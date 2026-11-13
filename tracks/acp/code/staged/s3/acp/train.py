"""ACP stage 1 on the X2: PPO (Table S6) for the contact-conditioned tracker with oracle contacts.

    python -m acp.train --data cfg/stage1_sets.json --run runs/s1_v1 [--envs 12000] [--iters 30000] [--resume]

Writes ckpt_<it>.pt + latest.pt in the run folder, log.jsonl, and the heartbeat /mnt/filesystem-g4/status/acp.json.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import time

import torch
import torch.nn as nn

from acp.data import Dataset
from acp import planner as P
from acp.env import DT, AcpEnv, G, W

# Table S6
STEPS, EPOCHS, MINIBATCHES, CLIP, VCOEF, GRADNORM, GAMMA, LAM = 24, 5, 8, 0.2, 1.0, 1.0, 0.99, 0.95
LR0, TARGET_KL, ENT = 1e-4, 0.02, 0.006
INIT_STD = 1.0          # paper-silent (rsl_rl default)


class Norm(nn.Module):
    """Running observation normalisation (paper-silent; rsl_rl's empirical normalisation)."""

    def __init__(self, dim):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4))

    @torch.no_grad()
    def update(self, x):
        b = x.shape[0]
        tot = self.count + b
        d = x.mean(0) - self.mean
        self.var.copy_((self.var * self.count + x.var(0, unbiased=False) * b + d**2 * self.count * b / tot) / tot)
        self.mean.add_(d * b / tot)
        self.count.copy_(tot)

    def forward(self, x):
        return ((x - self.mean) / (self.var.sqrt() + 1e-5)).clamp(-10.0, 10.0)


def mlp(i, o):
    return nn.Sequential(nn.Linear(i, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, o))


class ActorCritic(nn.Module):
    def __init__(self, actor_dim, critic_dim, act_dim):
        super().__init__()
        self.actor, self.critic = mlp(actor_dim, act_dim), mlp(critic_dim, 1)     # 512-256-128 ELU (Table S3 text)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(torch.tensor(INIT_STD).log())))
        self.an, self.cn = Norm(actor_dim), Norm(critic_dim)

    def dist(self, ao):
        return torch.distributions.Normal(self.actor(self.an(ao)), self.log_std.exp().expand(ao.shape[0], -1))

    def value(self, co):
        return self.critic(self.cn(co)).squeeze(-1)


def heartbeat(path, **kw):
    kw.update(track="acp", time=time.strftime("%Y-%m-%dT%H:%M:%S%z"), unix=time.time())
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(kw, fh)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--envs", type=int, default=12000)
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt-every", type=int, default=250)
    ap.add_argument("--latest-minutes", type=float, default=20.0, help="also write latest.pt when this many minutes passed since the last save (backup rule: at least every 30 min); 0 = off")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--njmax", type=int, default=300)
    ap.add_argument("--nconmax", type=int, default=64)
    ap.add_argument("--heartbeat", default="/mnt/filesystem-g4/status/acp.json")
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--init", default="", help="stage 2: the stage-1 checkpoint to start from (used when the run has no latest.pt)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--planner-arch", default="attn")
    ap.add_argument("--planner-init", default="", help="planner weights from acp.planner train (optional)")
    ap.add_argument("--rec-envs", type=int, default=3000)
    ap.add_argument("--replay", type=int, default=1500000)
    ap.add_argument("--u-obs", type=int, default=200); ap.add_argument("--u-rew", type=int, default=700); ap.add_argument("--n-ann", type=int, default=1000)
    ap.add_argument("--bench", type=int, default=0, help="only step the environment this many iterations and print the throughput")
    ap.add_argument("--goal-share", type=float, default=0.5, help="stage 3: share of goal environments (paper: not given)")
    ap.add_argument("--s3-warmup", type=int, default=500, help="stage 3: iterations with goal samples out of the actor loss (paper: 500)")
    args = ap.parse_args()
    dev = torch.device(args.device)
    target_kl, ent = {1: (TARGET_KL, ENT), 2: (0.01, 0.005), 3: (0.01, 0.002)}[args.stage]   # Table S6
    torch.manual_seed(args.seed)
    os.makedirs(args.run, exist_ok=True)
    data = Dataset(args.data, dev, cache=os.path.join(os.path.dirname(args.data), "labels_" + os.path.basename(args.data) + ".pt"))
    m = data.motion
    print(f"[data] clips {len(m.names)} frames {m.num_frames} ({m.num_frames / 50 / 60:.1f} min) prop slots {data.terrain.num_slots} "
          f"contact frac per body {[round(float(x), 3) for x in data.contact.float().mean(0)]} occ {[round(float(x), 3) for x in data.occ.float().mean(0)]}", flush=True)
    env = AcpEnv(data, args.envs, seed=args.seed, njmax=args.njmax, nconmax=args.nconmax)
    N, A = args.envs, env.NJ
    ac = ActorCritic(env.actor_dim, env.critic_dim, A).to(dev)
    opt = torch.optim.Adam(ac.parameters(), lr=LR0)
    lr, it0 = LR0, 0
    latest = os.path.join(args.run, "latest.pt")
    ck = None
    if not (args.resume and os.path.exists(latest)) and args.init:
        latest_src = args.init
    else:
        latest_src = latest
    if (args.resume or args.init) and os.path.exists(latest_src):
        ck = torch.load(latest_src, map_location=dev)
        ac.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        lr, it0 = ck["lr"], ck["it"]
        env.phase.copy_(ck["phase"]) if ck["phase"].shape == env.phase.shape else env.phase.fill_(float(ck["phase"].mean()))
        print(f"[resume] {latest_src} at iteration {it0}, lr {lr:.2e}, mean phase {float(env.phase.mean()):.3f}", flush=True)
    print(f"[env] envs {N} actor {env.actor_dim} critic {env.critic_dim} actions {A}", flush=True)
    pl = popt = replay = None
    s2_start = it0
    if args.stage >= 2:
        env.csi = False                                           # Stage 2: initial phase U(0, 0.98)
        hd = 5 * env.PROP
        pl = P.build(args.planner_arch, hd).to(dev)
        if ck is not None and ck.get("planner") is not None:
            pl.load_state_dict(ck["planner"]); s2_start = ck.get("s2_start", it0)
        elif args.planner_init:
            pk = torch.load(args.planner_init, map_location=dev)
            assert pk["arch"] == args.planner_arch, (pk["arch"], args.planner_arch)
            pl.load_state_dict(pk["net"])
        else:                                                    # input statistics from the tracker's running normalisers
            pl.h_mu.copy_(ac.an.mean[:hd]); pl.h_sd.copy_(ac.an.var[:hd].sqrt().clamp(min=1e-3))
            sm, sv = ac.cn.mean[env.PROP:env.PROP + 187], ac.cn.var[env.PROP:env.PROP + 187]
            pl.s_mu.fill_(float(sm.mean())); pl.s_sd.fill_(float((sv.mean() + sm.var()).sqrt().clamp(min=1e-3)))
        popt = torch.optim.Adam(pl.parameters(), lr=1e-4)        # Table S5
        if ck is not None and ck.get("popt") is not None:
            popt.load_state_dict(ck["popt"])
        pl.eval()
        env.planner = pl
        env.rec = P.Recorder(min(args.rec_envs, N), hd, dev)
        replay = P.Replay(args.replay, hd, dev)
        print(f"[stage 2] planner {args.planner_arch} ({sum(p.numel() for p in pl.parameters())} parameters), stage-2 start iteration {s2_start}, "
              f"u_obs {args.u_obs} u_rew {args.u_rew} N_ann {args.n_ann}, recorder envs {env.rec.n}", flush=True)
    vfs = vfs_n = vopt = None
    s3_start = it0
    gm = torch.zeros(N, dtype=torch.bool, device=dev)
    if args.stage == 3:                                           # goal environments, V_fs (fixed lr 1e-3), curriculum state
        env.setup_goals(args.goal_share)
        gm = env.goal_env
        vfs, vfs_n = mlp(env.goal_critic_dim, 1).to(dev), Norm(env.goal_critic_dim).to(dev)
        vopt = torch.optim.Adam(vfs.parameters(), lr=1e-3)
        if ck is not None and ck.get("vfs") is not None:
            vfs.load_state_dict(ck["vfs"]); vfs_n.load_state_dict(ck["vfs_norm"]); vopt.load_state_dict(ck["vopt"])
            s3_start = ck.get("s3_start", it0)
            if ck.get("g_level") is not None and ck["g_level"].shape[0] == N:
                env.g_level.copy_(ck["g_level"]); env.g_ema.copy_(ck["g_ema"])
        print(f"[stage 3] goal envs {int(gm.sum())} of {N}, families {env.data.goal_families}, stage-3 start iteration {s3_start}, "
              f"warm-up {args.s3_warmup}, V_fs input {env.goal_critic_dim}", flush=True)
    pstat = dict(aff=0.0, occ=0.0, new=0, updates=0)
    env.reset_all()
    ao, co = env.observe()
    B = N * STEPS
    gdim = env.goal_critic_dim if args.stage == 3 else 1
    buf = dict(ao=torch.zeros(STEPS, N, env.actor_dim, device=dev), co=torch.zeros(STEPS, N, env.critic_dim, device=dev),
               cg=torch.zeros(STEPS, N, gdim, device=dev),
               a=torch.zeros(STEPS, N, A, device=dev), logp=torch.zeros(STEPS, N, device=dev), r=torch.zeros(STEPS, N, device=dev),
               v=torch.zeros(STEPS, N, device=dev), done=torch.zeros(STEPS, N, device=dev), mu=torch.zeros(STEPS, N, A, device=dev),
               std=torch.zeros(STEPS, N, A, device=dev))
    acc, nacc = {}, 0
    ep = dict(done=0.0, term=0.0, fin=0.0, bad=0.0)
    nset = len(data.set_names)
    set_done, set_fin = torch.zeros(nset, device=dev), torch.zeros(nset, device=dev)
    t_log, steps_log = time.time(), 0
    logf = open(os.path.join(args.run, "log.jsonl"), "a")
    metf = open(os.path.join(args.run, "metrics.jsonl"), "a")     # one row per iteration: every paper term, raw and weighted
    cnt, gpu = {}, {}
    t_it = time.time()
    t_saved = time.time()
    for it in range(it0, args.iters):
        it_acc = {}
        if args.stage >= 2:                                       # Eq. 3: linear schedules, the reward one starts later
            u = it - s2_start
            env.eta_obs = min(1.0, max(0.0, (u - args.u_obs) / args.n_ann))
            env.eta_rew = min(1.0, max(0.0, (u - args.u_rew) / args.n_ann))
            env.plan_stats.zero_()
        it_set_done, it_set_fin = torch.zeros(nset, device=dev), torch.zeros(nset, device=dev)
        with torch.no_grad():
            for s in range(STEPS):
                ac.an.update(ao)
                if args.stage == 3:
                    cg = env.critic_goal
                    if bool((~gm).any()):
                        ac.cn.update(co[~gm])
                    vfs_n.update(cg[gm])
                    buf["cg"][s] = cg
                else:
                    ac.cn.update(co)
                dist = ac.dist(ao)
                a = dist.sample()
                v = ac.value(co)
                if args.stage == 3:
                    v = torch.where(gm, vfs(vfs_n(cg)).squeeze(-1), v)
                buf["ao"][s], buf["co"][s], buf["a"][s], buf["v"][s] = ao, co, a, v
                buf["logp"][s], buf["mu"][s], buf["std"][s] = dist.log_prob(a).sum(-1), dist.mean, dist.stddev
                r, term, timeout, info = env.step(a)
                buf["r"][s] = r + GAMMA * v * timeout.float()            # bootstrap on time-outs (rsl_rl)
                buf["done"][s] = (term | timeout).float()
                ao, co = env.observe()
                for k, x in info["terms"].items():
                    acc[k] = acc.get(k, 0.0) + x
                for k in ("e_pos", "e_rot", "e_joint", "e_contact_pos", "contact_match", "c_ref_frac", "c_meas_frac", "speed", "ref_speed"):
                    acc[k] = acc.get(k, 0.0) + info[k]
                acc["reward"] = acc.get("reward", 0.0) + r.mean()
                nacc += 1
                for k in ("term_pos", "term_rot", "timeout", "place_sum", "place_n"):
                    cnt[k] = cnt.get(k, 0.0) + info[k]
                for k, x in info["terms"].items():
                    it_acc[k] = it_acc.get(k, 0.0) + x
                for k in ("e_pos", "e_rot", "e_joint", "e_contact_pos", "contact_match", "c_ref_frac", "c_meas_frac", "speed", "ref_speed",
                          "done", "term", "finished", "bad", "term_pos", "term_rot", "timeout", "place_sum", "place_n"):
                    it_acc[k] = it_acc.get(k, 0.0) + info[k]
                it_acc["reward"] = it_acc.get("reward", 0.0) + r.mean()
                it_set_done.index_add_(0, info["set"], info["done_mask"].float()); it_set_fin.index_add_(0, info["set"], info["finished_mask"].float())
                if args.stage == 3:
                    for k in ("goal_done", "goal_success", "goal_fall", "goal_wp"):
                        it_acc[k] = it_acc.get(k, 0.0) + info[k]
                ep["done"] += float(info["done"]); ep["term"] += float(info["term"]); ep["fin"] += float(info["finished"]); ep["bad"] += float(info["bad"])
                set_done.index_add_(0, info["set"], info["done_mask"].float()); set_fin.index_add_(0, info["set"], info["finished_mask"].float())
            steps_log += B
            if args.bench:
                if it + 1 - it0 >= args.bench:
                    dt = time.time() - t_log
                    print(f"[bench] envs {N}: {steps_log / dt:.0f} env-steps/s ({dt / args.bench:.2f} s per {STEPS}-step rollout), "
                          f"GPU mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GB torch", flush=True)
                    return
                if it == it0:
                    t_log, steps_log = time.time(), 0                 # skip the compile
                continue
            last_v = ac.value(co)
            if args.stage == 3:
                last_v = torch.where(gm, vfs(vfs_n(env.critic_goal)).squeeze(-1), last_v)
            adv = torch.zeros_like(buf["r"])
            g = torch.zeros(N, device=dev)
            for s in reversed(range(STEPS)):
                nv = last_v if s == STEPS - 1 else buf["v"][s + 1]
                nd = 1.0 - buf["done"][s]
                delta = buf["r"][s] + GAMMA * nv * nd - buf["v"][s]
                g = delta + GAMMA * LAM * nd * g
                adv[s] = g
            ret = adv + buf["v"]
            if args.stage == 3:                                   # advantages normalised separately per environment type
                for msk in (gm, ~gm):
                    if bool(msk.any()):
                        x = adv[:, msk]
                        adv[:, msk] = (x - x.mean()) / (x.std() + 1e-8)
            else:
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        flat = {k: x.reshape(B, *x.shape[2:]) for k, x in buf.items()}
        adv_f, ret_f = adv.reshape(B), ret.reshape(B)
        gm_f = gm.unsqueeze(0).expand(STEPS, -1).reshape(B)
        warm = args.stage == 3 and (it - s3_start) < args.s3_warmup
        vgl = []
        kls, vls, pls = [], [], []
        for _ in range(EPOCHS):
            perm = torch.randperm(B, device=dev)
            for mb in perm.chunk(MINIBATCHES):
                dist = ac.dist(flat["ao"][mb])
                logp = dist.log_prob(flat["a"][mb]).sum(-1)
                v = ac.value(flat["co"][mb])
                with torch.no_grad():
                    mu0, s0 = flat["mu"][mb], flat["std"][mb]
                    kl = (torch.log(dist.stddev / s0) + (s0**2 + (mu0 - dist.mean) ** 2) / (2 * dist.stddev**2) - 0.5).sum(-1).mean()
                    if kl > 2 * target_kl:
                        lr = max(1e-5, lr / 1.5)
                    elif kl < target_kl / 2 and kl > 0:
                        lr = min(1e-2, lr * 1.5)
                    for gp in opt.param_groups:
                        gp["lr"] = lr
                ratio = (logp - flat["logp"][mb]).exp()
                a_ = adv_f[mb]
                pw = torch.ones_like(a_) if not warm else (~gm_f[mb]).float()   # warm-up: goal samples out of the actor loss
                pl_ = -(torch.min(ratio * a_, ratio.clamp(1 - CLIP, 1 + CLIP) * a_) * pw).sum() / pw.sum().clamp(min=1.0)
                v0 = flat["v"][mb]
                if args.stage == 3:
                    g_ = gm_f[mb]
                    v = torch.where(g_, vfs(vfs_n(flat["cg"][mb])).squeeze(-1), v)
                vc = v0 + (v - v0).clamp(-CLIP, CLIP)
                vle = torch.max((v - ret_f[mb]) ** 2, (vc - ret_f[mb]) ** 2)
                if args.stage == 3:                               # each critic on its own environments
                    vl = (vle * (~g_).float()).sum() / (~g_).float().sum().clamp(min=1.0)
                    vlg = (vle * g_.float()).sum() / g_.float().sum().clamp(min=1.0)
                    vgl.append(vlg.detach())
                else:
                    vl, vlg = vle.mean(), 0.0
                loss = pl_ + VCOEF * (vl + vlg) - ent * dist.entropy().sum(-1).mean()
                opt.zero_grad(set_to_none=True)
                if vopt is not None:
                    vopt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), GRADNORM)
                opt.step()
                if vopt is not None:
                    nn.utils.clip_grad_norm_(vfs.parameters(), GRADNORM)
                    vopt.step()
                kls.append(kl); vls.append(vl.detach()); pls.append(pl_.detach())
        n = it + 1
        if args.stage >= 2:                                       # planner: hindsight samples in, ten updates of 4,096 (Table S5)
            pstat = dict(aff=0.0, occ=0.0, new=env.rec.flush(replay), updates=0)
            if replay.size >= 1000:
                pl.train()
                for _ in range(10):
                    lp, la, lo = P.planner_loss(pl, *replay.sample(4096))
                    popt.zero_grad(set_to_none=True); lp.backward(); popt.step()
                    pstat["aff"] += float(la) / 10; pstat["occ"] += float(lo) / 10; pstat["updates"] += 1
                pl.eval()
        if not torch.isfinite(torch.stack(vls)).all():
            raise RuntimeError(f"non-finite loss at iteration {n}")
        # ---- per-iteration metrics row (names as in the paper's Table 3; legend in state/acp.md)
        if n % args.log_every == 0 or not gpu:
            try:
                o = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, timeout=10).stdout.split(",")
                gpu = dict(gpu_util_pct=float(o[0]), gpu_mem_gb=round(float(o[1]) / 1024, 1))
            except Exception:
                gpu = dict(gpu_util_pct=-1.0, gpu_mem_gb=-1.0)
        fa = {k: float(x) for k, x in it_acc.items()}
        dn = max(fa["done"], 1.0)
        now = time.time()
        wts = env.weights()
        mrow = dict(it=n, time=time.strftime("%Y-%m-%dT%H:%M:%S"), stage=args.stage, envs=N, env_steps=B, env_steps_total_this_process=(n - it0) * B,
                    iter_s=round(now - t_it, 3), env_steps_per_s=round(B / max(now - t_it, 1e-6)), **gpu,
                    reward_per_step=round(fa["reward"] / STEPS, 6), reward_max_per_step=round(float(info["reward_max"]), 4))
        for k in wts:
            mrow["raw/" + k] = round(fa[k] / STEPS, 6)                       # mean of r_i over envs and steps
            mrow["weighted/" + k] = round(DT * wts[k] * fa[k] / STEPS, 6)    # its share of the per-step reward: dt * w_i * r_i
            mrow["weight/" + k] = round(wts[k], 6)
        if args.stage == 3:
            gd = max(fa["goal_done"], 1.0)
            lv = env.g_level[gm].float()
            mrow.update({"stage3/iteration": n - s3_start, "stage3/warmup": int(warm), "goal/episodes": int(fa["goal_done"]),
                         "goal/success": round(fa["goal_success"] / gd, 4), "goal/fall": round(fa["goal_fall"] / gd, 4),
                         "goal/waypoint_index_mean": round(fa["goal_wp"] / STEPS, 3), "goal/level_mean": round(float(lv.mean()), 3),
                         "ppo/value_loss_goal": round(float(torch.stack(vgl).mean()), 5) if vgl else None})
            for fi, fname in enumerate(env.data.goal_families):
                sel = gm & (env.g_fam == fi)
                mrow["goal/level/" + fname] = round(float(env.g_level[sel].float().mean()), 3)
            for k in G:
                mrow["raw/goal/" + k] = round(fa.get("goal/" + k, 0.0) / STEPS, 6)
                mrow["weighted/goal/" + k] = round(DT * G[k] * fa.get("goal/" + k, 0.0) / STEPS, 6)
                mrow["weight/goal/" + k] = G[k]
        if args.stage >= 2:
            ps = [float(x) for x in env.plan_stats]
            mrow.update({"stage2/eta_obs": round(env.eta_obs, 4), "stage2/eta_rew": round(env.eta_rew, 4), "stage2/iteration": n - s2_start,
                         "planner/loss_affordance": round(pstat["aff"], 5), "planner/loss_occupancy": round(pstat["occ"], 5),
                         "planner/updates": pstat["updates"], "planner/new_samples": pstat["new"], "planner/replay_size": replay.size,
                         "planner/recorded_episodes": env.rec.episodes, "planner/recorded_successes": env.rec.successes,
                         "planner/fed_to_tracker_share": round(ps[2] / max(ps[3], 1.0), 4),
                         "planner/target_vs_reference_m": round(ps[0] / max(ps[1], 1.0), 4)})
        mrow.update({"err/base_pos_m": round(fa["e_pos"] / STEPS, 5), "err/base_rot_rad": round(fa["e_rot"] / STEPS, 5),
                     "err/joint_pos_rad_mean_abs": round(fa["e_joint"] / STEPS, 5), "err/contact_pos_base_m": round(fa["e_contact_pos"] / STEPS, 5),
                     "err/contact_placement_at_onset_m": round(fa["place_sum"] / max(fa["place_n"], 1.0), 5), "contact_onsets": int(fa["place_n"]),
                     "contact/label_agreement": round(fa["contact_match"] / STEPS, 5), "contact/feet_ref_frac": round(fa["c_ref_frac"] / STEPS, 4),
                     "contact/feet_measured_frac": round(fa["c_meas_frac"] / STEPS, 4), "speed/base_xy_mps": round(fa["speed"] / STEPS, 4),
                     "speed/ref_xy_mps": round(fa["ref_speed"] / STEPS, 4),
                     "episodes": int(fa["done"]), "end/finished_phase099": round(fa["finished"] / dn, 4),
                     "end/timeout_before_phase099": round((fa["timeout"] - fa["finished"]) / dn, 4),
                     "end/term_base_pos_gt_0.5m": round(fa["term_pos"] / dn, 4), "end/term_base_rot_gt_1rad": round(fa["term_rot"] / dn, 4),
                     "end/term_diverged": round(fa["bad"] / dn, 4),
                     "csi/phase_mean": round(float(env.phase.mean()), 4), "csi/phase_p10": round(float(env.phase.quantile(0.1)), 3),
                     "csi/phase_p50": round(float(env.phase.quantile(0.5)), 3), "csi/phase_p90": round(float(env.phase.quantile(0.9)), 3),
                     "ppo/lr": lr, "ppo/kl": round(float(torch.stack(kls).mean()), 5), "ppo/value_loss": round(float(torch.stack(vls).mean()), 5),
                     "ppo/policy_loss": round(float(torch.stack(pls).mean()), 5), "ppo/action_std": round(float(ac.log_std.exp().mean()), 4),
                     "ppo/value_mean": round(float(buf["v"].mean()), 4), "ppo/return_mean": round(float(ret.mean()), 4)})
        for i in range(nset):
            mrow["finish/" + data.set_names[i]] = round(float(it_set_fin[i] / it_set_done[i].clamp(min=1)), 3)
            mrow["episodes/" + data.set_names[i]] = int(it_set_done[i])
        metf.write(json.dumps(mrow) + "\n"); metf.flush()
        t_it = now
        if n % args.log_every == 0 or n == args.iters:
            dt = time.time() - t_log
            row = {k: round(float(x) / nacc, 5) for k, x in acc.items()}
            row.update(it=n, fps=round(steps_log / dt), lr=lr, kl=round(float(torch.stack(kls).mean()), 5),
                       value_loss=round(float(torch.stack(vls).mean()), 5), std=round(float(ac.log_std.exp().mean()), 4),
                       csi_phase=round(float(env.phase.mean()), 4), csi_p10=round(float(env.phase.quantile(0.1)), 3),
                       csi_p90=round(float(env.phase.quantile(0.9)), 3),
                       term_frac=round(ep["term"] / max(ep["done"], 1), 4), finish_frac=round(ep["fin"] / max(ep["done"], 1), 4),
                       episodes=int(ep["done"]), diverged=int(ep["bad"]),
                       finish_by_set={data.set_names[i]: round(float(set_fin[i] / set_done[i].clamp(min=1)), 3) for i in range(nset)},
                       time=time.strftime("%Y-%m-%dT%H:%M:%S"))
            logf.write(json.dumps(row) + "\n"); logf.flush()
            print(f"it {n} fps {row['fps']} rew {row['reward']:.4f} e_pos {row['e_pos']:.3f} e_joint {row['e_joint']:.3f} "
                  f"contact_match {row['contact_match']:.3f} term {row['term_frac']:.3f} finish {row['finish_frac']:.3f} "
                  f"csi {row['csi_phase']:.3f} kl {row['kl']:.4f} lr {lr:.1e} std {row['std']:.3f}", flush=True)
            cks = sorted(glob.glob(os.path.join(args.run, "ckpt_*.pt")))
            heartbeat(args.heartbeat, state="training", stage=args.stage, run=os.path.abspath(args.run), iteration=n, iterations=args.iters,
                      newest_checkpoint=os.path.abspath(cks[-1]) if cks else None, fps=row["fps"], reward=row["reward"],
                      csi_phase=row["csi_phase"], finish_frac=row["finish_frac"], term_frac=row["term_frac"], e_pos=row["e_pos"])
            acc, nacc = {}, 0
            ep = dict(done=0.0, term=0.0, fin=0.0, bad=0.0)
            set_done.zero_(); set_fin.zero_()
            t_log, steps_log = time.time(), 0
        save_num = n % args.ckpt_every == 0 or n == args.iters
        save_latest = save_num or (args.latest_minutes > 0 and time.time() - t_saved >= 60.0 * args.latest_minutes)
        if save_latest:
            ck = dict(model=ac.state_dict(), opt=opt.state_dict(), lr=lr, it=n, phase=env.phase.clone(), actor_dim=env.actor_dim,
                      critic_dim=env.critic_dim, act_dim=A, data=args.data, joint_names=env.x2.joint_names, stage=args.stage,
                      planner=pl.state_dict() if pl is not None else None, popt=popt.state_dict() if popt is not None else None,
                      planner_arch=args.planner_arch if pl is not None else None, s2_start=s2_start, x2_exclude_pairs=os.environ.get("X2_EXCLUDE_PAIRS", ""),
                      vfs=vfs.state_dict() if vfs is not None else None, vfs_norm=vfs_n.state_dict() if vfs_n is not None else None,
                      vopt=vopt.state_dict() if vopt is not None else None, s3_start=s3_start,
                      g_level=env.g_level.clone() if args.stage == 3 else None, g_ema=env.g_ema.clone() if args.stage == 3 else None,
                      goal_share=args.goal_share if args.stage == 3 else None)
            if save_num:
                path = os.path.join(args.run, f"ckpt_{n:06d}.pt")
                torch.save(ck, path + ".tmp"); os.replace(path + ".tmp", path)
            torch.save(ck, latest + ".tmp"); os.replace(latest + ".tmp", latest); t_saved = time.time()
            if not save_num:
                print(f"[save] latest.pt at iteration {n} (time-based, every {args.latest_minutes} min)", flush=True)
    heartbeat(args.heartbeat, state="finished", stage=args.stage, run=os.path.abspath(args.run), iteration=args.iters, iterations=args.iters,
              newest_checkpoint=os.path.abspath(os.path.join(args.run, f"ckpt_{args.iters:06d}.pt")))


if __name__ == "__main__":
    main()
