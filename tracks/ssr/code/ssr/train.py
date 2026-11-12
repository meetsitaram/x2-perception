"""SSR training: single-stage PPO with three critics, latent-space mirror augmentation, imagined foothold guidance, and
terrain-specific AMP (paper Sec. 3, Table 9). Four Adam optimizers at 5e-4.

python -m ssr.train --num_envs 4096 --iters 20000 --name run1
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F

from ssr import mirror as M
from ssr import networks as N
from ssr import terrain as T
from ssr.env import AMP_DIM, AMP_FRAMES, DEPTH_N, H, HB_NX, HB_NY, HF_NX, HF_NY, PRIV_DIM, PROP_DIM, SSREnv

ROOT = pathlib.Path(__file__).resolve().parents[1]
WEIGHTS = torch.tensor([1.0, 0.25, 0.2])       # w_l, w_f, w_s
# Table 6 weights, for the log: raw term = weighted term / weight
TERM_W = dict(lin_vel=1.0, ang_vel=0.8, orientation=0.5, ang_vel_xy=0.25, base_height=0.4, action_rate=-0.12, smoothness=-0.06, dof_vel=-0.96, dof_torque=-0.6,
              dof_dev=-1.8, dof_pos_lim=-1.5, dof_vel_lim=-6.0, dof_tau_lim=-6.0, stand_still=-0.12, single_support=0.2, impact=-1.3, slip=-0.2, air_time=-2.0,
              stumble=-2.0, lateral=0.08)


def gpu_now():
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
        u, m = o.strip().split(",")
        return int(u), int(m)
    except Exception:
        return -1, -1
GAMMA, LAM, CLIP, ENT, VF, KL_TARGET = 0.99, 0.95, 0.2, 0.005, 1.0, 0.01
STEPS, EPOCHS, MINI = 24, 5, 4
A_CLIP = 8.0


class AMPData:
    def __init__(self, device, min_speed=0.0, file="data/amp.npz", balance=False):
        z = np.load(ROOT / file)
        self.frames, self.valid, self.weight, self.fed = [], [], [], {}
        for name in T.TYPES:
            f = torch.as_tensor(z[name], device=device, dtype=torch.float32)
            f = torch.cat((f, M.amp(f)))                    # mirrored copy
            n = len(z[name])
            st = z[name + "_starts"]
            ok = np.ones(n, bool)
            for s in st:                                    # a 5-frame window may not cross a clip start
                ok[max(0, s - AMP_FRAMES + 1):s] = False
            ok[-AMP_FRAMES + 1:] = False
            if min_speed > 0:                               # windows where the body is not travelling are not locomotion: a prior that contains standing rewards standing
                sp = np.linalg.norm(z[name][:, 3:5], axis=1)
                win = np.convolve(sp, np.ones(AMP_FRAMES) / AMP_FRAMES)[AMP_FRAMES - 1:]     # mean speed of the window starting at each frame
                win = np.concatenate((win, np.zeros(n - len(win))))[:n]
                ok &= win >= min_speed
            sp_w = np.convolve(np.linalg.norm(z[name][:, 3:5], axis=1), np.ones(AMP_FRAMES) / AMP_FRAMES)[AMP_FRAMES - 1:]
            sp_w = np.concatenate((sp_w, np.zeros(n - len(sp_w))))[:n]
            wt = np.ones(n)
            if balance:                                     # equal reference mass per 0.2 m/s speed bin, so the prior does not favour one walking speed over the commanded range;
                b = np.digitize(sp_w, [0.2, 0.4, 0.6, 0.8, 1.0])   # no window is weighted more than 5x the uniform share (few fast clips must not dominate)
                cnt = np.bincount(b[ok], minlength=6).astype(float)
                wt = np.where(cnt[b] > 0, 1.0 / np.maximum(cnt[b], 1.0), 0.0)
                wt = np.minimum(wt, 5.0 / max(ok.sum(), 1))
            wv = wt[ok] / wt[ok].sum()
            q = lambda p_: float(sp_w[ok][np.argsort(sp_w[ok])][np.searchsorted(np.cumsum(wv[np.argsort(sp_w[ok])]), p_)])
            self.fed[name] = dict(seconds=round(float(ok.sum()) / 50), speed_p10=round(q(0.1), 2), p25=round(q(0.25), 2), p50=round(q(0.5), 2), p75=round(q(0.75), 2), p90=round(q(0.9), 2),
                                  share_over_0p8=round(float(wv[sp_w[ok] >= 0.8].sum()), 3))
            self.weight.append(torch.as_tensor(np.concatenate((wt[ok], wt[ok])), device=device, dtype=torch.float32))
            self.kept = getattr(self, "kept", {})
            self.kept[name] = round(float(ok.mean()), 3)
            ok = torch.as_tensor(np.concatenate((ok, ok)), device=device)
            self.frames.append(f)
            self.valid.append(ok.nonzero().squeeze(-1))

    def sample(self, i, n):
        v = self.valid[i]
        s = v[torch.multinomial(self.weight[i], n, replacement=True)]
        return self.frames[i][s[:, None] + torch.arange(AMP_FRAMES, device=v.device)].reshape(n, -1)


def disc_index(env):
    return torch.where(env.free, torch.full_like(env.type, T.FLAT), env.type)


def style_reward(D, x, idx):
    r = torch.zeros(len(x), device=x.device)
    with torch.no_grad():
        for i in range(5):
            m = idx == i
            if bool(m.any()):
                r[m] = (1 - 0.25 * (D(x[m], i).squeeze(-1) - 1) ** 2).clamp(min=0.0)
    return r


def safe_step(opt, params, loss, stats, max_norm=None):
    """One optimizer step, skipped when the loss or a gradient is not finite (one bad batch must not poison the networks)."""
    opt.zero_grad(set_to_none=True)
    if not torch.isfinite(loss):
        stats["skipped"] = stats.get("skipped", 0) + 1
        return
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(params, max_norm if max_norm else 1e9)
    if not torch.isfinite(norm):
        opt.zero_grad(set_to_none=True)
        stats["skipped"] = stats.get("skipped", 0) + 1
        return
    opt.step()


def all_finite(*mods):
    return all(torch.isfinite(p).all() for m in mods for p in m.parameters())


def train(a):
    dev = a.device
    os.environ["X2_PLANT"] = a.plant                 # read by ssr.plant_config.load_plant when the model is built
    out = pathlib.Path(a.out_root or os.environ.get("SSR_RUNS") or ROOT / "runs") / a.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(a)))
    env = SSREnv(a.num_envs, dev, seed=a.seed, camera=a.camera, map_memory=a.map_memory)
    amp = AMPData(dev, a.prior_min_speed, a.amp_file, a.prior_balance)
    print("motion prior as fed to each discriminator (window mean planar speed, m/s):", json.dumps(amp.fed), flush=True)
    print("motion prior windows kept:", amp.kept, flush=True)
    if a.rsi_frac > 0:
        env.set_rsi(amp.frames[T.FLAT], a.rsi_frac, a.rsi_min_speed)
        print("walking-start frames:", len(env.rsi_ok), "of", len(env.rsi_frames), flush=True)
    env.imit_w = a.imit_w
    env.stuck_term = a.stuck_term
    env.flat_only = a.flat_only or a.flat_until > 0
    env.feet_air_w, env.feet_air_min = a.feet_air_w, a.feet_air_min
    env.only_terms = set(a.only_terms.split(",")) if a.only_terms else None
    if a.flat_only or a.flat_until > 0:
        env.reset(torch.arange(a.num_envs, device=dev))
        env.refresh()
        env._contacts()
        env._update_obs(torch.arange(a.num_envs, device=dev))     # also refreshes the stored foot positions: without it the first step saw the feet jump (one huge slip penalty at iteration 0 in d4 and t_b)
    enc, aux, actor, critics, foot, disc = (m.to(dev) for m in (N.Encoder(a.map_memory), N.Aux(), N.MoEActor(), N.Critics(), N.Foothold(), N.Discriminators()))
    lr_ppo = 5e-4
    actor.log_std.data.fill_(math.log(a.init_std))
    opt_ppo = torch.optim.Adam(list(actor.parameters()) + list(critics.parameters()), lr=lr_ppo)
    opt_enc = torch.optim.Adam(list(enc.parameters()) + list(aux.parameters()), lr=5e-4)
    opt_foot = torch.optim.Adam(foot.parameters(), lr=5e-4)
    opt_disc = torch.optim.Adam(disc.parameters(), lr=5e-4, weight_decay=1e-4)
    start = 0
    if a.resume:
        ck = torch.load(a.resume, map_location=dev, weights_only=False)
        for k, m in (("enc", enc), ("aux", aux), ("actor", actor), ("critics", critics), ("foot", foot), ("disc", disc)):
            m.load_state_dict(ck[k])
        for k, o in (("opt_ppo", opt_ppo), ("opt_enc", opt_enc), ("opt_foot", opt_foot), ("opt_disc", opt_disc)):
            o.load_state_dict(ck[k])
        start = ck["iter"] + 1
        pick = torch.arange(a.num_envs) % len(ck["level"])          # a different robot count than the checkpoint's: reuse its levels in order
        env.level[:] = ck["level"][pick].to(dev)
        env.type[:] = ck["type"][pick].to(dev)
        env.guide_on = bool(ck.get("guide_on", False))
        lr_ppo = opt_ppo.param_groups[0]["lr"]
        print("resumed", a.resume, start, flush=True)
    if a.flat_until > 0:
        env.flat_only = not (start >= a.flat_until and bool((env.type != T.FLAT).any()))   # resumed after the switch: keep the loaded levels

    N_, Tn = a.num_envs, STEPS
    h = enc.init_h(N_, dev)
    z0 = lambda *s: torch.zeros(Tn, N_, *s, device=dev)
    ep_r, ep_len = torch.zeros(N_, device=dev), torch.zeros(N_, device=dev)
    w = WEIGHTS.to(dev).clone()
    w[2] = a.w_style
    log_f = open(out / "log.jsonl", "a")
    t_start = time.time()
    done_stats = {"ret": [], "len": []}
    gate_start, gate_hits, air_td = (ck.get("gate_start") if a.resume else None), 0, 0.0   # t_e progress gate: iteration its ramp-off began
    for it in range(start, a.iters):
        t0 = time.time()
        B = dict(hist=z0(H, PROP_DIM), depth=z0(DEPTH_N, DEPTH_N), obs=z0(N.ACTOR_DIM), priv=z0(PRIV_DIM), a=z0(21), mu=z0(21), logp=z0(), v=z0(3), r=z0(3),
                 done=z0(), term=z0(), amp=z0(AMP_DIM * AMP_FRAMES), idx=torch.zeros(Tn, N_, dtype=torch.long, device=dev),
                 hb=z0(HB_NX * HB_NY), hf=z0(2 * HF_NX * HF_NY), sole=z0(2, 3), touch=torch.zeros(Tn, N_, 2, dtype=torch.bool, device=dev),
                 base=z0(2), yaw=z0(), nprop=z0(PROP_DIM))
        if a.flat_until > 0 and env.flat_only and it >= a.flat_until:      # flat phase over: terrain curriculum from level 0
            env.flat_only = False
            allr = torch.arange(N_, device=dev)
            env.type[:] = torch.randint(0, 5, (N_,), device=dev) if a.obst_ramp <= 0 else torch.where(torch.rand(N_, device=dev) < a.obst0, torch.randint(0, 4, (N_,), device=dev), torch.full((N_,), T.FLAT, device=dev))
            env.level[:] = 0
            env.ep[:] = 1                                              # so reset() keeps these types and levels (it redraws them when no step was taken yet)
            env.reset(allr)
            env.refresh()
            env._contacts()
            env._update_obs(allr)
            h = h * 0.0
            ep_r.zero_(); ep_len.zero_()
            print("flat phase over at", it, "; terrain curriculum on, types", torch.bincount(env.type, minlength=5).tolist(), flush=True)
        env.obst_target = None if (a.obst_ramp <= 0 or a.flat_until <= 0 or env.flat_only or it >= a.flat_until + a.obst_ramp) else a.obst0 + (0.8 - a.obst0) * (it - a.flat_until) / a.obst_ramp
        env.boot = max(0.0, 1.0 - it / a.boot_iters) if a.boot_iters > 0 else 0.0     # early schedule, not in the paper: off unless asked for
        env.air_bonus = 8.0 * env.boot
        ramp = 1.0 if a.stage_b <= 0 else min(1.0, max(0.0, (it - a.stage_a) / max(a.stage_b - a.stage_a, 1)))   # staged schedule: 0 before stage_a, 1 from stage_b on
        env.late_scale = ramp
        cf = 1.0 if a.cmd_cur_iters <= 0 else min(1.0, max(0.0, (it - a.cmd_cur_iters) / a.cmd_cur_iters))   # command curriculum: held until cmd_cur_iters, released by 2x
        env.v_min = a.cmd_min0 * (1.0 - cf)
        env.sig_v = 0.25 if a.sig_v0 <= 0 else a.sig_v0 + (0.25 - a.sig_v0) * cf
        env.stuck_term = a.stuck_term and (a.stuck_until <= 0 or it < a.stuck_until)
        env.prog_gate = 0.0 if a.prog_gate <= 0 else a.prog_gate * (1.0 if gate_start is None else max(0.0, 1.0 - (it - gate_start) / max(a.gate_ramp, 1)))
        w[2] = a.w_style * ramp
        if a.style_from > 0:                         # style by schedule from style_from (overrides the latch)
            w[2] = a.w_style * ramp * min(1.0, max(0.0, (it - a.style_from) / max(a.gate_ramp, 1)))
        elif a.style_low >= 0:                         # t_f: style held low until real swing appears, then ramped to full
            w[2] = w[2] * (a.style_low if gate_start is None else min(1.0, a.style_low + (1.0 - a.style_low) * (it - gate_start) / max(a.gate_ramp, 1)))
        env.gait = {}
        env.cnt = {}
        sty_sum, sty_n = torch.zeros(5, device=dev), torch.zeros(5, device=dev)
        h0 = h.detach().clone()
        B_map = torch.zeros(Tn, N_, 2, 64, 64, device=dev) if a.map_memory else None
        rs = [0.0, 0.0, 0.0]
        term_sum = {}
        guard = {}
        with torch.no_grad():
            for t in range(Tn):
                e = enc.step(env.hist, env.depth, h, map_img=env.map_img)
                if a.map_memory:
                    B_map[t] = env.map_img
                h = e["h"]
                vh = aux.est(env.hist.flatten(1))
                o = torch.cat((env.prop, vh, e["z"]), -1)
                actor.norm.update(o)
                critics.norm.update(env.priv)
                foot.norm.update(env.priv)
                mu = actor(o)
                std = actor.std()
                act = mu + std * torch.randn_like(mu)                # probability of the sampled action; the environment clamps its own copy
                logp = (-0.5 * ((act - mu) / std) ** 2 - std.log() - 0.9189385).sum(-1)
                v = critics(env.priv)
                fm, fs = foot(env.priv, act)
                for k, val in (("hist", env.hist), ("depth", env.depth), ("obs", o), ("priv", env.priv), ("a", act), ("mu", mu), ("logp", logp), ("v", v),
                               ("hb", env.h_b), ("hf", env.h_f.reshape(N_, -1)), ("base", env.s["p"][:, :2]), ("yaw", env.s["yaw"])):
                    B[k][t] = val
                idx = disc_index(env)
                out_s = env.step(act.clamp(-A_CLIP, A_CLIP), fm, fs)
                r_s = style_reward(disc, env.amp_hist.flatten(1), idx)
                B["amp"][t], B["idx"][t] = env.amp_hist.flatten(1), idx
                sty_sum.index_add_(0, idx, r_s)
                sty_n.index_add_(0, idx, torch.ones_like(r_s))
                r = torch.stack((out_s["r_l"], out_s["r_f"], r_s), -1)
                r = torch.nan_to_num(r, 0.0, 0.0, 0.0)
                for j in range(3):
                    rs[j] += r[:, j].mean().item() / Tn       # logged per step, unscaled
                r = r * a.rew_scale                              # legged_gym convention (rewards x dt): the same objective, returns of a size the critics can fit
                timeout = out_s["timeout"] & ~out_s["term"]
                r = r + GAMMA * v * timeout[:, None].float()        # bootstrap time limits
                B["r"][t], B["done"][t], B["term"][t] = r, out_s["done"].float(), out_s["term"].float()
                B["sole"][t], B["touch"][t], B["nprop"][t] = out_s["sole"], out_s["touchdown"], env.prop
                for k_, v_ in env.terms.items():
                    term_sum[k_] = term_sum.get(k_, 0.0) + torch.nan_to_num(v_).mean().item() / Tn
                tot = (r * w).sum(-1)
                ep_r += tot
                ep_len += 1
                d = out_s["done"]
                if bool(d.any()):
                    done_stats["ret"] += ep_r[d].tolist()
                    done_stats["len"] += ep_len[d].tolist()
                    ep_r[d], ep_len[d] = 0.0, 0.0
                    h[d] = 0.0
            v_last = critics(env.priv)
        # GAE per critic
        adv = torch.zeros_like(B["r"])
        g = torch.zeros(N_, 3, device=dev)
        for t in reversed(range(Tn)):
            nv = v_last if t == Tn - 1 else B["v"][t + 1]
            nd = (1 - B["done"][t])[:, None]
            delta = B["r"][t] + GAMMA * nv * nd - B["v"][t]
            g = delta + GAMMA * LAM * nd * g
            adv[t] = g
        ret = adv + B["v"]
        adv_sd = (adv.std((0, 1)) * w).tolist()        # each group's weighted advantage spread: its pull on the policy under joint normalisation
        if a.adv_norm == "separate":                  # each critic's advantage standardised, then weighted: the weights then set each group's share of the policy gradient
            A = (((adv - adv.mean((0, 1))) / adv.std((0, 1)).clamp(min=a.adv_floor)) * w).sum(-1)     # the floor keeps a group with a near-constant reward (footholds on flat ground) from being blown up into noise
        else:
            A = (adv * w).sum(-1)
        A = (A - A.mean()) / (A.std() + 1e-6)
        # PPO with mirrored samples (Sec. 3.3)
        o, s, ac, lp, R, Vo, Ad = (x.flatten(0, 1) for x in (B["obs"], B["priv"], B["a"], B["logp"], ret, B["v"], A))
        mu_old_all, std_old = B["mu"].flatten(0, 1), actor.std().detach().clone()
        lab = B["idx"].flatten(0, 1)
        gate_hit, gate_n = 0.0, 0
        total, batch = len(o), len(o) // MINI
        kl_m = pi_l = v_l = 0.0
        std_scale = None
        for ep in range(EPOCHS):
            perm = torch.randperm(total, device=dev)
            for m in range(MINI):
                b = perm[m * batch:(m + 1) * batch]
                o_b = torch.cat((o[b], M.actor_obs(o[b])))
                s_b = torch.cat((s[b], M.priv(s[b])))
                a_b = torch.cat((ac[b], M.joints(ac[b])))
                old, Rb, Vb, Adb = lp[b].repeat(2), R[b].repeat(2, 1), Vo[b].repeat(2, 1), Ad[b].repeat(2)
                mu = actor(o_b)
                std = actor.std()
                new = (-0.5 * ((a_b - mu) / std) ** 2 - std.log() - 0.9189385).sum(-1)
                ent = (std.log() + 1.4189385).sum()
                ratio = (new - old).clamp(-20.0, 5.0).exp()
                l1 = -(Adb * ratio)
                l2 = -(Adb * ratio.clamp(1 - CLIP, 1 + CLIP))
                pi_loss = torch.max(l1, l2).mean()
                v = critics(s_b)
                vc = Vb + (v - Vb).clamp(-CLIP, CLIP)
                v_loss = torch.max((v - Rb) ** 2, (vc - Rb) ** 2).mean()
                gate = actor.gate(actor.norm(o_b))                 # soft label: terrain category i -> expert i; the deployed policy never needs the label
                gate_loss = F.cross_entropy(gate, lab[b].repeat(2))
                gate_hit += (gate.argmax(-1) == lab[b].repeat(2)).float().sum().item()
                gate_n += len(o_b)
                loss = pi_loss + VF * v_loss - a.ent * ent + a.gate_prior * gate_loss
                safe_step(opt_ppo, list(actor.parameters()) + list(critics.parameters()), loss, guard, 1.0)
                with torch.no_grad():                               # true Gaussian KL, adaptive step per mini-batch (rsl_rl's schedule)
                    # the original samples only: the mirrored ones also carry the policy's left/right asymmetry, which is not a step
                    kl = (torch.log(std / std_old) + (std_old ** 2 + (mu_old_all[b] - mu[:len(b)]) ** 2) / (2 * std ** 2) - 0.5).sum(-1).mean()
                    kl_m += kl.item() / (EPOCHS * MINI)
                    lr_ppo = max(1e-5, lr_ppo / 1.5) if kl.item() > KL_TARGET * 2 else (min(1e-3, lr_ppo * 1.5) if kl.item() < KL_TARGET / 2 else lr_ppo)
                    if a.fixed_lr > 0:
                        lr_ppo = a.fixed_lr
                    for g_ in opt_ppo.param_groups:
                        g_["lr"] = lr_ppo
                    pi_l += pi_loss.item() / (EPOCHS * MINI)
                    v_l += v_loss.item() / (EPOCHS * MINI)
        # encoder + estimator: GRU rolled over the stored sequence
        chunk = max(1, N_ // 4)
        enc_l = {"hb": 0.0, "hf": 0.0, "np": 0.0, "kl": 0.0, "v": 0.0}
        order = torch.randperm(N_, device=dev)
        for c in range(0, N_, chunk):
            ids = order[c:c + chunk]
            hc = h0[ids]
            loss = 0.0
            L = dict(hb=0.0, hf=0.0, np=0.0, kl=0.0, v=0.0)
            for t in range(Tn):
                e = enc.step(B["hist"][t, ids], B["depth"][t, ids], hc, map_img=None if B_map is None else B_map[t, ids])
                hc = e["h"]
                L["hb"] = L["hb"] + F.mse_loss(aux.dec_b(e["z_b"]), B["hb"][t, ids])
                L["hf"] = L["hf"] + F.mse_loss(aux.dec_f(e["z_f"]), B["hf"][t, ids])
                L["np"] = L["np"] + F.mse_loss(aux.dec_p(e["z_p"]), B["nprop"][t, ids])
                L["kl"] = L["kl"] - 0.5 * (1 + e["logv"] - e["mu"] ** 2 - e["logv"].exp()).mean()
                L["v"] = L["v"] + F.mse_loss(aux.est(B["hist"][t, ids].flatten(1)), B["priv"][t, ids, PROP_DIM:PROP_DIM + 3])
                keep = (1 - B["done"][t, ids])[:, None]
                hc = hc * keep
            loss = (2.0 * L["hb"] + 1.0 * L["hf"] + 5.0 * L["np"] + 1.0 * L["kl"] + 2.0 * L["v"]) / Tn
            safe_step(opt_enc, list(enc.parameters()) + list(aux.parameters()), loss, guard, 1.0)
            for k in enc_l:
                enc_l[k] += float(L[k].detach()) / Tn / max(1, (N_ // chunk))
        # foothold imagination (Eq. 1): first valid future contact per foot, in the heading frame at t
        with torch.no_grad():
            tgt = torch.zeros(Tn, N_, 2, 2, device=dev)
            ok = torch.zeros(Tn, N_, 2, dtype=torch.bool, device=dev)
            have = torch.zeros(N_, 2, dtype=torch.bool, device=dev)
            pos = torch.zeros(N_, 2, 2, device=dev)
            for t in reversed(range(Tn)):
                have &= (B["done"][t] < 0.5)[:, None]
                td = B["touch"][t]
                pos = torch.where(td[..., None], B["sole"][t][..., :2], pos)
                have |= td
                d = pos - B["base"][t][:, None]
                c, sn = torch.cos(B["yaw"][t])[:, None], torch.sin(B["yaw"][t])[:, None]
                tgt[t] = torch.stack((c * d[..., 0] + sn * d[..., 1], -sn * d[..., 0] + c * d[..., 1]), -1)
                ok[t] = have
        fm, fs = foot(s, ac)
        tg, mk = tgt.flatten(0, 1), ok.flatten(0, 1).float()
        nll = ((tg - fm).pow(2).sum(-1) / (2 * fs ** 2) + 2 * fs.log()) * mk
        foot_loss = nll.sum() / mk.sum().clamp(min=1.0)
        safe_step(opt_foot, list(foot.parameters()), foot_loss, guard, 1.0)
        f_err = ((tg - fm).norm(dim=-1) * mk).sum().item() / max(1.0, mk.sum().item())
        # multi-discriminator AMP
        ax, ai = B["amp"].flatten(0, 1), B["idx"].flatten(0, 1)
        disc.norm.update(ax)
        d_real = d_fake = 0.0
        disc_log = {}
        for i in range(5):
            mk_i = ai == i
            if int(mk_i.sum()) < 16:
                continue
            xf = ax[mk_i]
            xf = xf[torch.randperm(len(xf), device=dev)[:4096]]
            xr = amp.sample(i, len(xf))
            xr.requires_grad_(True)
            dr, df = disc(xr, i), disc(xf, i)
            gp = torch.autograd.grad(dr.sum(), xr, create_graph=True)[0].pow(2).sum(-1).mean()
            loss = ((dr - 1) ** 2).mean() + ((df + 1) ** 2).mean() + 20.0 * gp
            safe_step(opt_disc, list(disc.parameters()), loss, guard, 10.0)
            disc_log[T.TYPES[i]] = dict(loss=round(loss.item(), 4), d_data=round(dr.mean().item(), 3), d_policy=round(df.mean().item(), 3), grad_pen=round(gp.item(), 5),
                                        style_reward=round((sty_sum[i] / sty_n[i].clamp(min=1)).item(), 4), share=round((sty_n[i] / sty_n.sum()).item(), 3))
            d_real += dr.mean().item() / 5
            d_fake += df.mean().item() / 5
        cnt_ = env.cnt
        air_td = float(cnt_.get("air_sum", 0.0)) / max(float(cnt_.get("air_n", 0.0)), 1.0)
        gate_hits = gate_hits + 1 if air_td > a.gate_air else 0
        if (a.prog_gate > 0 or a.style_low >= 0) and gate_start is None and gate_hits >= 5:
            gate_start = it + 1
            print("progress gate: touchdown air time", round(air_td, 3), "for 5 iterations; ramping the gate off from", gate_start, flush=True)
        if it % a.log_every == 0:
            lv = [round(env.level[(env.type == c) & ~env.free].float().mean().item(), 2) for c in range(5)]
            cnt = {k_: float(v_) for k_, v_ in env.cnt.items()}
            n_done = max(cnt.get("done_any", 0.0), 1.0)
            gu, gm = gpu_now()
            extra = dict(terms_raw={k_: round(v_ / TERM_W[k_], 4) for k_, v_ in term_sum.items() if k_ in TERM_W},
                         group_weighted=dict(locomotion=round(rs[0] * float(w[0]), 4), foothold=round(rs[1] * float(w[1]), 4), style=round(rs[2] * float(w[2]), 4)),
                         disc=disc_log, foot_nll=round(float(foot_loss.detach()), 4), entropy_std=[round(x, 3) for x in actor.std().tolist()],
                         done_share={k_[5:]: round(v_ / n_done, 4) for k_, v_ in cnt.items() if k_.startswith("done_") and k_ != "done_any"}, dones_per_iter=cnt.get("done_any", 0.0),
                         foothold=dict(touchdowns=cnt.get("td_n", 0.0), msr=round(cnt.get("td_support", 0.0) / max(cnt.get("td_n", 0.0), 1.0), 4),
                                       sfr=round(cnt.get("td_safe", 0.0) / max(cnt.get("td_n", 0.0), 1.0), 4)),
                         levels_by_type=dict(zip(T.TYPES, lv)), level_max_share=round((env.level[~env.free] >= T.NLEVEL - 1).float().mean().item(), 4),
                         samples_per_s=round(N_ * Tn / max(time.time() - t0, 1e-6)), num_envs=N_, air_td=round(air_td, 3), air_long_share=round(float(cnt.get("air_long", 0.0)) / max(float(cnt.get("air_n", 0.0)), 1.0), 3), prog_mean=round(float(cnt.get("prog_sum", 0.0)) / max(float(cnt.get("prog_n", 0.0)), 1.0), 3), prog_gate=round(env.prog_gate, 3), gate_start=gate_start, flat_phase=bool(env.flat_only), obst_share=round(float((env.type != T.FLAT).float().mean()), 3), obst_target=env.obst_target, w_style_now=round(float(w[2]), 4), gpu_util=gu, gpu_mem_mb=gm)
            rec = dict(it=it, dt=round(time.time() - t0, 2), elapsed_h=round((time.time() - t_start) / 3600, 3), r_l=round(rs[0], 3), r_f=round(rs[1], 3), r_s=round(rs[2], 3),
                       ep_ret=round(float(np.mean(done_stats["ret"])) if done_stats["ret"] else 0.0, 2), ep_len_s=round(float(np.mean(done_stats["len"]) if done_stats["len"] else 0.0) * 0.02, 2),
                       n_done=len(done_stats["len"]), level=lv, kl=round(kl_m, 4), lr=lr_ppo, pi=round(pi_l, 4), v=round(v_l, 3), std=round(actor.std().mean().item(), 3),
                       hb=round(enc_l["hb"], 4), hf=round(enc_l["hf"], 4), np=round(enc_l["np"], 4), v_est=round(enc_l["v"], 4), foot_cm=round(100 * f_err, 1),
                       d_r=round(d_real, 3), d_f=round(d_fake, 3), gpu_gb=round(torch.cuda.max_memory_allocated() / 1e9, 1), nan=env.log.get("nan", 0), touch=env.log.get("touch", 0), stuck=env.log.get("stuck", 0), gate_acc=round(gate_hit / max(gate_n, 1), 3), skipped=guard.get("skipped", 0), **extra, guide_on=env.guide_on, stage_ramp=round(ramp, 3), v_min=round(env.v_min, 3), sig_v=round(env.sig_v, 4), stuck_on=bool(env.stuck_term), adv_sd=[round(x, 5) for x in adv_sd], lvl_mean=round(env.log.get("level", 0.0), 2),
                       gait={k_: round(v_ / max(env.gait.get("n", 1), 1), 3) for k_, v_ in env.gait.items() if k_ != "n"}, terms={k_: round(v_, 4) for k_, v_ in term_sum.items()})
            print(json.dumps(rec), flush=True)
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()
            done_stats = {"ret": [], "len": []}
        if not all_finite(enc, aux, actor, critics, foot, disc):
            raise SystemExit("non-finite network parameters at iteration %d: not saving; the supervisor resumes from the last good checkpoint" % it)
        if it % a.save_every == 0 or it == a.iters - 1:
            ck = dict(iter=it, enc=enc.state_dict(), aux=aux.state_dict(), actor=actor.state_dict(), critics=critics.state_dict(), foot=foot.state_dict(), disc=disc.state_dict(),
                      opt_ppo=opt_ppo.state_dict(), opt_enc=opt_enc.state_dict(), opt_foot=opt_foot.state_dict(), opt_disc=opt_disc.state_dict(), level=env.level.cpu(), type=env.type.cpu(),
                      guide_on=env.guide_on, gate_start=gate_start, args=vars(a))
            torch.save(ck, out / "last.tmp")
            if (out / "last.pt").exists():
                os.replace(out / "last.pt", out / "prev.pt")      # the newest two are always kept
            os.replace(out / "last.tmp", out / "last.pt")     # never a half-written last.pt
            if it % a.ckpt_every == 0:
                torch.save(ck, out / f"ckpt_{it:06d}.pt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_envs", type=int, default=4096)
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--name", default="dev")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default="")
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--save_every", type=int, default=100)
    ap.add_argument("--camera", default="head", choices=("head", "chest", "chest62", "chest85", "torso60", "chest_rs"))
    ap.add_argument("--plant", default="mc_20261024", help="user decision 2026-11-11: the measured motion-controller gains; vendor_20260823 = runs r1-r4, d1, d2")
    ap.add_argument("--map_memory", action="store_true")
    ap.add_argument("--gate_prior", type=float, default=0.0, help="not in the paper; 0.02 was the earlier setting")
    ap.add_argument("--out_root", default="")
    ap.add_argument("--ckpt_every", type=int, default=1000)
    ap.add_argument("--rsi_frac", type=float, default=0.5, help="share of walking episodes started from captured walking frames; 0 = default pose only")
    ap.add_argument("--init_std", type=float, default=1.0, help="rsl_rl default; the earlier runs used 0.5")
    ap.add_argument("--rew_scale", type=float, default=0.02, help="all rewards x control dt (legged_gym); 1.0 = earlier behaviour")
    ap.add_argument("--adv_norm", default="joint", choices=("joint", "separate"))
    ap.add_argument("--w_style", type=float, default=0.2)
    ap.add_argument("--cmd_cur_iters", type=int, default=0, help="command curriculum (deviation): hold cmd_min0 and sig_v0 until this iteration, release linearly by twice it; 0 = off")
    ap.add_argument("--cmd_min0", type=float, default=0.0, help="minimum commanded speed while the curriculum holds (m/s)")
    ap.add_argument("--sig_v0", type=float, default=0.0, help="velocity-tracking kernel width while it holds (paper 0.25); 0 = paper")
    ap.add_argument("--stuck_until", type=int, default=0, help="with --stuck_term: switch the no-progress termination off from this iteration; 0 = keep")
    ap.add_argument("--stage_a", type=int, default=0, help="staged schedule (deviation from the paper): until this iteration only tracking + posture terms, no style")
    ap.add_argument("--stage_b", type=int, default=0, help="...then the remaining Table 6 terms and the style weight ramp linearly to full by this iteration; 0 = off")
    ap.add_argument("--amp_file", default="data/amp.npz", help="data/amp_v2.npz = flat set rebuilt from all 999 s of selected flat clips")
    ap.add_argument("--prior_balance", action="store_true", help="sample reference windows with equal mass per 0.2 m/s speed bin")
    ap.add_argument("--ent", type=float, default=ENT, help="entropy coefficient (paper 0.005)")
    ap.add_argument("--flat_only", action="store_true", help="diagnosis: all robots on the open flat tile, no curriculum")
    ap.add_argument("--only_terms", default="", help="diagnosis: comma list of locomotion terms to keep; the rest are zeroed")
    ap.add_argument("--fixed_lr", type=float, default=0.0, help="diagnosis: fixed PPO step size instead of the KL-adaptive one")
    ap.add_argument("--prior_min_speed", type=float, default=0.0, help="drop discriminator reference windows whose mean planar speed is below this (m/s); 0 = keep all")
    ap.add_argument("--rsi_min_speed", type=float, default=0.0, help="walking starts only from frames moving forward at least this fast (m/s); 0 = any frame")
    ap.add_argument("--adv_floor", type=float, default=2e-3)
    ap.add_argument("--boot_iters", type=int, default=0, help="early schedule length (earlier runs: 3000); 0 = the paper's rewards from the start")
    ap.add_argument("--style_from", type=int, default=0, help="deviation: style weight 0 before this iteration, ramped to --w_style over --gate_ramp; overrides --style_low; 0 = off")
    ap.add_argument("--feet_air_w", type=float, default=0.0, help="deviation: feet-air-time reward weight (per touchdown, W * clamp(air - feet_air_min, 0, 0.4 - feet_air_min)); 0 = off")
    ap.add_argument("--feet_air_min", type=float, default=0.1, help="air time below which a touchdown earns nothing")
    ap.add_argument("--obst0", type=float, default=0.25, help="with --flat_until and --obst_ramp: share of robots on obstacle tiles at the switch")
    ap.add_argument("--obst_ramp", type=int, default=0, help="with --flat_until: iterations over which the obstacle share rises from --obst0 to 0.8; 0 = all at once")
    ap.add_argument("--flat_until", type=int, default=0, help="deviation: all robots on the flat tile until this iteration, then terrain types redrawn at level 0 and the curriculum on; 0 = off")
    ap.add_argument("--style_low", type=float, default=-1.0, help="t_f (deviation): style weight x this until touchdown air time > --gate_air for 5 iterations, then ramped to full over --gate_ramp; -1 = off")
    ap.add_argument("--prog_gate", type=float, default=0.0, help="t_e (deviation): multiply orientation, yaw, base height, roll/pitch rate and lateral terms by 1 - g(1 - progress) for robots with a moving command; 0 = off")
    ap.add_argument("--gate_air", type=float, default=0.2, help="with --prog_gate: start ramping the gate off once mean touchdown air time exceeds this for 5 iterations")
    ap.add_argument("--gate_ramp", type=int, default=500, help="with --prog_gate: iterations over which the gate ramps off")
    ap.add_argument("--stuck_term", action="store_true", help="end episodes that make no progress (not in the paper)")
    ap.add_argument("--imit_w", type=float, default=0.0)
    train(ap.parse_args())
