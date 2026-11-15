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

import sys

if "--paper_terrain" in sys.argv:                   # v15 terrain layout (ssr/terrain.py PAPER): read at import, so set before the ssr imports
    os.environ["X2_TERRAIN_PAPER"] = "1"

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
    def __init__(self, device, min_speed=0.0, file="data/amp.npz", balance=False, q_shift=None, turn_min=0.0):
        z = np.load(ROOT / file)
        self.frames, self.valid, self.weight, self.fed = [], [], [], {}
        for name in T.TYPES:
            f = torch.as_tensor(z[name], device=device, dtype=torch.float32)
            if q_shift is not None:                         # step 2: frames store q - plant default; re-express against the new default
                f[:, 9:30] += torch.as_tensor(q_shift, device=device, dtype=torch.float32)
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
                keep = win >= min_speed
                if turn_min > 0:                            # deviation (2026-11-12): turning windows (|yaw rate| >= turn_min) stay even when slow
                    wz_ = np.convolve(np.abs(z[name][:, 8]), np.ones(AMP_FRAMES) / AMP_FRAMES)[AMP_FRAMES - 1:]
                    wz_ = np.concatenate((wz_, np.zeros(n - len(wz_))))[:n]
                    keep |= wz_ >= turn_min
                ok &= keep
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
    """v15: one discriminator per terrain type, no flat one (Table 11); omnidirectional robots on the free area keep the
    discriminator of the terrain they came from (v14: a separate flat discriminator)."""
    if T.PAPER:
        return env.type
    return torch.where(env.free, torch.full_like(env.type, T.FLAT), env.type)   # v14: the omni robots use the flat discriminator


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



def diag_stats(adv, w, B, std, a_floor, H=10):
    """Frozen-policy diagnostic (--diag_out): which way each advantage setting pushes forward speed and double support on the
    same rollouts. y = mean vx / commanded speed over the next H steps (walkers told >= 0.3 m/s, same episode), z = double-support
    share over the same window. corr(A, y) > 0: the PPO step favours the actions followed by faster walking."""
    Tn, N_ = adv.shape[:2]
    vxr, wk, dbl, dn = B["vxr"], B["wk"], B["dbl"], B["done"]
    y = torch.zeros(Tn, N_, device=adv.device); z = torch.zeros_like(y); ok = torch.zeros_like(y, dtype=torch.bool)
    for t in range(Tn - H + 1):
        alive = (dn[t:t + H - 1].sum(0) == 0) if H > 1 else torch.ones(N_, dtype=torch.bool, device=adv.device)
        ok[t] = alive & (wk[t:t + H].min(0).values > 0.5)
        y[t], z[t] = vxr[t:t + H].mean(0), dbl[t:t + H].mean(0)
    m = ok.flatten()
    if int(m.sum()) < 50:
        return dict(n=int(m.sum()))
    sd = adv.std((0, 1))
    Aj = (adv * w).sum(-1)
    As = (((adv - adv.mean((0, 1))) / sd.clamp(min=a_floor)) * w).sum(-1)
    Ab = lambda b: (1 - b) * Aj / (Aj.std() + 1e-6) + b * As / (As.std() + 1e-6)
    X = dict(joint=Aj, separate=As, blend10=Ab(0.1), blend50=Ab(0.5), loco=adv[..., 0], foot=adv[..., 1], style=adv[..., 2])
    yy, zz = y.flatten()[m], z.flatten()[m]
    def corr(u, v):
        u, v = u - u.mean(), v - v.mean()
        return float((u * v).mean() / (u.std() * v.std() + 1e-12))
    out = dict(n=int(m.sum()), y_mean=round(float(yy.mean()), 4), z_mean=round(float(zz.mean()), 4), adv_sd=[round(float(x), 5) for x in sd])
    for k, v in X.items():
        vv = v.flatten()[m]
        out["cy_" + k], out["cz_" + k] = round(corr(vv, yy), 4), round(corr(vv, zz), 4)
    # each group's part of cov(A, y) under joint (weights w) and separate (weights w / sd) normalisation, as shares of the total
    cov = torch.stack([((adv[..., k].flatten()[m] - adv[..., k].flatten()[m].mean()) * (yy - yy.mean())).mean() for k in range(3)])
    cj, cs = cov * w, cov * w / sd.clamp(min=a_floor)
    out["covshare_joint"] = [round(float(x / cj.abs().sum()), 4) for x in cj]
    out["covshare_separate"] = [round(float(x / cs.abs().sum()), 4) for x in cs]
    # the same through the action noise only (independent of the state): speed direction beta estimated on half of the robots,
    # evaluated on the other half; corr(A, eps . beta) > 0 means the step moves the mean action toward the faster outcome
    eps = ((B["a"] - B["mu"]) / std).flatten(0, 1)[m]
    half = (torch.arange(N_, device=adv.device).repeat(Tn, 1).flatten()[m] % 2) == 0
    for tag, tgt in (("y", yy), ("z", zz)):
        r_ = {}
        for k, v in X.items():
            vv = v.flatten()[m]; acc = 0.0
            for h in (half, ~half):
                beta = (eps[h] * (tgt[h] - tgt[h].mean())[:, None]).mean(0)
                acc += 0.5 * corr(vv[~h], eps[~h] @ beta)
            r_[k] = round(acc, 4)
        out["eps_c" + tag] = r_
    return out


def train(a):
    dev = a.device
    if a.torch_seed >= 0:                            # 2026-11-13 v4 restart: seed torch / numpy / random (terrain keeps --seed); default -1 = unseeded as before
        import random as _random
        torch.manual_seed(a.torch_seed); torch.cuda.manual_seed_all(a.torch_seed); np.random.seed(a.torch_seed); _random.seed(a.torch_seed)
        print("torch seed", a.torch_seed, flush=True)
    os.environ["X2_PLANT"] = a.plant                 # read by ssr.plant_config.load_plant when the model is built
    out = pathlib.Path(a.out_root or os.environ.get("SSR_RUNS") or ROOT / "runs") / a.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(a)))
    if a.excl_pelvis_hiproll:
        os.environ["X2_EXCL_PELVIS_HIPROLL"] = "1"
    if a.default_from_ref:
        os.environ["X2_DEFAULT_FROM_REF"] = "1"
    if a.action_scale_paper:
        os.environ["X2_ACTION_SCALE_PAPER"] = "1"
    if a.ext_frac > 0:                               # extended terrain map (user decision 2026-11-12)
        from ssr import terrain as T_
        _r = T_.add_extended(T_.generate(a.seed), types=tuple(a.ext_types.split(",")), variants=a.ext_variants, min_along=a.ext_min_clear, stairs_inset=a.ext_stairs_inset, prop_routes=a.ext_prop_routes, stations=a.ext_stations, slope_turns=a.ext_slope_turns)
        env = SSREnv(a.num_envs, dev, seed=a.seed, camera=a.camera, map_memory=a.map_memory, raster=_r, tiles=True, ext_phys=a.ext_phys)
        env.ext_type[:] = torch.randint(0, len(_r["ext"]["types"]), (a.num_envs,), device=dev)
        print("extended map:", len(_r["ext"]["tiles"]), "tiles,", len(_r["ext"]["routes"]), "routes, raster", _r["z"].shape, flush=True)
    else:
        env = SSREnv(a.num_envs, dev, seed=a.seed, camera=a.camera, map_memory=a.map_memory)
    if a.dof_dev_ref:
        _pa = json.load(open(ROOT.parent / "eval/port_audit.json"))["reference_vs_default"]["flat"]
        _ref = env.q0.clone()
        for _i, _n in enumerate(__import__("ssr.x2", fromlist=["NAMES21"]).NAMES21[:12]):
            _ref[_i] = float(_pa[_n]["mean"])
        env.q_dev_ref = _ref
        print("dof_dev measured from the reference mean leg pose:", [round(float(v_), 3) for v_ in _ref[:12]], flush=True)
    if a.excl_pelvis_hiproll:
        print("pelvis / hip-roll contacts excluded: nexclude =", env.x2.model.nexclude, flush=True)
    _shift = None
    if a.default_from_ref:
        _x = env.x2
        _shift = (_x.default_q_plant - _x.default_q)[_x.idx21]
        print("default leg pose from the reference mean; prior shifted by", [round(float(v_), 3) for v_ in _shift[:12]], flush=True)
    amp = AMPData(dev, a.prior_min_speed, a.amp_file, a.prior_balance, q_shift=_shift, turn_min=a.prior_turn_min)
    print("motion prior as fed to each discriminator (window mean planar speed, m/s):", json.dumps(amp.fed), flush=True)
    print("motion prior windows kept:", amp.kept, flush=True)
    if a.rsi_frac > 0:
        if T.FLAT >= len(amp.frames):
            raise SystemExit("--rsi_frac > 0 needs a flat prior set; v15 has none (pass --rsi_frac 0)")
        env.set_rsi(amp.frames[T.FLAT], a.rsi_frac, a.rsi_min_speed)
        print("walking-start frames:", len(env.rsi_ok), "of", len(env.rsi_frames), flush=True)
    env.imit_w = a.imit_w
    env.stuck_term = a.stuck_term
    env.flat_only = a.flat_only or a.flat_until > 0
    env.feet_air_w, env.feet_air_min = a.feet_air_w, a.feet_air_min
    env.omni_frac, env.omni_turn, env.omni_slow = a.omni_frac, a.omni_turn, a.omni_slow
    env.swing_h_w, env.swing_h_target = a.swing_h_w, a.swing_h_target
    env.back_scale, env.drift_w = a.back_scale, a.drift_w
    env.creep_w, env.ext_hold_r = a.creep_w, a.ext_hold_r
    env.rel_kernel = a.rel_kernel
    env.edge_stop_p = a.edge_stop
    env.side_soft = a.side_soft
    env.omni_top, env.promote_half, env.vy_clamp = a.omni_top, a.promote_half, not a.no_vy_clamp      # v15 paper items (A.1, A.2)
    env.cur_rudin, env.cur_fair, env.swing_prog = a.cur_rudin, a.cur_fair, a.swing_prog            # v20 stairs refusal (default off)
    if a.climb_types:
        env.climb_types = torch.tensor([nm in a.climb_types.split(",") for nm in T.TYPES], device=dev)
        env.climb_side_noterm, env.climb_creep = a.climb_side_noterm, a.climb_creep
        print("v20 climb tiles", [nm for nm in T.TYPES if nm in a.climb_types.split(",")], "side no-term", a.climb_side_noterm, "creep", a.climb_creep, flush=True)
    if a.cur_rudin or a.swing_prog:
        print("v20 curriculum rudin", a.cur_rudin, "fair", a.cur_fair, "swing_prog", a.swing_prog, flush=True)
    env.slope_back_ban, env.q_noise, env.foot_base_frame = not a.slope_back, a.init_q_noise, a.foot_base_frame
    env.beam_hmax = a.beam_hmax
    if a.beam_hmax != 30.0:
        print("v21 beam heading +-", a.beam_hmax, "deg", flush=True)
    env.home_type = env.type.clone() if a.ext_home else None
    env.simcheck = a.simcheck
    env.ext_frac = a.ext_frac
    env.resample_steps = int(round(a.resample_s / 0.02)) if a.resample_s > 0 else 0
    env.only_terms = set(a.only_terms.split(",")) if a.only_terms else None
    if a.flat_only or a.flat_until > 0:
        env.reset(torch.arange(a.num_envs, device=dev))
        env.refresh()
        env._contacts()
        env._update_obs(torch.arange(a.num_envs, device=dev))     # also refreshes the stored foot positions: without it the first step saw the feet jump (one huge slip penalty at iteration 0 in d4 and t_b)
    enc, aux, actor, critics, foot, disc = (m.to(dev) for m in (N.Encoder(a.map_memory), N.Aux(), N.MoEActor(), N.Critics(), N.Foothold(), N.Discriminators()))
    if a.disc_part_w:                                             # v15 (user rule 2026-11-13): style input weighted by body part, on the normalised input
        wl, wa, wh = (float(x_) for x_ in a.disc_part_w.split(","))  # legs / pelvis / waist, shoulders / elbows, wrists (hand points) / head
        pw = torch.full((AMP_DIM,), wl, device=dev)                  # 0:9 gravity, base lin / ang vel (pelvis); 9:21 leg q; 21 waist q; 30:42 leg qd; 42 waist qd; 57:63 feet
        pw[22:30], pw[43:51] = wa, wa                                # arm q, arm qd (shoulder pitch / roll / yaw, elbow; the 21-joint set has no wrist joints)
        pw[51:57] = wh                                               # hand / wrist positions; the AMP frame has no head feature
        disc.part_w = pw.repeat(AMP_FRAMES)
        print("discriminator input weights per feature (one frame):", pw.tolist(), flush=True)
    if a.disc_mask_arms:
        mk = torch.ones(AMP_DIM, device=dev)
        mk[22:30], mk[43:51], mk[51:57] = 0.0, 0.0, 0.0           # arm q, arm qd, wrist positions
        disc.mask = mk.repeat(AMP_FRAMES)
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
        if getattr(env, "ext_on", False) and "ext_level" in ck:   # 2026-11-13: restore the extended-map levels per robot
            env.ext_level[:] = ck["ext_level"][pick].to(dev)
            env.ext_type[:] = ck["ext_type"][pick].to(dev)
            print("restored extended-map levels, mean", round(float(env.ext_level.float().mean()), 2), flush=True)
        lr_ppo = opt_ppo.param_groups[0]["lr"]
        if env.home_type is not None:                                # fix 2026-11-13: --ext_home's home types follow the resumed types (was: the types before the resume)
            env.home_type[:] = env.type
        print("resumed", a.resume, start, flush=True)
    warm_until = -1
    if a.init_from and not a.resume:                 # v15: continue the policy from another run's checkpoint; discriminators and optimisers fresh
        ck = torch.load(a.init_from, map_location=dev, weights_only=False)
        for k, m in (("enc", enc), ("aux", aux), ("actor", actor), ("critics", critics), ("foot", foot)):
            m.load_state_dict(ck[k])
        start = ck["iter"] + 1
        pick = torch.arange(a.num_envs) % len(ck["level"])
        env.level[:] = ck["level"][pick].to(dev)
        env.type[:] = ck["type"][pick].to(dev)
        was_flat = (env.type == T.DISCRETE) if T.PAPER else torch.zeros_like(env.type, dtype=torch.bool)   # paper layout: index 4 was "flat" up to v14, those robots start the discrete terrain at level 0; v14 layout: unchanged
        env.level[was_flat] = 0
        env.ep[:] = 1                                               # keep these types / levels at the first reset
        if env.home_type is not None:
            env.home_type[:] = env.type
        lr_ppo = float(ck["opt_ppo"]["param_groups"][0]["lr"])
        for g_ in opt_ppo.param_groups:
            g_["lr"] = lr_ppo
        warm_until = start + a.critic_warmup
        allr = torch.arange(a.num_envs, device=dev)
        env.reset(allr)
        env.refresh()
        env._contacts()
        env._update_obs(allr)
        print("v15 init from", a.init_from, "iter", ck["iter"], "; discriminators fresh; types", torch.bincount(env.type, minlength=5).tolist(),
              "; discrete (was flat) robots at level 0:", int(was_flat.sum()), "; critic warm-up (policy frozen) until", warm_until, flush=True)
    elif a.resume:
        warm_until = int(ck.get("warm_until", -1))
    if a.resume and a.warm_until_abs > 0:            # 2026-11-13 v4 restart: policy frozen (critics + discriminators train) until this absolute iteration; idempotent over crash resumes
        warm_until = max(warm_until, a.warm_until_abs)
        print("policy frozen (critic / discriminator warm-up) until", warm_until, flush=True)
    # E7 (2026-11-13): arm action noise. Arms were hidden from the discriminators until 11,350, so their stds drifted up under the
    # entropy bonus (0.5-1.0 x action scale); since the discriminators see arms, that noise collapses the style reward. Off by default.
    ARM_IDX, ANKLE_ROLL_IDX = list(range(13, 21)), [5, 11]           # NAMES21: right / left shoulder pitch, roll, yaw, elbow; ankle roll R, L
    std_caps = [(ARM_IDX, math.log(a.arm_std_cap))] if a.arm_std_cap > 0 else []
    if a.ankle_roll_std_cap > 0:
        std_caps.append((ANKLE_ROLL_IDX, math.log(a.ankle_roll_std_cap)))
    arm_std_reset_done = bool(ck.get("arm_std_reset_done", False)) if a.resume else False
    # E7b (2026-11-14): the reset is once per run AND per value. v18 saves carry done=True without a value (theirs was 0.3), so a
    # different --arm_std_reset (0.2) resets again once; a crash-resume of a save made with the same value does not.
    arm_bias_shift_done = bool(ck.get("arm_bias_shift_done", False)) if a.resume else False
    arm_std_reset_val = float(ck.get("arm_std_reset_val", 0.3 if arm_std_reset_done else 0.0)) if a.resume else 0.0

    def _cap_std():
        with torch.no_grad():
            for idx_, lc_ in std_caps:
                actor.log_std.data[idx_] = actor.log_std.data[idx_].clamp(max=lc_)

    if a.arm_std_reset > 0 or std_caps:
        print("arm std before reset / cap:", [round(x, 3) for x in actor.std()[ARM_IDX].tolist()],
              "ankle roll", [round(x, 3) for x in actor.std()[ANKLE_ROLL_IDX].tolist()], flush=True)
    if a.arm_std_reset > 0 and (not arm_std_reset_done or abs(arm_std_reset_val - a.arm_std_reset) > 1e-9):           # once per run: the flag is saved in the checkpoints, so a crash-resume does not reset again
        with torch.no_grad():
            actor.log_std.data[ARM_IDX] = math.log(a.arm_std_reset)
        st_ = opt_ppo.state.get(actor.log_std, {})
        if "exp_avg" in st_:                                        # drop Adam's first moment for these entries (it carries the old upward drift)
            st_["exp_avg"][ARM_IDX] = 0.0
        arm_std_reset_done, arm_std_reset_val = True, a.arm_std_reset
        print("arm std reset to", a.arm_std_reset, flush=True)
    _cap_std()
    if a.arm_std_reset > 0 or std_caps:
        print("arm std after reset / cap:", [round(x, 3) for x in actor.std()[ARM_IDX].tolist()],
              "ankle roll", [round(x, 3) for x in actor.std()[ANKLE_ROLL_IDX].tolist()], "caps", [(i[0], round(math.exp(c), 3)) for i, c in std_caps], flush=True)
    # E7b (2026-11-14): arm action-bound loss. env.step clamps the PD target to the joint range, so an action mean past a stop costs
    # nothing and gives PPO no signal (r7 v18 16,500: elbow mean 6.8 -> target far past the 0.0 rad stop, pinned in every sample).
    # Penalise the arm action MEANS outside the action interval that maps into [q_lo + m, q_hi - m] (squared excess, as the AMP / ASE
    # bound loss; the SSR paper is silent on action bounds). Off by default (coef 0).
    if a.arm_bound_coef > 0:
        _ab_lo = ((env.q_lo + a.arm_bound_margin - env.q0) / env.scale)[ARM_IDX].to(dev)
        _ab_hi = ((env.q_hi - a.arm_bound_margin - env.q0) / env.scale)[ARM_IDX].to(dev)
        print("arm action bounds (action units) lo", [round(x, 2) for x in _ab_lo.tolist()], "hi", [round(x, 2) for x in _ab_hi.tolist()],
              "coef", a.arm_bound_coef, "margin rad", a.arm_bound_margin, flush=True)
    if a.diag_disc_from:                              # diagnostic: another run's discriminators (e.g. v14's amp_v4 ones) judge this run's motion
        disc.load_state_dict(torch.load(a.diag_disc_from, map_location=dev, weights_only=False)["disc"])
        print("diag: discriminators from", a.diag_disc_from, flush=True)
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
    _ab_from = a.arm_bound_from if a.arm_bound_from > 0 else start
    if a.arm_bound_coef > 0 and a.arm_bound_ramp > 0:
        print("arm bound coef ramps 0 ->", a.arm_bound_coef, "over", a.arm_bound_ramp, "iterations from", _ab_from, flush=True)
    for it in range(start, a.iters):
        t0 = time.time()
        B = dict(hist=z0(H, PROP_DIM), depth=z0(DEPTH_N, DEPTH_N), obs=z0(N.ACTOR_DIM), priv=z0(PRIV_DIM), a=z0(21), mu=z0(21), logp=z0(), v=z0(3), r=z0(3),
                 done=z0(), term=z0(), amp=z0(AMP_DIM * AMP_FRAMES), idx=torch.zeros(Tn, N_, dtype=torch.long, device=dev),
                 hb=z0(HB_NX * HB_NY), hf=z0(2 * HF_NX * HF_NY), sole=z0(2, 3), touch=torch.zeros(Tn, N_, 2, dtype=torch.bool, device=dev),
                 base=z0(2), yaw=z0(), nprop=z0(PROP_DIM), p3=z0(3), R=z0(3, 3), vxr=z0(), wk=z0(), dbl=z0())
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
        if a.sig_v_to > 0:                           # deviation (r6d, 2026-11-12): velocity kernel narrowed for slow commands, ramped from sig_v_from over sig_v_ramp iterations
            env.sig_v = 0.25 + (a.sig_v_to - 0.25) * min(1.0, max(0.0, (it - a.sig_v_from) / max(a.sig_v_ramp, 1)))
        b_paper = 1.0 if a.to_paper_ramp <= 0 else min(1.0, max(0.0, (it - warm_until) / a.to_paper_ramp))   # v16: resume transition v14 -> paper
        if a.to_paper_ramp > 0 and a.sig_v_start > 0:
            env.sig_v = a.sig_v_start + (0.25 - a.sig_v_start) * b_paper      # v16: Table 7 sigma_v 0.25 reached at warm-up + to_paper_ramp
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
                               ("hb", env.h_b), ("hf", env.h_f.reshape(N_, -1)), ("base", env.s["p"][:, :2]), ("yaw", env.s["yaw"]), ("p3", env.s["p"]), ("R", env.s["R"])):
                    B[k][t] = val
                idx = disc_index(env)
                wk_ = (~env.free & (env.speed.abs() >= 0.3)).float()
                out_s = env.step(act.clamp(-A_CLIP, A_CLIP), fm, fs)
                if a.diag_out:
                    B["vxr"][t] = env.qvel[:, 0] / env.speed.where(env.speed.abs() >= 0.3, torch.ones_like(env.speed))
                    B["wk"][t], B["dbl"][t] = wk_, (out_s["contact"].sum(-1) == 2).float()
                r_s = style_reward(disc, env.amp_hist.flatten(1), idx)
                B["amp"][t], B["idx"][t] = env.amp_hist.flatten(1), idx
                sty_sum.index_add_(0, idx, r_s)
                sty_n.index_add_(0, idx, torch.ones_like(r_s))
                r = torch.stack((out_s["r_l"], out_s["r_f"], r_s), -1)
                r = torch.nan_to_num(r, 0.0, 0.0, 0.0)
                if a.simcheck:
                    r = torch.where(out_s["sim_fault"][:, None], torch.zeros_like(r), r)   # simcheck: a fault step earns nothing
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
            A = (((adv - adv.mean((0, 1))) / adv.std((0, 1)).clamp(min=a.adv_floor)) * w).sum(-1)
            if a.to_paper_ramp > 0 and b_paper < 1.0:   # v16: blend from v14's joint normalisation (shares by raw spread) to the paper's separate one
                Aj = (adv * w).sum(-1)
                A = (1.0 - b_paper) * Aj / (Aj.std() + 1e-6) + b_paper * A / (A.std() + 1e-6)     # the floor keeps a group with a near-constant reward (footholds on flat ground) from being blown up into noise
        else:
            A = (adv * w).sum(-1)
        if a.diag_out:
            with open(a.diag_out, "a") as f_:
                f_.write(json.dumps(dict(it=it, sig_v=env.sig_v, **diag_stats(adv, w, B, actor.std().detach(), a.adv_floor))) + "\n")
        A = (A - A.mean()) / (A.std() + 1e-6)
        # PPO with mirrored samples (Sec. 3.3)
        o, s, ac, lp, R, Vo, Ad = (x.flatten(0, 1) for x in (B["obs"], B["priv"], B["a"], B["logp"], ret, B["v"], A))
        mu_old_all, std_old = B["mu"].flatten(0, 1), actor.std().detach().clone()
        lab = B["idx"].flatten(0, 1)
        gate_hit, gate_n = 0.0, 0
        total, batch = len(o), len(o) // MINI
        kl_m = pi_l = v_l = bnd_l = 0.0
        bnd_c = a.arm_bound_coef * (min(1.0, max(0.0, (it - _ab_from + 1) / a.arm_bound_ramp)) if a.arm_bound_ramp > 0 else 1.0) if a.arm_bound_coef > 0 else 0.0
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
                if a.arm_bound_coef > 0 and (arm_bias_shift_done or not a.arm_bias_shift):   # E7b arm action-bound loss (off by default); with the shift, only after it (trial c8: bound loss before the shift = KL 72)
                    mu_a = mu[:, ARM_IDX]
                    bnd = ((mu_a - _ab_hi).clamp(min=0) ** 2 + (_ab_lo - mu_a).clamp(min=0) ** 2).sum(-1).mean()
                    loss = loss + bnd_c * bnd
                    bnd_l += bnd.item() / (EPOCHS * MINI)
                if it < warm_until or a.diag_out:            # v15 critic warm-up (and the --diag_out replay) after the policy resume: value fit only, the policy is not changed
                    loss = VF * v_loss
                safe_step(opt_ppo, list(actor.parameters()) + list(critics.parameters()), loss, guard, 1.0)
                _cap_std()                                                   # E7 caps (no-op when off)
                with torch.no_grad():                               # true Gaussian KL, adaptive step per mini-batch (rsl_rl's schedule)
                    # the original samples only: the mirrored ones also carry the policy's left/right asymmetry, which is not a step
                    kl = (torch.log(std / std_old) + (std_old ** 2 + (mu_old_all[b] - mu[:len(b)]) ** 2) / (2 * std ** 2) - 0.5).sum(-1).mean()
                    kl_m += kl.item() / (EPOCHS * MINI)
                    lr_ppo = max(1e-5, lr_ppo / 1.5) if kl.item() > KL_TARGET * 2 else (min(1e-3, lr_ppo * 1.5) if kl.item() < KL_TARGET / 2 else lr_ppo)
                    if a.fixed_lr > 0:
                        lr_ppo = a.fixed_lr
                    if it < warm_until or a.diag_out:
                        lr_ppo = opt_ppo.param_groups[0]["lr"]          # the KL of an unchanged policy says nothing: keep the step
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
            pos3 = torch.zeros(N_, 2, 3, device=dev)
            for t in reversed(range(Tn)):
                have &= (B["done"][t] < 0.5)[:, None]
                td = B["touch"][t]
                pos3 = torch.where(td[..., None], B["sole"][t], pos3)
                pos = pos3[..., :2]
                have |= td
                if a.foot_base_frame:                        # v15 (Eq. 1): base-frame xy of the landing point, R^T (p_sole - p_base)
                    tgt[t] = ((pos3 - B["p3"][t][:, None]) @ B["R"][t])[..., :2]
                else:
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
            g_in = torch.autograd.grad(dr.sum(), xr, create_graph=True)[0]
            gp = g_in.pow(2).sum(-1).mean()
            with torch.no_grad():                            # which body parts the discriminator reads: share of the squared input gradient
                g2 = g_in.detach().pow(2).reshape(len(g_in), AMP_FRAMES, AMP_DIM).sum((0, 1))
                g2 = g2 / g2.sum().clamp(min=1e-12)
                part = dict(pelvis=g2[0:9].sum(), legs=g2[9:21].sum() + g2[30:42].sum() + g2[57:63].sum(), waist=g2[21] + g2[42],
                            arms=g2[22:30].sum() + g2[43:51].sum(), hands=g2[51:57].sum())
            loss = ((dr - 1) ** 2).mean() + ((df + 1) ** 2).mean() + 20.0 * gp
            if not a.diag_disc_frozen:
                safe_step(opt_disc, list(disc.parameters()), loss, guard, 10.0)
            disc_log[T.TYPES[i]] = dict(loss=round(loss.item(), 4), d_data=round(dr.mean().item(), 3), d_policy=round(df.mean().item(), 3), grad_pen=round(gp.item(), 5), grad_share={k_: round(float(v_), 3) for k_, v_ in part.items()},
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
                         disc=disc_log, foot_nll=round(float(foot_loss.detach()), 4), entropy_std=[round(x, 3) for x in actor.std().tolist()], **({"arm_bound": round(bnd_l, 4), "arm_bound_c": round(bnd_c, 4)} if a.arm_bound_coef > 0 else {}),
                         done_share={k_[5:]: round(v_ / n_done, 4) for k_, v_ in cnt.items() if k_.startswith("done_") and k_ != "done_any"}, dones_per_iter=cnt.get("done_any", 0.0),
                         foothold=dict(touchdowns=cnt.get("td_n", 0.0), msr=round(cnt.get("td_support", 0.0) / max(cnt.get("td_n", 0.0), 1.0), 4),
                                       sfr=round(cnt.get("td_safe", 0.0) / max(cnt.get("td_n", 0.0), 1.0), 4)),
                         levels_by_type=dict(zip(T.TYPES, lv)), level_max_share=round((env.level[~env.free] >= T.NLEVEL - 1).float().mean().item(), 4),
                         samples_per_s=round(N_ * Tn / max(time.time() - t0, 1e-6)), num_envs=N_, air_td=round(air_td, 3), air_long_share=round(float(cnt.get("air_long", 0.0)) / max(float(cnt.get("air_n", 0.0)), 1.0), 3), prog_mean=round(float(cnt.get("prog_sum", 0.0)) / max(float(cnt.get("prog_n", 0.0)), 1.0), 3), prog_gate=round(env.prog_gate, 3), gate_start=gate_start, flat_phase=bool(env.flat_only), obst_share=round(float((env.type != T.FLAT).float().mean()), 3), obst_target=env.obst_target, w_style_now=round(float(w[2]), 4), gpu_util=gu, gpu_mem_mb=gm, ext=env.ext_stats(), edge_stop=dict(start=cnt.get("es_start", 0.0), done=cnt.get("es_done", 0.0)),
                         omni_share=round(float(env.free.float().mean()), 4), critic_warmup=bool(it < warm_until), to_paper=round(b_paper, 3),
                         style_by_group={k_: v_["style_reward"] for k_, v_ in disc_log.items()})
            rec = dict(it=it, dt=round(time.time() - t0, 2), elapsed_h=round((time.time() - t_start) / 3600, 3), r_l=round(rs[0], 3), r_f=round(rs[1], 3), r_s=round(rs[2], 3),
                       ep_ret=round(float(np.mean(done_stats["ret"])) if done_stats["ret"] else 0.0, 2), ep_len_s=round(float(np.mean(done_stats["len"]) if done_stats["len"] else 0.0) * 0.02, 2),
                       n_done=len(done_stats["len"]), level=lv, kl=round(kl_m, 4), lr=lr_ppo, pi=round(pi_l, 4), v=round(v_l, 3), std=round(actor.std().mean().item(), 3),
                       hb=round(enc_l["hb"], 4), hf=round(enc_l["hf"], 4), np=round(enc_l["np"], 4), v_est=round(enc_l["v"], 4), foot_cm=round(100 * f_err, 1),
                       d_r=round(d_real, 3), d_f=round(d_fake, 3), gpu_gb=round(torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0, 1), nan=env.log.get("nan", 0), touch=env.log.get("touch", 0), stuck=env.log.get("stuck", 0), gate_acc=round(gate_hit / max(gate_n, 1), 3), skipped=guard.get("skipped", 0), **extra, guide_on=env.guide_on, stage_ramp=round(ramp, 3), v_min=round(env.v_min, 3), sig_v=round(env.sig_v, 4), stuck_on=bool(env.stuck_term), adv_sd=[round(x, 5) for x in adv_sd], lvl_mean=round(env.log.get("level", 0.0), 2),
                       gait={k_: round(v_ / max(env.gait.get("n", 1), 1), 3) for k_, v_ in env.gait.items() if k_ != "n"}, terms={k_: round(v_, 4) for k_, v_ in term_sum.items()})
            print(json.dumps(rec), flush=True)
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()
            done_stats = {"ret": [], "len": []}
        if a.arm_bias_shift and not arm_bias_shift_done:           # E7b one-time shift of arm means past a stop (end of iteration; see FINDINGS E7b)
            with torch.no_grad():
                ob_ = B["obs"].flatten(0, 1)
                ob_ = ob_[::max(1, len(ob_) // 65536)]
                x_ = actor.norm(ob_); w_ = actor.gate(x_).softmax(-1)
                pre_ = (torch.stack([e(x_) for e in actor.experts], -1) * w_[..., None, :]).sum(-1)
                mu_ = 8.0 * torch.tanh(pre_ / 8.0)
                lo_ = (env.q_lo + a.arm_bound_margin - env.q0) / env.scale
                hi_ = (env.q_hi - a.arm_bound_margin - env.q0) / env.scale
                viol_ = [j_ for j_ in ARM_IDX if float(mu_[:, j_].mean()) > float(hi_[j_]) or float(mu_[:, j_].mean()) < float(lo_[j_])]
                PAIR_ = {13: 17, 14: 18, 15: 19, 16: 20}
                for r_, l_ in PAIR_.items():                       # mirror pairs move together (the encoder's tied bias needs equal deltas)
                    if (r_ in viol_) != (l_ in viol_):
                        viol_ = sorted(set(viol_) | {r_, l_})
                shifted_ = {}
                dA_ = torch.zeros(21, device=dev); dQ_ = torch.zeros(21, device=dev)
                for r_, l_ in PAIR_.items():
                    if r_ not in viol_:
                        continue
                    m_ = 0.5 * (float(mu_[:, r_].mean()) + float(mu_[:, l_].mean()))
                    if a.arm_shift_to == "edge":
                        side_hi = m_ > 0.5 * (float(hi_[r_]) + float(hi_[l_]))
                        tgt_ = (min(float(hi_[r_]), float(hi_[l_])) - 0.25) if side_hi else (max(float(lo_[r_]), float(lo_[l_])) + 0.25)
                    else:
                        tgt_ = 0.0
                    dA_[r_] = dA_[l_] = tgt_ - m_                          # the same action delta on both sides
                    dQ_[r_] = dQ_[l_] = 0.5 * sum(float(env.scale[j_]) * tgt_ - float(ob_[:, 9 + j_].mean()) for j_ in (r_, l_))   # new PD target - measured q (obs q is relative to default)
                    for j_ in (r_, l_):                               # pre-tanh shift giving that mean action (mean over the batch)
                        d_ = 0.0
                        for _ in range(20):                           # Newton on mean(8 tanh((pre + d)/8)) = mean(mu) + delta
                            f_ = (8.0 * torch.tanh((pre_[:, j_] + d_) / 8.0)).mean() - (mu_[:, j_].mean() + dA_[j_])
                            g_ = (1.0 - torch.tanh((pre_[:, j_] + d_) / 8.0) ** 2).mean().clamp(min=1e-3)
                            d_ = d_ - float(f_ / g_)
                        for e_ in actor.experts:
                            e_[-1].bias[j_] += d_
                            st_ = opt_ppo.state.get(e_[-1].bias, {})
                            if "exp_avg" in st_:
                                st_["exp_avg"][j_] = 0.0
                        shifted_[j_] = (round(float(mu_[:, j_].mean()), 2), round(float(dA_[j_]), 2), round(d_, 2))
                if shifted_:
                    st8_ = lambda X: X.flatten(0, 1)[::max(1, X.shape[0] * X.shape[1] // 8192)]
                    hB_, sB_, aB_ = st8_(B["hist"]), st8_(B["priv"]), st8_(B["a"])
                    e0_, f0_, v0_ = aux.est(hB_.flatten(1)), foot(sB_, aB_)[0], critics(sB_)
                    # every network that reads the previous action: same output for the same deviation from the new mean
                    oi_ = 51                                           # previous action in prop (and priv) = dims 51:72
                    d72_ = torch.zeros(PROP_DIM, device=dev); d72_[9:30] = dQ_; d72_[oi_:oi_ + 21] = dA_
                    for nm_ in (actor.norm, critics.norm, foot.norm):
                        nm_.mean[:PROP_DIM] += d72_
                    W_ = foot.net[0].weight                            # foothold model: current action is a raw input after priv
                    foot.net[0].bias -= W_[:, PRIV_DIM:PRIV_DIM + 21] @ dA_
                    dh_ = d72_.expand(1, H, PROP_DIM).clone()
                    L0_ = aux.est[0]                                   # velocity estimator: raw history
                    L0_.bias -= L0_.weight @ dh_.flatten(1)[0]
                    E0_ = enc.prop.net[0]                              # encoder: structured history, tied EqLinear
                    y_ = E0_(enc.structure(dh_)) - E0_(enc.structure(torch.zeros_like(dh_)))
                    yl_, yr_ = y_[0].chunk(2, -1)
                    E0_.bias -= 0.5 * (yl_ + yr_)
                    enc_res_ = float((yl_ - yr_).abs().max()) * 0.5
                    dP_ = torch.nn.functional.pad(d72_, (0, sB_.shape[1] - PROP_DIM))
                    chk_ = dict(est=round(float((aux.est((hB_ + dh_).flatten(1)) - e0_).abs().max()), 6),
                                foot=round(float((foot(sB_ + dP_, aB_ + dA_)[0] - f0_).abs().max()), 6),
                                critic=round(float((critics(sB_ + dP_) - v0_).abs().max()), 6))
                mu2_ = actor(ob_ + torch.nn.functional.pad(d72_, (0, ob_.shape[1] - PROP_DIM)) if shifted_ else ob_)
            arm_bias_shift_done = True
            if a.arm_shift_lr > 0:
                lr_ppo = a.arm_shift_lr
                for g_ in opt_ppo.param_groups:
                    g_["lr"] = lr_ppo
            print("arm bias shift at end of", it, "{joint: (mean action before, action delta, pre-tanh shift)}", shifted_, "q delta", [round(float(x), 3) for x in dQ_[ARM_IDX]],
                  "arm means after (inputs moved by the delta)", [round(float(x), 2) for x in mu2_[:, ARM_IDX].mean(0)],
                  "max change of the other joints", round(float((mu2_ - mu_)[:, [k for k in range(21) if k not in shifted_]].abs().max()), 6),
                  "encoder bias residual", round(enc_res_, 6) if shifted_ else 0.0, "consumer checks", chk_ if shifted_ else {}, flush=True)
        if not all_finite(enc, aux, actor, critics, foot, disc):
            raise SystemExit("non-finite network parameters at iteration %d: not saving; the supervisor resumes from the last good checkpoint" % it)
        if it % a.save_every == 0 or it == a.iters - 1:
            ck = dict(iter=it, enc=enc.state_dict(), aux=aux.state_dict(), actor=actor.state_dict(), critics=critics.state_dict(), foot=foot.state_dict(), disc=disc.state_dict(),
                      opt_ppo=opt_ppo.state_dict(), opt_enc=opt_enc.state_dict(), opt_foot=opt_foot.state_dict(), opt_disc=opt_disc.state_dict(), level=env.level.cpu(), type=env.type.cpu(),
                      guide_on=env.guide_on, gate_start=gate_start, args=vars(a), warm_until=warm_until, arm_std_reset_done=arm_std_reset_done, arm_std_reset_val=arm_std_reset_val, arm_bias_shift_done=arm_bias_shift_done)
            if getattr(env, "ext_on", False):                     # 2026-11-13: extended-map curriculum survives a resume (was reset to 0 at every restart)
                ck.update(ext_level=env.ext_level.cpu(), ext_type=env.ext_type.cpu())
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
    ap.add_argument("--ext_frac", type=float, default=0.0, help="extended map: share of tile robots on route tiles (user decision 2026-11-12); 0 = off")
    ap.add_argument("--ext_types", default="tight_spots,log_corridors,homes,street_front,stairs")
    ap.add_argument("--ext_variants", type=int, default=2)
    ap.add_argument("--ext_stairs_inset", action="store_true", help="v13: stair tiles from extended_v1_2_stairs_inset (foot inside the tile, 700 routes)")
    ap.add_argument("--ext_prop_routes", action="store_true", help="v13: add extended_v1_1_prop_routes (approach shapes on beams, beam fields, blocks, logs, wedges)")
    ap.add_argument("--ext_slope_turns", type=int, default=0, help="v14: slope turn stations per tile (out to a 8-22 deg plane, turn in place, back); 0 = off")
    ap.add_argument("--back_scale", type=float, default=1.0, help="v14 (user rule 2026-11-12): lin_vel reward x this while moving backward (< 1); also no backward commands on slope tiles; 1 = off")
    ap.add_argument("--creep_w", type=float, default=0.0, help="v14: weight of the progress share under 0.10-0.35 m/s commands (the stop-and-stand fix); 0 = off")
    ap.add_argument("--ext_hold_r", type=float, default=0.20, help="v14: route end hold radius (the eval disc is 0.20; 0.10 trains arriving inside it)")
    ap.add_argument("--drift_w", type=float, default=0.0, help="v14: penalty weight on planar speed under a stand or turn-in-place command; 0 = off")
    ap.add_argument("--ext_stations", type=int, default=0, help="v13: wall stations per tile (approach to 0.15-0.22 m of a wall face + departure); 0 = off")
    ap.add_argument("--ext_phys", type=int, default=0, help="extended map: K physical wall boxes per world, contact graded by body part (user rule 2026-11-12); 0 = vision-only walls")
    ap.add_argument("--ext_min_clear", type=float, default=0.0, help="extended map: drop routes whose least obstacle clearance along the route is under this (m); 0.39 = the X2 fits (SSR 2026-11-12)")
    ap.add_argument("--side_soft", type=float, default=0.0, help="deviation: side-obstacle 0.06 m band costs this per step instead of ending the episode (only overlap ends it); 0 = off")
    ap.add_argument("--prior_turn_min", type=float, default=0.0, help="deviation: keep prior windows with mean |yaw rate| >= this even under --prior_min_speed; 0 = off")
    ap.add_argument("--edge_stop", type=float, default=0.0, help="deviation: share of tile episodes that stop for 2 s at the first down edge (>= 0.15 m drop 0.45 m ahead)")
    ap.add_argument("--rel_kernel", action="store_true", help="deviation: velocity and yaw kernels narrowed for small commands (width x (max(|c|, c_min) / c_ref)^2, c_ref 0.6 m/s / 0.8 rad/s, c_min 0.3 / 0.4)")
    ap.add_argument("--swing_h_w", type=float, default=0.0, help="deviation: swing-height reward weight (sum over swing feet of min(h / target, 1) x min(|v_xy|, 1), moving commands); 0 = off")
    ap.add_argument("--swing_h_target", type=float, default=0.10)
    ap.add_argument("--omni_frac", type=float, default=0.1, help="share of omnidirectional robots (0.1 = as before)")
    ap.add_argument("--omni_turn", type=float, default=0.0, help="deviation: share of omni robots told to turn in place (|wz| 0.3-1.2)")
    ap.add_argument("--omni_slow", type=float, default=0.0, help="deviation: share of omni robots told to walk slowly forward (0.1-0.4 m/s)")
    ap.add_argument("--resample_s", type=float, default=0.0, help="deviation: new command every this many seconds within an episode; 0 = one per episode")
    ap.add_argument("--sig_v_to", type=float, default=0.0, help="deviation: velocity kernel width ramped from 0.25 to this; 0 = off")
    ap.add_argument("--sig_v_from", type=int, default=0)
    ap.add_argument("--sig_v_ramp", type=int, default=300)
    ap.add_argument("--action_scale_paper", action="store_true", help="paper Table 14 action scales (0.25 / 0.2); NOT to be run without the user's agreement (plant rule)")
    ap.add_argument("--default_from_ref", action="store_true", help="deviation (step 2, fresh runs only): default leg pose = reference mean")
    ap.add_argument("--excl_pelvis_hiproll", action="store_true", help="deviation (step 1): no pelvis / hip-roll link contacts")
    ap.add_argument("--dof_dev_ref", action="store_true", help="deviation (step 1): dof_dev from the reference mean leg pose")
    ap.add_argument("--disc_mask_arms", action="store_true", help="deviation: the style discriminators do not see arm joints or wrist positions")
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
    ap.add_argument("--init_from", default="", help="v15: start the policy (encoder, actor, critics, foothold model) from this checkpoint; discriminators and optimisers fresh; ignored when --resume is given")
    ap.add_argument("--to_paper_ramp", type=int, default=0, help="v16 (resume transition): iterations after the critic warm-up over which the advantage normalisation goes joint -> separate and sigma_v sig_v_start -> 0.25; 0 = paper values at once")
    ap.add_argument("--sig_v_start", type=float, default=0.0, help="v16: velocity kernel at the start of --to_paper_ramp (v14 ended at 0.1); 0 = 0.25 throughout")
    ap.add_argument("--critic_warmup", type=int, default=0, help="v15: with --init_from, iterations of value fit (and discriminator training) with the policy frozen")
    ap.add_argument("--disc_part_w", default="", help="v15 (user rule 2026-11-13): 'legs,arms,hands' weights on the normalised style input, e.g. 1.0,0.3,0.0")
    ap.add_argument("--omni_top", action="store_true", help="v15 (A.2): omnidirectional commands only after a robot cleared the top level")
    ap.add_argument("--cur_rudin", action="store_true", help="v20 stairs refusal: tile curriculum as in [30] (up: walked > 4 m; down: walked < 0.5 x mean |cmd| x 20 s)")
    ap.add_argument("--cur_fair", type=float, default=0.0, help="v20: with --cur_rudin, tile episodes with mean |cmd| below this move no level")
    ap.add_argument("--swing_prog", action="store_true", help="v20: swing-height pay x progress share")
    ap.add_argument("--climb_types", default="", help="v20: tile types for --climb_side_noterm / --climb_creep, e.g. stairs,platform")
    ap.add_argument("--climb_side_noterm", action="store_true", help="v20: side-obstacle touch not terminal on climb tiles (graze cost stays)")
    ap.add_argument("--climb_creep", action="store_true", help="v20: creep progress pay at every moving command on climb tiles")
    ap.add_argument("--promote_half", action="store_true", help="v15 (A.1, [30]): level up when the robot walked more than half the tile")
    ap.add_argument("--no_vy_clamp", action="store_true", help="v15 (A.2): projected command not clamped for tile robots")
    ap.add_argument("--slope_back", action="store_true", help="v15 (Table 4, user rule 2026-11-13): backward commands on slopes even with --back_scale < 1")
    ap.add_argument("--init_q_noise", type=float, default=0.05, help="initial joint noise U(-x, x) rad (not in the paper)")
    ap.add_argument("--beam_hmax", type=float, default=30.0, help="v21 (Table 4): beam-tile heading range in deg; beam = the gap slot (paper 15); 30 = v14-v20")
    ap.add_argument("--foot_base_frame", action="store_true", help="v15 (Eq. 1): foothold targets in the full base frame")
    ap.add_argument("--ext_home", action="store_true", help="v15: route robots return to their own tile type; flat routes use its discriminator")
    ap.add_argument("--paper_terrain", action="store_true", help="v15 (A.1): the paper's five types (gap slot -> beam, discrete in the old flat slot) plus an open FREE column; read at import (see the top of this file)")
    ap.add_argument("--simcheck", action="store_true", help="v15 (TRACK_RULES simulator checks): simcheck faults are truncations with reward 0 and no level move; default v14: NaN / |qvel| > 60 terminate")
    ap.add_argument("--diag_out", default="", help="diagnostic replay: policy frozen (critics and discriminators train), per-iteration advantage / speed statistics appended to this file")
    ap.add_argument("--diag_disc_from", default="", help="diagnostic: load the discriminators of this checkpoint after any resume")
    ap.add_argument("--diag_disc_frozen", action="store_true", help="diagnostic: do not train the discriminators")
    ap.add_argument("--torch_seed", type=int, default=-1, help="seed torch / numpy / random at start (-1 = unseeded, the old behaviour)")
    ap.add_argument("--warm_until_abs", type=int, default=0, help="with --resume: policy frozen until this absolute iteration (critics and discriminators train)")
    ap.add_argument("--arm_std_reset", type=float, default=0.0, help="E7: set the 8 arm action stds to this value once per run (at the start or the first resume; 0 = off)")
    ap.add_argument("--arm_std_cap", type=float, default=0.0, help="E7: cap the 8 arm action stds at this value after every policy update (0 = off)")
    ap.add_argument("--arm_bound_coef", type=float, default=0.0, help="E7b: squared-excess loss on the 8 arm action means outside the joint range (0 = off)")
    ap.add_argument("--arm_bias_shift", action="store_true", help="E7b: once per run, shift arm action means that sit past a joint stop to the default pose (bias of every expert)")
    ap.add_argument("--arm_shift_to", default="default", choices=["default", "edge"], help="E7b: where the bias shift puts the arm means (action 0, or just inside the bound)")
    ap.add_argument("--arm_shift_lr", type=float, default=0.0, help="E7b: PPO lr set to this at the bias shift (0 = keep)")
    ap.add_argument("--arm_bound_ramp", type=int, default=0, help="E7b: ramp the bound coefficient linearly from 0 over this many iterations (0 = full at once)")
    ap.add_argument("--arm_bound_from", type=int, default=0, help="E7b: absolute iteration where the ramp starts (0 = first iteration of this process)")
    ap.add_argument("--arm_bound_margin", type=float, default=0.0, help="E7b: keep the arm PD targets of the mean this many rad inside the joint range")
    ap.add_argument("--ankle_roll_std_cap", type=float, default=0.0, help="E7: cap the two ankle-roll action stds at this value (0 = off)")
    train(ap.parse_args())
