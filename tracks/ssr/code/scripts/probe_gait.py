"""Gait numbers for a checkpoint on the open flat tile, deterministic actions, no randomisation (a quick check, not an eval).

python scripts/probe_gait.py --ckpt ../runs/d1/last.pt --speed 0.6 --out ../eval/d1_probe
Prints: survival, speed, base height, yaw rate, cadence, step length, swing clearance, air time, support shares; saves the
simulator state (qpos per step) of the first robots for rendering.
"""
import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from ssr import networks as N
from ssr import terrain as T
from ssr import x2 as X
from ssr.env import SSREnv


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num", type=int, default=256)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--speed", type=float, default=0.6)
    ap.add_argument("--tag", default="gait")
    ap.add_argument("--evaltag", default="")
    ap.add_argument("--noise", action="store_true", help="sample actions with the policy std, as in training")
    a = ap.parse_args()
    import os
    DEV = os.environ.get("PROBE_DEVICE", "cuda:0")
    dev = torch.device(DEV)
    n = a.num
    env = SSREnv(n, DEV, train=False, tiles=True, camera=os.environ.get("PROBE_CAMERA", "chest_rs"))   # was the default head camera until 2026-11-12 05:05
    env.flat_only = True
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    enc, aux, actor = N.Encoder(False).to(dev), N.Aux().to(dev), N.MoEActor().to(dev)
    enc.load_state_dict(ck["enc"]); aux.load_state_dict(ck["aux"]); actor.load_state_dict(ck["actor"])
    [m.eval() for m in (enc, aux, actor)]
    ids = torch.arange(n, device=dev)
    env.type[:], env.level[:] = T.FLAT, 0
    env.reset(ids)
    env.speed[:], env.heading[:] = a.speed, 0.0
    env.qpos[:, 3], env.qpos[:, 4:7] = 1.0, 0.0
    env.refresh(); env._update_cmd(); env._contacts(); env._update_obs(ids); env._update_cmd()
    h = enc.init_h(n, dev)
    alive = torch.ones(n, dtype=torch.bool, device=dev)
    keep = min(8, n)
    traj = np.zeros((a.steps, keep, env.qpos.shape[1]), np.float32)
    acc = dict(vx=0.0, h=0.0, wz=0.0, wz_abs=0.0, wz_cmd=0.0, wxy=0.0, single=0.0, double=0.0, flight=0.0, n=0.0, tilt=0.0)
    last_td_x = torch.full((n, 2), float("nan"), device=dev)
    air_max = torch.zeros(n, 2, device=dev)
    air_t = torch.zeros(n, 2, device=dev)
    prev_c = torch.ones(n, 2, dtype=torch.bool, device=dev)
    steps_len, clear, airs, tds = [], [], [], 0.0
    terms = {}
    lim, qsum = torch.zeros(21, device=dev), torch.zeros(21, device=dev)
    std = actor.std()
    for t in range(a.steps):
        e = enc.step(env.hist, env.depth, h, sample=False)
        h = e["h"]
        o = torch.cat((env.prop, aux.est(env.hist.flatten(1)), e["z"]), -1)
        act = actor(o)
        if a.noise:
            act = act + std * torch.randn_like(act)
        r = env.step(act.clamp(-8, 8))
        alive &= ~r["term"]
        s = env.s
        w = alive.float()
        for k_, v_ in env.terms.items():                     # the paper's Table 6 terms, weighted, whatever the policy was trained on
            terms[k_] = terms.get(k_, 0.0) + float((v_ * w).sum())
        terms["r_l_total"] = terms.get("r_l_total", 0.0) + float((r["r_l"] * w).sum())
        c = r["contact"]
        ground = env.hmap(s["p"][:, 0], s["p"][:, 1])
        for k, v in (("vx", env.qvel[:, 0]), ("h", s["p"][:, 2] - ground), ("wz", s["w_b"][:, 2]), ("wz_abs", s["w_b"][:, 2].abs()), ("wz_cmd", env.cmd[:, 2].abs()),
                     ("wxy", s["w_b"][:, :2].norm(dim=-1)), ("single", (c.sum(-1) == 1).float()), ("double", (c.sum(-1) == 2).float()), ("flight", (c.sum(-1) == 0).float()),
                     ("tilt", s["g"][:, :2].norm(dim=-1))):
            acc[k] += float((v * w).sum())
        acc["n"] += float(w.sum())
        lim = lim + (((s["q"] < env.q_lo) | (s["q"] > env.q_hi)).float() * w[:, None]).sum(0)
        qsum = qsum + (s["q"] * w[:, None]).sum(0)
        sole_h = env.sole[..., 2] - env.hmap(env.sole[..., 0], env.sole[..., 1])
        air_max = torch.where(c, air_max, torch.maximum(air_max, sole_h))
        air_t = torch.where(c, air_t, air_t + X.CONTROL_DT)
        td = c & ~prev_c & alive[:, None]
        if bool(td.any()):
            tds += float(td.sum())
            x = env.sole[..., 0]
            d = (x - last_td_x)[td]
            steps_len.append(d[torch.isfinite(d)])
            clear.append(air_max[td]); airs.append(air_t[td])
            last_td_x = torch.where(td, x, last_td_x)
        air_max = torch.where(c, torch.zeros_like(air_max), air_max)
        air_t = torch.where(c, torch.zeros_like(air_t), air_t)
        prev_c = c.clone()
        traj[t] = env.qpos[:keep].cpu().numpy()
    cat = lambda L: torch.cat(L) if L else torch.zeros(1, device=dev)
    sl, cl, at = cat(steps_len), cat(clear), cat(airs)
    real = at >= 0.1                                     # air phases of 0.1 s or more: a step, not contact chatter
    m = max(acc["n"], 1.0)
    out = dict(ckpt=a.ckpt, iteration=int(ck["iter"]), robots=n, seconds=a.steps * X.CONTROL_DT, commanded_speed=a.speed, noise=a.noise, policy_std=round(float(std.mean()), 3),
               survived_share=round(float(alive.float().mean()), 3), speed_mps=round(acc["vx"] / m, 3), base_height_m=round(acc["h"] / m, 3),
               yaw_rate_mean=round(acc["wz"] / m, 3), yaw_rate_abs=round(acc["wz_abs"] / m, 3), yaw_rate_cmd_abs=round(acc["wz_cmd"] / m, 3), rollpitch_rate=round(acc["wxy"] / m, 3),
               tilt_gxy=round(acc["tilt"] / m, 3), single=round(acc["single"] / m, 3), double=round(acc["double"] / m, 3), flight=round(acc["flight"] / m, 3),
               touchdowns_per_foot_per_s=round(tds / (m * X.CONTROL_DT) / 2, 2),
               steps_per_foot_per_s=round(float(real.sum()) / (m * X.CONTROL_DT) / 2, 2),
               joints_past_limit={X.NAMES21[i]: dict(share=round(float(lim[i] / m), 3), mean_q=round(float(qsum[i] / m), 3), lo=round(float(env.q_lo[i]), 3), hi=round(float(env.q_hi[i]), 3)) for i in range(21) if float(lim[i] / m) > 0.02},
               reward_terms_weighted={k_: round(v_ / m, 4) for k_, v_ in terms.items()},
               air_time_s=dict(median=round(float(at.median()), 3), share_under_0p1=round(float((at < 0.1).float().mean()), 3)),
               step_length_m=dict(median_all=round(float(sl.median()), 3)),
               swing_clearance_m=dict(median_all=round(float(cl.median()), 3), median_real_steps=round(float(cl[real].median()) if bool(real.any()) else 0.0, 3),
                                      p90_real_steps=round(float(cl[real].quantile(0.9)) if bool(real.any()) else 0.0, 3)))
    p = pathlib.Path(a.out)
    p.mkdir(parents=True, exist_ok=True)
    (p / (a.tag + ("_noise.json" if a.noise else ".json"))).write_text(json.dumps(out, indent=1))
    if not a.noise:
        np.savez_compressed(p / "state.npz", qpos=traj, control_dt=X.CONTROL_DT, speed=a.speed, iteration=int(ck["iter"]))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
