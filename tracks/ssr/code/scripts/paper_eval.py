"""The paper's metrics for a checkpoint, per terrain and difficulty, plus the reward terms and training curves from the log.

Survive: 20 s without termination at a 1.0 m/s command, 1,000 randomized trials per setting (the paper's success rate, Sec. 4.1). A robot standing still scores 100 % on that,
so crossed adds the stricter test: it reached the far end of the tile, alive.
SFR / MSR: safe foothold rate (support ratio over 75 %) and mean support ratio, at touchdowns (Sec. 4.2).
Power and peak foot contact force (Table 1). Tracking under eight velocity directions and two yaw rates, and which foot leads over obstacles (Fig. 7).
Gate: mean expert weights per terrain.

python scripts/paper_eval.py --ckpt runs/chest/last.pt --out reports/chest
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

LEVELS = (4, 9, 14, 19)
DESC = {
    "ramp": lambda s: f"slope {20 * s:.0f} deg", "stairs": lambda s: f"rise {100 * (0.05 + 0.15 * s):.0f} cm, run {100 * (0.40 - 0.15 * s):.0f} cm",
    "beam": lambda s: f"beam {100 * (0.03 + 0.09 * s):.0f} cm tall", "platform": lambda s: f"platform {100 * (0.05 + 0.35 * s):.0f} cm",
    "flat": lambda s: "lane with doors and rails",
}
# the paper's real-robot and simulation figures, on its own terrains, for reference only
PAPER = {"stairs": "sim near 100 %; real 100 % (20/20) at 15 / 30 cm", "platform": "real 100 % (20/20) at 40 cm", "beam": "no equivalent terrain", "ramp": "slopes shown, no rate reported", "flat": "no equivalent terrain"}


@torch.no_grad()
def rollout(env, enc, aux, actor, steps, cmd=None):
    n = env.N
    dev = env.dev
    h = enc.init_h(n, dev)
    alive = torch.ones(n, dtype=torch.bool, device=dev)
    done_ok = torch.zeros(n, dtype=torch.bool, device=dev)
    stat = dict(power=torch.zeros(n, device=dev), t=torch.zeros(n, device=dev), peak=torch.zeros(n, device=dev), sup=[], gate=torch.zeros(5, device=dev), verr=torch.zeros(n, device=dev), werr=torch.zeros(n, device=dev))
    lead = torch.full((n,), -1, dtype=torch.long, device=dev)
    crossed = torch.zeros(n, dtype=torch.bool, device=dev)
    for _ in range(steps):
        if cmd is not None:
            env.cmd = cmd
            env.prop[:, 6:9] = cmd
            env.hist[:, -1, 6:9] = cmd
        e = enc.step(env.hist, env.depth, h, sample=False, map_img=env.map_img)
        h = e["h"]
        o = torch.cat((env.prop, aux.est(env.hist.flatten(1)), e["z"]), -1)
        stat["gate"] += actor.gate(actor.norm(o)).softmax(-1)[alive].sum(0) if bool(alive.any()) else 0.0
        c_used = env.cmd.clone()
        r = env.step(actor(o))
        s = env.s
        stat["power"] += (s["tau"] * s["qd"]).abs().sum(-1) * alive
        stat["t"] += alive.float()
        stat["peak"] = torch.max(stat["peak"], env.F.norm(dim=-1).amax(-1) * alive)
        stat["verr"] += (s["v_b"][:, :2] - c_used[:, :2]).norm(dim=-1) * alive
        stat["werr"] += (s["w_b"][:, 2] - c_used[:, 2]).abs() * alive
        td = r["touchdown"] & alive[:, None]
        if bool(td.any()):
            stat["sup"].append((1.0 - env.support_stance())[td])
            xr = env.sole[..., 0] - env.origin[:, 0, None]
            past = td & (xr > T.FLAT_RUN + 0.1)
            first = (lead < 0) & past.any(-1)
            lead = torch.where(first, past[:, 1].long(), lead)              # 1 = left foot first (feet are [R, L])
        done_ok |= r["timeout"] & alive & ~r["term"]
        crossed |= alive & (env.qpos[:, 0] - env.origin[:, 0] > T.TILE_X)
        alive &= ~r["term"]
    stat["crossed"] = crossed
    return alive | done_ok, stat, lead


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--trials", type=int, default=1000)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    dev = torch.device(a.device)
    run_dir = pathlib.Path(a.ckpt).parent
    args = json.load(open(run_dir / "args.json"))
    env = SSREnv(a.trials, a.device, train=False, tiles=True, camera=args.get("camera", "head"), map_memory=args.get("map_memory", False))
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    enc, aux, actor = N.Encoder(args.get("map_memory", False)).to(dev), N.Aux().to(dev), N.MoEActor().to(dev)
    enc.load_state_dict(ck["enc"]); aux.load_state_dict(ck["aux"]); actor.load_state_dict(ck["actor"])
    [m.eval() for m in (enc, aux, actor)]
    ids = torch.arange(a.trials, device=dev)
    rows, gate = [], {}
    for c, name in enumerate(T.TYPES):
        g = torch.zeros(5, device=dev)
        for lv in LEVELS:
            env.type[:], env.level[:] = c, lv
            env.ep[:] = 0
            env.reset(ids)
            env.speed[:], env.heading[:] = 1.0, 0.0
            env.qpos[:, 3], env.qpos[:, 6] = 1.0, 0.0
            env.refresh()
            env._update_cmd()
            env._contacts()
            env._update_obs(ids)
            env._update_cmd()
            ok, st, lead = rollout(env, enc, aux, actor, 1000)
            sup = torch.cat(st["sup"]) if st["sup"] else torch.zeros(1, device=dev)
            t = st["t"].clamp(min=1.0)
            L, R = int((lead == 1).sum()), int((lead == 0).sum())
            rows.append(dict(terrain=name, level=lv, setting=DESC[name](lv / (T.NLEVEL - 1)), survive=round(100 * ok.float().mean().item(), 1), crossed=round(100 * st["crossed"].float().mean().item(), 1), sfr=round(100 * (sup > 0.75).float().mean().item(), 1),
                             msr=round(100 * sup.mean().item(), 1), power_w=round((st["power"] / t).mean().item(), 1), peak_force_n=round(st["peak"].mean().item(), 1), left_lead_pct=round(100 * L / max(L + R, 1), 1), v_err=round((st["verr"] / t).mean().item(), 3)))
            g += st["gate"]
            print(json.dumps(rows[-1]), flush=True)
        gate[name] = [round(x, 3) for x in (g / g.sum()).tolist()]
    # tracking on open flat ground: eight directions and two yaw rates (Fig. 7)
    track = {}
    for label, c in (("+x", (0.8, 0, 0)), ("-x", (-0.8, 0, 0)), ("+y", (0, 0.3, 0)), ("-y", (0, -0.3, 0)), ("+x+y", (0.56, 0.28, 0)), ("+x-y", (0.56, -0.28, 0)), ("-x+y", (-0.56, 0.28, 0)), ("-x-y", (-0.56, -0.28, 0)), ("yaw+", (0, 0, 0.8)), ("yaw-", (0, 0, -0.8))):
        env.type[:], env.level[:] = T.FLAT, 0
        env.ep[:] = 0
        env.free[:] = True
        env.reset(ids)
        env.free[:] = True
        env.origin[:, 0], env.origin[:, 1] = T.FLAT * T.TILE_X, 0.0
        env.qpos[:, 0], env.qpos[:, 1] = T.FLAT * T.TILE_X + 4.0, 2.0 + env._u(a.trials) * 1.0
        env.qpos[:, 3], env.qpos[:, 6] = 1.0, 0.0
        env.refresh(); env._contacts(); env._update_obs(ids)
        cmd = torch.tensor(c, device=dev, dtype=torch.float32).expand(a.trials, -1)
        ok, st, _ = rollout(env, enc, aux, actor, 250, cmd=cmd)
        t = st["t"].clamp(min=1.0)
        track[label] = dict(survive=round(100 * ok.float().mean().item(), 1), v_err=round((st["verr"] / t).mean().item(), 3), yaw_err=round((st["werr"] / t).mean().item(), 3))
        print(label, track[label], flush=True)
    log = [json.loads(l) for l in open(run_dir / "log.jsonl")]
    last = log[-1]
    res = dict(ckpt=str(a.ckpt), iter=ck["iter"], camera=args.get("camera"), map_memory=args.get("map_memory"), rows=rows, gate=gate, tracking=track, paper=PAPER, final_log={k: last[k] for k in ("it", "r_l", "r_f", "r_s", "ep_len_s", "level", "foot_cm", "gate_acc", "terms") if k in last},
               curve=[dict(it=x["it"], ep_len=x["ep_len_s"], r_l=x["r_l"], r_f=x["r_f"], r_s=x["r_s"], level=round(float(np.mean(x["level"])), 2)) for x in log])
    if a.out:
        out = ROOT / a.out
        out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(str(out) + ".json", "w"), indent=1)
        print("wrote", str(out) + ".json")


if __name__ == "__main__":
    main()
