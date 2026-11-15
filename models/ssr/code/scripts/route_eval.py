"""Walk the apparatus mobility route with an SSR policy: N independent robots, velocity commands from the waypoints.

The policy only gets what the paper gives it: depth + proprioception + a velocity command (no reference, no map, no planner).
Physics: the course's walk height raster + wall boxes, as in the ACP evaluation. Vision: the raster with the tall geometry.
The apparatus is never in training.

python scripts/route_eval.py --ckpt runs/v1/last.pt --num 50 --out runs/v1/route
Saves one rollout (best robot) for scripts/render_route.py and prints the scorecard.
"""
import argparse
import json
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from ssr import networks as N
from ssr import x2 as X
from ssr.env import DEPTH_N, HB_NX, HB_NY, SSREnv
from ssr.route import MOBILITY, IN, SPAWN

def load_raster(path, pad=2.0):
    z = np.load(path)
    g = np.nan_to_num(np.maximum(z["grid"], 0.0)).astype(np.float32)
    n = int(round(pad / float(z["res"])))
    return dict(z=np.pad(g, n), x0=float(z["x0"]) - n * float(z["res"]), y0=float(z["y0"]) - n * float(z["res"]), res=float(z["res"]), base=0.1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--num", type=int, default=50)
    ap.add_argument("--out", default="")
    ap.add_argument("--seconds", type=float, default=150.0)
    ap.add_argument("--slow", type=float, default=0.45)
    ap.add_argument("--walk", type=float, default=0.75)
    ap.add_argument("--wp_r", type=float, default=0.22)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    dev = torch.device(a.device)
    walls = json.load(open(ROOT / "assets/route/cad_walls.json"))["boxes"]
    run_args = json.load(open(pathlib.Path(a.ckpt).parent / "args.json"))
    camera, map_memory = run_args.get("camera", "head"), run_args.get("map_memory", False)
    env = SSREnv(a.num, a.device, raster=load_raster(ROOT / "assets/route/cad_20mm_walk.npz"), vision=load_raster(ROOT / "assets/route/cad_20mm.npz"), walls=walls, train=False, camera=camera, map_memory=map_memory,
                 spawn=torch.tensor(SPAWN, device=dev, dtype=torch.float32).expand(a.num, -1))
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    enc, aux, actor, foot = N.Encoder(map_memory).to(dev), N.Aux().to(dev), N.MoEActor().to(dev), N.Foothold().to(dev)
    enc.load_state_dict(ck["enc"]); aux.load_state_dict(ck["aux"]); actor.load_state_dict(ck["actor"]); foot.load_state_dict(ck["foot"])
    [m.eval() for m in (enc, aux, actor, foot)]
    wps = torch.tensor([(x * IN, y * IN, math.radians(h), s, hold) for _, x, y, h, s, hold in MOBILITY], device=dev, dtype=torch.float32)
    W = len(wps)
    wi, hold, alive, finish = torch.zeros(a.num, dtype=torch.long, device=dev), torch.zeros(a.num, device=dev), torch.ones(a.num, dtype=torch.bool, device=dev), torch.full((a.num,), -1.0, device=dev)
    h = enc.init_h(a.num, dev)
    Tmax = int(a.seconds / X.CONTROL_DT)
    rec = {k: [] for k in ("qpos", "depth", "hb_pred", "hb_true", "fm", "fs", "contact", "sole", "wi", "cmd", "rho")}
    fail_wp = torch.full((a.num,), -1, dtype=torch.long, device=dev)
    deaths = []
    for t in range(Tmax):
        with torch.no_grad():
            s = env.s
            p, yaw = s["p"], s["yaw"]
            w = wps[wi.clamp(max=W - 1)]
            d = w[:, :2] - p[:, :2]
            dist = d.norm(dim=-1)
            arrive = (dist < a.wp_r) & alive
            hold = torch.where(arrive, hold + X.CONTROL_DT, torch.zeros_like(hold))
            # arrival: drive to the waypoint, then face its heading for its hold time
            turn = wrap(w[:, 2] - yaw)
            done_here = arrive & (hold >= w[:, 4]) & (turn.abs() < 0.25 + 10 * (w[:, 4] <= 0))
            wi = wi + done_here.long()
            hold[done_here] = 0.0
            finished = wi >= W
            finish = torch.where(finished & (finish < 0), torch.full_like(finish, t * X.CONTROL_DT), finish)
            w = wps[wi.clamp(max=W - 1)]
            d = w[:, :2] - p[:, :2]
            dist = d.norm(dim=-1)
            head = torch.atan2(d[:, 1], d[:, 0])
            near = dist < 0.5
            target_yaw = torch.where(near & (w[:, 4] > 0), w[:, 2], head)
            err = wrap(target_yaw - yaw)
            v = torch.where(w[:, 3] > 0.5, torch.full_like(dist, a.walk), torch.full_like(dist, a.slow))
            speed = (v * (1 - (err.abs() / 1.2).clamp(max=1.0)) * (dist / 0.3).clamp(max=1.0))
            c, sn = torch.cos(yaw), torch.sin(yaw)
            vx = (c * d[:, 0] + sn * d[:, 1]) / dist.clamp(min=1e-3) * speed
            vy = (-sn * d[:, 0] + c * d[:, 1]) / dist.clamp(min=1e-3) * speed
            cmd = torch.stack((vx, vy.clamp(-0.3, 0.3), (1.5 * err).clamp(-1.0, 1.0)), -1)
            cmd = torch.where((finished | ~alive)[:, None], torch.zeros_like(cmd), cmd)
            env.cmd = cmd
            env.prop[:, 6:9] = cmd
            env.hist[:, -1, 6:9] = cmd
            e = enc.step(env.hist, env.depth, h, sample=False, map_img=env.map_img)
            h = e["h"]
            o = torch.cat((env.prop, aux.est(env.hist.flatten(1)), e["z"]), -1)
            act = actor(o)
            fm, fs = foot(env.priv, act)
            hb_pred = aux.dec_b(e["z_b"])
            best = int((wi + alive.long() * 0.01).argmax())
            r = env.step(act, fm, fs)
            dead = r["term"] & alive
            if bool(dead.any()):
                fail_wp[dead] = wi[dead]
                deaths += [(int(i), int(wi[i]), round(t * X.CONTROL_DT, 1)) for i in dead.nonzero().squeeze(-1)]
            alive &= ~r["term"]
            rec["qpos"].append(env.qpos.clone().cpu().float())
            for k, v_ in (("depth", env.depth), ("hb_pred", hb_pred), ("hb_true", env.h_b), ("fm", fm), ("fs", fs), ("contact", r["contact"]), ("sole", env.sole), ("wi", wi), ("cmd", cmd), ("rho", env.rho)):
                rec[k].append(v_.cpu().float() if v_.dtype != torch.bool else v_.cpu())
        if int((finished | ~alive).sum()) == a.num:
            break
    nf = int((finish >= 0).sum())
    print(json.dumps(dict(robots=a.num, finished=nf, seconds=[round(float(finish[finish >= 0].min()), 1), round(float(finish[finish >= 0].median()), 1)] if nf else None,
                          mean_waypoints=round(float(wi.float().mean()), 1), of=W, fell=int((~alive).sum()), by_waypoint=torch.bincount(fail_wp[fail_wp >= 0], minlength=W).tolist()), indent=None), flush=True)
    if a.out:
        out = pathlib.Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        idx = int((wi.float() + (finish >= 0) * 100 + alive.float() * 0.1).argmax())
        Tn = len(rec["qpos"])
        np.savez_compressed(out / "best.npz", **{k: torch.stack(v)[:, idx].numpy() for k, v in rec.items()}, reached=int(wi[idx]), finished=bool(finish[idx] >= 0), T=Tn)
        np.save(out / "all_qpos.npy", torch.stack(rec["qpos"])[::2].numpy())
        json.dump(dict(deaths=deaths, finish=finish.tolist(), wi=wi.tolist()), open(out / "summary.json", "w"))


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


if __name__ == "__main__":
    main()
