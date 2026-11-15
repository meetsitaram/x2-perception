"""Eval-like stop probe (CPU, deterministic, open flat floor): the harness waypoint law (vx = clip(dist, 0.15, 0.6) cos err, wz 1.5 err,
zero vx past 60 deg) to a waypoint 2 m ahead with a 0.20 m disc; inside the disc a zero command. Per checkpoint: share that hold a
strict stand (speed < 0.15 m/s inside the disc for 1 s) within 10 s, median path length in 1 s after first entering the disc,
median speed when entering. Usage: ckpt..."""
import json, sys, math, torch, numpy as np
sys.path.insert(0, "/mnt/filesystem-g4/tracks/ssr/code")
from ssr import networks as N
from ssr.env import SSREnv
torch.set_grad_enabled(False)
Z = np.zeros((int(12 / 0.025) + 1, int(4 / 0.025) + 1), np.float32)
R = dict(z=Z, x0=0.0, y0=0.0, res=0.025, base=0.1, vision=dict(z=Z.copy(), x0=0.0, y0=0.0, res=0.025))
DT = 0.02
import os
FLOOR = float(os.environ.get("FLOOR", "0"))
for path in sys.argv[1:]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    enc, aux, actor = N.Encoder(False), N.Aux(), N.MoEActor()
    enc.load_state_dict(ck["enc"]); aux.load_state_dict(ck["aux"]); actor.load_state_dict(ck["actor"])
    n = 16
    sp = torch.tensor([[1.0, 2.0 + 0.03 * (i - 7.5), 0.0] for i in range(n)])
    env = SSREnv(n, "cpu", raster=R, train=False, tiles=False, spawn=sp, camera="chest_rs")
    env.cmd[:] = 0.0; h = enc.init_h(n, "cpu")
    tgt = torch.stack((torch.full((n,), 3.0), sp[:, 1]), -1)
    t_in = torch.full((n,), -1); held = torch.zeros(n, dtype=torch.bool); ok_t = torch.zeros(n); path_len = torch.zeros(n)
    v_in = torch.zeros(n); prev = None
    for t in range(int(12 / DT)):
        p, yaw = env.s["p"][:, :2], env.s["yaw"]
        vec = tgt - p; dist = vec.norm(dim=-1)
        err = torch.atan2(torch.sin(torch.atan2(vec[:, 1], vec[:, 0]) - yaw), torch.cos(torch.atan2(vec[:, 1], vec[:, 0]) - yaw))
        inside = dist < 0.20
        vx = torch.where(err.abs() > math.radians(60), torch.zeros(n), dist.clamp(0.15, 0.6).clamp(min=FLOOR) * torch.cos(err))
        wz = (1.5 * err).clamp(-1, 1)
        if t >= 50:
            env.cmd[:] = torch.where((inside | (t_in >= 0))[:, None], torch.zeros(n, 3), torch.stack((vx, torch.zeros(n), wz), -1))   # first entry latches the stand
        spd = env.s["v_w"][:, :2].norm(dim=-1) if "v_w" in env.s else env.s["v_b"][:, :2].norm(dim=-1)
        new = inside & (t_in < 0) & (t >= 50)
        v_in = torch.where(new, spd, v_in); t_in = torch.where(new, torch.full_like(t_in, t), t_in)
        if prev is not None:
            step_d = (p - prev).norm(dim=-1)
            path_len += torch.where((t_in >= 0) & (t - t_in <= 50), step_d, torch.zeros(n))
        prev = p.clone()
        good = inside & (spd < 0.15)
        ok_t = torch.where(good, ok_t + DT, torch.zeros(n)); held |= ok_t >= 1.0
        e = enc.step(env.hist, env.depth, h, sample=False); h = e["h"]
        env.step(actor(torch.cat((env.prop, aux.est(env.hist.flatten(1)), e["z"]), -1)).clamp(-8, 8))
    print(json.dumps(dict(ckpt=path.split("/")[-1], it=int(ck["iter"]), held=round(float(held.float().mean()), 2), reached=round(float((t_in >= 0).float().mean()), 2),
          path_1s_med=round(float(path_len.median()), 3), v_enter_med=round(float(v_in.median()), 3), fin_dist_med=round(float((tgt - env.s["p"][:, :2]).norm(dim=-1).median()), 3))), flush=True)
print("DONE")
