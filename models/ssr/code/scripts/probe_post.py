"""Does a post / door frame / wall seen in depth make the policy stand? (CPU, deterministic). Flat floor 8 x 4 m; the
obstacles are only in the vision raster (seen in depth, no collision), so a stall is the policy's choice, not a contact.
Robots start at (1, 2) facing +x after 1 s standing; command (v, 0, 0) for 8 s; distance covered and mean speed of the last 4 s."""
import json, sys, torch
import numpy as np
sys.path.insert(0, "/mnt/filesystem-g4/tracks/ssr/code")
from ssr import networks as N
from ssr.env import SSREnv
torch.set_grad_enabled(False)
RES = 0.025
def raster(kind):
    nx, ny = int(8 / RES) + 1, int(4 / RES) + 1
    z = np.zeros((nx, ny), np.float32); v = z.copy()
    I = lambda a: int(round(a / RES))
    def box(x0, x1, y0, y1): v[I(x0):I(x1), I(y0):I(y1)] = 1.3
    if kind == "post_left_0.4":      # post 0.12 m square, 0.4 m ahead, 0.35 m left of the path (like the door-side stall)
        box(1.34, 1.46, 2.29, 2.41)
    elif kind == "door_0.9":         # door jambs 0.12 m deep with 0.5 m wall stubs, opening 0.9 m, 0.6 m ahead
        box(1.6, 1.72, 2.45, 2.95); box(1.6, 1.72, 1.05, 1.55)
    elif kind == "wall_left_0.35":   # wall along the left of the path, 0.35 m from the centre line
        box(0.5, 4.0, 2.35, 2.43)
    elif kind == "posts_both_0.45":  # posts both sides 0.45 m from the line, every 1.2 m
        for x in (1.6, 2.8, 4.0): box(x, x + 0.08, 2.45, 2.53); box(x, x + 0.08, 1.47, 1.55)
    elif kind == "post_close":       # post 0.25 m ahead, 0.20-0.32 m left
        box(1.25, 1.37, 2.20, 2.32)
    elif kind == "stub_close":       # wall stub (door jamb + 0.5 m wall) 0.3 m ahead, from 0.25 m left outward
        box(1.30, 1.42, 2.25, 2.75)
    elif kind == "wall_left_0.25":
        box(0.5, 4.0, 2.25, 2.33)
    return dict(z=z, x0=0.0, y0=0.0, res=RES, base=0.1, vision=dict(z=v, x0=0.0, y0=0.0, res=RES))
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
enc, aux, actor = N.Encoder(False), N.Aux(), N.MoEActor()
enc.load_state_dict(ck["enc"]); aux.load_state_dict(ck["aux"]); actor.load_state_dict(ck["actor"])
n = 8; out = {"iteration": int(ck["iter"])}
for kind in sys.argv[2].split(","):
    for v in [float(x) for x in sys.argv[3].split(",")]:
        sp = torch.tensor([[1.0, 2.0 + 0.04 * (i - 3.5), 0.0] for i in range(n)])
        env = SSREnv(n, "cpu", raster=raster(kind), train=False, tiles=False, spawn=sp, camera="chest_rs")
        env.cmd[:] = 0.0; h = enc.init_h(n, "cpu"); x0 = None; acc = torch.zeros(n)
        for t in range(450):
            if t == 50: env.cmd[:] = torch.tensor([v, 0.0, 0.0]); x0 = env.s["p"][:, 0].clone()
            e = enc.step(env.hist, env.depth, h, sample=False); h = e["h"]
            env.step(actor(torch.cat((env.prop, aux.est(env.hist.flatten(1)), e["z"]), -1)).clamp(-8, 8))
            if t >= 250: acc += env.s["v_b"][:, 0] / 200
        out[f"{kind}@{v}"] = dict(dist=round(float((env.s["p"][:, 0] - x0).mean()), 2), v_last4=round(float(acc.mean()), 2),
                                  stalled=round(float((acc < 0.1).float().mean()), 2))
print(json.dumps(out))
