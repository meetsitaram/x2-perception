"""Wall-refusal trend (CPU, deterministic): open floor with one wall (1.3 m, vision only) along the left of the path at
distance d from the centre line; 64 robots start on the line (+-3 cm, yaw +-10 deg) after 1 s standing, command 0.37 m/s for
8 s. Stalled = mean forward speed over the last 4 s < 0.1 m/s."""
import json, math, sys, torch
import numpy as np
sys.path.insert(0, "/mnt/filesystem-g4/tracks/ssr/code")
from ssr import networks as N
from ssr.env import SSREnv
torch.set_grad_enabled(False)
RES = 0.025
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
enc, aux, actor = N.Encoder(False), N.Aux(), N.MoEActor()
enc.load_state_dict(ck["enc"]); aux.load_state_dict(ck["aux"]); actor.load_state_dict(ck["actor"])
n = 64; out = {"iteration": int(ck["iter"]), "robots": n}
g = torch.Generator().manual_seed(11)
dy = (torch.rand(n, generator=g) * 2 - 1) * 0.03; yaw = (torch.rand(n, generator=g) * 2 - 1) * math.radians(10)
for d in [float(x) for x in sys.argv[2].split(",")]:
    nx, ny = int(8 / RES) + 1, int(4 / RES) + 1
    z = np.zeros((nx, ny), np.float32); v = z.copy()
    v[int(0.5 / RES):int(5.0 / RES), int((2.0 + d) / RES):int((2.0 + d + 0.08) / RES)] = 1.3
    R = dict(z=z, x0=0.0, y0=0.0, res=RES, base=0.1, vision=dict(z=v, x0=0.0, y0=0.0, res=RES))
    sp = torch.stack((torch.full((n,), 1.0), 2.0 + dy, yaw), -1)
    env = SSREnv(n, "cpu", raster=R, train=False, tiles=False, spawn=sp, camera="chest_rs")
    env.cmd[:] = 0.0; h = enc.init_h(n, "cpu"); acc = torch.zeros(n)
    for t in range(450):
        if t == 50: env.cmd[:] = torch.tensor([0.37, 0.0, 0.0])
        e = enc.step(env.hist, env.depth, h, sample=False); h = e["h"]
        env.step(actor(torch.cat((env.prop, aux.est(env.hist.flatten(1)), e["z"]), -1)).clamp(-8, 8))
        if t >= 250: acc += env.s["v_b"][:, 0] / 200
    out[str(d)] = dict(stalled=round(float((acc < 0.1).float().mean()), 3), v_last4=round(float(acc.mean()), 3))
    print(json.dumps(out), flush=True)
