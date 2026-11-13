"""Stop probe (CPU, deterministic, open flat floor): stand 1 s, walk at v1 m/s for 3 s, then a zero command for 8 s.
Prints per checkpoint: distance in the first 1 s after the stop command (median, p90), planar drift speed in the last 4 s, survival."""
import json, sys, torch, numpy as np
sys.path.insert(0, "/mnt/filesystem-g4/tracks/ssr/code")
from ssr import networks as N
from ssr.env import SSREnv
torch.set_grad_enabled(False)
Z = np.zeros((int(12 / 0.025) + 1, int(4 / 0.025) + 1), np.float32)
R = dict(z=Z, x0=0.0, y0=0.0, res=0.025, base=0.1, vision=dict(z=Z.copy(), x0=0.0, y0=0.0, res=0.025))
v1 = float(sys.argv[1])
for path in sys.argv[2:]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    enc, aux, actor = N.Encoder(False), N.Aux(), N.MoEActor()
    enc.load_state_dict(ck["enc"]); aux.load_state_dict(ck["aux"]); actor.load_state_dict(ck["actor"])
    n = 16
    sp = torch.tensor([[1.0, 2.0 + 0.03 * (i - 7.5), 0.0] for i in range(n)])
    env = SSREnv(n, "cpu", raster=R, train=False, tiles=False, spawn=sp, camera="chest_rs")
    env.cmd[:] = 0.0; h = enc.init_h(n, "cpu"); p0 = p1 = None; drift = torch.zeros(n); alive = torch.ones(n, dtype=torch.bool)
    for t in range(50 + 150 + 400):
        if t == 50: env.cmd[:] = torch.tensor([v1, 0.0, 0.0])
        if t == 200: env.cmd[:] = 0.0; p0 = env.s["p"][:, :2].clone()
        if t == 250: p1 = env.s["p"][:, :2].clone()
        e = enc.step(env.hist, env.depth, h, sample=False); h = e["h"]
        out = env.step(actor(torch.cat((env.prop, aux.est(env.hist.flatten(1)), e["z"]), -1)).clamp(-8, 8))
        alive &= ~out["done"]
        if t >= 400: drift += env.s["v_b"][:, :2].norm(dim=-1) / 200
    d1 = (p1 - p0).norm(dim=-1)
    print(json.dumps(dict(ckpt=path.split("/")[-1], it=int(ck["iter"]), v1=v1, d1s_med=round(float(d1.median()), 3), d1s_p90=round(float(d1.quantile(0.9)), 3),
          drift_last4=round(float(drift.median()), 3), alive=round(float(alive.float().mean()), 2))), flush=True)
print("DONE")
