"""Command step-down probe (CPU, deterministic, open flat floor): walk at 0.6 m/s for 3 s, then the command steps to v2 for 8 s.
Per v2: share still walking (mean vx over the last 4 s >= 0.5 v2), mean vx in the last 4 s, distance travelled after the step."""
import json, sys, torch
import numpy as np
sys.path.insert(0, "/mnt/filesystem-g4/tracks/ssr/code")
from ssr import networks as N
from ssr.env import SSREnv
torch.set_grad_enabled(False)
z = np.zeros((int(12 / 0.025) + 1, int(4 / 0.025) + 1), np.float32)
R = dict(z=z, x0=0.0, y0=0.0, res=0.025, base=0.1, vision=dict(z=z.copy(), x0=0.0, y0=0.0, res=0.025))
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
enc, aux, actor = N.Encoder(False), N.Aux(), N.MoEActor()
enc.load_state_dict(ck["enc"]); aux.load_state_dict(ck["aux"]); actor.load_state_dict(ck["actor"])
n = 16; out = {"iteration": int(ck["iter"])}
for v2 in [float(x) for x in sys.argv[2].split(",")]:
    sp = torch.tensor([[1.0, 2.0 + 0.03 * (i - 7.5), 0.0] for i in range(n)])
    env = SSREnv(n, "cpu", raster=R, train=False, tiles=False, spawn=sp, camera="chest_rs")
    env.cmd[:] = 0.0; h = enc.init_h(n, "cpu"); acc = torch.zeros(n); x_step = None
    for t in range(50 + 150 + 400):
        if t == 50: env.cmd[:] = torch.tensor([0.6, 0.0, 0.0])
        if t == 200: env.cmd[:] = torch.tensor([v2, 0.0, 0.0]); x_step = env.s["p"][:, 0].clone()
        e = enc.step(env.hist, env.depth, h, sample=False); h = e["h"]
        env.step(actor(torch.cat((env.prop, aux.est(env.hist.flatten(1)), e["z"]), -1)).clamp(-8, 8))
        if t >= 400: acc += env.s["v_b"][:, 0] / 200
    out[str(v2)] = dict(walking=round(float((acc >= 0.5 * v2).float().mean()), 2), vx_last4=round(float(acc.mean()), 3),
                        dist_after_step=round(float((env.s["p"][:, 0] - x_step).mean()), 2))
print(json.dumps(out))
