"""Turn-in-place and start-from-stand probe (CPU, deterministic, flat). Start: default stand; command fixed for 8 s; mean over the last 4 s (as the eval)."""
import json, os, sys, torch
sys.path.insert(0, "/mnt/filesystem-g4/tracks/ssr/code")
from ssr import networks as N, terrain as T
from ssr.env import SSREnv
ck_path, cam = sys.argv[1], sys.argv[2]
cmds = [(0, 0, 0.2), (0, 0, -0.2), (0, 0, 0.3), (0, 0, -0.3), (0, 0, 0.4), (0, 0, -0.4), (0, 0, 1.0), (0, 0, -1.0), (0.3, 0, 1.0), (0.3, 0, -1.0), (0.4, 0, 0.5)]
torch.set_grad_enabled(False)
env = SSREnv(8, "cpu", train=False, tiles=True, camera=cam); env.flat_only = True
ck = torch.load(ck_path, map_location="cpu", weights_only=False)
enc, aux, actor = N.Encoder(False), N.Aux(), N.MoEActor()
enc.load_state_dict(ck["enc"]); aux.load_state_dict(ck["aux"]); actor.load_state_dict(ck["actor"])
out = {}
for c in cmds:
    ids = torch.arange(8); env.type[:], env.level[:] = T.FLAT, 0; env.reset(ids)
    env.free[:] = True; env.cmd[:] = torch.tensor(c, dtype=torch.float32)
    env.qpos[:, 3], env.qpos[:, 4:7] = 1.0, 0.0
    env.refresh(); env._contacts(); env._update_obs(ids)
    h = enc.init_h(8, "cpu"); wz, vx, n, alive = 0.0, 0.0, 0, torch.ones(8, dtype=torch.bool); p0 = None
    for t in range(400):
        e = enc.step(env.hist, env.depth, h, sample=False); h = e["h"]
        a = actor(torch.cat((env.prop, aux.est(env.hist.flatten(1)), e["z"]), -1))
        env.cmd[:] = torch.tensor(c, dtype=torch.float32)
        r = env.step(a.clamp(-8, 8)); alive &= ~r["term"]
        if t == 200: p0 = env.s["p"][:, :2].clone()
        if t >= 200:
            wz += float(env.s["w_b"][:, 2][alive].mean()) if alive.any() else 0.0; vx += float(env.s["v_b"][:, 0][alive].mean()) if alive.any() else 0.0; n += 1
    out[str(c)] = dict(yaw_rate=round(wz / n, 3), vx=round(vx / n, 3), drift_m_4s=round(float((env.s["p"][:, :2] - p0).norm(dim=-1).mean()), 3), survived=round(float(alive.float().mean()), 2))
print(json.dumps(out))
