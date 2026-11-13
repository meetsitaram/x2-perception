"""Start-from-a-stand probe (CPU, deterministic policy, flat): N robots stand 1 s at zero command with start yaw U(-15, 15) deg,
then get speed v along heading 0; per robot, mean forward speed over the last 4 s of 8 s. Reports the share that started
(mean >= 0.5 v) and the mean. Seeded, so base and candidate see the same starts."""
import json, math, sys, torch
sys.path.insert(0, "/mnt/filesystem-g4/tracks/ssr/code")
from ssr import networks as N, terrain as T
from ssr.env import SSREnv
torch.set_grad_enabled(False)
ckpt, n = sys.argv[1], int(sys.argv[2]); speeds = [float(x) for x in sys.argv[3].split(",")]
env = SSREnv(n, "cpu", train=False, tiles=True, camera="chest_rs", seed=7); env.flat_only = True
ck = torch.load(ckpt, map_location="cpu", weights_only=False)
enc, aux, actor = N.Encoder(False), N.Aux(), N.MoEActor()
enc.load_state_dict(ck["enc"]); aux.load_state_dict(ck["aux"]); actor.load_state_dict(ck["actor"])
g = torch.Generator().manual_seed(1234); yaw0 = (torch.rand(n, generator=g) * 2 - 1) * math.radians(15)
out = {"ckpt": ckpt, "iteration": int(ck["iter"]), "robots": n}
for v in speeds:
    ids = torch.arange(n); env.type[:], env.level[:] = T.FLAT, 0; env.reset(ids)
    env.qpos[:, 3], env.qpos[:, 4], env.qpos[:, 5], env.qpos[:, 6] = torch.cos(yaw0 / 2), 0.0, 0.0, torch.sin(yaw0 / 2)
    env.heading[:] = 0.0; env.speed[:] = 0.0
    env.refresh(); env._contacts(); env._update_obs(ids); env._update_cmd()
    h = enc.init_h(n, "cpu"); acc = torch.zeros(n); alive = torch.ones(n, dtype=torch.bool)
    for t in range(450):
        if t == 50: env.speed[:] = v
        e = enc.step(env.hist, env.depth, h, sample=False); h = e["h"]
        r = env.step(actor(torch.cat((env.prop, aux.est(env.hist.flatten(1)), e["z"]), -1)).clamp(-8, 8)); alive &= ~r["term"]
        if t >= 250: acc += env.qvel[:, 0] / 200
    started = (acc >= 0.5 * v) & alive
    out[str(v)] = dict(started=round(float(started.float().mean()), 3), mean_vx=round(float(acc.mean()), 3), survived=round(float(alive.float().mean()), 3))
print(json.dumps(out))
