"""CPU: diverged episodes vs contact depth, single height field vs X2_HF_CHUNK split. Usage: ckpt n level steps"""
import sys, os, time, json, torch, numpy as np
sys.path.insert(0, "/mnt/filesystem-g4/tracks/ssr/code")
import warp as wp
from ssr import networks as N, terrain as T
from ssr.env import SSREnv
torch.set_grad_enabled(False); torch.manual_seed(0)
TY = ("tight_spots", "log_corridors", "homes", "street_front", "stairs", "wedges", "single_beams", "beam_fields", "blocks")
ck, n, lvl, steps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
t0 = time.time()
r = T.add_extended(T.generate(0), types=TY, stairs_inset=True, prop_routes=True, stations=1)
env = SSREnv(n, "cpu", raster=r, tiles=True, train=True, camera="chest_rs", seed=5, ext_phys=24)
env.ext_frac = 1.0; env.side_soft = 2.0
env.ext_type[:] = torch.arange(n) % len(TY); env.ext_level[:] = lvl
env.ep[:] = 1; ids = torch.arange(n); env.reset(ids); env.refresh(); env._contacts(); env._update_obs(ids); env._update_cmd()
c = torch.load(ck, map_location="cpu", weights_only=False)
enc, aux, actor = N.Encoder(False), N.Aux(), N.MoEActor()
enc.load_state_dict(c["enc"]); aux.load_state_dict(c["aux"]); actor.load_state_dict(c["actor"])
h = enc.init_h(n, "cpu")
cdist = wp.to_torch(env.dd.contact.dist); cw = env.c_world
nhf = int((env.mm.geom_type.numpy() == 1).sum()) if hasattr(env, "mm") else -1
prev_depth = torch.zeros(n); prev_v = torch.zeros(n)
events = []; ndone = 0; mind_all = []
for t in range(steps):
    e = enc.step(env.hist, env.depth, h, sample=False); h = e["h"]
    out = env.step(actor(torch.cat((env.prop, aux.est(env.hist.flatten(1)), e["z"]), -1)).clamp(-8, 8))
    k = min(int(env.nacon.max()), cdist.shape[0])
    depth = torch.zeros(n)
    if k:
        d = cdist[:k]; w = cw[:k].long(); ok = w >= 0
        depth.index_reduce_(0, w[ok], (-d[ok]).clamp(min=0), "amax")
    v = env.qvel[:, :3].norm(dim=-1)
    for i in env.bad.nonzero().flatten().tolist():
        events.append(dict(t=t, w=i, type=TY[i % len(TY)], depth_at=round(float(depth[i]), 3), depth_prev=round(float(prev_depth[i]), 3),
                           v_prev=round(float(prev_v[i]), 2)))
    mind_all.append(float(depth[~env.bad].max()) if (~env.bad).any() else 0.0)
    ndone += int(out["done"].sum())
    prev_depth, prev_v = depth.clone(), v.clone()
    h = torch.where(out["done"][:, None], torch.zeros_like(h), h)
deep = sum(e["depth_at"] > 0.05 or e["depth_prev"] > 0.05 for e in events)
print(json.dumps(dict(chunk=os.environ.get("X2_HF_CHUNK", "0"), sim_s=round(time.time() - t0), steps=steps, n=n, done=ndone, diverged=len(events),
      div_with_depth_gt_5cm=deep, max_depth_nonbad_p99=round(float(np.percentile(mind_all, 99)), 3), max_depth_nonbad=round(max(mind_all), 3),
      events=events[:12])))
print("DONE")
