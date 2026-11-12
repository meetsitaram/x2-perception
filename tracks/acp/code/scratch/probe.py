import sys, time, torch, mujoco, mujoco_warp as mjw, warp as wp, numpy as np
sys.path.insert(0, "/mnt/filesystem-g4/tracks/acp/code")
from gear_sonic.supermimic.assets import build_x2
x2 = build_x2(foot_terrain=True); m = x2.model
print("nbody", m.nbody, "ngeom", m.ngeom, "nu", m.nu, "cone", m.opt.cone, "solver", m.opt.solver, "iters", m.opt.iterations)
print("joints", x2.joint_names)
print("scale", np.round(x2.action_scale,3)); print("kp", np.round(x2.kp,1)); print("q0", np.round(x2.default_q,2))
N = int(sys.argv[1]) if len(sys.argv) > 1 else 64
d = mujoco.MjData(m); d.qpos[2] = 0.72; d.qpos[3] = 1; d.qpos[7:] = x2.default_q; mujoco.mj_forward(m, d)
with wp.ScopedDevice("cuda:0"):
    mm = mjw.put_model(m); dd = mjw.put_data(m, d, nworld=N, njmax=300, nconmax=64)
ctrl = wp.to_torch(dd.ctrl); ctrl[:] = torch.tensor(x2.default_q, device="cuda", dtype=torch.float32)
for _ in range(200): mjw.step(mm, dd)
qpos = wp.to_torch(dd.qpos); print("z after 1s", qpos[:3, 2])
c = dd.contact
print("contact fields", [k for k in dir(c) if not k.startswith("_")])
print("data fields w/ con/efc/cfrc", [k for k in dir(dd) if any(s in k for s in ("con", "efc", "cfrc", "nacon", "nefc"))])
for k in ("dist", "geom", "frame", "worldid", "efc_address", "dim", "type"):
    if hasattr(c, k): print(k, getattr(c, k).shape, getattr(c, k).dtype)
print("efc fields", [k for k in dir(dd.efc) if not k.startswith("_")] if hasattr(dd, "efc") else None)
for k in ("nacon", "ncon", "nefc"):
    if hasattr(dd, k): print(k, getattr(dd, k).numpy() if hasattr(getattr(dd, k), "numpy") else getattr(dd, k))
