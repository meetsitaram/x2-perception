"""Diag (ACP agent 2026-11-12): is the v2 ejection a large-hfield defect? Drop the foot (spheres only, AB contact) at one site of
a raster, on the full raster and on crops of it, in MuJoCo Warp CPU and in MuJoCo C. Prints deepest contact and final height."""
import sys, numpy as np, mujoco, warp as wp, torch
sys.path.insert(0, "/mnt/filesystem-g4/tracks/acp/code")
import mujoco_warp as mjw
from gear_sonic.supermimic.assets import build_x2
R = np.load(sys.argv[1]); ix, iy = int(sys.argv[2]), int(sys.argv[3]); v = float(sys.argv[4])
full = build_x2(None, foot_terrain=True).model
foot = mujoco.mj_name2id(full, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
sph = [g for g in range(full.ngeom) if full.geom_bodyid[g] == foot and full.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE and full.geom_contype[g]]
def model(grid, x0, y0, res):
    nx, ny = grid.shape; top = max(float(grid.max()), 1e-3)
    sp = mujoco.MjSpec(); sp.option.timestep = full.opt.timestep; sp.option.integrator = full.opt.integrator
    hf = sp.add_hfield(name="t", nrow=ny, ncol=nx, size=[(nx - 1) * res / 2, (ny - 1) * res / 2, top, 0.05])
    hf.userdata = (grid.T / top).astype(np.float32).ravel().tolist()
    g = sp.worldbody.add_geom(name="t", type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname="t", pos=[x0 + (nx - 1) * res / 2, y0 + (ny - 1) * res / 2, 0])
    g.conaffinity = 1; g.solref = [0.01, 1]; g.priority = 1
    b = sp.worldbody.add_body(name="f"); b.add_freejoint()
    for i, gi in enumerate(sph):
        s = b.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=full.geom_size[gi].tolist(), pos=full.geom_pos[gi].tolist()); s.mass = 10 / len(sph)
    return sp.compile()
grid, res, x0, y0 = R["grid"].astype(np.float32), float(R["res"]), float(R["x0"]), float(R["y0"])
low = min(full.geom_pos[g][2] - full.geom_size[g][0] for g in sph)
h = grid[ix - 4:ix + 5, iy - 2:iy + 3].max()
for half in [0, 400, 100, 25]:
    if half: g2 = grid[ix - half:ix + half + 1, iy - half:iy + half + 1]; ox, oy = x0 + (ix - half) * res, y0 + (iy - half) * res
    else: g2, ox, oy = grid, x0, y0
    m = model(g2, ox, oy, res)
    for eng in ("warp", "C"):
        d = mujoco.MjData(m); d.qpos[:3] = [x0 + ix * res, y0 + iy * res, h - low + 0.002]; d.qpos[3] = 1; d.qvel[0] = 1; d.qvel[2] = -v
        deep = 0.0
        if eng == "C":
            for k in range(160):
                mujoco.mj_step(m, d); deep = max(deep, -min([d.contact[i].dist for i in range(d.ncon)] or [0]))
            z = d.qpos[2]
        else:
            with wp.ScopedDevice("cpu"):
                mm = mjw.put_model(m); dd = mjw.put_data(m, d, nworld=1, nconmax=200, njmax=600)
                for k in range(160):
                    mjw.step(mm, dd); n = int(dd.nacon.numpy()[0]); deep = max(deep, -min(dd.contact.dist.numpy()[:n].tolist() or [0]))
                z = float(dd.qpos.numpy()[0, 2])
        print(f"crop half {half or full} shape {g2.shape} {eng}: deepest {1000*deep:.1f} mm, final z above site {z - h:.3f} m", flush=True)
