"""Check the chunked heightfield (ACP agent 2026-11-12): geom count and surface equal to the raster (ray casts at vertices)."""
import sys
sys.path.insert(0, "/mnt/filesystem-g4/tracks/acp/code")
import numpy as np, mujoco
from gear_sonic.supermimic.assets import build_x2
import gear_sonic.supermimic.terrain as T
if len(sys.argv) > 1: T.HF_CHUNK = int(sys.argv[1])
for f in ["tp_ramp_hmap_x30_40mm", "hmap_ext_v2_40mm"]:
    z = np.load(f"/mnt/filesystem-g4/tracks/acp/data/clips/{f}.npz")
    r = dict(x0=float(z["x0"]), y0=float(z["y0"]), res=float(z["res"]), grid=z["grid"].astype(np.float32))
    m = build_x2(None, raster=r, foot_terrain=True).model
    hg = [g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_HFIELD]
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    rng = np.random.default_rng(1); err = []
    G = r["grid"]
    for _ in range(400):
        ix = int(rng.integers(1, G.shape[0] - 1)); iy = int(rng.integers(1, G.shape[1] - 1))
        if rng.random() < 0.5:   # seams
            ix = (ix // 399) * 399 if ix >= 399 else ix
        p = np.array([r["x0"] + ix * r["res"], r["y0"] + iy * r["res"], 10.0]); gid = np.zeros(1, np.int32)
        dist = mujoco.mj_ray(m, d, p, np.array([0, 0, -1.0]), None, 1, -1, gid)
        err.append(abs((10 - dist) - G[ix, iy]))
    print(f, "hfield geoms", len(hg), "first", mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, hg[0]),
          "conaffinity", int(m.geom_conaffinity[hg[0]]), "solref", m.geom_solref[hg[0]].tolist(), "ray err max mm", round(1000 * max(err), 2), "n>1cm", int((np.array(err) > 0.01).sum()), "of", len(err))
