p = "/mnt/filesystem-g4/tracks/acp/code/gear_sonic/supermimic/terrain.py"
s = open(p).read()
if "HF_CHUNK" in s:
    raise SystemExit("already patched")
old = s[s.index("def add_hfield(spec, raster: dict, base: float = 0.05):"):]
old = old[:old.index("    return spec") + len("    return spec")]
new = '''HF_CHUNK = 400   # vertices per side of one heightfield geom (ACP agent 2026-11-12, see below)


def add_hfield(spec, raster: dict, base: float = 0.05):
    """The course raster as static heightfield geoms (cell centres on the raster's points).
    Chunked (ACP agent 2026-11-12, state/acp.md 17c): MuJoCo Warp mis-collides on one 2,260 x 1,819 field (a foot dropped on a
    flat 1.66 m platform got a 9 m deep contact and was thrown 240 m; MuJoCo C and any crop up to 681 x 763 are correct), so the
    raster is split into pieces of at most HF_CHUNK vertices per side that share their border vertices (same surface). A raster
    that fits in one piece gives one geom named "course", as before.
    ACP_HF_FIX=1 (fixes A + B of state/acp.md 16y, measured with tools/drop_test.py): the field collides only with conaffinity
    bit 1 (sole spheres and other robot geoms; the terrain foot boxes no longer sink into it, they still meet props), and its
    contact is stiffer (solref 0.01 s, priority 1; 0.005 clamps to 2 dt). Default 0 keeps the running stage-2 physics."""
    import os
    import mujoco
    import numpy as np
    fix = os.environ.get("ACP_HF_FIX", "0") == "1"
    grid = raster["grid"]
    NX, NY = grid.shape
    res = raster["res"]
    step = HF_CHUNK - 1
    xs = list(range(0, max(NX - 1, 1), step))
    ys = list(range(0, max(NY - 1, 1), step))
    one = len(xs) == 1 and len(ys) == 1
    for i0 in xs:
        for j0 in ys:
            sub = grid[i0:min(i0 + HF_CHUNK, NX), j0:min(j0 + HF_CHUNK, NY)]
            nx, ny = sub.shape
            if nx < 2 or ny < 2:
                continue
            name = "course" if one else f"course_{i0}_{j0}"
            top = max(float(sub.max()), 1e-3)
            hf = spec.add_hfield(name=name, nrow=ny, ncol=nx, size=[(nx - 1) * res / 2, (ny - 1) * res / 2, top, base])
            hf.userdata = (sub.T / top).astype(np.float32).ravel().tolist()        # rows along y, columns along x
            g = spec.worldbody.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname=name,
                                        pos=[raster["x0"] + (i0 + (nx - 1) / 2) * res, raster["y0"] + (j0 + (ny - 1) / 2) * res, 0.0],
                                        rgba=[0.55, 0.5, 0.45, 1.0])
            g.conaffinity = 1 if fix else (1 | TERRAIN_BIT)
            if fix:
                g.solref = [0.01, 1.0]
                g.priority = 1
    return spec'''
s = s.replace(old, new)
open(p, "w").write(s)
print("patched")
