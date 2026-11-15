p = "/mnt/filesystem-g4/tracks/ssr/code/ssr/x2.py"
s = open(p).read()
old = '''        hf = spec.add_hfield(name="terrain", nrow=ny, ncol=nx, size=[(nx - 1) * res / 2, (ny - 1) * res / 2, top, base])
        hf.userdata = ((z - zmin) / top).T.astype(np.float32).ravel().tolist()
        g = spec.worldbody.add_geom(name="terrain", type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname="terrain",
                                    pos=[hfield["x0"] + (nx - 1) * res / 2, hfield["y0"] + (ny - 1) * res / 2, zmin],
                                    rgba=[0.55, 0.5, 0.45, 1.0])
        g.contype, g.conaffinity = 0, TERRAIN_BIT
'''
new = '''        import os as _os2
        CH = int(_os2.environ.get("X2_HF_CHUNK", "0"))                 # 2026-11-12: >0 splits the raster into fields of <= CH x CH vertices
        if CH <= 1:                                                     # (MuJoCo Warp mis-collides on one very large field; ACP finding)
            hf = spec.add_hfield(name="terrain", nrow=ny, ncol=nx, size=[(nx - 1) * res / 2, (ny - 1) * res / 2, top, base])
            hf.userdata = ((z - zmin) / top).T.astype(np.float32).ravel().tolist()
            g = spec.worldbody.add_geom(name="terrain", type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname="terrain",
                                        pos=[hfield["x0"] + (nx - 1) * res / 2, hfield["y0"] + (ny - 1) * res / 2, zmin],
                                        rgba=[0.55, 0.5, 0.45, 1.0])
            g.contype, g.conaffinity = 0, TERRAIN_BIT
        else:
            for i0 in range(0, nx - 1, CH - 1):                           # chunks share their edge vertices
                for j0 in range(0, ny - 1, CH - 1):
                    sub = z[i0:min(i0 + CH, nx), j0:min(j0 + CH, ny)]
                    sx, sy = sub.shape
                    if sx < 2 or sy < 2:
                        continue
                    smin = float(sub.min()); stop = max(float(sub.max()) - smin, 1e-3)
                    nm = f"terrain_{i0}_{j0}"
                    hf = spec.add_hfield(name=nm, nrow=sy, ncol=sx, size=[(sx - 1) * res / 2, (sy - 1) * res / 2, stop, base])
                    hf.userdata = ((sub - smin) / stop).T.astype(np.float32).ravel().tolist()
                    g = spec.worldbody.add_geom(name=nm, type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname=nm,
                                                pos=[hfield["x0"] + (i0 + (sx - 1) / 2) * res, hfield["y0"] + (j0 + (sy - 1) / 2) * res, smin],
                                                rgba=[0.55, 0.5, 0.45, 1.0])
                    g.contype, g.conaffinity = 0, TERRAIN_BIT
'''
assert s.count(old) == 1
open(p, "w").write(s.replace(old, new))
print("patched")
