"""Foot drop test on ACP's own height field (ACP agent, 2026-11-12; after SSR's eval/diag_1812/drop_test.py).
A 10 kg foot carrying the X2's left-foot collision geoms of the training model (sole spheres + the terrain foot box, contype /
conaffinity / solref / friction / margin copied from build_x2(foot_terrain=True)) is dropped onto a real ACP raster at sites of
three kinds (flat, slope, edge = a height jump > 8 cm between neighbouring cells under the foot), vz = -1..4.5 m/s with 1 m/s
sideways, then a 150 N sideways pull for 0.2 s. Same options as training (timestep, integrator, MuJoCo Warp on CPU).
Per case: deepest contact (sphere / box), time with any contact deeper than 1 cm, time from first to last such step (release),
horizontal-normal contacts, distance moved by the pull.
Usage: drop_test.py RASTER.npz VARIANT [res_override]   VARIANT: base | A | AB | ABsr005 (A: hfield conaffinity 1 only, so
the terrain foot box ignores the field; B: hfield solref 0.01 / priority 1)."""
import sys, json, numpy as np, mujoco, warp as wp, torch
sys.path.insert(0, "/mnt/filesystem-g4/tracks/acp/code")
import mujoco_warp as mjw
from gear_sonic.supermimic.assets import build_x2
from gear_sonic.supermimic.terrain import TERRAIN_BIT
RAST, VAR = sys.argv[1], sys.argv[2]
z = np.load(RAST); grid = z["grid"].astype(np.float32); res = float(z["res"]); x0, y0 = float(z["x0"]), float(z["y0"])
full = build_x2(None, foot_terrain=True).model
foot = mujoco.mj_name2id(full, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
geoms = [g for g in range(full.ngeom) if full.geom_bodyid[g] == foot and (full.geom_contype[g] or full.geom_conaffinity[g])]
nx, ny = grid.shape
# sites: local stats over the foot footprint (0.26 x 0.12 m)
rng = np.random.default_rng(0)
fx, fy = int(0.13 / res) + 1, int(0.06 / res) + 1
from numpy.lib.stride_tricks import sliding_window_view as swv
pad = np.pad(grid, ((fx, fx), (fy, fy)), mode="edge")
win = swv(pad, (2 * fx + 1, 2 * fy + 1))
lo, hi = win.min(axis=(2, 3)), win.max(axis=(2, 3))
jump = np.maximum(np.abs(np.diff(grid, axis=0, prepend=grid[:1])), np.abs(np.diff(grid, axis=1, prepend=grid[:, :1])))
jpad = swv(np.pad(jump, ((fx, fx), (fy, fy))), (2 * fx + 1, 2 * fy + 1)).max(axis=(2, 3))
border = np.zeros_like(grid, bool); border[fx * 2:-fx * 2, fy * 2:-fy * 2] = True
kinds = {"flat": border & (hi - lo < 0.005) & (hi > 0.02),
         "slope": border & (hi - lo > 0.03) & (hi - lo < 0.15) & (jpad < 0.02),
         "edge": border & (jpad > 0.08) & (jump > 0.08)}
sites = []
for k, msk in kinds.items():
    idx = np.argwhere(msk)
    if not len(idx):
        k2 = border & (hi - lo < 0.005) if k == "flat" else None
        if k2 is None: continue
        idx = np.argwhere(k2)
    for i in rng.choice(len(idx), size=min(2, len(idx)), replace=False):
        ix, iy = idx[i]; sites.append((k, int(ix), int(iy), float(hi[ix, iy])))
spec = mujoco.MjSpec()
spec.option.timestep = full.opt.timestep; spec.option.integrator = full.opt.integrator; spec.option.cone = full.opt.cone
spec.option.impratio = full.opt.impratio
if VAR.startswith("train"):   # the training code's own add_hfield (chunked; ACP_HF_FIX from the environment); train1 = one geom
    import os, gear_sonic.supermimic.terrain as TT
    if VAR == "train1": TT.HF_CHUNK = 10 ** 6
    TT.add_hfield(spec, dict(grid=grid, res=res, x0=x0, y0=y0))
    VAR = VAR + ("_fix" if os.environ.get("ACP_HF_FIX") == "1" else "_nofix")
else:
    top = max(float(grid.max()), 1e-3)
    hf = spec.add_hfield(name="t", nrow=ny, ncol=nx, size=[(nx - 1) * res / 2, (ny - 1) * res / 2, top, 0.05])
    hf.userdata = (grid.T / top).astype(np.float32).ravel().tolist()
    g = spec.worldbody.add_geom(name="t", type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname="t",
                                pos=[x0 + (nx - 1) * res / 2, y0 + (ny - 1) * res / 2, 0.0])
    g.contype = 0
    g.conaffinity = 1 if VAR.startswith("A") else (1 | TERRAIN_BIT)
    if VAR.startswith("AB"):
        g.solref = [0.005 if "sr005" in VAR else 0.01, 1.0]; g.priority = 1
nhf = sum(1 for _ in spec.geoms if _.type == mujoco.mjtGeom.mjGEOM_HFIELD)
b = spec.worldbody.add_body(name="foot", pos=[0, 0, 0.3]); b.add_freejoint()
isbox = []
for i, gi in enumerate(geoms):
    s = b.add_geom(name=f"g{i}", type=mujoco.mjtGeom(int(full.geom_type[gi])), size=full.geom_size[gi].tolist(),
                   pos=full.geom_pos[gi].tolist(), quat=full.geom_quat[gi].tolist())
    s.contype, s.conaffinity = int(full.geom_contype[gi]), 0
    s.solref, s.solimp, s.friction = full.geom_solref[gi].tolist(), full.geom_solimp[gi].tolist(), full.geom_friction[gi].tolist()
    s.margin, s.gap, s.condim, s.priority = float(full.geom_margin[gi]), float(full.geom_gap[gi]), int(full.geom_condim[gi]), int(full.geom_priority[gi])
    s.mass = 10.0 / len(geoms)
    isbox.append(full.geom_type[gi] == mujoco.mjtGeom.mjGEOM_BOX)
m = spec.compile()
isbox = np.array([False] * nhf + isbox)  # the first nhf geoms are the field
d = mujoco.MjData(m); mujoco.mj_forward(m, d)
low = min(float(d.geom_xpos[i + nhf][2] - (m.geom_size[i + nhf][0] if m.geom_type[i + nhf] == mujoco.mjtGeom.mjGEOM_SPHERE else m.geom_size[i + nhf][2])) - 0.3
          for i in range(len(geoms)))
speeds = [1.0, 2.0, 3.0, 4.0, 4.5]
cases = [(v, s) for s in sites for v in speeds]; W = len(cases)
with wp.ScopedDevice("cpu"):
    mm = mjw.put_model(m); dd = mjw.put_data(m, d, nworld=W, nconmax=200, njmax=600)
    T = wp.to_torch
    qpos, qvel, xfrc = T(dd.qpos), T(dd.qvel), T(dd.xfrc_applied)
    for w, (v, (k, ix, iy, h)) in enumerate(cases):
        qpos[w, :3] = torch.tensor([x0 + ix * res, y0 + iy * res, h - low + 0.002]); qpos[w, 3:7] = torch.tensor([1.0, 0, 0, 0])
        qvel[w, :] = 0; qvel[w, 0] = 1.0; qvel[w, 2] = -v
    dt = m.opt.timestep
    ds, db = np.zeros(W), np.zeros(W); t1 = np.zeros(W); first = np.full(W, -1.0); last = np.full(W, -1.0)
    hmax = np.zeros(W, int); nmax = np.zeros(W, int); x_at = np.zeros(W); x_end = np.zeros(W)
    for kstep in range(int(0.8 / dt)):
        t = kstep * dt
        if abs(t - 0.10) < 1e-9: x_at[:] = qpos[:, 0].numpy(); xfrc[:, 1, 0] = 150.0
        if abs(t - 0.30) < 1e-9: x_end[:] = qpos[:, 0].numpy(); xfrc[:, 1, 0] = 0.0
        mjw.step(mm, dd)
        n = min(int(T(dd.nacon)[0]), dd.contact.dist.shape[0])
        if n == 0: continue
        wid = T(dd.contact.worldid)[:n].numpy(); dist = T(dd.contact.dist)[:n].numpy(); fr = T(dd.contact.frame)[:n].numpy()
        gg = T(dd.contact.geom)[:n].numpy(); bx = isbox[gg].any(1)
        dep = -dist; hor = np.abs(fr[:, 0, 2]) < 0.7
        for w in range(W):
            sel = wid == w
            if not sel.any(): continue
            nmax[w] = max(nmax[w], int(sel.sum())); hmax[w] = max(hmax[w], int((hor & sel).sum()))
            if (sel & ~bx).any(): ds[w] = max(ds[w], dep[sel & ~bx].max())
            if (sel & bx).any(): db[w] = max(db[w], dep[sel & bx].max())
            if dep[sel].max() > 0.01:
                t1[w] += dt; last[w] = t
                if first[w] < 0: first[w] = t
    fin = qpos[:, 2].numpy()
out = []
for w, (v, (k, ix, iy, h)) in enumerate(cases):
    out.append(dict(kind=k, v=v, site=[ix, iy], deep_sphere_mm=round(1000 * ds[w], 1), deep_box_mm=round(1000 * db[w], 1),
                    t_over_1cm_s=round(t1[w], 3), release_s=round(last[w] - first[w] + dt, 3) if first[w] >= 0 else 0.0,
                    horiz_max=int(hmax[w]), contacts_max=int(nmax[w]), pull_moved_m=round(float(x_end[w] - x_at[w]), 3),
                    final_z_above_site=round(float(fin[w] - h), 3)))
summ = {}
for k in kinds:
    r = [o for o in out if o["kind"] == k]
    if r: summ[k] = dict(n=len(r), deep_sphere_max=max(o["deep_sphere_mm"] for o in r), deep_box_max=max(o["deep_box_mm"] for o in r),
                         t_over_1cm_max=max(o["t_over_1cm_s"] for o in r), release_max=max(o["release_s"] for o in r),
                         pull_min=min(o["pull_moved_m"] for o in r), contacts_max=max(o["contacts_max"] for o in r))
print(json.dumps(dict(raster=RAST, res=res, variant=VAR, ngeom_foot=len(geoms), nbox=int(isbox.sum()), summary=summ, cases=out)))
