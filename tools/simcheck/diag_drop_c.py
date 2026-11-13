"""Diagnostic: repeat failing foot-drop cases of a preflight report in MuJoCo C (CPU) and in MuJoCo Warp, same model, and
print the deepest contact of each, plus where the site sits relative to the height-field chunk borders.
usage (track venv, PYTHONPATH=/mnt/filesystem-g4/tools): python -m simcheck.diag_drop_c report.json [--hf-fix 1] [--cfg ...]"""
import argparse
import importlib
import json

import mujoco
import numpy as np

from simcheck import core

ap = argparse.ArgumentParser()
ap.add_argument("report")
ap.add_argument("--cfg", default=None)
ap.add_argument("--hf-fix", default=None)
ap.add_argument("--hf-chunk", default=None)
a = ap.parse_args()
rep = json.load(open(a.report))
b = importlib.import_module(f"simcheck.adapters.{rep['track']}").build(dict(cfg=a.cfg, hf_fix=a.hf_fix, hf_chunk=a.hf_chunk))
m = b["model"]
fm, _, _ = core.foot_model(m, b["foot"])
fb = mujoco.mj_name2id(fm, mujoco.mjtObj.mjOBJ_BODY, "foot")
fails = [c for c in rep["checks"] if c["check"] == "foot_drop"][0]["failed_cases"]
seen = set()
for f in fails:
    key = (f["x"], f["y"], f["v"])
    if key in seen:
        continue
    seen.add(key)
    d = mujoco.MjData(fm)
    mujoco.mj_forward(fm, d)
    h = core._ground(fm, d, (f["x"], f["y"]), fb)
    d.qpos[:3] = [0, 0, 0.3]
    mujoco.mj_forward(fm, d)
    low = min(core._bottom(fm, d, g) for g in range(fm.ngeom) if fm.geom_bodyid[g] == fb) - 0.3
    d = mujoco.MjData(fm)
    d.qpos[:3] = [f["x"], f["y"], h - low + 0.002]
    d.qpos[3:7] = [1, 0, 0, 0]
    d.qvel[0], d.qvel[2] = core.DROP_SIDE_V, -f["v"]
    deep, t_over, x_at = 0.0, 0.0, None
    p = np.array(f.get("pull_dir", [1.0, 0.0]) + [0.0])
    for ks in range(int(round(core.DROP_T / fm.opt.timestep))):
        t = ks * fm.opt.timestep
        if abs(t - core.PULL_T0) < 1e-9:
            x_at = d.qpos[:2].copy(); d.xfrc_applied[fb, :3] = core.PULL_N * p
        if abs(t - core.PULL_T1) < 1e-9:
            moved = float((d.qpos[:2] - x_at) @ p[:2]); d.xfrc_applied[fb, :3] = 0
        mujoco.mj_step(fm, d)
        dm = max([-d.contact[i].dist for i in range(d.ncon)] + [0.0])
        deep = max(deep, dm)
        t_over += fm.opt.timestep * (dm > core.DROP_DEPTH)
    # chunk position: nearest height-field geom border (x, y distance in m)
    dd = mujoco.MjData(fm); mujoco.mj_forward(fm, dd)
    bd = []
    for g in range(fm.ngeom):
        if fm.geom_type[g] == mujoco.mjtGeom.mjGEOM_HFIELD:
            sx, sy = fm.hfield_size[fm.geom_dataid[g]][:2]
            cx, cy = dd.geom_xpos[g][:2]
            if abs(f["x"] - cx) <= sx and abs(f["y"] - cy) <= sy:
                bd.append(round(float(min(sx - abs(f["x"] - cx), sy - abs(f["y"] - cy))), 3))
    print(json.dumps(dict(kind=f["kind"], v=f["v"], x=f["x"], y=f["y"], warp_deep_mm=f["deep_mm"], warp_t=f["t_over_1cm_s"],
                          warp_pull=f["pull_moved_m"], c_deep_mm=round(1000 * deep, 1), c_t=round(t_over, 3), c_pull=round(moved, 3),
                          dist_to_field_border_m=bd)))
