"""One still of several moments of an apparatus episode in one frame (TDP / page figure): the same robot drawn at each chosen
time, each with ACP's overlay from the maps it was given (render_park.py's drawing), camera from a fixed pose.

    python3 models/acp_depth/render_still.py --scene runs/park_scene --run runs/vis_fine3/app --extra runs/vis_fine3/extra_app/ep0001.npz \
        --times 6.3,14.4,28.0,44.0 --cam 5.6,1.2,0.5,180,-26,7.0 --out fig.jpg

--cam: lookat x, y, z in the apparatus frame (m), azimuth, elevation (deg), distance (m).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_park as P  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--seed", type=int, default=2007)
    ap.add_argument("--extra", required=True)
    ap.add_argument("--times", required=True)
    ap.add_argument("--cam", required=True)
    ap.add_argument("--size", default="1600x900")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fovy", type=float, default=0.0, help="camera vertical field of view (deg); 0 = the scene's")
    ap.add_argument("--fade", action="store_true", help="fade rails / walls between the camera and the last robot")
    a = ap.parse_args()
    W, H = (int(v) for v in a.size.split("x"))
    scene = pathlib.Path(a.scene)
    sh = np.array([t["shift"] for t in json.load(open(scene / "scene.json"))["tiles"] if t["course"].startswith("apparatus")][0], float)
    run = pathlib.Path(a.run)
    rb = P.Robot("app", "apparatus", run / "traj" / ("%d.npz" % a.seed), a.extra, sh, run / "results.json")
    m, adr = P.build(scene / "scene.xml", 1, (W, H))
    A = adr[0]
    if a.fovy:
        m.vis.global_.fovy = a.fovy
    d = mujoco.MjData(m)
    feet = [m.body("%s_ankle_roll_link" % s).id for s in ("left", "right")]
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    for g in (0, 1, 2, 4, 5):
        opt.geomgroup[g] = 1
    ren = mujoco.Renderer(m, H, W, max_geom=60000)
    cx, cy, cz, az, el, dist = (float(v) for v in a.cam.split(","))
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [cx + sh[0], cy + sh[1], cz + sh[2]]
    cam.azimuth, cam.elevation, cam.distance = az, el, dist
    d.qpos[A + 2] = -50.0
    mujoco.mj_kinematics(m, d)
    ren.update_scene(d, cam, opt)
    scn = ren.scene
    pert = mujoco.MjvPerturb()
    for tl in (float(v) for v in a.times.split(",")):
        d.qpos[A:A + P.NQ] = rb.qpos[rb.k(tl)]
        mujoco.mj_kinematics(m, d)
        mujoco.mjv_addGeoms(m, d, opt, pert, mujoco.mjtCatBit.mjCAT_DYNAMIC, scn)

        class F:
            xpos = d.xpos.copy()
        P.draw_overlay(scn, rb, tl, F, feet)
    # rails and walls between the camera and the robot in focus (the last time given) are drawn nearly transparent
    import math
    pf = rb.qpos[rb.k(float(a.times.split(",")[-1]))][:3]
    azr, elr = math.radians(az), math.radians(el)
    look = np.array(cam.lookat)
    eye = look - dist * np.array([math.cos(elr) * math.cos(azr), math.cos(elr) * math.sin(azr), math.sin(elr)])
    for tgt in ((pf + [0, 0, 0.3], pf - [0, 0, 0.3], pf - [0, 0, 0.62]) if a.fade else ()):
        ab = tgt - eye
        for gi in range(scn.ngeom):
            gm = scn.geoms[gi]
            if gm.objtype != mujoco.mjtObj.mjOBJ_GEOM or m.geom_group[gm.objid] not in (0, 5):
                continue
            c = np.array(gm.pos)
            tt = float(np.clip((c - eye) @ ab / (ab @ ab), 0.0, 1.0))
            if tt < 0.9 and np.linalg.norm(eye + tt * ab - c) < float(m.geom_rbound[gm.objid]) + 0.15:
                gm.rgba[3] = min(gm.rgba[3], 0.12)
    from PIL import Image
    Image.fromarray(ren.render()).save(a.out, quality=92)
    print("->", a.out)


if __name__ == "__main__":
    main()
