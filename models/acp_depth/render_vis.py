"""Videos of ACP driven by the depth student, with the chest depth camera inset bottom right (eval/render.py + the depth
overlay of eval/overlays.py, with ACP's chest camera: the bracket pose at the training centre, 61.6 + 4 deg down, roll -1.5).

    python3 models/acp_depth/render_vis.py RUN_DIR SEED OUT.mp4 "caption" [--extra EP.npz] [--t0 S --t1 S]

--extra: the run's ACPD_EXTRA file for that episode: ACP's overlay (eval/overlays.py ACPOverlay) drawn from the maps ACP was
given, i.e. the student's estimate in a student run: planner tiles, tracker dots, planned landing rings, stance squares.
Apparatus: drawn from the eval's physics, or dressed from APPARATUS_SCENE only when that is the eval's own source scene.
"""
import os
import math
import pathlib
import sys

import numpy as np

# No apparatus dressing by default: one map per render (user, 2026-11-17: "don't have both overlapped together"). terrain-park's
# CAD export is a stale CAD revision (docs/FINDINGS.md B-app1) and drew a second pair of 4x4s over the eval's; without
# APPARATUS_SCENE the apparatus is drawn from the eval's own physics. APPARATUS_SCENE may name a scene that IS the eval's
# geometry (loco-manip-apparatus sim/scene_x2_env_mobility.xml, the source of its raster and walls).
REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "eval"))
import overlays  # noqa: E402
import render  # noqa: E402

PITCH, ROLL = 61.6 + 4.0, -1.5


def cam_R():
    p, r = math.radians(PITCH), math.radians(ROLL)
    Ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
    return Ry @ Rx                                    # camera frame (x forward, y left, z up) in torso_link


class Both:
    """ACP's overlay in the scene + the depth inset in the image."""

    def __init__(self, extra):
        self.inset = ChestDepthInset()
        self.acp = overlays.ACPOverlay(extra) if extra else None
        self.cameras = self.inset.cameras
        self.legend_text = overlays.ACPOverlay.legend_text if extra else None

    def scene(self, scn, x, d, app, t, z, k):
        if self.acp is not None:
            self.acp.scene(scn, x, d, app, t, z, k)
        self.inset.scene(scn, x, d, app, t, z, k)

    def image(self, im, dr, t, top, bottom, font):
        self.inset.minimal = getattr(self, "minimal", True)
        self.inset.image(im, dr, t, top, bottom, font)


class ChestDepthInset(overlays.DepthOverlay):
    """The inset shows what the student saw: the physics geometry the camera is ray-cast against (robot groups 1-2,
    terrain 4, walls 5), never the render-only dressing (group 0), which can differ from the physics (the robot can walk
    through a drawn-only board, and the camera inside it went black)."""

    def scene(self, scn, x, d, app, t, z, k):
        if self.ren is None:
            import mujoco
            m = x.model
            self.m, self.cid = m, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "realsense_sim")
            self.ren = mujoco.Renderer(m, self.RH, self.RW, max_geom=20000)
            self.ren.enable_depth_rendering()
            self.opt = mujoco.MjvOption()
            self.opt.geomgroup[:] = 0
            for g in (1, 2, 4, 5):
                self.opt.geomgroup[g] = 1
        self.ren.update_scene(d, camera=self.cid, scene_option=self.opt)
        self.depth = self.ren.render().copy()

    def __init__(self):
        self.e = dict(cam_R=cam_R(), cam_body="torso_link", cam_pos=np.array([0.131, 0.0, 0.259]),
                      t=np.zeros(1), depth=np.zeros((1, 45, 80)))
        self.track = "acp_depth"
        R = self.e["cam_R"]
        xa, ya = R @ np.array([0.0, -1.0, 0.0]), R @ np.array([0.0, 0.0, 1.0])
        self.cameras = [dict(name="realsense_sim", body="torso_link", pos=[0.131, 0.0, 0.259],
                             xyaxes=[float(v) for v in np.r_[xa, ya]], fovy=self.FOVY)]
        self.ren = None
        self.legend_text = None
        self.minimal = True


if __name__ == "__main__":
    run, seed, out, cap = pathlib.Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
    a = sys.argv[5:]
    t0 = float(a[a.index("--t0") + 1]) if "--t0" in a else None
    t1 = float(a[a.index("--t1") + 1]) if "--t1" in a else None
    extra = a[a.index("--extra") + 1] if "--extra" in a else None
    p = render.render(run, seed, "chase", "acp_depth", t0=t0, t1=t1, caption=cap, out_path=out, overlay=Both(extra))
    print("->", out)
