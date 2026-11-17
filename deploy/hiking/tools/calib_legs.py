"""Calibrate the chest camera's mount angle from the robot's own legs (user, 2026-11-16: "use the leg positions to calibrate first
rather than relying on the 4x4 alone"). The legs and feet are where the joint angles put them, independent of anything placed
on the floor.

    (ulimit -v 6000000; python3 calib_legs.py --cap deploy/hiking/logs/d435/cal_MMDD_HHMM)

Sim: the robot in the captured pose (compare_mount.sim_view's setup) and the trained camera, ray-cast at 160 x 90 with the same
pinhole as the real frame (nominal 87 x 58 deg); a pixel is "robot" when its ray hits a robot geom. Real: points 2-15 cm above the fitted floor plane and nearer than 0.85 m (feet and lower shins; the 4x4 is ~1 m away);
the sim mask uses the same height band, so hands and arms never count (a strapped or damaged arm cannot skew the fit). The mount's pitch and roll
offsets (camera frame, added to the trained 61.6 deg) are searched for the best overlap (IoU) of the two masks.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import mujoco
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import compare_mount as C  # noqa: E402
from hiking_deploy import core  # noqa: E402

GW, GH = 160, 90
BAND = (0.02, 0.15)       # m above the floor: feet and lower shins only. Hands and arms are ignored: they hang higher, and an
                          # arm can be off its joint angles (2026-11-16: a damaged wrist, the hand strapped out of the way)


def pose_robot(m, d, state):
    q = state["q"]
    R_t = C.quat_xyzw_to_mat(*state["imu"]["xyzw"])
    R_p, _ = core.pelvis_from_torso(R_t, np.zeros(3), [q[n] for n in core.WAIST], np.zeros(3))
    yaw = math.atan2(R_p[1, 0], R_p[0, 0])
    c, s = math.cos(-yaw), math.sin(-yaw)
    R_p = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ R_p
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, R_p.ravel())
    d.qpos[:3] = 0.0
    d.qpos[3:7] = quat
    for j in range(m.njnt):
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE and m.joint(j).name in q:
            d.qpos[m.jnt_qposadr[j]] = q[m.joint(j).name]
    mujoco.mj_kinematics(m, d)
    soles = [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.endswith("ankle_roll_link")
             and m.geom_contype[g] and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE]
    d.qpos[2] = -min(d.geom_xpos[g][2] - m.geom_size[g][0] for g in soles)
    mujoco.mj_kinematics(m, d)


def rays(W=GW, H=GH, hfov=C.HFOV, vfov=C.VFOV):
    """Unit rays in the camera frame (x forward, y left, z up), the same pinhole as compare_mount.real_points."""
    fx, fy = W / 2 / math.tan(math.radians(hfov / 2)), H / 2 / math.tan(math.radians(vfov / 2))
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    r = np.stack([np.ones_like(u), -(u - W / 2) / fx, -(v - H / 2) / fy], -1).reshape(-1, 3)
    return r / np.linalg.norm(r, axis=1, keepdims=True)


def rot_cam(pitch_deg, roll_deg):
    """Extra rotation in the camera frame: pitch about y (positive = looks further down), roll about x."""
    p, r = math.radians(pitch_deg), math.radians(roll_deg)
    Ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
    return Ry @ Rx


def sim_robot_mask(m, d, card, dpitch, droll):
    from hiking_deploy.sim_depth import quat_wxyz_to_mat
    body = m.body(card.depth["body"]).id
    R_off = quat_wxyz_to_mat(card.depth["quat_wxyz"])
    p = d.xpos[body] + d.xmat[body].reshape(3, 3) @ np.array(card.depth["pos_in_body"], float)
    R = d.xmat[body].reshape(3, 3) @ R_off @ rot_cam(dpitch, droll)
    dirs = np.ascontiguousarray(rays() @ R.T)
    n = len(dirs)
    gid, dist = np.zeros(n, np.int32), np.zeros(n)
    gg = np.zeros(6, np.uint8)
    gg[1] = gg[2] = gg[4] = 1                 # robot visual geoms (groups 1, 2) and the floor (4)
    mujoco.mj_multiRay(m, d, p, dirs.ravel(), gg, 1, -1, gid, dist, None, n, 10.0)
    hit_z = p[2] + dist * dirs[:, 2]                 # world height of each hit (the floor is z = 0)
    robot = np.array([g >= 0 and m.body_rootid[m.geom_bodyid[g]] != 0 for g in gid]) & (hit_z > BAND[0]) & (hit_z < BAND[1])
    return robot.reshape(GH, GW), (dist * (dirs @ R[:, 0])).reshape(GH, GW), p, R


def real_robot_mask(z):
    P = C.real_points(z)
    H = z.shape[0]
    mask = (z > 0.3) & (z < 4.0)
    mask[int(0.55 * H):] = False
    n, h, _ = C.floor_fit(P, mask)
    above = P @ n + h
    robot = (z > 0) & (above > BAND[0]) & (above < BAND[1]) & (z < 0.85)
    # down-sample to GH x GW: a cell is robot if most of its valid pixels are
    Hs, Ws = z.shape[0] // GH, z.shape[1] // GW
    cell = robot[:GH * Hs, :GW * Ws].reshape(GH, Hs, GW, Ws).mean((1, 3))
    valid = (z[:GH * Hs, :GW * Ws] > 0).reshape(GH, Hs, GW, Ws).mean((1, 3))
    return cell > 0.5, valid > 0.3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", required=True)
    ap.add_argument("--depth", default="depth640x360.raw")
    ap.add_argument("--scene", default=str(HERE.parent / "sim" / "scenes" / "flat" / "scene.xml"))
    ap.add_argument("--model", default=str(HERE.parent / "models" / "x2-hiking-in-the-wild"))
    a = ap.parse_args()
    cap = pathlib.Path(a.cap)
    state = json.load(open(cap / "x2_state.json"))
    z = np.fromfile(cap / a.depth, np.uint16).reshape(360, 640).astype(float) / 1000.0
    m = mujoco.MjModel.from_xml_path(a.scene)
    d = mujoco.MjData(m)
    pose_robot(m, d, state)
    card = core.Card.load(a.model)
    real, valid = real_robot_mask(z)

    def iou(dp, dr):
        sim = sim_robot_mask(m, d, card, dp, dr)[0]
        inter = (sim & real & valid).sum()
        union = ((sim | real) & valid).sum()
        return inter / max(union, 1)

    best = max(((iou(dp, dr), dp, dr) for dp in np.arange(-10, 25.01, 1.0) for dr in np.arange(-6, 6.01, 2.0)))
    _, p0, r0 = best
    best = max(((iou(dp, dr), dp, dr) for dp in np.arange(p0 - 1.5, p0 + 1.51, 0.25) for dr in np.arange(r0 - 2, r0 + 2.01, 0.5)))
    score, dp, dr = best
    base = iou(0.0, 0.0)
    curve = [(p, round(iou(p, dr), 3)) for p in np.arange(dp - 6, dp + 6.01, 2.0)]
    print("legs/feet overlap (IoU): trained mount %.3f | best %.3f at pitch %+.2f deg (further down = +), roll %+.2f deg"
          % (base, score, dp, dr))
    print("IoU around the best pitch:", curve)
    print("=> real mount ~ %.1f deg down from the torso (trained 61.6), roll %+.1f deg; real robot pixels %d, sim %d (best)"
          % (61.6 + dp, dr, int(real.sum()), int(sim_robot_mask(m, d, card, dp, dr)[0].sum())))
    from PIL import Image
    def overlay(sim):
        img = np.zeros((GH, GW, 3), np.uint8)
        img[real & valid] = (230, 120, 40)
        img[sim] = (60, 140, 255)
        img[sim & real] = (240, 240, 240)
        img[~valid] = (img[~valid] * 0.4).astype(np.uint8)
        return Image.fromarray(img).resize((GW * 4, GH * 4), Image.NEAREST)
    canvas = Image.new("RGB", (GW * 8 + 10, GH * 4), (20, 20, 20))
    canvas.paste(overlay(sim_robot_mask(m, d, card, 0.0, 0.0)[0]), (0, 0))
    canvas.paste(overlay(sim_robot_mask(m, d, card, dp, dr)[0]), (GW * 4 + 10, 0))
    out = cap / "calib_legs.png"
    canvas.save(out)
    print("overlay (orange = real legs, blue = sim legs, white = both; left = trained mount, right = best fit):", out)


if __name__ == "__main__":
    main()
