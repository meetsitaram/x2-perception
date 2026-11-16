"""Does the real chest camera see what the trained camera would see in the robot's current pose? (user, 2026-11-16: "check if
this view matches the sim if the robot were standing in this exact pose")

    python3 compare_mount.py --state robot_state.json --depth depth360.raw [--size 640x360] --scene sim/scenes/flat/scene.xml

state: one sample of the robot's joint positions and torso IMU (xyzw), read on PC2. depth: one raw Z16 frame (mm) of the
full 87 x 58 deg view (640x360 or 848x480; 640x480 is cropped). Sim: the robot in that pose on flat floor (pelvis from the
torso IMU by the waist FK, yaw removed, lowest sole on z = 0), the trained camera ray-cast (hiking_deploy.sim_depth).
Both are reduced to (a) the floor plane seen by the camera: optical-axis angle below horizontal and camera height, and
(b) the trained raw 64 x 36 image, compared row by row. Output: numbers on stdout and a side-by-side PNG.
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
sys.path.insert(0, str(HERE.parent))
from hiking_deploy import core  # noqa: E402
from hiking_deploy.sim_depth import ChestDepthCaster  # noqa: E402

def quat_xyzw_to_mat(x, y, z, w):          # as hiking_deploy.node (which needs rclpy; this runs on a plain host)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


HFOV, VFOV = 87.0, 58.0                     # D435 depth, nominal (no calibration read yet)


def sim_view(scene, model, state):
    m = mujoco.MjModel.from_xml_path(scene)
    d = mujoco.MjData(m)
    card = core.Card.load(model)
    q = state["q"]
    R_t = quat_xyzw_to_mat(*state["imu"]["xyzw"])
    R_p, _ = core.pelvis_from_torso(R_t, np.zeros(3), [q[n] for n in core.WAIST], np.zeros(3))
    yaw = math.atan2(R_p[1, 0], R_p[0, 0])
    c, s = math.cos(-yaw), math.sin(-yaw)
    R_p = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ R_p
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, R_p.ravel())
    d.qpos[3:7] = quat
    for j in range(m.njnt):
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE and m.joint(j).name in q:
            d.qpos[m.jnt_qposadr[j]] = q[m.joint(j).name]
    mujoco.mj_kinematics(m, d)
    soles = [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.endswith("ankle_roll_link")
             and m.geom_contype[g] and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE]
    d.qpos[:3] = 0.0                          # qpos0 holds the MJCF spawn height; measure from z = 0
    mujoco.mj_kinematics(m, d)
    d.qpos[2] = -min(d.geom_xpos[g][2] - m.geom_size[g][0] for g in soles)
    mujoco.mj_kinematics(m, d)
    cam = ChestDepthCaster(m, card.depth)
    p, R = cam.pose(d)                       # camera frame: x forward, y left, z up
    axis = R[:, 0]
    torso = d.xmat[m.body("torso_link").id].reshape(3, 3)
    return dict(raw=cam.render(d), height=float(p[2]), down_deg=float(-math.degrees(math.asin(axis[2]))),
                torso_pitch_deg=float(math.degrees(math.atan2(-torso[2, 0], torso[2, 2]))), card=card)


def real_points(z, hfov=HFOV, vfov=VFOV):
    """Z16 frame (metres) -> camera-frame points (x forward, y left, z up), pinhole with the nominal FOV."""
    H, W = z.shape
    fx, fy = W / 2 / math.tan(math.radians(hfov / 2)), H / 2 / math.tan(math.radians(vfov / 2))
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    return np.stack([z, -(u - W / 2) / fx * z, -(v - H / 2) / fy * z], -1)


def floor_fit(P, mask, iters=300, tol=0.02, rng=np.random.default_rng(0)):
    pts = P[mask]
    best = None
    for _ in range(iters):
        a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        inl = np.abs((pts - a) @ n) < tol
        if best is None or inl.sum() > best[0].sum():
            best = (inl, n, a)
    inl = best[0]
    Q = pts[inl]
    cen = Q.mean(0)
    n = np.linalg.svd(Q - cen, full_matrices=False)[2][-1]   # economy SVD: the full one is N x N (80 GB; hung the desktop)
    if n @ cen > 0:                          # normal pointing from the floor towards the camera
        n = -n
    return n, float(abs(n @ cen)), float(inl.mean())


def real_view(z):
    P = real_points(z)
    H = z.shape[0]
    mask = (z > 0.3) & (z < 4.0)
    mask[int(0.55 * H):] = False             # the floor ahead, not the feet at the bottom
    n, h, frac = floor_fit(P, mask)
    # n is the floor's up direction in the camera frame; the optical axis is camera x
    down = -math.degrees(math.asin(np.clip(n[0], -1, 1)))   # floor up-vector . optical axis = -sin(angle below horizontal)
    return dict(height=h, down_deg=down, roll_deg=math.degrees(math.atan2(n[1], n[2])), inliers=frac)


def to_raw64(z):
    zz = np.where(z > 0, z, np.inf)
    H, W = z.shape
    ys, xs = np.linspace(0, H, 37).astype(int), np.linspace(0, W, 65).astype(int)
    return np.array([[zz[ys[i]:ys[i + 1], xs[j]:xs[j + 1]].min() for j in range(64)] for i in range(36)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--depth", required=True)
    ap.add_argument("--size", default="640x360")
    ap.add_argument("--scene", default=str(HERE.parent / "sim" / "scenes" / "flat" / "scene.xml"))
    ap.add_argument("--model", default=str(HERE.parent / "models" / "x2-hiking-in-the-wild"))
    ap.add_argument("--out", default=str(HERE.parent / "sim" / "renders" / "mount_compare.png"))
    a = ap.parse_args()
    W, H = (int(v) for v in a.size.split("x"))
    z = np.fromfile(a.depth, np.uint16)[:W * H].reshape(H, W).astype(float) / 1000.0
    state = json.load(open(a.state))
    sim, real = sim_view(a.scene, a.model, state), real_view(z)
    print("pose: knees %.2f / %.2f rad, waist pitch %.2f rad, torso pitch (IMU, sim) %+.1f deg"
          % (state["q"]["left_knee_joint"], state["q"]["right_knee_joint"], state["q"]["waist_pitch_joint"], sim["torso_pitch_deg"]))
    print("camera optical axis below horizontal: trained mount in this pose %.1f deg | real camera %.1f deg | real - trained %+.1f deg"
          % (sim["down_deg"], real["down_deg"], real["down_deg"] - sim["down_deg"]))
    print("camera height above the floor:        trained %.3f m | real %.3f m | real roll %+.1f deg (floor fit, %.0f%% inliers)"
          % (sim["height"], real["height"], real["roll_deg"], 100 * real["inliers"]))
    r_sim, r_real = sim["raw"], to_raw64(z)
    med = lambda r: np.array([np.median(row[np.isfinite(row)]) if np.isfinite(row).any() else np.nan for row in r])  # noqa: E731
    ms, mr = med(r_sim), med(r_real)
    print("raw 64x36 row medians (top -> bottom, every 5th), m:")
    print("  trained:", np.round(ms[::5], 2))
    print("  real:   ", np.round(mr[::5], 2))
    from PIL import Image, ImageDraw
    def img(r):
        v = np.clip(np.where(np.isfinite(r), r, 3.0) / 3.0, 0, 1)
        return Image.fromarray((v * 255).astype(np.uint8)).resize((64 * 8, 36 * 8), Image.NEAREST).convert("RGB")
    canvas = Image.new("RGB", (64 * 16 + 30, 36 * 8 + 50), (24, 26, 30))
    canvas.paste(img(r_sim), (10, 40))
    canvas.paste(img(r_real), (64 * 8 + 20, 40))
    dr = ImageDraw.Draw(canvas)
    dr.text((10, 10), "trained camera, sim, this pose: %.1f deg down, %.2f m" % (sim["down_deg"], sim["height"]), fill=(230, 230, 230))
    dr.text((64 * 8 + 20, 10), "real chest camera: %.1f deg down, %.2f m" % (real["down_deg"], real["height"]), fill=(230, 230, 230))
    canvas.save(a.out)
    print("image:", a.out)


if __name__ == "__main__":
    main()
