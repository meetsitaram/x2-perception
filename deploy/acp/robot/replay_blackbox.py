"""Desktop: replay a robot black box (blackbox.py's run_<time>/) in MuJoCo as a video: the recorded joint angles and the torso
IMU's attitude (pelvis via the waist FK). The black box has no position, so it is rebuilt as leg odometry: the lower foot is the
stance foot and stays where it landed (on the floor), and the pelvis follows from the joint angles; when the other foot becomes
the lower one it is planted where it is. Steps, leans and tips move the whole robot as recorded. Kinematic playback, not a
simulation: it shows what the robot did (on a gantry the harness carried part of the weight, which this cannot show). Overlay: time, node state, pelvis tilt, policy
step time; the robot is tinted red while the node is limp (DAMP / SAFE).

    python3 replay_blackbox.py RUN_DIR out.mp4 [--from -2 --to 80]   (seconds from the policy start)
"""
import argparse
import os
import sys

import cv2
import mujoco
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "hiking"))
from hiking_deploy.core import WAIST, pelvis_from_torso  # noqa: E402

MJCF = os.environ.get("X2_MJCF", os.path.expanduser(
    "~/Projects/sonic-final-cut/astra/worktrees/runtime/gear_sonic/data/assets/robot_description/mjcf/x2_ultra.xml"))   # needs its meshes


def quat_wxyz_to_mat(w, x, y, z):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def mat_to_quat_wxyz(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.ravel())
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("out")
    ap.add_argument("--from", dest="t_from", type=float, default=-2.0)
    ap.add_argument("--to", dest="t_to", type=float, default=1e9)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--size", default="960x720")
    a = ap.parse_args()
    W, H = (int(v) for v in a.size.split("x"))
    tick = pd.read_csv(os.path.join(a.run, "tick.csv"))
    qp = pd.read_csv(os.path.join(a.run, "joint_pos.csv"))
    imu = pd.read_csv(os.path.join(a.run, "imu.csv"))
    pol = tick[tick.state.isin(["STAND", "WALK"])]
    t0 = pol.t.iloc[0] if len(pol) else tick.t.iloc[0]

    spec = mujoco.MjSpec.from_file(MJCF)
    w = spec.worldbody
    if not any(g.type == mujoco.mjtGeom.mjGEOM_PLANE for g in spec.geoms):     # the X2 MJCF has its own floor: a second plane at
        w.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[5, 5, 0.1], rgba=[0.55, 0.57, 0.6, 1])   # the same height showed as a double floor
    w.add_light(pos=[1, -1, 3], dir=[-0.3, 0.3, -1], diffuse=[0.8, 0.8, 0.8])
    m = spec.compile()
    m.vis.global_.offwidth, m.vis.global_.offheight = W, H
    d = mujoco.MjData(m)
    jadr = {}
    for j in range(m.njnt):
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
            jadr[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)] = m.jnt_qposadr[j]
    names = [c for c in qp.columns if c != "t" and c in jadr]
    def body_geoms(side):
        return [g for g in range(m.ngeom) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or "")
                == "%s_ankle_roll_link" % side]
    feet = {sd: body_geoms(sd) for sd in ("left", "right")}

    def sole(sd):                                        # the foot's lowest point (geom centres minus part of their radius)
        pts = np.array([d.geom_xpos[g] - [0, 0, m.geom_rbound[g] * 0.3] for g in feet[sd]])
        return pts[np.argmin(pts[:, 2])]
    stance, anchor = None, None
    base_rgba = m.geom_rgba.copy()
    r = mujoco.Renderer(m, H, W)
    cam = mujoco.MjvCamera()
    cam.type, cam.distance, cam.elevation, cam.azimuth = mujoco.mjtCamera.mjCAMERA_FREE, 2.6, -12.0, 135.0
    wr = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (W, H))
    times = np.arange(max(a.t_from, qp.t.iloc[0] - t0), min(a.t_to, qp.t.iloc[-1] - t0), 1.0 / a.fps)
    iq, ii, it = np.searchsorted(qp.t.values, times + t0), np.searchsorted(imu.t.values, times + t0), np.searchsorted(tick.t.values, times + t0)
    for k, tt in enumerate(times):
        row = qp.iloc[min(iq[k], len(qp) - 1)]
        im = imu.iloc[min(ii[k], len(imu) - 1)]
        tk = tick.iloc[min(it[k], len(tick) - 1)]
        for n in names:
            d.qpos[jadr[n]] = row[n]
        R_t = quat_wxyz_to_mat(im.qw, im.qx, im.qy, im.qz)
        R_p, _ = pelvis_from_torso(R_t, np.zeros(3), np.array([row[n] for n in WAIST]), np.zeros(3))
        base = np.array(d.qpos[0:3]) if stance is not None else np.array([0.0, 0.0, 1.0])
        d.qpos[0:3] = base
        d.qpos[3:7] = mat_to_quat_wxyz(R_p)
        mujoco.mj_kinematics(m, d)
        so = {sd: sole(sd) for sd in feet}
        low = min(so, key=lambda sd: so[sd][2])
        if stance is None or (low != stance and so[low][2] < so[stance][2] - 0.01):   # 1 cm hysteresis
            stance = low
            anchor = np.array([so[stance][0], so[stance][1], 0.0])                     # planted where it is, on the floor
        d.qpos[0:3] = base + (anchor - so[stance])
        mujoco.mj_forward(m, d)
        limp = tk.state in ("DAMP", "SAFE")
        m.geom_rgba[:] = base_rgba
        if limp:
            m.geom_rgba[:, 0] = np.minimum(1.0, base_rgba[:, 0] * 0.6 + 0.5)
            m.geom_rgba[:, 1:3] *= 0.5
        cam.lookat[:] = cam.lookat * 0.9 + 0.1 * np.array([d.qpos[0], d.qpos[1], 0.6]) if k else (d.qpos[0], d.qpos[1], 0.6)
        r.update_scene(d, cam)
        img = cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR)
        tilt = np.degrees(np.arccos(np.clip(-im.gz, -1, 1)))
        for j, (txt, col) in enumerate((("t %+6.2f s   %s" % (tt, tk.state), (255, 255, 255)),
                                         ("pelvis tilt %4.1f deg   gyro %.1f rad/s" % (tilt, np.sqrt(im.wx ** 2 + im.wy ** 2 + im.wz ** 2)),
                                          (255, 255, 255) if tilt < 10 else (80, 80, 255)),
                                         ("pad: walk %+.2f  turn %+.2f%s   stance %s   policy step %.0f ms" % (
                                             tk.vx, tk.wz, "" if tk.deadman else " (no deadman)", stance, tk.step_ms), (200, 200, 200)))):
            cv2.putText(img, txt, (16, 36 + 30 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
            cv2.putText(img, txt, (16, 36 + 30 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
        wr.write(img)
    wr.release()
    tmp = a.out + ".h264.mp4"                            # OpenCV's mp4v plays almost nowhere: re-encode as H.264
    if os.system("ffmpeg -y -loglevel error -i '%s' -c:v libx264 -pix_fmt yuv420p -crf 22 -movflags +faststart '%s'" % (a.out, tmp)) == 0:
        os.replace(tmp, a.out)
    print("%d frames (%.1f s from the policy start %+.1f to %+.1f) -> %s" % (len(times), len(times) / a.fps, times[0], times[-1], a.out))


if __name__ == "__main__":
    main()
