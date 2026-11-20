"""PC2: what the deployed ACP stack perceives, without commanding anything. The live chest depth, torso IMU and joint states go
through the same ACPCore.step the deploy node runs (models/student.onnx -> fine map -> planner scan -> contact targets, goal 0),
for --seconds; every camera frame's result is saved to an npz for render_perception.py on the desktop. Subscribes only.

    python3 perception_view.py --acp models/acp --student models/student.onnx --cam-pitch 8.25 --cam-roll -3.5 --out /tmp/x.npz
"""
import argparse
import os
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, Imu

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "hiking"))
from aimdk_msgs.msg import JointStateArray  # noqa: E402
from acp_deploy import core  # noqa: E402
from hiking_deploy.node import quat_xyzw_to_mat  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acp", required=True)
    ap.add_argument("--student", required=True)
    ap.add_argument("--cam-pitch", type=float, default=0.0)
    ap.add_argument("--cam-roll", type=float, default=0.0)
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--goal", default="0,0,0", help="planner goal (gx, gy, dpsi) as the node gets it from the pad; 1,0,0 = walk ahead")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    acp = core.ACPCore(a.acp, a.student, threads=2)
    rclpy.init()
    n = Node("x2p_perception_view")
    st = dict(depth=None, seq=0, imu=None, q={}, qd={})

    def on_depth(m):
        st["depth"] = np.frombuffer(m.data, np.float32).reshape(m.height, m.width).copy()
        st["seq"] += 1

    def on_imu(m):
        o, w = m.orientation, m.angular_velocity
        st["imu"] = (quat_xyzw_to_mat(o.x, o.y, o.z, o.w), np.array([w.x, w.y, w.z]))

    def on_js(m):
        for j in m.joints:
            st["q"][j.name], st["qd"][j.name] = j.position, j.velocity

    n.create_subscription(Image, "/x2_acp/chest_depth", on_depth, qos_profile_sensor_data)
    n.create_subscription(Imu, "/aima/hal/imu/torso/state", on_imu, qos_profile_sensor_data)
    for g in ("leg", "waist", "arm", "head"):
        n.create_subscription(JointStateArray, "/aima/hal/joint/%s/state" % g, on_js, qos_profile_sensor_data)
    out, used, t_end = {k: [] for k in ("depth", "fine", "plan", "track", "targets", "occ", "camT", "R_pelvis")}, 0, None
    while rclpy.ok():
        rclpy.spin_once(n, timeout_sec=0.01)
        if st["depth"] is None or st["imu"] is None or any(j not in st["q"] for j in acp.joints) or st["seq"] == used:
            continue
        t_end = t_end or time.monotonic() + a.seconds
        used = st["seq"]
        _, info = acp.step(st["imu"][0], st["imu"][1], st["q"], st["qd"], np.array([float(v) for v in a.goal.split(",")]), depth=st["depth"],
                           cam_pitch=a.cam_pitch, cam_roll=a.cam_roll)
        q = np.array([st["q"][j] for j in acp.joints])
        camT = core.cam_T(info["R_pelvis"], q[acp.iw], a.cam_pitch, a.cam_roll)
        for k, v in (("depth", st["depth"]), ("fine", acp.fine), ("plan", info["plan"]), ("track", info["track"]),
                     ("targets", info["targets"]), ("occ", info["occ"]), ("camT", camT), ("R_pelvis", info["R_pelvis"])):
            out[k].append(np.asarray(v, np.float32))
        if time.monotonic() > t_end:
            break
    rclpy.try_shutdown()
    np.savez(a.out, **{k: np.array(v) for k, v in out.items()}, joints=np.array(acp.joints),
             q=np.array([st["q"][j] for j in acp.joints]), cam=np.array([a.cam_pitch, a.cam_roll]))
    print("%d frames -> %s" % (len(out["fine"]), a.out))


if __name__ == "__main__":
    main()
