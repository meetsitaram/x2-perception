"""PC2, read-only: record the joint command / state bus for --seconds (whoever drives the robot: MC, SONIC or our relay) for the
stiffness fit and the timing comparison. Leg + waist state (position, velocity, measured effort), the torso IMU, and the leg +
waist commands, each with its arrival time. Subscribes only; commands nothing.

    python3 bus_record.py --seconds 60 --out /tmp/bus.npz
"""
import argparse
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu

from aimdk_msgs.msg import JointCommandArray, JointStateArray


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rclpy.init()
    n = Node("x2p_bus_record")
    rec = {"state": [], "cmd": [], "imu": []}
    names = {}

    def on_state(g, m):
        names.setdefault("state_" + g, [j.name for j in m.joints])
        rec["state"].append((time.monotonic(), g, [(j.position, j.velocity, j.effort) for j in m.joints]))

    def on_cmd(g, m):
        names.setdefault("cmd_" + g, [j.name for j in m.joints])
        rec["cmd"].append((time.monotonic(), g, [(j.position, j.velocity, j.effort, j.stiffness, j.damping) for j in m.joints]))

    for g in ("leg", "waist"):
        n.create_subscription(JointStateArray, "/aima/hal/joint/%s/state" % g, lambda m, g=g: on_state(g, m), qos_profile_sensor_data)
        n.create_subscription(JointCommandArray, "/aima/hal/joint/%s/command" % g, lambda m, g=g: on_cmd(g, m), qos_profile_sensor_data)
    n.create_subscription(Imu, "/aima/hal/imu/torso/state", lambda m: rec["imu"].append(
        (time.monotonic(), m.orientation.w, m.orientation.x, m.orientation.y, m.orientation.z,
         m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z)), qos_profile_sensor_data)
    t_end = time.monotonic() + a.seconds
    while time.monotonic() < t_end:
        rclpy.spin_once(n, timeout_sec=0.0005)
    rclpy.try_shutdown()
    out = {"imu": np.array(rec["imu"]), "names": np.array([names], dtype=object)}
    for kind in ("state", "cmd"):
        for g in ("leg", "waist"):
            rows = [(t, *np.ravel(v)) for t, gg, v in rec[kind] if gg == g]
            out["%s_%s" % (kind, g)] = np.array(rows)
    np.savez(a.out, **out)
    print({k: v.shape for k, v in out.items() if k != "names"})


if __name__ == "__main__":
    main()
