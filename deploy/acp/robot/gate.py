"""PC2 pre-flight for ritual_start_x2perception.sh: exit 0 only when every joint group, the torso IMU and the chest depth are
publishing (each heard within the last 0.5 s, waited for up to X2P_GATE_WAIT_S, default 45 s). Subscribes only; commands nothing.

--mc-silent [S]: exit 0 if nothing publishes leg joint commands for S s (default 1; MC is not driving the robot), else 1. The
launch script uses it to skip stopping an MC that is already stopped, and to confirm the stop worked (the `aima em` CLI has hung
before, so its return code alone is not trusted)."""
import os
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, Imu

from aimdk_msgs.msg import JointCommandArray, JointStateArray

TOPICS = {"/aima/hal/joint/%s/state" % g: JointStateArray for g in ("leg", "waist", "arm", "head")}
TOPICS["/aima/hal/imu/torso/state"] = Imu
TOPICS["/x2_acp/chest_depth"] = Image


def mc_silent(secs):
    rclpy.init()
    n = Node("x2p_mc_check")
    heard = []
    n.create_subscription(JointCommandArray, "/aima/hal/joint/leg/command", lambda m: heard.append(1), qos_profile_sensor_data)
    t_end = time.monotonic() + secs
    while time.monotonic() < t_end and not heard:
        rclpy.spin_once(n, timeout_sec=0.05)
    rclpy.try_shutdown()
    print("mc-check: leg commands %s" % ("HEARD (something is commanding)" if heard else "silent for %.1f s" % secs))
    sys.exit(1 if heard else 0)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--mc-silent":
        mc_silent(float(sys.argv[2]) if len(sys.argv) > 2 else 1.0)
    rclpy.init()
    n = Node("x2p_gate")
    seen = {}
    for t, typ in TOPICS.items():
        n.create_subscription(typ, t, lambda m, t=t: seen.__setitem__(t, time.monotonic()), qos_profile_sensor_data)
    t_end = time.monotonic() + float(os.environ.get("X2P_GATE_WAIT_S", "45"))     # a ritual just after boot waits for the robot
    while time.monotonic() < t_end:
        rclpy.spin_once(n, timeout_sec=0.05)
        now = time.monotonic()
        if all(now - seen.get(t, -1e9) < 0.5 for t in TOPICS):
            print("gate: all topics fresh: " + ", ".join(TOPICS))
            rclpy.try_shutdown()
            sys.exit(0)
    print("gate: MISSING " + ", ".join(t for t in TOPICS if time.monotonic() - seen.get(t, -1e9) >= 0.5))
    rclpy.try_shutdown()
    sys.exit(1)


if __name__ == "__main__":
    main()
