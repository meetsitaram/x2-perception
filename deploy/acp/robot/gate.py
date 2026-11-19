"""PC2 pre-flight for ritual_start_x2perception.sh: exit 0 only when every joint group, the torso IMU and the chest depth are
publishing (each heard within the last 0.5 s, waited for up to 15 s). Subscribes only; commands nothing."""
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, Imu

from aimdk_msgs.msg import JointStateArray

TOPICS = {"/aima/hal/joint/%s/state" % g: JointStateArray for g in ("leg", "waist", "arm", "head")}
TOPICS["/aima/hal/imu/torso/state"] = Imu
TOPICS["/x2_acp/chest_depth"] = Image


def main():
    rclpy.init()
    n = Node("x2p_gate")
    seen = {}
    for t, typ in TOPICS.items():
        n.create_subscription(typ, t, lambda m, t=t: seen.__setitem__(t, time.monotonic()), qos_profile_sensor_data)
    t_end = time.monotonic() + 15.0
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
