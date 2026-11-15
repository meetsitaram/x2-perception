"""SSR on the real X2 (ROS 2 Humble, AimDK topics, run on PC2 alongside the HAL with motion control stopped).

Default is DRY RUN: it reads the sensors, runs the policy at 50 Hz and logs what it would command, and publishes nothing.
--arm publishes joint commands. Hang the robot on the gantry first. The motion-control app must be stopped, the way
gear_sonic_deploy/deploy_x2.sh does it (Y/n gate, MC stop_app).

    python3 deploy/ssr_x2_node.py --onnx deploy/ssr_x2.onnx             # dry run
    python3 deploy/ssr_x2_node.py --onnx deploy/ssr_x2.onnx --arm --vx 0.3

Safety: 2 s ramp from the measured pose to the policy; the node stops commanding and falls back to a damped hold if the pelvis tilts
past 40 deg, the IMU or a joint stream goes stale for 100 ms, the depth stream for 300 ms, or the policy produces a non-finite action.
"""
import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import rclpy
from aimdk_msgs.msg import JointCommandArray, JointStateArray
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, Imu

from ssr_core import Core, preprocess_depth
from ssr import x2 as X

GROUPS = dict(leg=(0, 12), waist=(12, 3), arm=(15, 14), head=(29, 2))
RAMP_S, TILT_MAX = 2.0, np.deg2rad(40.0)


class SSRNode(Node):
    def __init__(self, a):
        super().__init__("ssr_x2")
        self.a = a
        self.x2 = X.build(foot_boxes=False)
        self.core = Core(a.onnx, self.x2)
        self.q, self.qd = np.zeros(31), np.zeros(31)
        self.seen, self.stamp = {}, {}
        self.quat, self.gyro, self.depth = None, None, None
        self.pubs = {}
        for g, (st, n) in GROUPS.items():
            self.create_subscription(JointStateArray, f"/aima/hal/joint/{g}/state", self._joint(g, st, n), qos_profile_sensor_data)
            self.pubs[g] = self.create_publisher(JointCommandArray, f"/aima/hal/joint/{g}/command", qos_profile_sensor_data)
        self.create_subscription(Imu, a.imu_topic, self._imu, qos_profile_sensor_data)
        self.create_subscription(Image, a.depth_topic, self._depth, qos_profile_sensor_data)
        self.t0 = None
        self.q_start = None
        self.fault = None
        self.cmd = np.array([a.vx, a.vy, a.wz], np.float32)
        self.timer = self.create_timer(X.CONTROL_DT, self.tick)
        self.get_logger().warn("ARMED: will command the robot" if a.arm else "DRY RUN: nothing is published")

    def _joint(self, g, st, n):
        def cb(msg):
            if len(msg.joints) != n:
                self.fault = f"{g}: {len(msg.joints)} joints"
                return
            for i, j in enumerate(msg.joints):
                if j.name != self.x2.joint_names[st + i]:
                    self.fault = f"joint order: {j.name} != {self.x2.joint_names[st + i]}"
                    return
                self.q[st + i], self.qd[st + i] = j.position, j.velocity
            self.stamp[g] = time.monotonic()
        return cb

    def _imu(self, m):
        self.quat, self.gyro = np.array([m.orientation.w, m.orientation.x, m.orientation.y, m.orientation.z]), np.array([m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z])
        self.stamp["imu"] = time.monotonic()

    def _depth(self, m):
        if m.encoding not in ("16UC1", "mono16"):
            self.fault = f"depth encoding {m.encoding}"
            return
        self.depth = preprocess_depth(np.frombuffer(m.data, np.uint16).reshape(m.height, m.width), flip_180=not self.a.no_flip)
        self.stamp["depth"] = time.monotonic()

    def publish(self, target, kp, kd):
        if not self.a.arm:
            return
        for g, (st, n) in GROUPS.items():
            msg = JointCommandArray()
            msg.joints = []
            for i in range(st, st + n):
                from aimdk_msgs.msg import JointCommand
                j = JointCommand()
                j.name, j.position, j.velocity, j.effort, j.stiffness, j.damping = self.x2.joint_names[i], float(target[i]), 0.0, 0.0, float(kp[i]), float(kd[i])
                msg.joints.append(j)
            self.pubs[g].publish(msg)

    def tick(self):
        now = time.monotonic()
        needed = {"leg": 0.1, "waist": 0.1, "arm": 0.1, "head": 0.1, "imu": 0.1, "depth": 0.3}
        if self.fault is None and self.t0 is not None:
            for k, age in needed.items():
                if now - self.stamp.get(k, 0.0) > age:
                    self.fault = f"stale {k}"
        if self.fault:
            self.get_logger().error(f"FAULT {self.fault}: damped hold", throttle_duration_sec=1.0)
            self.publish(self.q.copy(), self.x2.kp * 0.0, self.x2.kd)
            return
        if not all(k in self.stamp for k in needed):
            self.get_logger().info("waiting for streams: " + ", ".join(k for k in needed if k not in self.stamp), throttle_duration_sec=2.0)
            return
        if self.t0 is None:
            self.t0, self.q_start = now, self.q.copy()
        target = self.core.step(self.q.copy(), self.qd.copy(), self.quat, self.gyro, self.depth, self.cmd)
        if not np.isfinite(target).all():
            self.fault = "non-finite action"
            return
        g = -(np.array([2 * (self.quat[1] * self.quat[3] - self.quat[0] * self.quat[2]), 2 * (self.quat[2] * self.quat[3] + self.quat[0] * self.quat[1]), 1 - 2 * (self.quat[1] ** 2 + self.quat[2] ** 2)]))
        if np.arccos(np.clip(-g[2], -1, 1)) > TILT_MAX:
            self.fault = "tilt"
            return
        s = min((now - self.t0) / RAMP_S, 1.0)
        out = self.q_start * (1 - s) + target * s
        self.publish(out, self.x2.kp, self.x2.kd)
        if int((now - self.t0) * 50) % 50 == 0:
            self.get_logger().info(f"t {now - self.t0:5.1f} ramp {s:.2f} action max {np.abs(self.core.a).max():.2f} depth mean {self.depth.mean():.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--arm", action="store_true", help="publish commands (without it: dry run)")
    ap.add_argument("--vx", type=float, default=0.0)
    ap.add_argument("--vy", type=float, default=0.0)
    ap.add_argument("--wz", type=float, default=0.0)
    ap.add_argument("--imu_topic", default="/aima/hal/imu/torso/state")
    ap.add_argument("--depth_topic", default="/aima/hal/sensor/rgbd_head_front/depth_image", help="check with ros2 topic list")
    ap.add_argument("--no_flip", action="store_true", help="the depth stream is already upright")
    a = ap.parse_args()
    rclpy.init()
    n = SSRNode(a)
    try:
        rclpy.spin(n)
    finally:
        n.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
