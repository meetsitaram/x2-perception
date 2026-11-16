"""Sim stand-in for the chest RealSense: ray-casts the trained chest depth from the bridge's state and publishes
/x2_hiking/chest_depth (sensor_msgs/Image 32FC1, raw 64 x 36 z-depth in metres, inf = no return) at 50 Hz.

    python3 depth_pub.py --scene DIR/scene.xml --model MODEL_DIR [--hz 50] [--latency-ms 0]

Pose: the bridge's ground-truth pelvis qpos (ZMQ robot_pose, tcp://127.0.0.1:5570, JSON); joints: the bridge's joint states.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import threading
import time

import mujoco
import numpy as np
import rclpy
import zmq
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from aimdk_msgs.msg import JointStateArray

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from hiking_deploy import core  # noqa: E402
from hiking_deploy.sim_depth import ChestDepthCaster  # noqa: E402


class Mirror:
    """The bridge's robot state in a local MjData: ground-truth pelvis qpos (ZMQ robot_pose) + joint states (ROS)."""

    def __init__(self, node: Node, scene, pose_port=5570):
        self.m = mujoco.MjModel.from_xml_path(scene)
        self.d = mujoco.MjData(self.m)
        self.qadr = {self.m.joint(j).name: self.m.jnt_qposadr[j] for j in range(self.m.njnt)
                     if self.m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE}
        self.lock = threading.Lock()
        self.root, self.joints = None, {}
        self.sub = zmq.Context.instance().socket(zmq.SUB)
        self.sub.setsockopt(zmq.CONFLATE, 1)
        self.sub.setsockopt(zmq.SUBSCRIBE, b"robot_pose")
        self.sub.connect("tcp://127.0.0.1:%d" % pose_port)
        for g in ("leg", "waist", "arm", "head"):
            node.create_subscription(JointStateArray, "/aima/hal/joint/%s/state" % g, self.on_state, qos_profile_sensor_data)

    def on_state(self, m):
        with self.lock:
            for j in m.joints:
                self.joints[j.name] = j.position

    def update(self):
        """Copy the latest state into self.d (kinematics only). False until both pose and joints have arrived."""
        try:
            while True:
                self.root = json.loads(self.sub.recv(zmq.NOBLOCK).split(b" ", 1)[1])["pelvis_qpos_wxyz"]
        except zmq.Again:
            pass
        if self.root is None or not self.joints:
            return False
        self.d.qpos[:7] = self.root
        with self.lock:
            for n, v in self.joints.items():
                if n in self.qadr:
                    self.d.qpos[self.qadr[n]] = v
        mujoco.mj_kinematics(self.m, self.d)
        return True


class DepthPub(Node):
    def __init__(self, a):
        super().__init__("x2_hiking_sim_depth")
        self.mir = Mirror(self, a.scene, a.pose_port)
        self.m, self.d = self.mir.m, self.mir.d
        card = core.Card.load(a.model)
        self.caster = ChestDepthCaster(self.m, card.depth)
        self.H, self.W = int(card.depth["raw_size"][1]), int(card.depth["raw_size"][0])
        self.delay = collections.deque()
        self.latency = a.latency_ms / 1e3
        self.pub = self.create_publisher(Image, a.topic, qos_profile_sensor_data)
        self.create_timer(1.0 / a.hz, self.tick)
        self.n, self.t_cast, self.t_last = 0, 0.0, time.monotonic()
        self.get_logger().info("scene %s, %d x %d rays, %g Hz, latency %g ms" % (a.scene, self.W, self.H, a.hz, a.latency_ms))

    def tick(self):
        if not self.mir.update():
            return
        t0 = time.monotonic()
        z = self.caster.render(self.d).astype(np.float32)
        self.t_cast += time.monotonic() - t0
        self.delay.append((t0 + self.latency, z))
        while self.delay and self.delay[0][0] <= time.monotonic():
            _, z = self.delay.popleft()
            img = Image(height=self.H, width=self.W, encoding="32FC1", is_bigendian=0, step=4 * self.W, data=z.tobytes())
            img.header.stamp = self.get_clock().now().to_msg()
            img.header.frame_id = "chest_depth"
            self.pub.publish(img)
            self.n += 1
        if t0 - self.t_last > 5.0:
            self.get_logger().info("%.1f Hz, cast %.1f ms" % (self.n / (t0 - self.t_last), 1e3 * self.t_cast / max(self.n, 1)))
            self.n, self.t_cast, self.t_last = 0, 0.0, t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--latency-ms", type=float, default=0.0)
    ap.add_argument("--pose-port", type=int, default=5570)
    ap.add_argument("--topic", default="/x2_hiking/chest_depth")
    a = ap.parse_args()
    rclpy.init()
    n = DepthPub(a)
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
