"""Flat-floor depth: the chest image the trained camera would see if the robot stood on flat ground, made from the robot's own
state only (no camera). For a first real-robot walk before the RealSense is mounted (user, 2026-11-15: "test the model on the
real robot without depth camera ... fake flat terrain to make it walk straight first").

    python3 -m hiking_deploy.flat_depth --mjcf X2_FLAT_SCENE.xml --model MODEL_DIR [--hz 50]

Pose: torso IMU orientation -> pelvis orientation by the waist FK (core.pelvis_from_torso, as the policy's observation);
joints from the joint states; height such that the lowest sole touches z = 0. Ray-cast like the sim camera (the robot's own
toes / hands in view, a plane at z = 0) and published on /x2_hiking/chest_depth (32FC1, raw 64 x 36 metres), so the deploy
node is unchanged. Valid only on flat ground: anything real in front of the robot is invisible to the policy.
"""
from __future__ import annotations

import argparse
import threading
import time

import mujoco
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, Imu

from aimdk_msgs.msg import JointStateArray

from . import core
from .node import quat_xyzw_to_mat
from .sim_depth import ChestDepthCaster


class FlatDepth(Node):
    def __init__(self, a):
        super().__init__("x2_hiking_flat_depth")
        self.m = mujoco.MjModel.from_xml_path(a.mjcf)
        self.d = mujoco.MjData(self.m)
        card = core.Card.load(a.model)
        self.caster = ChestDepthCaster(self.m, card.depth)
        self.H, self.W = int(card.depth["raw_size"][1]), int(card.depth["raw_size"][0])
        self.qadr = {self.m.joint(j).name: self.m.jnt_qposadr[j] for j in range(self.m.njnt)
                     if self.m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE}
        self.soles = [g for g in range(self.m.ngeom) if self.m.body(self.m.geom_bodyid[g]).name.endswith("ankle_roll_link")
                      and self.m.geom_contype[g] and self.m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE]
        self.lock = threading.Lock()
        self.q, self.qd, self.imu = {}, {}, None
        for g in ("leg", "waist", "arm", "head"):
            self.create_subscription(JointStateArray, "/aima/hal/joint/%s/state" % g, self.on_state, qos_profile_sensor_data)
        self.create_subscription(Imu, "/aima/hal/imu/torso/state", self.on_imu, qos_profile_sensor_data)
        self.pub = self.create_publisher(Image, a.topic, qos_profile_sensor_data)
        self.create_timer(1.0 / a.hz, self.tick)
        self.n, self.t_cast, self.t_last = 0, 0.0, time.monotonic()
        self.get_logger().info("flat-floor depth from IMU + joints (%d sole spheres), %g Hz -> %s" % (len(self.soles), a.hz, a.topic))

    def on_state(self, m):
        with self.lock:
            for j in m.joints:
                self.q[j.name], self.qd[j.name] = j.position, j.velocity

    def on_imu(self, m):
        o, w = m.orientation, m.angular_velocity
        with self.lock:
            self.imu = (quat_xyzw_to_mat(o.x, o.y, o.z, o.w), np.array([w.x, w.y, w.z]))

    def tick(self):
        with self.lock:
            if self.imu is None or not all(n in self.q for n in core.WAIST):
                return
            q, qd, (R_t, w_t) = dict(self.q), dict(self.qd), self.imu
        t0 = time.monotonic()
        R_p, _ = core.pelvis_from_torso(R_t, w_t, [q[n] for n in core.WAIST], [qd[n] for n in core.WAIST])
        R_p = R_p.copy()
        yaw = np.arctan2(R_p[1, 0], R_p[0, 0])          # yaw does not matter on a plane; remove it
        c, s = np.cos(-yaw), np.sin(-yaw)
        R_p = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ R_p
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, R_p.ravel())
        d = self.d
        d.qpos[:3] = 0.0
        d.qpos[3:7] = quat
        for n, v in q.items():
            if n in self.qadr:
                d.qpos[self.qadr[n]] = v
        mujoco.mj_kinematics(self.m, d)
        low = min(d.geom_xpos[g][2] - self.m.geom_size[g][0] for g in self.soles)
        d.qpos[2] = -low                                  # the lowest sole on the floor
        mujoco.mj_kinematics(self.m, d)
        z = self.caster.render(d).astype(np.float32)
        img = Image(height=self.H, width=self.W, encoding="32FC1", is_bigendian=0, step=4 * self.W, data=z.tobytes())
        img.header.stamp = self.get_clock().now().to_msg()
        img.header.frame_id = "chest_depth_flat"
        self.pub.publish(img)
        self.n += 1
        self.t_cast += time.monotonic() - t0
        if t0 - self.t_last > 5.0:
            self.get_logger().info("%.1f Hz, %.1f ms per frame" % (self.n / (t0 - self.t_last), 1e3 * self.t_cast / max(self.n, 1)))
            self.n, self.t_cast, self.t_last = 0, 0.0, t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mjcf", required=True, help="the robot on a plane (e.g. deploy/hiking/sim/scenes/flat/scene.xml)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--topic", default="/x2_hiking/chest_depth")
    a = ap.parse_args()
    rclpy.init()
    n = FlatDepth(a)
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
