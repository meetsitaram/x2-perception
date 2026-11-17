"""Sim stand-in for the chest RealSense as ACP's student was trained on it: ray-casts the 80 x 45 z-depth (87 x 58 deg, on
torso_link at the bracket pose, the robot's own body + terrain + walls, 0 = no return; as eval/adapters/acp_depth.ChestCam)
from the bridge's state and publishes /x2_acp/chest_depth (sensor_msgs/Image 32FC1) at 25 Hz.

    python3 acp_depth_pub.py --scene DIR/scene.xml [--hz 25] [--cam-pitch 0 --cam-roll 0] [--latency-ms 0]

--cam-pitch / --cam-roll: where the simulated camera really points (deg off the trained pose); the deploy node gets its own
values (its calibration), so a mismatch between the two is a calibration error.
--truth-scans (diagnostic, sim only): also publish ACP's two scans from the scene (/x2_acp/true_scans, Float32MultiArray,
187 + 63 heights minus pelvis height, terrain + walls ray-cast down at the eval's grid points); the node uses them in place of
the student only with its own --truth-scans. Separates student error from everything else in the stack.
"""
from __future__ import annotations

import argparse
import collections
import math
import os
import sys
import time

import mujoco
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "hiking", "sim"))
from acp_deploy.core import CAM_DOWN, CAM_POS, _rot_x, _rot_y  # noqa: E402
from depth_pub import Mirror  # noqa: E402  (Hiking's: the bridge's pose + joints in a local MjData)

W, H, HFOV, VFOV = 80, 45, 87.0, 58.0


def heading_grid(nx, ny, x0, step=0.1):                 # as eval/adapters/common.heading_grid (x-major)
    gx, gy = np.meshgrid(x0 + step * np.arange(nx), step * (np.arange(ny) - (ny - 1) / 2), indexing="ij")
    return np.stack((gx.ravel(), gy.ravel()), 1)


GRIDS = np.concatenate((heading_grid(17, 11, -0.8), heading_grid(9, 7, -0.2)))


class TrueScans:
    def __init__(self, m):
        self.m = m
        self.gg = np.zeros(6, np.uint8)
        self.gg[[4, 5]] = 1
        n = len(GRIDS)
        self.gid, self.dist = np.zeros(n, np.int32), np.zeros(n)
        self.dirs = np.tile([0.0, 0.0, -1.0], n)

    def __call__(self, d):
        p = d.xpos[self.m.body("pelvis").id]
        R = d.xmat[self.m.body("pelvis").id].reshape(3, 3)
        h = math.atan2(R[1, 0], R[0, 0])
        c, s = math.cos(h), math.sin(h)
        xy = p[:2] + GRIDS @ np.array([[c, s], [-s, c]])
        top = 0.6 + p[2]                                  # under door lintels and overhangs (3 m hit the door header)
        pts = np.column_stack((xy, np.full(len(xy), top)))
        hs = np.zeros(len(xy))
        for i in range(len(xy)):                         # one point at a time: mj_multiRay takes one origin
            hs[i] = top - mujoco.mj_ray(self.m, d, pts[i], self.dirs[:3], self.gg, 1, -1, self.gid[i:i + 1])
        return np.clip(hs - p[2], -2.0, 2.0).astype(np.float32)


class ChestCam:
    def __init__(self, m, pitch=0.0, roll=0.0):
        self.m = m
        fx, fy = W / 2 / math.tan(math.radians(HFOV / 2)), H / 2 / math.tan(math.radians(VFOV / 2))
        u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
        r = np.stack([np.ones_like(u), -(u - W / 2) / fx, -(v - H / 2) / fy], -1).reshape(-1, 3)
        self.rays = r / np.linalg.norm(r, axis=1, keepdims=True)
        self.xcos = self.rays[:, 0]
        self.body = m.body("torso_link").id
        self.gg = np.zeros(6, np.uint8)
        self.gg[[1, 2, 4, 5]] = 1
        self.gid, self.dist = np.zeros(len(self.rays), np.int32), np.zeros(len(self.rays))
        self.R_off = _rot_y(CAM_DOWN + pitch) @ _rot_x(roll)

    def render(self, d):
        Rb = d.xmat[self.body].reshape(3, 3)
        p = d.xpos[self.body] + Rb @ CAM_POS
        dirs = np.ascontiguousarray(self.rays @ (Rb @ self.R_off).T)
        mujoco.mj_multiRay(self.m, d, p, dirs.ravel(), self.gg, 1, -1, self.gid, self.dist, None, len(dirs), 6.0)
        return np.where(self.gid >= 0, self.dist * self.xcos, 0.0).reshape(H, W)


class DepthPub(Node):
    def __init__(self, a):
        super().__init__("x2_acp_sim_depth")
        self.mir = Mirror(self, a.scene, a.pose_port)
        self.cam = ChestCam(self.mir.m, a.cam_pitch, a.cam_roll)
        self.delay, self.latency = collections.deque(), a.latency_ms / 1e3
        self.pub = self.create_publisher(Image, a.topic, qos_profile_sensor_data)
        self.truth = TrueScans(self.mir.m) if a.truth_scans else None
        if self.truth:
            self.tpub = self.create_publisher(Float32MultiArray, "/x2_acp/true_scans", qos_profile_sensor_data)
        self.create_timer(1.0 / a.hz, self.tick)
        self.n, self.t_cast, self.t_last = 0, 0.0, time.monotonic()
        self.get_logger().info("scene %s, %d x %d, %g Hz, camera %+.1f / %+.1f deg, latency %g ms"
                               % (a.scene, W, H, a.hz, a.cam_pitch, a.cam_roll, a.latency_ms))

    def tick(self):
        if not self.mir.update():
            return
        t0 = time.monotonic()
        z = self.cam.render(self.mir.d).astype(np.float32)
        if self.truth:
            self.tpub.publish(Float32MultiArray(data=self.truth(self.mir.d).tolist()))
        self.t_cast += time.monotonic() - t0
        self.delay.append((t0 + self.latency, z))
        while self.delay and self.delay[0][0] <= time.monotonic():
            _, z = self.delay.popleft()
            img = Image(height=H, width=W, encoding="32FC1", is_bigendian=0, step=4 * W, data=z.tobytes())
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
    ap.add_argument("--hz", type=float, default=25.0)
    ap.add_argument("--cam-pitch", type=float, default=0.0)
    ap.add_argument("--cam-roll", type=float, default=0.0)
    ap.add_argument("--latency-ms", type=float, default=0.0)
    ap.add_argument("--pose-port", type=int, default=5570)
    ap.add_argument("--topic", default="/x2_acp/chest_depth")
    ap.add_argument("--truth-scans", action="store_true", help="diagnostic: also publish the true scans (sim only)")
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
