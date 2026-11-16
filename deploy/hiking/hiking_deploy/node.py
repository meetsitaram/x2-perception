"""Hiking deploy node (rclpy): pad command -> Hiking policy -> /aima/hal/joint/*/command. Same node in sim and on the robot.

    python3 -m hiking_deploy.node --model DIR [--limits-mjcf X2.xml] [--writer-hz 500]

Inputs: /aima/hal/joint/{leg,waist,arm,head}/state (aimdk_msgs/JointStateArray), /aima/hal/imu/torso/state (sensor_msgs/Imu,
torso frame), /x2_hiking/chest_depth (sensor_msgs/Image 32FC1, raw 64 x 36 z-depth in metres), pad over UDP JSON (pad.py).
States: WAIT (publishes nothing; the bridge's standby hold or MC has the robot) -Y-> STAND / WALK: the policy runs from the
measured pose, its targets blended in over 1 s (--blend-s). No PD-only ramp by default: Hiking's default pose does not stand
under PD alone and a 2.5 s ramp to it from the MC stand fell over in sim without support (--ramp-s brings it back). WALK while the deadman is held and the pad and the depth
are fresh (pad 0.25 s, depth 0.2 s): the pad's command; otherwise STAND: zero command. Pad controls as SONIC's (pad.py).
B event: DAMP (kp 0). E-stop, tilt over 60 deg, or joint/IMU data older than 0.1 s: SAFE (damp, latched; clear the e-stop,
then Y to ramp again).
"""
from __future__ import annotations

import argparse
import json
import math
import socket
import threading
import time
import xml.etree.ElementTree as ET

import numpy as np
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, Imu
from std_msgs.msg import String

from aimdk_msgs.msg import JointCommand, JointCommandArray, JointStateArray

from . import core

GROUPS = ("leg", "waist", "arm", "head")
PREFIX = "/aima/hal/joint/"
DT = 0.02
STALE_STATE_S, STALE_DEPTH_S, STALE_PAD_S = 0.1, 0.2, 0.25
TILT_GZ = -0.5                       # projected gravity z in the pelvis frame above this (tilt > 60 deg) -> SAFE
DAMP_KD = 3.0


def quat_xyzw_to_mat(x, y, z, w):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def mjcf_limits(path):
    """Joint ranges (rad) from an MJCF, by name; no MuJoCo needed (PC2)."""
    out = {}
    for j in ET.parse(path).getroot().iter("joint"):
        r = j.get("range")
        if j.get("name") and r:
            lo, hi = (float(v) for v in r.split())
            out[j.get("name")] = (lo, hi)
    return out


class HikingNode(Node):
    def __init__(self, a):
        super().__init__("x2_hiking_deploy")
        lim = mjcf_limits(a.limits_mjcf) if a.limits_mjcf else None
        self.ctl = core.HikingController(a.model, clip=a.clip, limits=lim)
        card = self.ctl.card
        self.jo = card.joint_order
        self.q0 = {n: card.default_pose[n] for n in self.jo}
        self.kp, self.kd = card.kp, card.kd
        self.lock = threading.Lock()
        self.names = {}                                  # group -> joint names in the robot's order (from its state)
        self.q, self.qd, self.t_state = {}, {}, {}
        self.imu = None                                  # (R, omega, t)
        self.depth = None                                # (frame 36 x 32, t)
        self.pad = dict(vx=0.0, vy=0.0, wz=0.0, deadman=False, estop=False, ev={}, t=0.0)
        self.ev_seen = None
        self.state, self.t_enter = "WAIT", time.monotonic()
        self.ramp_from, self.hold_targets, self.targets, self.gains = None, None, None, None
        self.walk_t0 = 0.0
        self.cmd_msgs = None                             # prebuilt per-group messages the writer sends
        self.stats = dict(n=0, dt=[], clipped=0, amax=0.0)
        self.debug = a.debug
        self.ramp_s, self.blend_s = a.ramp_s, a.blend_s
        self.sim_reset_port, self.reset_at = a.sim_reset_port, 0.0

        st = MutuallyExclusiveCallbackGroup()
        for g in GROUPS:
            self.create_subscription(JointStateArray, PREFIX + g + "/state", lambda m, g=g: self.on_state(g, m),
                                     qos_profile_sensor_data, callback_group=st)
        self.create_subscription(Imu, "/aima/hal/imu/torso/state", self.on_imu, qos_profile_sensor_data, callback_group=st)
        self.create_subscription(Image, a.depth_topic, self.on_depth, qos_profile_sensor_data, callback_group=st)
        self.pubs = {g: self.create_publisher(JointCommandArray, PREFIX + g + "/command", qos_profile_sensor_data) for g in GROUPS}
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", a.pad_port))
        self.sock.setblocking(False)
        self.create_timer(DT, self.step, callback_group=MutuallyExclusiveCallbackGroup())
        self.create_timer(1.0 / a.writer_hz, self.write, callback_group=MutuallyExclusiveCallbackGroup())
        self.create_timer(1.0, self.report, callback_group=st)
        self.status_pub = self.create_publisher(String, "/x2_hiking/status", 10)
        self.create_timer(0.05, self.publish_status, callback_group=st)   # 20 Hz: the viewer's camera follows the right stick
        self.get_logger().info("Hiking %s: clip %s, limits %s, writer %d Hz, pad udp :%d, depth %s"
                               % (card.raw.get("checkpoint", "?"), a.clip, "mjcf" if lim else "none", a.writer_hz, a.pad_port, a.depth_topic))

    # ---- inputs ----
    def on_state(self, g, m):
        now = time.monotonic()
        with self.lock:
            if g not in self.names:
                self.names[g] = [j.name for j in m.joints]
            for j in m.joints:
                self.q[j.name], self.qd[j.name] = j.position, j.velocity
            self.t_state[g] = now

    def on_imu(self, m):
        o, w = m.orientation, m.angular_velocity
        R = quat_xyzw_to_mat(o.x, o.y, o.z, o.w)
        with self.lock:
            self.imu = (R, np.array([w.x, w.y, w.z]), time.monotonic())

    def on_depth(self, m):
        z = np.frombuffer(m.data, np.float32).reshape(m.height, m.width)
        f = core.preprocess_depth(z, self.ctl.card.depth)
        with self.lock:
            self.depth = (f, time.monotonic())

    def read_pad(self):
        while True:
            try:
                data, addr = self.sock.recvfrom(4096)
            except BlockingIOError:
                return
            try:
                p = json.loads(data)
            except ValueError:
                continue
            p["t"] = time.monotonic()
            self.pad, self.pad_addr = p, addr

    # ---- control ----
    def enter(self, s, why=""):
        self.get_logger().info("%s -> %s %s" % (self.state, s, why))
        self.state, self.t_enter = s, time.monotonic()

    def start_policy(self, now, why=""):
        self.ctl.reset()
        self.walk_t0 = now
        self.enter("STAND", why)

    def events(self):
        ev = self.pad.get("ev", {})
        if self.ev_seen is None:
            self.ev_seen = dict(ev)
            return set()
        fired = {k for k, v in ev.items() if v > self.ev_seen.get(k, 0)}
        self.ev_seen = dict(ev)
        return fired

    def step(self):
        t0 = time.monotonic()
        self.read_pad()
        with self.lock:
            have = len(self.names) == len(GROUPS) and self.imu is not None
            q, qd = dict(self.q), dict(self.qd)
            imu, depth = self.imu, self.depth
            t_state = min(self.t_state.values()) if self.t_state else 0.0
        fired = self.events()
        if "reset" in fired and self.sim_reset_port:
            self.cmd_msgs = None                         # stop commanding first, then ask the bridge (bridge_run.py)
            self.reset_at = time.monotonic() + 0.3
            self.enter("WAIT", "(reset)")
        if self.reset_at and time.monotonic() >= self.reset_at:
            self.reset_at = 0.0
            self.sock.sendto(b"reset", ("127.0.0.1", self.sim_reset_port))
            self.ctl.reset()
        if not have:
            return
        now = time.monotonic()
        stale = now - min(t_state, imu[2]) > STALE_STATE_S
        if self.pad.get("estop") and self.state not in ("SAFE",):
            self.enter("SAFE", "(e-stop)")
        elif stale and self.state not in ("WAIT", "SAFE"):
            self.enter("SAFE", "(joint/IMU data %.0f ms old)" % (1e3 * (now - min(t_state, imu[2]))))
        if "b" in fired and self.state != "WAIT":
            self.enter("DAMP", "(B)")
        if "y" in fired:
            if self.state in ("WAIT", "DAMP") or (self.state == "SAFE" and not self.pad.get("estop")):
                self.ramp_from = {n: q[n] for n in self.jo}
                if self.ramp_s > 0:
                    self.enter("RAMP", "(Y)")
                else:
                    self.start_policy(now, "(Y: policy from the measured pose)")

        s, el = self.state, now - self.t_enter
        gains = (self.kp, self.kd)
        if s == "WAIT":
            return
        if s in ("DAMP", "SAFE"):
            targets, gains = {n: q[n] for n in self.jo}, ({n: 0.0 for n in self.jo}, {n: DAMP_KD for n in self.jo})
        elif s == "RAMP":
            a = min(el / self.ramp_s, 1.0)
            a = a * a * (3 - 2 * a)
            targets = {n: (1 - a) * self.ramp_from[n] + a * self.q0[n] for n in self.jo}
            if el >= self.ramp_s:
                self.ramp_from = dict(self.q0)
                self.start_policy(now)
        else:   # STAND / WALK: the policy
            p = self.pad
            ok_pad = p.get("deadman") and now - p.get("t", 0.0) < STALE_PAD_S
            ok_depth = depth is not None and now - depth[1] < STALE_DEPTH_S
            if (ok_pad and ok_depth) != (s == "WALK"):
                self.enter("WALK" if ok_pad and ok_depth else "STAND", "" if ok_depth else "(depth stale)")
            cmd = np.array([p.get("vx", 0.0), p.get("vy", 0.0), p.get("wz", 0.0)]) if ok_pad and ok_depth else np.zeros(3)
            frame = depth[0] if depth is not None else np.ones((36, 32), np.float32)
            pol, info = self.ctl.step(imu[0], imu[1], q, qd, cmd, frame)
            gp = core.projected_gravity(info["R_pelvis"])
            if self.debug and now - self.walk_t0 < 3.0:
                self.get_logger().info("dbg t %.2f g %s w %s cmd %s |a| %.1f a_wp %.1f q_wp %.3f" % (
                    now - self.walk_t0, np.round(gp, 3), np.round(info["omega_pelvis"], 2), np.round(cmd, 2),
                    np.abs(info["a_raw"]).max(), info["a_raw"][self.ctl.card.action_order.index("waist_pitch_joint")],
                    q["waist_pitch_joint"]))
            if gp[2] > TILT_GZ:
                self.enter("SAFE", "(tilt)")
            b = min((now - self.walk_t0) / self.blend_s, 1.0)
            targets = {n: (1 - b) * self.ramp_from[n] + b * pol[n] for n in self.jo}
            self.stats["clipped"] += int(self.ctl.amap.clipped.sum()) if self.ctl.amap.clipped is not None else 0
            self.stats["amax"] = max(self.stats["amax"], float(np.abs(info["a_raw"]).max()))
            self.cmd_now = cmd
        self.targets = targets
        msgs = {}
        for g in GROUPS:
            m = JointCommandArray()
            m.joints = [JointCommand(name=n, position=float(targets[n]), velocity=0.0, effort=0.0,
                                     stiffness=float(gains[0][n]), damping=float(gains[1][n])) for n in self.names[g]]
            msgs[g] = m
        self.cmd_msgs = msgs
        self.stats["n"] += 1
        self.stats["dt"].append(time.monotonic() - t0)

    def write(self):
        msgs = self.cmd_msgs
        if msgs is None or self.state == "WAIT":
            return
        stamp = self.get_clock().now().to_msg()
        for g, m in msgs.items():
            m.header.stamp = stamp
            self.pubs[g].publish(m)

    HINT = dict(WAIT="hold L1+L2+R1+R2 3 s, then Y, to stand (fallen? View/Back 1 s resets)", RAMP="ramping to the stand pose",
                STAND="hold L2 + left stick: up = walk, sideways = turn; right stick = camera; View 1 s = reset (sim)", WALK="release L2 to stop; L1 / R1 speed -/+",
                DAMP="L1+L2+R1+R2 3 s, then Y, to stand again; View/Back 1 s = reset (sim)",
                SAFE="safe hold: View/Back 1 s resets (sim); or clear the e-stop, then L1+L2+R1+R2 3 s + Y")

    def publish_status(self):
        p = self.pad
        cmd = getattr(self, "cmd_now", np.zeros(3)) if self.state in ("STAND", "WALK") else np.zeros(3)
        pad_ok = time.monotonic() - p.get("t", 0.0) < STALE_PAD_S
        line = ("%s|%s|cmd %+.2f %+.2f %+.2f|pad %s deadman %s%s" % (
            self.state, self.HINT[self.state], *cmd, "ok" if pad_ok else "LOST", "on" if p.get("deadman") else "off",
            " E-STOP" if p.get("estop") else "")) + "|" + (json.dumps(p["hud"]) if pad_ok and isinstance(p.get("hud"), dict)
                                                         else p.get("hud", "") if pad_ok else "pad: no packets")
        self.status_pub.publish(String(data=line))
        if getattr(self, "pad_addr", None):
            try:
                self.sock.sendto(line.encode(), self.pad_addr)
            except OSError:
                pass

    def report(self):
        dt = np.array(self.stats["dt"]) * 1e3
        d_age = (time.monotonic() - self.depth[1]) * 1e3 if self.depth else float("nan")
        cmd = getattr(self, "cmd_now", np.zeros(3)) if self.state in ("STAND", "WALK") else np.zeros(3)
        self.get_logger().info("%-4s cmd %+.2f %+.2f %+.2f  step p50 %.1f p99 %.1f ms  depth age %.0f ms  |a|max %.1f  clipped %d"
                               % (self.state, *cmd, np.percentile(dt, 50) if len(dt) else 0, np.percentile(dt, 99) if len(dt) else 0,
                                  d_age, self.stats["amax"], self.stats["clipped"]))
        self.stats.update(dt=[], amax=0.0, clipped=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--limits-mjcf", default=None)
    ap.add_argument("--clip", type=float, default=20.0)
    ap.add_argument("--writer-hz", type=float, default=500.0)
    ap.add_argument("--pad-port", type=int, default=5690)
    ap.add_argument("--depth-topic", default="/x2_hiking/chest_depth")
    ap.add_argument("--ramp-s", type=float, default=0.0,
                    help="PD ramp to the default pose before the policy (0 = none: the policy starts from the measured pose; "
                         "a PD-only ramp fell over in sim without support)")
    ap.add_argument("--blend-s", type=float, default=1.0, help="policy targets blended in from the start pose over this")
    ap.add_argument("--sim-reset-port", type=int, default=0,
                    help="sim only: pad reset -> WAIT, then 'reset' to the bridge wrapper on this UDP port (0 = off)")
    ap.add_argument("--debug", action="store_true", help="log every step of the first 3 s of WALK")
    a = ap.parse_args()
    rclpy.init()
    node = HikingNode(a)
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
