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
# Pure damping (kp 0) fitted from the vendor MC's own damping mode (gwbc-x2 safety.cpp kVendorDampKd), by joint name.
_VD = (9.0, 8.0, 8.0, 4.0, 7.5, 3.0)
VENDOR_DAMP_KD = dict(zip(
    ["left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
     "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
     "waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint",
     "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_yaw_joint", "left_wrist_pitch_joint", "left_wrist_roll_joint",
     "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_yaw_joint", "right_wrist_pitch_joint", "right_wrist_roll_joint",
     "head_yaw_joint", "head_pitch_joint"],
    _VD + _VD + (5.0, 6.25, 4.0) + (4.0, 5.0, 3.0, 3.0, 3.0, 1.0, 1.0) * 2 + (0.0, 0.0)))
# Assisted stand-up (user, 2026-11-19: start damped like upstream GR00T WBC's G1, "a leaky assist - as i move the robot, it
# also tries to go to its pose until we reach standing pose"): target = the stand pose, stiffness rising with progress.
UPRIGHT_GZ, QUIET_W, QUIET_QD, QUIET_S = -0.95, 0.5, 1.0, 2.0     # gate: torso upright and still for 2 s (or Y)
ASSIST_MIN, ASSIST_RAMP_S, ASSIST_FULL_S = 0.05, 3.0, 4.0          # stiffness share: >= 5 %; never faster than 3 s to
                                                                  # full; full by 4 s even with no help (as SONIC RECOVER)
ASSIST_DONE_RAD, ASSIST_DONE_S = 0.08, 0.5                        # full stiffness, upright, and every joint within
                                                                  # 0.08 rad or all joints still, for 0.5 s


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
    """The state machine, safety and command writer; the policy behind it is the four hooks setup_policy, prep_depth,
    reset_policy and policy (deploy/acp/acp_deploy/node.py runs ACP through the same node)."""
    NODE = "x2_hiking_deploy"

    def __init__(self, a):
        super().__init__(self.NODE)
        label = self.setup_policy(a)                     # sets self.jo, self.q0, self.kp, self.kd (dicts by joint name)
        self.lock = threading.Lock()
        self.names = {}                                  # group -> joint names in the robot's order (from its state)
        self.q, self.qd, self.t_state = {}, {}, {}
        self.imu = None                                  # (R, omega, t)
        self.depth = None                                # (policy frame, t, sequence number)
        self.depth_seq = 0
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
        self.damped_start = a.start == "damp"
        self.quiet_t, self.done_t, self.assist_e0, self.assist_share = 0.0, 0.0, 1.0, 0.0
        self.band_released = False
        self.start_pending = False
        self.assist_on_start, self.assist_started = a.assist_on_start, False
        self.no_publish = a.no_publish
        if self.damped_start:
            self.state = "INIT"

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
        self.get_logger().info("%s: writer %d Hz, pad udp :%d, depth %s" % (label, a.writer_hz, a.pad_port, a.depth_topic))

    # ---- the policy (Hiking) ----
    def setup_policy(self, a):
        lim = mjcf_limits(a.limits_mjcf) if a.limits_mjcf else None
        self.ctl = core.HikingController(a.model, clip=a.clip, limits=lim)
        card = self.ctl.card
        self.jo = card.joint_order
        self.q0 = {n: card.default_pose[n] for n in self.jo}
        self.kp, self.kd = card.kp, card.kd
        self.blank_depth = np.ones((36, 32), np.float32)
        return "Hiking %s: clip %s, limits %s" % (card.raw.get("checkpoint", "?"), a.clip, "mjcf" if lim else "none")

    def prep_depth(self, z):
        return core.preprocess_depth(z, self.ctl.card.depth)

    def reset_policy(self):
        self.ctl.reset()

    def observe(self, imu, q, qd, depth, now):
        """Called every step outside STAND / WALK (DAMP, ASSIST, IDLE, SAFE, WAIT, also with --no-publish). Hiking: nothing."""

    def policy(self, imu, q, qd, cmd, depth, now):
        """-> (joint targets by name, pelvis rotation). depth: (frame, t, seq) or None."""
        frame = depth[0] if depth is not None else self.blank_depth
        pol, info = self.ctl.step(imu[0], imu[1], q, qd, cmd, frame)
        if self.debug and now - self.walk_t0 < 3.0:
            gp = core.projected_gravity(info["R_pelvis"])
            self.get_logger().info("dbg t %.2f g %s w %s cmd %s |a| %.1f a_wp %.1f q_wp %.3f" % (
                now - self.walk_t0, np.round(gp, 3), np.round(info["omega_pelvis"], 2), np.round(cmd, 2),
                np.abs(info["a_raw"]).max(), info["a_raw"][self.ctl.card.action_order.index("waist_pitch_joint")],
                q["waist_pitch_joint"]))
        self.stats["clipped"] += int(self.ctl.amap.clipped.sum()) if self.ctl.amap.clipped is not None else 0
        self.stats["amax"] = max(self.stats["amax"], float(np.abs(info["a_raw"]).max()))
        return pol, info["R_pelvis"]

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
        f = self.prep_depth(z)
        with self.lock:
            self.depth_seq += 1
            self.depth = (f, time.monotonic(), self.depth_seq)

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
        self.reset_policy()
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
            self.enter("INIT" if self.damped_start else "WAIT", "(reset)")
        if self.reset_at and time.monotonic() >= self.reset_at:
            self.reset_at = 0.0
            self.sock.sendto(b"reset", ("127.0.0.1", self.sim_reset_port))
            self.reset_policy()
        if not have:
            return
        now = time.monotonic()
        stale = now - min(t_state, imu[2]) > STALE_STATE_S
        if self.state == "INIT":                         # first data: limp (damped start), nothing stiff until the assist
            self.enter("DAMP", "(damped start)")
        if self.pad.get("estop") and self.state not in ("SAFE",):
            self.enter("SAFE", "(e-stop)")
        elif stale and self.state not in ("WAIT", "SAFE"):
            self.enter("SAFE", "(joint/IMU data %.0f ms old)" % (1e3 * (now - min(t_state, imu[2]))))
        if self.state == "SAFE" and not self.pad.get("estop") and not stale and self.damped_start:
            self.enter("DAMP", "(e-stop cleared: assist to stand again)")
        if "b" in fired and self.state != "WAIT":
            self.enter("DAMP", "(B)")
        # pelvis attitude from the torso IMU through the waist (in damp the limp waist lets the torso slump while whoever holds
        # the robot holds the hips): gravity z in the pelvis frame, -1 upright
        R_p, _ = core.pelvis_from_torso(imu[0], imu[1], np.array([q[n] for n in core.WAIST]), np.array([qd[n] for n in core.WAIST]))
        gz = float(core.projected_gravity(R_p)[2])
        if self.debug and self.state in ("DAMP", "ASSIST", "IDLE") and now - getattr(self, "_dbg_t", 0.0) > 0.5:
            self._dbg_t = now
            jq = max(self.jo, key=lambda n: abs(qd[n]))
            je = max(self.jo, key=lambda n: abs(self.q0[n] - q[n]))
            self.get_logger().info("dbg %s gz %+.3f |w| %.2f |qd|max %.2f (%s) err %.2f (%s) a %.2f" % (
                self.state, gz, float(np.abs(imu[1]).max()), abs(qd[jq]), jq, abs(self.q0[je] - q[je]), je, self.assist_share))
        # the start sequence (L1+L2+R1+R2 3 s, then Y) starts the assist; standing, the policy follows by itself (user,
        # 2026-11-19: "the sequence has to happen after we press the start sequence" - the operator's hands are on the robot)
        if self.state == "DAMP" and self.assist_on_start and not self.assist_started:
            self.assist_started = True                   # the robot ritual (SONIC's pad daemon) was the start sequence
            fired.add("y")
        if self.state == "DAMP" and "y" in fired:
            self.assist_e0 = max(max(abs(self.q0[n] - q[n]) for n in self.jo), 1e-3)
            self.done_t = 0.0
            self.enter("ASSIST", "(start sequence: leaky assist to the stand pose, then the policy)")
            fired.discard("y")
        if "y" in fired:
            if self.state == "IDLE" or (not self.damped_start and (self.state in ("WAIT", "DAMP")
                                                                   or (self.state == "SAFE" and not self.pad.get("estop")))):
                self.ramp_from = {n: q[n] for n in self.jo}
                if self.ramp_s > 0:
                    self.enter("RAMP", "(Y)")
                else:
                    self.start_policy(now, "(Y: policy from the measured pose)")

        s, el = self.state, now - self.t_enter
        gains = (self.kp, self.kd)
        if s not in ("STAND", "WALK"):                   # perception for the screen outside the policy (ACP); never stops control
            try:
                self.observe(imu, q, qd, depth, now)
            except Exception as e:                       # noqa: BLE001
                self.get_logger().warning("observe failed: %s" % e, throttle_duration_sec=5.0)
        if s in ("WAIT", "INIT"):
            return
        if s in ("DAMP", "SAFE"):
            kd = VENDOR_DAMP_KD if self.damped_start else {n: DAMP_KD for n in self.jo}
            targets, gains = {n: q[n] for n in self.jo}, ({n: 0.0 for n in self.jo}, {n: kd.get(n, DAMP_KD) for n in self.jo})
        elif s == "ASSIST":                              # leaky: aims at the stand pose, stiffer as it gets closer
            err = max(abs(self.q0[n] - q[n]) for n in self.jo)
            prog = 1.0 - min(err / self.assist_e0, 1.0)
            a = min(max(ASSIST_MIN + (1 - ASSIST_MIN) * prog, el / ASSIST_FULL_S), max(ASSIST_MIN, el / ASSIST_RAMP_S), 1.0)
            targets = dict(self.q0)
            gains = ({n: a * self.kp[n] for n in self.jo},
                     {n: (1 - a) * VENDOR_DAMP_KD.get(n, DAMP_KD) + a * self.kd[n] for n in self.jo})
            if gz > TILT_GZ:
                self.enter("DAMP", "(tilt during the assist)")
            elif a > 0.99 and gz < UPRIGHT_GZ and (err < ASSIST_DONE_RAD or max(abs(qd[n]) for n in self.jo) < 0.05):
                # at the pose, or as close as stiffness alone gets it (gravity sag, e.g. the waist) and settled
                self.done_t = self.done_t or now
                if now - self.done_t >= ASSIST_DONE_S:
                    self.ramp_from = {n: q[n] for n in self.jo}
                    self.start_policy(now, "(standing: the policy takes over)")
            else:
                self.done_t = 0.0
            self.assist_share = a
        elif s == "IDLE":                                # holding the stand pose at full gains
            targets = dict(self.q0)
            if gz > TILT_GZ:
                self.enter("DAMP", "(tilt while standing)")
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
            pol, R_pelvis = self.policy(imu, q, qd, cmd, depth, now)
            if core.projected_gravity(R_pelvis)[2] > TILT_GZ:
                self.enter("SAFE", "(tilt)")
            b = min((now - self.walk_t0) / self.blend_s, 1.0)
            targets = {n: (1 - b) * self.ramp_from[n] + b * pol[n] for n in self.jo}
            self.cmd_now = cmd
            if self.sim_reset_port and self.damped_start and not self.band_released and now - self.walk_t0 > 3.0:
                self.band_released = True                # sim gantry: lower the robot once the policy has held it 3 s
                self.sock.sendto(b"release", ("127.0.0.1", self.sim_reset_port))
                self.get_logger().info("sim: asked the bridge to release the gantry band")
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
        if msgs is None or self.state == "WAIT" or self.no_publish:
            return
        stamp = self.get_clock().now().to_msg()
        for g, m in msgs.items():
            m.header.stamp = stamp
            self.pubs[g].publish(m)

    HINT = dict(WAIT="hold L1+L2+R1+R2 3 s, then Y, to stand (fallen? View/Back 1 s resets)", RAMP="ramping to the stand pose",
                STAND="hold L2 + left stick: up = walk, sideways = turn; right stick = camera; View 1 s = reset (sim)", WALK="release L2 to stop; L1 / R1 speed -/+",
                DAMP="limp: L1+L2+R1+R2 3 s, then Y: assist to standing, then the policy; View/Back 1 s = reset (sim)",
                INIT="waiting for joint and IMU data",
                ASSIST="assisting to the stand pose: guide the robot up; it stiffens as it nears standing, then the policy starts",
                IDLE="standing: L1+L2+R1+R2 3 s, then Y, starts the policy",
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


def main(node_cls=None, add_args=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=node_cls is None)
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
    ap.add_argument("--assist-on-start", action="store_true",
                    help="robot launch: the start ritual already happened (SONIC's pad daemon ran our launch script), so go "
                         "from DAMP straight into the assist once the data is fresh")
    ap.add_argument("--no-publish", action="store_true",
                    help="observe only: run the whole pipeline (IMU, joints, depth, pad, policy, states) but never send a "
                         "joint command (MC must keep the robot; never combine with stopping MC)")
    ap.add_argument("--start", choices=("damp", "wait"), default="damp",
                    help="damp (default, the real-robot start): limp from the first data, then a leaky assist to the stand pose "
                         "(upright and still for 2 s, or Y), IDLE, Y = policy; e-stop -> limp, cleared -> assist again. "
                         "wait: the old sim start (nothing published until Y, policy from the measured pose)")
    if add_args:
        add_args(ap)
    a = ap.parse_args()
    rclpy.init()
    node = (node_cls or HikingNode)(a)
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
