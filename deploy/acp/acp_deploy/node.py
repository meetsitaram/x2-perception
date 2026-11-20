"""ACP deploy node (rclpy): pad -> goal -> ACP (fine depth student + planner + tracker) -> /aima/hal/joint/*/command.
Same node in sim and on the robot; the state machine, safety and command writer are Hiking's (hiking_deploy/node.py).

    python3 -m acp_deploy.node --acp DIR --student student.pt [--cam-pitch 0 --cam-roll 0]

Depth: /x2_acp/chest_depth (sensor_msgs/Image 32FC1, 80 x 45 z-depth in metres, 0 / inf / nan = no return, 25 Hz); each new
frame goes through the student once, the scans are reused in between (as in the eval). Pad (SONIC layout, pad.py): ACP is
goal-driven, so the stick becomes the goal the eval gives on the way to a waypoint at bearing b: goal = (cos b, sin b) and
heading still to turn = b. Left stick up (vx over --v-walk) = walk: a unit goal (ACP sets its own pace: a goal shorter than 1
means "arrived" and it stops, seen in sim 2026-11-17); sideways = turn: b = wz / WZ_MAX x --bearing-max rad. Turn stick
released: b holds the heading it was released at (torso IMU yaw; without it ACP drifted 0.2 rad over 5 m). No walk: goal 0
and the same turn / heading hold (turn in place, or stand).
"""
from __future__ import annotations

import math
import os
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "hiking"))
from hiking_deploy import node as hnode  # noqa: E402
from hiking_deploy.pad import WZ_MAX  # noqa: E402

from . import core  # noqa: E402


class PadGoal:
    """(vx, vy, wz) from the pad + the pelvis yaw -> ACP goal (gx, gy, dpsi) in the heading frame."""

    def __init__(self, v_walk=0.05, bearing_max=1.0, turn_dead=0.05):
        self.v_walk, self.bmax, self.dead = v_walk, bearing_max, turn_dead
        self.hold = None

    def reset(self):
        self.hold = None

    def __call__(self, cmd, yaw):
        vx, _, wz = (float(c) for c in cmd)
        u = max(-1.0, min(1.0, wz / WZ_MAX))
        if abs(u) > self.dead or self.hold is None:
            self.hold = yaw
        if abs(u) > self.dead:
            b = u * self.bmax
        else:
            e = math.atan2(math.sin(self.hold - yaw), math.cos(self.hold - yaw))
            b = max(-self.bmax, min(self.bmax, e))
        n = 1.0 if vx > self.v_walk else 0.0
        return np.array([n * math.cos(b), n * math.sin(b), b])


class ACPNode(hnode.HikingNode):
    NODE = "x2_acp_deploy"

    def setup_policy(self, a):
        self.core = core.ACPCore(a.acp, a.student, threads=a.threads, accel=a.accel,
                                 planner_accel=a.planner_accel)
        c = self.core
        self.jo = c.joints
        self.q0 = dict(zip(c.joints, c.q0))
        self.kp, self.kd = dict(zip(c.joints, c.kp)), dict(zip(c.joints, c.kd))
        self.cam = (a.cam_pitch, a.cam_roll)
        self.pad_goal = PadGoal(a.v_walk, a.bearing_max)
        self.seq_used = 0
        self.goal_now = np.zeros(3)
        self.truth, self.true_scans = a.truth_scans, None
        self.goal_from_pad = a.goal_from_pad
        self.fine_shm, self.fine_t = a.fine_shm, 0.0
        # the student in its own thread (user, 2026-11-19): the control step only reads the newest scans, so a slow student
        # (about 20 ms on PC2's CPU) never delays the joint targets
        self.student_hz, self.idle_hz = a.student_hz, a.student_idle_hz
        self.student_reset = False
        self.sstats = dict(ms=[], age=[])
        threading.Thread(target=self.student_loop, daemon=True, name="acp-student").start()
        if a.truth_scans:
            from std_msgs.msg import Float32MultiArray
            self.create_subscription(Float32MultiArray, "/x2_acp/true_scans", self.on_true_scans, hnode.qos_profile_sensor_data)
        return "ACP %s + student %s: camera offset pitch %+.1f roll %+.1f deg, %d thread(s), planner on %s, tracker %s, student on %s " \
               "(own thread, %.0f Hz in the policy, %.0f Hz outside)" % (
            os.path.basename(os.path.abspath(a.acp)), os.path.basename(a.student), *self.cam, a.threads,
            c.providers["planner"], c.providers["tracker"], c.providers["student"], a.student_hz, a.student_idle_hz)

    def on_true_scans(self, m):
        v = np.asarray(m.data, float)
        self.true_scans = (v[:187], v[187:])

    def prep_depth(self, z):
        z = np.nan_to_num(np.asarray(z, np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if z.shape != (45, 80):
            raise ValueError("ACP chest depth must be 80 x 45, got %s" % (z.shape[::-1],))
        return z

    def reset_policy(self):
        self.core.reset(keep_scans=True)     # the policy's own state; the student thread restarts its memory at the next frame
        self.student_reset = True
        self.pad_goal.reset()

    def share_fine(self, now):
        """The newest fine map for the face screen (face_depth_capture.py): one ~9 KB file in shared memory, replaced in place
        (written aside, then renamed: the reader never sees half a file), at most 5 times a second. No ROS traffic."""
        if not self.fine_shm or self.core.fine is None or now - self.fine_t < 0.2:
            return
        self.fine_t = now
        tmp = self.fine_shm + ".tmp"
        with open(tmp, "wb") as f:
            np.save(f, np.asarray(self.core.fine, np.float32))
        os.replace(tmp, self.fine_shm)

    def policy_fields(self):
        n = len(self.core.joints)
        return ([f"a{i}" for i in range(n)] + ["goal_x", "goal_y", "goal_dpsi"] + [f"tg{i}" for i in range(27)]
                + [f"occ{i}" for i in range(9)] + [f"plan{i}" for i in range(187)] + [f"track{i}" for i in range(63)]
                + [f"prop{i}" for i in range(len(self.core.q0) * 3 + 6)] + ["scan_age_ms", "planner_ms", "tracker_ms"])

    def student_loop(self):
        """The student on the newest camera frame with the joint / IMU state of that moment: about --student-hz in the policy
        (25 Hz, the trained camera rate, from the D435's 30 Hz), --student-idle-hz outside it (the face screen only). Its
        memory restarts when the policy starts, as in the eval."""
        c, seq, next_t, last_log = self.core, 0, 0.0, time.monotonic()
        while not hasattr(self, "bb"):                   # the node's __init__ finishes after setup_policy starts this thread
            time.sleep(0.01)
        while True:
            time.sleep(0.003)
            in_policy = self.state in ("STAND", "WALK")
            with self.lock:
                depth, imu = self.depth, self.imu
                q, qd = dict(self.q), dict(self.qd)
            now = time.monotonic()
            if depth is None or imu is None or depth[2] == seq or now - depth[1] > hnode.STALE_DEPTH_S \
                    or any(n not in q for n in c.joints):
                continue
            period = 1.0 / (self.student_hz if in_policy else self.idle_hz)
            if now < next_t - 0.5 / 30.0:                # half a 30 Hz frame early is still this slot
                continue
            next_t = max(next_t + period, now) if now - next_t < period else now + period
            seq = depth[2]
            if self.student_reset:
                self.student_reset = False
                c.h = None
            if self.truth and self.true_scans is not None and in_policy:
                c.scans = self.true_scans                # diagnostic (sim): the scene's scans instead of the student's
                continue
            t0 = time.monotonic()
            try:
                c.perceive(imu[0], imu[1], q, qd, depth[0], *self.cam)
            except Exception as e:                       # noqa: BLE001 -- the policy keeps the last scans; logged
                self.get_logger().warning("student failed: %s" % e, throttle_duration_sec=5.0)
                continue
            t1 = time.monotonic()
            self.sstats["ms"].append(1e3 * (t1 - t0))
            self.sstats["age"].append(1e3 * (t1 - depth[1]))
            self.share_fine(t1)
            self.scan_t = depth[1]
            if t1 - last_log > 5.0:
                ms, age = np.array(self.sstats["ms"]), np.array(self.sstats["age"])
                self.get_logger().info("student %.1f Hz: run p50 %.1f p99 %.1f ms, map age at publish p50 %.0f p99 %.0f ms" % (
                    len(ms) / (t1 - last_log), np.percentile(ms, 50), np.percentile(ms, 99), np.percentile(age, 50),
                    np.percentile(age, 99)))
                self.sstats = dict(ms=[], age=[])
                last_log = t1

    def policy(self, imu, q, qd, cmd, depth, now):
        R_t = imu[0]
        goal = self.pad_goal(cmd, math.atan2(R_t[1, 0], R_t[0, 0]))     # torso yaw: the heading hold needs only its change
        g = self.pad.get("goal") if self.goal_from_pad else None
        if g is not None and self.pad.get("deadman"):                  # sim test: the eval's own goal (route_pad.py --eval-goal)
            goal = np.asarray(g, float)
        # the scans come from the student thread (student_loop); before its first map the core falls back to one inline run
        targets, info = self.core.step(imu[0], imu[1], q, qd, goal, depth=None, cam_pitch=self.cam[0], cam_roll=self.cam[1])
        self.goal_now = goal
        self.bb_policy = np.concatenate((info["action"], goal, info["targets"].ravel(), info["occ"], info["plan"], info["track"],
                                         info["prop"], [1e3 * (now - getattr(self, "scan_t", now)), info["planner_ms"],
                                                        info["tracker_ms"]]))
        self.stats["amax"] = max(self.stats["amax"], float(np.abs(info["action"]).max()))
        if self.debug and now - self.walk_t0 < 3.0:
            self.get_logger().info("dbg t %.2f goal %s |a| %.1f plan h [%.2f %.2f] targets %d" % (
                now - self.walk_t0, np.round(goal, 2), np.abs(info["action"]).max(), info["plan"].min(), info["plan"].max(),
                int((info["occ"] > 0.5).sum())))
        return dict(zip(self.jo, targets)), info["R_pelvis"]


def _args(ap):
    ap.add_argument("--acp", required=True, help="ACP hand-over folder (actor.npz, planner_attn.onnx)")
    ap.add_argument("--student", required=True, help="fine depth student, TorchScript (models/acp_depth)")
    ap.add_argument("--cam-pitch", type=float, default=0.0, help="mounted camera's pitch offset from the trained pose (deg, + = down)")
    ap.add_argument("--cam-roll", type=float, default=0.0)
    ap.add_argument("--v-walk", type=float, default=0.05, help="pad vx (m/s) over which ACP walks (at its own pace)")
    ap.add_argument("--bearing-max", type=float, default=1.0, help="goal bearing (rad) at full turn stick")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--goal-from-pad", action="store_true",
                    help="SIM TEST only: a pad packet's 'goal' (gx, gy, dpsi, as the eval computes it) replaces the stick mapping")
    ap.add_argument("--truth-scans", action="store_true",
                    help="DIAGNOSTIC, sim only: ACP on the scene's true scans (/x2_acp/true_scans) instead of the student")
    ap.add_argument("--accel", choices=("cpu", "cuda", "trt"), default="cpu",
                    help="planner and ONNX student on the CPU, CUDA or TensorRT (falls back to what onnxruntime offers; logged)")
    ap.add_argument("--planner-accel", choices=("cpu", "cuda", "trt"), default="cpu",
                    help="the planner's own choice (default cpu: sharing the GPU with the student thread slowed the policy to "
                         "19 Hz on PC2, 2026-11-19)")
    ap.add_argument("--student-hz", type=float, default=25.0, help="student rate in the policy (the trained camera rate)")
    ap.add_argument("--student-idle-hz", type=float, default=25.0,
                    help="student rate outside the policy (default the trained 25 Hz too: the screen shows what the policy would get)")
    ap.add_argument("--fine-shm", default="/dev/shm/x2p_fine.npy",
                    help="the newest fine map (52 x 44 float32 .npy, height minus root height) for the face screen; '' = off")
    ap.set_defaults(depth_topic="/x2_acp/chest_depth")


def main():
    hnode.main(ACPNode, _args)


if __name__ == "__main__":
    main()
