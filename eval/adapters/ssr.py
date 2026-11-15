"""SSR hand-over adapter (eval monitor, 2026-11-12): one ONNX graph (proprioception history, chest depth, GRU state ->
21 actions, GRU state), as the SSR track exports it to tracks/ssr/eval/export/<run>_<iter>/ (policy.onnx,
adapter_card.json, obs_check.npz). Folder: SSR_HANDOVER_DIR.

Observation, from the card (term by term, checked against obs_check.npz by tools/ssr_check.py):
  hist (5, 72), oldest first, at reset all rows = the first frame; frame = pelvis angular velocity (body frame, the free
  joint's qvel[3:6]), projected gravity (-R[2, :]), command (vx, vy, wz in the heading frame), q - default_q (21, card
  order), qd (21), the previous raw actor output (21), clipped to +-40.
  depth (36, 36): pinhole on torso_link at (0.126, 0, 0.16), pitched down 1.0385 rad, rays as the card's
  camera_rays_cam_frame; ray length t to the terrain and walls (mj_ray) or to the limb spheres x2.OCCLUDERS (the robot's
  own legs and arms, as in training), whichever is nearer, march range 7.5 m; z = t * ray_x; (clip(z, 0.1, 3) - 0.1) / 2.9.
  gru (256): zeros at reset, gru_out fed back.
Action: q_target = clip(default_q + scale * clip(a, -8, 8), q_lo, q_hi) on the 21 commanded joints; the other 10 joints
held at default_q with the same gains. Plant from the card (kp, kd, effort, armature per joint; mc_20261024).

Command: the harness's standard velocity command from the route waypoints (harness.Trial.command): heading P control
1.5 x error capped at wmax, forward speed min(distance, vmax) x cos(error) with a 0.15 m/s floor, zero forward speed while
the error is over 60 degrees, zero speed and turn to the station heading inside a station's disc. SSR trained with no
waypoints (speed U(0, 1.2) along a drawn world heading, yaw rate 1.0 x heading error capped at 1.2): vmax 0.6 m/s,
wmax 1.0 rad/s, inside its training ranges.
"""
from __future__ import annotations

import json
import os

import numpy as np

from .base import Adapter, Command, View

# x2.OCCLUDERS of the SSR training code (tracks/ssr/code/ssr/x2.py, copied 2026-11-12): (body, offset in body frame, radius)
OCCLUDERS = [(f"{s}_{b}", p, r) for s in ("left", "right") for b, p, r in (
    ("hip_yaw_link", (0.0, 0.0, -0.07), 0.08), ("hip_yaw_link", (0.0, 0.0, -0.20), 0.075), ("knee_link", (0.0, 0.0, 0.0), 0.07), ("knee_link", (0.0, 0.0, -0.10), 0.055),
    ("knee_link", (0.0, 0.0, -0.20), 0.05), ("ankle_roll_link", (-0.03, 0.0, -0.035), 0.05), ("ankle_roll_link", (0.06, 0.0, -0.04), 0.05), ("ankle_roll_link", (0.13, 0.0, -0.05), 0.04),
    ("elbow_link", (0.0, 0.0, 0.0), 0.05), ("wrist_roll_link", (0.05, 0.0, 0.0), 0.05))]


def _vec(card_map, names):
    return np.array([float(card_map[n]) for n in names])


class SSRHandover(Adapter):
    name = "ssr"
    control_dt = 0.02
    vmax = 0.6
    wmax = 1.0

    def __init__(self, d=None):
        import onnxruntime as ort
        d = d or os.environ["SSR_HANDOVER_DIR"]
        self.dir = d
        c = self.card = json.load(open(os.path.join(d, "adapter_card.json")))
        missing = [k for k in ("plant", "kp", "kd") if not c.get(k)]
        if missing:
            raise ValueError("hand-over refused: adapter_card.json does not state %s" % ", ".join(missing))
        so = ort.SessionOptions()
        so.intra_op_num_threads = so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(os.path.join(d, c.get("policy_file", "policy.onnx")), so, providers=["CPUExecutionProvider"])
        self.j21 = list(c["joint_names"])
        self.plant_name = str(c["plant"])
        self.trained_model = c.get("trained_model", "MuJoCo Warp, X2 MJCF sha256 4f6a4f5b (pelvis - hip_roll hull contact ACTIVE)")
        dc = c["observation"]["depth"]
        self.cam_body, self.cam_pos, self.cam_pitch = dc["body"], np.array(dc["pos_in_body"], float), float(dc["pitch_down_rad"])
        self.near, self.far, self.march = float(dc["near"]), float(dc["far"]), float(dc["ray_march_range"])
        # training's ray march reaches at most tmin + iters * step_cap (3.88 m for chest_rs); a farther hit reads as "no hit"
        # (t = ray_march_range). Cards from 2026-11-13 15:40 UTC on carry it; older cards: exact cast (fix 2026-11-13).
        self.hit_range = float(dc["ray_hit_range"]) if "ray_hit_range" in dc else None
        z = np.load(os.path.join(d, "obs_check.npz"))
        self.rays_cam = z["camera_rays_cam_frame"].astype(float)          # (1296, 3), unit, camera frame x fwd y left z up
        self.rays_cam /= np.linalg.norm(self.rays_cam, axis=1, keepdims=True)
        self.n = int(dc["shape"][0])
        self.occlude = os.environ.get("SSR_OCCLUDERS", "1") != "0"
        self.cmd_mode = os.environ.get("SSR_CMD", "standard")
        if self.cmd_mode == "keep":
            self.vmax = float(os.environ.get("SSR_VMAX", "0.6"))
        self.checkpoint = "SSR %s, iteration %s | depth with limb spheres%s | command %s" % (
            os.path.basename(d.rstrip("/\\")), c.get("iteration"), "" if self.occlude else " OFF (eval variant)",
            {"keep": "constant %.2f m/s to the disc (eval variant)" % self.vmax,
             "floor40": "FLOOR40 (eval variant): harness command with forward >= 0.40 m/s outside the disc, turns >= 0.6 rad/s"}.get(self.cmd_mode, "harness standard"))
        self.observes = ("proprioception history 5 x 72 (incl. the velocity command) + chest depth 36 x 36 (terrain, walls and the "
                         "robot's own limb spheres) + GRU state")

    # ---- plant and joints -------------------------------------------------------------------------------------------
    def plant(self, joint_names):
        c = self.card
        self.names31 = list(joint_names)
        self.idx21 = np.array([self.names31.index(n) for n in self.j21])
        self.q0_31 = _vec(c["default_q"], self.names31)
        self.q0 = self.q0_31[self.idx21]
        self.scale = _vec(c["action_scale"], self.j21)
        self.q_lo, self.q_hi = _vec(c["action"]["q_lo"], self.j21), _vec(c["action"]["q_hi"], self.j21)
        return dict(kp=_vec(c["kp"], self.names31), kd=_vec(c["kd"], self.names31), effort=_vec(c["effort_limit"], self.names31),
                    armature=_vec(c["armature"], self.names31), default_q=self.q0_31)

    def _ensure(self, view):
        if not hasattr(self, "idx21"):
            self.plant(view.joint_names)
        if not hasattr(self, "_occ"):
            m = view.m
            self._occ = [(m.body(b).id, np.array(p, float), float(r)) for b, p, r in OCCLUDERS]
            self._cam_id = m.body(self.cam_body).id

    # ---- observation ------------------------------------------------------------------------------------------------
    def frame(self, view: View, cmd_vel):
        _, _, R = view.body_pose("pelvis")
        w = view.d.qvel[3:6]
        q, qd = view.q()[self.idx21], view.qd()[self.idx21]
        f = np.concatenate((w, -R[2, :], np.asarray(cmd_vel, float), q - self.q0, qd, self.a_prev))
        return np.clip(np.nan_to_num(f), -40.0, 40.0)

    def camera(self, view: View):
        d = view.d
        p, R = d.xpos[self._cam_id], d.xmat[self._cam_id].reshape(3, 3)
        c, s = np.cos(self.cam_pitch), np.sin(self.cam_pitch)
        Rc = R @ np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        return p + R @ self.cam_pos, self.rays_cam @ Rc.T

    def ray_lengths(self, view: View):
        o, dw = self.camera(view)
        t = view.rays(np.tile(o, (len(dw), 1)), dw, far=self.march)
        if self.hit_range is not None:
            t = np.where(t > self.hit_range, self.march, t)
        if self.occlude:
            d = view.d
            for b, off, r in self._occ:
                cen = d.xpos[b] + d.xmat[b].reshape(3, 3) @ off
                oc = cen - o
                bb = dw @ oc
                disc = bb * bb - oc @ oc + r * r
                hit = (disc > 0) & (bb > 0)
                tk = np.maximum(bb - np.sqrt(np.maximum(disc, 0.0)), 0.0)
                t = np.where(hit, np.minimum(t, tk), t)
        return t

    def depth(self, view: View):
        t = self.ray_lengths(view)
        z = t * self.rays_cam[:, 0]
        return ((np.clip(z, self.near, self.far) - self.near) / (self.far - self.near)).reshape(self.n, self.n)

    # ---- control ----------------------------------------------------------------------------------------------------
    def reset(self, view: View, rng):
        self._ensure(view)
        self.a_prev = np.zeros(21)
        self.gru = np.zeros((1, 256), np.float32)
        self.hist = None

    def step(self, hist, depth, gru):
        a, g = self.sess.run(None, {"hist": hist[None].astype(np.float32), "depth": depth[None].astype(np.float32), "gru": gru})
        return a[0].astype(float), g

    def command_vel(self, view: View, cmd: Command):
        """SSR_CMD=standard (default): the harness's command. SSR_CMD=keep (eval variant, 2026-11-12): the way SSR trained,
        a constant speed vmax towards the waypoint (rotated into the heading frame, vy capped at 0.4) with yaw rate
        clip(1.0 x heading error, 1.2), no slowing before the arrival disc; inside a station's disc the harness's command
        (stand, turn to face). Reason: in MuJoCo r6b 1600 does not walk at all from standing under 0.6 m/s and stops when the
        standard command slows to 0.36 m/s 0.36 m before a waypoint."""
        if self.cmd_mode == "floor40":
            return self._floor40(cmd)
        if self.cmd_mode != "keep" or cmd.hold or cmd.total == 0:
            return cmd.vel
        p, q, _ = view.body_pose("pelvis")
        yaw = np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
        vec = cmd.target_xy - p[:2]
        u = vec / max(np.linalg.norm(vec), 1e-9) * self.vmax
        c, s = np.cos(yaw), np.sin(yaw)
        vx, vy = c * u[0] + s * u[1], -s * u[0] + c * u[1]
        err = cmd.heading_err                      # the harness's: direction to the waypoint, or the kept heading (backward / sideways)
        return np.array([vx, float(np.clip(vy, -0.4, 0.4)), float(np.clip(1.0 * err, -1.2, 1.2))])

    def _floor40(self, cmd: Command):
        """SSR_CMD=floor40 (eval variant, coordinator 2026-11-12 ~14:30, asked by the SSR track): the harness's standard command with
        a speed floor. Outside the arrival disc, heading error under 60 deg: forward = max(min(distance, vmax), 0.40) x cos(error),
        yaw as the harness; inside a station's disc (hold): zero forward plus the harness's turn to the station heading; heading
        error over 60 deg (turn on the spot): yaw magnitude at least 0.6 rad/s. Waypoints that keep a heading (backward / sideways
        tests, vy != 0) get the harness's command unchanged. Reason: r7 2250 does not walk under about 0.3 m/s, and the standard
        command falls under that inside about 0.3 m of a waypoint, so it stops outside the 0.20 m disc."""
        if cmd.total == 0 or cmd.hold or abs(cmd.vel[1]) > 1e-9:
            return cmd.vel
        err = float(cmd.heading_err)
        if abs(err) > np.radians(60):
            return np.array([0.0, 0.0, float(np.sign(err) * max(abs(cmd.vel[2]), 0.6))])
        vx = max(min(float(cmd.dist), self.vmax), 0.40) * np.cos(err)
        return np.array([vx, 0.0, float(cmd.vel[2])])

    def act(self, view: View, cmd: Command):
        self._ensure(view)
        self.last_command = self.command_vel(view, cmd)
        f = self.frame(view, self.last_command)
        self.hist = np.tile(f, (5, 1)) if self.hist is None else np.vstack((self.hist[1:], f))
        a, self.gru = self.step(self.hist, self.depth(view), self.gru)
        self.a_prev = a
        q21 = np.clip(self.q0 + self.scale * np.clip(a, -8.0, 8.0), self.q_lo, self.q_hi)
        q = self.q0_31.copy()
        q[self.idx21] = q21
        return q
