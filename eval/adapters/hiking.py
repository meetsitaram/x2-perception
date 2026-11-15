"""Hiking in the Wild (X2 port) hand-over adapter (eval monitor, 2026-11-12). Folder: HIKING_HANDOVER_DIR, as the Hiking
track exports it (policy.onnx: obs (1, 10016) -> actions (1, 29); adapter_card.json; obs_check.npz). The observation is
rebuilt as the track's own MuJoCo port does it (models/hiking/x2_port/sim2sim/sim2sim_mujoco_x2.py, X2Sim.observe and
DepthCamera, read only), on the eval's model:

  proprio (800): per term 8 control steps [t-7 .. t], oldest first, term after term, filled with the first value at reset:
    base_ang_vel x 0.25 (pelvis link frame), projected gravity, command (vx, vy, wz; zeros in the reset step's
    observation), q - default (31, card joint_order), qd x 0.05 (31), previous raw action (29, action_joint_order).
  depth (8 x 36 x 32): chest camera on torso_link at the card's pos / quat, pinhole 64 x 36 from the card's apertures,
    ray cast against terrain, walls AND the robot's own visual meshes (geom group 1), hits nearer than min distance re-cast
    from it; z = distance x ray_x, clipped at the far limit; crop columns 16..48; gaussian blur 3 x 3 sigma 1 (reflect);
    clip 0..3 m, / 3. Frames every 5th control step from a 37-frame history, oldest first; the newest is one control
    step old (depth_lag 1, as recorded in Isaac on run 1).
Action: q_target = default + action_scale x action on the 29 action joints; head joints held at default. Plant from the
card (per-joint kp, kd, effort, armature of env.yaml; the card's plant label).

Command (HIKING_CMD, default "card"): the card's rule for waypoints, which is the training command generator with the
current waypoint as its target: vx = clip(2 x forward distance to the waypoint, 0, vmax) with vmax 0.7 m/s (training
0.6-0.8), vy = 0, yaw rate = clip(2 x bearing error, -1, 1); inside a station's disc the harness's command (stand, turn
to the station heading). Training zeroes the command within 0.4 m of its target; the eval's arrival disc is 0.20 m,
so this rule keeps going to the disc (HIKING_STOP_R, 0.20 m; final-station stop rule 2026-11-13). HIKING_CMD=standard: the harness's standard command.
"""
from __future__ import annotations

import json
import math
import os

import mujoco
import numpy as np

from .base import Adapter, Command, View

TERMS = ("base_ang_vel", "projected_gravity", "velocity_commands", "joint_pos", "joint_vel", "actions")


def _blur(img, k, sigma):
    half = (k - 1) * 0.5
    x = np.linspace(-half, half, k)
    ker = np.exp(-0.5 * (x / sigma) ** 2)
    ker /= ker.sum()
    p = k // 2
    a = np.pad(img, ((p, p), (p, p)), mode="reflect")
    a = sum(ker[i] * a[:, i:i + img.shape[1]] for i in range(k))
    a = sum(ker[i] * a[i:i + img.shape[0], :] for i in range(k))
    return a


def _quat_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


RING_VMIN = float(os.environ.get("HIKING_RING_VMIN", "0.5"))   # m/s: least forward command between 0.20 m and 0.4 m of a station
RING_T = float(os.environ.get("HIKING_RING_T", "6.0"))         # s in that ring without reaching the disc before a stand
BRAKE_D = float(os.environ.get("HIKING_BRAKE_D", "0.35"))    # m before a station's 0.20 m disc where "brake" zeroes the forward command


class HikingHandover(Adapter):
    name = "hiking"
    control_dt = 0.02
    vmax = 0.7
    wmax = 1.0

    def __init__(self, d=None):
        import onnxruntime as ort
        d = d or os.environ["HIKING_HANDOVER_DIR"]
        self.dir = d
        c = self.card = json.load(open(os.path.join(d, "adapter_card.json")))
        missing = [k for k in ("kp", "kd", "effort_limit", "armature", "default_pose", "action_scale") if not c.get(k)]
        if missing:
            raise ValueError("hand-over refused: adapter_card.json lacks %s" % ", ".join(missing))
        so = ort.SessionOptions()
        so.intra_op_num_threads = so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(os.path.join(d, "policy.onnx"), so, providers=["CPUExecutionProvider"])
        self.jo, self.ao = list(c["joint_order"]), list(c["action_joint_order"])
        self.terms = {t["name"]: t for t in c["observation"]["terms"]}
        self.H = int(self.terms["base_ang_vel"]["history_steps"])
        dc = c["depth"]
        self.dc = dc
        W, Hh = int(dc["raw_size"][0]), int(dc["raw_size"][1])
        fx, fy = W * dc["focal_length"] / dc["horizontal_aperture"], Hh * dc["focal_length"] / dc["vertical_aperture"]
        u, v = np.meshgrid(np.arange(W), np.arange(Hh), indexing="xy")
        dirs = np.stack([np.ones((Hh, W)), -(u + 0.5 - W / 2) / fx, -(v + 0.5 - Hh / 2) / fy], -1)
        dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
        t, b, l, r = dc["crop_region"]
        self.dirs = dirs[t:Hh - b, l:W - r]                      # (36, 32, 3) camera frame x fwd, y left, z up
        assert self.dirs.shape[:2] == tuple(dc["frame_shape"]), (self.dirs.shape, dc["frame_shape"])
        self.R_off, self.p_off = _quat_mat(dc["quat_wxyz"]), np.array(dc["pos_in_body"], float)
        self.lo, self.hi = [float(v) for v in dc["depth_range_m"]]
        self.min_dist = float(dc["min_distance_m"])
        self.n_stack, self.skip, self.hist_frames, self.lag = int(dc["stack_frames"]), int(dc["stack_skip"]), int(dc["history_frames"]), 1
        self.plant_name = c.get("plant") or "%s (per-joint env.yaml gains of run %s)" % (c["control"].get("gains_label", "card gains"), c.get("run"))
        self.cmd_mode = os.environ.get("HIKING_CMD", "card")
        self.trained_model = c.get("trained_model", "Isaac Lab, training URDF convex hulls, self-collision on (pelvis - hip_roll hull contact presumably ACTIVE)")
        self.vmax = float(os.environ.get("HIKING_VMAX", "0.7"))
        self.card_stop_r = float(c.get("command", {}).get("target_stop_radius_m", 0.4))   # training's stop radius (documentation)
        # Final-station stop (user, 2026-11-13, docs/EVAL.md): command toward a station until inside the eval's 0.20 m disc,
        # not the card's 0.4 m (which left 109 of 180 extended-map failures standing 0.20-0.45 m short). HIKING_STOP_R=0.4
        # gives the rule used before 2026-11-13 16:43 UTC.
        self.stop_r = float(os.environ.get("HIKING_STOP_R", "0.20"))
        self.self_view = os.environ.get("HIKING_SELF_DEPTH", "1") != "0"
        self.checkpoint = "Hiking %s model_%s | command %s" % (c.get("run"), c.get("iteration"),
                                                              {"card": "card rule (training generator to the waypoint; stand within %.2f m of a station or with it behind within 0.4 m; card 0.4 m), vmax %.2f" % (self.stop_r, self.vmax),
                                                               "card_v1": "card rule, implementation before 2026-11-12 10:15 (no stop radius), vmax %.2f" % self.vmax, "floor": "at least 0.45 m/s to the disc, zero in a station (eval variant), vmax %.2f" % self.vmax,
                                                               "brake": "BRAKE EARLY (eval variant): card rule, zero forward from %.2f m before a station's disc, a stand for a target behind within 0.5 m, vmax %.2f" % (BRAKE_D, self.vmax)}.get(self.cmd_mode, "harness standard"))
        self.observes = "proprioception history 8 steps (incl. the velocity command) + 8 chest depth frames 36 x 32 (terrain, walls, own body)"

    def plant(self, joint_names):
        c = self.card
        self.names31 = list(joint_names)
        self.i31 = np.array([self.names31.index(n) for n in self.jo])          # card joint_order -> model index
        self.i29 = np.array([self.names31.index(n) for n in self.ao])
        self.q0_31 = np.array([float(c["default_pose"][n]) for n in self.names31])
        self.q0_jo = self.q0_31[self.i31]
        self.scale = np.array([float(c["action_scale"][n]) for n in self.ao])
        f = lambda k: np.array([float(c[k][n]) for n in self.names31])
        return dict(kp=f("kp"), kd=f("kd"), effort=f("effort_limit"), armature=f("armature"), default_q=self.q0_31)

    def _ensure(self, view):
        if not hasattr(self, "i31"):
            self.plant(view.joint_names)
        if not hasattr(self, "_gg"):
            m = view.m
            self._cam_body = m.body(self.dc["body"]).id
            self._pelvis = m.body("pelvis").id
            self._gg = np.zeros(6, np.uint8)
            self._gg[4] = self._gg[5] = 1                          # eval terrain and walls
            if self.self_view:
                self._gg[1] = 1                                    # the robot's visual meshes (the training camera sees its own links)

    # ---- observation ------------------------------------------------------------------------------------------------
    def depth_frame(self, view: View):
        m, d = view.m, view.d
        R_b = d.xmat[self._cam_body].reshape(3, 3)
        p = d.xpos[self._cam_body] + R_b @ self.p_off
        R = R_b @ self.R_off
        dc = self.dirs.reshape(-1, 3)
        dw = np.ascontiguousarray(dc @ R.T)
        n = len(dc)
        gid, dist = np.zeros(n, np.int32), np.zeros(n)
        mujoco.mj_multiRay(m, d, p, dw.ravel(), self._gg, 1, -1, gid, dist, None, n, 1e6)
        close = (dist >= 0) & (dist < self.min_dist)
        if close.any():
            g1 = np.zeros(1, np.int32)
            for k in np.nonzero(close)[0]:
                r = mujoco.mj_ray(m, d, p + self.min_dist * dw[k], dw[k], self._gg, 1, -1, g1)
                dist[k] = r + self.min_dist if r >= 0 else -1
        self.robot_hits = int(((gid >= 0) & (m.geom_group[np.maximum(gid, 0)] == 1)).sum())
        dist[dist < 0] = np.inf
        z = dist * dc[:, 0]
        z[~np.isfinite(z)] = 1e3
        img = _blur(z.reshape(self.dirs.shape[:2]).astype(np.float32), int(self.dc["blur"]["kernel_size"]), float(self.dc["blur"]["sigma"]))
        return ((np.clip(img, self.lo, self.hi) - self.lo) / (self.hi - self.lo)).astype(np.float32)

    def proprio(self, view: View, cmd_vel):
        v6 = np.zeros(6)
        mujoco.mj_objectVelocity(view.m, view.d, mujoco.mjtObj.mjOBJ_XBODY, self._pelvis, v6, 1)
        R = view.d.xmat[self._pelvis].reshape(3, 3)
        q, qd = view.q()[self.i31], view.qd()[self.i31]
        raw = dict(base_ang_vel=v6[:3], projected_gravity=R.T @ np.array([0.0, 0.0, -1.0]), velocity_commands=np.asarray(cmd_vel, float),
                   joint_pos=q - self.q0_jo, joint_vel=qd, actions=self.a_prev)
        out = {}
        for k in TERMS:
            v = np.asarray(raw[k], float)
            s = self.terms[k].get("scale")
            out[k] = v * s if s is not None else v
        return out

    def observe(self, view: View, cmd_vel):
        parts = []
        for k, v in self.proprio(view, cmd_vel).items():
            self.hist[k] = [v] * self.H if k not in self.hist else self.hist[k][1:] + [v]
            parts.append(np.concatenate(self.hist[k]))
        f = self.depth_frame(view)
        need = self.hist_frames + self.lag
        self.dhist = [f] * need if not self.dhist else self.dhist[1:] + [f]
        L = len(self.dhist)
        idx = [max(L - 1 - self.lag - o, 0) for o in range((self.n_stack - 1) * self.skip, -1, -self.skip)]
        parts.append(np.stack([self.dhist[i] for i in idx], 0).ravel())
        return np.concatenate(parts).astype(np.float32)

    # ---- command ----------------------------------------------------------------------------------------------------
    def command_vel(self, view: View, cmd: Command):
        if self.cmd_mode == "brake":
            return self._brake(view, cmd)
        if self.cmd_mode == "card":
            return self._card(view, cmd)
        if cmd.hold and self.cmd_mode == "floor":
            return np.zeros(3) if abs(cmd.vel[2]) < 1e-9 else np.array([0.0, 0.0, float(np.clip(cmd.vel[2], -1.0, 1.0))])
        if self.cmd_mode not in ("card_v1", "floor") or cmd.hold or cmd.total == 0:
            return cmd.vel
        if self.cmd_mode == "floor":
            # HIKING_CMD=floor (coordinator, 2026-11-12): forward command at least 0.45 m/s until the arrival disc, zero inside a
            # station's disc (the trained range: 0.6-1.0 m/s walking, exact zeros standing); yaw as the card's rule
            p, q, _ = view.body_pose("pelvis")
            yaw = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
            vec = cmd.target_xy - p[:2]
            err = (math.atan2(vec[1], vec[0]) - yaw + math.pi) % (2 * math.pi) - math.pi
            return np.array([max(0.45, min(2.0 * float(np.linalg.norm(vec)), self.vmax)), 0.0, min(max(2.0 * err, -1.0), 1.0)])
        p, q, _ = view.body_pose("pelvis")
        yaw = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
        vec = cmd.target_xy - p[:2]
        xb = math.cos(yaw) * vec[0] + math.sin(yaw) * vec[1]
        err = (math.atan2(vec[1], vec[0]) - yaw + math.pi) % (2 * math.pi) - math.pi
        return np.array([min(max(2.0 * xb, 0.0), self.vmax), 0.0, min(max(2.0 * err, -1.0), 1.0)])

    def _card(self, view: View, cmd: Command):
        """HIKING_CMD=card, the STANDARD Hiking command from 2026-11-12 10:15 (coordinator, after the Hiking track pointed at its
        card): the card's training generator with the waypoint as its target. vx = clip(2 * forward distance, 0, vmax), vy = 0,
        yaw rate = clip(2 * bearing error, -1, 1) (a target behind gives vx 0 and a turn on the spot, then a walk); at a station,
        within HIKING_STOP_R (0.20 m since 2026-11-13; the card's target_stop_radius_m 0.4 m before) of it, or with it behind
        the robot within 0.4 m: a stand (vx = vy = 0), with yaw only toward the station heading when
        it has one (the turn tests need it; the card itself commands all zeros there). Non-station waypoints: no stop (the card's
        "for_waypoints" rule). The implementation used before 10:15 is HIKING_CMD=card_v1 (no 0.4 m stop; inside the disc the
        harness's yaw correction; outside it, pure-yaw corrections at up to 1 rad/s that made the robot circle)."""
        if cmd.total == 0:
            return cmd.vel
        p, q, _ = view.body_pose("pelvis")
        yaw = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
        vec = cmd.target_xy - p[:2]
        dist = float(np.linalg.norm(vec))
        xb = math.cos(yaw) * vec[0] + math.sin(yaw) * vec[1]
        # stand when inside the stop radius, or when an overshot station (target behind) is still within the card's 0.4 m:
        # no pure-yaw turn on the spot there (it made the robot circle in card_v1); the eval counts a stable stand within
        # 0.4 m of the final station as "stopped short"
        ring = cmd.station and self.stop_r <= dist < max(self.card_stop_r, self.stop_r)
        if cmd.index != getattr(self, "_ring_wp", None):          # time in the ring, per waypoint
            self._ring_wp, self.ring_t = cmd.index, 0.0
        if ring:
            self.ring_t += self.control_dt
        # in the ring between 0.20 m and the card's 0.4 m: forward at least RING_VMIN (the policy marks time in place under about
        # 0.4 m/s); after RING_T s in the ring without reaching the disc, a stand (the eval scores it "stopped short")
        give_up = ring and self.ring_t > RING_T
        if cmd.station and (dist < self.stop_r or give_up or (xb <= 0.0 and dist < max(self.card_stop_r, self.stop_r))):
            if cmd.target_heading is None:
                return np.zeros(3)
            e = (cmd.target_heading - yaw + math.pi) % (2 * math.pi) - math.pi
            return np.array([0.0, 0.0, min(max(2.0 * e, -1.0), 1.0)])
        err = (math.atan2(vec[1], vec[0]) - yaw + math.pi) % (2 * math.pi) - math.pi
        vx = min(max(2.0 * xb, 0.0), self.vmax)
        if ring and xb > 0.0:
            vx = max(vx, min(RING_VMIN, self.vmax) * max(0.0, math.cos(err)))
        return np.array([vx, 0.0, min(max(2.0 * err, -1.0), 1.0)])

    def _brake(self, view: View, cmd: Command):
        """HIKING_CMD=brake (coordinator 2026-11-12 10:01): the card rule, except at a station: zero forward command once the
        pelvis is within RADIUS + BRAKE_D of it (the policy runs on about 0.34 m after a zero command), a stand (all zero, or
        a pure turn toward the station heading once inside the disc) while holding, and a target that has ended up behind
        the robot within 0.5 m gets a stand instead of a pure-yaw correction while the robot still moves, then (stopped, outside
        the disc) a turn on the spot at up to 0.8 rad/s to face it. If it has stopped (under 0.1 m/s) outside the
        disc with the target ahead, it creeps on at 0.2 m/s."""
        p, q, _ = view.body_pose("pelvis")
        yaw = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
        if cmd.total == 0:
            return cmd.vel
        if cmd.hold:                                       # inside the station's disc: stand; turn on the spot only to face it
            return np.array([0.0, 0.0, float(np.clip(cmd.vel[2], -1.0, 1.0))])
        vec = cmd.target_xy - p[:2]
        dist = float(np.linalg.norm(vec))
        xb = math.cos(yaw) * vec[0] + math.sin(yaw) * vec[1]
        err = (math.atan2(vec[1], vec[0]) - yaw + math.pi) % (2 * math.pi) - math.pi
        wz = min(max(2.0 * err, -1.0), 1.0)
        if cmd.station:
            speed = float(np.hypot(view.d.qvel[0], view.d.qvel[1]))
            if xb < 0 and dist < 0.5:                      # overshot: stand while still moving (no circling); once stopped
                if speed > 0.1 or dist < 0.18:             # outside the disc, turn on the spot (at most 0.8 rad/s) to face it
                    return np.zeros(3)
                return np.array([0.0, 0.0, min(max(2.0 * err, -0.8), 0.8)])
            if dist < 0.20 + BRAKE_D:
                return np.array([0.2 if (speed < 0.1 and xb > 0.05 and abs(err) < 0.5) else 0.0, 0.0, wz if abs(err) < 0.5 else 0.0])
        return np.array([min(max(2.0 * xb, 0.0), self.vmax), 0.0, wz])

    # ---- control ----------------------------------------------------------------------------------------------------
    def reset(self, view: View, rng):
        self._ensure(view)
        self.a_prev = np.zeros(len(self.ao))
        self.hist, self.dhist = {}, []
        self.first = True

    def policy(self, obs):
        return self.sess.run(None, {"obs": obs[None]})[0][0].astype(float)

    def act(self, view: View, cmd: Command):
        self._ensure(view)
        v = np.zeros(3) if self.first else self.command_vel(view, cmd)
        self.last_command = v      # the reset step's observation has a zero command
        self.first = False
        obs = self.observe(view, v)
        a = self.policy(obs)
        self.a_prev = a
        q = self.q0_31.copy()
        q[self.i29] += self.scale * a
        return q
