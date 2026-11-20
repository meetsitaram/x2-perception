"""Hiking in the Wild (X2, checkpoint 24,500) deploy core: pure numpy + onnxruntime, no ROS, no MuJoCo.

The same module runs in the docker sim and on PC2. Everything the policy needs is read from the export's
`adapter_card.json` at run time (joint orders, default pose, action scale, gains, observation terms, depth optics).
Reference implementation it is tested against: `eval/adapters/hiking.py` (HikingHandover.proprio / observe / act).

Inputs per 50 Hz control step:
  - torso IMU (orientation wxyz in the world frame, angular velocity in the torso frame), as the robot publishes it on
    /aima/hal/imu/torso/state; converted to the pelvis frame the policy was trained on with the waist joints
    (`pelvis_from_torso`),
  - joint position / velocity by name (31 joints),
  - the velocity command (vx, vy, wz),
  - the newest chest depth image: z-depth in metres, raw 64 x 36 (cols x rows) pinhole (`preprocess_depth`).
Output: 31 joint position targets by name (29 policy joints + the two head joints held at default).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

TERMS = ("base_ang_vel", "projected_gravity", "velocity_commands", "joint_pos", "joint_vel", "actions")
WAIST = ("waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint")   # pelvis -> torso_link chain, axes z, y, x


# ---- rotations ------------------------------------------------------------------------------------------------------
def quat_wxyz_to_mat(q):
    w, x, y, z = (float(v) for v in q)
    n = w * w + x * x + y * y + z * z
    s = 2.0 / n if n > 0 else 0.0
    return np.array([[1 - s * (y * y + z * z), s * (x * y - w * z), s * (x * z + w * y)],
                     [s * (x * y + w * z), 1 - s * (x * x + z * z), s * (y * z - w * x)],
                     [s * (x * z - w * y), s * (y * z + w * x), 1 - s * (x * x + y * y)]])


def _rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# Where the robot's /aima/hal/imu/torso/state really sits. False: on torso_link (above the waist joints), converted to the pelvis
# by the waist FK below. True: below the waist (on the pelvis side), used as the pelvis directly. The real X2 measured as True
# (2026-11-19, FINDINGS X13: the IMU+waist pelvis pitch differed from the feet-flat leg kinematics by -0.93 x the waist pitch;
# SONIC uses the raw IMU as its base). The deploy node sets this (--imu-on-pelvis); the sim bridge can emulate either.
IMU_ON_PELVIS = False


def pelvis_from_torso(R_world_torso, omega_torso, q_waist, qd_waist):
    """Torso IMU -> pelvis frame by waist FK (X2 MJCF: pelvis -> waist_yaw_link (z) -> waist_pitch_link (y) ->
    torso_link (x), identity body rotations, so R_pelvis_torso = Rz(yaw) Ry(pitch) Rx(roll)).

    R_world_torso: (3, 3) torso orientation; omega_torso: (3,) torso angular velocity in the torso frame;
    q_waist, qd_waist: (yaw, pitch, roll) positions and rates. Returns (R_world_pelvis, omega_pelvis in the pelvis frame).
    omega_torso (in pelvis) = omega_pelvis + qd_yaw z + qd_pitch Rz y + qd_roll Rz Ry x.
    """
    if IMU_ON_PELVIS:                                    # the IMU already gives the pelvis
        return np.asarray(R_world_torso, float), np.asarray(omega_torso, float)
    y, p, r = (float(v) for v in q_waist)
    Rz, Ry = _rz(y), _ry(p)
    R_pt = Rz @ Ry @ _rx(r)
    rel = (float(qd_waist[0]) * np.array([0.0, 0.0, 1.0]) + float(qd_waist[1]) * Rz[:, 1]
           + float(qd_waist[2]) * (Rz @ Ry)[:, 0])
    return R_world_torso @ R_pt.T, R_pt @ np.asarray(omega_torso, float) - rel


def projected_gravity(R_world_body):
    return R_world_body.T @ np.array([0.0, 0.0, -1.0])


# ---- card -----------------------------------------------------------------------------------------------------------
@dataclass
class Card:
    path: str
    raw: dict
    joint_order: list           # 31, the observation's joint order
    action_order: list          # 29, the policy output order
    held: list                  # head joints held at default
    default_pose: dict          # name -> rad
    action_scale: np.ndarray    # (29,) action order
    kp: dict
    kd: dict
    terms: dict
    history: int
    depth: dict
    obs_size: int

    @classmethod
    def load(cls, model_dir):
        p = os.path.join(model_dir, "adapter_card.json")
        c = json.load(open(p))
        missing = [k for k in ("kp", "kd", "default_pose", "action_scale", "joint_order", "action_joint_order") if not c.get(k)]
        if missing:
            raise ValueError("adapter_card.json lacks %s" % ", ".join(missing))
        terms = {t["name"]: t for t in c["observation"]["terms"]}
        ao = list(c["action_joint_order"])
        return cls(path=p, raw=c, joint_order=list(c["joint_order"]), action_order=ao, held=list(c.get("held_joints", [])),
                   default_pose={k: float(v) for k, v in c["default_pose"].items()},
                   action_scale=np.array([float(c["action_scale"][n]) for n in ao]),
                   kp={k: float(v) for k, v in c["kp"].items()}, kd={k: float(v) for k, v in c["kd"].items()},
                   terms=terms, history=int(terms["base_ang_vel"]["history_steps"]), depth=c["depth"],
                   obs_size=int(c["observation"]["total"]))


# ---- depth ----------------------------------------------------------------------------------------------------------
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


def camera_rays(depth_card):
    """(rows, cols, 3) unit ray directions of the RAW image in the camera frame (x forward, y left, z up), as trained."""
    W, H = int(depth_card["raw_size"][0]), int(depth_card["raw_size"][1])
    f = depth_card["focal_length"]
    fx, fy = W * f / depth_card["horizontal_aperture"], H * f / depth_card["vertical_aperture"]
    u, v = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
    d = np.stack([np.ones((H, W)), -(u + 0.5 - W / 2) / fx, -(v + 0.5 - H / 2) / fy], -1)
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def preprocess_depth(z_raw, depth_card):
    """Raw z-depth (rows 36 x cols 64, metres; inf / nan / <= 0 = no return) -> policy frame (36 x 32, 0..1):
    crop, gaussian blur (reflect), clip to the depth range, normalise. No return counts as far (1e3 m before the clip)."""
    z = np.array(z_raw, dtype=np.float64)
    H, W = int(depth_card["raw_size"][1]), int(depth_card["raw_size"][0])
    if z.shape != (H, W):
        raise ValueError("depth image %s, expected %s (rows, cols)" % (z.shape, (H, W)))
    z[~np.isfinite(z) | (z <= 0)] = 1e3
    t, b, l, r = depth_card["crop_region"]
    z = z[t:H - b, l:W - r]
    img = _blur(z.astype(np.float32), int(depth_card["blur"]["kernel_size"]), float(depth_card["blur"]["sigma"]))
    lo, hi = (float(v) for v in depth_card["depth_range_m"])
    return ((np.clip(img, lo, hi) - lo) / (hi - lo)).astype(np.float32)


# ---- observation ----------------------------------------------------------------------------------------------------
class ObsBuilder:
    """Proprio history (8 steps per term, oldest first, filled with the first value at reset) + depth stack (8 frames
    every 5th step from a 37-frame history, the newest one control step old)."""

    def __init__(self, card: Card, depth_lag=1):
        self.c = card
        dc = card.depth
        self.n_stack, self.skip, self.hist_frames = int(dc["stack_frames"]), int(dc["stack_skip"]), int(dc["history_frames"])
        self.lag = int(depth_lag)
        self.q0 = np.array([card.default_pose[n] for n in card.joint_order])
        self.reset()

    def reset(self):
        self.hist, self.dhist = {}, []
        self.a_prev = np.zeros(len(self.c.action_order))

    def step(self, omega_pelvis, grav_pelvis, cmd, q_jo, qd_jo, depth_frame):
        """q_jo, qd_jo: (31,) in card joint_order. depth_frame: (36, 32) preprocessed. Returns obs (obs_size,) float32."""
        raw = dict(base_ang_vel=omega_pelvis, projected_gravity=grav_pelvis, velocity_commands=cmd,
                   joint_pos=np.asarray(q_jo, float) - self.q0, joint_vel=qd_jo, actions=self.a_prev)
        parts = []
        for k in TERMS:
            v = np.asarray(raw[k], float)
            s = self.c.terms[k].get("scale")
            v = v * s if s is not None else v
            self.hist[k] = [v] * self.c.history if k not in self.hist else self.hist[k][1:] + [v]
            parts.append(np.concatenate(self.hist[k]))
        need = self.hist_frames + self.lag
        f = np.asarray(depth_frame, np.float32)
        self.dhist = [f] * need if not self.dhist else self.dhist[1:] + [f]
        L = len(self.dhist)
        idx = [max(L - 1 - self.lag - o, 0) for o in range((self.n_stack - 1) * self.skip, -1, -self.skip)]
        parts.append(np.stack([self.dhist[i] for i in idx], 0).ravel())
        obs = np.concatenate(parts).astype(np.float32)
        if obs.shape[0] != self.c.obs_size:
            raise RuntimeError("obs size %d != card %d" % (obs.shape[0], self.c.obs_size))
        return obs


# ---- actions --------------------------------------------------------------------------------------------------------
@dataclass
class ActionMap:
    """Raw policy action (29, action order) -> clipped action -> 31 targets by name, clamped to joint limits.

    clip: default |a| <= clip on every joint, per-joint overrides in clip_joint (user, 2026-11-15: +-20, waist_pitch +-50;
    checkpoint 24,500 trained without a clip). The clipped action is what is fed back as the last action, as the training
    wrapper's clip does (models/hiking/LEDGER.md row 35c). limits: name -> (lo, hi) rad, applied to targets."""
    card: Card
    clip: float | None = 20.0
    clip_joint: dict = field(default_factory=lambda: {"waist_pitch_joint": 50.0})
    limits: dict | None = None

    def __post_init__(self):
        ao = self.card.action_order
        self.q0_ao = np.array([self.card.default_pose[n] for n in ao])
        if self.clip is None:
            self.bound = None
        else:
            self.bound = np.array([float(self.clip_joint.get(n, self.clip)) for n in ao])
        self.lo = self.hi = None
        if self.limits:
            self.lo = np.array([self.limits[n][0] if n in self.limits else -np.inf for n in ao])
            self.hi = np.array([self.limits[n][1] if n in self.limits else np.inf for n in ao])
        self.clipped = np.zeros(len(ao), bool)

    def apply(self, a_raw):
        a = np.asarray(a_raw, float)
        if not np.all(np.isfinite(a)):
            raise FloatingPointError("non-finite policy action")
        if self.bound is not None:
            ac = np.clip(a, -self.bound, self.bound)
            self.clipped = ac != a
            a = ac
        q = self.q0_ao + self.card.action_scale * a
        if self.lo is not None:
            q = np.clip(q, self.lo, self.hi)
        targets = dict(zip(self.card.action_order, q))
        for n in self.card.held:
            targets[n] = self.card.default_pose[n]
        return a, targets


# ---- policy ---------------------------------------------------------------------------------------------------------
class Policy:
    def __init__(self, model_dir, threads=1, providers=("CPUExecutionProvider",)):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = so.inter_op_num_threads = int(threads)
        self.sess = ort.InferenceSession(os.path.join(model_dir, "policy.onnx"), so, providers=list(providers))
        self.inp = self.sess.get_inputs()[0].name

    def __call__(self, obs):
        return self.sess.run(None, {self.inp: np.asarray(obs, np.float32)[None]})[0][0].astype(np.float64)


class HikingController:
    """One control step = obs -> policy -> action map. The ROS node and the offline tests both drive this."""

    def __init__(self, model_dir, clip=20.0, clip_joint=None, limits=None, depth_lag=1, threads=1):
        self.card = Card.load(model_dir)
        self.obs = ObsBuilder(self.card, depth_lag=depth_lag)
        self.amap = ActionMap(self.card, clip=clip, clip_joint={"waist_pitch_joint": 50.0} if clip_joint is None else clip_joint,
                              limits=limits)
        self.policy = Policy(model_dir, threads=threads)
        jo = self.card.joint_order
        self.iw = [jo.index(n) for n in WAIST]

    def reset(self):
        self.obs.reset()

    def step(self, R_world_torso, omega_torso, q_by_name, qd_by_name, cmd, depth_frame):
        jo = self.card.joint_order
        q = np.array([q_by_name[n] for n in jo])
        qd = np.array([qd_by_name[n] for n in jo])
        R_p, w_p = pelvis_from_torso(R_world_torso, omega_torso, q[self.iw], qd[self.iw])
        obs = self.obs.step(w_p, projected_gravity(R_p), cmd, q, qd, depth_frame)
        a_raw = self.policy(obs)
        a, targets = self.amap.apply(a_raw)
        self.obs.a_prev = a
        return targets, dict(obs=obs, a_raw=a_raw, a=a, omega_pelvis=w_p, R_pelvis=R_p)
