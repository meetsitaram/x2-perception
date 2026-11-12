"""The robot-side policy step, shared by the ROS 2 node and the offline sim test, so the sim test exercises the deploy code."""
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import onnxruntime as ort
import yaml

from ssr.env import DEPTH_MAX, DEPTH_MIN, DEPTH_N, H, PROP_DIM

ROOT = pathlib.Path(__file__).resolve().parents[1]
CROP = 720                     # square centre crop of the 1280 x 720 Gemini 335 stream, the sim camera's field of view


def preprocess_depth(mm: np.ndarray, flip_180: bool = True) -> np.ndarray:
    """uint16 millimetres (720, 1280) -> (36, 36) in [0, 1] as in training. The Orbbec's raw stream is upside-down on the
    X2 (x2-groot-vla/docs/CAMERAS.md). Holes (0) read as far, like a ray that hits nothing within range."""
    h, w = mm.shape
    y0, x0 = (h - CROP) // 2, (w - CROP) // 2
    d = mm[y0:y0 + CROP, x0:x0 + CROP].astype(np.float32) / 1000.0
    if flip_180:
        d = d[::-1, ::-1]
    bad = d <= 0.0
    d = np.clip(d, DEPTH_MIN, DEPTH_MAX)
    d[bad] = DEPTH_MAX
    k = CROP // DEPTH_N
    d = d[: k * DEPTH_N, : k * DEPTH_N].reshape(DEPTH_N, k, DEPTH_N, k).mean((1, 3))
    return ((d - DEPTH_MIN) / (DEPTH_MAX - DEPTH_MIN)).astype(np.float32)


def rot(axis, a):
    c, s = math.cos(a), math.sin(a)
    return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]), "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]), "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]


def quat_to_mat(q):  # wxyz
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def torso_to_pelvis(R_t, w_t, q_waist, qd_waist):
    """Training uses the pelvis; the robot's IMU is on torso_link. q_waist: yaw, pitch, roll (z, y, x, the joint chain's order).
    Returns R_pelvis, angular velocity in the pelvis frame."""
    y, p, r = q_waist
    R_w = rot("z", y) @ rot("y", p) @ rot("x", r)                   # torso in pelvis
    Rp = R_t @ R_w.T
    wy = np.array([0, 0, qd_waist[0]])
    wp = rot("z", y) @ np.array([0, qd_waist[1], 0])
    wr = rot("z", y) @ rot("y", p) @ np.array([qd_waist[2], 0, 0])
    return Rp, R_w @ w_t - (wy + wp + wr)


class Core:
    def __init__(self, onnx_path: str, x2):
        self.sess = ort.InferenceSession(onnx_path, providers=ort.get_available_providers())
        self.idx, self.q0_all, self.scale_all = x2.idx21, x2.default_q.copy(), x2.action_scale.copy()
        self.q_lo, self.q_hi = x2.q_lo, x2.q_hi
        self.q0, self.scale = self.q0_all[self.idx], self.scale_all[self.idx]
        self.names = x2.joint_names
        self.kp, self.kd = x2.kp, x2.kd
        self.waist = [x2.joint_names.index(n) for n in ("waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint")]
        self.reset()

    def reset(self):
        self.a = np.zeros(21, np.float32)
        self.hist = None
        self.gru = np.zeros((1, 256), np.float32)

    def step(self, q31, qd31, quat_wxyz, gyro, depth36, cmd):
        """q31, qd31 in MuJoCo joint order; quat_wxyz, gyro of torso_link (world from torso; body-frame rate). Returns 31 position targets."""
        R_t = quat_to_mat(np.asarray(quat_wxyz))
        Rp, w = torso_to_pelvis(R_t, np.asarray(gyro), q31[self.waist], qd31[self.waist])
        g = -Rp[2, :]
        q, qd = q31[self.idx], qd31[self.idx]
        o = np.concatenate((w, g, cmd, q - self.q0, qd, self.a)).astype(np.float32)
        self.hist = np.repeat(o[None], H, 0) if self.hist is None else np.concatenate((self.hist[1:], o[None]))
        r = self.sess.run(None, {"hist": self.hist[None], "depth": depth36[None], "gru": self.gru})
        self.a, self.gru = r[0][0], r[1]
        target = self.q0_all.copy()
        target[self.idx] = np.clip(self.q0 + self.scale * self.a, self.q_lo[self.idx], self.q_hi[self.idx])
        return target
