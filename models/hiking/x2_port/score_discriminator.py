"""Offline: what style reward does a checkpoint's discriminator give to walking, to standing, and to a perfect stand?

  CUDA_VISIBLE_DEVICES= python score_discriminator.py <checkpoint.pt> <motion.npz> <x2|g1> [more motion.npz ...]

Builds the discriminator's input the way the task does (group amp_policy / amp_reference: 10 frames at 50 Hz, oldest first, of
projected gravity, joint position minus default, joint velocity x 0.05, base linear velocity, base angular velocity, all in the
root frame, joints in Isaac order) from a motion file, and applies reward = clamp(1 - 0.25 (D - 1)^2, 0) (the "quad" type; the
policy gets 0.25 x this). Reference velocities here are finite differences of the file (the task interpolates and differences the
same file); a synthetic stand is the default pose with zero velocities and gravity straight down.
CPU only. Approximations: no interpolation, clip seams skipped by a speed test, windows only from the file's own frames.
"""
import sys

import numpy as np
import torch

CKPT, ROBOT = sys.argv[1], sys.argv[3]
FILES = [sys.argv[2]] + sys.argv[4:]
X2_JOINTS = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint", "left_hip_roll_joint", "right_hip_roll_joint",
    "waist_pitch_joint", "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_roll_joint", "left_knee_joint", "right_knee_joint",
    "head_yaw_joint", "left_shoulder_pitch_joint", "right_shoulder_pitch_joint", "left_ankle_pitch_joint",
    "right_ankle_pitch_joint", "head_pitch_joint", "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint", "left_shoulder_yaw_joint", "right_shoulder_yaw_joint", "left_elbow_joint",
    "right_elbow_joint", "left_wrist_yaw_joint", "right_wrist_yaw_joint", "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
]
G1_JOINTS = [
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint", "waist_pitch_joint", "left_shoulder_roll_joint",
    "right_shoulder_roll_joint", "waist_roll_joint", "left_shoulder_yaw_joint", "right_shoulder_yaw_joint", "waist_yaw_joint",
    "left_elbow_joint", "right_elbow_joint", "left_hip_pitch_joint", "right_hip_pitch_joint", "left_wrist_roll_joint",
    "right_wrist_roll_joint", "left_hip_roll_joint", "right_hip_roll_joint", "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "left_wrist_yaw_joint", "right_wrist_yaw_joint", "left_knee_joint",
    "right_knee_joint", "left_ankle_pitch_joint", "right_ankle_pitch_joint", "left_ankle_roll_joint", "right_ankle_roll_joint",
]
JOINTS = X2_JOINTS if ROBOT == "x2" else G1_JOINTS
ELBOW = -0.6 if ROBOT == "x2" else 0.6


def default(j):
    if j.endswith("hip_pitch_joint"): return -0.312
    if j.endswith("knee_joint"): return 0.669
    if j.endswith("ankle_pitch_joint"): return -0.363
    if j.endswith("elbow_joint"): return ELBOW
    if j == "left_shoulder_roll_joint": return 0.2
    if j == "right_shoulder_roll_joint": return -0.2
    if j.endswith("shoulder_pitch_joint"): return 0.2
    return 0.0


Q0 = np.array([default(j) for j in JOINTS])
NJ, H = len(JOINTS), 10


def rot(q):  # wxyz -> (T,3,3)
    w, x, y, z = q.T
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], 1)


def frames(path):
    d = np.load(path, allow_pickle=True)
    fps = float(d["framerate"])
    assert abs(fps - 50) < 1e-6
    names = [str(n) for n in d["joint_names"]]
    q = d["joint_pos"][:, [names.index(j) for j in JOINTS]].astype(np.float64)
    p, quat = d["base_pos_w"].astype(np.float64), d["base_quat_w"].astype(np.float64)
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    R = rot(quat)
    qd = np.diff(q, axis=0) * fps
    vw = np.diff(p, axis=0) * fps
    dR = (R[1:] - R[:-1]) * fps
    W = np.einsum("tji,tjk->tik", R[:-1], dR)                 # R^T dR = [w_b]x
    wb = np.stack([W[:, 2, 1], W[:, 0, 2], W[:, 1, 0]], -1)
    vb = np.einsum("tji,tj->ti", R[:-1], vw)
    g = -R[:-1, 2, :]                                          # R^T (0,0,-1)
    speed = np.linalg.norm(vw[:, :2], axis=1)
    return dict(g=g, q=q[:-1] - Q0, qd=qd * 0.05, v=vb, w=wb, speed=speed, jspeed=np.abs(qd).max(1))


def windows(f, idx):
    out = []
    for t in idx:
        sl = slice(t - H + 1, t + 1)
        out.append(np.concatenate([f[k][sl].reshape(-1) for k in ("g", "q", "qd", "v", "w")]))
    return np.array(out, np.float32)


SIZES = dict(g=3 * H, q=NJ * H, qd=NJ * H, v=3 * H, w=3 * H)
OFF = {}
o = 0
for k, n in SIZES.items():
    OFF[k] = slice(o, o + n); o += n
sd = torch.load(CKPT, map_location="cpu", weights_only=False)["discriminator"]
ws = [(k, v) for k, v in sd.items() if v.ndim == 2]
bs = [v for k, v in sd.items() if v.ndim == 1]
assert ws[0][1].shape[1] == o, ("discriminator input size", ws[0][1].shape, "built", o, list(sd.keys()))
print("discriminator layers", [tuple(w.shape) for _, w in ws], "keys", [k for k in sd if "normal" in k.lower()] or "no normalizer")


def D(x):
    x = torch.from_numpy(x)
    for i, ((_, w), b) in enumerate(zip(ws, bs)):
        x = x @ w.T + b
        if i < len(ws) - 1:
            x = torch.relu(x)
    return x.squeeze(-1).numpy()


def reward(x):
    return np.clip(1 - 0.25 * (D(x) - 1) ** 2, 0, None)


def report(label, x):
    d, r = D(x), reward(x)
    print("  %-52s n %5d | D mean %+.2f (10-90%%: %+.2f..%+.2f) | style reward mean %.3f" % (label, len(x), d.mean(), np.percentile(d, 10), np.percentile(d, 90), r.mean()))
    return r.mean()


stand = np.zeros((1, o), np.float32)
stand[0, OFF["g"]] = np.tile([0, 0, -1.0], H)
rng = np.random.default_rng(0)
print("checkpoint", CKPT)
for path in FILES:
    f = frames(path)
    T = len(f["speed"])
    ok = np.array([t for t in range(H, T) if f["speed"][t - H + 1:t + 1].max() < 3.0])       # no clip seam inside the window
    ms = np.array([f["speed"][t - H + 1:t + 1].mean() for t in ok])
    mj = np.array([f["jspeed"][t - H + 1:t + 1].max() for t in ok])
    walk, still = ok[ms > 0.3], ok[(ms < 0.05) & (mj < 0.5)]
    print(path.split("/")[-1].split("\\")[-1], ": %d frames, walking windows %d, standing windows %d" % (T, len(walk), len(still)))
    xw = windows(f, rng.choice(walk, min(4000, len(walk)), replace=False))
    report("reference, walking (mean speed > 0.3 m/s)", xw)
    if len(still):
        report("reference, standing stretches", windows(f, still[:4000]))
    report("synthetic perfect stand (default pose, zero velocity)", stand)
    lean = stand.copy(); a = np.radians(9.0); lean[0, OFF["g"]] = np.tile([-np.sin(a), 0, -np.cos(a)], H)   # leaning back 9 deg
    report("synthetic stand leaning back 9 deg", lean)
    # which inputs carry the difference: walking windows with one group replaced by the stand's values, and the reverse
    for k in SIZES:
        x = xw.copy(); x[:, OFF[k]] = stand[0, OFF[k]]
        y = np.repeat(stand, len(xw), 0); y[:, OFF[k]] = xw[:, OFF[k]]
        print("    group %-3s: walking with this group set to the stand's -> reward %.3f | stand with this group taken from walking -> %.3f" % (k, reward(x).mean(), reward(y).mean()))
