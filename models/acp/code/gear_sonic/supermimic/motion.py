"""Reference-motion library for SuperMimic X2: SONIC motion pkls resampled to the 50 Hz control rate.

Input: the same joblib pkls the SONIC trainer loads (clip name -> dict with ``root_trans_offset``
(T,3) world, ``root_rot`` (T,4) xyzw, ``dof`` (T,31) MuJoCo order, ``fps``). Every clip is
resampled once to 50 Hz (linear for positions/joints, slerp for the root rotation) and all clips
are concatenated into flat GPU tensors indexed by a global frame id ``f``.

Stored per frame (~75 floats, ~14 GB fp32 for the full 264 h corpus): root pose and velocities,
joint positions and velocities. Body poses/velocities of the tracked bodies are computed on demand
with :class:`X2Kinematics` (cheap, and storing them for every body would not fit for the full set).

Velocities (paper-silent): central differences at 50 Hz for the reference state (reset and reward).
The goal's joint velocities use the backward difference, as the deploy tokenizer builds them
(``eval_x2_mujoco.build_tokenizer_obs``).
"""

from __future__ import annotations

import math
import pathlib

import joblib
import numpy as np
import torch

from gear_sonic.supermimic.assets import CONTROL_DT
from gear_sonic.supermimic.kinematics import X2Kinematics, mm3, mtv3, quat_wxyz_to_mat
from gear_sonic.supermimic.robot_cfg import CFG

BIN_FRAMES = int(round(1.0 / CONTROL_DT))  # paper App. D.1: 1 s bins
K_FUTURE = 10  # paper App. A: K = 10 future reference frames
FUTURE_STRIDE = int(round(0.1 / CONTROL_DT))  # deploy contract: 0.1 s apart (paper-silent)
GOAL_AHEAD_S = 1.5  # s: ref_free goal command, the reference root this far ahead (= the contact horizon, contacts.HORIZON)
# Some clips (e.g. x2_recorded_gestures) are stored floor-relative with the root at z = 0. They are detected from the
# FEET, not the root: a root-height rule (eval_x2_mujoco's "max root z < 0.3 -> +0.60") also lifts grounded deep
# crouches 0.6 m into the air. A clip whose lowest body origin sits more than FLOOR_RELATIVE_Z below the floor (median)
# is shifted so that median lands at X2_LOWEST_ORIGIN, the lowest-origin height of a grounded X2 stance.
X2_LOWEST_ORIGIN = 0.073
FLOOR_RELATIVE_Z = -0.3

# IsaacLab joint order of the deploy contract (eval_x2_mujoco.IL_TO_MJ_DOF): x_il = x_mj[IL_TO_MJ]
IL_TO_MJ = [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 29, 15, 22, 4, 10, 30, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]

# BeyondMimic's tracked bodies (anchor = pelvis), the set SONIC's X2 config uses.
TRACKED_BODIES = [
    "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link", "torso_link",
    "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
    "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
]  # fmt: skip
if CFG:  # another robot: its own bodies and floor height; its action order is its MuJoCo order
    IL_TO_MJ = list(range(CFG["num_joints"]))
    TRACKED_BODIES = CFG["tracked_bodies"]
    X2_LOWEST_ORIGIN = CFG["lowest_origin"]


# ---------------------------------------------------------------- rotation helpers (torch, wxyz)
def quat_slerp(q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    dot = (q0 * q1).sum(-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)
    dot = dot.abs().clamp(max=1.0)
    theta = torch.acos(dot)
    s = torch.sin(theta)
    t = t.unsqueeze(-1)
    w0 = torch.where(s > 1e-6, torch.sin((1 - t) * theta) / s.clamp(min=1e-12), 1 - t)
    w1 = torch.where(s > 1e-6, torch.sin(t * theta) / s.clamp(min=1e-12), t)
    q = w0 * q0 + w1 * q1
    return q / q.norm(dim=-1, keepdim=True)


def so3_log(R: torch.Tensor) -> torch.Tensor:
    """Rotation matrix -> rotation vector (stable near 0 and pi)."""
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos = ((tr - 1) / 2).clamp(-1 + 1e-7, 1 - 1e-7)
    theta = torch.acos(cos)
    v = torch.stack((R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]), -1)
    scale = torch.where(theta > 1e-4, theta / (2 * torch.sin(theta)), 0.5 + theta**2 / 12)
    return v * scale.unsqueeze(-1)


def _central_diff(x: torch.Tensor, dt: float) -> torch.Tensor:
    g = torch.empty_like(x)
    if x.shape[0] == 1:
        return torch.zeros_like(x)
    g[1:-1] = (x[2:] - x[:-2]) / (2 * dt)
    g[0] = (x[1] - x[0]) / dt
    g[-1] = (x[-1] - x[-2]) / dt
    return g


class _RotView:
    """``root_R[f]`` computed from the stored quaternions (saves 9 floats/frame)."""

    def __init__(self, quat: torch.Tensor):
        self.quat = quat

    def __getitem__(self, f):
        return quat_wxyz_to_mat(self.quat[f])


class _BackDiffView:
    """Backward-difference joint velocity ``(q[f] - q[f-1]) / dt``, 0 at a clip's first frame (deploy tokenizer)."""

    def __init__(self, q: torch.Tensor, clip_start: torch.Tensor, clip_id: torch.Tensor):
        self.q, self.clip_start, self.clip_id = q, clip_start, clip_id

    def __getitem__(self, f):
        prev = torch.maximum(f - 1, self.clip_start[self.clip_id[f]])
        return (self.q[f] - self.q[prev]) / CONTROL_DT


def _resample(clip: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One clip -> (root_p (n,3), root_quat_wxyz (n,4), q (n,31)) at 50 Hz."""
    fps = float(clip["fps"])
    p = torch.as_tensor(np.asarray(clip["root_trans_offset"], dtype=np.float64))
    xyzw = torch.as_tensor(np.asarray(clip["root_rot"], dtype=np.float64))
    quat = torch.cat((xyzw[:, 3:4], xyzw[:, :3]), -1)
    quat = quat / quat.norm(dim=-1, keepdim=True)
    q = torch.as_tensor(np.asarray(clip["dof"], dtype=np.float64))
    T = q.shape[0]
    if T == 1:
        return p.float(), quat.float(), q.float()
    dur = (T - 1) / fps
    n = int(math.floor(dur / CONTROL_DT + 1e-9)) + 1
    s = torch.arange(n, dtype=torch.float64) * CONTROL_DT * fps
    i0 = s.floor().long().clamp(max=T - 2)
    a = (s - i0).clamp(0, 1)
    lerp = lambda x: x[i0] * (1 - a[:, None]) + x[i0 + 1] * a[:, None]  # noqa: E731
    return lerp(p).float(), quat_slerp(quat[i0], quat[i0 + 1], a).float(), lerp(q).float()


def ground_floor_relative(p: torch.Tensor, quat: torch.Tensor, q: torch.Tensor, kin: X2Kinematics) -> tuple[torch.Tensor, float]:
    """Return (root positions, applied z shift); shifts only clips stored floor-relative (see FLOOR_RELATIVE_Z)."""
    idx = torch.arange(0, p.shape[0], max(1, p.shape[0] // 200))
    dev = kin.device
    pp, R, qq = p[idx].to(dev), quat_wxyz_to_mat(quat[idx]).to(dev), q[idx].to(dev)
    z = torch.zeros_like(pp)
    zmin = kin.forward(pp, R, z, z, qq, torch.zeros_like(qq))[0][..., 2].min(-1).values
    med = float(zmin.median())
    if med >= FLOOR_RELATIVE_Z:
        return p, 0.0
    shift = X2_LOWEST_ORIGIN - med
    p = p.clone()
    p[:, 2] += shift
    return p, shift


class MotionLib:
    def __init__(self, clips: dict[str, dict], kin: X2Kinematics, device: str | torch.device = "cuda"):
        self.device = torch.device(device)
        self.kin = kin
        self.names = list(clips.keys())
        P, Q, J, lens = [], [], [], []
        self.grounded_shift = {}
        for name in self.names:
            p, quat, q = _resample(clips[name])
            p, shift = ground_floor_relative(p, quat, q, kin)
            if shift:
                self.grounded_shift[name] = shift
            P.append(p), Q.append(quat), J.append(q), lens.append(q.shape[0])
        if self.grounded_shift:
            print(f"[motion] grounded {len(self.grounded_shift)} floor-relative clips (z shift "
                  f"{min(self.grounded_shift.values()):.3f}..{max(self.grounded_shift.values()):.3f} m)", flush=True)
        dev = self.device
        self.clip_len = torch.tensor(lens, device=dev)
        self.clip_start = torch.cumsum(torch.tensor([0] + lens[:-1], device=dev), 0)
        self.clip_end = self.clip_start + self.clip_len - 1  # last valid frame (inclusive)
        self.num_frames = int(sum(lens))
        self.clip_id = torch.repeat_interleave(torch.arange(len(lens), device=dev), self.clip_len)

        root_v, root_w, qd = [], [], []
        for p, quat, q in zip(P, Q, J):
            R = quat_wxyz_to_mat(quat)
            w = torch.zeros_like(p)
            if p.shape[0] > 1:
                w[1:-1] = so3_log(R[2:] @ R[:-2].transpose(-1, -2)) / (2 * CONTROL_DT)
                w[0] = so3_log(R[1] @ R[0].T) / CONTROL_DT
                w[-1] = so3_log(R[-1] @ R[-2].T) / CONTROL_DT
            root_v.append(_central_diff(p, CONTROL_DT)), root_w.append(w), qd.append(_central_diff(q, CONTROL_DT))
        cat = lambda xs: torch.cat(xs).to(dev)  # noqa: E731
        # stored per frame: 3 + 4 + 3 + 3 + 31 + 31 = 75 floats; everything else is computed on demand
        self.root_p, self.root_quat = cat(P), cat(Q)
        self.root_v, self.root_w = cat(root_v), cat(root_w)
        self.q, self.qd = cat(J), cat(qd)
        self.root_R = _RotView(self.root_quat)
        self.qd_back = _BackDiffView(self.q, self.clip_start, self.clip_id)

        # 1 s bins for return-based reset sampling (paper App. D.1)
        nb = torch.div(self.clip_len + BIN_FRAMES - 1, BIN_FRAMES, rounding_mode="floor")
        self.num_bins = int(nb.sum())
        self.bin_clip = torch.repeat_interleave(torch.arange(len(lens), device=dev), nb)
        first_bin = torch.cumsum(torch.cat((torch.zeros(1, device=dev, dtype=nb.dtype), nb[:-1])), 0)
        within = torch.arange(self.num_bins, device=dev) - first_bin[self.bin_clip]
        self.bin_start = self.clip_start[self.bin_clip] + within * BIN_FRAMES
        self.frame_bin = first_bin[self.clip_id] + torch.div(
            torch.arange(self.num_frames, device=dev) - self.clip_start[self.clip_id], BIN_FRAMES, rounding_mode="floor")

        self.tracked = kin.body_index(TRACKED_BODIES)
        self.il_to_mj = torch.tensor(IL_TO_MJ, device=dev)

    @staticmethod
    def load_pkls(paths: list[str | pathlib.Path], kin: X2Kinematics, device="cuda", keys: list[str] | None = None) -> MotionLib:
        clips: dict[str, dict] = {}
        for path in paths:
            data = joblib.load(path)
            for k, v in data.items():
                if keys is None or k in keys:
                    clips[k] = v
        if keys is not None:
            missing = sorted(set(keys) - set(clips))
            assert not missing, f"{len(missing)} requested clips not found, e.g. {missing[:3]}"
        return MotionLib(clips, kin, device)

    # ------------------------------------------------------------ queries
    def ref_state(self, f: torch.Tensor):
        """(p, R, v, w, q, qd) of the reference root/joints at frames f (world frame)."""
        return self.root_p[f], self.root_R[f], self.root_v[f], self.root_w[f], self.q[f], self.qd[f]

    def ref_bodies(self, f: torch.Tensor):
        """Tracked-body (pos, R, linvel, angvel) of the reference at frames f: (..., 14, ...)."""
        P, R, V, W = self.kin.forward(*self.ref_state(f))
        t = self.tracked
        return P[..., t, :], R[..., t, :, :], V[..., t, :], W[..., t, :]

    def future_frames(self, f: torch.Tensor) -> torch.Tensor:
        """Frames f + 5k, k = 0..9 (k = 0 is the current frame), clamped to the clip end: (B, 10)."""
        fut = f.unsqueeze(-1) + FUTURE_STRIDE * torch.arange(K_FUTURE, device=f.device)
        return torch.minimum(fut, self.clip_end[self.clip_id[f]].unsqueeze(-1))

    def root_ahead(self, f: torch.Tensor, p: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        """(B, 30) the reference root at the K_FUTURE goal frames relative to the robot root (p, R), in the robot's
        heading frame: where the robot must be, which the goal (joints, orientation) does not say. Detached."""
        from gear_sonic.supermimic.mdp import yaw_mat

        fut = self.future_frames(f)
        d = self.root_p[fut] - p.detach().unsqueeze(-2)                        # (B, 10, 3)
        H = yaw_mat(R.detach()).unsqueeze(-3)
        return (H * d.unsqueeze(-1)).sum(-2).flatten(-2)                       # H^T d

    def goal_cmd(self, f: torch.Tensor, p: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        """(B, 30) ACP's goal command for the reference-free policy (ref_free): a bounded horizontal direction to the
        reference root GOAL_AHEAD_S ahead (the offset in the robot's heading frame, scaled to length <= 1 m) and the
        relative heading there (cos, sin), zero-padded to the 30-wide reference-position slot it replaces. Detached."""
        from gear_sonic.supermimic.mdp import yaw_mat

        g = torch.minimum(f + int(round(GOAL_AHEAD_S / CONTROL_DT)), self.clip_end[self.clip_id[f]])
        H = yaw_mat(R.detach())
        d = mtv3(H, self.root_p[g] - p.detach())[..., :2]
        d = d / d.norm(dim=-1, keepdim=True).clamp(min=1.0)
        Hg = yaw_mat(self.root_R[g])
        rel = H.transpose(-1, -2) @ Hg                                        # goal heading in the robot's heading frame
        x = torch.cat((d, rel[..., 0, 0:1], rel[..., 1, 0:1]), -1)
        gs = getattr(self, "goals", None)
        if gs is not None and gs.count:  # ACP stage 3 goal clips (goal.py): the tile's goal, heading +x
            dg = mtv3(H, gs.pos[self.clip_id[f]] - p.detach())[..., :2]
            dg = dg / dg.norm(dim=-1, keepdim=True).clamp(min=1.0)
            xg = torch.cat((dg, H[..., 0, 0:1], H[..., 0, 1:2]), -1)       # H^T I: heading of +x in the robot frame
            x = torch.where(gs.is_goal[self.clip_id[f]].unsqueeze(-1), xg, x)
        return torch.cat((x, x.new_zeros(*x.shape[:-1], 26)), -1)

    def goal(self, f: torch.Tensor, robot_R: torch.Tensor) -> torch.Tensor:
        """Deploy-contract 680-D goal (the ``build_tokenizer_obs`` layout, anchor-ori mode ``b``).

        ``cat(jpos(10x31), jvel(10x31)).reshape(10, 62)`` (IsaacLab joint order; the "buggy temporal
        reshape" kept as deployed) next to the per-frame 6D of ``robot_R^T R_ref(f_k)`` flattened
        row-major as ``mat[:, :2]``. ``robot_R`` (B,3,3) is the robot root rotation (differentiable).
        """
        fut = self.future_frames(f)  # (B, 10)
        B = f.shape[0]
        jpos = self.q[fut][..., self.il_to_mj].reshape(B, -1)
        jvel = self.qd_back[fut][..., self.il_to_mj].reshape(B, -1)
        cmd = torch.cat((jpos, jvel), -1).reshape(B, K_FUTURE, 2 * len(IL_TO_MJ))
        rel = mm3(robot_R.transpose(-1, -2).unsqueeze(1), self.root_R[fut])  # (B, 10, 3, 3)
        ori = rel[..., :2].reshape(B, K_FUTURE, 6)
        return torch.cat((cmd, ori), -1).reshape(B, K_FUTURE * (2 * len(IL_TO_MJ) + 6))
