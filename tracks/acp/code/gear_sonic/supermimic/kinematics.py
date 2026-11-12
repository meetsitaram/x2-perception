"""Differentiable batched forward kinematics of the X2, built from the compiled MuJoCo model.

Taking the tree (body offsets, hinge axes) from the same ``MjModel`` that the simulator steps keeps
the analytic reward/observation heads consistent with the physics by construction. Gives world
poses and world linear/angular velocities of every body (the paper's reward uses body linear and
angular velocity), from the root state and joint state.

Conventions: rotation matrices; root linear velocity of the pelvis origin in world frame, root
angular velocity in world frame; ``q``/``qd`` in MuJoCo joint order.
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch


def quat_wxyz_to_mat(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ),
        -1,
    ).reshape(*q.shape[:-1], 3, 3)  # fmt: skip


def mm3(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """A @ B for (..., 3, 3) matrices as broadcast multiply + sum (elementwise kernels, fusable; no tiny GEMMs)."""
    return (A.unsqueeze(-1) * B.unsqueeze(-3)).sum(-2)


def mv3(A: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """A @ v for (..., 3, 3) and (..., 3)."""
    return (A * v.unsqueeze(-2)).sum(-1)


def mtv3(A: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """A^T @ v for (..., 3, 3) and (..., 3)."""
    return (A * v.unsqueeze(-1)).sum(-2)


def axis_angle_to_mat(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rodrigues for unit axes (..., 3) and angles (...)."""
    x, y, z = axis.unbind(-1)
    zero = torch.zeros_like(x)
    K = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(*axis.shape[:-1], 3, 3)
    s, c = torch.sin(angle)[..., None, None], torch.cos(angle)[..., None, None]
    eye = torch.eye(3, device=axis.device, dtype=axis.dtype).expand_as(K)
    return eye + s * K + (1 - c) * mm3(K, K)


class X2Kinematics:
    def __init__(self, model: mujoco.MjModel, device: str | torch.device = "cpu"):
        assert model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
        self.device = torch.device(device)
        self.model = model
        nb = model.nbody
        # body 0 = world, body 1 = pelvis (free joint); every other body has exactly one hinge
        self.names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(1, nb)]
        self.parent = [int(model.body_parentid[i]) - 1 for i in range(1, nb)]  # -1 = world
        jbody = {int(model.jnt_bodyid[j]): j for j in range(1, model.njnt)}
        hinge_of_body, axes = [], []
        for b in range(2, nb):
            j = jbody[b]
            assert model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE and np.allclose(model.jnt_pos[j], 0.0)
            hinge_of_body.append(int(model.jnt_qposadr[j]) - 7)
            axes.append(model.jnt_axis[j])
        for b in range(1, nb):
            assert self.parent[b - 1] < b - 1, "bodies must be topologically ordered"
        f = dict(device=self.device, dtype=torch.float32)
        self.hinge_of_body = torch.tensor(hinge_of_body, device=self.device)  # body k+1 -> q index
        self.axis = torch.tensor(np.array(axes), **f)  # (nb-2, 3)
        self.offset = torch.tensor(model.body_pos[2:], **f)  # in parent frame
        self.local_R = quat_wxyz_to_mat(torch.tensor(model.body_quat[2:], **f))
        self.num_bodies = nb - 1
        self._build_levels()

    def _build_levels(self):
        """Bodies grouped by tree depth; each level's parents are positions in the previous level's tensor."""
        depth = [0] * self.num_bodies
        for i in range(1, self.num_bodies):
            depth[i] = depth[self.parent[i]] + 1
        groups = [[0]] + [[i for i in range(1, self.num_bodies) if depth[i] == d] for d in range(1, max(depth) + 1)]
        dev = self.device
        self.level_rows, self.level_parent_pos = [], []
        for d in range(1, len(groups)):
            prev = groups[d - 1]
            self.level_rows.append(torch.tensor([i - 1 for i in groups[d]], device=dev))  # hinge/offset rows
            self.level_parent_pos.append(torch.tensor([prev.index(self.parent[i]) for i in groups[d]], device=dev))
        order = [b for g in groups for b in g]  # concatenation order of the level tensors
        self.unpermute = torch.tensor([order.index(b) for b in range(self.num_bodies)], device=dev)

    def body_index(self, names: list[str]) -> torch.Tensor:
        return torch.tensor([self.names.index(n) for n in names], device=self.device)

    def forward(self, p, R, v, w, q, qd):
        """Root state (p, R, v, w) + joints (q, qd) -> body (pos, rot, linvel, angvel), all world frame.

        Shapes: p/v/w (..., 3), R (..., 3, 3), q/qd (..., 31) -> (..., nb, 3), (..., nb, 3, 3).
        """
        lead = p.shape[:-1]
        qi, qdi = q[..., self.hinge_of_body], qd[..., self.hinge_of_body]  # (..., nb-1)
        joint_R = mm3(self.local_R, axis_angle_to_mat(self.axis.expand(*lead, -1, -1), qi))
        # one tensor per tree level (no full-size copies), assembled once at the end
        Ps, Rs, Vs, Ws = [p.unsqueeze(-2)], [R.unsqueeze(-3)], [v.unsqueeze(-2)], [w.unsqueeze(-2)]
        for k, par in zip(self.level_rows, self.level_parent_pos):
            Rp, pp, vp, wp = Rs[-1][..., par, :, :], Ps[-1][..., par, :], Vs[-1][..., par, :], Ws[-1][..., par, :]
            r = mv3(Rp, self.offset[k])  # parent->child origin, world
            Rc = mm3(Rp, joint_R[..., k, :, :])
            a = mv3(Rc, self.axis[k])  # hinge axis in world (the joint frame is the child frame, jnt_pos = 0)
            Ps.append(pp + r)
            Rs.append(Rc)
            Vs.append(vp + torch.cross(wp, r, dim=-1))
            Ws.append(wp + a * qdi[..., k].unsqueeze(-1))
        u = self.unpermute
        return (torch.cat(Ps, -2)[..., u, :], torch.cat(Rs, -3)[..., u, :, :], torch.cat(Vs, -2)[..., u, :],
                torch.cat(Ws, -2)[..., u, :])  # fmt: skip
