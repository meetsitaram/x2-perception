"""simcheck.episode_end: tell a simulator fault from a policy failure at an episode end, and treat the fault as a truncation.

THE RULE (docs/TRACK_RULES.md, "Simulator checks"; coordinator 2026-11-12 after FINDINGS B8-B10):
  A world's step is a SIMULATOR FAULT when any of these holds after the physics step:
    nan          any qpos or qvel is not finite
    qvel         max |qvel| over the track's own bound (ACP 200, SSR 60: what each already calls "diverged")
    depth        any contact deeper than DEPTH_M = 0.05 m (no real contact of the X2 is 5 cm deep; B10 had metres)
    speed_jump   any robot body's linear speed changes by more than JUMP_MS = 20 m/s between two control steps
                 (positions differenced; a thrown body, B10 threw a foot 240 m)
  A simulator fault ends the episode as a TRUNCATION: time-out style, the value is bootstrapped (r_t + gamma V(s_t), as the
  tracks already do for time limits), the step's reward is set to 0 (it was computed on a broken state), no termination
  penalty, and it does not move any curriculum (CSI phase, terrain level). It is never counted as a termination.
  Everything else that ends an episode early is the policy's: termination as before.
  Count the faults by cause and report their share of episode ends in the heartbeat (simcheck.alarm).

Use (torch, on the device of the simulator data; works on mujoco_warp's Data through wp.to_torch):
    det = FaultDetector(num_envs, device, qvel_max=200.0, control_dt=0.02)       # once
    ...physics...
    fault, why = det.update(qpos, qvel, xpos_robot, contact_depth_per_world(dd, N))   # each control step
    term, timeout, r = truncate_faults(term, timeout, r, fault)
    det.reset(reset_ids)                                                            # after the env resets worlds
"""
from __future__ import annotations

DEPTH_M = 0.05
JUMP_MS = 20.0
CAUSES = ("nan", "qvel", "depth", "speed_jump")


def contact_depth_per_world(dd, nworld: int):
    """Deepest contact per world (m, >= 0) from mujoco_warp Data, on the data's device."""
    import torch
    import warp as wp
    T = wp.to_torch
    dist = T(dd.contact.dist)
    n = min(int(T(dd.nacon)[0]), dist.shape[0])
    out = torch.zeros(nworld, device=dist.device)
    if n:
        dep = torch.nan_to_num(-dist[:n], nan=1e9, posinf=1e9, neginf=0.0)
        out.scatter_reduce_(0, T(dd.contact.worldid)[:n].long(), dep, reduce="amax")
    return out


class FaultDetector:
    def __init__(self, num_envs: int, device, qvel_max: float, control_dt: float, depth_m: float = DEPTH_M,
                 jump_ms: float = JUMP_MS):
        import torch
        self.N, self.dev = num_envs, device
        self.qvel_max, self.dt, self.depth_m, self.jump_ms = qvel_max, control_dt, depth_m, jump_ms
        self.xprev = None
        self.vprev = None
        self.valid = torch.zeros(num_envs, dtype=torch.int8, device=device)   # 0: no history, 1: position, 2: velocity
        self.counts = {k: 0 for k in CAUSES}
        self.counts["any"] = 0

    def update(self, qpos, qvel, xpos=None, depth=None):
        """qpos (N, nq), qvel (N, nv), xpos (N, nbody, 3) robot body positions or None, depth (N,) or None.
        Returns (fault (N,) bool, why dict of (N,) bool)."""
        import torch
        why = {"nan": ~(torch.isfinite(qpos).all(-1) & torch.isfinite(qvel).all(-1))}
        why["qvel"] = torch.nan_to_num(qvel.abs(), nan=0.0).amax(-1) > self.qvel_max
        why["depth"] = (depth > self.depth_m) if depth is not None else torch.zeros_like(why["nan"])
        jump = torch.zeros_like(why["nan"])
        if xpos is not None:
            x = torch.nan_to_num(xpos.detach(), nan=1e6)
            if self.xprev is not None:
                v = (x - self.xprev) / self.dt
                if self.vprev is not None:
                    jump = ((v - self.vprev).norm(dim=-1).amax(-1) > self.jump_ms) & (self.valid >= 2)
                self.vprev = v
                self.valid = torch.clamp(self.valid + 1, max=2)
            else:
                self.valid = torch.ones_like(self.valid)
            self.xprev = x.clone()
        why["speed_jump"] = jump
        fault = why["nan"] | why["qvel"] | why["depth"] | why["speed_jump"]
        for k in CAUSES:
            self.counts[k] += int(why[k].sum())
        self.counts["any"] += int(fault.sum())
        return fault, why

    def reset(self, ids):
        """Call with the ids of the worlds the env reset (their bodies jumped there)."""
        if ids is not None and len(ids):
            self.valid[ids] = 0


def truncate_faults(term, timeout, reward, fault):
    """term, timeout (N,) bool, reward (N,) or (N, k): faults leave `term`, join `timeout`, and get reward 0."""
    import torch
    term = term & ~fault
    timeout = timeout | fault
    r = torch.where(fault.view(-1, *([1] * (reward.dim() - 1))), torch.zeros_like(reward), reward)
    return term, timeout, r


def classify(qpos, qvel, depth=None, speed_jump=None, qvel_max: float = 200.0):
    """Stateless form for logs and replays: (N,) bool simulator fault from states, the deepest contact (m) and the largest
    body speed change (m/s) per world. numpy or torch."""
    try:
        import torch
        if isinstance(qpos, torch.Tensor):
            f = ~(torch.isfinite(qpos).all(-1) & torch.isfinite(qvel).all(-1)) | (torch.nan_to_num(qvel.abs()).amax(-1) > qvel_max)
            if depth is not None:
                f = f | (depth > DEPTH_M)
            if speed_jump is not None:
                f = f | (speed_jump > JUMP_MS)
            return f
    except ImportError:
        pass
    import numpy as np
    qpos, qvel = np.asarray(qpos), np.asarray(qvel)
    f = ~(np.isfinite(qpos).all(-1) & np.isfinite(qvel).all(-1)) | (np.nan_to_num(np.abs(qvel)).max(-1) > qvel_max)
    if depth is not None:
        f |= np.asarray(depth) > DEPTH_M
    if speed_jump is not None:
        f |= np.asarray(speed_jump) > JUMP_MS
    return f
