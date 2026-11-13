"""simcheck for Hiking (docs/TRACK_RULES.md "Simulator checks", 2026-11-12; from /mnt/filesystem-g4/tools/simcheck/patches/hiking.diff,
moved into its own module so the CPU pre-flight can test it): a simulator fault (NaN, a body speed over 50 m/s, a body speed jump
over 20 m/s in one step) ends the episode as a time-out (bootstrapped, no termination penalty) and is never a failure: every
failure term ignores a faulted env. HIKING_X2_SIMFAULT=0 turns it off. Wired by staged.py (v5) at the end of __post_init__."""
from __future__ import annotations

import functools
import inspect
import os

import torch


def sim_fault(env, jump_ms: float = 20.0, speed_max: float = 50.0, asset_cfg=None):
    """(num_envs,) bool, computed once per env step (cached): NaN in the robot state, a body speed over speed_max, or a body
    speed change over jump_ms since the previous step (not on the first step after a reset)."""
    step = int(env.common_step_counter)
    c = getattr(env, "_simfault_cache", None)
    if c is not None and c[0] == step:
        return c[1]
    a = env.scene["robot" if asset_cfg is None else asset_cfg.name]
    v = a.data.body_lin_vel_w
    nan = ~(torch.isfinite(a.data.root_state_w).all(-1) & torch.isfinite(v).flatten(1).all(-1) & torch.isfinite(a.data.joint_pos).all(-1))
    vv = torch.nan_to_num(v, nan=0.0, posinf=1e6, neginf=-1e6)
    f = nan | (vv.norm(dim=-1).amax(-1) > speed_max)
    prev = getattr(env, "_simfault_vprev", None)
    if prev is not None:
        f = f | (((vv - prev).norm(dim=-1).amax(-1) > jump_ms) & (env.episode_length_buf > 1))
    env._simfault_vprev = vv.clone()
    env._simfault_cache = (step, f)
    return f


def _unless_sim_fault(func):
    # 2026-11-13: functools.wraps sets __wrapped__, so Isaac Lab's term check (inspect.signature) sees the wrapped
    # function's parameters; without it *args/**kwargs counted as mandatory (v5b construction check 02:06).
    @functools.wraps(func)
    def wrapped(env, *args, **kwargs):
        return func(env, *args, **kwargs) & ~sim_fault(env)
    wrapped.__name__ = getattr(func, "__name__", "term")
    return wrapped


def install(cfg, DoneTerm) -> list:
    """Wrap every non-time-out termination function; add `sim_fault` as a time-out. Returns the wrapped term names."""
    if os.environ.get("HIKING_X2_SIMFAULT", "1") != "1":
        return []
    done = []
    for name, t in list(vars(cfg.terminations).items()):
        if isinstance(t, DoneTerm) and not t.time_out and inspect.isfunction(t.func):
            t.func = _unless_sim_fault(t.func)
            done.append(name)
    cfg.terminations.sim_fault = DoneTerm(func=sim_fault, time_out=True)
    return done
