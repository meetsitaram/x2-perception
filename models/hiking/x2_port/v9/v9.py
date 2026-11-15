"""v9 (2026-11-14, user decisions "Hiking v9 decisions" in docs/USER_RULES.md): v8 + the paper's command mix + no touchdown reward.
Applied last by extended.X2ParkourExtendedFixedEnvCfg when HIKING_X2_V9=1 (after apply_fair, which v9 keeps).

Launcher part (env vars, no code here): stand share 0.05 (release rel_standing_envs), turn-in-place share 0 and slow-forward /
slow-yaw shares 0 (the release has neither: paper obstacle columns get U(0.45, 0.8) m/s toward a far target, yaw from the heading
rule, clipped to +-1 rad/s), option t removed (the paper has no touchdown term; feet_air_time 0.5 stays).

Code part:
1. Paper Rough (Stand) column (HIKING_X2_PAPER_STAND=1). The release samples the perlin_rough_stand column's random velocity from
   ranges lin_vel_x (0, 0), lin_vel_y (0, 0), ang_vel_z (-1, 1): turn in place and stand only. Our extended task widened the shared
   random range to vx 0..0.8, vy +-0.3 for its velocity column; here the stand column's robots get the release ranges back, the
   extended velocity column (perlin_rough_velocity) keeps the wide ones (user's map). Command class V9Command.
2. Per-type curriculum log (always on with v9): at every reset, per paper terrain type, running means (EMA 0.02) of the promotion
   score (tracking_exp_vel_xy, = sum of exp(-err^2/0.25) / max episode length), the up / down shares of eligible episodes, the share
   of eligible episodes that ended by a termination (not a time-out), and the eligible share. Curriculum term `by_type`, keys
   Curriculum/by_type/<stat>_<type>. Answers why square_gaps sits at rows 0-1 (open item of 2026-11-14).
3. Contact weight scales (OPTIONAL, default 1.0 = unchanged, user decision pending): HIKING_X2_STUMBLE_SCALE and
   HIKING_X2_CONTACT_SCALE multiply feet_stumble and wall_contact_graded per robot; HIKING_X2_SCALE_SCOPE=paper (default) applies
   them only to robots on the paper's terrain columns (the extended map keeps the user's weights), =all to every robot. With both
   scales at 1.0 nothing is wrapped (cfg identical to v8's terms).
"""
from __future__ import annotations

import inspect
import os

import numpy as np
import torch

from isaaclab.utils import configclass

from .fair import FairSlowRouteFarTargetPoseVelocityCommand, FairSlowRouteFarTargetPoseVelocityCommandCfg, tracking_exp_vel_paper_cmd

PAPER_TYPES = ("perlin_rough", "perlin_rough_stand", "square_gaps", "pyramid_stairs", "pyramid_stairs_high", "pyramid_stairs_inv",
               "pyramid_stairs_inv_high", "boxes", "mesh_boxes", "hf_pyramid_slope_inv")


def column_type_names(generator_cfg) -> list:
    """Sub-terrain name of every terrain column, by the release's proportion rule (pose_velocity_command.py)."""
    props = np.array([s.proportion for s in generator_cfg.sub_terrains.values()], dtype=float)
    props /= props.sum()
    names = list(generator_cfg.sub_terrains.keys())
    out = []
    for i in range(generator_cfg.num_cols):
        out.append(names[int(np.min(np.where(i / generator_cfg.num_cols + 0.001 < np.cumsum(props))[0]))])
    return out


def env_type_names(env) -> list:
    """Per robot: sub-terrain name of its column (cached)."""
    c = _st.get("types")
    if c is None:
        cols = column_type_names(env.scene.terrain.cfg.terrain_generator)
        tt = env.scene.terrain.terrain_types.cpu().numpy()
        c = _st["types"] = [cols[int(t)] for t in tt]
        _st["paper_mask"] = torch.tensor([n in PAPER_TYPES for n in c], device=env.device)
    return c


_st = {}


def set_stand_ranges(lo_hi_x: torch.Tensor, lo_hi_y: torch.Tensor, mask: torch.Tensor) -> None:
    """In place: robots in mask get the release stand-column random ranges vx (0, 0), vy (0, 0); yaw range untouched."""
    lo_hi_x[mask] = 0.0
    lo_hi_y[mask] = 0.0


class V9Command(FairSlowRouteFarTargetPoseVelocityCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if os.environ.get("HIKING_X2_PAPER_STAND", "0") == "1":
            names = column_type_names(self.terrain.cfg.terrain_generator)
            tt = self.terrain.terrain_types.cpu().numpy()
            mask = torch.tensor([names[int(t)] == "perlin_rough_stand" for t in tt], device=self.device) & self.random_velocity_indices
            set_stand_ranges(self.random_lin_vel_x_range, self.random_lin_vel_y_range, mask)
            print("[X2 v9] paper stand column: %d robots get vx (0,0) vy (0,0) wz %s; velocity column robots keep vx %s vy %s"
                  % (int(mask.sum()), tuple(cfg.ranges.ang_vel_z), tuple(cfg.ranges.lin_vel_x), tuple(cfg.ranges.lin_vel_y)), flush=True)


@configclass
class V9CommandCfg(FairSlowRouteFarTargetPoseVelocityCommandCfg):
    class_type: type = V9Command


def type_stats_update(acc: dict, types: list, score: torch.Tensor, up: torch.Tensor, down: torch.Tensor, eligible: torch.Tensor,
                      terminated: torch.Tensor, a: float = 0.02) -> None:
    """acc[type] = dict of EMAs over this type's resets: score, up, down (of eligible), early (termination share of eligible), elig."""
    for t in set(types):
        if t not in PAPER_TYPES:
            continue
        sel = torch.tensor([x == t for x in types], dtype=torch.bool)
        if not sel.any():
            continue
        e = eligible[sel]
        d = acc.setdefault(t, dict(score=0.0, up=0.0, down=0.0, early=0.0, elig=0.0, n=0))
        k = min(1.0, a * int(sel.sum()))
        d["elig"] += k * (float(e.float().mean()) - d["elig"]); d["n"] += int(sel.sum())
        if e.any():
            k = min(1.0, a * int(e.sum()))
            for key, v in (("score", score[sel][e]), ("up", up[sel][e].float()), ("down", down[sel][e].float()),
                           ("early", terminated[sel][e].float())):
                d[key] += k * (float(v.mean()) - d[key])


def tracking_exp_vel_paper_cmd_v9(env, env_ids, lin_vel_threshold: tuple = (0.3, 0.6), ang_vel_threshold: tuple = (0.0, 0.0),
                                  asset_cfg=None) -> torch.Tensor:
    """v8's curriculum term unchanged, plus the per-type statistics of these resets (read before the term moves the levels)."""
    try:
        command = env.command_manager.get_term("base_velocity")
        ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
        xy = command.metrics["tracking_exp_vel_xy"][ids]
        frozen = getattr(command, "nonpaper_ep", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))[ids]
        term = env.termination_manager.terminated[ids]
        names = env_type_names(env)
        types = [names[i] for i in ids.tolist()]
        type_stats_update(_st.setdefault("acc", {}), types, xy.cpu(), (xy > lin_vel_threshold[1]).cpu(),
                          (xy < lin_vel_threshold[0]).cpu(), (~frozen).cpu(), term.cpu())
    except Exception as ex:  # a log must never stop training
        if not _st.get("warned"):
            print("[X2 v9] by_type log error (ignored):", repr(ex), flush=True); _st["warned"] = True
    return tracking_exp_vel_paper_cmd(env, env_ids, lin_vel_threshold, ang_vel_threshold, asset_cfg)


def by_type(env, env_ids) -> dict:
    out = {}
    for t, d in _st.get("acc", {}).items():
        for k in ("score", "up", "down", "early", "elig"):
            out["%s_%s" % (k, t)] = float(d[k])
    return out


def scaled(func, scale: float, scope: str):
    def f(env, **kw):
        r = func(env, **kw)
        if scope == "all":
            return r * scale
        env_type_names(env)
        return r * torch.where(_st["paper_mask"], torch.full_like(r, scale), torch.ones_like(r))
    f.__name__ = "%s_x%g_%s" % (func.__name__, scale, scope)
    f.__signature__ = inspect.signature(func)   # Isaac Lab's term check reads the parameters of the wrapped term
    return f


def apply_v9(cfg) -> None:
    from isaaclab.managers import CurriculumTermCfg as CurrTerm
    old = cfg.commands.base_velocity
    new = V9CommandCfg()
    for k, v in old.__dict__.items():
        if k != "class_type" and hasattr(new, k):
            setattr(new, k, v)
    cfg.commands.base_velocity = new
    cfg.curriculum.terrain_levels.func = tracking_exp_vel_paper_cmd_v9
    cfg.curriculum.by_type = CurrTerm(func=by_type)
    r = cfg.rewards.rewards
    scope = os.environ.get("HIKING_X2_SCALE_SCOPE", "paper")
    for name, var in (("feet_stumble", "HIKING_X2_STUMBLE_SCALE"), ("wall_contact_graded", "HIKING_X2_CONTACT_SCALE")):
        s = float(os.environ.get(var, "1.0"))
        t = getattr(r, name, None)
        if t is not None and s != 1.0:
            t.func = scaled(t.func, s, scope)
