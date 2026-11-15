"""v8 (2026-11-14, Hiking priority: close the gap to the paper's Table 2 on the paper's terrains). Two changes, each behind an
env var, applied last by extended.X2ParkourExtendedFixedEnvCfg when HIKING_X2_FAIR=1:

1. Paper curriculum signal restored (HIKING_X2_FAIR=1). The paper promotes / demotes a robot on the episode mean of
   exp(-|v_cmd - v|^2 / 0.5^2) (> 0.6 up, < 0.3 down; tracking_exp_vel) with every robot on an obstacle column commanded
   U(0.45, 0.8) m/s toward a far target (5% stand). Our additions (stand share 0.10, turn in place 0.15, slow forward cap
   U(0.1, 0.45) for 25%) give many robots a command a REFUSING robot tracks well: Monte Carlo (state/hiking.md, 2026-11-14) puts
   a robot that stops at the obstacle foot at 7% up / 59% down under the paper's commands and 42% up / 21% down under ours, so the
   rows saturate (mean level 6.0 = recycling equilibrium on every type) while the MuJoCo eval stalls at rows 5 and 8 (all 0/20
   failures are 'stalled' at the obstacle foot, pelvis 0.67 m, command 0.7 m/s). Here an episode that had any stand, turn or
   slow-forward command is neither promoted nor demoted; all other episodes use the paper's rule unchanged.
2. Touchdown pay only for progress (HIKING_X2_AIR_TD_PROGRESS=1). Option t (feet_air_touchdown_refcap, weight 140; ours, the
   paper has feet_air_time 0.5) pays about 26 per episode (linear tracking 30) for every touchdown under a moving command,
   including marking time in front of an obstacle (eval: repeated edge steps at the same x on platform_up L5). Under a walking
   command (|v_cmd xy| > 0.1) the pay is multiplied by clamp(v . v_cmd / |v_cmd|^2, 0, 1); turn-in-place pay is unchanged.
"""
from __future__ import annotations

import os

import torch

from .slow import SlowRouteFarTargetPoseVelocityCommand, SlowRouteFarTargetPoseVelocityCommandCfg
from .staged import feet_air_touchdown_refcap

from isaaclab.utils import configclass


def update_flags(nonpaper: torch.Tensor, ids: torch.Tensor, standing: torch.Tensor, turning: torch.Tensor,
                 vcap: torch.Tensor) -> None:
    """In place: an episode stops counting for the curriculum once one of its commands is a stand, a turn or slow-forward."""
    nonpaper[ids] |= standing[ids] | turning[ids] | torch.isfinite(vcap[ids])


class FairSlowRouteFarTargetPoseVelocityCommand(SlowRouteFarTargetPoseVelocityCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.nonpaper_ep = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.metrics["paper_cmd_episode"] = torch.zeros(self.num_envs, device=self.device)
        print("[X2 fair] curriculum moves only on paper-command episodes (stand / turn / slow-forward episodes keep their level)",
              flush=True)

    def reset(self, env_ids=None):
        # the curriculum has already read nonpaper_ep for these robots (ManagerBasedRLEnv._reset_idx: curriculum first)
        ids = slice(None) if env_ids is None else torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self.metrics["paper_cmd_episode"][ids] = (~self.nonpaper_ep[ids]).float()
        self.nonpaper_ep[ids] = False
        return super().reset(env_ids)

    def _resample(self, env_ids):
        super()._resample(env_ids)
        if len(env_ids) == 0:
            return
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        update_flags(self.nonpaper_ep, ids, self.is_standing_env, self.is_turn_env, self.vcap)


@configclass
class FairSlowRouteFarTargetPoseVelocityCommandCfg(SlowRouteFarTargetPoseVelocityCommandCfg):
    class_type: type = FairSlowRouteFarTargetPoseVelocityCommand


def promotion_masks(xy: torch.Tensor, yaw: torch.Tensor, frozen: torch.Tensor, lin_th=(0.3, 0.6), ang_th=(0.0, 0.0)):
    """The paper's tracking_exp_vel rule, with frozen robots neither moved up nor down."""
    move_up = (xy > lin_th[1]) & (yaw > ang_th[1]) & ~frozen
    move_down = (xy < lin_th[0]) & ~move_up & ~frozen
    return move_up, move_down


def tracking_exp_vel_paper_cmd(env, env_ids, lin_vel_threshold: tuple = (0.3, 0.6), ang_vel_threshold: tuple = (0.0, 0.0),
                               asset_cfg=None) -> torch.Tensor:
    terrain = env.scene.terrain
    command = env.command_manager.get_term("base_velocity")
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    frozen = getattr(command, "nonpaper_ep", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))[ids]
    up, down = promotion_masks(command.metrics["tracking_exp_vel_xy"][ids], command.metrics["tracking_exp_vel_yaw"][ids],
                               frozen, lin_vel_threshold, ang_vel_threshold)
    terrain.update_env_origins(ids, up, down)
    return torch.mean(terrain.terrain_levels.float())


def progress_ratio(v_b: torch.Tensor, cmd: torch.Tensor) -> torch.Tensor:
    n2 = (cmd[:, :2] ** 2).sum(dim=1).clamp(min=1e-6)
    return ((v_b[:, :2] * cmd[:, :2]).sum(dim=1) / n2).clamp(0.0, 1.0)


def feet_air_touchdown_refcap_progress(env, command_name: str, sensor_cfg, threshold: float = 0.05,
                                       min_cmd: float = 0.1) -> torch.Tensor:
    pay = feet_air_touchdown_refcap(env, command_name, sensor_cfg, threshold, min_cmd)
    cmd = env.command_manager.get_command(command_name)
    walking = torch.norm(cmd[:, :2], dim=1) > min_cmd
    prog = progress_ratio(env.scene["robot"].data.root_lin_vel_b, cmd)
    return torch.where(walking, pay * prog, pay)


def apply_fair(cfg) -> None:
    old = cfg.commands.base_velocity
    new = FairSlowRouteFarTargetPoseVelocityCommandCfg()
    for k, v in old.__dict__.items():
        if k != "class_type" and hasattr(new, k):
            setattr(new, k, v)
    cfg.commands.base_velocity = new
    cfg.curriculum.terrain_levels.func = tracking_exp_vel_paper_cmd
    if os.environ.get("HIKING_X2_AIR_TD_PROGRESS", "0") == "1":
        cfg.rewards.rewards.feet_air_touchdown.func = feet_air_touchdown_refcap_progress
