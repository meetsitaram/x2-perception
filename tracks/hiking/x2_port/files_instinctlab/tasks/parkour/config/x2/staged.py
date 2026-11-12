"""Candidate deviations for getting the X2 off the stand, as options of one task (`Instinct-Parkour-Staged-Amp-X2-v0`).

STATUS 2026-11-12: written on the CPU; the schedule and the two kernels are unit-tested with plain tensors (test_staged.py); the
task has NEVER been constructed in Isaac. With no option set it is exactly the run-2 task (`X2ParkourTorsoCamEnvCfg`).

Options are read from the environment when the config is built (the supervisor passes its environment to the trainer):
  HIKING_X2_OPTS   comma list of letters, e.g. "a,b,c"
  HIKING_X2_STAGE  "hold,end" in training iterations (default "500,1500"): every schedule holds its early value until `hold`,
                   then moves linearly to the paper's value at `end`, and stays there.
  HIKING_ITER_OFFSET  set by the supervisor to the checkpoint iteration a resume starts from, so schedules continue after a restart
                   (the env's step counter restarts at zero in each process).

  a  staged rewards: `is_alive` x 0.3 early (HIKING_X2_ALIVE_EARLY overrides the 0.3; run 3 uses 0.3); the penalties a first step pays (`action_rate_l2`, `feet_slide`, `dof_acc_l2`,
     `feet_flat_ori`) x 0.2 early.
  b  tracking: linear-tracking kernel std 0.25 early (zero speed pays exp(-0.63^2/0.25^2) = 0.002 for a 0.63 m/s command; the
     paper's 0.5 pays 0.20); yaw tracking multiplied by forward progress (v_x / (0.5 x commanded v_x), clipped to 0..1) for robots
     commanded to move, so turning on the spot earns nothing early. The gate fades out with the schedule.
  c  commands: no robots told to stand and a higher minimum speed (0.6 m/s) for the whole run (not scheduled: these are config
     values read once); targets are redrawn until they are at least 1.5 m from the robot, so the command is not zero from the start
     (about 15% of robots otherwise). The stand column keeps its zero commands.
  d  not moving: `dont_wait` x 4 early (HIKING_X2_WAIT_EARLY overrides the 4; run 3 uses 2); and an early termination `stalled`: a robot commanded to move whose average speed since
     the episode start is under 0.1 m/s after 4 s is reset. The termination is switched off at `end`.
  e  `feet_air_time` x 4 early.
  s  style reward coefficient (0.25) x 0.3 early (HIKING_X2_STYLE_EARLY), patched into the trainer's WasabiPPO.
     Every staged run (with or without s) logs `style_coef` and `style_reward_weighted` per step.
Every option is a listed deviation from the paper, with its schedule; the run's state file must say which were on.
"""
from __future__ import annotations

import os

import torch

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import instinctlab.tasks.parkour.mdp as mdp
from instinctlab.tasks.parkour.mdp.commands.commands_cfg import PoseVelocityCommandCfg
from instinctlab.tasks.parkour.mdp.commands.pose_velocity_command import PoseVelocityCommand

from .x2_parkour_target_amp_cfg import X2ParkourTorsoCamEnvCfg

STEPS_PER_ITERATION = 24


def _opts():
    return {o.strip() for o in os.environ.get("HIKING_X2_OPTS", "").split(",") if o.strip()}


def _stage():
    hold, end = (float(v) for v in os.environ.get("HIKING_X2_STAGE", "500,1500").split(","))
    return hold, end


def iteration(env) -> float:
    return float(os.environ.get("HIKING_ITER_OFFSET", "0")) + env.common_step_counter / STEPS_PER_ITERATION


def schedule(it: float, early: float, hold: float, end: float, final: float = 1.0) -> float:
    """`early` until `hold`, linear to `final` at `end`, `final` afterwards."""
    if it <= hold:
        return early
    if it >= end:
        return final
    return early + (final - early) * (it - hold) / (end - hold)


def _s(env, early: float, final: float = 1.0) -> float:
    hold, end = _stage()
    return schedule(iteration(env), early, hold, end, final)


# ---- a: staged rewards (each returns the paper's term times the schedule) ----
def is_alive_staged(env, early_scale: float = 0.3) -> torch.Tensor:
    return mdp.is_alive(env) * _s(env, early_scale)


def action_rate_staged(env, early_scale: float = 0.2) -> torch.Tensor:
    return mdp.action_rate_l2(env) * _s(env, early_scale)


def feet_slide_staged(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg, threshold: float, early_scale: float = 0.2):
    return mdp.contact_slide(env, sensor_cfg=sensor_cfg, asset_cfg=asset_cfg, threshold=threshold) * _s(env, early_scale)


def dof_acc_staged(env, asset_cfg: SceneEntityCfg, early_scale: float = 0.2) -> torch.Tensor:
    return mdp.joint_acc_l2(env, asset_cfg=asset_cfg) * _s(env, early_scale)


def feet_flat_ori_staged(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg, early_scale: float = 0.2) -> torch.Tensor:
    return mdp.feet_orientation_contact(env, sensor_cfg=sensor_cfg, asset_cfg=asset_cfg) * _s(env, early_scale)


# ---- b: tracking ----
def lin_tracking_kernel(cmd_xy: torch.Tensor, vel_xy: torch.Tensor, std: float) -> torch.Tensor:
    return torch.exp(-torch.sum(torch.square(cmd_xy - vel_xy), dim=1) / std**2)


def progress_gate(cmd_x: torch.Tensor, vel_x: torch.Tensor, strength: float) -> torch.Tensor:
    """1 for robots not commanded forward; for the others v_x / (0.5 cmd_x) clipped to 0..1, blended to 1 as strength goes to 0."""
    moving = cmd_x > 0.1
    g = torch.clamp(vel_x / (0.5 * cmd_x.clamp(min=1e-3)), 0.0, 1.0)
    g = 1.0 - strength * (1.0 - g)
    return torch.where(moving, g, torch.ones_like(g))


def track_lin_vel_xy_exp_staged(env, command_name: str, std: float, early_std: float = 0.25,
                                asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    cur = _s(env, early_std, std)
    return lin_tracking_kernel(env.command_manager.get_command(command_name)[:, :2], env.scene[asset_cfg.name].data.root_lin_vel_b[:, :2], cur)


def track_ang_vel_z_exp_gated(env, command_name: str, std: float,
                              asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    cmd = env.command_manager.get_command(command_name)
    data = env.scene[asset_cfg.name].data
    base = torch.exp(-torch.square(cmd[:, 2] - data.root_ang_vel_b[:, 2]) / std**2)
    return base * progress_gate(cmd[:, 0], data.root_lin_vel_b[:, 0], _s(env, 1.0, 0.0))


# ---- c: commands ----
class FarTargetPoseVelocityCommand(PoseVelocityCommand):
    """The paper's command, with targets redrawn (up to 8 times) until they are at least `min_target_distance` from the robot."""

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        env_ids = torch.as_tensor(env_ids, device=self.device)
        min_d = getattr(self.cfg, "min_target_distance", 0.0)
        for _ in range(8):
            d = torch.norm(self.pos_command_w[env_ids, :2] - self.robot.data.root_pos_w[env_ids, :2], dim=1)
            near = env_ids[d < min_d]
            if len(near) == 0:
                break
            ids = torch.randint(0, self.valid_targets.shape[2], size=(len(near),), device=self.device)
            self.pos_command_w[near] = self.valid_targets[self.terrain.terrain_levels[near], self.terrain.terrain_types[near], ids]


@configclass
class FarTargetPoseVelocityCommandCfg(PoseVelocityCommandCfg):
    class_type: type = FarTargetPoseVelocityCommand
    min_target_distance: float = 1.5


# ---- d: not moving ----
def dont_wait_staged(env, command_name: str, early_scale: float = 4.0) -> torch.Tensor:
    return mdp.dont_wait(env, command_name=command_name) * _s(env, early_scale)


def stalled(env, command_name: str, min_speed: float = 0.1, grace_s: float = 4.0,
            asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Commanded to move, yet the average speed since the episode start (distance from the spawn origin / time) is under min_speed
    after grace_s. Off once the schedule ends."""
    hold, end = _stage()
    if iteration(env) >= end:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    t = env.episode_length_buf * env.step_dt
    dist = torch.norm(env.scene[asset_cfg.name].data.root_pos_w[:, :2] - env.scene.env_origins[:, :2], dim=1)
    moving = env.command_manager.get_command(command_name)[:, 0] > 0.3
    return moving & (t > grace_s) & (dist < min_speed * t)


# ---- e: air time ----
def feet_air_time_staged(env, command_name: str, vel_threshold: float, sensor_cfg: SceneEntityCfg, early_scale: float = 4.0):
    return mdp.feet_air_time(env, command_name=command_name, vel_threshold=vel_threshold, sensor_cfg=sensor_cfg) * _s(env, early_scale)



# ---- s: style reward coefficient (patches the trainer's WasabiPPO class; logs the weighted style reward per step) ----
def install_style_schedule(algo_cls, early: float):
    """Before every env step set `discriminator_reward_coef` = base x schedule(iteration, early); iteration = HIKING_ITER_OFFSET
    + env steps since this process started / 24 (the same count as the env's). With early = 1.0 the coefficient is unchanged.
    After the step adds `style_coef` and `style_reward_weighted` (coef x D-reward, what PPO adds to each step's reward, not
    multiplied by dt) to the trainer's per-step log ("Mean step ..."). Installs once per process."""
    if getattr(algo_cls, "_hiking_style_installed", False):
        return
    orig = algo_cls.process_env_step

    def process_env_step(self, rewards, dones, infos, next_obs, next_extra_obs):
        if not hasattr(self, "_hiking_style_base"):
            self._hiking_style_base = float(self.discriminator_reward_coef)
            self._hiking_style_steps = 0
        hold, end = _stage()
        it = float(os.environ.get("HIKING_ITER_OFFSET", "0")) + self._hiking_style_steps / STEPS_PER_ITERATION
        coef = self._hiking_style_base * schedule(it, early, hold, end)
        self.discriminator_reward_coef = coef
        self._hiking_style_steps += 1
        out = orig(self, rewards, dones, infos, next_obs, next_extra_obs)
        step = infos.get("step") if isinstance(infos, dict) else None
        if step is not None and "discriminator_reward" in step:
            step["style_coef"] = torch.tensor(float(coef))
            step["style_reward_weighted"] = coef * step["discriminator_reward"]
        return out

    algo_cls.process_env_step = process_env_step
    algo_cls._hiking_style_installed = True
# ---- end s ----


@configclass
class X2ParkourStagedEnvCfg(X2ParkourTorsoCamEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        opts = _opts()
        unknown = opts - set("abcdes")
        assert not unknown, "unknown HIKING_X2_OPTS: %s" % sorted(unknown)
        from instinct_rl.algorithms.wasabi import WasabiPPO
        install_style_schedule(WasabiPPO, float(os.environ.get("HIKING_X2_STYLE_EARLY", "0.3")) if "s" in opts else 1.0)
        r = self.rewards.rewards
        if "a" in opts:
            r.is_alive = RewTerm(func=is_alive_staged, weight=r.is_alive.weight,
                                 params={"early_scale": float(os.environ.get("HIKING_X2_ALIVE_EARLY", "0.3"))})
            r.action_rate_l2 = RewTerm(func=action_rate_staged, weight=r.action_rate_l2.weight)
            r.feet_slide = RewTerm(func=feet_slide_staged, weight=r.feet_slide.weight, params=dict(r.feet_slide.params))
            r.dof_acc_l2 = RewTerm(func=dof_acc_staged, weight=r.dof_acc_l2.weight, params=dict(r.dof_acc_l2.params))
            r.feet_flat_ori = RewTerm(func=feet_flat_ori_staged, weight=r.feet_flat_ori.weight, params=dict(r.feet_flat_ori.params))
        if "b" in opts:
            r.track_lin_vel_xy_exp = RewTerm(func=track_lin_vel_xy_exp_staged, weight=r.track_lin_vel_xy_exp.weight,
                                             params=dict(r.track_lin_vel_xy_exp.params))
            r.track_ang_vel_z_exp = RewTerm(func=track_ang_vel_z_exp_gated, weight=r.track_ang_vel_z_exp.weight,
                                            params=dict(r.track_ang_vel_z_exp.params))
        if "c" in opts:
            old = self.commands.base_velocity
            new = FarTargetPoseVelocityCommandCfg()
            for k, v in old.__dict__.items():
                if k != "class_type" and hasattr(new, k):
                    setattr(new, k, v)
            new.rel_standing_envs = 0.0
            for name, rng in new.velocity_ranges.items():
                if rng["lin_vel_x"][1] > 0:
                    rng["lin_vel_x"] = (max(rng["lin_vel_x"][0], 0.6), rng["lin_vel_x"][1])
            self.commands.base_velocity = new
        if "d" in opts:
            r.dont_wait = RewTerm(func=dont_wait_staged, weight=r.dont_wait.weight,
                                  params=dict(r.dont_wait.params, early_scale=float(os.environ.get("HIKING_X2_WAIT_EARLY", "4.0"))))
            self.terminations.stalled = DoneTerm(func=stalled, params={"command_name": "base_velocity"})
        if "e" in opts:
            r.feet_air_time = RewTerm(func=feet_air_time_staged, weight=r.feet_air_time.weight, params=dict(r.feet_air_time.params))
