"""Option v (v4k, 2026-11-12 21:30): slow forward, slow yaw and a still turn, after the eval of model_4750.
Evidence: on the apparatus legs 3 and 4 (44 and 25 stalls of 50) the robot stood inside the station disc while commanded a stand
turn of 0.1-0.5 rad/s; leg 5 stalls at vx 0.7 below the stairs. Training never asks for yaw under 0.4 rad/s with a cost for not
turning (`dont_turn_wait` needs |cmd| > 0.4) and never asks for forward speed under 0.45 m/s (VX_MIN, route vmax floor). Turners
drift 0.20 m/s (train.log turn/planar_speed); in the eval's turn_in_place all 38 failures ended with the heading within 1.4 deg
but 0.23-0.39 m off the station.
Per robot, drawn at each command resample (robots on the paper's random-velocity columns are left alone):
  - share HIKING_X2_SLOW_FWD_SHARE (0.25): forward command capped at U(HIKING_X2_SLOW_VX = 0.1, 0.45) m/s (route robots too:
    the user wants slow, careful final approaches; `overspeed` (o) already charges speed above the command);
  - share HIKING_X2_SLOW_TURN_SHARE (0.4): |yaw command| capped at U(0.1, 0.4) rad/s (turners and walkers).
Rewards: `turn_effort` replaces `dont_turn_wait`: 1 per step while the command is a turn (|vx| < 0.1, |yaw| > 0.1) and the robot
turns slower than half the command (weight -0.5, as before); `turn_drift`: planar speed (m/s) under the same turn command (weight
HIKING_X2_TURN_DRIFT_WEIGHT, -2.0, on top of `stop_motion`'s -2.0 with STOP_ANY_YAW=1).
Metrics (Metrics/base_velocity/...): slow_fwd_ratio (forward speed / capped command), slow_turn_ratio (yaw rate / capped command),
turn_drift (planar speed while turning)."""
import os

import torch
from isaaclab.utils import configclass

from .routes import RouteFarTargetPoseVelocityCommand, RouteFarTargetPoseVelocityCommandCfg

FWD_SHARE = float(os.environ.get("HIKING_X2_SLOW_FWD_SHARE", "0.25"))
TURN_SHARE = float(os.environ.get("HIKING_X2_SLOW_TURN_SHARE", "0.4"))
VX_LO = float(os.environ.get("HIKING_X2_SLOW_VX", "0.1"))


def cap_commands(vel, vcap, wcap):
    """vel (N, 3) body command; vcap, wcap (N,) caps (inf = none). Forward part capped from above (backward untouched), |yaw| capped."""
    out = vel.clone()
    out[:, 0] = torch.where(out[:, 0] > 0, torch.minimum(out[:, 0], vcap), out[:, 0])
    out[:, 2] = torch.sign(out[:, 2]) * torch.minimum(out[:, 2].abs(), wcap)
    return out


def _ratio(actual, cmd, mask):
    r = torch.where(mask, actual / torch.clamp(cmd, min=1e-3), torch.zeros_like(actual))
    return torch.clamp(r, -2.0, 3.0)


class SlowRouteFarTargetPoseVelocityCommand(RouteFarTargetPoseVelocityCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        inf = torch.full((self.num_envs,), float("inf"), device=self.device)
        self.vcap, self.wcap = inf.clone(), inf.clone()
        for k in ("slow_fwd_ratio", "slow_turn_ratio", "turn_drift"):
            self.metrics[k] = torch.zeros(self.num_envs, device=self.device)

    def _resample(self, env_ids):
        super()._resample(env_ids)
        if len(env_ids) == 0:
            return
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        n = len(ids)
        free = ~self.random_velocity_indices[ids]
        f = (torch.rand(n, device=self.device) < FWD_SHARE) & free
        w = (torch.rand(n, device=self.device) < TURN_SHARE) & free
        inf = torch.full((n,), float("inf"), device=self.device)
        self.vcap[ids] = torch.where(f, torch.empty(n, device=self.device).uniform_(VX_LO, 0.45), inf)
        self.wcap[ids] = torch.where(w, torch.empty(n, device=self.device).uniform_(0.1, 0.4), inf)

    def _update_command(self):
        super()._update_command()
        self.vel_command_b[:] = cap_commands(self.vel_command_b, self.vcap, self.wcap)
        c = self.vel_command_b
        vb = self.robot.data.root_lin_vel_b
        wz = self.robot.data.root_ang_vel_b[:, 2]
        fm = torch.isfinite(self.vcap) & (c[:, 0] > 0.05)
        tm = torch.isfinite(self.wcap) & (c[:, 2].abs() > 0.05)
        turning = (torch.norm(c[:, :2], dim=1) < 0.1) & (c[:, 2].abs() > 0.1)
        a = 0.02  # running mean over about one second
        self.metrics["slow_fwd_ratio"].lerp_(_ratio(vb[:, 0], c[:, 0], fm), a)
        self.metrics["slow_turn_ratio"].lerp_(_ratio(wz * torch.sign(c[:, 2]), c[:, 2].abs(), tm), a)
        self.metrics["turn_drift"].lerp_(torch.norm(vb[:, :2], dim=1) * turning.float(), a)


@configclass
class SlowRouteFarTargetPoseVelocityCommandCfg(RouteFarTargetPoseVelocityCommandCfg):
    class_type: type = SlowRouteFarTargetPoseVelocityCommand


def turn_effort(env, command_name: str) -> torch.Tensor:
    cmd = env.command_manager.get_command(command_name)
    wz = env.scene["robot"].data.root_ang_vel_b[:, 2]
    turn = (torch.norm(cmd[:, :2], dim=1) < 0.1) & (cmd[:, 2].abs() > 0.1)
    return (turn & (wz * torch.sign(cmd[:, 2]) < 0.5 * cmd[:, 2].abs())).float()


def turn_drift(env, command_name: str) -> torch.Tensor:
    cmd = env.command_manager.get_command(command_name)
    turn = (torch.norm(cmd[:, :2], dim=1) < 0.1) & (cmd[:, 2].abs() > 0.1)
    return torch.norm(env.scene["robot"].data.root_lin_vel_b[:, :2], dim=1) * turn.float()
