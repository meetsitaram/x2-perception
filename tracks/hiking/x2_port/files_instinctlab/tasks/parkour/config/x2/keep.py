"""Options k (keep terrain levels across restarts) and y (turn in place), added 2026-11-12 for run 4's continuation.

k  `level_keeper` curriculum term. Every HIKING_X2_LEVELS_EVERY_S (300 s) it writes the per-robot terrain levels and types to
   HIKING_X2_LEVELS_OUT (torch file, atomic replace). On its first call it restores levels: from HIKING_X2_LEVELS_IN if that file
   exists and its types equal this run's, else from HIKING_X2_LEVELS_MEANS ("type:mean,..." per terrain type, e.g. read from the
   log at the checkpoint: each robot gets floor(mean) or floor(mean)+1 so the type's mean is kept). After a restore every robot is
   moved at its next natural reset (HIKING_X2_LEVELS_FORCE_RESET=1 times every robot out at once: not used after 1,750). Without these variables it does nothing.
   Logs `levels_restored` (1 after a restore) and `turn/*` (option y).
y  turn in place. At each command resample a share HIKING_X2_TURN_SHARE (0.15) of the robots that are not on the random-velocity
   (stand) column become turners for HIKING_X2_TURN_TIME (4-6 s): their command is [0, 0, clip(2 x bearing error to the target,
   -1, 1)], i.e. the paper's yaw rule with the forward part removed: turn on the spot to face a target at least 1.5 m away, then
   stand facing it. This is what an apparatus station needs (turn to the next waypoint, then walk). `dont_turn_wait` (weight -0.5,
   the paper's dont_wait weight) costs 1 per step while a turner is commanded more than 0.4 rad/s and turns slower than 0.2 rad/s.
   A listed departure from the paper (its only zero-forward yaw commands are on the perlin_rough_stand column, 5% of robots).
Errors in k are printed once and swallowed; errors in y raise (it changes the task).
"""
from __future__ import annotations

import os
import time

import torch

from isaaclab.utils import configclass

from .staged import FarTargetPoseVelocityCommand, FarTargetPoseVelocityCommandCfg

_k = {}


class TurnFarTargetPoseVelocityCommand(FarTargetPoseVelocityCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.is_turn_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _resample(self, env_ids):
        super()._resample(env_ids)
        if len(env_ids) == 0:
            return
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        turn = (torch.rand(len(ids), device=self.device) < self.cfg.turn_share) & ~self.random_velocity_indices[ids]
        self.is_turn_env[ids] = turn
        tid = ids[turn]
        if len(tid):
            self.time_left[tid] = self.time_left[tid].uniform_(*self.cfg.turn_time_range)

    def _update_command(self):
        super()._update_command()
        self.vel_command_b[self.is_turn_env, :2] = 0.0


@configclass
class TurnFarTargetPoseVelocityCommandCfg(FarTargetPoseVelocityCommandCfg):
    class_type: type = TurnFarTargetPoseVelocityCommand
    turn_share: float = 0.15
    turn_time_range: tuple = (4.0, 6.0)


def dont_turn_wait(env, command_name: str) -> torch.Tensor:
    cmd = env.command_manager.get_command(command_name)
    wz = env.scene["robot"].data.root_ang_vel_b[:, 2]
    return ((cmd[:, 0].abs() < 0.1) & (cmd[:, 2].abs() > 0.4) & (wz * torch.sign(cmd[:, 2]) < 0.2)).float()


def _restore(env, t):
    pin = os.environ.get("HIKING_X2_LEVELS_IN", "")
    if pin and os.path.exists(pin):
        d = torch.load(pin, map_location="cpu")
        if d["types"].shape == t.terrain_types.shape and torch.equal(d["types"].to(t.terrain_types.device), t.terrain_types):
            t.terrain_levels[:] = d["levels"].to(t.terrain_levels.device)
            return "file %s (iteration %.0f)" % (pin, d.get("iteration", -1))
        print("[X2 level_keeper] %s does not match this run's terrain types; trying the means" % pin, flush=True)
    means = os.environ.get("HIKING_X2_LEVELS_MEANS", "")
    if not means:
        return None
    from .monitor import _terrain_names
    names, col_type = _terrain_names(env)
    want = {k: float(v) for k, v in (p.split(":") for p in means.split(","))}
    types = t.terrain_types
    for col in range(int(types.max().item()) + 1):
        name = names[int(col_type[col])]
        if name not in want:
            continue
        ids = (types == col).nonzero().flatten()
        m = min(max(want[name], 0.0), float(t.max_terrain_level - 1))
        base = int(m)
        up = torch.rand(len(ids), device=types.device) < (m - base)
        t.terrain_levels[ids] = base + up.long()
    return "means " + means


def level_keeper(env, env_ids):
    try:
        t = env.scene.terrain
        out = {}
        if "init" not in _k:
            _k["init"], _k["last"], _k["restored"] = True, time.time(), 0.0
            how = _restore(env, t)
            if how:
                t.env_origins[:] = t.terrain_origins[t.terrain_levels, t.terrain_types]
                if os.environ.get("HIKING_X2_LEVELS_FORCE_RESET", "0") == "1":  # off: at 1,750 the forced time-out cost every robot a level
                    env.episode_length_buf[:] = env.max_episode_length  # (score of a 1-step episode) and synchronised all resets
                _k["restored"] = 1.0
                _k["pinned"] = t.terrain_levels.clone()
                _k["pin_until"] = env.common_step_counter + int(os.environ.get("HIKING_X2_LEVELS_PIN_STEPS", "1500"))
                print("[X2 level_keeper] restored terrain levels from %s: mean %.2f" % (how, t.terrain_levels.float().mean().item()), flush=True)
        # pin: for HIKING_X2_LEVELS_PIN_STEPS (1,500 control steps) after a restore, the curriculum's moves are undone for every
        # robot that resets (this term runs after `terrain_levels` and before the reset events place the robot). Reason
        # (2026-11-12, restarts at 1,750 and 2,750): after a restore the first episodes end early (robots still on their spawn
        # tiles, synchronised starts), their tracking scores are summed over steps / 1,000, so each such reset moved a robot
        # down a level: 5.42 -> 3.16 mean in 22 iterations with no change to the task.
        if "pinned" in _k and env_ids is not None:
            if env.common_step_counter < _k["pin_until"]:
                ids = torch.as_tensor(env_ids, device=t.terrain_levels.device, dtype=torch.long)
                t.terrain_levels[ids] = _k["pinned"][ids]
                t.env_origins[ids] = t.terrain_origins[t.terrain_levels[ids], t.terrain_types[ids]]
            else:
                del _k["pinned"]
        # every key on every call: instinct_rl's logger keys its means on the first step's dict and raises KeyError when a key
        # goes missing (12:27 and 12:42 crashes, when the pin released and "levels_pinned" disappeared)
        out["levels_pinned"] = 1.0 if "pinned" in _k else 0.0
        pout = os.environ.get("HIKING_X2_LEVELS_OUT", "")
        if pout and time.time() - _k["last"] > float(os.environ.get("HIKING_X2_LEVELS_EVERY_S", "300")):
            _k["last"] = time.time()
            from .staged import iteration
            torch.save({"levels": t.terrain_levels.cpu(), "types": t.terrain_types.cpu(), "iteration": iteration(env),
                        "time": time.time()}, pout + ".tmp")
            os.replace(pout + ".tmp", pout)
        out["levels_restored"] = _k["restored"]
        cmd_term = env.command_manager.get_term("base_velocity")
        if hasattr(cmd_term, "is_turn_env"):
            te = cmd_term.is_turn_env
            cmd = env.command_manager.get_command("base_velocity")
            act = te & (cmd[:, 2].abs() > 0.4)
            out["turn/share"] = te.float().mean().item()
            if act.any():
                wz = env.scene["robot"].data.root_ang_vel_b[act, 2]
                vxy = torch.norm(env.scene["robot"].data.root_lin_vel_b[act, :2], dim=1)
                _k["turn"] = (cmd[act, 2].abs().mean().item(), (wz * torch.sign(cmd[act, 2])).mean().item(), vxy.mean().item())
            out["turn/cmd_abs"], out["turn/rate_along_cmd"], out["turn/planar_speed"] = _k.get("turn", (0.0, 0.0, 0.0))  # last value when none active
        # routes (option w): share of robots on a route, share of those that reached their final station, mean waypoint index
        if hasattr(cmd_term, "on_route"):
            o = cmd_term.on_route
            _k["route"] = (o.float().mean().item(), cmd_term.route_done[o].float().mean().item() if o.any() else _k.get("route", (0, 0, 0))[1],
                           cmd_term.route_i[o].float().mean().item() if o.any() else _k.get("route", (0, 0, 0))[2])
            out["route/share"], out["route/final_reached"], out["route/waypoint_index"] = _k["route"]
        # artefact watch (2026-11-12, user: "Could isaacsim training also have such issues that we are not seeing?"), per group
        # (each new tile type, "paper" for the paper's columns): share of robots with any body force over 3 x / 10 x body weight
        # (1,240 / 4,120 N), the largest force, share of feet in contact more than 1 cm below the ground under them (foot
        # height scanners), and share of robots in their first 0.2 s with a non-foot contact over 50 N (spawn check)
        AGROUPS = ["paper", "ext_single_beams", "ext_beam_fields", "ext_blocks", "ext_wedges", "ext_tight_spots", "ext_log_corridors",
                   "ext_homes", "ext_street_front", "ext_stairs", "perlin_rough_velocity"]
        try:
            if "agroup_of_col" not in _k:
                from .monitor import _terrain_names
                names, col_type = _terrain_names(env)
                g = [AGROUPS.index(names[int(c)]) if names[int(c)] in AGROUPS else 0 for c in col_type.tolist()]
                _k["agroup_of_col"] = torch.tensor(g, device=t.terrain_types.device)
                cs = env.scene.sensors["contact_forces"]
                _k["feet_ids"] = [cs.body_names.index(n) for n in ("left_ankle_roll_link", "right_ankle_roll_link")]
                _k["weight"] = float(env.scene["robot"].data.default_mass[0].sum()) * 9.81
            cs = env.scene.sensors["contact_forces"]
            f = torch.norm(cs.data.net_forces_w, dim=-1)                       # (N, bodies)
            fmax = f.max(dim=1)[0]
            other = f.clone(); other[:, _k["feet_ids"]] = 0.0
            robot = env.scene["robot"]
            foot_z = robot.data.body_pos_w[:, robot.find_bodies(["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True)[0], 2] - 0.073
            ground = torch.stack([env.scene.sensors[n].data.ray_hits_w[..., 2].max(dim=1)[0] for n in ("left_height_scanner", "right_height_scanner")], 1)
            ground = torch.nan_to_num(ground, nan=-1e3, posinf=-1e3, neginf=-1e3)
            sunk = ((ground - foot_z) > 0.01) & (f[:, _k["feet_ids"]] > 1.0)
            early = env.episode_length_buf < 10
            grp = _k["agroup_of_col"][t.terrain_types]
            for gi, gname in enumerate(AGROUPS):
                sel = grp == gi
                if sel.any():
                    _k["a3_" + gname] = (fmax[sel] > 3 * _k["weight"]).float().mean().item()
                    _k["a10_" + gname] = (fmax[sel] > 10 * _k["weight"]).float().mean().item()
                    _k["amax_" + gname] = max(_k.get("amax_" + gname, 0.0), fmax[sel].max().item())  # running max since start
                    _k["sunk_" + gname] = sunk[sel].float().mean().item()
                    se = sel & early
                    if se.any():
                        _k["spawn_" + gname] = (other[se].max(dim=1)[0] > 50.0).float().mean().item()
        except Exception as e:
            if not _k.get("artefact_warned"):
                _k["artefact_warned"] = True
                print("[X2 level_keeper] artefact log off: %r" % (e,), flush=True)
        for gname in AGROUPS:
            for p in ("a3_", "a10_", "amax_", "sunk_", "spawn_"):
                out["artefact/" + p + gname] = _k.get(p + gname, 0.0)
        # v4j logs (option g, contact.py): episode ends and route arrivals per tile group, forces per body group, tunnelling,
        # route robots' speed near tall props
        from .staged import _opts
        if "g" in _opts() and "agroup_of_col" in _k:
            try:
                from .contact import log_v4j
                log_v4j(env, env_ids, out, _k, AGROUPS, _k["agroup_of_col"][t.terrain_types])
            except Exception as e:
                if not _k.get("v4j_warned"):
                    _k["v4j_warned"] = True
                    print("[X2 level_keeper] v4j log off: %r" % (e,), flush=True)
        # posture watch (2026-11-12, user: "the torso seems to lean backward while walking"): torso_link forward pitch (deg,
        # negative = leaning back) and waist pitch (deg) of robots commanded forward, by terrain group; last value kept
        GROUPS = {"flat": ("perlin_rough",), "stairs": ("pyramid_stairs", "pyramid_stairs_high", "pyramid_stairs_inv", "pyramid_stairs_inv_high"),
                  "boxes": ("boxes", "mesh_boxes"), "gaps": ("square_gaps",)}
        try:
            robot = env.scene["robot"]
            if "torso_id" not in _k:
                from .monitor import _terrain_names
                names, col_type = _terrain_names(env)
                _k["torso_id"] = robot.find_bodies("torso_link")[0][0]
                _k["waist_id"] = robot.find_joints("waist_pitch_joint")[0][0]
                _k["group_cols"] = {g: torch.isin(col_type, torch.tensor([names.index(n) for n in ns_ if n in names], device=col_type.device)).nonzero().flatten()
                                    for g, ns_ in GROUPS.items()}
            q = robot.data.body_quat_w[:, _k["torso_id"]]
            pitch = torch.rad2deg(torch.asin(torch.clamp(2 * (q[:, 0] * q[:, 2] - q[:, 1] * q[:, 3]), -1, 1)))
            waist = torch.rad2deg(robot.data.joint_pos[:, _k["waist_id"]])
            mov = env.command_manager.get_command("base_velocity")[:, 0] > 0.3
            for g, cols in _k["group_cols"].items():
                sel = mov & torch.isin(t.terrain_types, cols.to(t.terrain_types.device))
                if sel.any():
                    _k["pt_" + g] = pitch[sel].mean().item()
                    _k["wp_" + g] = waist[sel].mean().item()
        except Exception as e:
            if not _k.get("posture_warned"):
                _k["posture_warned"] = True
                print("[X2 level_keeper] posture log off: %r" % (e,), flush=True)
        for g in GROUPS:
            out["posture/torso_pitch_" + g] = _k.get("pt_" + g, 0.0)
            out["posture/waist_pitch_" + g] = _k.get("wp_" + g, 0.0)
        _k["keys"] = list(out)
        return out
    except Exception as e:  # logging and level keeping must never stop training
        if not _k.get("warned"):
            _k["warned"] = True
            print("[X2 level_keeper] disabled after error: %r" % (e,), flush=True)
        return {k: 0.0 for k in _k.get("keys", [])}


def stop_motion(env, command_name: str, cmd_eps: float = 0.1) -> torch.Tensor:
    """Option z (2026-11-12): planar root speed (m/s) while the command is a stand (|v cmd| < 0.1 m/s and |yaw cmd| < 0.1 rad/s).
    The paper's terms do not charge motion under a zero command: `stand_still` is a joint-pose term (sum |q - q0| - 4) and the
    speed kernel exp(-v^2 / 0.25) keeps 70% at 0.3 m/s. Weight HIKING_X2_STOP_WEIGHT (default -2.0): a 0.5 m/s coast for 1 s
    costs 2 x 0.02 x 0.5 x 50 = 1.0."""
    cmd = env.command_manager.get_command(command_name)
    still = torch.norm(cmd[:, :2], dim=1) < cmd_eps
    if os.environ.get("HIKING_X2_STOP_ANY_YAW", "0") != "1":  # default: stand commands only; 1: also turn-in-place commands
        still = still & (cmd[:, 2].abs() < cmd_eps)
    v = torch.norm(env.scene["robot"].data.root_lin_vel_b[:, :2], dim=1)
    return v * still.float()


def overspeed(env, command_name: str, min_cmd: float = 0.1) -> torch.Tensor:
    """Option o (2026-11-12): forward speed above the forward command, m/s, for robots commanded forward (> 0.1 m/s); 0 at or
    under the command, 0 for stand and turn commands. Weight HIKING_X2_OVERSPEED_WEIGHT (default -2.0, times dt by the manager).
    Reason: the speed kernel exp(-e^2 / 0.25) makes overspeed cheap (+0.24 m/s keeps 79%), and the touchdown air-time reward
    pays longer, faster steps; at 3,125 the policy made 0.30 / 0.47 / 0.58 m/s at 0.2 / 0.35 / 0.5 and stopped 0.47 m."""
    cmd = env.command_manager.get_command(command_name)[:, 0]
    vx = env.scene["robot"].data.root_lin_vel_b[:, 0]
    return torch.clamp(vx - cmd, min=0.0) * (cmd > min_cmd).float()
