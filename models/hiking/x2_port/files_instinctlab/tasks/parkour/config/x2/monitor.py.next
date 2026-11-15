"""Log-only signals for the X2 run (no effect on rewards, observations or the terrain curriculum).

Registered as a curriculum term because curriculum terms are the one manager whose returned dict is printed every iteration
("Mean episode Curriculum/walk_monitor/<name>"). The values are a snapshot over ALL robots at the moment some robots reset
(every control step has resets with thousands of robots), averaged by the trainer over the iteration.

  cmd_speed            mean commanded forward speed, robots commanded to move (> 0.1 m/s)
  actual_speed         mean forward speed in the root frame of those same robots
  speed_ratio          actual_speed / cmd_speed
  moving_cmd_share     share of robots commanded to move
  double_support       share of moving-commanded robots with both feet in contact (> 1 N vertical)
  no_support           share with neither foot in contact
  step_rate_hz         touchdowns per robot per second (both feet together), moving-commanded robots
  air_per_touchdown_s  measured: mean air time of the swing that ended, over the touchdowns at this step (moving-commanded)
  touchdown_air_ge_0p2 share of those touchdowns whose swing lasted at least 0.2 s
  level/<terrain type> mean terrain row (0-9) of the robots on that terrain type
Any error here is swallowed (printed once): logging must never stop training.
"""
import numpy as np
import torch

_state = {}


def _terrain_names(env):
    terrain = env.scene.terrain
    gen = terrain.cfg.terrain_generator
    names = list(gen.sub_terrains.keys())
    prop = np.array([c.proportion for c in gen.sub_terrains.values()], dtype=float)
    cum = np.cumsum(prop / prop.sum())
    cols = gen.num_cols
    # same rule as the terrain generator's curriculum layout: column -> sub-terrain by cumulative proportion
    col_type = [int(np.min(np.where(i / cols + 0.001 < cum)[0])) for i in range(cols)]
    return names, torch.tensor(col_type, device=env.device)


def walk_monitor(env, env_ids, command_name: str = "base_velocity", sensor_name: str = "contact_forces"):
    try:
        robot = env.scene["robot"]
        sensor = env.scene[sensor_name]
        if "feet" not in _state:
            _state["feet"] = sensor.find_bodies(".*_ankle_roll_link")[0]
            _state["names"], _state["col_type"] = _terrain_names(env)
        cmd = env.command_manager.get_command(command_name)
        moving = cmd[:, 0] > 0.1
        n = moving.sum().clamp(min=1)
        vx = robot.data.root_lin_vel_b[:, 0]
        cmd_speed = (cmd[:, 0] * moving).sum() / n
        actual = (vx * moving).sum() / n
        contact = sensor.data.net_forces_w[:, _state["feet"], 2] > 1.0
        both = contact.all(dim=1) & moving
        none = (~contact).all(dim=1) & moving
        first = sensor.compute_first_contact(env.step_dt)[:, _state["feet"]].float()
        touchdown = first.sum(dim=1)
        last_air = sensor.data.last_air_time[:, _state["feet"]]
        td = first * moving.unsqueeze(1).float()
        n_td = td.sum()
        out = {
            "cmd_speed": cmd_speed.item(),
            "actual_speed": actual.item(),
            "speed_ratio": (actual / cmd_speed.clamp(min=1e-3)).item(),
            "moving_cmd_share": moving.float().mean().item(),
            "double_support": (both.sum() / n).item(),
            "no_support": (none.sum() / n).item(),
            "step_rate_hz": ((touchdown * moving).sum() / n / env.step_dt).item(),
            "air_per_touchdown_s": ((last_air * td).sum() / n_td).item() if n_td > 0 else float("nan"),
            "touchdown_air_ge_0p2": (((last_air >= 0.2).float() * td).sum() / n_td).item() if n_td > 0 else float("nan"),
        }
        terrain = env.scene.terrain
        types = _state["col_type"][terrain.terrain_types]
        levels = terrain.terrain_levels.float()
        for i, name in enumerate(_state["names"]):
            m = types == i
            if m.any():
                out["level/" + name] = levels[m].mean().item()
        return out
    except Exception as e:  # noqa: BLE001
        if not _state.get("warned"):
            _state["warned"] = True
            print("[X2 walk_monitor] disabled after error: %r" % (e,), flush=True)
        return {}
