"""Reproduce Tables 1 and 2 of "Hiking in the Wild" (arXiv 2601.07718 v2) on the released G1 task.

The authors did not release evaluation code. Every choice the paper leaves open is listed in docs/EVAL_NOTES.md.
NOT YET RUN: this file has only been exercised through its CPU-only unit tests.

Layout: the parent process (no Isaac Sim, no GPU) launches one headless worker per terrain, sequentially, because Isaac Sim
can only own one simulation per process. Each worker launches Isaac Sim and writes <out>.parts/<terrain>.json. Only one GPU
job may run at a time, so do not start this while training runs.

Usage:
  python scripts/eval_paper.py --checkpoint PATH [--num_envs 1000] [--terrains stair_up,gap] [--out results.json]
Anything else on the command line (for example --device cuda:0) is passed through to AppLauncher in the workers.
"""
import argparse
import copy
import json
import math
import os
import re
import subprocess
import sys
import time
import traceback

# numpy and stdlib only at module level: torch and isaaclab must be imported after mujoco and AppLauncher in the worker,
# and the unit tests import this module without touching the GPU.
import numpy as np

ROOT = "/mnt/filesystem-g4/tracks/hiking/code/hiking-in-the-wild"
PAPER_TXT = "/mnt/filesystem-g4/tracks/hiking/code/x2_port/paper_v2.txt"
TASK = "Instinct-Parkour-Target-Amp-X2-v0"
SEPARATOR = "=" * 110

# Appendix C of the paper.
SUCCESS_RADIUS_M = 0.25
TRIAL_S = 15.0
# Paper A.5: "target radius r = 0.2". The training config uses 0.4 (parkour_env_cfg.py:649), which would zero the velocity
# command 0.15 m outside the success radius. See docs/EVAL_NOTES.md item 6.
DEFAULT_STOP_RADIUS_M = 0.2
TABLE1_STEPS = 10_000
# A flat-patch target this close to the spawn point would succeed at t=0. See docs/EVAL_NOTES.md item 5.
MIN_START_DIST_M = 0.5

# name -> (code sub-terrain key from parkour_env_cfg.py, overrides giving Table 8 geometry, paper's column title).
# Direction: the robot spawns at the tile centre, so a non-inverted pyramid (centre high) is walked DOWN and an inverted one UP.
# Table 7 lists them in this same order (Stairs Down 15%, Platform Down 10%, Stairs Up 15%, Platform Up 10%).
# Audit item: slope_range is rise over run, not radians (hf_terrains.py:79-90, height_max = slope * size / 2), so the evaluated
# ramp uses tan(0.40) = 0.4228 to be a true 0.40 rad ramp. (Rounding to the vertical scale gives about 0.399 rad.)
TERRAINS = {
    "large_box": ("boxes", {"obstacle_height_range": (0.40, 0.40)}, "Large Box"),
    "ramp": ("hf_pyramid_slope_inv", {"slope_range": (math.tan(0.40), math.tan(0.40))}, "Ramp"),
    "small_box": (
        "mesh_boxes",
        {
            "box_height_mean": 0.30,
            "box_height_range": 0.05,
            "box_length_mean": 0.40,
            "box_length_range": 0.10,
            "box_width_mean": 0.40,
            "box_width_range": 0.10,
        },
        "Small Box",
    ),
    "rough": ("perlin_rough", {"noise_scale": 0.05}, "Rough"),
    "stair_down": ("pyramid_stairs", {"step_height_range": (0.20, 0.20)}, "Stair Down"),
    "platform_down": ("pyramid_stairs_high", {"step_height_range": (0.40, 0.40)}, "Platform down"),
    "stair_up": ("pyramid_stairs_inv", {"step_height_range": (0.20, 0.20)}, "Stair Up"),
    "platform_up": ("pyramid_stairs_inv_high", {"step_height_range": (0.40, 0.40)}, "Platform up"),
    "gap": ("square_gaps", {"gap_distance_range": (0.60, 0.60)}, "Gap"),
}
# Paper Table 1 rows, mapped to evaluated terrains (see docs/EVAL_NOTES.md item 11 for Deep Gap).
TABLE1 = {"stair_up": "Stair Up", "stair_down": "Stair Down", "gap": "Deep Gap", "small_box": "Small Box"}


def load_paper_tables(path=PAPER_TXT):
    """Read the paper's "Ours" rows from the v2 text. Returns {"table2": {terrain: (sr, time)}, "table1": {terrain: (sr, landing)}}."""
    text = open(path, encoding="utf8", errors="replace").read()
    ours = re.search(r"\bOurs((?:\s+[\d.]+){18})\b", text)
    if ours is None:
        raise RuntimeError("Could not find the 'Ours' row of Table 2 in " + path)
    values = [float(v) for v in ours.group(1).split()]
    names = list(TERRAINS)
    table2 = {t: (values[i], values[i + len(names)]) for i, t in enumerate(names)}
    table1 = {}
    for terrain, title in TABLE1.items():
        m = re.search(re.escape(title) + r"\s+([\d.]+)%\s+[\d.]+%\s+([\d.]+)\s+[\d.]+", text)
        if m is None:
            raise RuntimeError("Could not find Table 1 row %r in %s" % (title, path))
        table1[terrain] = (float(m.group(1)), float(m.group(2)))
    return {"table2": table2, "table1": table1}


class TrialTracker:
    """Counts trials one env at a time from the (done, distance) signal after each env.step.

    Isaac Lab resets a done env inside step(), so after the step its robot, target and contact state belong to the NEW episode.
    A done on this step therefore closes the trial and its distance is never looked at. A trial is tallied exactly once, when
    it closes: termination, or the deadline. Success is latched at the first reach, so termination after reaching keeps the
    success (a stricter count is kept in n_reached_then_terminated). A trial that closed at the deadline waits for its env's next
    reset before the next trial starts, and with max_trials the env stops being counted. Trials still open at the end are dropped.
    """

    def __init__(self, num_envs, horizon_steps, radius=SUCCESS_RADIUS_M, max_trials=None):
        self.horizon = int(horizon_steps)
        self.radius = float(radius)
        self.max_trials = max_trials
        self.open = np.ones(num_envs, dtype=bool)
        self.steps = np.zeros(num_envs, dtype=np.int64)
        self.reach_step = np.full(num_envs, -1, dtype=np.int64)
        self.trials_closed = np.zeros(num_envs, dtype=np.int64)
        self.n_trials = 0
        self.n_success = 0
        self.n_terminated = 0  # closed by a done before reaching
        self.n_timed_out = 0  # still not reached at the deadline
        self.n_reached_then_terminated = 0  # reached, then a done before the deadline; still a success
        self.sum_reach_steps = 0

    def update(self, done, dist):
        done = np.asarray(done, dtype=bool)
        dist = np.asarray(dist, dtype=float)
        running = self.open.copy()
        self.steps[running] += 1
        reach_now = running & ~done & (self.reach_step < 0) & (dist <= self.radius) & (self.steps <= self.horizon)
        self.reach_step[reach_now] = self.steps[reach_now]
        close_done = running & done
        close_deadline = running & ~done & (self.steps >= self.horizon)
        self._close(close_done, terminated=True)
        self._close(close_deadline, terminated=False)
        # An env that reset on this step begins its next trial on the following step, unless it has used its trials.
        restart = done & ~self.open
        if self.max_trials is not None:
            restart &= self.trials_closed < self.max_trials
        self.open[restart] = True
        self.steps[restart] = 0
        self.reach_step[restart] = -1

    def _close(self, mask, terminated):
        if not mask.any():
            return
        reached = mask & (self.reach_step >= 0)
        n = int(mask.sum())
        self.n_trials += n
        self.trials_closed[mask] += 1
        self.n_success += int(reached.sum())
        self.sum_reach_steps += int(self.reach_step[reached].sum())
        if terminated:
            self.n_terminated += n - int(reached.sum())
            self.n_reached_then_terminated += int(reached.sum())
        else:
            self.n_timed_out += n - int(reached.sum())
        self.open[mask] = False

    @property
    def finished(self):
        return self.max_trials is not None and bool((self.trials_closed >= self.max_trials).all())

    def summary(self, dt):
        success_pct = 100.0 * self.n_success / self.n_trials if self.n_trials else None
        mean_time = dt * self.sum_reach_steps / self.n_success if self.n_success else None
        strict = 100.0 * (self.n_success - self.n_reached_then_terminated) / self.n_trials if self.n_trials else None
        return {
            "n_trials": self.n_trials,
            "n_success": self.n_success,
            "success_rate_pct": success_pct,
            "mean_reaching_time_s": mean_time,
            "n_terminated_before_reach": self.n_terminated,
            "n_timed_out": self.n_timed_out,
            "n_reached_then_terminated": self.n_reached_then_terminated,
            "strict_success_rate_pct": strict,
        }


def filter_near_targets(patches, origins_xy, min_dist, rng):
    """Replace flat-patch targets closer than min_dist to the spawn point by another valid patch of the same tile.

    patches: (rows, cols, num_patches, 3). origins_xy: (rows, cols, 2) or (rows, cols, 3). Returns (patches, n_replaced, n_tiles_without_valid).
    """
    patches = np.array(patches, copy=True)
    rows, cols, num_patches = patches.shape[:3]
    n_replaced = n_empty = 0
    for r in range(rows):
        for c in range(cols):
            near = np.linalg.norm(patches[r, c, :, :2] - origins_xy[r, c, :2], axis=1) < min_dist
            if not near.any():
                continue
            valid = np.where(~near)[0]
            if valid.size == 0:
                n_empty += 1
                continue
            n_replaced += int(near.sum())
            patches[r, c, near] = patches[r, c, rng.choice(valid, size=int(near.sum()))]
    return patches, n_replaced, n_empty


def interaction_mask(heights, threshold):
    """True where the terrain under the base, within the scanner footprint, spans more than `threshold`. heights: (envs, rays)."""
    valid = np.isfinite(heights)
    top = np.where(valid, heights, -np.inf).max(axis=1)
    bottom = np.where(valid, heights, np.inf).min(axis=1)
    return valid.any(axis=1) & ((top - bottom) > threshold)


def landing_fractions(sole_z, hit_z, tol):
    """Fraction of sole rays with terrain within `tol` of the sole. sole_z: (n,), hit_z: (n, rays). A missed ray is unsupported."""
    supported = np.isfinite(hit_z) & ((sole_z[:, None] - hit_z) <= tol)
    return supported.mean(axis=1)


def landing_step(first_contact, done, gate, sole_z, hit_z, tol):
    """Sum and count of landing fractions for feet that touched down this step inside the interaction range. Skips reset envs."""
    select = np.asarray(first_contact, bool) & ~np.asarray(done, bool) & np.asarray(gate, bool)
    if not select.any():
        return 0.0, 0
    return float(landing_fractions(sole_z[select], hit_z[select], tol).sum()), int(select.sum())


def apply_eval_overrides(env_cfg, terrain, stop_radius, seed):
    """Reduce the training config to the evaluation protocol. Attribute access only, so it is unit-tested with a fake config."""
    key, overrides, _ = TERRAINS[terrain]
    env_cfg.seed = seed
    # Time out one control step after the 15 s deadline so a robot's state at exactly t=15 s is still visible (not reset).
    dt = env_cfg.sim.dt * env_cfg.decimation
    env_cfg.episode_length_s = TRIAL_S + 0.5 * dt
    # Terrain: one Table 8 sub-terrain on the training grid. curriculum=True is the tile layout, not the terrain-level curriculum.
    gen = env_cfg.scene.terrain.terrain_generator
    sub = copy.deepcopy(gen.sub_terrains[key])
    sub.proportion = 1.0
    sub.wall_prob = [0.0, 0.0, 0.0, 0.0]  # as G1ParkourRoughEnvCfg_PLAY
    for name, value in overrides.items():
        setattr(sub, name, value)
    gen.sub_terrains = {key: sub}
    gen.seed = seed
    gen.use_cache = False
    env_cfg.scene.terrain.max_init_terrain_level = None  # spread robots over all rows
    # Terrain-level curriculum off.
    env_cfg.curriculum.terrain_levels = None
    # Command: same PoseVelocityCommand and flat-patch targets, one target per trial, no standing envs.
    cmd = env_cfg.commands.base_velocity
    cmd.resampling_time_range = (1.0e9, 1.0e9)
    cmd.rel_standing_envs = 0.0
    cmd.random_velocity_terrain = None
    cmd.velocity_ranges = {key: cmd.velocity_ranges[key]}
    cmd.target_dis_threshold = stop_radius
    cmd.debug_vis = False
    # Training-only randomization, removed exactly as G1ParkourRoughEnvCfg_PLAY does. Base pose/velocity jitter at reset stays,
    # as in PLAY. There are no push events in this task's EventCfg. Terminations are kept unchanged (PLAY drops root_height; we must not).
    env_cfg.events.physics_material = None
    env_cfg.events.reset_robot_joints.params = {"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)}
    return env_cfg


def format_cell(value, digits=2, missing="n/a"):
    return missing if value is None else ("%.*f" % (digits, value))


def format_table2(paper, ours):
    """The paper's Table 2 layout with its 'Ours' row beside this run. '-' = no success (paper's notation), n/a = not run."""
    names = list(TERRAINS)
    titles = [TERRAINS[t][2] for t in names]
    def row(label, cells):
        return "  ".join([label.ljust(16)] + [c.rjust(max(len(titles[i]), 7)) for i, c in enumerate(cells)])
    lines = ["Table 2: Quantitative results in simulation (%d terrains)" % len(names)]
    for title, column, digits, unit in (("Success Rate (%)", 0, 2, "success_rate_pct"), ("Mean Reaching Time (s)", 1, 2, "mean_reaching_time_s")):
        lines.append(title)
        lines.append(row("Method", titles))
        lines.append(row("Ours (paper)", [format_cell(paper["table2"][t][column], digits) for t in names]))
        cells = []
        for t in names:
            r = (ours.get(t) or {}).get("table2")
            if r is None:
                cells.append("n/a")
            elif unit == "mean_reaching_time_s" and r["n_success"] == 0:
                cells.append("-")
            else:
                cells.append(format_cell(r[unit], digits))
        lines.append(row("This run", cells))
    return "\n".join(lines)


def format_table1(paper, ours):
    lines = ["Table 1: Edge penalization in simulation (Ours only; No Edge and Real SR are not reproduced here)"]
    lines.append("%-12s %16s %16s %18s %18s" % ("Terrain Type", "Sim SR Ours(paper)", "Sim SR this run", "Landing Ours(paper)", "Landing this run"))
    for t, title in TABLE1.items():
        p_sr, p_land = paper["table1"][t]
        r = (ours.get(t) or {}).get("table1")
        sr = "n/a" if r is None else format_cell(r["success_rate_pct"], 2)
        land = "n/a" if r is None else format_cell(r["landing_area_fraction"], 2, missing="-")
        lines.append("%-12s %16.2f %16s %18.2f %18s" % (title, p_sr, sr, p_land, land))
    return "\n".join(lines)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="Path of the model_*.pt checkpoint to evaluate.")
    p.add_argument("--num_envs", type=int, default=1000, help="Parallel robots per terrain (paper: 1000).")
    p.add_argument("--terrains", default=",".join(TERRAINS), help="Comma list of: " + ", ".join(TERRAINS))
    p.add_argument("--out", default="eval_paper_results.json", help="Path of the output JSON file.")
    p.add_argument("--trials_per_env", type=int, default=1, help="Table 2 trials counted per robot (see docs/EVAL_NOTES.md item 4).")
    p.add_argument("--table1_steps", type=int, default=TABLE1_STEPS, help="Table 1 policy steps (paper: 10,000).")
    p.add_argument("--stop_radius", type=float, default=DEFAULT_STOP_RADIUS_M, help="Distance at which the velocity command drops to zero.")
    p.add_argument("--landing_tol", type=float, default=0.05, help="Sole-to-terrain clearance still counted as supported (m).")
    p.add_argument("--interaction_threshold", type=float, default=0.15, help="Terrain height span under the base scanner counted as an obstacle (m).")
    p.add_argument("--seed", type=int, default=42, help="Seed for the env, terrain layout and target filtering.")
    p.add_argument("--worker_timeout", type=float, default=4 * 3600.0, help="Kill a terrain worker after this many seconds.")
    return p


def parent_main():
    args, passthrough = build_parser().parse_known_args()
    terrains = [t.strip() for t in args.terrains.split(",") if t.strip()]
    unknown = [t for t in terrains if t not in TERRAINS]
    if unknown or not terrains:
        raise SystemExit("Unknown terrains: %s. Valid: %s" % (unknown, list(TERRAINS)))
    if not os.path.isfile(args.checkpoint):
        raise SystemExit("Checkpoint not found: " + args.checkpoint)
    paper = load_paper_tables()
    out = os.path.abspath(args.out)
    parts = out + ".parts"
    os.makedirs(parts, exist_ok=True)
    env = dict(os.environ, OMNI_KIT_ACCEPT_EULA="YES", PYTHONUNBUFFERED="1")
    results, errors = {}, {}
    for terrain in terrains:
        part, log_path = os.path.join(parts, terrain + ".json"), os.path.join(parts, terrain + ".log")
        if os.path.exists(part):
            os.remove(part)
        cmd = [
            sys.executable, os.path.abspath(__file__), "--worker", "--worker_terrain", terrain, "--worker_out", part,
            "--checkpoint", os.path.abspath(args.checkpoint), "--num_envs", str(args.num_envs), "--terrains", terrain,
            "--trials_per_env", str(args.trials_per_env), "--table1_steps", str(args.table1_steps),
            "--stop_radius", str(args.stop_radius), "--landing_tol", str(args.landing_tol),
            "--interaction_threshold", str(args.interaction_threshold), "--seed", str(args.seed), "--headless",
        ] + passthrough
        print("[eval] %s: starting worker, log %s" % (terrain, log_path), flush=True)
        t0 = time.time()
        with open(log_path, "wb") as log:
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                code = proc.wait(timeout=args.worker_timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                code = "timeout"
        # Trust the result file, not the exit code: a clean result followed by a hanging or crashing Kit shutdown is still a result.
        data = json.load(open(part, encoding="utf8")) if os.path.exists(part) else None
        if data and data.get("complete"):
            results[terrain] = data
            print("[eval] %s: done in %.0f s (exit %s)" % (terrain, time.time() - t0, code), flush=True)
        else:
            errors[terrain] = (data or {}).get("error", "worker exit %s, no result file; see %s" % (code, log_path))
            print("[eval] %s: FAILED: %s" % (terrain, errors[terrain]), flush=True)
    report = "\n\n".join([format_table2(paper, results), format_table1(paper, results)])
    print("\n" + SEPARATOR + "\n" + report + "\n" + SEPARATOR)
    payload = {
        "paper": "arXiv 2601.07718 v2, 'Ours' rows, read from docs/paper/v2.txt",
        "checkpoint": os.path.abspath(args.checkpoint),
        "args": vars(args),
        "paper_ours": paper,
        "results": results,
        "errors": errors,
        "report": report,
    }
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, out)
    print("[eval] wrote " + out)
    return 1 if errors else 0


def worker_main():
    """Runs inside the terrain subprocess. Mirrors run_train.py for the Windows launch quirks."""
    import mujoco  # noqa: F401  (must be imported before AppLauncher on Windows, see run_train.py)

    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    # ROOT holds a checkout named instinct_rl that would shadow the installed package.
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != ROOT]
    os.chdir(os.path.join(ROOT, "InstinctLab"))

    from isaaclab.app import AppLauncher

    parser = build_parser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--worker_terrain", required=True)
    parser.add_argument("--worker_out", required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    app_launcher = AppLauncher(args)  # noqa: F841  (starts Isaac Sim; keep a reference until the os._exit below)

    try:
        data = _worker_run(args)
        _write_json(args.worker_out, data)
        code = 0
    except Exception:
        traceback.print_exc()
        _write_json(args.worker_out, {"terrain": args.worker_terrain, "complete": False, "error": traceback.format_exc()})
        code = 1
    # The result is already on disk. Skip the Kit shutdown so a slow or hanging exit cannot hold the GPU until the parent's timeout.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def _write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def _worker_run(args):
    import gymnasium as gym
    import torch
    from instinct_rl.runners import OnPolicyRunner

    from isaaclab.sensors import RayCasterCfg, patterns
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    import instinctlab.tasks  # noqa: F401  (registers the gym ids, as train.py:103)
    from instinctlab.utils.wrappers import InstinctRlVecEnvWrapper

    terrain = args.worker_terrain
    run_table1 = terrain in TABLE1

    # Same loading path as tasks/parkour/scripts/play.py; load_cfg_from_registry is what cli_args.parse_instinct_rl_cfg calls.
    env_cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs, use_fabric=True)
    agent_cfg = load_cfg_from_registry(TASK, "instinct_rl_cfg_entry_point")
    apply_eval_overrides(env_cfg, terrain, args.stop_radius, args.seed)
    sole_height_offset = env_cfg.rewards.rewards.feet_at_plane.params["height_offset"]
    if run_table1:
        _add_landing_sensors(env_cfg, RayCasterCfg, patterns)

    env = gym.make(TASK, cfg=env_cfg, render_mode=None)
    env = InstinctRlVecEnvWrapper(env)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    base_env = env.unwrapped
    term = base_env.command_manager.get_term("base_velocity")
    robot = base_env.scene["robot"]
    dt = base_env.step_dt
    horizon = int(round(TRIAL_S / dt))

    # Keep the real flat-patch target mechanism but drop targets that are trivially close to the spawn point.
    rng = np.random.default_rng(args.seed)
    patches, n_replaced, n_empty = filter_near_targets(
        term.valid_targets.cpu().numpy(), base_env.scene.terrain.terrain_origins.cpu().numpy(), MIN_START_DIST_M, rng
    )
    term.valid_targets[:] = torch.as_tensor(patches, device=term.valid_targets.device, dtype=term.valid_targets.dtype)
    print("[eval] flat-patch targets replaced: %d, tiles with no valid target: %d" % (n_replaced, n_empty), flush=True)

    def distances():
        return torch.norm(term.pos_command_w[:, :2] - robot.data.root_pos_w[:, :2], dim=1).cpu().numpy()

    def run(num_steps, tracker, on_step=None):
        # Everything inside one inference_mode block, including reset: a reset outside it after an earlier block
        # would do in-place writes on tensors created inside inference mode and raise.
        with torch.inference_mode():
            obs, _ = env.reset()
            start = distances()
            rec = [] if (os.environ.get("HIKING_OBS_CHECK") and not getattr(run, "recorded", False)) else None
            for _ in range(num_steps):
                actions = policy(obs)
                if rec is not None and len(rec) < 50:
                    rec.append(dict(obs=obs[0].cpu().numpy(), actions=actions[0].cpu().numpy(),
                                    root_pos_w=robot.data.root_pos_w[0].cpu().numpy(), root_quat_w=robot.data.root_quat_w[0].cpu().numpy(),
                                    root_lin_vel_b=robot.data.root_lin_vel_b[0].cpu().numpy(), root_ang_vel_b=robot.data.root_ang_vel_b[0].cpu().numpy(),
                                    projected_gravity_b=robot.data.projected_gravity_b[0].cpu().numpy(),
                                    joint_pos=robot.data.joint_pos[0].cpu().numpy(), joint_vel=robot.data.joint_vel[0].cpu().numpy(),
                                    command=base_env.command_manager.get_command("base_velocity")[0].cpu().numpy()))
                    if len(rec) == 50:
                        np.savez(os.environ["HIKING_OBS_CHECK"], joint_names=np.array(robot.joint_names),
                                 default_joint_pos=robot.data.default_joint_pos[0].cpu().numpy(),
                                 **{k: np.stack([r[k] for r in rec]) for k in rec[0]})
                        run.recorded = True
                        print("[eval] wrote obs check", os.environ["HIKING_OBS_CHECK"], flush=True)
                obs, _, dones, _ = env.step(actions)
                done = dones.bool().cpu().numpy()
                tracker.update(done, distances())
                if on_step is not None:
                    on_step(done)
                if tracker.finished:
                    break
        return start

    data = {"terrain": terrain, "complete": True, "dt": dt, "horizon_steps": horizon,
            "targets_replaced": n_replaced, "tiles_without_valid_target": n_empty}

    # Table 2: at most trials_per_env trials per robot. Episodes run horizon + 1 steps (one trial) unless a robot terminates.
    tracker = TrialTracker(args.num_envs, horizon, max_trials=args.trials_per_env)
    start = run((horizon + 1) * args.trials_per_env, tracker)
    data["table2"] = tracker.summary(dt)
    data["table2"]["target_start_distance_m"] = {"mean": float(start.mean()), "min": float(start.min()), "max": float(start.max())}
    print("[eval] table2", data["table2"], flush=True)

    if run_table1:
        data["table1"] = _run_table1(args, env, base_env, robot, dt, horizon, sole_height_offset, run)
        print("[eval] table1", data["table1"], flush=True)
    return data


def _add_landing_sensors(env_cfg, RayCasterCfg, patterns):
    """Height rays under each foot sole, and a 1 x 1 m grid under the base for the interaction gate. Never observed by the policy."""
    for side in ("left", "right"):
        setattr(
            env_cfg.scene,
            "eval_%s_sole_scanner" % side,
            RayCasterCfg(
                prim_path="{ENV_REGEX_NS}/Robot/%s_ankle_roll_link" % side,
                # Sole area from the shoe URDF collision cylinders: x in [-0.054, 0.132], y in +-0.036, so centred on x = 0.039 m.
                offset=RayCasterCfg.OffsetCfg(pos=(0.037, 0.0, 20.0)),  # X2 sole: x -0.070..0.144, y +-0.065
                ray_alignment="yaw",
                pattern_cfg=patterns.GridPatternCfg(resolution=0.02, size=[0.18, 0.10]),
                debug_vis=False,
                mesh_prim_paths=["/World/ground"],
                update_period=0.02,
            ),
        )
    env_cfg.scene.eval_base_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/pelvis",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.0, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        update_period=0.02,
    )


def _run_table1(args, env, base_env, robot, dt, horizon, sole_height_offset, run):
    scene = base_env.scene
    contact = scene["contact_forces"]
    base_scanner = scene["eval_base_scanner"]
    feet = []
    contact_ids, contact_names = contact.find_bodies(".*_ankle_roll_link")
    for side in ("left", "right"):
        name = "%s_ankle_roll_link" % side
        feet.append(
            (
                side,
                contact_ids[contact_names.index(name)],
                robot.find_bodies(name)[0][0],
                scene["eval_%s_sole_scanner" % side],
            )
        )
    total, count = 0.0, 0
    per_foot = {}

    def on_step(done):
        nonlocal total, count
        first = contact.compute_first_contact(dt).cpu().numpy()
        heights = base_scanner.data.ray_hits_w[..., 2].cpu().numpy()
        gate = interaction_mask(heights, args.interaction_threshold)
        for side, contact_id, body_id, sole_sensor in feet:
            sole_z = (robot.data.body_pos_w[:, body_id, 2] - sole_height_offset).cpu().numpy()
            hit_z = sole_sensor.data.ray_hits_w[..., 2].cpu().numpy()
            s, n = landing_step(first[:, contact_id], done, gate, sole_z, hit_z, args.landing_tol)
            total += s
            count += n
            per_foot[side] = per_foot.get(side, 0) + n

    tracker = TrialTracker(args.num_envs, horizon)  # rolling: every episode is one trial, unfinished ones are dropped
    run(args.table1_steps, tracker, on_step)
    result = tracker.summary(dt)
    result["landing_area_fraction"] = total / count if count else None
    result["n_landing_events"] = count
    result["n_landing_events_per_foot"] = per_foot
    return result


if __name__ == "__main__":
    if "--worker" in sys.argv:
        worker_main()
    else:
        sys.exit(parent_main())
