"""Extended training task for the X2 (second task name, `Instinct-Parkour-Extended-Amp-X2-v0`).

STATUS 2026-11-11: written on the CPU while run 2 trains; NOT yet started in Isaac Sim, so untested there. Do not launch a
training run from it before `check_x2_env.py --task Instinct-Parkour-Extended-Amp-X2-v0` passes.

Adds to the run-2 task (`X2ParkourTorsoCamEnvCfg`: paper terrains, chest camera, mc_20261024 gains):
1. Terrains: the paper's nine types keep their 20 columns; 12 new columns hold tiles of the exported set
   `/mnt/filesystem-g4/data/terrains/extended_v1` (single beams, beam fields, blocks, wedges, tight spots, log corridors, homes,
   street front, stairs x2) and two columns of rough ground driven by velocity commands. One third of the field is new, as the user
   asked. Tile geometry is built from the set's boxes and meshes (not from the 2 cm rasters, which would be 320,000 triangles a tile).
2. Commands: on the velocity-command columns the robot gets a held velocity instead of a target: forward -0.4..0.8 m/s, sideways
   -0.3..0.3 m/s, turn -1..1 rad/s (backward, sideways and turn-in-place walking). The paper's "stand" column shares these ranges
   (the command class has one range for all velocity-mode columns); 5% of robots are still told to stand.
3. Rewards (user: "INCLUDE THEM"): `feet_stumble` (a moving foot pushed sideways harder than it is loaded), `foot_clearance` (a
   swing foot moving low over the ground under it). The paper's edge term (`volume_points_penetration`) stays as the foothold term.
   Weights are starting points.

Not done here (listed in state/hiking.md): the set's checked routes as spawn and target pairs (targets are flat patches sampled on
each tile, robots start at the tile's widest-clearance spawn, spawn yaw is not applied); backward and sideways clips in the motion
reference; graded knee and shin costs; sticky side faces; obstacle-only contact sensing.
The two challenge apparatuses are not in the tile set as layouts or routes (see the set's README).
"""
from __future__ import annotations

import copy
import json
import os

import numpy as np
import torch
import trimesh

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import FlatPatchSamplingCfg
from isaaclab.terrains.terrain_generator_cfg import SubTerrainBaseCfg
from isaaclab.utils import configclass

from instinctlab.tasks.parkour.config.parkour_env_cfg import ROUGH_TERRAINS_CFG

from .x2_parkour_target_amp_cfg import X2_SOLE_BELOW_ANKLE, X2ParkourTorsoCamEnvCfg

EXTENDED_DIR = os.environ.get("HIKING_EXTENDED_TERRAINS", "/mnt/filesystem-g4/data/terrains/extended_v1")
_cache = {}
_CALLS = {}  # tile_type -> [(tile index, spawn (x, y, z) in tile-centred coordinates)] in generator order (routes.py)


def _load(tile_type: str, root: str):
    if tile_type not in _cache:
        z = np.load(os.path.join(root, tile_type + ".npz"))
        meta = json.load(open(os.path.join(root, tile_type + ".json")))["tiles"]
        _cache[tile_type] = (z, meta)
    return _cache[tile_type]


def _quat_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def extended_tile_terrain(difficulty: float, cfg: "ExtendedTileTerrainCfg"):
    """One exported tile as a sub-terrain. The row's difficulty picks the level (0-9); the seed variant is random.

    Returns the meshes in the generator's convention (tile corner at the origin, so the set's centred coordinates are shifted by
    half the tile size) and the spawn origin: the kept spawn pose with the widest clearance, at its floor height.
    """
    z, meta = _load(cfg.tile_type, cfg.root)
    level = int(np.clip(np.floor(difficulty * 10), 0, 9))
    cand = [t for t in meta if t["level"] == level and t["spawns"]] or [t for t in meta if t["spawns"]]
    tile = cand[np.random.randint(len(cand))]
    i = tile["index"]
    shift = np.array([cfg.size[0] / 2, cfg.size[1] / 2, 0.0])
    meshes = []
    b0, b1 = int(z["box_start"][i]), int(z["box_start"][i + 1])
    zmin = 0.0
    for b in range(b0, b1):
        t = np.eye(4)
        t[:3, :3] = _quat_matrix(z["box_quat"][b])
        t[:3, 3] = z["box_center"][b] + shift
        m = trimesh.creation.box(extents=2 * z["box_half"][b], transform=t)
        zmin = min(zmin, float(m.bounds[0, 2]))
        meshes.append(m)
    m0, m1 = int(z["mesh_start"][i]), int(z["mesh_start"][i + 1])
    for k in range(m0, m1):
        v = z["mesh_vertices"][int(z["mesh_vert_start"][k]):int(z["mesh_vert_start"][k + 1])] + shift
        f = z["mesh_faces"][int(z["mesh_face_start"][k]):int(z["mesh_face_start"][k + 1])]
        meshes.append(trimesh.Trimesh(vertices=v, faces=f, process=False))
        zmin = min(zmin, float(v[:, 2].min()))
    # a slab under everything, so a hole in a tile cannot let a robot fall out of the world
    t = np.eye(4)
    t[:3, 3] = (cfg.size[0] / 2, cfg.size[1] / 2, zmin - 0.25)
    meshes.append(trimesh.creation.box(extents=(cfg.size[0], cfg.size[1], 0.5 - 0.02), transform=t))
    spawn = max(tile["spawns"], key=lambda s: s.get("clearance_m", 0.0))
    _CALLS.setdefault(cfg.tile_type, []).append((int(i), [float(spawn["x"]), float(spawn["y"]), float(spawn.get("z", 0.0))]))
    origin = np.array([spawn["x"] + shift[0], spawn["y"] + shift[1], spawn.get("z", 0.0)])
    return meshes, origin


@configclass
class ExtendedTileTerrainCfg(SubTerrainBaseCfg):
    function = extended_tile_terrain
    tile_type: str = "single_beams"
    root: str = EXTENDED_DIR


def _targets(z_hi=0.45):
    # targets only on floor-like surfaces: tops of walls, counters and cars (0.9 m and more) are excluded by height
    return {"target": FlatPatchSamplingCfg(num_patches=50, patch_radius=[0.05, 0.10, 0.15, 0.20], max_height_diff=0.05,
                                           z_range=(-0.6, z_hi))}


# name -> (tile type, columns of 32, highest target above the tile's floor)
EXTENDED_COLUMNS = {
    "ext_single_beams": ("single_beams", 1, 0.45),
    "ext_beam_fields": ("beam_fields", 1, 0.45),
    "ext_blocks": ("blocks", 1, 0.45),
    "ext_wedges": ("wedges", 1, 0.6),
    "ext_tight_spots": ("tight_spots", 1, 0.45),
    "ext_log_corridors": ("log_corridors", 1, 0.45),
    "ext_homes": ("homes", 1, 0.45),
    "ext_street_front": ("street_front", 1, 0.45),
    "ext_stairs": ("stairs", 2, 3.5),
}
VELOCITY_COLUMNS = 2
PAPER_COLUMNS, TOTAL_COLUMNS = 20, 32

EXTENDED_TERRAINS_CFG = copy.deepcopy(ROUGH_TERRAINS_CFG)
EXTENDED_TERRAINS_CFG.num_cols = TOTAL_COLUMNS
_paper_total = sum(c.proportion for c in EXTENDED_TERRAINS_CFG.sub_terrains.values())
for _c in EXTENDED_TERRAINS_CFG.sub_terrains.values():
    _c.proportion = _c.proportion / _paper_total * PAPER_COLUMNS / TOTAL_COLUMNS
for _name, (_type, _cols, _zhi) in EXTENDED_COLUMNS.items():
    EXTENDED_TERRAINS_CFG.sub_terrains[_name] = ExtendedTileTerrainCfg(
        proportion=_cols / TOTAL_COLUMNS, tile_type=_type, size=EXTENDED_TERRAINS_CFG.size, flat_patch_sampling=_targets(_zhi)
    )
_vel = copy.deepcopy(ROUGH_TERRAINS_CFG.sub_terrains["perlin_rough"])
_vel.proportion = VELOCITY_COLUMNS / TOTAL_COLUMNS
EXTENDED_TERRAINS_CFG.sub_terrains["perlin_rough_velocity"] = _vel


##
# Rewards
##
def feet_stumble(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg, force_ratio: float = 3.0,
                 min_force: float = 5.0, speed_threshold: float = 0.15, force_scale: float = 100.0) -> torch.Tensor:
    """A moving foot hit from the side: horizontal contact force over `force_ratio` times the vertical one (the SSR paper's ratio;
    SOLO uses 1, TACT-ful 4 with the same 0.15 m/s gate), scaled by the force so a hard hit costs more than a graze (0.2 to 2)."""
    f = env.scene.sensors[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids]
    horizontal = torch.norm(f[..., :2], dim=-1)
    vertical = torch.abs(f[..., 2])
    speed = torch.norm(env.scene[asset_cfg.name].data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=-1)
    hit = (horizontal > force_ratio * vertical) & (horizontal > min_force) & (speed > speed_threshold)
    return torch.sum(hit.float() * torch.clamp(horizontal / force_scale, 0.2, 2.0), dim=1)


def foot_clearance(env, contact_sensor_cfg: SceneEntityCfg, left_height_scanner_cfg: SceneEntityCfg,
                   right_height_scanner_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg, sole_offset: float,
                   target_height: float = 0.06) -> torch.Tensor:
    """A swing foot moving low: (target - sole height above the ground under it), clipped at zero, times the foot's horizontal speed
    (TACT-ful's term, one-sided so high steps on stairs are free). Body order of the cfgs: left, right."""
    robot = env.scene[asset_cfg.name]
    forces = env.scene.sensors[contact_sensor_cfg.name].data.net_forces_w_history[:, :, contact_sensor_cfg.body_ids]
    swing = torch.max(torch.norm(forces, dim=-1), dim=1)[0] < 1.0
    foot_z = robot.data.body_pos_w[:, asset_cfg.body_ids, 2] - sole_offset
    ground = torch.stack([
        env.scene.sensors[left_height_scanner_cfg.name].data.ray_hits_w[..., 2].max(dim=1)[0],
        env.scene.sensors[right_height_scanner_cfg.name].data.ray_hits_w[..., 2].max(dim=1)[0],
    ], dim=1)
    ground = torch.nan_to_num(ground, nan=0.0, posinf=0.0, neginf=0.0)
    speed = torch.norm(robot.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=-1)
    low = torch.clamp(target_height - (foot_z - ground), min=0.0, max=target_height)
    return torch.sum(swing.float() * low * speed, dim=1)


def _apply_extended(self):
    """The extended task's changes, applied after the base task's __post_init__ (so after run 4's options c / y replaced the
    command class: the ranges below go into that command's config)."""
    self.scene.terrain.terrain_generator = EXTENDED_TERRAINS_CFG
    # commands: every new column needs a speed range (a column without one would never be told to move)
    cmd = self.commands.base_velocity
    for name in EXTENDED_COLUMNS:
        cmd.velocity_ranges[name] = {"lin_vel_x": (0.45, 0.8), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)}
    cmd.velocity_ranges["perlin_rough_velocity"] = {"lin_vel_x": (0.0, 0.0), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (0.0, 0.0)}
    cmd.random_velocity_terrain = ["perlin_rough_stand", "perlin_rough_velocity"]
    cmd.ranges.lin_vel_x = (float(os.environ.get("HIKING_X2_VX_BACK", "-0.4")), 0.8)  # v4k: 0 (user 2026-11-12: never step back)
    cmd.ranges.lin_vel_y = (-0.3, 0.3)
    cmd.ranges.ang_vel_z = (-1.0, 1.0)
    # rewards (body_names preserve_order so index 0 is the left foot, as the two height scanners are ordered)
    feet = ["left_ankle_roll_link", "right_ankle_roll_link"]
    self.rewards.rewards.feet_stumble = RewTerm(
        func=feet_stumble,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=feet, preserve_order=True),
            "asset_cfg": SceneEntityCfg("robot", body_names=feet, preserve_order=True),
        },
    )
    self.rewards.rewards.foot_clearance = RewTerm(
        func=foot_clearance,
        weight=-0.5,
        params={
            "contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=feet, preserve_order=True),
            "left_height_scanner_cfg": SceneEntityCfg("left_height_scanner"),
            "right_height_scanner_cfg": SceneEntityCfg("right_height_scanner"),
            "asset_cfg": SceneEntityCfg("robot", body_names=feet, preserve_order=True),
            "sole_offset": X2_SOLE_BELOW_ANKLE,
        },
    )


@configclass
class X2ParkourExtendedEnvCfg(X2ParkourTorsoCamEnvCfg):
    """The 2026-11-11 version: extended terrains on the run-2 task (no run-4 fixes). Kept for reference; not trained."""
    def __post_init__(self):
        super().__post_init__()
        _apply_extended(self)


from .fixed import X2ParkourFixedEnvCfg  # noqa: E402


@configclass
class X2ParkourExtendedFixedEnvCfg(X2ParkourFixedEnvCfg):
    """2026-11-12 (user: the extended map goes into Hiking's training now): run 4's task (fixes, HIKING_X2_OPTS options, level
    keeper, turn in place, overspeed, sliding) plus the extended terrain columns, velocity-mode columns, stumble and clearance
    terms. Same observations and actions as run 4, so v4g's checkpoints continue on it."""
    def __post_init__(self):
        super().__post_init__()
        _apply_extended(self)
        from .staged import _opts
        if "g" in _opts():  # v4j: wall / prop contact graded by body part, torso ends only when resting on something (contact.py)
            from .contact import apply_v4j
            apply_v4j(self)
