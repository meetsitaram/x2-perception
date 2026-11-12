"""Hiking in the Wild (parkour target AMP) on the AgiBot X2 Ultra.

The terrains, commands, rewards, observations, curriculum and randomisation of the paper are inherited unchanged from
ParkourEnvCfg. Only the robot differs: asset and actuators, 29 commanded joints (head held), foot geometry numbers, depth camera
mount, root-height threshold scaled to the X2, and an X2 motion reference.
"""
import copy
import math
import os

from isaaclab.envs import ViewerCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import instinctlab.envs.mdp as instinct_mdp
from instinctlab.assets.agibot_x2 import (
    X2_ACTION_JOINTS,
    X2_CFG,
    X2_LINKS,
    X2_symmetric_augmentation_joint_mapping,
    X2_symmetric_augmentation_joint_reverse_buf,
    x2_action_scale,
    x2_delayed_actuators_mc,
)
from instinctlab.motion_reference import MotionReferenceManagerCfg
from instinctlab.motion_reference.motion_files.amass_motion_cfg import AmassMotionCfg as AmassMotionCfgBase
from instinctlab.motion_reference.utils import motion_interpolate_bilinear
from instinctlab.sensors import NoisyGroupedRayCasterCameraCfg, get_link_prim_targets
from instinctlab.tasks.parkour.config.parkour_env_cfg import ROUGH_TERRAINS_CFG, ParkourEnvCfg

from .monitor import walk_monitor

X2_MOTION_DIR = os.environ.get("HIKING_X2_MOTION_DIR", "/mnt/filesystem-g4/data/motions/hiking_x2")

# X2 foot (ankle_roll_link frame, from the sole contact points of the model): sole 0.073 m below the ankle origin,
# x -0.070..0.144, y +-0.065.
X2_SOLE_BELOW_ANKLE = 0.073
# Built-in Orbbec Gemini 335 (URDF joint rgbd_head_front on head_pitch_link): 40 deg below horizontal with the head at zero.
X2_CAMERA_POS = (0.05761, -0.011183, -0.04837)
X2_CAMERA_PITCH_DOWN = math.radians(40.0)


@configclass
class AmassMotionCfg(AmassMotionCfgBase):
    path = X2_MOTION_DIR
    retargetting_func = None
    # v2 (from iteration 750): standing stretches cut to 3% (v1 had 21%, the G1 release 5%); v1 yaml: x2_parkour_motion_without_run.yaml
    filtered_motion_selection_filepath = os.path.join(X2_MOTION_DIR, os.environ.get("HIKING_X2_MOTION_YAML", "x2_parkour_motion_without_run_v2.yaml"))
    motion_start_from_middle_range = [0.0, 0.9]
    motion_start_height_offset = 0.0
    ensure_link_below_zero_ground = False
    buffer_device = "output_device"
    motion_interpolate_func = motion_interpolate_bilinear
    velocity_estimation_method = "frontward"


motion_reference_cfg = MotionReferenceManagerCfg(
    prim_path="{ENV_REGEX_NS}/Robot/pelvis",
    robot_model_path=X2_CFG.spawn.asset_path,
    reference_prim_path="/World/envs/env_.*/RobotReference/pelvis",
    symmetric_augmentation_link_mapping=[0, 1, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12],
    symmetric_augmentation_joint_mapping=X2_symmetric_augmentation_joint_mapping,
    symmetric_augmentation_joint_reverse_buf=X2_symmetric_augmentation_joint_reverse_buf,
    frame_interval_s=0.02,
    update_period=0.02,
    num_frames=10,
    motion_buffers={
        "run_walk": AmassMotionCfg(),
    },
    link_of_interests=[
        "pelvis",
        "torso_link",
        "left_shoulder_roll_link",
        "right_shoulder_roll_link",
        "left_elbow_link",
        "right_elbow_link",
        "left_wrist_roll_link",
        "right_wrist_roll_link",
        "left_hip_roll_link",
        "right_hip_roll_link",
        "left_knee_link",
        "right_knee_link",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    ],
    mp_split_method="Even",
)

ROUGH_TERRAINS_CFG_PLAY = copy.deepcopy(ROUGH_TERRAINS_CFG)
for sub_terrain_name, sub_terrain_cfg in ROUGH_TERRAINS_CFG_PLAY.sub_terrains.items():
    sub_terrain_cfg.wall_prob = [0.0, 0.0, 0.0, 0.0]


@configclass
class X2ParkourEnvCfg(ParkourEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # robot
        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG
        self.scene.robot = X2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.motion_reference = motion_reference_cfg
        # 29 commanded joints, BeyondMimic action scale from the X2 plant
        self.actions.joint_pos.joint_names = X2_ACTION_JOINTS
        self.actions.joint_pos.scale = x2_action_scale
        # depth camera: the head camera the X2 ships with
        self.scene.camera.prim_path = "{ENV_REGEX_NS}/Robot/head_pitch_link"
        self.scene.camera.offset = NoisyGroupedRayCasterCameraCfg.OffsetCfg(
            pos=X2_CAMERA_POS,
            rot=(math.cos(X2_CAMERA_PITCH_DOWN / 2), 0.0, math.sin(X2_CAMERA_PITCH_DOWN / 2), 0.0),
            convention="world",
        )
        self.scene.camera.mesh_prim_paths.extend(get_link_prim_targets(X2_LINKS))
        # foot volume points: the box of the G1 task (inset from the sole outline, from 5 mm below the sole to 35 mm above it)
        g = self.scene.leg_volume_points.points_generator
        g.x_min, g.x_max = -0.045, 0.12
        g.y_min, g.y_max = -0.05, 0.05
        g.z_min, g.z_max = -(X2_SOLE_BELOW_ANKLE + 0.005), -(X2_SOLE_BELOW_ANKLE - 0.035)
        self.rewards.rewards.feet_at_plane.params["height_offset"] = X2_SOLE_BELOW_ANKLE
        # root is the pelvis (0.641 m standing) instead of the torso of the G1 (about 0.83 m): same fraction of standing height
        self.terminations.root_height.params["minimum_height"] = 0.39
        # The G1 task penalises the tilt of its root (torso, flat_orientation_l2) and of the pelvis link (pelvis_orientation_l2).
        # The X2 root is the pelvis, so the link term takes the torso: both bodies are penalised, as on the G1.
        self.rewards.rewards.pelvis_orientation_l2.params["asset_cfg"].body_names = "torso_link"
        # log-only: commanded vs actual speed, support shares, step rate, terrain level per type (monitor.py)
        self.curriculum.walk_monitor = CurrTerm(func=walk_monitor)


# Run 2 camera (user, 2026-11-11): the Intel RealSense (D435 class) the user fitted on the X2's chest, centred, pitched down.
# A departure from the paper's camera at the user's direction. Calibration: the user's infrared frame (the depth imager's view,
# 87 x 58 deg): the robot's toe tips just inside the bottom edge, and the line 0.985 m ahead of the toe line at 27% of the image
# height from the top. Solved for the simulated standing pose with the lens 3 cm proud of the chest shell (shell front from the
# torso mesh, x = 0.101 m at that height; chest_solve.py): lens 1.055 m above the ground, 61.6 deg below horizontal, 1 cm ahead of
# the toe tips; flat-ground view from the toe tips to 1.66 m ahead. (If the real lens sits 12 cm behind the toe tips, as the
# coordinator read off the photos, the same landmarks give 0.90 m and 53.4 deg, to 1.86 m: the user's measured height and angle
# will settle it.) All 36 rows kept (the paper keeps the lower 18), the paper's left/right crop, range 0-3 m (paper 2.5).
X2_TORSO_CAMERA_POS = (0.131, 0.0, 0.259)
X2_TORSO_CAMERA_PITCH_DOWN = math.radians(61.6)
X2_TORSO_CAMERA_FOV_DEG = (87.0, 58.0)
# Camera mount randomisation per robot, redrawn at each episode reset (user, 2026-11-11: the real lens height is "somewhere in
# between" 0.92 and 1.05 m): height -10.5..+5 cm (0.95 to 1.105 m), forward +-2 cm, pitch +-4 deg, roll and yaw +-2 deg.
# OFF unless HIKING_X2_CAM_RAND=1 is in the environment of the training process: it has not run in Isaac yet, and an unattended
# auto-resume must not pick up untested code. Turn it on at a watched restart (start_x2_parkour_v2.sh takes the variable through).
X2_CAMERA_MOUNT_RANGES = {
    "x": (-0.02, 0.02),
    "z": (-0.105, 0.05),
    "roll": (-math.radians(2.0), math.radians(2.0)),
    "pitch": (-math.radians(4.0), math.radians(4.0)),
    "yaw": (-math.radians(2.0), math.radians(2.0)),
}


@configclass
class X2ParkourTorsoCamEnvCfg(X2ParkourEnvCfg):
    """Run 2: measured PD gains (plant mc_20261024) and the chest RealSense camera. Everything else as X2ParkourEnvCfg."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.actuators = x2_delayed_actuators_mc
        self.scene.camera.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"
        self.scene.camera.offset = NoisyGroupedRayCasterCameraCfg.OffsetCfg(
            pos=X2_TORSO_CAMERA_POS,
            rot=(math.cos(X2_TORSO_CAMERA_PITCH_DOWN / 2), 0.0, math.sin(X2_TORSO_CAMERA_PITCH_DOWN / 2), 0.0),
            convention="world",
        )
        self.scene.camera.pattern_cfg.horizontal_aperture = 2 * math.tan(math.radians(X2_TORSO_CAMERA_FOV_DEG[0]) / 2)
        self.scene.camera.pattern_cfg.vertical_aperture = 2 * math.tan(math.radians(X2_TORSO_CAMERA_FOV_DEG[1]) / 2)
        self.scene.camera.noise_pipeline["crop_and_resize"].crop_region = (0, 0, 16, 16)
        self.scene.camera.noise_pipeline["depth_normalization"].depth_range = (0.0, 3.0)
        if os.environ.get("HIKING_X2_CAM_RAND") == "1":
            self.events.camera_mount = EventTerm(
                func=instinct_mdp.randomize_camera_offsets,
                mode="reset",
                params={"asset_cfg": SceneEntityCfg("camera"), "offset_pose_ranges": X2_CAMERA_MOUNT_RANGES},
            )


@configclass
class X2ParkourEnvCfg_PLAY(X2ParkourEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG_PLAY
        self.scene.num_envs = 10
        self.viewer = ViewerCfg(
            eye=[4.0, 0.75, 1.0],
            lookat=[0.0, 0.75, 0.0],
            origin_type="asset_root",
            asset_name="robot",
        )
        self.scene.env_spacing = 2.5
        self.episode_length_s = 10
        self.terminations.root_height = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 4
            self.scene.terrain.terrain_generator.num_cols = 10
        self.scene.leg_volume_points.debug_vis = True
        self.commands.base_velocity.debug_vis = True
        self.events.physics_material = None
        self.events.reset_robot_joints.params = {
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        }
