"""AgiBot X2 Ultra for the parkour (Hiking in the Wild) task.

Model: the shared X2 URDF (/mnt/filesystem-g4/data/robots/x2_ultra/urdf/x2_ultra.urdf, mesh feet), root = pelvis.
Actuator plant: vendor datasheet 2026-08-23 (gwbc-v6 config/robot_plant/x2_ultra.yaml), with the same BeyondMimic rule the G1
task uses: kp = armature * (2 pi 10 Hz)^2, kd = 2 * 2.0 * armature * (2 pi 10 Hz), action scale = 0.25 * effort / kp.
Delayed PD actuators (0-2 steps) in runs 1-3. Correction (audit 2026-11-12): the G1 run used implicit PD with no delay
(its apply_shoe_config replaces the delayed actuators); run 4 (config/x2/fixed.py) uses implicit PD like the G1.
"""
import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# The shared URDF with its massless base_link (fixed to pelvis) removed, so the root body is named pelvis: see x2_port/robot/README.md
X2_URDF = os.environ.get("HIKING_X2_URDF", "/mnt/filesystem-g4/tracks/hiking/code/x2_port/robot/x2_ultra_pelvis_root.urdf")

ARMATURE_PF52 = 0.003132992  # 24 N.m
ARMATURE_PF70 = 0.008840000  # 36 N.m
ARMATURE_PF90 = 0.029079810  # 120 N.m
ARMATURE_SMALL = 0.00425  # wrist pitch/roll, head: vendor value not supplied (estimate)

NATURAL_FREQ = 10 * 2.0 * 3.1415926535
DAMPING_RATIO = 2.0

# joint pattern -> (armature, effort N.m, velocity rad/s)
_PLANT = {
    ".*_hip_pitch_joint": (ARMATURE_PF90, 120.0, 11.936),
    ".*_hip_roll_joint": (ARMATURE_PF90, 120.0, 11.936),
    ".*_hip_yaw_joint": (ARMATURE_PF90, 120.0, 11.936),
    ".*_knee_joint": (ARMATURE_PF90, 120.0, 11.936),
    ".*_ankle_pitch_joint": (ARMATURE_PF70, 36.0, 13.088),
    ".*_ankle_roll_joint": (ARMATURE_PF52, 24.0, 15.077),
    "waist_yaw_joint": (ARMATURE_PF90, 120.0, 11.936),
    "waist_pitch_joint": (ARMATURE_PF52, 24.0, 13.088),
    "waist_roll_joint": (ARMATURE_PF52, 24.0, 13.088),
    ".*_shoulder_pitch_joint": (ARMATURE_PF70, 36.0, 13.088),
    ".*_shoulder_roll_joint": (ARMATURE_PF70, 36.0, 13.088),
    ".*_shoulder_yaw_joint": (ARMATURE_PF52, 24.0, 15.077),
    ".*_elbow_joint": (ARMATURE_PF52, 24.0, 15.077),
    ".*_wrist_yaw_joint": (ARMATURE_PF52, 24.0, 15.077),
    ".*_wrist_pitch_joint": (ARMATURE_SMALL, 6.0, 20.944),
    ".*_wrist_roll_joint": (ARMATURE_SMALL, 6.0, 20.944),
    "head_yaw_joint": (ARMATURE_SMALL, 2.6, 6.019),
    "head_pitch_joint": (ARMATURE_SMALL, 0.6, 6.28),
}
_GROUPS = {
    "legs": [".*_hip_pitch_joint", ".*_hip_roll_joint", ".*_hip_yaw_joint", ".*_knee_joint"],
    "feet": [".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
    "waist": ["waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint"],
    "arms": [
        ".*_shoulder_pitch_joint",
        ".*_shoulder_roll_joint",
        ".*_shoulder_yaw_joint",
        ".*_elbow_joint",
        ".*_wrist_yaw_joint",
        ".*_wrist_pitch_joint",
        ".*_wrist_roll_joint",
    ],
    "head": ["head_yaw_joint", "head_pitch_joint"],
}


def _stiffness(a):
    return a * NATURAL_FREQ**2


def _damping(a):
    return 2.0 * DAMPING_RATIO * a * NATURAL_FREQ


# Plant mc_20261024 (data/robots/x2/robot_plant/x2_ultra_mc_20261024.yaml): the PD gains the vendor motion controller commands,
# measured on the robot; the user's choice for run 2 ("the ones used for supermimic training last iteration"). Armature, effort
# and default pose are the vendor values above; only stiffness and damping differ. Keys are substrings, first match wins.
# The action scale is NOT derived from these: it stays 0.25 * effort / (armature * w^2), as in that plant's plant_config.py.
_MC_STIFFNESS = {
    "hip": 100.0, "knee": 150.0, "ankle_pitch": 40.0, "ankle_roll": 30.0, "waist": 40.0, "shoulder_pitch": 30.0,
    "shoulder_roll": 20.0, "shoulder_yaw": 20.0, "elbow": 50.0, "wrist_yaw": 50.0, "wrist_pitch": 20.0, "wrist_roll": 20.0,
    "head_yaw": 3.4, "head_pitch": 10.0,
}
_MC_DAMPING = {
    "hip_pitch": 4.0, "hip": 3.0, "knee": 5.0, "ankle_pitch": 3.0, "ankle_roll": 2.0, "waist_yaw": 8.0, "waist": 5.0,
    "shoulder": 1.0, "elbow": 1.0, "wrist": 1.0, "head_yaw": 0.114, "head_pitch": 0.8,
}


def _first(table, pattern):
    name = pattern.replace(".*_", "")
    for key, value in table.items():
        if key in name:
            return value
    raise KeyError(pattern)


def make_x2_delayed_actuators(plant="armature_rule"):
    """plant: "armature_rule" (kp, kd from the armature; run 1) or "mc_20261024" (measured gains; run 2)."""
    assert plant in ("armature_rule", "mc_20261024"), plant
    mc = plant == "mc_20261024"
    return {
        group: DelayedPDActuatorCfg(
            joint_names_expr=names,
            effort_limit_sim={n: _PLANT[n][1] for n in names},
            velocity_limit_sim={n: _PLANT[n][2] for n in names},
            stiffness={n: (_first(_MC_STIFFNESS, n) if mc else _stiffness(_PLANT[n][0])) for n in names},
            damping={n: (_first(_MC_DAMPING, n) if mc else _damping(_PLANT[n][0])) for n in names},
            armature={n: _PLANT[n][0] for n in names},
            min_delay=0,
            max_delay=2,
        )
        for group, names in _GROUPS.items()
    }


x2_delayed_actuators = make_x2_delayed_actuators("armature_rule")
x2_delayed_actuators_mc = make_x2_delayed_actuators("mc_20261024")

# The policy commands 29 joints. The two head joints (2.6 and 0.6 N.m, carrying the depth camera) are held at zero by their PD.
X2_ACTION_JOINTS = [n for g in ("legs", "feet", "waist", "arms") for n in _GROUPS[g]]
x2_action_scale = {n: 0.25 * _PLANT[n][1] / _stiffness(_PLANT[n][0]) for n in X2_ACTION_JOINTS}

# Link pairs two joints apart whose collision meshes (convex hulls) overlap in every pose: PhysX only skips parent-child pairs, so
# with self-collision on these push against each other with thousands of newtons from the first step (measured at spawn:
# knee/ankle_roll 8 kN, torso/head_pitch 20 kN, wrist_yaw/wrist_roll 6 kN). They are filtered; all other self-collisions stay on.
X2_FILTERED_PAIRS = [("torso_link", "head_pitch_link")] + [
    (f"{s}_{a}", f"{s}_{b}")
    for s in ("left", "right")
    for a, b in (("knee_link", "ankle_roll_link"), ("wrist_yaw_link", "wrist_roll_link"))
]


def spawn_x2(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """spawn_from_urdf, then filter the always-overlapping link pairs."""
    from pxr import UsdPhysics

    prim = sim_utils.spawn_from_urdf(prim_path, cfg, translation, orientation, **kwargs)
    stage = prim.GetStage()
    n = 0
    for root in sim_utils.find_matching_prims(prim_path):
        for a, b in X2_FILTERED_PAIRS:
            pa, pb = stage.GetPrimAtPath(f"{root.GetPath()}/{a}"), stage.GetPrimAtPath(f"{root.GetPath()}/{b}")
            if not (pa.IsValid() and pb.IsValid()):
                raise RuntimeError(f"X2 link prims not found for the collision filter: {root.GetPath()}/{a}, {b}")
            UsdPhysics.FilteredPairsAPI.Apply(pa).CreateFilteredPairsRel().AddTarget(pb.GetPath())
            n += 1
    print(f"[X2] filtered {n} always-overlapping link pairs under {prim_path}")
    return prim


X2_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        func=spawn_x2,
        fix_base=False,
        replace_cylinders_with_capsules=True,
        merge_fixed_joints=True,
        asset_path=X2_URDF,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # pelvis is 0.641 m above the soles in this pose (MuJoCo FK); the G1 task spawns about 7 cm above standing.
        pos=(0.0, 0.0, 0.71),
        joint_pos={
            ".*_hip_pitch_joint": -0.312,
            ".*_knee_joint": 0.669,
            ".*_ankle_pitch_joint": -0.363,
            ".*_elbow_joint": -0.6,
            "left_shoulder_roll_joint": 0.2,
            "left_shoulder_pitch_joint": 0.2,
            "right_shoulder_roll_joint": -0.2,
            "right_shoulder_pitch_joint": 0.2,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators=x2_delayed_actuators,
)

# Joint order Isaac Lab gives this URDF (breadth first, alphabetical per level). Checked at run time by
# x2_port/check_x2_env.py; the mirror maps below are in this order.
X2_ISAAC_JOINTS = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_pitch_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_roll_joint",
    "left_knee_joint",
    "right_knee_joint",
    "head_yaw_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "head_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
]


def _mirror_name(n):
    return "right_" + n[5:] if n.startswith("left_") else "left_" + n[6:] if n.startswith("right_") else n


# mirrored[i] = original[mapping[i]] * sign[i]. Every X2 joint axis is the link x, y or z on both sides (URDF), so a
# left-right mirror negates roll (x) and yaw (z) joints and keeps pitch (y) joints.
X2_symmetric_augmentation_joint_mapping = [X2_ISAAC_JOINTS.index(_mirror_name(n)) for n in X2_ISAAC_JOINTS]
X2_symmetric_augmentation_joint_reverse_buf = [-1 if ("_roll_" in n or "_yaw_" in n) else 1 for n in X2_ISAAC_JOINTS]

X2_LINKS = ["pelvis", "waist_yaw_link", "waist_pitch_link", "torso_link", "head_yaw_link", "head_pitch_link"] + [
    f"{s}_{b}_link"
    for s in ("left", "right")
    for b in (
        "hip_pitch",
        "hip_roll",
        "hip_yaw",
        "knee",
        "ankle_pitch",
        "ankle_roll",
        "shoulder_pitch",
        "shoulder_roll",
        "shoulder_yaw",
        "elbow",
        "wrist_yaw",
        "wrist_pitch",
        "wrist_roll",
    )
]
