"""Run 4 (`Instinct-Parkour-Fixed-Amp-X2-v0`): the staged task with the port fixes from the independent audit
(`state/hiking_port_audit.md`, 2026-11-12). Each item is checked on the CPU (`code/x2_port/run4/`); departures from the paper
are marked.

  a  Discriminator input: joint positions and velocities of the 12 leg joints only (hip pitch/roll/yaw, knee, ankle pitch/roll,
     both sides), plus the base terms the G1 used (projected gravity, base linear and angular velocity). Head, arms and waist are
     dropped: the head is passive (the reference holds it at 0), the reference arms sit far from the X2's and collide with its
     thighs, and the reference waist leans 13 deg. The G1's discriminator saw all joints, but its reference arms are static at the
     default pose, so it effectively judged the legs. Departure (input selection).
  b  Default pose: legs from the v2 reference mean (left/right symmetrised), waist 0 (the reference mean of +0.229 rad is the
     lean), arms and head unchanged. Old (copied from the G1 numbers): hip pitch -0.312, knee 0.669, ankle pitch -0.363, hip
     roll/yaw 0, ankle roll 0. New: hip pitch -0.136, hip roll -/+0.043, hip yaw +/-0.181, knee 0.379, ankle pitch -0.114, ankle
     roll +/-0.057 (left/right). As on the G1, whose default equals its reference mean within 0.01 rad. Pelvis 0.679 m above the
     soles in this pose (MuJoCo FK); spawn at 0.75 m (about 7 cm drop, as before).
  c  Action scale per joint = the G1's per joint type (hip pitch/yaw and waist yaw 0.548, hip roll and knee 0.351, ankles, waist
     roll/pitch, shoulders, elbows and wrist roll 0.439, wrist pitch/yaw 0.0745) instead of 0.25 x effort / (armature x w^2)
     (hip 0.261, ankle pitch 0.258). Same action range in radians as the G1.
  d  Self-collision: pelvis vs hip-roll links and the same-side wrist links vs hip pitch/roll/yaw links are filtered, beside the
     five pairs filtered before (census over 3,000 reference frames: pelvis/hip_roll overlap in 50-57% of frames, wrist/hip in
     6-38%). Foot, ankle and knee pairs between the legs stay on (1-5% of frames: real leg crossings).
  e  soft_joint_pos_limit_factor 1.0 (was 0.9): the reference reaches the hard limits of hip roll, shoulder roll, ankle roll and
     elbow, so with 0.9 its normal poses sit in the penalty band. Departure (the G1 task uses 0.9 with wider G1 ranges).
  f  Actuators: implicit PD, no delay, as the G1 run actually ran (its `apply_shoe_config` replaces the delayed actuators with
     implicit ones). Gains, effort and velocity limits and armature as before (plant mc_20261024).
"""
from __future__ import annotations

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import instinctlab.assets.agibot_x2 as X

from .staged import X2ParkourStagedEnvCfg

SIDES = ("left", "right")
LEG_JOINTS = [f"{s}_{j}_joint" for s in SIDES for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]

RUN4_DEFAULT_POSE = {
    ".*_hip_pitch_joint": -0.136,
    "left_hip_roll_joint": -0.043,
    "right_hip_roll_joint": 0.043,
    "left_hip_yaw_joint": 0.181,
    "right_hip_yaw_joint": -0.181,
    ".*_knee_joint": 0.379,
    ".*_ankle_pitch_joint": -0.114,
    "left_ankle_roll_joint": 0.057,
    "right_ankle_roll_joint": -0.057,
    "waist_.*_joint": 0.0,
    ".*_elbow_joint": -0.6,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
}
RUN4_SPAWN_Z = 0.75

_G1_HIP_PY, _G1_HIP_R, _G1_DEFAULT, _G1_WRIST_PY = 0.5475464652142303, 0.3506614663788243, 0.43857731392336724, 0.07450087032950714
RUN4_ACTION_SCALE = {
    ".*_hip_yaw_joint": _G1_HIP_PY, ".*_hip_roll_joint": _G1_HIP_R, ".*_hip_pitch_joint": _G1_HIP_PY, ".*_knee_joint": _G1_HIP_R,
    ".*_ankle_pitch_joint": _G1_DEFAULT, ".*_ankle_roll_joint": _G1_DEFAULT,
    "waist_roll_joint": _G1_DEFAULT, "waist_pitch_joint": _G1_DEFAULT, "waist_yaw_joint": _G1_HIP_PY,
    ".*_shoulder_pitch_joint": _G1_DEFAULT, ".*_shoulder_roll_joint": _G1_DEFAULT, ".*_shoulder_yaw_joint": _G1_DEFAULT,
    ".*_elbow_joint": _G1_DEFAULT, ".*_wrist_roll_joint": _G1_DEFAULT, ".*_wrist_pitch_joint": _G1_WRIST_PY,
    ".*_wrist_yaw_joint": _G1_WRIST_PY,
}

RUN4_EXTRA_FILTERED_PAIRS = [("pelvis", f"{s}_hip_roll_link") for s in SIDES] + [
    (f"{s}_{h}_link", f"{s}_{w}_link") for s in SIDES for h in ("hip_pitch", "hip_roll", "hip_yaw") for w in ("wrist_yaw", "wrist_pitch", "wrist_roll")
]


def spawn_x2_run4(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """spawn_x2 (five always-overlapping pairs filtered), then the run-4 pairs from the reference census."""
    import isaaclab.sim as sim_utils
    from pxr import UsdPhysics

    prim = X.spawn_x2(prim_path, cfg, translation, orientation, **kwargs)
    stage = prim.GetStage()
    n = 0
    for root in sim_utils.find_matching_prims(prim_path):
        for a, b in RUN4_EXTRA_FILTERED_PAIRS:
            pa, pb = stage.GetPrimAtPath(f"{root.GetPath()}/{a}"), stage.GetPrimAtPath(f"{root.GetPath()}/{b}")
            if not (pa.IsValid() and pb.IsValid()):
                raise RuntimeError(f"X2 link prims not found for the run-4 collision filter: {root.GetPath()}/{a}, {b}")
            UsdPhysics.FilteredPairsAPI.Apply(pa).CreateFilteredPairsRel().AddTarget(pb.GetPath())
            n += 1
    print(f"[X2 run 4] filtered {n} more link pairs under {prim_path}")
    return prim


def make_x2_implicit_actuators_mc():
    """Implicit PD (no delay) with the mc_20261024 gains, effort / velocity limits and armature of the delayed set."""
    return {
        group: ImplicitActuatorCfg(
            joint_names_expr=names,
            effort_limit_sim={n: X._PLANT[n][1] for n in names},
            velocity_limit_sim={n: X._PLANT[n][2] for n in names},
            stiffness={n: X._first(X._MC_STIFFNESS, n) for n in names},
            damping={n: X._first(X._MC_DAMPING, n) for n in names},
            armature={n: X._PLANT[n][0] for n in names},
        )
        for group, names in X._GROUPS.items()
    }


@configclass
class X2ParkourFixedEnvCfg(X2ParkourStagedEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        robot = self.scene.robot
        self.scene.robot = robot.replace(
            spawn=robot.spawn.replace(func=spawn_x2_run4),
            init_state=robot.init_state.replace(pos=(0.0, 0.0, RUN4_SPAWN_Z), joint_pos=dict(RUN4_DEFAULT_POSE)),
            soft_joint_pos_limit_factor=1.0,
            actuators=make_x2_implicit_actuators_mc(),
        )
        self.actions.joint_pos.scale = dict(RUN4_ACTION_SCALE)
        for group in (self.observations.amp_policy, self.observations.amp_reference):
            for term in ("joint_pos_rel", "joint_vel"):
                old = getattr(group, term).params["asset_cfg"]  # a new object, so no other group that shares it changes
                getattr(group, term).params["asset_cfg"] = SceneEntityCfg(old.name, joint_names=list(LEG_JOINTS))
