"""Build the X2 MuJoCo model used for SuperMimic data collection.

Base: ``mjcf/x2_ultra.xml`` as-is (bodies, inertia, collision geoms, 12 foot spheres per foot,
waist limits from the real-robot capture, joint frictionloss 0.3). Changed at load time:

* timestep 5 ms, implicitfast (paper: 50 Hz control, 4 x 5 ms substeps);
* per-joint armature from the plant (``plant_config.load_plant``), as SONIC training does;
* the torque ``<motor>`` actuators are replaced by in-sim PD position servos at the plant
  kp/kd with force range = plant effort, one per joint in qpos order (actuator i drives
  joint i, so ``ctrl`` is in MuJoCo joint order).
"""

from __future__ import annotations

import pathlib
import sys
from dataclasses import dataclass

import mujoco
import numpy as np
import yaml

REPO = pathlib.Path(__file__).resolve().parents[2]
# plant_config imports no isaaclab, but the robots package __init__ does (same trick as eval_x2_mujoco.py)
_PLANT_DIR = REPO / "gear_sonic/envs/manager_env/robots"
if str(_PLANT_DIR) not in sys.path:
    sys.path.insert(0, str(_PLANT_DIR))
from plant_config import Plant, load_plant  # noqa: E402

from gear_sonic.supermimic.robot_cfg import CFG  # noqa: E402

MJCF_PATH = CFG["mjcf"] if CFG else REPO / "gear_sonic/data/assets/robot_description/mjcf/x2_ultra.xml"
NUM_HINGES = CFG["num_joints"] if CFG else 31

SIM_DT = 0.005
DECIMATION = 4
CONTROL_DT = SIM_DT * DECIMATION


@dataclass
class X2Model:
    model: mujoco.MjModel
    plant: Plant
    joint_names: list[str]  # MuJoCo qpos order (31 hinges)
    kp: np.ndarray
    kd: np.ndarray
    effort: np.ndarray
    armature: np.ndarray
    action_scale: np.ndarray
    default_q: np.ndarray
    q_lo: np.ndarray
    q_hi: np.ndarray


# Foot shell collider (operator, 2026-10-22: the real feet hit each other; the MJCF foot shell is visual-only and only the
# 12 sole spheres collide, so simulated feet could pass through each other). Box = the ankle_roll visual mesh extent
# in the link frame (x -0.072..0.148, y +-0.065, z -0.073..0.032). It touches the floor never: it has no bitmask
# collisions and only explicit pairs with the OTHER foot's box and the other leg's knee / ankle-pitch links.
FOOT_SHELL_POS = (0.038, 0.0, -0.0205)
FOOT_SHELL_HALF = (0.110, 0.065, 0.0525)


def _add_foot_shells(spec: mujoco.MjSpec):
    bodies = {b.name: b for b in spec.bodies}
    for side in ("left", "right"):
        g = bodies[f"{side}_ankle_roll_link"].add_geom(name=f"{side}_foot_shell", type=mujoco.mjtGeom.mjGEOM_BOX)
        g.size = FOOT_SHELL_HALF
        g.pos = FOOT_SHELL_POS
        g.contype, g.conaffinity, g.group = 0, 0, 3
        g.mass = 0.0  # collision only; keep the URDF inertials
        g.density = 0.0
    leg_geoms = {}
    for geom in spec.geoms:
        b = geom.parent.name if hasattr(geom, "parent") else ""
        for side in ("left", "right"):
            if b in (f"{side}_knee_link", f"{side}_ankle_pitch_link") and (geom.contype or geom.conaffinity):
                if not geom.name:
                    geom.name = f"{b}_collision_{len(leg_geoms.setdefault(side, []))}"
                leg_geoms.setdefault(side, []).append(geom.name)
    spec.add_pair(geomname1="left_foot_shell", geomname2="right_foot_shell")
    for side, other in (("left", "right"), ("right", "left")):
        for name in leg_geoms.get(other, []):
            spec.add_pair(geomname1=f"{side}_foot_shell", geomname2=name)


# Terrain foot box: the foot shell's footprint, its underside TERRAIN_FOOT_LIFT above the sole spheres' lowest points
# (ankle frame z -0.073), so standing on any flat surface only the spheres touch; the box meets riser faces, edges and
# box sides.
TERRAIN_FOOT_LIFT = 0.015
_TF_BOTTOM, _TF_TOP = -0.073 + TERRAIN_FOOT_LIFT, FOOT_SHELL_POS[2] + FOOT_SHELL_HALF[2]
TERRAIN_FOOT_POS = (FOOT_SHELL_POS[0], FOOT_SHELL_POS[1], (_TF_BOTTOM + _TF_TOP) / 2)
TERRAIN_FOOT_HALF = (FOOT_SHELL_HALF[0], FOOT_SHELL_HALF[1], (_TF_TOP - _TF_BOTTOM) / 2)


def _add_terrain_feet(spec: mujoco.MjSpec):
    """Foot boxes that collide only with the terrain (props, heightfield; terrain.TERRAIN_BIT), not with the floor or
    the robot, and sit TERRAIN_FOOT_LIFT above the sole spheres: floor and tread contact stay the 12 sole spheres the
    policy was trained on, while a toe or heel can no longer pass into a stair riser or a box side."""
    from gear_sonic.supermimic.terrain import TERRAIN_BIT

    bodies = {b.name: b for b in spec.bodies}
    for side in ("left", "right"):
        g = bodies[f"{side}_ankle_roll_link"].add_geom(name=f"{side}_foot_terrain", type=mujoco.mjtGeom.mjGEOM_BOX)
        g.size, g.pos = TERRAIN_FOOT_HALF, TERRAIN_FOOT_POS
        g.contype, g.conaffinity, g.group = TERRAIN_BIT, 0, 3
        g.mass, g.density = 0.0, 0.0


def build_x2(plant_name: str | None = None, mjcf_path: pathlib.Path = MJCF_PATH, foot_shells: bool = False,
             prop_slots: list | None = None, raster: dict | None = None, foot_terrain: bool = False,
             walls: list | None = None) -> X2Model:
    # foot_shells stays OFF until tested in mujoco_warp and in a local run (operator 2026-10-22: the first Nebius run
    # uses only what local_v2 verified).
    plant = Plant(yaml.safe_load(CFG["plant"].read_text(encoding="utf-8")), CFG["plant"]) if CFG else load_plant(plant_name)
    spec = mujoco.MjSpec.from_file(str(mjcf_path))
    if CFG:  # visual meshes: no collision, and the inertials are explicit, so they only cost compile time and GPU memory
        for g in [g for g in spec.geoms if g.type == mujoco.mjtGeom.mjGEOM_MESH]:
            spec.delete(g)
        for a in list(spec.meshes):
            spec.delete(a)
    spec.option.timestep = SIM_DT
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

    hinges = [j for j in spec.joints if j.type == mujoco.mjtJoint.mjJNT_HINGE]
    names = [j.name for j in hinges]
    assert len(names) == NUM_HINGES, names

    for act in list(spec.actuators):
        spec.delete(act)
    kp, kd, eff, arm = (np.array([f(n) for n in names]) for f in (plant.kp_for, plant.kd_for, plant.effort_for, plant.armature_for))
    for j, n, p, d, e, a in zip(hinges, names, kp, kd, eff, arm):
        j.armature = a
        act = spec.add_actuator(name=f"pd_{n}", target=n, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        act.gainprm[0] = p
        act.biasprm[:3] = [0.0, -p, -d]
        act.ctrllimited = mujoco.mjtLimited.mjLIMITED_FALSE
        act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
        act.forcerange = [-e, e]

    if foot_shells:
        _add_foot_shells(spec)
    if prop_slots:  # terrain: one mocap box per prop slot (terrain.Terrain), placed per world at reset
        from gear_sonic.supermimic.terrain import add_slots

        add_slots(spec, prop_slots)
    if raster is not None:  # the course height raster as a static heightfield, shared by every world
        from gear_sonic.supermimic.terrain import add_hfield

        add_hfield(spec, raster)
    if walls:  # tall course geometry (walls, door jambs, posts, rails) as static boxes, shared by every world
        from gear_sonic.supermimic.terrain import add_walls

        add_walls(spec, walls)
    if foot_terrain:
        _add_terrain_feet(spec)
    model = spec.compile()
    jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names]
    assert [int(model.jnt_qposadr[i]) for i in jids] == list(range(7, 7 + NUM_HINGES)), "hinges must follow the free joint in qpos"
    rng = model.jnt_range[jids]
    return X2Model(
        model=model, plant=plant, joint_names=names, kp=kp, kd=kd, effort=eff, armature=arm,
        action_scale=np.array([plant.action_scale_for(n) for n in names]),
        default_q=np.array([plant.default_pos_for(n) for n in names]),
        q_lo=rng[:, 0].copy(), q_hi=rng[:, 1].copy(),
    )  # fmt: skip


if __name__ == "__main__":
    x2 = build_x2()
    m = x2.model
    print(f"plant={x2.plant.name} nq={m.nq} nv={m.nv} nu={m.nu} ngeom={m.ngeom} dt={m.opt.timestep}")
    for i in (0, 4, 13, 22, 29):
        print(f"  {x2.joint_names[i]:28s} kp={x2.kp[i]:8.2f} kd={x2.kd[i]:6.3f} eff={x2.effort[i]:6.1f} "
              f"arm={x2.armature[i]:.5f} scale={x2.action_scale[i]:.4f} q0={x2.default_q[i]:+.3f}")  # fmt: skip
