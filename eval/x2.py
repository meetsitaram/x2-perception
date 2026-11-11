"""The X2 Ultra in MuJoCo (CPU) on an apparatus, for the common eval. Same model files for every track.

Robot files: PA_ROBOT_DIR, else the desktop copy (ssr-x2/assets/robot), else /mnt/filesystem-g4/data/robots/x2_ultra.
Physics: SIM_DT, implicitfast, PD inside the simulator (affine position actuators, torque-limited). The gains come from
the adapter (each track trains against its own plant); the default is config/x2_plant.json, the X2 deploy plant.
Collision: the robot's own collision geoms (12 sole spheres per foot, convex meshes elsewhere) against the walk raster
(a height field) and the wall boxes.
"""
from __future__ import annotations

import json
import math
import os
import pathlib
from dataclasses import dataclass, field

import mujoco
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SIM_DT = 0.0025
GROUP_TERRAIN, GROUP_WALL = 4, 5          # geom groups (robot: visual 1 / 2, collision 3)
FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")
SOLE_HALF = (0.1125, 0.05)                # m: sole rectangle, centred SOLE_X ahead of the ankle, SOLE_Z below it
SOLE_X, SOLE_Z = 0.038, 0.073
SOLE_R = 0.02                             # m: radius the 12 sole spheres per foot are given (see build)
STAND_Z = 0.68                            # m: pelvis height of the MJCF's standing pose (straight legs)


def robot_dir() -> pathlib.Path:
    for p in (os.environ.get("PA_ROBOT_DIR"), "C:/Users/PC/projects/ssr-x2/assets/robot", "/mnt/filesystem-g4/data/robots/x2_ultra"):
        if p and (pathlib.Path(p) / "mjcf" / "x2_ultra.xml").exists():
            return pathlib.Path(p)
    raise FileNotFoundError("X2 model not found; set PA_ROBOT_DIR to the folder holding mjcf/x2_ultra.xml")


def _match(table, joint, default=None):
    for k, v in table.items():
        if k in joint:
            return float(v)
    if default is None:
        raise KeyError(joint)
    return default


def default_plant(joint_names):
    """kp, kd, effort, armature, default_q arrays (model joint order) of the deploy plant."""
    p = json.load(open(HERE / "config" / "x2_plant.json"))
    w = 2 * math.pi * p["natural_freq_hz"]
    arm = np.array([_match(p["armature"], j) for j in joint_names])
    return dict(kp=arm * w * w, kd=2 * p["damping_ratio"] * arm * w, effort=np.array([_match(p["effort"], j) for j in joint_names]),
                armature=arm, default_q=np.array([_match(p["default_joint_pos"], j, 0.0) for j in joint_names]))


@dataclass
class X2:
    model: mujoco.MjModel
    joint_names: list
    plant: dict
    qadr: np.ndarray
    dadr: np.ndarray
    terrain_geom: int
    wall_geoms: set
    body_group: dict = field(default_factory=dict)       # body id -> "foot" | "shin" | "knee" | "thigh" | "trunk" | "arm" | "head"
    foot_ids: tuple = ()


def joint_names_of(mjcf: pathlib.Path):
    spec = mujoco.MjSpec.from_file(str(mjcf))
    return [j.name for j in spec.joints if j.type == mujoco.mjtJoint.mjJNT_HINGE]


def _group_of(body_name: str):
    n = body_name
    if "ankle_roll" in n:
        return "foot"
    if "ankle_pitch" in n:
        return "shin"                      # the ankle pitch link sits at the bottom of the shin
    if "knee" in n:
        return "knee"                      # knee_link is the shank: knee joint down to the ankle
    if "hip" in n:
        return "thigh"
    if "head" in n:
        return "head"
    if any(k in n for k in ("shoulder", "elbow", "wrist")):
        return "arm"
    return "trunk"                         # pelvis, waist links, torso


def build(app=None, plant_override: dict | None = None, friction: float = 1.0, render_size=(1280, 720),
          wall_alpha: float = 1.0) -> X2:
    """app: apparatus.Apparatus or None (flat plane). plant_override: any of kp, kd, effort, armature, default_q (31,)."""
    rd = robot_dir()
    spec = mujoco.MjSpec.from_file(str(rd / "mjcf" / "x2_ultra.xml"))
    for cand in (rd / "urdf" / "x2_ultra" / "meshes", rd / "urdf" / "meshes"):
        if cand.exists():
            spec.meshdir = str(cand)
            break
    spec.option.timestep = SIM_DT
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    hinges = [j for j in spec.joints if j.type == mujoco.mjtJoint.mjJNT_HINGE]
    names = [j.name for j in hinges]
    assert len(names) == 31, len(names)
    plant = default_plant(names)
    for k, v in (plant_override or {}).items():
        plant[k] = np.asarray(v, float)
        assert plant[k].shape == (31,), k
    for a in list(spec.actuators):
        spec.delete(a)
    for j, n, p, d, e, arm in zip(hinges, names, plant["kp"], plant["kd"], plant["effort"], plant["armature"]):
        j.armature = arm
        act = spec.add_actuator(name="pd_" + n, target=n, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        act.gaintype, act.biastype = mujoco.mjtGain.mjGAIN_FIXED, mujoco.mjtBias.mjBIAS_AFFINE
        act.gainprm[0] = p
        act.biasprm[:3] = [0.0, -p, -d]
        act.ctrllimited = mujoco.mjtLimited.mjLIMITED_FALSE
        act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
        act.forcerange = [-e, e]
    # Sole spheres are 5 mm in the MJCF. Against a height field a sphere that small gets tilted contact normals on flat
    # ground (27 % of standing contacts off vertical, down to 0.70). At SOLE_R = 2 cm with the same lowest point it is
    # 15 %, never below 0.92. ssr-x2 makes the same change for the same reason.
    for g in spec.geoms:
        if g.type == mujoco.mjtGeom.mjGEOM_SPHERE and g.parent.name.endswith("ankle_roll_link"):
            g.pos = [g.pos[0], g.pos[1], g.pos[2] + SOLE_R - g.size[0]]
            g.size = [SOLE_R, 0, 0]
    for g in spec.geoms:                                   # the MJCF's own floor plane goes; the apparatus is the floor
        if g.type == mujoco.mjtGeom.mjGEOM_PLANE:
            spec.delete(g)
            break
    mat = spec.add_material(name="deck", rgba=[0.74, 0.70, 0.62, 1])
    if app is None:
        g = spec.worldbody.add_geom(name="terrain", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[20, 20, 0.1], material="deck")
    else:
        z = app.grid
        zmin, top = float(z.min()), max(float(z.max() - z.min()), 1e-3)
        hf = spec.add_hfield(name="terrain", nrow=app.ny, ncol=app.nx, size=[(app.nx - 1) * app.res / 2, (app.ny - 1) * app.res / 2, top, 0.2])
        hf.userdata = ((z - zmin) / top).T.astype(np.float32).ravel().tolist()
        g = spec.worldbody.add_geom(name="terrain", type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname="terrain", material="deck",
                                    pos=[app.x0 + (app.nx - 1) * app.res / 2, app.y0 + (app.ny - 1) * app.res / 2, zmin])
    g.group, g.contype, g.conaffinity = GROUP_TERRAIN, 1, 1
    g.friction = [friction, 0.005, 0.0001]
    spec.add_material(name="wall", rgba=[0.80, 0.66, 0.45, wall_alpha])
    for i, b in enumerate(app.boxes if app is not None else []):
        w = spec.worldbody.add_geom(name="wall_%d" % i, type=mujoco.mjtGeom.mjGEOM_BOX, size=list(b["half"]), pos=list(b["center"]),
                                    quat=list(b["quat"]), material="wall")
        w.group, w.contype, w.conaffinity = GROUP_WALL, 1, 1
        w.friction = [friction, 0.005, 0.0001]
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.70, 0.78, 0.88], rgb2=[0.25, 0.28, 0.33], width=256, height=256)
    spec.visual.global_.offwidth, spec.visual.global_.offheight = render_size
    spec.visual.headlight.ambient = [0.45, 0.45, 0.45]
    spec.visual.headlight.diffuse = [0.55, 0.55, 0.55]
    spec.visual.quality.shadowsize = 4096
    spec.worldbody.add_light(name="sun", pos=[0, 0, 10], dir=[0.3, 0.2, -1], diffuse=[0.5, 0.5, 0.5], castshadow=True,
                             type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    m = spec.compile()
    jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names]
    x = X2(m, names, plant, np.array([m.jnt_qposadr[i] for i in jid]), np.array([m.jnt_dofadr[i] for i in jid]),
           mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "terrain"),
           {g for g in range(m.ngeom) if m.geom_group[g] == GROUP_WALL})
    for b in range(1, m.nbody):
        x.body_group[b] = _group_of(m.body(b).name)
    x.foot_ids = tuple(m.body(n).id for n in FOOT_BODIES)
    return x


def yaw_of(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi
