"""X2 Ultra MuJoCo model for SSR: 21 actuated joints as in the paper, the other 10 held at default by their PD.

Paper joint block order (App. B.1): [R leg, L leg, waist yaw, R arm, L arm]; leg = hip pitch/roll/yaw, knee,
ankle pitch/roll; arm = shoulder pitch/roll/yaw, elbow. Everything policy-facing uses that order (`idx21`).
Physics: 200 Hz, implicitfast, in-sim PD at the deploy plant's gains, the same build as SuperMimic.
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass

import mujoco
import numpy as np

from ssr.plant_config import load_plant

ROOT = pathlib.Path(__file__).resolve().parents[1]
MJCF = ROOT / "assets/robot/mjcf/x2_ultra.xml"
SIM_DT, DECIMATION = 0.005, 4
CONTROL_DT = SIM_DT * DECIMATION
TERRAIN_BIT = 2

_LEG = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
_ARM = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow")
NAMES21 = ([f"right_{j}_joint" for j in _LEG] + [f"left_{j}_joint" for j in _LEG] + ["waist_yaw_joint"]
           + [f"right_{j}_joint" for j in _ARM] + [f"left_{j}_joint" for j in _ARM])
FOOT_BODIES = ("right_ankle_roll_link", "left_ankle_roll_link")       # order: R, L (the paper's foot order)
HAND_BODIES = ("right_wrist_roll_link", "left_wrist_roll_link")
# The built-in Orbbec Gemini 335 (URDF joint rgbd_head_front): on head_pitch_link, 40 deg below horizontal at head 0.
CAM_POS = (0.05761, 0.0, -0.04837)                                     # y (-1.1 cm in the URDF) centred: mirror symmetry
CAM_PITCH_DOWN = float(np.deg2rad(40.0))
CAM_FX_720 = 692.4                                                     # factory fx at 1280x720
CAM_FOV = float(2 * np.arctan(360.0 / CAM_FX_720))                     # square centre crop of the stream
# Experiments on where the depth camera sits. head: the robot's built-in Gemini 335. chest: an added down-looking camera on the torso,
# in front of the chest shell, same optics.
CAMERAS = {
    "head": dict(body="head_pitch_link", pos=CAM_POS, pitch=CAM_PITCH_DOWN),
    "chest": dict(body="torso_link", pos=(0.155, 0.0, 0.20), pitch=float(np.deg2rad(75.0))),
    "chest62": dict(body="torso_link", pos=(0.155, 0.0, 0.20), pitch=float(np.deg2rad(62.0))),
    "chest85": dict(body="torso_link", pos=(0.155, 0.0, 0.20), pitch=float(np.deg2rad(85.0))),
    # User spec 2026-11-11: see the ground from about 10 cm behind the toe tips to 2 m ahead, feet in view, range 3 m. Mount point PROVISIONAL (the G1 torso camera
    # offset of the Hiking track put on the X2 torso link, y centred) until the Hiking track fixes the X2 offset; fov = square field of view of the 36 x 36 image.
    # The user's real setup (2026-11-11): a RealSense D435-class camera held on the chest, centred, just above the chest sensor window, pitched down (tilt B).
    # Depth field of view 87 x 58 deg resampled to the 36 x 36 image. Pose solved by the coordinator from the user's infrared frame: 0.90 m high, lens 0.12 m behind the toe tips, 53 deg down; mount point to be replaced by the Hiking track's value.
    "chest_rs": dict(body="torso_link", pos=(0.126, 0.0, 0.16), pitch=float(np.deg2rad(59.5)), fov=float(np.deg2rad(58.0)), fov_h=float(np.deg2rad(87.0)), dmax=3.0, march=38,
                     dr_pos=(0.02, 0.01, 0.05), dr_rpy_deg=(2.0, 4.0, 2.0)),   # per-episode mount randomisation (user, 2026-11-11): forward +-2 cm, height +-5 cm, pitch +-4 deg, roll / yaw +-2 deg
    "torso60": dict(body="torso_link", pos=(0.0488, 0.0, 0.4378), pitch=float(np.deg2rad(60.0)), fov=float(np.deg2rad(64.0)), dmax=3.0, march=38),
}
# Self-occlusion in the depth image: spheres on the limbs (body, local position, radius)
OCCLUDERS = [(f"{s}_{b}", p, r) for s in ("left", "right") for b, p, r in (
    ("hip_yaw_link", (0.0, 0.0, -0.07), 0.08), ("hip_yaw_link", (0.0, 0.0, -0.20), 0.075), ("knee_link", (0.0, 0.0, 0.0), 0.07), ("knee_link", (0.0, 0.0, -0.10), 0.055),
    ("knee_link", (0.0, 0.0, -0.20), 0.05), ("ankle_roll_link", (-0.03, 0.0, -0.035), 0.05), ("ankle_roll_link", (0.06, 0.0, -0.04), 0.05), ("ankle_roll_link", (0.13, 0.0, -0.05), 0.04),
    ("elbow_link", (0.0, 0.0, 0.0), 0.05), ("wrist_roll_link", (0.05, 0.0, 0.0), 0.05))]
SOLE_HALF = (0.1125, 0.05)                                             # paper: 22.5 x 10 cm sole patch
SOLE_OFFSET_X = 0.038                                                  # sole centre ahead of the ankle (mesh extent)
SOLE_Z = 0.073                                                         # sole below the ankle origin
TERMINATE_BODIES = ("left_elbow_link", "right_elbow_link")   # contact forces; trunk / thighs / head end by clearance (env._done)
CLEARANCE = (("pelvis", 0.12), ("torso_link", 0.15), ("head_pitch_link", 0.10), ("left_hip_yaw_link", 0.08), ("right_hip_yaw_link", 0.08))


@dataclass
class X2:
    model: mujoco.MjModel
    joint_names: list
    idx21: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    effort: np.ndarray
    action_scale: np.ndarray
    default_q: np.ndarray
    q_lo: np.ndarray
    q_hi: np.ndarray
    qd_max: np.ndarray
    default_q_plant: np.ndarray = None     # the plant's default pose (differs from default_q under X2_DEFAULT_FROM_REF)


def _terrain_guards(spec):
    """Only spheres touch the heightfield (mesh links on a 2.5 cm heightfield overflow mujoco_warp's collision buffer and
    cost most of the step). The 12 sole spheres also get the terrain bit. Added here: toe / heel / side spheres 15 mm above
    the sole so a foot cannot pass into a riser, and spheres along the shin and arm. Spheres over about 5 cm also overflow it,
    so the trunk, thighs and head end the episode by terrain clearance instead (env._done). Meshes
    still collide with each other and with wall boxes."""
    bodies = {b.name: b for b in spec.bodies}
    def add(body, name, pos, r):
        g = bodies[body].add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_SPHERE)
        g.size, g.pos = (r, 0, 0), pos
        g.contype, g.conaffinity, g.group = TERRAIN_BIT, 0, 3
        g.mass, g.density = 0.0, 0.0
    for g in spec.geoms:
        if g.type == mujoco.mjtGeom.mjGEOM_SPHERE and g.contype and g.parent.name.endswith("ankle_roll_link"):
            g.contype |= TERRAIN_BIT
            r_sole = float(__import__("os").environ.get("SSR_SOLE_R", "0.02"))
            g.pos = (g.pos[0], g.pos[1], g.pos[2] + r_sole - g.size[0])  # same lowest point, a sphere big enough for the heightfield
            g.size = (r_sole, 0, 0)
    import os
    mode = os.environ.get("SSR_GUARD", "all")
    if mode == "sole":
        return
    r, z = 0.022, -0.073 + 0.015 + 0.022
    for side in ("left", "right"):
        for i, p in enumerate([(x, y, z) for x in (-0.05, 0.03, 0.125) for y in (-0.04, 0.04)]):
            add(f"{side}_ankle_roll_link", f"{side}_foot_guard{i}", p, r)
        if mode == "foot":
            continue
        shin = np.array(bodies[f"{side}_ankle_pitch_link"].pos)
        for i, t in enumerate((0.0, 0.33, 0.66)):
            add(f"{side}_knee_link", f"{side}_shin_guard{i}", tuple(shin * t), 0.05)
        add(f"{side}_elbow_link", f"{side}_elbow_guard", (0.0, 0.0, 0.0), 0.045)
        add(f"{side}_wrist_roll_link", f"{side}_hand_guard", (0.04, 0.0, 0.0), 0.045)


def build(hfield: dict | None = None, walls: list | None = None, foot_boxes: bool = True) -> X2:
    """hfield: dict(z (nx, ny) metres, x0, y0, res, base); walls: boxes (center, half, quat wxyz)."""
    plant = load_plant()
    spec = mujoco.MjSpec.from_file(str(MJCF))
    spec.option.timestep = SIM_DT
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    hinges = [j for j in spec.joints if j.type == mujoco.mjtJoint.mjJNT_HINGE]
    names = [j.name for j in hinges]
    assert len(names) == 31
    for a in list(spec.actuators):
        spec.delete(a)
    kp, kd, eff, arm = (np.array([f(n) for n in names]) for f in (plant.kp_for, plant.kd_for, plant.effort_for, plant.armature_for))
    # The policy does not command waist pitch and roll (paper: 21 joints, waist yaw only). At the plant's gains (kp 12) they cannot hold the
    # torso, which sagged onto its limits (20 deg pitch, 30 deg roll), so they are held stiffly at zero, here and on the robot.
    if not plant.stiffness:                                              # armature-rule plants only; a plant with measured gains (mc_20261024) is used as it is
        for n_ in ("waist_pitch_joint", "waist_roll_joint"):
            i_ = names.index(n_)
            kp[i_], kd[i_] = 120.0, 6.0
    for j, n, p, d, e, a in zip(hinges, names, kp, kd, eff, arm):
        j.armature = a
        act = spec.add_actuator(name=f"pd_{n}", target=n, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        act.gaintype, act.biastype = mujoco.mjtGain.mjGAIN_FIXED, mujoco.mjtBias.mjBIAS_AFFINE
        act.gainprm[0] = p
        act.biasprm[:3] = [0.0, -p, -d]
        act.ctrllimited = mujoco.mjtLimited.mjLIMITED_FALSE
        act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
        act.forcerange = [-e, e]
    if hfield is not None:
        z = hfield["z"]
        nx, ny = z.shape
        res, base = hfield["res"], hfield.get("base", 0.1)
        zmin, zmax = float(z.min()), float(z.max())
        top = max(zmax - zmin, 1e-3)
        import os as _os2
        CH = int(_os2.environ.get("X2_HF_CHUNK", "400"))                 # 2026-11-12: >0 splits the raster into fields of <= CH x CH vertices
        if CH <= 1:                                                     # (MuJoCo Warp mis-collides on one very large field; ACP finding)
            hf = spec.add_hfield(name="terrain", nrow=ny, ncol=nx, size=[(nx - 1) * res / 2, (ny - 1) * res / 2, top, base])
            hf.userdata = ((z - zmin) / top).T.astype(np.float32).ravel().tolist()
            g = spec.worldbody.add_geom(name="terrain", type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname="terrain",
                                        pos=[hfield["x0"] + (nx - 1) * res / 2, hfield["y0"] + (ny - 1) * res / 2, zmin],
                                        rgba=[0.55, 0.5, 0.45, 1.0])
            g.contype, g.conaffinity = 0, TERRAIN_BIT
        else:
            for i0 in range(0, nx - 1, CH - 1):                           # chunks share their edge vertices
                for j0 in range(0, ny - 1, CH - 1):
                    sub = z[i0:min(i0 + CH, nx), j0:min(j0 + CH, ny)]
                    sx, sy = sub.shape
                    if sx < 2 or sy < 2:
                        continue
                    smin = float(sub.min()); stop = max(float(sub.max()) - smin, 1e-3)
                    nm = f"terrain_{i0}_{j0}"
                    hf = spec.add_hfield(name=nm, nrow=sy, ncol=sx, size=[(sx - 1) * res / 2, (sy - 1) * res / 2, stop, base])
                    hf.userdata = ((sub - smin) / stop).T.astype(np.float32).ravel().tolist()
                    g = spec.worldbody.add_geom(name=nm, type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname=nm,
                                                pos=[hfield["x0"] + (i0 + (sx - 1) / 2) * res, hfield["y0"] + (j0 + (sy - 1) / 2) * res, smin],
                                                rgba=[0.55, 0.5, 0.45, 1.0])
                    g.contype, g.conaffinity = 0, TERRAIN_BIT
        for geom in spec.geoms:                                         # the hfield is the floor
            if geom.type == mujoco.mjtGeom.mjGEOM_PLANE:
                spec.delete(geom)
                break
    for i, b in enumerate(walls or []):
        g = spec.worldbody.add_geom(name=f"wall_{i}", type=mujoco.mjtGeom.mjGEOM_BOX, size=b["half"], pos=b["center"],
                                    quat=b["quat"], rgba=[0.7, 0.62, 0.5, 1.0])
        g.conaffinity = 1 | TERRAIN_BIT
    if foot_boxes:
        _terrain_guards(spec)
    import os as _os
    if _os.environ.get("X2_EXCL_PELVIS_HIPROLL") == "1":                   # deviation (step 1, 2026-11-12): hull overlap in reference poses
        for _s in ("left", "right"):
            spec.add_exclude(bodyname1="pelvis", bodyname2=f"{_s}_hip_roll_link")
    m = spec.compile()
    jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names]
    assert [int(m.jnt_qposadr[i]) for i in jid] == list(range(7, 38))
    rng = m.jnt_range[jid]
    limits = json.load(open(ROOT / "config/joint_velocity_limits.json"))     # URDF joint velocity limits
    qd_max = np.array([limits[n] for n in names])
    dq_plant = np.array([plant.default_pos_for(n) for n in names])
    dq = dq_plant.copy()
    if _os.environ.get("X2_DEFAULT_FROM_REF") == "1":                     # deviation (step 2, 2026-11-12): default leg pose = reference mean
        _ref = json.load(open(ROOT / "config/default_ref_v1.json"))["default_q"]
        for _n, _v in _ref.items():
            dq[names.index(_n)] = _v
    asc = np.array([plant.action_scale_for(n) for n in names])
    if _os.environ.get("X2_ACTION_SCALE_PAPER") == "1":                  # paper Table 14 (needs the user's agreement under the plant rule): 0.25 legs / waist, 0.2 arms
        asc = np.array([0.2 if ("shoulder" in n or "elbow" in n or "wrist" in n) else 0.25 for n in names])
    return X2(m, names, np.array([names.index(n) for n in NAMES21]), kp, kd, eff,
              asc, dq,
              rng[:, 0].copy(), rng[:, 1].copy(), qd_max, dq_plant)


# ---- mirror (App. B.1), in the 21-joint order. Checked against the model's kinematics in tests/test_mirror.py.
S_LEG = np.array([1, -1, -1, 1, 1, -1.0])
S_ARM = np.array([1, -1, -1, 1.0])
MIRROR_PERM = np.concatenate((np.arange(6, 12), np.arange(0, 6), [12], np.arange(17, 21), np.arange(13, 17)))
MIRROR_SIGN = np.concatenate((S_LEG, S_LEG, [-1.0], S_ARM, S_ARM))
