"""CPU-only MuJoCo sim2sim player for the "Hiking in the Wild" parkour policy on the AgiBot X2 Ultra.

Port of hiking-in-the-wild/scripts/sim2sim_mujoco.py (Unitree G1). Never uses the GPU for compute: CUDA is hidden and
ONNX runs on CPUExecutionProvider. (The optional mp4 uses MuJoCo's OpenGL offscreen renderer for drawing only.)

Every run setting is read from the run's params/env.yaml (--run) and, when an export folder with adapter_card.json is
given (--export), asserted equal to the card: PD gains, armature, effort / velocity limits, default pose, action scale,
observation layout, command rule, and the depth camera (body, offset, quaternion, apertures = field of view, crop, depth
range, frame shape, history). Nothing of run 1 is hard-coded; run 1 (head camera, armature-rule gains, 18x32 frames,
5408 obs) and run 2 (torso camera, measured gains, 36x32 frames, 10016 obs) both load. Robot facts that are not in
env.yaml stay in this file: the URDF, the filtered collision pairs of the training asset, the sole spheres.

What it reproduces from the Isaac training env:
  * robot: the training URDF (x2_ultra.urdf) with the massless base_link removed -> pelvis is the free root, fixed links
    fused, collision = convex hull of each link's collision mesh (MuJoCo collides meshes as their convex hull),
    self-collision on except the pairs the training filtered (torso/head_pitch, knee/ankle_roll, wrist_yaw/wrist_roll)
  * actuators: explicit PD, tau = kp (q* - q) - kd qd clipped to the effort limit, recomputed every 5 ms physics step;
    armature per joint; sim dt 0.005, decimation 4 (policy 50 Hz); q* = default + action_scale * action for the 29
    commanded joints, the two head joints are held at 0 by their own PD
  * observation: base_ang_vel*0.25 | projected_gravity | velocity_commands | joint_pos-default (31) |
    joint_vel*0.05 (31) | last action (29), each an 8-step history oldest first; then 8 depth frames oldest first
    (run 1: 18x32 -> 5408 values; run 2: 36x32 -> 10016)
  * depth: ray-cast pinhole camera on the body named in env.yaml (run 1: head_pitch_link, 40 deg down; run 2:
    torso_link, 61.6 deg down), min distance 0.1 m, crop, gaussian blur k3 sigma 1, clip/normalise to the depth range,
    37-frame 50 Hz history, 8 frames every 5th; rays hit the terrain and the robot's own visual meshes
  * command: PoseVelocityCommand from a target point: vx = clip(2*x_b, 0, vmax), vy = 0, yaw rate = clip(2*heading
    error, -1, 1), all zero inside the 0.4 m stop radius

Deviations from Isaac (PhysX) -- none of them tuned:
  * physics engine: MuJoCo soft-constraint contacts (default solref/solimp, pyramidal cone, Newton solver) instead of
    PhysX TGS; integrator implicitfast. Friction 1.0 (training randomises the robot material 0.3..1.6 and multiplies by
    the terrain's 1.0); restitution 0.
  * collision hulls: MuJoCo uses the exact convex hull of each mesh; PhysX convex-hull cooking limits a hull to 64
    vertices. Hull-on-box contacts use MuJoCo multiccd (on by default in MuJoCo 3.16: several contact points for a flat sole).
  * --foot spheres replaces the foot hull by the 12 sole spheres (5 mm) of the gwbc MJCF: NOT what training used.
  * action delay: training randomises 0..2 physics steps of command delay per robot; here a fixed --delay-steps (default 0).
  * no observation noise (training adds uniform noise: gyro +-0.2, gravity +-0.05, joint pos +-0.01, joint vel +-0.5).
  * depth timing: in the Isaac recording the newest depth frame of control step t is the image of step t-1 (see
    validation.txt); --depth-lag 1 (default) reproduces that, 0 uses the current image.
  * PhysX clamps joint velocity to velocity_limit_sim; MuJoCo does not (exceedances are counted and reported).
  * joint limits: URDF hard limits in both; no joint friction / damping in either (URDF has none).
  * terrain: flat plane, or a clean stair tile made of boxes (20 cm risers, 30 cm treads); the Isaac tile is a 5 cm
    height-field mesh with perlin roughness and optional walls, which are not reproduced.
  * no domain randomisation, no pushes, no reset noise (unless a recorded state is given).

usage (from this folder, python = C:/Users/PC/.venvs/hiking/Scripts/python.exe):
  python sim2sim_mujoco_x2.py --name x2_ckpt500_mujoco_flat --terrain flat --seconds 14 --vmax 0.74      # run 1 (default)
  python sim2sim_mujoco_x2.py --name x2_ckpt500_mujoco_stairs --terrain stairs --seconds 14 --vmax 0.74
  python sim2sim_mujoco_x2.py --run assets/run2 --export assets/eval_v2_250 --name x2v2_ckpt250_mujoco_flat --no-video
  python validate_x2.py            # run 1 against the Isaac recording, writes validation.txt
  python depth_check_run2.py       # run-2 camera against the acceptance landmarks, writes depth_check_run2.txt
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # hard rule: never use the GPU for compute

import argparse
import csv
import json
import math
import pathlib
import re
import tempfile
import time

import mujoco
import numpy as np
import yaml

HERE = pathlib.Path(__file__).resolve().parent
ASSETS = HERE / "assets"
DEFAULT_RUN = ASSETS / "run"  # CLI default only (run 1, the run with an Isaac recording): folder with env.yaml
DEFAULT_EXPORT = ASSETS / "eval"  # CLI default only: folder with policy.onnx + adapter_card.json
X2_DESC = pathlib.Path(r"C:\Users\PC\projects\gwbc-v6\gear_sonic\data\assets\robot_description")
URDF = X2_DESC / "urdf" / "x2_ultra" / "x2_ultra.urdf"
MESHDIR = X2_DESC / "urdf" / "x2_ultra" / "meshes"
OUT_DIR = HERE

ROOT_BODY = "pelvis"
FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")
# pairs filtered in training (instinctlab/assets/agibot_x2.py X2_FILTERED_PAIRS): their hulls overlap in every pose
FILTERED_PAIRS = [("torso_link", "head_pitch_link")] + [
    (f"{s}_{a}", f"{s}_{b}") for s in ("left", "right")
    for a, b in (("knee_link", "ankle_roll_link"), ("wrist_yaw_link", "wrist_roll_link"))]
# sole spheres of gwbc-v6 mjcf/x2_ultra.xml (ankle_roll_link frame, radius 5 mm), only for --foot spheres
FOOT_SPHERES = [(-0.05, 0.05, -0.068), (-0.065, 0.025, -0.068), (-0.05, -0.05, -0.068), (-0.065, -0.025, -0.068),
                (0.11, 0.05, -0.068), (0.11, -0.05, -0.068), (-0.02, -0.06, -0.068), (-0.02, 0.06, -0.068),
                (0.07, -0.06, -0.068), (0.07, 0.06, -0.068), (0.139, 0.0, -0.066), (0.128, 0.0, -0.02)]
FOOT_SPHERE_R = 0.005

GROUP_COLLISION = 0  # collision hulls (invisible)
GROUP_ROBOT_VIS = 1  # URDF visual meshes: rendered, and ray-cast targets like the camera's .../visuals prims
GROUP_TERRAIN = 3

# fall = any of the training terminations (env.yaml terminations): root lower than this above the ground under it,
# tilt beyond bad_orientation.limit_angle, or torso_link touching the terrain
FALL_TILT_RAD = 1.0


# ----------------------------------------------------------------------------------------------------------------------
# config
# ----------------------------------------------------------------------------------------------------------------------
def _match(table, name, default=None):
    """Isaac-style per-joint lookup: scalar or {regex: value} (regex must fully match)."""
    if table is None:
        return default
    if not isinstance(table, dict):
        return float(table)
    for k, v in table.items():
        if re.fullmatch(k, name):
            return float(v)
    return default


def find_env_yaml(run) -> pathlib.Path:
    """`run` may be the env.yaml itself, a folder holding it, or a training run folder (params/env.yaml)."""
    run = pathlib.Path(run)
    for cand in (run, run / "env.yaml", run / "params" / "env.yaml"):
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"no env.yaml under {run}")


def isaac_joint_order(urdf_text: str):
    """Joint order Isaac Lab gives the articulation: breadth first from the root link, the moving joints under each link
    in alphabetical order (links behind fixed joints are merged into their parent)."""
    import xml.etree.ElementTree as ET

    rob = ET.fromstring(re.sub(r"<\?xml[^>]*\?>", "", urdf_text))
    kids, children = {}, set()
    for j in rob.findall("joint"):
        par, ch = j.find("parent").get("link"), j.find("child").get("link")
        kids.setdefault(par, []).append((j.get("name"), j.get("type"), ch))
        children.add(ch)
    roots = [l.get("name") for l in rob.findall("link") if l.get("name") not in children]
    assert len(roots) == 1, roots

    def moving(link):  # moving joints directly under `link`, looking through fixed joints
        out = []
        for name, typ, ch in kids.get(link, []):
            out += moving(ch) if typ == "fixed" else [(name, ch)]
        return out

    order, queue = [], [roots[0]]
    while queue:
        link = queue.pop(0)
        for name, ch in sorted(moving(link)):
            order.append(name)
            queue.append(ch)
    return order


class EnvConfig:
    """Everything the player needs, read from one run's env.yaml (no run-specific constants in this file). Joint order
    from the URDF (Isaac's breadth-first rule). If an adapter card is given, its numbers (joint order, default pose, gains,
    limits, armature, action scale, observation layout, depth camera) are asserted equal to the yaml's."""

    def __init__(self, run=DEFAULT_RUN, card_json=None):
        self.env_yaml = find_env_yaml(run)
        with open(self.env_yaml) as f:
            self.cfg = c = yaml.unsafe_load(f)
        self.card = card = json.load(open(card_json)) if card_json is not None and pathlib.Path(card_json).is_file() else None
        self.joints = isaac_joint_order(training_urdf_text())  # 31, Isaac articulation order
        if card is not None:
            assert self.joints == list(card["joint_order"]), "joint order: URDF breadth-first vs card"
        self.sim_dt = float(c["sim"]["dt"])
        self.decimation = int(c["decimation"])
        rob = c["scene"]["robot"]
        n = len(self.joints)
        self.default_q = np.zeros(n)
        for expr, v in rob["init_state"]["joint_pos"].items():
            for i, j in enumerate(self.joints):
                if re.fullmatch(expr, j):
                    self.default_q[i] = v
        self.init_z = float(rob["init_state"]["pos"][2])
        self.kp, self.kd, self.armature, self.effort, self.vel_limit = (np.zeros(n) for _ in range(5))
        self.max_delay = 0
        for an, ac in rob["actuators"].items():
            assert ac["class_type"].split(":")[-1] == "DelayedPDActuator", ac["class_type"]
            self.max_delay = max(self.max_delay, int(ac["max_delay"]))
            for i, j in enumerate(self.joints):
                if any(re.fullmatch(e, j) for e in ac["joint_names_expr"]):
                    self.kp[i] = _match(ac["stiffness"], j)
                    self.kd[i] = _match(ac["damping"], j)
                    self.armature[i] = _match(ac["armature"], j, 0.0)
                    self.effort[i] = _match(ac["effort_limit_sim"], j)
                    self.vel_limit[i] = _match(ac["velocity_limit_sim"], j)
        assert (self.kp > 0).all(), "a joint has no actuator"
        act = c["actions"]["joint_pos"]
        assert act["use_default_offset"] and act["clip"] is None
        self.action_joints = [j for j in self.joints if any(re.fullmatch(e, j) for e in act["joint_names"])]  # 29
        self.action_idx = np.array([self.joints.index(j) for j in self.action_joints])
        assert card is None or self.action_joints == list(card["action_joint_order"]), "action joint order: card vs env.yaml"
        self.action_scale = np.array([_match(act["scale"], j) for j in self.action_joints])
        # the armature rule of run 1 (kp = armature (2 pi 10 Hz)^2, kd = 4 armature 2 pi 10 Hz) or other per-joint gains
        w0 = 2 * math.pi * 10.0
        self.gains_rule = bool(np.allclose(self.kp, self.armature * w0 ** 2, rtol=1e-4) and np.allclose(self.kd, 4 * self.armature * w0, rtol=1e-4))
        # cross-check the card
        for key, arr, names in () if card is None else (("default_pose", self.default_q, self.joints), ("kp", self.kp, self.joints),
                                ("kd", self.kd, self.joints), ("effort_limit", self.effort, self.joints),
                                ("velocity_limit", self.vel_limit, self.joints), ("armature", self.armature, self.joints),
                                ("action_scale", self.action_scale, self.action_joints)):
            cv = np.array([card[key][j] for j in names])
            assert np.allclose(cv, arr, rtol=1e-6, atol=1e-9), f"card {key} differs from env.yaml"
        assert card is None or abs(1.0 / (self.sim_dt * self.decimation) - card["control"]["policy_rate_hz"]) < 1e-9
        # observations (policy group), in config order
        pol = c["observations"]["policy"]
        skip = {"concatenate_terms", "concatenate_dim", "enable_corruption", "history_length", "flatten_history_dim"}
        self.obs_terms = [(k, v) for k, v in pol.items() if k not in skip and v is not None]
        self.term_dim = {"base_ang_vel": 3, "projected_gravity": 3, "velocity_commands": 3, "joint_pos": n, "joint_vel": n,
                         "actions": len(self.action_joints)}
        cam0 = c["scene"]["camera"]
        ct, cb, cl, cr = cam0["noise_pipeline"]["crop_and_resize"]["crop_region"]  # rows cut at the top / bottom, columns left / right
        self.frame_shape = (int(cam0["pattern_cfg"]["height"]) - ct - cb, int(cam0["pattern_cfg"]["width"]) - cl - cr)  # (rows, cols)
        self.layout = {}  # name -> (start, size, history)
        off = 0
        for name, tc in self.obs_terms:
            if name == "depth_image":
                hist = int(tc["params"]["num_output_frames"])
                size = hist * self.frame_shape[0] * self.frame_shape[1]
            else:
                hist = int(tc.get("history_length") or 1)
                size = self.term_dim[name] * hist
            self.layout[name] = (off, size, hist)
            off += size
        self.obs_dim = off
        for t in ([] if card is None else card["observation"]["terms"]):
            assert self.layout[t["name"]][:2] == (t["start"], t["size"]), f"obs layout {t['name']}: card vs env.yaml"
        assert card is None or self.obs_dim == card["observation"]["total"]
        self.proprio_dim = self.layout["depth_image"][0]
        # command
        cmd = c["commands"]["base_velocity"]
        self.cmd = dict(
            kv=float(cmd["velocity_control_stiffness"]), kh=float(cmd["heading_control_stiffness"]),
            only_pos_x=bool(cmd["only_positive_lin_vel_x"]), yaw_range=tuple(cmd["ranges"]["ang_vel_z"]),
            dist_thr=float(cmd["target_dis_threshold"]), lin_thr=float(cmd["lin_vel_threshold"]),
            ang_thr=float(cmd["ang_vel_threshold"]),
            vx_ranges={k: tuple(v["lin_vel_x"]) for k, v in cmd["velocity_ranges"].items()})
        # camera
        cam = c["scene"]["camera"]
        self.cam_body = cam["prim_path"].split("/")[-1]
        pc = cam["pattern_cfg"]
        self.cam = dict(
            pos=np.array(cam["offset"]["pos"], float),
            quat=np.array(cam["offset"]["rot"], float),  # (w,x,y,z), convention "world" (x fwd, z up)
            convention=cam["offset"]["convention"], width=int(pc["width"]), height=int(pc["height"]),
            fx=pc["width"] * pc["focal_length"] / pc["horizontal_aperture"],
            fy=pc["height"] * pc["focal_length"] / pc["vertical_aperture"],
            cx=pc["horizontal_aperture_offset"] * pc["width"] * pc["focal_length"] / pc["horizontal_aperture"] + pc["width"] / 2,
            cy=pc["vertical_aperture_offset"] * pc["height"] * pc["focal_length"] / pc["vertical_aperture"] + pc["height"] / 2,
            min_dist=float(cam["min_distance"]), max_dist=float(cam["max_distance"]),
            clipping=cam["depth_clipping_behavior"], update_period=float(cam["update_period"]),
            history=int(cam["data_histories"]["distance_to_image_plane_noised"]),
            pipeline=list(cam["noise_pipeline"].keys()),
            crop=tuple(cam["noise_pipeline"]["crop_and_resize"]["crop_region"]),
            resize=cam["noise_pipeline"]["crop_and_resize"]["resize_shape"],
            blur_k=int(cam["noise_pipeline"]["gaussian_blur"]["kernel_size"]),
            blur_s=float(cam["noise_pipeline"]["gaussian_blur"]["sigma"]),
            drange=tuple(cam["noise_pipeline"]["depth_normalization"]["depth_range"]),
            normalize=bool(cam["noise_pipeline"]["depth_normalization"]["normalize"]),
            orange=tuple(cam["noise_pipeline"]["depth_normalization"]["output_range"]))
        assert self.cam["convention"] == "world" and self.cam["clipping"] == "max" and self.cam["resize"] is None
        assert self.cam["pipeline"] == ["crop_and_resize", "gaussian_blur", "depth_normalization"], self.cam["pipeline"]
        assert abs(self.cam["update_period"] - self.sim_dt * self.decimation) < 1e-9
        cq = self.cam["quat"] / np.linalg.norm(self.cam["quat"])
        self.cam["pitch_down_deg"] = math.degrees(math.asin(max(-1.0, min(1.0, 2 * (cq[0] * cq[2] - cq[3] * cq[1])))))  # optical axis below the body's horizontal
        self.cam["fov_deg"] = (2 * math.degrees(math.atan(self.cam["width"] / (2 * self.cam["fx"]))), 2 * math.degrees(math.atan(self.cam["height"] / (2 * self.cam["fy"]))))
        if card is not None and "depth" in card:  # card vs env.yaml for the camera
            cd = card["depth"]
            assert cd["body"] == self.cam_body and np.allclose(cd["pos_in_body"], self.cam["pos"], atol=1e-6), "camera mount: card vs env.yaml"
            assert list(cd["raw_size"]) == [self.cam["width"], self.cam["height"]] and np.allclose(cd["fov_deg"], self.cam["fov_deg"], atol=0.02)
            if "frame_shape" in cd:  # cards written by the generalised export
                assert tuple(cd["frame_shape"]) == self.frame_shape and np.allclose(cd["quat_wxyz"], cq, atol=1e-5)
                assert np.allclose(cd["depth_range_m"], self.cam["drange"]) and tuple(cd["crop_region"]) == tuple(self.cam["crop"])
        dp = pol["depth_image"]["params"]
        self.depth_skip = int(dp["history_skip_frames"])
        self.depth_frames = int(dp["num_output_frames"])
        self.root_height_min = float(c["terminations"]["root_height"]["params"]["minimum_height"])


# ----------------------------------------------------------------------------------------------------------------------
# policy
# ----------------------------------------------------------------------------------------------------------------------
class OnnxPolicy:
    """policy.onnx: flat observation (n, 5408) -> actions (n, 29). CPU only."""

    def __init__(self, path, threads: int = 2):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.obs_dim = self.sess.get_inputs()[0].shape[-1]
        self.num_actions = self.sess.get_outputs()[0].shape[-1]

    def __call__(self, obs):
        obs = np.asarray(obs, np.float32)
        single = obs.ndim == 1
        out = self.sess.run(None, {self.inp: obs[None] if single else obs})[0]
        return out[0] if single else out


# ----------------------------------------------------------------------------------------------------------------------
# terrain
# ----------------------------------------------------------------------------------------------------------------------
class Terrain:
    """Robot spawns at the origin facing +x.
    flat            z = 0 plane (collides everywhere, but MuJoCo rays and the renderer only see it within +-40 m of the origin)
    stairs          straight flight up: flat until x = `start`, then `n_steps` steps (riser, tread) across the full
                    width, then a flat top landing. With the defaults (start 1.25 m = half the 2.5 m platform of the
                    training tile, 5 steps of 0.20 x 0.30 m) the top is at +1.0 m from x = 2.75 m and the paper's target
                    3.7 m ahead lies on the landing.
    stairs_pyramid  the same profile on all four sides (inverted pyramid: the robot starts in the pit, like the Isaac
                    pyramid_stairs_inv tile; no perlin roughness, no walls)."""

    def __init__(self, kind="flat", riser=0.20, tread=0.30, n_steps=5, start=1.25, half_width=4.0, landing=3.0):
        self.kind = kind
        self.riser, self.tread, self.n_steps, self.start = riser, tread, n_steps, start
        self.boxes = []  # (xmin, xmax, ymin, ymax, top z): solid from z = 0
        if kind == "flat":
            return
        end = start + n_steps * tread + landing
        for k in range(1, n_steps + 1):
            a = start + (k - 1) * tread
            if kind == "stairs":
                self.boxes.append((a, end, -half_width, half_width, k * riser))
            elif kind == "stairs_pyramid":
                self.boxes += [(a, end, -end, end, k * riser), (-end, -a, -end, end, k * riser),
                               (-end, end, a, end, k * riser), (-end, end, -end, -a, k * riser)]
            else:
                raise ValueError(kind)

    def height(self, x, y):
        h = 0.0
        for (x0, x1, y0, y1, top) in self.boxes:
            if x0 <= x <= x1 and y0 <= y <= y1:
                h = max(h, top)
        return h

    def describe(self):
        if self.kind == "flat":
            return "flat plane"
        top = self.n_steps * self.riser
        shape = "straight flight" if self.kind == "stairs" else "inverted pyramid (pit)"
        return (f"{shape}: flat for {self.start:.2f} m, then {self.n_steps} steps of {self.riser*100:.0f} cm riser x "
                f"{self.tread*100:.0f} cm tread, landing at +{top:.2f} m from x = {self.start + self.n_steps*self.tread:.2f} m")

    def add_to_spec(self, spec):
        tex = spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                               rgb1=[0.55, 0.57, 0.6], rgb2=[0.68, 0.7, 0.72], width=512, height=512)
        mat = spec.add_material(name="terrain_mat", rgba=[0.62, 0.64, 0.66, 1])
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
        mat.texrepeat = [2, 2]  # 0.5 m checker squares with texuniform
        mat.texuniform = True
        geoms = [spec.worldbody.add_geom(name="terrain", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[40, 40, 0.1], material="terrain_mat")]
        for i, (x0, x1, y0, y1, top) in enumerate(self.boxes):
            shade = 0.80 if (round(top / self.riser) % 2) else 0.66  # alternate tread shades so single steps read in the video
            geoms.append(spec.worldbody.add_geom(
                name=f"terrain_box{i}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[(x1 - x0) / 2, (y1 - y0) / 2, top / 2],
                pos=[(x0 + x1) / 2, (y0 + y1) / 2, top / 2], rgba=[shade, shade * 0.97, shade * 0.9, 1]))
        for g in geoms:
            g.group = GROUP_TERRAIN
            g.contype, g.conaffinity = 1, 1
            g.friction = [1.0, 0.005, 0.0001]
        return geoms


# ----------------------------------------------------------------------------------------------------------------------
# MuJoCo model of the training robot
# ----------------------------------------------------------------------------------------------------------------------
def training_urdf_text(urdf: pathlib.Path = URDF, meshdir: pathlib.Path = MESHDIR) -> str:
    """The shared X2 URDF as training used it: massless base_link (fixed to pelvis) removed -> root body = pelvis."""
    txt = urdf.read_text()
    txt = re.sub(r"<mujoco>.*?</mujoco>", "", txt, flags=re.S)
    txt, nsub = re.subn(r'<link name="base_link"\s*/>\s*<joint name="base_link_joint".*?</joint>', "", txt, flags=re.S)
    assert nsub == 1, "base_link / base_link_joint not found in the URDF"
    inj = (f'<mujoco><compiler meshdir="{meshdir.as_posix()}" strippath="true" discardvisual="false" '
           f'fusestatic="true" balanceinertia="false"/></mujoco>')
    return re.sub(r"(<robot[^>]*>)", lambda mm: mm.group(1) + inj, txt, count=1)


def build_model(ec: EnvConfig, terrain: Terrain, foot: str = "hull", self_collision: bool = True, render_size=(1280, 720)):
    """Compile the MuJoCo model: training URDF + free root + armature + torque actuators (explicit PD is applied in
    X2Sim.physics_step) + terrain. Returns mjModel."""
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="sim2sim_x2_")) / "x2_ultra_pelvis_root.urdf"
    tmp.write_text(training_urdf_text())
    spec = mujoco.MjSpec.from_file(str(tmp))
    root = spec.worldbody.first_body()
    assert root.name == ROOT_BODY, root.name
    root.add_freejoint(name="root")
    spec.option.timestep = ec.sim_dt
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.gravity = list(ec.cfg["sim"]["gravity"])
    for g in spec.geoms:
        assert g.type == mujoco.mjtGeom.mjGEOM_MESH, "the X2 URDF has mesh geoms only"
        if g.contype == 0:  # URDF <visual>: render + depth ray-cast only
            g.group = GROUP_ROBOT_VIS
        else:  # URDF <collision>: MuJoCo collides the convex hull of the mesh
            g.contype, g.conaffinity = (1, 1) if self_collision else (2, 1)
            g.friction = [1.0, 0.005, 0.0001]
            g.group = GROUP_COLLISION
            g.rgba = [0.9, 0.4, 0.1, 0.0]
    if foot == "spheres":
        for fb in FOOT_BODIES:
            b = spec.body(fb)
            for g in b.geoms:
                if g.group == GROUP_COLLISION:
                    g.contype, g.conaffinity = 0, 0
            for k, p in enumerate(FOOT_SPHERES):
                s = b.add_geom(name=f"{fb}_sole{k}", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[FOOT_SPHERE_R, 0, 0], pos=list(p))
                s.contype, s.conaffinity = (1, 1) if self_collision else (2, 1)
                s.friction = [1.0, 0.005, 0.0001]
                s.group = GROUP_COLLISION
                s.rgba = [0.9, 0.4, 0.1, 0.0]
                s.mass = 0.0
    elif foot != "hull":
        raise ValueError(foot)
    for a, b in FILTERED_PAIRS:
        spec.add_exclude(bodyname1=a, bodyname2=b)
    for i, jn in enumerate(ec.joints):
        j = spec.joint(jn)
        j.armature = ec.armature[i]  # URDF has no <dynamics>: joint damping / friction stay 0 like Isaac
        a = spec.add_actuator(name=jn, target=jn, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        a.gaintype = mujoco.mjtGain.mjGAIN_FIXED  # ctrl = torque (explicit PD computed in python every physics step)
        a.gainprm = [1.0] + [0.0] * 9
        a.biastype = mujoco.mjtBias.mjBIAS_NONE
        a.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
        a.forcerange = [-ec.effort[i], ec.effort[i]]
        a.ctrllimited = mujoco.mjtLimited.mjLIMITED_FALSE
    terrain.add_to_spec(spec)
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.62, 0.72, 0.85], rgb2=[0.1, 0.12, 0.16], width=256, height=256)
    spec.visual.global_.offwidth, spec.visual.global_.offheight = render_size
    spec.visual.headlight.ambient = [0.45, 0.45, 0.45]
    spec.visual.headlight.diffuse = [0.55, 0.55, 0.55]
    spec.visual.quality.shadowsize = 4096
    spec.worldbody.add_light(name="sun", pos=[0, 0, 10], dir=[0.3, 0.2, -1], diffuse=[0.5, 0.5, 0.5], castshadow=True,
                             type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    return spec.compile()


# ----------------------------------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------------------------------
def quat_to_mat(q):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, float))
    return R.reshape(3, 3)


def yaw_of(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def roll_pitch_of(q):
    w, x, y, z = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    return roll, pitch


def wrap_to_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def gaussian_blur_reflect(img, k, sigma):
    """torchvision GaussianBlur(kernel_size=k, sigma=sigma): separable kernel, reflect padding."""
    half = (k - 1) * 0.5
    x = np.linspace(-half, half, k)
    ker = np.exp(-0.5 * (x / sigma) ** 2)
    ker /= ker.sum()
    p = k // 2
    a = np.pad(img, ((p, p), (p, p)), mode="reflect")
    a = sum(ker[i] * a[:, i:i + img.shape[1]] for i in range(k))  # horizontal
    a = sum(ker[i] * a[i:i + img.shape[0], :] for i in range(k))  # vertical
    return a


def target_command(ec: EnvConfig, root_pos, root_quat, target, vmax):
    """PoseVelocityCommand._update_command for one robot: [vx, vy, yaw_rate] in the root frame.
    vx = clip(kv * x_b, 0, vmax); vy = clip(kv * y_b, -0, 0) = 0 (lin_vel_y range is (0,0) on every terrain);
    yaw rate = clip(kh * wrap(bearing - yaw), -1, 1); everything 0 when the target is within dist_thr (0.4 m)."""
    c = ec.cmd
    vec = np.asarray(target, float) - np.asarray(root_pos, float)
    dist = float(np.linalg.norm(vec[:2]))
    yaw = yaw_of(root_quat)
    cy, sy = math.cos(yaw), math.sin(yaw)
    xb, yb = cy * vec[0] + sy * vec[1], -sy * vec[0] + cy * vec[1]  # quat_apply_inverse(yaw_quat)
    vx = min(max(xb * c["kv"], 0.0 if c["only_pos_x"] else -vmax), vmax)
    vy = min(max(yb * c["kv"], -0.0), 0.0)
    wz = min(max(wrap_to_pi(math.atan2(vec[1], vec[0]) - yaw) * c["kh"], c["yaw_range"][0]), c["yaw_range"][1])
    cmd = np.array([vx, vy, wz]) * float(dist > c["dist_thr"])
    cmd[:2] *= float(np.linalg.norm(cmd[:2]) > c["lin_thr"])
    cmd[2] *= float(abs(cmd[2]) > c["ang_thr"])
    return cmd


class DepthCamera:
    """NoisyGroupedRayCasterCamera on the body named in env.yaml: z-depth (distance to image plane) by ray-casting the
    terrain and the robot's own visual meshes; frame() returns the processed image the policy sees (ec.frame_shape, 0..1)."""

    def __init__(self, ec: EnvConfig, m):
        c = self.c = ec.cam
        W, H = c["width"], c["height"]
        u, v = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")  # row-major (H, W): row 0 = top, col 0 = left
        px = (u + 0.5 - c["cx"]) / c["fx"]
        py = (v + 0.5 - c["cy"]) / c["fy"]
        d = np.stack([np.ones_like(px), -px, -py], -1)  # camera frame "world" convention: x fwd, y left, z up
        self.dirs_full = d / np.linalg.norm(d, axis=-1, keepdims=True)  # (H, W, 3)
        t, b, l, r = c["crop"]
        self.crop_slices = (slice(t, H - b), slice(l, W - r))
        self.R_off = quat_to_mat(c["quat"])
        self.p_off = c["pos"]
        self.body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, ec.cam_body)
        assert self.body >= 0
        self.gg = np.zeros(6, np.uint8)
        self.gg[GROUP_TERRAIN] = 1
        self.gg[GROUP_ROBOT_VIS] = 1
        self.hit_close = 0
        self.robot_hits = 0

    def pose(self, d):
        R_b = d.xmat[self.body].reshape(3, 3)
        return d.xpos[self.body] + R_b @ self.p_off, R_b @ self.R_off

    def cast(self, m, d, dirs_cam):
        """z-depth for an (h, w, 3) block of camera-frame unit directions. Also sets self.last_geom (h, w) geom ids."""
        c = self.c
        shp = dirs_cam.shape[:2]
        dc = dirs_cam.reshape(-1, 3)
        n = len(dc)
        p, R = self.pose(d)
        dirs_w = np.ascontiguousarray(dc @ R.T)
        geomid = np.zeros(n, np.int32)
        dist = np.zeros(n)
        mujoco.mj_multiRay(m, d, p, dirs_w.ravel(), self.gg, 1, -1, geomid, dist, None, n, 1e6)
        close = (dist >= 0) & (dist < c["min_dist"])
        self.hit_close = int(close.sum())
        if close.any():  # the warp ray-caster ignores hits closer than min_distance -> re-cast those rays from min_dist
            gid = np.zeros(1, np.int32)
            for k in np.nonzero(close)[0]:
                r = mujoco.mj_ray(m, d, p + c["min_dist"] * dirs_w[k], dirs_w[k], self.gg, 1, -1, gid)
                dist[k] = r + c["min_dist"] if r >= 0 else -1
                geomid[k] = gid[0] if r >= 0 else -1
        self.last_geom = geomid.reshape(shp)
        self.robot_hits = int(((geomid >= 0) & (m.geom_group[np.maximum(geomid, 0)] == GROUP_ROBOT_VIS)).sum())
        dist[dist < 0] = np.inf
        z = dist * dc[:, 0]  # distance_to_image_plane (camera x is forward)
        z = np.minimum(z, c["max_dist"])  # depth_clipping_behavior="max"
        z[~np.isfinite(z)] = c["max_dist"]
        return z.reshape(shp).astype(np.float32)

    def raw_full(self, m, d):
        """Full 36x64 raw z-depth in metres (not needed by the policy; for inspection)."""
        return self.cast(m, d, self.dirs_full)

    def process(self, cropped):
        """crop is already applied: blur k3 s1 (reflect) -> clip to depth_range -> normalise to 0..1."""
        c = self.c
        img = gaussian_blur_reflect(cropped.astype(np.float32), c["blur_k"], c["blur_s"]).astype(np.float32)
        lo, hi = c["drange"]
        img = np.clip(img, lo, hi)
        if c["normalize"]:
            img = (img - lo) / (hi - lo) * (c["orange"][1] - c["orange"][0]) + c["orange"][0]
        return img.astype(np.float32)

    def frame(self, m, d):
        """Processed policy frame (ec.frame_shape). Only the cropped block is ray-cast (the blur runs after the crop)."""
        return self.process(self.cast(m, d, self.dirs_full[self.crop_slices]))


# ----------------------------------------------------------------------------------------------------------------------
# the simulator wrapper
# ----------------------------------------------------------------------------------------------------------------------
class X2Sim:
    """One X2 in MuJoCo with the training observation / action interface.

        sim = X2Sim(EnvConfig("assets/run2"), Terrain("flat"))
        sim.reset(root_pos=(0, 0, ec.init_z))
        obs = sim.observe(cmd)          # (ec.obs_dim,) float32, histories updated
        sim.step(policy(obs))           # 4 physics steps of explicit PD
    """

    def __init__(self, ec: EnvConfig, terrain: Terrain, foot="hull", self_collision=True, delay_steps=0, depth_lag=1,
                 render_size=(1280, 720)):
        self.ec, self.terrain, self.foot = ec, terrain, foot
        self.m = m = build_model(ec, terrain, foot, self_collision, render_size)
        self.d = mujoco.MjData(m)
        nid = lambda t, n: mujoco.mj_name2id(m, t, n)
        self.root = nid(mujoco.mjtObj.mjOBJ_BODY, ROOT_BODY)
        self.torso = nid(mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self.qadr = np.array([m.jnt_qposadr[nid(mujoco.mjtObj.mjOBJ_JOINT, j)] for j in ec.joints])
        self.dadr = np.array([m.jnt_dofadr[nid(mujoco.mjtObj.mjOBJ_JOINT, j)] for j in ec.joints])
        self.act_ids = np.array([nid(mujoco.mjtObj.mjOBJ_ACTUATOR, j) for j in ec.joints])
        self.feet = [nid(mujoco.mjtObj.mjOBJ_BODY, n) for n in FOOT_BODIES]
        self.terrain_geoms = {g for g in range(m.ngeom) if m.geom_group[g] == GROUP_TERRAIN}
        self.cam = DepthCamera(ec, m)
        self.delay_steps = int(delay_steps)
        self.depth_lag = int(depth_lag)
        self.reset((0.0, 0.0, terrain.height(0, 0) + ec.init_z))

    # --- state -------------------------------------------------------------------------------------------------------
    def set_state(self, root_pos, root_quat=(1, 0, 0, 0), joint_pos=None, joint_vel=None, root_lin_vel_b=None, root_ang_vel_b=None):
        """Set the MuJoCo state from Isaac-style quantities (joint arrays in Isaac order, quaternion wxyz, root
        velocities in the root frame). Does not touch the observation histories."""
        m, d, ec = self.m, self.d, self.ec
        q = np.asarray(root_quat, float)
        q = q / np.linalg.norm(q)
        d.qpos[:3] = root_pos
        d.qpos[3:7] = q
        d.qpos[self.qadr] = ec.default_q if joint_pos is None else joint_pos
        d.qvel[:] = 0
        if joint_vel is not None:
            d.qvel[self.dadr] = joint_vel
        if root_lin_vel_b is not None:
            d.qvel[:3] = quat_to_mat(q) @ np.asarray(root_lin_vel_b, float)  # free joint: linear velocity in the world frame
        if root_ang_vel_b is not None:
            d.qvel[3:6] = root_ang_vel_b  # free joint: angular velocity in the body frame
        mujoco.mj_forward(m, d)

    def reset(self, root_pos, yaw=0.0, root_quat=None, **state):
        """Reset like Isaac: default joint pose, zero velocity (or the given state), cleared histories."""
        if root_quat is None:
            root_quat = (math.cos(yaw / 2), 0, 0, math.sin(yaw / 2))
        self.set_state(root_pos, root_quat, **state)
        self.last_action = np.zeros(len(self.ec.action_joints))
        self.hist = {}
        self.depth_hist = []
        self.q_target = self.ec.default_q.copy()
        self.target_queue = [self.q_target.copy() for _ in range(self.delay_steps)]
        self.vel_limit_hits = 0
        self.torque_sat_steps = 0
        self.phys_steps = 0

    def root_state(self):
        """(pos_w, quat_wxyz, R, lin_vel_b, ang_vel_b) of the pelvis."""
        d = self.d
        v6 = np.zeros(6)
        mujoco.mj_objectVelocity(self.m, d, mujoco.mjtObj.mjOBJ_XBODY, self.root, v6, 1)  # [ang, lin] in the body (link) frame; mjOBJ_BODY would be the inertial frame
        return d.xpos[self.root].copy(), d.xquat[self.root].copy(), d.xmat[self.root].reshape(3, 3).copy(), v6[3:].copy(), v6[:3].copy()

    def joint_pos(self):
        return self.d.qpos[self.qadr].copy()

    def joint_vel(self):
        return self.d.qvel[self.dadr].copy()

    # --- observation -------------------------------------------------------------------------------------------------
    def proprio_terms(self, cmd):
        """Newest-step value of every proprio term, already scaled, in observation order (no history, no noise)."""
        _, _, R, _, w_b = self.root_state()
        raw = {
            "base_ang_vel": w_b,
            "projected_gravity": R.T @ np.array([0.0, 0.0, -1.0]),
            "velocity_commands": np.asarray(cmd, float),
            "joint_pos": self.joint_pos() - self.ec.default_q,
            "joint_vel": self.joint_vel(),  # default joint velocity is 0
            "actions": self.last_action.copy(),  # previous raw policy output (29)
        }
        out = {}
        for name, tc in self.ec.obs_terms:
            if name == "depth_image":
                continue
            v = np.asarray(raw[name], float)
            if tc.get("clip") is not None:
                v = np.clip(v, tc["clip"][0], tc["clip"][1])
            if tc.get("scale") is not None:
                v = v * tc["scale"]
            out[name] = v
        return out

    def depth_frame(self):
        return self.cam.frame(self.m, self.d)

    def depth_stack(self):
        """(8, rows, cols): frames every 5th control step, oldest first, newest last; the newest one is `depth_lag` control
        steps old (1 in the Isaac recording)."""
        L = len(self.depth_hist)
        skip, n = self.ec.depth_skip, self.ec.depth_frames
        idx = [max(L - 1 - self.depth_lag - o, 0) for o in range((n - 1) * skip, -1, -skip)]
        return np.stack([self.depth_hist[i] for i in idx], 0)

    def observe(self, cmd):
        """Flat policy observation (ec.obs_dim,) for the current state; pushes one entry into every history.
        Layout: for each proprio term 8 steps [t-7 .. t] concatenated (term after term), then the depth stack flattened
        (frame, row, column). On the first call after reset every history is filled with the first value."""
        parts = []
        for name, v in self.proprio_terms(cmd).items():
            L = self.ec.layout[name][2]
            self.hist[name] = [v] * L if name not in self.hist else self.hist[name][1:] + [v]
            parts.append(np.concatenate(self.hist[name]))
        frame = self.depth_frame()
        need = self.ec.cam["history"] + self.depth_lag
        self.depth_hist = [frame] * need if not self.depth_hist else self.depth_hist[1:] + [frame]
        self.last_frame = frame
        parts.append(self.depth_stack().ravel())
        obs = np.concatenate(parts).astype(np.float32)
        assert obs.shape == (self.ec.obs_dim,), obs.shape
        return obs

    # --- control -----------------------------------------------------------------------------------------------------
    def pd_torque(self, q_target):
        ec = self.ec
        tau = ec.kp * (q_target - self.d.qpos[self.qadr]) - ec.kd * self.d.qvel[self.dadr]
        return np.clip(tau, -ec.effort, ec.effort), tau

    def physics_step(self):
        """One 5 ms step: explicit PD torque from the current state (like Isaac's DelayedPDActuator every sub-step)."""
        if self.delay_steps > 0:
            self.target_queue.append(self.q_target.copy())
            tgt = self.target_queue.pop(0)
        else:
            tgt = self.q_target
        tau, unclipped = self.pd_torque(tgt)
        self.d.ctrl[self.act_ids] = tau
        self.last_tau = tau
        mujoco.mj_step(self.m, self.d)
        self.phys_steps += 1
        self.vel_limit_hits += int((np.abs(self.d.qvel[self.dadr]) > self.ec.vel_limit).any())
        self.torque_sat_steps += int((np.abs(unclipped) > self.ec.effort).any())

    def step(self, action):
        """Apply one policy action (29, action_joint_order) for one control step (decimation physics steps)."""
        ec = self.ec
        self.last_action = np.asarray(action, float).copy()
        self.q_target = ec.default_q.copy()  # head joints stay at their default (0)
        self.q_target[ec.action_idx] += ec.action_scale * self.last_action
        for _ in range(ec.decimation):
            self.physics_step()

    # --- contacts ----------------------------------------------------------------------------------------------------
    def foot_forces(self):
        """(2, 3) world-frame force of the terrain on the left / right foot body (N)."""
        m, d = self.m, self.d
        F = np.zeros((2, 3))
        f6 = np.zeros(6)
        for k in range(d.ncon):
            con = d.contact[k]
            if not ((con.geom1 in self.terrain_geoms) ^ (con.geom2 in self.terrain_geoms)):
                continue
            g_robot = con.geom2 if con.geom1 in self.terrain_geoms else con.geom1
            b = m.geom_bodyid[g_robot]
            if b in self.feet:
                mujoco.mj_contactForce(m, d, k, f6)
                fw = con.frame.reshape(3, 3).T @ f6[:3]  # force on geom2 from geom1, world frame
                F[self.feet.index(b)] += fw if g_robot == con.geom2 else -fw
        return F

    def other_terrain_contacts(self):
        """Names of robot bodies other than the feet that touch the terrain."""
        m, d = self.m, self.d
        out = set()
        for k in range(d.ncon):
            con = d.contact[k]
            if (con.geom1 in self.terrain_geoms) ^ (con.geom2 in self.terrain_geoms):
                g = con.geom2 if con.geom1 in self.terrain_geoms else con.geom1
                b = m.geom_bodyid[g]
                if b not in self.feet:
                    out.add(m.body(b).name)
        return out

    def self_contacts(self):
        m, d = self.m, self.d
        out = set()
        for k in range(d.ncon):
            con = d.contact[k]
            if con.geom1 not in self.terrain_geoms and con.geom2 not in self.terrain_geoms and con.dist < 0:
                out.add(tuple(sorted((m.body(m.geom_bodyid[con.geom1]).name, m.body(m.geom_bodyid[con.geom2]).name))))
        return out

    def fallen(self):
        """Training termination conditions: (reason or None)."""
        pos, q, R, _, _ = self.root_state()
        h = pos[2] - self.terrain.height(pos[0], pos[1])
        tilt = math.acos(max(-1.0, min(1.0, -(R.T @ np.array([0, 0, -1.0]))[2])))
        if h < self.ec.root_height_min:
            return f"pelvis {h:.2f} m above the ground (< {self.ec.root_height_min})"
        if tilt > FALL_TILT_RAD:
            return f"tilt {math.degrees(tilt):.0f} deg (> {math.degrees(FALL_TILT_RAD):.0f})"
        if "torso_link" in self.other_terrain_contacts():
            return "torso_link touches the terrain"
        return None


# ----------------------------------------------------------------------------------------------------------------------
# closed-loop run, CSV, summary
# ----------------------------------------------------------------------------------------------------------------------
def run_closed_loop(sim: X2Sim, policy, seconds, target=None, vmax=0.74, cmd_fn=None, verbose=True):
    """Closed loop for `seconds`. The command comes from `target` (training rule) or cmd_fn(step, sim) -> (3,).
    Does not stop on success or fall: the whole clip is simulated. Returns a dict of per-control-step arrays; every row
    is the state at the start of the control step, before the action of that step is applied."""
    ec = sim.ec
    dt = ec.sim_dt * ec.decimation
    steps = int(round(seconds / dt))
    keys = ("t", "cmd", "root_pos", "root_quat", "lin_vel_b", "ang_vel_b", "joint_pos", "action", "feet_force", "qpos",
            "depth", "other_contacts", "robot_depth_px", "tau_max_frac")
    rec = {k: [] for k in keys}
    fall_t, fall_reason, success_t = None, None, None
    obs0 = None
    wall = time.time()
    for s in range(steps):
        t = s * dt
        pos, q, R, v_b, w_b = sim.root_state()
        cmd = np.asarray(cmd_fn(s, sim), float) if cmd_fn is not None else target_command(ec, pos, q, target, vmax)
        obs = sim.observe(cmd)
        if obs0 is None:
            obs0 = obs.copy()
        action = policy(obs)
        rec["t"].append(t); rec["cmd"].append(cmd); rec["root_pos"].append(pos); rec["root_quat"].append(q)
        rec["lin_vel_b"].append(v_b); rec["ang_vel_b"].append(w_b); rec["joint_pos"].append(sim.joint_pos())
        rec["action"].append(np.asarray(action, float).copy()); rec["feet_force"].append(sim.foot_forces())
        rec["qpos"].append(sim.d.qpos.copy()); rec["depth"].append(sim.last_frame.copy())
        rec["other_contacts"].append(",".join(sorted(sim.other_terrain_contacts())))
        rec["robot_depth_px"].append(sim.cam.robot_hits)
        sim.step(action)
        rec["tau_max_frac"].append(float(np.max(np.abs(sim.last_tau) / ec.effort)))
        reason = sim.fallen()
        if fall_t is None and reason is not None:
            fall_t, fall_reason = t + dt, reason
            if verbose:
                print(f"[sim] FELL at t={fall_t:.2f} s: {reason}")
        pos = sim.root_state()[0]
        if success_t is None and target is not None and np.linalg.norm((np.asarray(target) - pos)[:2]) < ec.cmd["dist_thr"]:
            success_t = t + dt
            if verbose:
                print(f"[sim] reached the target at t={success_t:.2f} s")
        if verbose and s % 50 == 0:
            print(f"[sim] t={t:5.2f} pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}) cmd={np.round(cmd, 2)} |a|max={np.abs(action).max():.2f}"
                  f" feet Fz={np.round(rec['feet_force'][-1][:, 2])}")
    out = {k: np.array(v) for k, v in rec.items()}
    out.update(fall_t=fall_t, fall_reason=fall_reason, success_t=success_t, first_obs=obs0, wall_s=time.time() - wall,
               vel_limit_hits=sim.vel_limit_hits, torque_sat_steps=sim.torque_sat_steps, phys_steps=sim.phys_steps)
    return out


def touchdowns(t, Fz, on=60.0, off=10.0, min_gap=0.15):
    """Indices where a foot's vertical force rises above `on` N after having been below `off` N (hysteresis)."""
    td, state, last = [], Fz[0] > on, -1e9
    for k in range(1, len(t)):
        if state and Fz[k] < off:
            state = False
        elif not state and Fz[k] > on:
            state = True
            if t[k] - last >= min_gap:
                td.append(k)
                last = t[k]
    return td


CONTACT_FLAG_N = 5.0  # a foot counts as in contact when the terrain pushes on it with more than this (N, vertical)


def write_csv(rec, ec: EnvConfig, path):
    hdr = (["time_s", "cmd_vx", "cmd_vy", "cmd_yaw_rate", "base_vx_b", "base_vy_b", "base_wz_b", "base_x", "base_y", "base_z",
            "base_roll", "base_pitch", "left_foot_contact", "right_foot_contact"] + [f"q_{j}" for j in ec.joints])
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        for k in range(len(rec["t"])):
            roll, pitch = roll_pitch_of(rec["root_quat"][k])
            row = [rec["t"][k], *rec["cmd"][k], rec["lin_vel_b"][k][0], rec["lin_vel_b"][k][1], rec["ang_vel_b"][k][2],
                   *rec["root_pos"][k], roll, pitch,
                   int(rec["feet_force"][k][0, 2] > CONTACT_FLAG_N), int(rec["feet_force"][k][1, 2] > CONTACT_FLAG_N),
                   *rec["joint_pos"][k]]
            w.writerow([f"{v:.5f}" if isinstance(v, float) else v for v in map(lambda x: float(x) if not isinstance(x, int) else x, row)])


def summarize(rec, name, vmax):
    t, p = rec["t"], rec["root_pos"]
    dt = t[1] - t[0]
    dur = t[-1] + dt
    net = float(np.linalg.norm(p[-1, :2] - p[0, :2]))
    path = float(np.sum(np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1)))
    td = [touchdowns(t, rec["feet_force"][:, i, 2]) for i in range(2)]
    pitch = np.array([roll_pitch_of(q)[1] for q in rec["root_quat"]])
    roll = np.array([roll_pitch_of(q)[0] for q in rec["root_quat"]])
    moving = np.linalg.norm(rec["cmd"][:, :2], axis=1) > 0.05
    mean_cmd = float(rec["cmd"][moving, 0].mean()) if moving.any() else 0.0
    mean_vx = float(rec["lin_vel_b"][moving, 0].mean()) if moving.any() else 0.0
    first1 = float(np.linalg.norm(p[min(50, len(p) - 1), :2] - p[0, :2]))
    # after the landing from the spawn drop: how long is a foot ever off the ground, and how many lift-offs
    air_max, lifts = 0.0, 0
    for i in range(2):
        c = rec["feet_force"][:, i, 2] > CONTACT_FLAG_N
        if c.any():
            c = c[int(np.argmax(c)):]
            run = 0
            for v in c:
                run = 0 if v else run + 1
                lifts += int(run == 1)
                air_max = max(air_max, run * dt)
    landing = sum(int(len(x) > 0) for x in td)
    s = dict(name=name, duration_s=float(dur), vmax=vmax, net_distance_m=net, forward_x_m=float(p[-1, 0] - p[0, 0]),
             path_length_m=path, moved_first_second_m=first1, touchdowns_left=len(td[0]), touchdowns_right=len(td[1]),
             mean_cmd_vx=mean_cmd, mean_base_vx=mean_vx, mean_net_speed=net / dur, final_height_z=float(p[-1, 2]),
             foot_liftoffs_after_landing=lifts, longest_foot_airtime_after_landing_s=air_max,
             mean_pitch_deg_last_5s=float(np.degrees(pitch[-250:].mean())), mean_roll_deg_last_5s=float(np.degrees(roll[-250:].mean())),
             fell=rec["fall_t"] is not None, fall_t=rec["fall_t"], fall_reason=rec["fall_reason"],
             reached_target_t=rec["success_t"],
             double_support_frac=float(((rec["feet_force"][:, 0, 2] > CONTACT_FLAG_N) & (rec["feet_force"][:, 1, 2] > CONTACT_FLAG_N)).mean()),
             other_body_terrain_contacts=sorted({b for s_ in rec["other_contacts"] for b in s_.split(",") if b}),
             joint_vel_limit_exceed_phys_steps=int(rec["vel_limit_hits"]), torque_saturated_phys_steps=int(rec["torque_sat_steps"]),
             phys_steps=int(rec["phys_steps"]), wall_s=float(rec["wall_s"]))
    line = (f"{name}: {dur:.1f} s clip, commanded {mean_cmd:.2f} m/s (mean while commanded); moved {net:.2f} m net "
            f"({s['forward_x_m']:+.2f} m along x, path {path:.2f} m, {first1:.2f} m in the first second); "
            f"foot touchdowns L {len(td[0])} + R {len(td[1])} = {len(td[0]) + len(td[1])} (the first {landing} are the landing from the "
            f"7 cm spawn drop; after it {lifts} lift-offs, longest time a foot is off the ground {air_max:.2f} s); mean pelvis pitch over the "
            f"last 5 s {np.degrees(pitch[-250:].mean()):+.1f} deg (negative = leaning back); mean forward speed "
            f"{mean_vx:+.3f} m/s (body frame) / {net/dur:.3f} m/s net vs command {mean_cmd:.2f}; "
            + (f"FELL at {rec['fall_t']:.2f} s ({rec['fall_reason']})" if rec["fall_t"] is not None else "did not fall")
            + ("" if rec["success_t"] is None else f"; reached the target at {rec['success_t']:.2f} s"))
    return s, line


# ----------------------------------------------------------------------------------------------------------------------
# video
# ----------------------------------------------------------------------------------------------------------------------
def run_labels(ec: EnvConfig, run_name=None, checkpoint=None, plant=None):
    """(run name, checkpoint label, second caption line) built from the card / env.yaml, never from constants."""
    card = ec.card or {}
    ck = pathlib.PurePosixPath(str(card.get("checkpoint", "")).replace(chr(92), "/"))
    if run_name is None:
        run_name = ck.parent.parent.name if len(ck.parts) >= 3 else ec.env_yaml.parent.name
    if checkpoint is None:
        checkpoint = str(card["iteration"]) if "iteration" in card else "?"
    if plant is None:
        plant = (card.get("control") or {}).get("gains_label") or ("armature rule" if ec.gains_rule else "per-joint env.yaml gains")
    g = lambda j: f"{ec.kp[ec.joints.index(j)]:.0f}/{ec.kd[ec.joints.index(j)]:.1f}"
    c = ec.cam
    line2 = (f"gains {plant}: kp/kd hip pitch {g('left_hip_pitch_joint')}, knee {g('left_knee_joint')}, ankle pitch {g('left_ankle_pitch_joint')}"
             f"   |   depth camera on {ec.cam_body}, {c['pitch_down_deg']:.1f} deg down, {c['fov_deg'][0]:.0f} x {c['fov_deg'][1]:.0f} deg,"
             f" frame {ec.frame_shape[0]} x {ec.frame_shape[1]}, {c['drange'][0]:.0f}-{c['drange'][1]:.1f} m")
    return run_name, checkpoint, line2


def render_video(sim: X2Sim, rec, path, caption, view="chase", target=None, fps=25, size=(1280, 720), crf=24, caption2=None):
    """mp4 (H.264, real time). view='chase': three-quarter view from behind-right, nothing between camera and robot;
    view='legs': side view at shin height following the feet. Caption burned into every frame."""
    import imageio
    from PIL import Image, ImageDraw, ImageFont

    m = sim.m
    d = mujoco.MjData(m)
    W, H = size
    ren = mujoco.Renderer(m, H, W, max_geom=10000)
    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[GROUP_ROBOT_VIS] = 1
    opt.geomgroup[GROUP_TERRAIN] = 1
    try:
        font = ImageFont.truetype("arial.ttf", 22)
        small = ImageFont.truetype("arial.ttf", 17)
    except OSError:
        font = small = ImageFont.load_default()
    dt = sim.ec.sim_dt * sim.ec.decimation
    every = max(1, int(round(1.0 / (fps * dt))))
    w = imageio.get_writer(str(path), fps=1.0 / (every * dt), codec="libx264", macro_block_size=8,
                           pixelformat="yuv420p", ffmpeg_params=["-crf", str(crf), "-preset", "medium"])
    yaw0 = math.degrees(yaw_of(rec["root_quat"][0]))
    lk = None
    for k in range(0, len(rec["t"]), every):
        d.qpos[:] = rec["qpos"][k]
        mujoco.mj_forward(m, d)
        if view == "legs":
            tgt = 0.5 * (d.xpos[sim.feet[0]] + d.xpos[sim.feet[1]]) + [0, 0, 0.22]
            cam.distance, cam.elevation, cam.azimuth = 1.7, -6, yaw0 + 90.0  # camera on the robot's right, looking left
        else:
            tgt = rec["root_pos"][k] + [0, 0, 0.25]
            cam.distance, cam.elevation, cam.azimuth = 4.6, -12, yaw0 + 40.0  # behind and to the right
        lk = np.array(tgt) if lk is None else lk + 0.15 * (tgt - lk)
        cam.lookat[:] = lk
        ren.update_scene(d, cam, opt)
        scn = ren.scene
        if target is not None and scn.ngeom + 2 <= scn.maxgeom:
            # target marker: a thin pole floating 0.4..1.4 m above the ground at the target with a small flag on top
            base = np.array([target[0], target[1], sim.terrain.height(target[0], target[1])])
            mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_CYLINDER, np.array([0.012, 0.5, 0.0]),
                                base + [0, 0, 0.9], np.eye(3).ravel(), np.array([0.9, 0.15, 0.1, 1.0], np.float32))
            scn.ngeom += 1
            mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_BOX, np.array([0.004, 0.11, 0.07]),
                                base + [0, 0.11, 1.33], np.eye(3).ravel(), np.array([0.9, 0.15, 0.1, 1.0], np.float32))
            scn.ngeom += 1
        im = Image.fromarray(ren.render())
        dr = ImageDraw.Draw(im)
        dr.rectangle((0, 0, W, 38 if not caption2 else 62), fill=(15, 18, 24))
        dr.text((12, 7), caption, fill=(240, 240, 240), font=font)
        if caption2:
            dr.text((12, 36), caption2, fill=(205, 210, 220), font=small)
        c = rec["cmd"][k]
        v = rec["lin_vel_b"][k]
        vname = "legs close-up (side view, follows the feet)" if view == "legs" else "three-quarter chase view"
        dr.rectangle((0, H - 30, W, H), fill=(15, 18, 24))
        dr.text((12, H - 26), f"t = {rec['t'][k]:5.2f} s   command vx {c[0]:.2f} m/s, yaw rate {c[2]:+.2f} rad/s   actual vx {v[0]:+.2f} m/s"
                              f"   |   {vname}" + ("   |   red pole with flag = target (in the air, not a foothold)" if target is not None and view == "chase" else ""),
                fill=(250, 250, 250), font=small)
        w.append_data(np.asarray(im))
    w.close()
    ren.close()


# ----------------------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="output prefix: <name>.csv, <name>_chase.mp4, <name>_legs.mp4, <name>.json")
    ap.add_argument("--terrain", choices=["flat", "stairs", "stairs_pyramid"], default="flat")
    ap.add_argument("--foot", choices=["hull", "spheres"], default="hull", help="hull = convex hull of the foot mesh (training)")
    ap.add_argument("--target-dist", type=float, default=3.7, help="target this far ahead of the spawn (training tiles: 3.7 m)")
    ap.add_argument("--vmax", type=float, default=0.74, help="commanded forward speed (training samples 0.45..0.8 on stairs, ..1.0 rough)")
    ap.add_argument("--stand", action="store_true", help="zero command")
    ap.add_argument("--seconds", type=float, default=14.0)
    ap.add_argument("--delay-steps", type=int, default=0, help="fixed action delay in physics steps (training: random 0..2)")
    ap.add_argument("--depth-lag", type=int, default=1, help="age of the newest depth frame in control steps (Isaac recording: 1)")
    ap.add_argument("--no-self-collision", action="store_true")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--run", default=str(DEFAULT_RUN), help="env.yaml, a folder with env.yaml, or a run folder with params/env.yaml")
    ap.add_argument("--export", default=str(DEFAULT_EXPORT), help="export folder: policy.onnx and (optional) adapter_card.json")
    ap.add_argument("--run-name", default=None, help="caption: run name (default: from the card's checkpoint path)")
    ap.add_argument("--checkpoint", default=None, help="caption: checkpoint number (default: the card's iteration)")
    ap.add_argument("--plant-label", default=None, help="caption: name of the gain set (default: card control.gains_label, else derived)")
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    export = pathlib.Path(args.export)
    ec = EnvConfig(args.run, export / "adapter_card.json")
    if ec.card is None:
        print(f"[warn] no adapter_card.json in {export}: settings come from env.yaml only, nothing is cross-checked")
    policy = OnnxPolicy(export / "policy.onnx")
    run_name, ckpt, cap2 = run_labels(ec, args.run_name, args.checkpoint, args.plant_label)
    print(f"[run] {run_name} checkpoint {ckpt}; env.yaml {ec.env_yaml}; obs {ec.obs_dim} = {ec.proprio_dim} + 8 x {ec.frame_shape[0]} x {ec.frame_shape[1]}")
    print(f"[run] {cap2}")
    assert policy.obs_dim == ec.obs_dim and policy.num_actions == len(ec.action_joints)
    terrain = Terrain(args.terrain)
    sim = X2Sim(ec, terrain, foot=args.foot, self_collision=not args.no_self_collision, delay_steps=args.delay_steps,
                depth_lag=args.depth_lag)
    m = sim.m
    print(f"[model] nbody {m.nbody} ngeom {m.ngeom} nu {m.nu} mass {m.body_subtreemass[1]:.2f} kg dt {m.opt.timestep} foot={args.foot}"
          f" terrain: {terrain.describe()}")
    sim.reset((0.0, 0.0, terrain.height(0, 0) + ec.init_z))
    print("[model] self contacts at spawn:", sim.self_contacts() or "none")
    target = np.array([args.target_dist, 0.0, terrain.height(args.target_dist, 0.0)])
    if args.stand:
        rec = run_closed_loop(sim, policy, args.seconds, cmd_fn=lambda s, sim_: np.zeros(3))
        target = None
    else:
        rec = run_closed_loop(sim, policy, args.seconds, target=target, vmax=args.vmax)
    summary, line = summarize(rec, args.name, 0.0 if args.stand else args.vmax)
    summary.update(run=run_name, checkpoint=ckpt, env_yaml=str(ec.env_yaml), export=str(export), settings=cap2, obs_dim=ec.obs_dim,
                   terrain=terrain.describe(), foot=args.foot, delay_steps=args.delay_steps, depth_lag=args.depth_lag,
                   target=None if target is None else target.tolist(), summary_line=line)
    print("[summary]", line)
    write_csv(rec, ec, out / f"{args.name}.csv")
    json.dump(summary, open(out / f"{args.name}.json", "w"), indent=1)
    sp = out / "summary.txt"  # one line per clip; the line of this clip is replaced
    old = [l for l in (sp.read_text().splitlines() if sp.exists() else []) if l.strip() and not l.startswith(args.name + ":")]
    sp.write_text(chr(10).join(old + [line]) + chr(10))
    np.savez_compressed(out / f"{args.name}_rollout.npz", t=rec["t"], qpos=rec["qpos"], cmd=rec["cmd"], action=rec["action"],
                        feet_force=rec["feet_force"], depth=rec["depth"].astype(np.float32), joint_names=np.array(ec.joints),
                        first_obs=rec["first_obs"])
    if not args.no_video:
        cap = (f"MuJoCo, not the training simulator; X2, Hiking track, run {run_name}, checkpoint {ckpt}, commanded "
               f"{0.0 if args.stand else args.vmax:.2f} m/s")
        for view in ("chase", "legs"):
            p = out / f"{args.name}_{view}.mp4"
            render_video(sim, rec, p, cap, view=view, target=target, fps=args.fps, caption2=cap2)
            print("[video]", p, f"{p.stat().st_size/1e6:.2f} MB")


if __name__ == "__main__":
    main()
