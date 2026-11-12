"""Export an X2 parkour checkpoint for the eval monitor, on the CPU, without Isaac Sim (training keeps running).

  CUDA_VISIBLE_DEVICES= nice -n 10 ~/hiking-env/bin/python export_for_eval.py <run_dir>/model_<N>.pt [out_dir]
        [--obs recording.npz] [--task NAME] [--plant NAME] [--card-only]

Everything in adapter_card.json is derived from the run itself: <run_dir>/params/env.yaml (joints' PD gains, armature,
effort / velocity limits, default pose, action joints and scale, observation terms with their sizes, scales and noise,
the depth camera body / pose / optics / crop / range / history, the command settings) and <run_dir>/params/agent.yaml
(network, encoder output size). Nothing is hard-coded per run: run 1 (head camera, 18 x 32 frames, 5408 observations,
armature-rule gains) and run 2 (torso camera, 36 x 32 frames, 10016 observations, measured gains) both export.
The joint order is Isaac Lab's breadth-first order computed from the run's URDF (scene.robot.spawn.asset_path).

Writes into out_dir (default /mnt/filesystem-g4/tracks/hiking/eval/<run>_<iteration>/):
  0-depth_encoder.onnx   depth (1, frames, rows, cols) -> embedding
  actor.onnx             [proprio, depth embedding] -> actions
  policy.onnx            both in one file: flat observation (n, total) -> actions
  adapter_card.json      control rate, joints, default pose, action scale, PD gains, observation terms in order, command,
                         depth sensor, and the facts verified against a simulator recording (and on which run)
  obs_check.npz          50 observation vectors with the torch and ONNX actions (SYNTHETIC observations unless --obs given: this
                         proves ONNX == torch; it does not prove an adapter builds the observation the way the simulator does)

--task   task name for the card (default: read from <run root>/_supervisor/supervisor.log, else "unknown")
--plant  name of the gain set for the card (default: "armature rule" when the gains follow it, else "per-joint gains ...")
--card-only  write only adapter_card.json (no torch, no ONNX): works before a checkpoint exists, pass the path it will have
"""
import json
import math
import os
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np
import yaml


def _opt(name):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else None


CKPT = sys.argv[1]
RUN_DIR = os.path.dirname(os.path.abspath(CKPT))
RUN = os.path.basename(os.path.dirname(RUN_DIR))
IT = os.path.basename(CKPT).split("_")[1].split(".")[0]
OUT = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else \
    "/mnt/filesystem-g4/tracks/hiking/eval/%s_%s" % (RUN, IT)
OBS_FILE = _opt("--obs")
CARD_ONLY = "--card-only" in sys.argv
os.makedirs(OUT, exist_ok=True)

env = yaml.unsafe_load(open(os.path.join(RUN_DIR, "params/env.yaml")))
agent = yaml.unsafe_load(open(os.path.join(RUN_DIR, "params/agent.yaml")))


def match(table, name, default=None):
    """Isaac Lab per-joint lookup: a scalar, or {regex: value} where the regex must match the whole name."""
    if table is None:
        return default
    if not isinstance(table, dict):
        return float(table)
    for pat, v in table.items():
        if re.fullmatch(pat, name):
            return float(v)
    return default


def isaac_joint_order(urdf_path):
    """Isaac Lab articulation order: breadth first from the root link, the moving joints under a link alphabetically
    (links behind fixed joints are merged into their parent)."""
    rob = ET.parse(urdf_path).getroot()
    kids, children = {}, set()
    for j in rob.findall("joint"):
        par, ch = j.find("parent").get("link"), j.find("child").get("link")
        kids.setdefault(par, []).append((j.get("name"), j.get("type"), ch))
        children.add(ch)
    roots = [l.get("name") for l in rob.findall("link") if l.get("name") not in children]
    assert len(roots) == 1, roots

    def moving(link):
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
    return roots[0], order


# ---- robot: joints, gains, limits, default pose, actions (all from env.yaml) ----
rob = env["scene"]["robot"]
URDF = rob["spawn"]["asset_path"]
ROOT_BODY, JOINTS = isaac_joint_order(URDF)
PORT = "/mnt/filesystem-g4/tracks/hiking/code/x2_port/files_instinctlab/assets/agibot_x2.py"
joint_order_check = "URDF breadth-first order (not cross-checked: %s not readable)" % PORT
if os.path.exists(PORT):  # the order the port checks at run time inside Isaac (check_x2_env.py)
    src = open(PORT).read()
    if "X2_ISAAC_JOINTS" in src:
        ns = {}
        exec(src[src.index("X2_ISAAC_JOINTS = ["):src.index("]", src.index("X2_ISAAC_JOINTS = [")) + 1], ns)
        assert ns["X2_ISAAC_JOINTS"] == JOINTS, "URDF breadth-first joint order differs from X2_ISAAC_JOINTS in the port"
        joint_order_check = "URDF breadth-first order, equal to X2_ISAAC_JOINTS of the port (which check_x2_env.py checks inside Isaac)"

kp, kd, armature, effort, vel_limit, act_class, delay = {}, {}, {}, {}, {}, {}, {}
for group, ac in rob["actuators"].items():
    for j in JOINTS:
        if any(re.fullmatch(e, j) for e in ac["joint_names_expr"]):
            assert j not in kp, "joint %s is in two actuator groups" % j
            kp[j] = match(ac["stiffness"], j)
            kd[j] = match(ac["damping"], j)
            armature[j] = match(ac.get("armature"), j, 0.0)
            effort[j] = match(ac.get("effort_limit_sim"), j, match(ac.get("effort_limit"), j))
            vel_limit[j] = match(ac.get("velocity_limit_sim"), j, match(ac.get("velocity_limit"), j))
            act_class[j] = ac["class_type"].split(":")[-1]
            delay[j] = (int(ac.get("min_delay", 0)), int(ac.get("max_delay", 0)))
assert all(j in kp and kp[j] is not None and kd[j] is not None and effort[j] is not None for j in JOINTS), "a joint has no actuator / gains"
kp, kd, armature, effort, vel_limit = ({j: d[j] for j in JOINTS} for d in (kp, kd, armature, effort, vel_limit))
default_pose = {j: next((float(v) for p, v in rob["init_state"]["joint_pos"].items() if re.fullmatch(p, j)), 0.0) for j in JOINTS}

a_cfg = env["actions"]["joint_pos"]
assert a_cfg["class_type"].endswith("JointPositionAction") and a_cfg["use_default_offset"] and a_cfg["clip"] is None, a_cfg
ACTION_JOINTS = [j for j in JOINTS if any(re.fullmatch(e, j) for e in a_cfg["joint_names"])]
action_scale = {j: match(a_cfg["scale"], j) for j in ACTION_JOINTS}
HELD = [j for j in JOINTS if j not in ACTION_JOINTS]
NJ, NA = len(JOINTS), len(ACTION_JOINTS)

W0 = 2 * math.pi * 10.0
armature_rule = all(abs(kp[j] - armature[j] * W0**2) <= 1e-4 * kp[j] and abs(kd[j] - 4 * armature[j] * W0) <= 1e-4 * kd[j] for j in JOINTS)
PLANT = _opt("--plant")
gains_source = (PLANT + " (name given at export); " if PLANT else "") + (
    "armature rule: kp = armature * (2 pi 10 Hz)^2, kd = 4 * armature * 2 pi 10 Hz (checked on the env.yaml values)" if armature_rule
    else "per-joint stiffness / damping of env.yaml, NOT the armature rule (checked on the env.yaml values)")
scale_rule = all(abs(action_scale[j] - 0.25 * effort[j] / (armature[j] * W0**2)) <= 1e-6 for j in ACTION_JOINTS if armature[j] > 0)

sim_dt, decimation = float(env["sim"]["dt"]), int(env["decimation"])
control_dt = sim_dt * decimation
dmin, dmax = min(d[0] for d in delay.values()), max(d[1] for d in delay.values())

# ---- depth camera (all from env.yaml) ----
cam = env["scene"]["camera"]
pc, noise = cam["pattern_cfg"], cam["noise_pipeline"]
CAM_BODY = cam["prim_path"].split("/")[-1]
RAW_W, RAW_H = int(pc["width"]), int(pc["height"])
crop_top, crop_bottom, crop_left, crop_right = (int(v) for v in noise["crop_and_resize"]["crop_region"])
assert noise["crop_and_resize"]["resize_shape"] is None, "resize after the crop is not handled"
ROWS, COLS = RAW_H - crop_top - crop_bottom, RAW_W - crop_left - crop_right
fov = [2 * math.degrees(math.atan(pc["horizontal_aperture"] / (2 * pc["focal_length"]))),
       2 * math.degrees(math.atan(pc["vertical_aperture"] / (2 * pc["focal_length"])))]
quat = [float(v) for v in cam["offset"]["rot"]]
qn = math.sqrt(sum(v * v for v in quat))
qw, qx, qy, qz = (v / qn for v in quat)
pitch_down = math.degrees(math.asin(max(-1.0, min(1.0, 2 * (qw * qy - qz * qx)))))
only_pitch = abs(qx) < 1e-9 and abs(qz) < 1e-9
drange = [float(v) for v in noise["depth_normalization"]["depth_range"]]
orange = [float(v) for v in noise["depth_normalization"]["output_range"]]
cam_hist = int(cam["data_histories"]["distance_to_image_plane_noised"])
cam_rate = 1.0 / float(cam["update_period"])

# ---- observation layout (policy group, flattened in the config order) ----
SKIP = {"concatenate_terms", "concatenate_dim", "enable_corruption", "history_length", "flatten_history_dim"}
DIM = {"base_lin_vel": 3, "base_ang_vel": 3, "projected_gravity": 3, "velocity_commands": 3, "joint_pos": NJ, "joint_vel": NJ, "actions": NA}
BASE = {
    "base_lin_vel": "root (%s) linear velocity in the root frame, m/s",
    "base_ang_vel": "root (%s) angular velocity in the root frame, rad/s",
    "projected_gravity": "unit gravity vector in the root (%s) frame",
    "velocity_commands": "command [vx m/s, vy m/s, yaw rate rad/s] in the root frame",
    "joint_pos": "joint position minus default pose, rad, all %d joints in joint_order" % NJ,
    "joint_vel": "joint velocity, rad/s, all %d joints in joint_order" % NJ,
    "actions": "previous raw policy outputs (before scaling), %d values in action_joint_order" % NA,
}


def group_terms(group):
    """[(name, flat size, shape for the network, history steps, config)] of one observation group, in order."""
    out = []
    for name, tc in env["observations"][group].items():
        if name in SKIP or tc is None:
            continue
        if name == "depth_image":
            p = tc["params"]
            frames = max(int(p.get("num_output_frames", 0)), 1)
            out.append((name, frames * ROWS * COLS, (frames, ROWS, COLS), frames, tc))
        else:
            assert name in DIM, "observation term %s: size not known to this script" % name
            hist = int(tc.get("history_length") or 1)
            out.append((name, DIM[name] * hist, (DIM[name] * hist,), hist, tc))
    return out


POLICY_TERMS = group_terms("policy")
dp = env["observations"]["policy"]["depth_image"]["params"]
FRAMES, SKIP_FRAMES = int(dp["num_output_frames"]), int(dp["history_skip_frames"])
assert (FRAMES - 1) * SKIP_FRAMES + 1 <= cam_hist
PROPRIO = sum(n for name, n, *_ in POLICY_TERMS if name != "depth_image")
TOTAL = sum(n for _, n, *_ in POLICY_TERMS)
assert POLICY_TERMS[-1][0] == "depth_image", "the depth image is expected to be the last policy term"
obs_format = {g: {name: shape for name, _, shape, _, _ in group_terms(g)} for g in ("policy", "critic")}
EMB = int(agent["policy"]["encoder_configs"]["depth_encoder"]["output_size"])

terms, off = [], 0
for name, n, shape, hist, tc in POLICY_TERMS:
    if name == "depth_image":
        what = "%d depth frames, oldest first (the last one is the newest), %d control steps apart, each %d rows x %d columns, values %g..%g" % (
            FRAMES, SKIP_FRAMES, ROWS, COLS, orange[0], orange[1])
    else:
        what = BASE[name] % ROOT_BODY if "%s" in BASE[name] else BASE[name]
        if tc.get("scale") is not None:
            what += ", x %g" % tc["scale"]
        if tc.get("noise") is not None:
            what += "; training noise +-%g (before scaling)" % tc["noise"]["n_max"]
    terms.append(dict(name=name, start=off, size=n, history_steps=hist, shape=list(shape), scale=tc.get("scale"),
                      noise_half_width=(tc.get("noise") or {}).get("n_max"), what=what))
    off += n

# ---- command (from env.yaml) ----
cmd = env["commands"]["base_velocity"]
vr = {k: dict(lin_vel_x=list(v["lin_vel_x"]), lin_vel_y=list(v["lin_vel_y"]), ang_vel_z=list(v["ang_vel_z"])) for k, v in (cmd.get("velocity_ranges") or {}).items()}
moving = [v["lin_vel_x"] for k, v in vr.items() if v["lin_vel_x"][1] > 0]
vx_lo, vx_hi = (max(v[0] for v in moving), min(v[1] for v in moving)) if moving else (0.0, 0.0)  # the range every moving terrain was trained on
by_range = {}
for k, v in vr.items():
    by_range.setdefault("%g-%g" % tuple(v["lin_vel_x"]), []).append(k)
yaw_lo, yaw_hi = (float(v) for v in cmd["ranges"]["ang_vel_z"])

# ---- task name ----
TASK = _opt("--task")
if TASK is None:
    TASK = "unknown"
    sup = os.path.join(os.path.dirname(RUN_DIR), "_supervisor", "supervisor.log")
    if os.path.exists(sup):
        found = re.findall(r"task[ =]([\w\-]+)", open(sup).read())
        if found:
            TASK = found[-1]

# ---- the network: build on the CPU, load the weights, export, compare ----
obs_kind, err1, err2 = "not run (--card-only)", None, None
if not CARD_ONLY:
    import torch
    from instinct_rl import modules

    pol = dict(agent["policy"])
    ac = modules.build_actor_critic(pol.pop("class_name"), pol, obs_format, num_actions=NA, num_rewards=1)
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)["model_state_dict"]
    missing, unexpected = ac.load_state_dict(sd, strict=False)
    assert not [k for k in missing if not k.startswith("std")] and not unexpected, (missing, unexpected)
    ac.eval()

    rng = np.random.default_rng(0)
    if OBS_FILE:
        obs_np = np.load(OBS_FILE)["obs"].astype(np.float32)[:50]
        assert obs_np.shape[1] == TOTAL, "the recording has %d values per observation, this run needs %d" % (obs_np.shape[1], TOTAL)
        obs_kind = "recorded in the training simulator (%s)" % OBS_FILE
    else:
        obs_np = np.concatenate([rng.normal(0, 0.3, (50, PROPRIO)), rng.uniform(0, 1, (50, TOTAL - PROPRIO))], 1).astype(np.float32)
        obs_kind = "SYNTHETIC (random proprioception, uniform depth): checks ONNX against torch only"
    obs = torch.from_numpy(obs_np)
    with torch.no_grad():
        act_torch = ac.act_inference(obs).numpy()
    ac.export_as_onnx(obs[:1], OUT)

    class Whole(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m.act_inference(x)

    with torch.no_grad():
        torch.onnx.export(Whole(ac), obs[:1], os.path.join(OUT, "policy.onnx"), input_names=["obs"], output_names=["actions"],
                          opset_version=18, dynamic_axes={"obs": {0: "n"}, "actions": {0: "n"}})

    import onnxruntime as ort

    sess = ort.InferenceSession(os.path.join(OUT, "policy.onnx"), providers=["CPUExecutionProvider"])
    act_onnx = sess.run(None, {"obs": obs_np})[0]
    enc = ort.InferenceSession(os.path.join(OUT, "0-depth_encoder.onnx"), providers=["CPUExecutionProvider"])
    act = ort.InferenceSession(os.path.join(OUT, "actor.onnx"), providers=["CPUExecutionProvider"])
    two = np.concatenate([
        act.run(None, {act.get_inputs()[0].name: np.concatenate(
            [o[None, :PROPRIO], enc.run(None, {enc.get_inputs()[0].name: o[None, PROPRIO:].reshape(1, FRAMES, ROWS, COLS)})[0]], 1)})[0]
        for o in obs_np])
    err1, err2 = float(np.abs(act_onnx - act_torch).max()), float(np.abs(two - act_torch).max())
    print("max |onnx - torch|: policy.onnx %.2e, encoder+actor %.2e" % (err1, err2))
    assert err1 < 1e-4 and err2 < 1e-4
    np.savez(os.path.join(OUT, "obs_check.npz"), obs=obs_np, actions_torch=act_torch, actions_onnx=act_onnx,
             note=np.array(obs_kind), term_names=np.array([t[0] for t in POLICY_TERMS]), term_sizes=np.array([t[1] for t in POLICY_TERMS]))

# ---- the card ----
V1 = "verified on run 1 only (x2_parkour_v1 model_500, Isaac recording obs_check_sim_state.npz, 50 control steps; sim2sim validation.txt)"
same_as_run1 = CAM_BODY == "head_pitch_link" and (ROWS, COLS) == (18, 32)
card = dict(
    track="hiking", method="Hiking in the Wild (arXiv 2601.07718), X2 port", task=TASK, run=RUN,
    checkpoint=CKPT, iteration=int(IT), source="every value below is derived from %s and params/agent.yaml by export_for_eval.py" % os.path.join(RUN_DIR, "params/env.yaml"),
    obs_check=obs_kind, onnx_vs_torch_max_abs_error=None if err1 is None else max(err1, err2),
    files={"policy.onnx": "obs (n, %d) -> actions (n, %d)" % (TOTAL, NA),
           "0-depth_encoder.onnx + actor.onnx": "depth (1,%d,%d,%d) -> %d; [obs[:%d], %d] -> %d actions" % (FRAMES, ROWS, COLS, EMB, PROPRIO, EMB, NA)},
    robot=dict(model="%s (training URDF, root link %s, fixed links merged: %s; collision: %s%s)" % (
                   URDF, ROOT_BODY, rob["spawn"].get("merge_fixed_joints"), rob["spawn"].get("collider_type"),
                   ", self-collision on" if (rob["spawn"].get("articulation_props") or {}).get("enabled_self_collisions") else ""),
               root_body=ROOT_BODY, joint_order_source=joint_order_check,
               note="training simulator feet are the mesh hull, not the MJCF sole spheres"),
    control=dict(policy_rate_hz=round(1.0 / control_dt, 6), sim_dt_s=sim_dt, decimation=decimation,
                 law="tau = kp * (q_target - q) - kd * qd, clipped to effort; q_target = default_pose + action_scale * action",
                 actuator=sorted(set(act_class.values())), gains_source=gains_source, gains_label=PLANT or ("armature rule" if armature_rule else "per-joint env.yaml gains"),
                 gains_follow_armature_rule=armature_rule,
                 action_scale_rule="0.25 * effort / (armature * (2 pi 10 Hz)^2): %s" % ("holds for every action joint" if scale_rule else "does NOT hold; use the table"),
                 action_delay="training randomises %d-%d physics steps (%g-%g ms) of command delay" % (dmin, dmax, dmin * sim_dt * 1000, dmax * sim_dt * 1000),
                 action_delay_physics_steps=[dmin, dmax],
                 head=("%s are not policy outputs: hold at the default pose with their kp/kd" % " and ".join(HELD)) if HELD else "every joint is a policy output"),
    joint_order=JOINTS, action_joint_order=ACTION_JOINTS, held_joints=HELD,
    default_pose=default_pose, action_scale=action_scale,
    kp=kp, kd=kd, effort_limit=effort, velocity_limit=vel_limit, armature=armature,
    observation=dict(total=TOTAL, proprio=PROPRIO, terms=terms,
                     history="each proprio term is %d control steps flattened [t-%d .. t], oldest first, newest last; at reset the history is filled with the first value" % (
                         POLICY_TERMS[0][3], POLICY_TERMS[0][3] - 1),
                     joint_pos_and_vel_order="joint_order (%d%s)" % (NJ, ", includes the held joints" if HELD else ""),
                     actions_order="action_joint_order (%d)" % NA,
                     noise_enabled_in_training=bool(env["observations"]["policy"].get("enable_corruption"))),
    command=dict(obs="[vx, vy, yaw_rate] in the root frame", generator=cmd["class_type"].split(":")[-1],
                 velocity_control_stiffness=float(cmd["velocity_control_stiffness"]), heading_control_stiffness=float(cmd["heading_control_stiffness"]),
                 only_positive_lin_vel_x=bool(cmd["only_positive_lin_vel_x"]), yaw_rate_range=[yaw_lo, yaw_hi],
                 target_stop_radius_m=float(cmd["target_dis_threshold"]), standing_fraction=float(cmd["rel_standing_envs"]),
                 resampling_time_range_s=list(cmd["resampling_time_range"]), velocity_ranges_by_terrain=vr,
                 training_ranges="vx by terrain: %s; vy %s; yaw rate %g..%g rad/s" % (
                     "; ".join("%s m/s on %s" % (k, ", ".join(v)) for k, v in by_range.items()),
                     "0" if all(v["lin_vel_y"] == [0.0, 0.0] for v in vr.values()) else "see velocity_ranges_by_terrain", yaw_lo, yaw_hi),
                 how_training_makes_it="a target point on the terrain: vx = clip(%g * forward distance to the target, 0, sampled speed), yaw rate = clip(%g * heading error to the target, %g, %g), all zero within %g m of the target; %g%% of robots stand (all zeros)" % (
                     cmd["velocity_control_stiffness"], cmd["heading_control_stiffness"], yaw_lo, yaw_hi, cmd["target_dis_threshold"], 100 * cmd["rel_standing_envs"]),
                 for_waypoints="vx = chosen speed in %g-%g, vy = 0, yaw_rate = clip(%g * (bearing to waypoint - yaw), %g, %g); zeros to stand" % (
                     vx_lo, vx_hi, cmd["heading_control_stiffness"], yaw_lo, yaw_hi)),
    depth=dict(body=CAM_BODY, pos_in_body=[float(v) for v in cam["offset"]["pos"]], quat_wxyz=[qw, qx, qy, qz], convention=cam["offset"]["convention"],
               pitch_down_deg=pitch_down,
               orientation="x forward, z up camera frame %s (quaternion wxyz [%.5f, %.5f, %.5f, %.5f]) in the frame of %s" % (
                   ("pitched %.1f deg down about y" % pitch_down) if only_pitch else "rotated by the quaternion", qw, qx, qy, qz, CAM_BODY),
               type="distance to image plane (z depth), pinhole", raw_size=[RAW_W, RAW_H], fov_deg=[round(fov[0], 2), round(fov[1], 2)],
               focal_length=float(pc["focal_length"]), horizontal_aperture=float(pc["horizontal_aperture"]), vertical_aperture=float(pc["vertical_aperture"]),
               rate_hz=round(cam_rate, 6), min_distance_m=float(cam["min_distance"]), sees="terrain and the robot's own links",
               crop_region=[crop_top, crop_bottom, crop_left, crop_right], frame_shape=[ROWS, COLS], depth_range_m=drange, output_range=orange,
               blur=dict(kernel_size=int(noise["gaussian_blur"]["kernel_size"]), sigma=float(noise["gaussian_blur"]["sigma"])),
               history_frames=cam_hist, stack_frames=FRAMES, stack_skip=SKIP_FRAMES,
               preprocessing=["crop: rows %d..(%d-%d), columns %d..(%d-%d) -> %d x %d (crop_region top %d, bottom %d, left %d, right %d)" % (
                                  crop_top, RAW_H, crop_bottom, crop_left, RAW_W, crop_right, ROWS, COLS, crop_top, crop_bottom, crop_left, crop_right),
                              "gaussian blur %dx%d, sigma %g" % (noise["gaussian_blur"]["kernel_size"], noise["gaussian_blur"]["kernel_size"], noise["gaussian_blur"]["sigma"]),
                              "clip to %g..%g m and divide by %g (beyond range = %g)" % (drange[0], drange[1], drange[1] - drange[0], orange[1])],
               pipeline=list(noise.keys()),
               stack="%d frames taken every %dth frame from a %d-frame %g Hz history (spanning %.1f s), oldest first; the configured extra delay %s frames is cast to an integer and is 0 in practice" % (
                   FRAMES, SKIP_FRAMES, cam_hist, cam_rate, (FRAMES - 1) * SKIP_FRAMES / cam_rate, list(dp["delayed_frame_ranges"])),
               noise="the noise pipeline is exactly the steps in `pipeline` (no extra depth noise terms)"),
    verified=dict(
        about="facts checked outside Isaac by the MuJoCo port (sim2sim_mujoco_x2.py) against a simulator recording. " + (
            "This card is for run 1's observation layout." if same_as_run1 else
            "No simulator recording exists for THIS run's camera and frame shape: the items below were verified on run 1 only and hold here through the same code path, not by a direct comparison."),
        proprio_history="every proprio term is 8 control steps, OLDEST FIRST, newest last, filled with the first value at reset: exact match. " + V1,
        proprio_values="rebuilt from the recorded simulator state, all terms within the training noise bound; projected gravity from the wxyz root quaternion to 4e-8. " + V1,
        actions_term="the newest `actions` entry at step t is the policy output of step t-1 (zeros at the reset step): exact. " + V1,
        depth_stack_order="depth frames are OLDEST FIRST, the NEWEST is the LAST (index %d); neighbouring frames are %d control steps apart: exact. %s" % (FRAMES - 1, SKIP_FRAMES, V1),
        depth_timing="the newest depth frame of control step t is the image of step t-1 (one control step old): bit-identical frames at steps 0 and 1, and best fit at lag 1 against a ray-cast. " + V1,
        depth_orientation="row 0 is the top of the image (far), the last row is the bottom (near ground); column 0 is the camera's left: correlation +0.99 with a MuJoCo ray-cast, -0.97 when flipped. " + V1,
        depth_values="near-ground rows agree with a MuJoCo flat-ground ray-cast to 0.001 (2.5 mm); far rows were not comparable (Isaac terrain not available). " + V1,
        mujoco_velocity_frame="in MuJoCo read the root's body-frame velocity with mjOBJ_XBODY (link frame); mjOBJ_BODY is the inertial frame, rotated for the X2 pelvis, and gave a gyro error up to 2.8 rad/s. " + V1 + " (robot fact, independent of the run)",
        command_at_reset="the command in the observation of the reset step is [0, 0, 0]. " + V1,
        onnx="policy.onnx reproduces the recorded simulator actions to 1e-6 when fed the recorded observations. " + V1),
    not_verified=(["this run's depth image against Isaac (no recording of this run): camera pose, optics, crop %s and range %g m are read from env.yaml and only checked geometrically in MuJoCo" % ([crop_top, crop_bottom, crop_left, crop_right], drange[1]),
                   "this run's PD gains in closed loop against Isaac (no recording of this run)"] if not same_as_run1 else []) + [
        "depth on stairs (far rows) against Isaac: the Isaac terrain mesh was not available outside Isaac",
        "the robot's own body in the depth image against Isaac (not visible in the recorded poses of run 1)",
        "which action delay (0, 1 or 2 physics steps) a given robot has; the first control step after a reset is not reproduced exactly in MuJoCo"],
)
json.dump(card, open(os.path.join(OUT, "adapter_card.json"), "w"), indent=1)
print("observation %d = %d proprio + %d x %d x %d depth; camera on %s, %.1f deg down, %.2f x %.2f deg; gains: %s" % (
    TOTAL, PROPRIO, FRAMES, ROWS, COLS, CAM_BODY, pitch_down, fov[0], fov[1], gains_source))
print("EXPORT DONE", OUT, sorted(os.listdir(OUT)))
