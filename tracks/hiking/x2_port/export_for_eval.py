"""Export an X2 parkour checkpoint for the eval monitor, on the CPU, without Isaac Sim (training keeps running).

  CUDA_VISIBLE_DEVICES= ~/hiking-env/bin/python export_for_eval.py <checkpoint.pt> [out_dir]

Writes into out_dir (default /mnt/filesystem-g4/tracks/hiking/eval/<run>_<iteration>/):
  0-depth_encoder.onnx   depth (1, 8, 18, 32) -> 128
  actor.onnx             [proprio 800, depth embedding 128] -> 29 actions
  policy.onnx            both in one file: flat observation (1, 5408) -> 29 actions
  adapter_card.json      control rate, joints, default pose, action scale, PD gains, observation terms in order, command, depth sensor
  obs_check.npz          50 observation vectors with the torch and ONNX actions (SYNTHETIC observations unless --obs given: this
                         proves ONNX == torch; it does not prove an adapter builds the observation the way the simulator does)
"""
import json
import math
import os
import sys

import numpy as np
import torch
import yaml

CKPT = sys.argv[1]
RUN_DIR = os.path.dirname(CKPT)
IT = os.path.basename(CKPT).split("_")[1].split(".")[0]
OUT = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else \
    "/mnt/filesystem-g4/tracks/hiking/eval/%s_%s" % (os.path.basename(os.path.dirname(RUN_DIR)), IT)
OBS_FILE = sys.argv[sys.argv.index("--obs") + 1] if "--obs" in sys.argv else None
os.makedirs(OUT, exist_ok=True)

# ---- the X2 definition, from the port (no Isaac import: the asset file is read as text) ----
PORT = "/mnt/filesystem-g4/tracks/hiking/code/x2_port/files_instinctlab/assets/agibot_x2.py"
src = open(PORT).read()
ns = {"os": os}
exec(src[src.index("ARMATURE_PF52"):src.index("x2_delayed_actuators")], ns)
exec(src[src.index("# Joint order Isaac Lab"):src.index("X2_LINKS")], ns)
import re

JOINTS = ns["X2_ISAAC_JOINTS"]
PLANT, W, Z = ns["_PLANT"], ns["NATURAL_FREQ"], ns["DAMPING_RATIO"]


def plant(j):
    for pat, v in PLANT.items():
        if re.fullmatch(pat, j):
            return v
    raise KeyError(j)


DEFAULT = {".*_hip_pitch_joint": -0.312, ".*_knee_joint": 0.669, ".*_ankle_pitch_joint": -0.363, ".*_elbow_joint": -0.6,
           "left_shoulder_roll_joint": 0.2, "left_shoulder_pitch_joint": 0.2, "right_shoulder_roll_joint": -0.2,
           "right_shoulder_pitch_joint": 0.2}
default_q = [next((v for p, v in DEFAULT.items() if re.fullmatch(p, j)), 0.0) for j in JOINTS]
ACTION_JOINTS = [j for j in JOINTS if not j.startswith("head_")]
kp = {j: plant(j)[0] * W**2 for j in JOINTS}
kd = {j: 2 * Z * plant(j)[0] * W for j in JOINTS}

# ---- observation layout (policy group, flattened in this order; every term is an 8-step history, oldest first) ----
NJ, NA, H = len(JOINTS), len(ACTION_JOINTS), 8
TERMS = [
    ("base_ang_vel", 3 * H, "root (pelvis) angular velocity in the root frame, rad/s, x 0.25; training noise +-0.2"),
    ("projected_gravity", 3 * H, "unit gravity vector in the root (pelvis) frame; training noise +-0.05"),
    ("velocity_commands", 3 * H, "command [vx m/s, vy m/s, yaw rate rad/s] in the root frame"),
    ("joint_pos", NJ * H, "joint position minus default pose, rad, all 31 joints in joint_order; training noise +-0.01"),
    ("joint_vel", NJ * H, "joint velocity, rad/s, x 0.05, all 31 joints in joint_order; training noise +-0.5"),
    ("actions", NA * H, "previous raw policy outputs (before scaling), 29 values in action_joint_order"),
    ("depth_image", 8 * 18 * 32, "8 depth frames (1 newest .. every 5th frame of a 37-frame, 50 Hz history), each 18 x 32, values 0..1"),
]
PROPRIO = sum(n for name, n, _ in TERMS if name != "depth_image")
TOTAL = sum(n for _, n, _ in TERMS)
obs_format = {"policy": {name: ((8, 18, 32) if name == "depth_image" else (n,)) for name, n, _ in TERMS}}
obs_format["critic"] = {"base_lin_vel": (3 * H,), **obs_format["policy"]}

# ---- build the network on the CPU and load the weights ----
from instinct_rl import modules

agent = yaml.unsafe_load(open(os.path.join(RUN_DIR, "params/agent.yaml")))
pol = dict(agent["policy"])
ac = modules.build_actor_critic(pol.pop("class_name"), pol, obs_format, num_actions=NA, num_rewards=1)
sd = torch.load(CKPT, map_location="cpu", weights_only=False)["model_state_dict"]
missing, unexpected = ac.load_state_dict(sd, strict=False)
assert not [k for k in missing if not k.startswith("std")] and not unexpected, (missing, unexpected)
ac.eval()

rng = np.random.default_rng(0)
if OBS_FILE:
    obs_np = np.load(OBS_FILE)["obs"].astype(np.float32)[:50]
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
        [o[None, :PROPRIO], enc.run(None, {enc.get_inputs()[0].name: o[None, PROPRIO:].reshape(1, 8, 18, 32)})[0]], 1)})[0]
    for o in obs_np])
err1, err2 = float(np.abs(act_onnx - act_torch).max()), float(np.abs(two - act_torch).max())
print("max |onnx - torch|: policy.onnx %.2e, encoder+actor %.2e" % (err1, err2))
assert err1 < 1e-4 and err2 < 1e-4
np.savez(os.path.join(OUT, "obs_check.npz"), obs=obs_np, actions_torch=act_torch, actions_onnx=act_onnx,
         note=np.array(obs_kind), term_names=np.array([t[0] for t in TERMS]), term_sizes=np.array([t[1] for t in TERMS]))

off, terms = 0, []
for name, n, what in TERMS:
    terms.append(dict(name=name, start=off, size=n, history_steps=8, what=what))
    off += n
pitch = math.radians(40.0)
card = dict(
    track="hiking", method="Hiking in the Wild (arXiv 2601.07718), X2 port", task="Instinct-Parkour-Target-Amp-X2-v0",
    checkpoint=CKPT, iteration=int(IT), obs_check=obs_kind, onnx_vs_torch_max_abs_error=max(err1, err2),
    files={"policy.onnx": "obs (n, %d) -> actions (n, %d)" % (TOTAL, NA),
           "0-depth_encoder.onnx + actor.onnx": "depth (1,8,18,32) -> 128; [obs[:%d], 128] -> %d actions" % (PROPRIO, NA)},
    robot=dict(model="/mnt/filesystem-g4/data/robots/x2_ultra (URDF with base_link removed in training; MJCF for MuJoCo)",
               root_body="pelvis", note="training simulator feet are the mesh hull, not the MJCF sole spheres"),
    control=dict(policy_rate_hz=50, sim_dt_s=0.005, decimation=4,
                 law="tau = kp * (q_target - q) - kd * qd, clipped to effort; q_target = default_pose + action_scale * action",
                 action_delay="training randomises 0-2 physics steps (0-10 ms) of command delay",
                 head="head_yaw and head_pitch are not policy outputs: hold both at 0 with their kp/kd"),
    joint_order=JOINTS, action_joint_order=ACTION_JOINTS,
    default_pose={j: q for j, q in zip(JOINTS, default_q)},
    action_scale={j: 0.25 * plant(j)[1] / kp[j] for j in ACTION_JOINTS},
    kp=kp, kd=kd, effort_limit={j: plant(j)[1] for j in JOINTS}, velocity_limit={j: plant(j)[2] for j in JOINTS},
    armature={j: plant(j)[0] for j in JOINTS},
    observation=dict(total=TOTAL, proprio=PROPRIO, terms=terms,
                     history="each proprio term is 8 control steps flattened [t-7 .. t]; at reset the history is filled with the first value",
                     joint_pos_and_vel_order="joint_order (31, includes the head)", actions_order="action_joint_order (29)"),
    command=dict(obs="[vx, vy, yaw_rate] in the root frame", training_ranges="vx 0.45-1.0 m/s on rough ground, 0.45-0.8 on stairs, gaps, boxes, slopes; vy 0; yaw rate -1..1 rad/s",
                 how_training_makes_it="a target point on the terrain: vx is the sampled speed, yaw rate = clip(2.0 * heading error to the target, -1, 1); 5% of robots stand (all zeros)",
                 for_waypoints="vx = chosen speed in 0.45-0.8, vy = 0, yaw_rate = clip(2 * (bearing to waypoint - yaw), -1, 1); zeros to stand"),
    depth=dict(body="head_pitch_link", pos_in_body=[0.05761, -0.011183, -0.04837],
               orientation="x forward, z up camera frame pitched 40 deg down about y (quaternion wxyz [%.5f, 0, %.5f, 0]); matches the URDF joint rgbd_head_front" % (math.cos(pitch / 2), math.sin(pitch / 2)),
               type="distance to image plane (z depth), pinhole", raw_size=[64, 36], fov_deg=[89.51, 58.29], rate_hz=50,
               min_distance_m=0.1, sees="terrain and the robot's own links",
               preprocessing=["crop: rows 18..(36-0), columns 16..(64-16) -> 18 x 32 (crop_region top 18, bottom 0, left 16, right 16)",
                              "gaussian blur 3x3, sigma 1", "clip to 0..2.5 m and divide by 2.5 (beyond range = 1.0)"],
               stack="8 frames taken every 5th frame from a 37-frame 50 Hz history (spanning 0.7 s), plus a random 0-1 frame delay in training",
               noise="the base config's noise pipeline is exactly the three steps above (no extra depth noise terms enabled)"),
    not_verified=["the depth preprocessing list was read from the config, not compared numerically outside Isaac",
                  "frame order inside the 8-frame depth stack (proprio history is oldest first, checked on a simulator recording)", "no MuJoCo run of this policy yet"],
)
json.dump(card, open(os.path.join(OUT, "adapter_card.json"), "w"), indent=1)
print("EXPORT DONE", OUT, os.listdir(OUT))
