"""Quick check of the X2 parkour env (not a result): joint order, bodies, standing under zero actions, depth image.

  python check_x2_env.py --headless --num_envs 64 --steps 150 --out /mnt/filesystem-g4/tracks/hiking/runs/_smoke
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--task", default="Instinct-Parkour-Target-Amp-X2-v0")
p.add_argument("--num_envs", type=int, default=64)
p.add_argument("--steps", type=int, default=150)
p.add_argument("--out", default="/mnt/filesystem-g4/tracks/hiking/runs/_smoke")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch

import instinctlab.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
env = gym.make(args.task, cfg=cfg).unwrapped
robot = env.scene["robot"]
print("JOINTS", robot.joint_names)
print("BODIES", robot.body_names)
print("ACTION_DIM", env.action_manager.total_action_dim, "ACTION_JOINTS", env.action_manager.get_term("joint_pos")._joint_names)
if "X2" in args.task:
    from instinctlab.assets.agibot_x2 import X2_ISAAC_JOINTS

    assert list(robot.joint_names) == X2_ISAAC_JOINTS, "joint order differs from X2_ISAAC_JOINTS: the mirror maps are wrong"
    print("JOINT ORDER OK")
print("default root", robot.data.default_root_state[0, :3].tolist())
print("stiffness", dict(zip(robot.joint_names, robot.data.joint_stiffness[0].tolist())))
print("effort", dict(zip(robot.joint_names, robot.data.joint_effort_limits[0].tolist())))
import json

table = {}
for gname, act in robot.actuators.items():
    for k, j in enumerate(act.joint_names):
        table[j] = dict(group=gname, actuator=type(act).__name__, kp=float(act.stiffness[0, k]), kd=float(act.damping[0, k]),
                        effort_limit=float(act.effort_limit[0, k]), velocity_limit=float(act.velocity_limit[0, k]))
term = env.action_manager.get_term("joint_pos")
scale = term._scale[0].tolist() if hasattr(term._scale, "tolist") else [float(term._scale)] * len(term._joint_names)
for j, sc in zip(term._joint_names, scale):
    table[j]["action_scale"] = sc
for k, j in enumerate(robot.joint_names):
    table[j].update(armature=float(robot.data.joint_armature[0, k]), default_pos=float(robot.data.default_joint_pos[0, k]),
                    pos_limit=[float(x) for x in robot.data.joint_pos_limits[0, k]])
os.makedirs(args.out, exist_ok=True)
json.dump(dict(task=args.task, joints=table, body_mass={b: float(m) for b, m in zip(robot.body_names, robot.data.default_mass[0])},
               total_mass=float(robot.data.default_mass[0].sum())), open(os.path.join(args.out, "applied_%s.json" % args.task), "w"), indent=1)
print("GAINS", {j: (round(v["kp"], 2), round(v["kd"], 3)) for j, v in table.items()})
print("TOTAL MASS %.3f" % float(robot.data.default_mass[0].sum()))
cam = env.scene["camera"]
print("CAMERA prim", cam.cfg.prim_path, "offset", cam.cfg.offset.pos, cam.cfg.offset.rot)
obs, _ = env.reset()
for g, v in obs.items():
    print("OBS", g, {k: tuple(x.shape) for k, x in v.items()} if isinstance(v, dict) else tuple(v.shape))
os.makedirs(args.out, exist_ok=True)
n_done = torch.zeros(env.num_envs, device=env.device)
reasons = {}
z = []
for i in range(args.steps):
    obs, rew, term, trunc, info = env.step(torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device))
    n_done += (term | trunc).float()
    for name in env.termination_manager.active_terms:
        reasons[name] = reasons.get(name, 0) + int(env.termination_manager.get_term(name).sum())
    if i < 2:
        cs = env.scene["contact_forces"]
        f = cs.data.net_forces_w_history.norm(dim=-1).amax(dim=1)
        print("CONTACT step", i, {n: round(f[:, k].mean().item(), 1) for k, n in enumerate(cs.body_names) if f[:, k].max() > 1.0})
    z.append((robot.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]).mean().item())
    if i in (0, 25, 100):
        d = obs["policy"]["depth_image"]
        print("DEPTH step", i, tuple(d.shape), "min %.3f mean %.3f max %.3f" % (d.min(), d.mean(), d.max()),
              "frac<0.08 %.3f frac>0.99 %.3f" % ((d < 0.08).float().mean(), (d > 0.99).float().mean()))
        np.save(os.path.join(args.out, "depth_%s_%d.npy" % (args.task, i)), d[:32].cpu().numpy())
        np.save(os.path.join(args.out, "terrain_type_%s.npy" % args.task), env.scene.terrain.terrain_types[:32].cpu().numpy())
print("zero-action: steps", args.steps, "episodes ended per env %.2f" % n_done.mean().item(), "reasons", reasons)
print("mean root height above env origin: first %.3f, step 25 %.3f, last %.3f" % (z[0], z[min(25, len(z) - 1)], z[-1]))
print("reward keys", {k: round(v.mean().item(), 4) for k, v in rew.items()} if isinstance(rew, dict) else rew.mean().item())
print("CHECK DONE")
env.close()
app.close()
