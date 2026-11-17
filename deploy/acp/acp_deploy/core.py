"""ACP deploy core: everything from the robot's own sensors to joint targets, pure Python (numpy, onnxruntime, torch for the
student). The same code runs in the sim node, the parity test and on PC2. Plan: deploy/acp/PLAN.md.

Inputs per control step (50 Hz): the torso IMU (orientation as a rotation matrix, angular velocity in the torso frame), joint
positions / velocities by name, the goal (gx, gy in the heading frame, already bounded; dpsi rad) and, every other step, a
chest depth frame (45 x 80 z-depth metres, 0 = no return).
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "hiking"))
from hiking_deploy.core import WAIST, pelvis_from_torso  # noqa: E402  (waist FK, validated in Hiking's M1)

HIST = 5
CAM_POS = np.array([0.131, 0.0, 0.259])          # chest bracket on torso_link (docs/hardware/chest_camera_mount, variant H)
CAM_DOWN = 61.6                                  # deg below the torso's x axis (the trained pose)
WAIST_Z = 0.06706 + 0.0880106                    # pelvis -> waist_yaw_link -> waist_pitch_link (torso_link at the pitch link)


def _rot_y(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])


def _rot_x(deg):
    a = math.radians(deg)
    return np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])


def cam_T(R_pelvis, q_waist, cam_pitch=0.0, cam_roll=0.0):
    """The student's camera pose: rotation camera -> heading frame (3 x 3, flattened) and position in the heading frame
    relative to the pelvis, from the pelvis attitude (IMU via the waist FK) and the waist joints. cam_pitch / cam_roll: the
    mounted camera's offset from the trained pose (deg; the leg calibration, deploy/hiking/tools/mount_calib.py)."""
    yaw, pitch, roll = (float(v) for v in q_waist)
    c, s = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    R_pt = Rz @ _rot_y(pitch * 180 / math.pi) @ _rot_x(roll * 180 / math.pi)          # pelvis -> torso_link
    h = math.atan2(R_pelvis[1, 0], R_pelvis[0, 0])
    ch, sh = math.cos(h), math.sin(h)
    Rh = np.array([[ch, sh, 0], [-sh, ch, 0], [0, 0, 1]])                               # world -> heading frame
    R_hp = Rh @ R_pelvis                                                                # pelvis -> heading frame
    R_cam = R_hp @ R_pt @ _rot_y(CAM_DOWN + cam_pitch) @ _rot_x(cam_roll)
    p_cam = R_hp @ (np.array([0.0, 0.0, WAIST_Z]) + R_pt @ CAM_POS)
    return np.concatenate((R_cam.ravel(), p_cam)).astype(np.float32)


class Mlp:
    """actor.npz layers (torch layout, ELU between, none after the last), as eval/adapters/common.NpMLP."""

    def __init__(self, z, prefix="actor"):
        self.layers, i = [], 0
        while f"{prefix}_w{i}" in z:
            self.layers.append((np.asarray(z[f"{prefix}_w{i}"], np.float64), np.asarray(z[f"{prefix}_b{i}"], np.float64)))
            i += 1

    def __call__(self, x):
        for k, (w, b) in enumerate(self.layers):
            x = w @ x + b
            if k < len(self.layers) - 1:
                x = np.where(x > 0, x, np.expm1(np.minimum(x, 0)))
        return x


class ACPCore:
    def __init__(self, acp_dir, student_path, threads=1):
        import onnxruntime as ort
        import torch
        z = np.load(os.path.join(acp_dir, "actor.npz"), allow_pickle=True)
        self.actor = Mlp(z)
        self.mean, self.std = np.asarray(z["norm_mean"], np.float64), np.sqrt(np.asarray(z["norm_var"], np.float64)) + 1e-5
        self.q0, self.scale = np.asarray(z["default_q"], float), np.asarray(z["action_scale"], float)
        self.kp, self.kd = np.asarray(z["kp"], float), np.asarray(z["kd"], float)
        self.joints = [str(n) for n in z["joint_names"]]
        self.goal_r = float(z["goal_r"]) if "goal_r" in z else 0.3
        so = ort.SessionOptions()
        so.intra_op_num_threads = so.inter_op_num_threads = threads
        self.planner = ort.InferenceSession(os.path.join(acp_dir, "planner_attn.onnx"), sess_options=so, providers=["CPUExecutionProvider"])
        torch.set_num_threads(threads)
        self.torch = torch
        self.student = torch.jit.load(student_path, map_location="cpu").eval()
        self.iw = [self.joints.index(n) for n in WAIST]
        self.reset()

    def reset(self):
        self.a_prev = np.zeros(len(self.joints))
        self.hist, self.h, self.scans, self.fine = None, None, None, None

    def proprio(self, R_pelvis, w_pelvis, q, qd):
        return np.concatenate((w_pelvis, -R_pelvis[2, :], q - self.q0, qd, self.a_prev))

    def student_step(self, depth, prop, camT):
        """-> planner scan (187), tracker scan (63): height minus root height (clipped +-2 m, as ACP reads them)."""
        t = self.torch
        with t.no_grad():
            plan, track, self.h, fine = self.student(t.from_numpy(np.asarray(depth, np.float32))[None, None],
                                                     t.from_numpy(np.asarray(prop, np.float32))[None],
                                                     t.from_numpy(camT)[None], self.h if self.h is not None else t.zeros(0))
        self.fine = fine[0].numpy()
        self.scans = (np.clip(plan[0].numpy().astype(float), -2, 2), np.clip(track[0].numpy().astype(float), -2, 2))
        return self.scans

    def step(self, R_torso, w_torso, q_by_name, qd_by_name, goal, depth=None, cam_pitch=0.0, cam_roll=0.0):
        """goal: (gx, gy, dpsi). depth: a new camera frame (45 x 80) or None (the last scans are reused).
        -> (joint targets (31, ACP joint order), info)."""
        q = np.array([q_by_name[n] for n in self.joints])
        qd = np.array([qd_by_name[n] for n in self.joints])
        R_p, w_p = pelvis_from_torso(R_torso, w_torso, q[self.iw], qd[self.iw])
        prop = self.proprio(R_p, w_p, q, qd)
        if self.hist is None:
            self.hist = np.tile(prop, (HIST, 1))
        else:
            self.hist = np.concatenate((self.hist[1:], prop[None]), 0)
        if depth is not None or self.scans is None:
            if depth is None:
                depth = np.zeros((45, 80), np.float32)
            self.student_step(depth, prop, cam_T(R_p, q[self.iw], cam_pitch, cam_roll))
        plan, track = self.scans
        hist = self.hist.ravel()
        goal = np.asarray(goal, float)
        tg, occ = self.planner.run(["contact_targets", "occupancy"], {"proprio_history": hist[None].astype(np.float32),
                                                                       "goal_command": goal[None].astype(np.float32),
                                                                       "scan_17x11": plan[None].astype(np.float32)})
        tg, occ = tg[0].astype(float), occ[0].astype(float)
        tgt = tg * (occ > 0.5)[:, None]
        obs = np.concatenate((hist, track, tgt.ravel(), goal))
        a = np.clip(self.actor(np.clip((obs - self.mean) / self.std, -10, 10)), -20.0, 20.0)
        self.a_prev = a
        return self.q0 + self.scale * a, dict(obs=obs, action=a, plan=plan, track=track, targets=tg, occ=occ, R_pelvis=R_p, prop=prop)


def bounded_goal(d_heading, r):
    """A direction in the heading frame -> d / max(|d|, r) (unit far away, shrinking inside r), as the eval's bounded_goal."""
    d = np.asarray(d_heading, float)
    return d / max(float(np.linalg.norm(d)), r)
