"""ACP track adapter: the contact-conditioned tracker (and, from stage 2 on, its contact planner).

Written 2026-11-11 against the track's stage-1 code (run s1_v1, `code_snapshot/acp/env.py` and `train.py`), before any
route-walking checkpoint exists. NOT yet run with a real checkpoint: the layout below is read from that code and must be
re-checked against the stage the checkpoint comes from (the track's adapter card).

Actor input, in order (env.observe):
  1. proprioception history, 5 steps, oldest first; each step = base angular velocity in the body frame (3), projected
     gravity (3), joint position minus default (NJ), joint velocity (NJ), previous action (NJ); NJ = 31, MJCF joint order
  2. height scan, 9 x 7 points at 0.1 m in the heading frame, x from -0.2 m, y centred, x-major: terrain height minus
     root height, clipped to +-2 m (floor raster only, as in training: walls are not in the scan)
  3. nine contact targets (feet, knees, hands, elbows, pelvis), each a 3-vector in the body-yaw frame, zero when none
  4. goal command: direction to the goal in the heading frame, d / max(|d|, r), and the heading still to turn (rad)
Action: joint targets = default pose + action_scale * clip(action, +-20); 50 Hz; PD at the plant gains (kp, kd from the
export file). The policy mean is used (no sampling). Observation noise, delay and calibration offsets are training-only.

Stage 1 trains with contact targets read from reference clips; on a route there is no reference, so the targets must come
from the planner (stage 2 onward). Without a planner this adapter feeds zeros ("no contact planned"), which the stage-1
tracker never saw as a steady input: such a run is a smoke test of the plumbing, never a result.
"""
from __future__ import annotations

import math

import numpy as np

from .base import Adapter, Command, View
from .common import History, NpMLP, Normaliser, bounded_goal, heading_grid, heading_of, to_world

HIST = 5
SCAN = heading_grid(9, 7, -0.2)           # tracker scan, 0.8 x 0.6 m
PLAN_SCAN = heading_grid(17, 11, -0.8)    # planner scan, 1.6 x 1.0 m


class ACPAdapter(Adapter):
    name = "acp"
    control_dt = 0.02
    vmax = 0.8                           # not used by the policy (it is goal-driven); kept for the record
    goal_r = 0.3                         # stage 1: 0.3 m; stage 3 goal terrains: 0.5 m (set from the adapter card)
    observes = ("proprioception history (5 x 50 Hz), 9 x 7 height scan of the floor (0.1 m), goal direction and heading; "
                "contact targets from the planner (zeros if no planner is loaded)")

    def __init__(self):
        self.z = None
        self.planner = None              # callable(view, cmd, hist) -> (9, 3) targets in the body-yaw frame, (9,) valid

    def load(self, path):
        """`path`: the .npz written by tools/export_actor.py (actor layers, normaliser, action scale, default pose, gains)."""
        z = self.z = np.load(path, allow_pickle=True)
        self.actor = NpMLP(z, "actor")
        self.norm = Normaliser(z["norm_mean"], z["norm_var"])
        self.nj = int(self.actor.out_dim)
        assert self.actor.in_dim == HIST * (6 + 3 * self.nj) + 63 + 27 + 3, "actor input %d does not match the stage-1 layout" % self.actor.in_dim
        self.scale, self.q0 = np.asarray(z["action_scale"], float), np.asarray(z["default_q"], float)
        self.joints = [str(n) for n in z["joint_names"]]
        if "goal_r" in z:
            self.goal_r = float(z["goal_r"])
        self.checkpoint = "%s (%s)" % (str(z["source"]) if "source" in z else path, "with planner" if self.planner else "NO planner: contact targets are zeros")

    def plant(self, joint_names):
        assert self.joints == list(joint_names), "the export's joint order differs from the eval model's"
        return dict(kp=np.asarray(self.z["kp"], float), kd=np.asarray(self.z["kd"], float), default_q=self.q0)

    def _proprio(self, view: View):
        _, _, R = view.body_pose("pelvis")
        _, w_body = view.body_velocity("pelvis")
        return np.concatenate((w_body, -R[2, :], view.q() - self.q0, view.qd(), self.a_prev))

    def reset(self, view: View, rng):
        self.a_prev = np.zeros(self.nj)
        self.hist = History(HIST)
        self.hist.reset(self._proprio(view))

    def act(self, view: View, cmd: Command) -> np.ndarray:
        p, _, R = view.body_pose("pelvis")
        yaw = heading_of(R)
        self.hist.push(self._proprio(view))
        scan = np.clip(view.height_scan(to_world(SCAN, p[:2], yaw)) - p[2], -2.0, 2.0)
        tgt = np.zeros((9, 3))
        if self.planner is not None:
            t, ok = self.planner(view, cmd, self.hist)
            tgt = np.asarray(t, float) * np.asarray(ok, float)[:, None]
        want = cmd.target_heading if (cmd.hold and cmd.target_heading is not None) else math.atan2(cmd.target_xy[1] - p[1], cmd.target_xy[0] - p[0])
        dpsi = (want - yaw + math.pi) % (2 * math.pi) - math.pi
        goal = np.concatenate((bounded_goal(cmd.target_xy, p[:2], yaw, self.goal_r), [dpsi if cmd.dist > 0.05 or cmd.hold else 0.0]))
        obs = np.concatenate((self.hist.flat(), scan, tgt.ravel(), goal))
        a = np.clip(self.actor(self.norm(obs)), -20.0, 20.0)
        self.a_prev = a
        return self.q0 + self.scale * a
