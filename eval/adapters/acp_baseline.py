"""The 2026-11-02 baseline policy pair as an eval adapter: the stage-2 foot planner (planner_v2_hist1) feeding the
SuperMimic tracker acp3_v2 @ 1000, as `gwbc-v6/scripts/acp/eval/pico_rollout_planner.py` ran them (13 of 50 on our
apparatus under the old rule). For re-measuring the baseline only; this round's tracks do not use these checkpoints.

Tracker input (1817), from gear_sonic/supermimic (sim.policy_input, mdp.obs, networks.Policy):
  goal block 680     zeros (the policy is reference-free; its encoder output is then a constant)
  proprioception 990 10 steps, oldest first, term-major: body angular velocity (3 x 10), joint position minus default
                     (31 x 10), joint velocity (31 x 10), previous action (31 x 10), projected gravity (3 x 10); joints in
                     the deploy (IsaacLab) order
  scan 63            9 x 7 at 0.1 m in the heading frame from 0.2 m behind, floor height minus root height
  contacts 54        nine bodies x (target xyz * valid, time / 1.5 s or 1, valid, in contact); the planner fills the feet
  goal command 30    direction to the waypoint in the heading frame, d / max(|d|, 1 m) capped at 0.5; cos and sin of the
                     heading still to turn (the station's facing when the waypoint is a station); 26 zeros
Each block is normalised with its own running mean / variance (eps 0.01). Output: 31 actions in the deploy order, mapped
to MuJoCo order, clipped to +-20; joint target = default pose + action scale * action; 50 Hz; PD at the plant's gains.
Checked against the original code with tools/baseline_check.py (see agentic/state/eval.md for the numbers).
"""
from __future__ import annotations

import math
import os

import numpy as np

from .acp import HORIZON, FootMlpPlanner
from .base import Adapter, Command, View
from .common import bounded_goal, elu, heading_grid, heading_of, to_world

HERE = os.path.dirname(os.path.abspath(__file__))
EXPORT = os.environ.get("ACP_BASELINE_DIR", os.path.join(HERE, "..", "out", "baseline_export"))
SCAN = heading_grid(9, 7, -0.2)
HIST, NJ = 10, 31


class ACPBaseline(Adapter):
    name = "acp-baseline-2026-11-02"
    plant_name = "vendor_20260823"       # the plant acp3_v2 was trained with (its checkpoint cfg); the export carries its gains
    control_dt = 0.02
    goal_mag = 0.5
    observes = ("proprioception history (10 x 50 Hz), 9 x 7 height scan of the floor, goal direction and heading; foot contact "
                "targets from the stage-2 planner (17 x 11 scan, goal, current state)")

    def __init__(self):
        self.planner = FootMlpPlanner(os.path.join(EXPORT, "planner_v2_hist1.npz"), goal_mag=self.goal_mag, floor=0.0)
        # The original switched to the next waypoint at 0.5 m, so on the way to a waypoint that was not a station the goal
        # command always had its full length (0.5); only at stations did it shrink. With the eval's 0.20 m arrival the
        # distance drops below 0.5 at every waypoint, a command the pair never met there. Both were run on our apparatus (200 trials each):
        # shrinking (default) 15 successes, full length except at stations (ACP_GOAL=keep) 4.
        self.keep_mag = os.environ.get("ACP_GOAL", "shrink") == "keep"    # default shrink: measured better on our apparatus (15 vs 4 of 200)
        self.planner.keep_mag = self.keep_mag
        self.load(os.path.join(EXPORT, "acp3_v2_1000.npz"))
        self.debug = {}

    def load(self, path):
        z = self.z = np.load(path, allow_pickle=True)
        # float32, as the original network: half the time of float64 on one core, differences of 1e-6
        self.enc = [(np.asarray(z["enc_w%d" % k], np.float32), np.asarray(z["enc_b%d" % k], np.float32)) for k in range(5)]
        self.dec = [(np.asarray(z["dec_w%d" % k], np.float32), np.asarray(z["dec_b%d" % k], np.float32)) for k in range(5)]
        eps = float(z["norm_eps"])
        self.norm = {k: (np.asarray(z[k + "_mean"], np.float64), np.sqrt(np.asarray(z[k + "_var"], np.float64) + eps ** 2)) for k in ("goal", "prop", "scan", "contact", "ref")}
        self.il = np.asarray(z["il_to_mj"], int)                       # x_il = x_mj[il]
        self.mj_to_il = np.array([list(self.il).index(i) for i in range(NJ)])   # a_mj = a_il[mj_to_il]
        self.q0, self.scale = np.asarray(z["default_q"], float), np.asarray(z["action_scale"], float)
        self.joints = [str(n) for n in z["joint_names"]]
        g = (np.zeros(680) - self.norm["goal"][0]) / self.norm["goal"][1]
        self.zc = np.tanh(self._mlp(self.enc, g))                      # the encoder sees zeros: a constant
        self.checkpoint = "%s + %s; goal length %s" % (str(z["source"]), self.planner.name, "kept at 0.5 except at stations" if getattr(self, "keep_mag", True) else "shrinks with the distance")

    @staticmethod
    def _mlp(layers, x):
        x = np.asarray(x, np.float32)
        for k, (w, b) in enumerate(layers):
            x = w @ x + b
            if k < len(layers) - 1:
                x = elu(x)
        return x

    def plant(self, joint_names):
        assert self.joints == list(joint_names), "joint order of the export differs from the eval model"
        z = self.z
        return dict(kp=np.asarray(z["kp"], float), kd=np.asarray(z["kd"], float), effort=np.asarray(z["effort"], float),
                    armature=np.asarray(z["armature"], float), default_q=self.q0)

    def start_q(self, joint_names):
        """The pose the original runs started in (the first frame of the route reference), instead of the default pose."""
        return np.asarray(self.z["start_q"], float)

    def _frame(self, view: View):
        _, _, R = view.body_pose("pelvis")
        w_body = view.d.qvel[3:6].copy()                              # free joint: angular velocity in the body frame
        return np.concatenate((w_body, (view.q() - self.q0)[self.il], view.qd()[self.il], self.a_prev[self.il], -R[2, :]))

    def reset(self, view: View, rng):
        self.a_prev = np.zeros(NJ)
        self.hist = np.tile(self._frame(view), (HIST, 1))
        self.fresh = True
        self.planner.reset(view, rng)

    def act(self, view: View, cmd: Command) -> np.ndarray:
        if not self.fresh:
            self.hist = np.concatenate((self.hist[1:], self._frame(view)[None]), 0)
        self.fresh = False
        p, _, R = view.body_pose("pelvis")
        yaw = heading_of(R)
        prop = np.concatenate([self.hist[:, a:b].ravel() for a, b in ((0, 3), (3, 34), (34, 65), (65, 96), (96, 99))])
        scan = np.maximum(view.height_scan(to_world(SCAN, p[:2], yaw)), 0.0) - p[2]      # floor 0, as the original terrain code
        cs = self.planner(view, cmd)
        ok = cs["valid"].astype(float)
        con = np.concatenate((cs["target"] * ok[:, None], np.where(ok > 0, cs["dt"] / HORIZON, 1.0)[:, None], ok[:, None], cs["contact"].astype(float)[:, None]), 1).ravel()
        d = bounded_goal(cmd.target_xy, p[:2], yaw, 1.0)
        n = float(np.linalg.norm(d))
        if n > self.goal_mag or (self.keep_mag and not cmd.station and n > 1e-6):
            d = d * self.goal_mag / n
        a = math.atan2(d[1], d[0])
        if cmd.target_heading is not None:
            a = math.atan2(math.sin(cmd.target_heading - yaw), math.cos(cmd.target_heading - yaw))
        ref = np.concatenate(([d[0], d[1], math.cos(a), math.sin(a)], np.zeros(26)))
        nz = lambda k, x: (x - self.norm[k][0]) / self.norm[k][1]  # noqa: E731
        h = np.concatenate((self.zc, nz("prop", prop), nz("scan", scan), nz("contact", con), nz("ref", ref)))
        act = np.clip(self._mlp(self.dec, h)[self.mj_to_il].astype(np.float64), -20.0, 20.0)
        self.a_prev = act
        self.debug = dict(x=np.concatenate((np.zeros(680), prop, scan, con, ref)), contacts=con, action=act)
        return self.q0 + self.scale * act
