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

ACP is a pair: a contact planner (where each body touches next) and a tracker that executes those contacts. The tracker
always needs contact targets; only their SOURCE changes: reference clips in stage-1 training, the planner on a route (as
in the 2026-11-02 baseline, 13 of 50 on our apparatus: planner_v2_hist1 + tracker acp3_v2). So the adapter takes a
pluggable `ContactSource`, called every control step:
  NoContacts        zeros ("nothing planned"): a plumbing test only, never a result
  FootMlpPlanner    the earlier pipeline's stage-2 MLP planner for the two feet (gwbc-v6 scripts/acp/stage2/planner.py and
                    eval/pico_rollout_planner.py), from its weights exported to npz, with that eval's decode options. It was
                    trained on SuperMimic-tracker rollouts, so it serves the baseline re-run, not this round's ACP policies.
  (to come)         the track's attention planner for nine bodies, as ONNX, when the track exports it; or an interim
                    non-learned source if the ACP agent provides one (same interface)
"""
from __future__ import annotations

import math

import numpy as np

from .base import Adapter, Command, View
from .common import History, NpMLP, Normaliser, bounded_goal, heading_grid, heading_of, to_world

HIST = 5
SCAN = heading_grid(9, 7, -0.2)           # tracker scan, 0.8 x 0.6 m
PLAN_SCAN = heading_grid(17, 11, -0.8)    # this round's planner / critic scan, 1.6 x 1.0 m
NINE = ("left foot", "right foot", "left knee", "right knee", "left hand", "right hand", "left elbow", "right elbow", "pelvis")
HORIZON = 1.5                             # s: how far ahead a contact is planned


class ContactSource:
    """Where the tracker's contact targets come from. Called once per control step, before the tracker.

    Returns a dict for the nine bodies (order NINE): `target` (9, 3) next contact point in the body-yaw frame (x forward,
    y left, z up, relative to the root), `valid` (9,) a contact is coming within HORIZON, `dt` (9,) seconds until it,
    `contact` (9,) in contact now. Bodies a source does not plan stay invalid."""
    name = "none"

    def reset(self, view: View, rng):
        pass

    def __call__(self, view: View, cmd: Command) -> dict:
        return dict(target=np.zeros((9, 3)), valid=np.zeros(9, bool), dt=np.full(9, HORIZON), contact=np.zeros(9, bool))


class NoContacts(ContactSource):
    name = "no planner (zeros)"


class FootMlpPlanner(ContactSource):
    """The 2026-11-02 baseline's foot planner. Input (204): 17 x 11 scan (x -0.3 .. 1.3, y -0.5 .. 0.5, 0.1 m, heading
    frame, height minus root height), goal (direction in the heading frame, d / max(|d|, 1 m), capped at `goal_mag`;
    cos and sin of the heading still to turn), and 13 numbers of the current state (ankles xy in the heading frame,
    ankles z minus root z, root height over the ground, velocity in the heading frame, the root's z axis). Output per
    foot: weights over the 187 scan points, occupancy, time to contact, contact now.
    Decode as that eval ran it: `mode` (the strongest point, refined by the weights within 0.11 m), `step_mask` (no
    candidate whose surroundings, +-6 cm, step by more than this), `latch` (a target is frozen in the world once its
    touchdown is within this many seconds). NOT yet checked against the original code's outputs."""

    def __init__(self, npz, decode="mode", step_mask=0.03, latch=0.3, goal_mag=0.5, dt=0.02, floor=None):
        self.floor = floor               # the original terrain code never returns less than its floor (0): the ground below
        #                                  the deck around our apparatus (-0.108 m in the raster) read as 0 there
        z = np.load(npz, allow_pickle=True)
        self.name = "foot MLP planner: %s (decode %s, step mask %.2f, latch %.1f s)" % (str(z["source"]) if "source" in z else npz, decode, step_mask, latch)
        self.w = [(np.asarray(z["body_0_weight"], float), np.asarray(z["body_0_bias"], float)), (np.asarray(z["body_2_weight"], float), np.asarray(z["body_2_bias"], float))]
        self.ww, self.wb = np.asarray(z["w_weight"], float), np.asarray(z["w_bias"], float)
        self.hw, self.hb = np.asarray(z["head_weight"], float), np.asarray(z["head_bias"], float)
        self.mu, self.sd, self.pts = np.asarray(z["mu"], float), np.asarray(z["sd"], float), np.asarray(z["pts"], float)
        self.hd = int(z["hist_dim"])
        assert self.hd == 13, "only the current-step planner (--hist1) is ported; this file has a %d-number history" % self.hd
        self.decode, self.mask, self.latch, self.goal_mag, self.dt = decode, step_mask, latch, goal_mag, dt
        self.ankles = ("left_ankle_roll_link", "right_ankle_roll_link")

    def reset(self, view, rng):
        self.lt, self.lrem = np.zeros((2, 3)), np.zeros(2)

    def __call__(self, view: View, cmd: Command) -> dict:
        from .common import elu
        p, _, R = view.body_pose("pelvis")
        yaw = heading_of(R)
        c, s_ = math.cos(yaw), math.sin(yaw)
        W = to_world(self.pts, p[:2], yaw)
        hs = (lambda q: np.maximum(view.height_scan(q), self.floor)) if self.floor is not None else view.height_scan
        scan = hs(W) - p[2]
        d = bounded_goal(cmd.target_xy, p[:2], yaw, 1.0)
        n = float(np.linalg.norm(d))
        if self.goal_mag > 0 and n > self.goal_mag:
            d = d * self.goal_mag / n
        a = math.atan2(d[1], d[0])
        if cmd.target_heading is not None:
            a = math.atan2(math.sin(cmd.target_heading - yaw), math.cos(cmd.target_heading - yaw))
        goal = np.array([d[0], d[1], math.cos(a), math.sin(a)])
        A = np.array([view.body_pose(b)[0] for b in self.ankles])
        rel = A[:, :2] - p[:2]
        hx, hy = c * rel[:, 0] + s_ * rel[:, 1], -s_ * rel[:, 0] + c * rel[:, 1]
        v = view.d.qvel[:3]
        feat = np.concatenate(([hx[0], hy[0], hx[1], hy[1]], A[:, 2] - p[2], [p[2] - float(hs(p[None, :2])[0])],
                               [c * v[0] + s_ * v[1], -s_ * v[0] + c * v[1], v[2]], R[2, :]))
        h = (np.concatenate((scan, goal, feat)) - self.mu) / self.sd
        for w, b in self.w:
            h = elu(w @ h + b)
        logits = (self.ww @ h + self.wb).reshape(2, -1)
        o = (self.hw @ h + self.hb).reshape(2, 3)
        sig = lambda x: 1.0 / (1.0 + np.exp(-x))  # noqa: E731
        if self.mask > 0:                                   # foothold scoring on the scanned terrain
            off = np.array([[0, 0], [0.06, 0], [-0.06, 0], [0, 0.06], [0, -0.06]])
            hh = np.stack([hs(W + np.array([c * o_[0] - s_ * o_[1], s_ * o_[0] + c * o_[1]])) for o_ in off], 1)
            ok = (hh.max(1) - hh.min(1)) <= self.mask
            if ok.any():
                logits = np.where(ok[None], logits, -1e9)
        w = np.exp(logits - logits.max(1, keepdims=True))
        w /= w.sum(1, keepdims=True)
        if self.decode == "mode":
            best = self.pts[w.argmax(1)]
            near = np.linalg.norm(self.pts[None] - best[:, None], axis=-1) <= 0.11
            w = w * near
            w /= np.maximum(w.sum(1, keepdims=True), 1e-9)
        tgt = np.concatenate((w @ self.pts, (w * scan[None]).sum(1, keepdims=True)), 1)       # (2, 3) heading frame, z rel root
        po, pdt, pc = sig(o[:, 0]), sig(o[:, 1]) * HORIZON, sig(o[:, 2])
        if self.latch > 0:                                  # commit to a foothold shortly before touchdown
            wt = np.stack((p[0] + c * tgt[:, 0] - s_ * tgt[:, 1], p[1] + s_ * tgt[:, 0] + c * tgt[:, 1], p[2] + tgt[:, 2]), 1)
            new = (po > 0.5) & (pdt < self.latch) & (self.lrem <= 0)
            self.lt = np.where(new[:, None], wt, self.lt)
            self.lrem = np.where(new, pdt + 0.15, self.lrem)
            on = self.lrem > 0
            rx, ry = self.lt[:, 0] - p[0], self.lt[:, 1] - p[1]
            lt = np.stack((c * rx + s_ * ry, -s_ * rx + c * ry, self.lt[:, 2] - p[2]), 1)
            tgt = np.where(on[:, None], lt, tgt)
            pdt, po = np.where(on, np.maximum(self.lrem, 0.0), pdt), np.where(on, 1.0, po)
            self.lrem = self.lrem - self.dt
        out = ContactSource.__call__(self, view, cmd)
        out["target"][:2], out["valid"][:2], out["dt"][:2], out["contact"][:2] = tgt, po > 0.5, pdt, pc > 0.5
        return out


class ACPAdapter(Adapter):
    name = "acp"
    control_dt = 0.02
    vmax = 0.8                           # not used by the policy (it is goal-driven); kept for the record
    goal_r = 0.3                         # stage 1: 0.3 m; stage 3 goal terrains: 0.5 m (set from the adapter card)
    observes = ("proprioception history (5 x 50 Hz), 9 x 7 height scan of the floor (0.1 m), goal direction and heading; "
                "contact targets from the planner (zeros if no planner is loaded)")

    contact_feat = 3                     # per body in the tracker input: 3 = target * valid (this round's PPO tracker);
    #                                      6 = target * valid, dt / 1.5 (1 if none), valid, in contact (the earlier pipeline)

    def __init__(self, contacts: ContactSource | None = None):
        self.z = None
        self.contacts = contacts or NoContacts()
        self.plan_log = []               # per step: the planned foot targets in the world (for renders and checks)

    def load(self, path):
        """`path`: the .npz written by tools/export_actor.py (actor layers, normaliser, action scale, default pose, gains)."""
        z = self.z = np.load(path, allow_pickle=True)
        self.actor = NpMLP(z, "actor")
        self.norm = Normaliser(z["norm_mean"], z["norm_var"])
        self.nj = int(self.actor.out_dim)
        assert self.actor.in_dim == HIST * (6 + 3 * self.nj) + 63 + 9 * self.contact_feat + 3, "actor input %d does not match the stage-1 layout" % self.actor.in_dim
        self.scale, self.q0 = np.asarray(z["action_scale"], float), np.asarray(z["default_q"], float)
        self.joints = [str(n) for n in z["joint_names"]]
        if "goal_r" in z:
            self.goal_r = float(z["goal_r"])
        self.checkpoint = "%s | contacts from: %s" % (str(z["source"]) if "source" in z else path, self.contacts.name)
        self.observes = self.observes.replace("contact targets from the planner (zeros if no planner is loaded)", "contact targets from: " + self.contacts.name)

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
        self.contacts.reset(view, rng)
        self.plan_log = []

    def act(self, view: View, cmd: Command) -> np.ndarray:
        p, _, R = view.body_pose("pelvis")
        yaw = heading_of(R)
        self.hist.push(self._proprio(view))
        scan = np.clip(view.height_scan(to_world(SCAN, p[:2], yaw)) - p[2], -2.0, 2.0)
        cs = self.contacts(view, cmd)
        ok = np.asarray(cs["valid"], float)
        tgt = np.asarray(cs["target"], float) * ok[:, None]
        if self.contact_feat == 6:
            tgt = np.concatenate((tgt, np.where(ok > 0, np.asarray(cs["dt"], float) / HORIZON, 1.0)[:, None], ok[:, None],
                                  np.asarray(cs["contact"], float)[:, None]), 1)
        c_, s_ = math.cos(yaw), math.sin(yaw)
        self.plan_log.append([(p[0] + c_ * t[0] - s_ * t[1], p[1] + s_ * t[0] + c_ * t[1], p[2] + t[2], bool(v)) for t, v in zip(cs["target"][:2], cs["valid"][:2])])
        want = cmd.target_heading if (cmd.hold and cmd.target_heading is not None) else math.atan2(cmd.target_xy[1] - p[1], cmd.target_xy[0] - p[0])
        dpsi = (want - yaw + math.pi) % (2 * math.pi) - math.pi
        goal = np.concatenate((bounded_goal(cmd.target_xy, p[:2], yaw, self.goal_r), [dpsi if cmd.dist > 0.05 or cmd.hold else 0.0]))
        obs = np.concatenate((self.hist.flat(), scan, tgt.ravel(), goal))
        a = np.clip(self.actor(self.norm(obs)), -20.0, 20.0)
        self.a_prev = a
        return self.q0 + self.scale * a


class ACPBaselinePlannerTest(ACPAdapter):
    """Plumbing test: this round's tracker layout (random or real weights) with the baseline foot planner as the contact
    source. `--checkpoint tracker.npz` and the planner npz in the environment variable ACP_PLANNER_NPZ."""

    def __init__(self):
        import os
        super().__init__(FootMlpPlanner(os.environ["ACP_PLANNER_NPZ"]))
