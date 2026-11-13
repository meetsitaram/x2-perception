"""The common MuJoCo eval (CPU): one X2, one attempt at an apparatus route, scored as docs/EVAL.md says.

    python eval/harness.py --adapter zero --apparatus ours --trials 3 --out eval/out/zero_ours
    python eval/harness.py --adapter stand --apparatus ours --trials 1 --seconds 10 --out ...     (standing check)
    python eval/harness.py --kinematic --apparatus nist --out ...      (no physics: the pelvis is carried along the route to
                                                                         check waypoints, sections and cameras; never a result)

Output folder: results.json (every trial), summary.json (the table row), traj/<seed>.npz for the trials to render
(render.py). All numbers in metres, seconds, newtons. Constants below are the eval definition; change them only together
with docs/EVAL.md.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import pathlib
import sys
import time

import mujoco
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import x2 as X  # noqa: E402
from adapters.base import Adapter, Command, View  # noqa: E402
from apparatus import Apparatus  # noqa: E402

# ---- the eval definition (docs/EVAL.md) ---------------------------------------------------------------------------
RADIUS = 0.20            # m: a waypoint is reached when the pelvis is this close in the horizontal plane ...
Z_TOL = 0.30             # m: ... and within this of standing height above the waypoint's ground
END_STAND = 1.0          # s: after the last waypoint the robot must stay up this long
# Strict stand (coordinator's decision 2026-11-12, implementing the user's "a station is a 1 s stand"; to be confirmed by the
# user; docs/EVAL.md "Strict stand"). For the whole 1 s of a station's hold and of the end stand: the pelvis inside the disc,
# the two feet together carrying at least STRICT_LOAD of the robot's weight (so never both at 0 N) and the pelvis' horizontal
# speed under STRICT_SPEED; and the robot still upright STRICT_POST_END s after the end stand. Checked on the same trajectory
# as the disc-only rule (waypoints advance by the disc-only rule, so the disc-only count is unchanged and comparable; a station
# held loosely is not retried). The end stand is checked this way only when the last waypoint is itself a station.
STRICT_LOAD = 0.5        # share of the robot's weight on the feet (sum of both feet's vertical loads)
STRICT_SPEED = 0.15      # m/s: horizontal pelvis speed
STRICT_POST_END = 2.0    # s after the end stand: still upright (the trial runs on this long with a stand command)
STAND_RULE = os.environ.get("STAND_RULE", "strict")   # "strict": a station's 1 s counts only while standing (main, from
#                          2026-11-12); "disc": the old rule (1 s in the disc), run beside it for comparison with earlier results;
#                          in a "disc" run the strict conditions are still checked on the window that counted (strict_success)
END_RETRY = os.environ.get("END_RETRY", "1") == "1"   # coordinator 2026-11-12 11:50: in a strict run the end stand, like a
#                          station, counts once a continuous 1 s strict stand is reached at the last station (a correction
#                          restarts the 1 s; limit: the time limit, or 45 s after the last waypoint), then 2 s upright.
#                          Default on since 2026-11-12 12:06 UTC (END_RETRY=0 gives the earlier end stand).
TRAVEL_AFTER = 1.0       # s: "travel after a station": the pelvis path length in this time after the station counted
TIME_LIMIT = 180.0       # s per trial (recommended value; the user has not confirmed it)
STALL = 45.0             # s without reaching a new waypoint (and not holding at a station): the trial ends as "stalled"
FALL_H = 0.35            # m: fall when the pelvis is lower than this above the lowest ground under the pelvis and feet
FALL_TILT = 60.0         # deg: fall when the pelvis leans further than this from upright
OFF_COURSE = 1.0         # m: failure when the pelvis is further than this from the route (or off the raster)
F_MIN = 5.0              # N: contacts weaker than this are ignored
SIDE_NZ = 0.5            # a contact is "horizontal-dominant" when |normal z| is below this (surface steeper than 60 deg)
SIDE_RELIEF = 0.03       # m: ... and the raster within 4 cm of the contact point spans more than this (a face is there)
# side hits (docs/EVAL.md, "Side hits and the stumble penalty")
HIT_F, HIT_J = 20.0, 0.5 # N, N s: a side contact is a hit when its peak horizontal force or its horizontal impulse reaches these;
SCUFF_P = 0.02           #         below both it is a scuff, which costs this much and does not spoil a clean success
HIT_BASE, HIT_J_W, HIT_J_REF = 0.1, 0.4, 10.0      # penalty of a hit: base + J_W * min(1, impulse / J_REF) ...
HIT_D_W, HIT_DV_REF, HIT_TILT_REF = 0.5, 0.30, 10.0  # ... + D_W * min(1, max(pelvis speed change / DV_REF, tilt change / TILT_REF))
HIT_WINDOW = 0.3         # s after the contact starts: the impact. Peak force, impulse and pelvis disturbance are taken in it
BEAM_NEAR = 0.12         # m: a face this close to a 4x4's centre line is the 4x4's
CLASSES = ("4x4", "wedge", "stair")
TRAP_DEPTH, TRAP_MIN = 0.01, 0.1      # m, s: a foot "trapped" = a contact deeper than this held at least this long (audit class a/b)
CLAMP_DEPTH = 0.005                    # m: a head / trunk / limb contact deeper than this in a wall box, held TRAP_MIN, is a clamp
JLIM_EXCESS = 0.15                     # rad: a joint this far past its limit (MuJoCo's limits are soft); 0.05 until 2026-11-12 21:10 UTC (fired in ~every trial)
ANKLE_ROLL_EXCESS = 0.05               # rad: ankle roll (+-0.262 rad range) past its limit, reported as its own time (coordinator 2026-11-12)
MEASURE = "v2 (2026-11-12): loads, peaks and impulses from the summed force vector per foot / per contact episode; wall boxes count in the load"
EVENT_GAP = 0.2          # s: contacts of the same kind on the same side closer than this are one event
STANCE_ON, STANCE_OFF = 40.0, 10.0   # N: a foot lands when its vertical load rises above ON after being below OFF
EDGE_RANGE = 0.03        # m: an edge landing: the ground under the sole departs this much from a plane (peak to peak)
SEED0 = 1000             # trial i uses seed SEED0 + i, for every track
RANDOM = dict(           # the "randomised" protocol (the headline number); "nominal" keeps only the small start jitter
    start_xy=0.10, start_yaw_deg=15.0, friction=(0.5, 1.0), mass_scale=(0.95, 1.05), torso_payload=(0.0, 3.0),
    push_every=(6.0, 10.0), push_dv=(0.2, 0.5))
NOMINAL = dict(start_xy=0.03, start_yaw_deg=3.0, friction=(1.0, 1.0), mass_scale=(1.0, 1.0), torso_payload=(0.0, 0.0),
               push_every=None, push_dv=(0.0, 0.0))


def _seg_cross(p, q, a, b):
    """Distance along a-b (m) at which segment p-q crosses it, or None."""
    p, q, a, b = (np.asarray(v, float) for v in (p, q, a, b))
    r, s = q - p, b - a
    den = r[0] * s[1] - r[1] * s[0]
    if abs(den) < 1e-12:
        return None
    t = ((a[0] - p[0]) * s[1] - (a[1] - p[1]) * s[0]) / den
    u = ((a[0] - p[0]) * r[1] - (a[1] - p[1]) * r[0]) / den
    return float(u * np.linalg.norm(s)) if 0 <= t <= 1 and 0 <= u <= 1 else None


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, c - h), min(1.0, c + h))


def load_adapter(name: str, checkpoint=None) -> Adapter:
    """`name` is a key of ADAPTERS in adapters/stand.py or `module:Class` under adapters/ (e.g. ssr:SSRAdapter)."""
    if ":" in name:
        mod, cls = name.split(":")
        a = getattr(importlib.import_module("adapters." + mod), cls)()
    else:
        from adapters.stand import ADAPTERS
        a = ADAPTERS[name]()
    if checkpoint:
        a.checkpoint = checkpoint
        if hasattr(a, "load"):
            a.load(checkpoint)
    return a


class Trial:
    def __init__(self, x: X.X2, app: Apparatus | None, adapter: Adapter, nominal_mass, nominal_inertia):
        self.x, self.m, self.app, self.adapter = x, x.model, app, adapter
        self.d = mujoco.MjData(self.m)
        self.view = View(x, self.d, app)
        self.pelvis = self.m.body("pelvis").id
        self.torso = self.m.body("torso_link").id
        self.mass0, self.inertia0 = nominal_mass, nominal_inertia
        self.sub = max(1, int(round(adapter.control_dt / X.SIM_DT)))
        self.path = app.path() if app is not None else np.zeros((1, 2))
        self.beams = app.route.get("beams", []) if app is not None else []      # 4x4 stepovers: every one must be crossed
        self.zones = app.route.get("zones", []) if app is not None else []
        self.beam_lines = app.route.get("beam_lines", self.beams) if app is not None else []   # every 4x4, for the side-hit class
        self.wps = app.waypoints if app is not None else []
        self._f6 = np.zeros(6)
        sx, sy = np.meshgrid(np.linspace(-1, 1, 7) * X.SOLE_HALF[0] + X.SOLE_X, np.linspace(-1, 1, 3) * X.SOLE_HALF[1], indexing="ij")
        self.sole_pts = np.stack([sx.ravel(), sy.ravel(), np.full(sx.size, -X.SOLE_Z)], 1)

    # ---- set-up -----------------------------------------------------------------------------------------------------
    def reset(self, seed: int, protocol: dict):
        m, d, x = self.m, self.d, self.x
        rng = np.random.default_rng(seed)
        r = {}
        r["dx"], r["dy"] = (float(v) for v in rng.uniform(-protocol["start_xy"], protocol["start_xy"], 2))
        r["dyaw_deg"] = float(rng.uniform(-protocol["start_yaw_deg"], protocol["start_yaw_deg"]))
        r["friction"] = float(rng.uniform(*protocol["friction"]))
        r["mass_scale"] = float(rng.uniform(*protocol["mass_scale"]))
        r["torso_payload"] = float(rng.uniform(*protocol["torso_payload"]))
        pushes, t = [], 0.0
        while protocol["push_every"] is not None and t < TIME_LIMIT:
            t += float(rng.uniform(*protocol["push_every"]))
            a, dv = float(rng.uniform(0, 2 * math.pi)), float(rng.uniform(*protocol["push_dv"]))
            pushes.append((t, dv * math.cos(a), dv * math.sin(a)))
        self.pushes, self.rand = pushes, r
        m.geom_friction[:, 0] = r["friction"]              # MuJoCo takes the larger of the two geoms', so all get it
        m.body_mass[:] = self.mass0 * r["mass_scale"]
        m.body_inertia[:] = self.inertia0 * r["mass_scale"]
        k = (m.body_mass[self.torso] + r["torso_payload"]) / m.body_mass[self.torso]
        m.body_mass[self.torso] *= k
        m.body_inertia[self.torso] *= k
        mujoco.mj_setConst(m, d)
        mujoco.mj_resetData(m, d)
        sx, sy, syaw = self.app.spawn if self.app is not None else (0.0, 0.0, 0.0)
        yaw = math.radians(syaw + r["dyaw_deg"])
        d.qpos[:3] = [sx + r["dx"], sy + r["dy"], 1.0]
        d.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        sq = getattr(self.adapter, "start_q", None)           # an adapter may ask for the pose its policy's runs start in
        d.qpos[x.qadr] = sq(x.joint_names) if sq else x.plant["default_q"]
        mujoco.mj_forward(m, d)
        gap = min(self._sole_gap(b) for b in x.foot_ids)   # drop until the lowest sole point is 2 mm above its ground
        d.qpos[2] -= gap - 0.002
        mujoco.mj_forward(m, d)
        self.h0 = float(d.qpos[2] - self.ground(d.qpos[0], d.qpos[1]))
        self.wp, self.hold_t, self.reach_t, self.last_progress = 0, 0.0, [], 0.0
        self.done_t = None
        self.disc_reason, self.strict_fail, self.stations, self.win, self.end_win = None, None, [], [], []
        self.end_breaks, self.end_ok_t = 0, None
        self.weight = float(m.body_mass.sum() * 9.81)
        self.push_done = []
        self.beam_cross = {b["name"]: [] for b in self.beams}      # times the pelvis path crossed each beam's centre line
        self.prev_xy = d.qpos[:2].copy()
        self.events, self.last_event = [], {}
        self.steps = {b: [] for b in x.foot_ids}
        self.stance = {b: True for b in x.foot_ids}
        self.load = {b: 0.0 for b in x.foot_ids}
        self.contact_steps = 0
        self.flag_runs = {}                                  # (kind, part) -> start time of the current deep-contact run
        self.flag_eps = []                                   # [kind, part, t0, t1, x, y]
        self.jlim_t, self.jlim_max, self.ankroll_t = [], 0.0, []
        lim = [j for j in range(m.njnt) if m.jnt_limited[j] and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
        self._jl = (np.array([m.jnt_qposadr[j] for j in lim], int), m.jnt_range[lim, 0].copy(), m.jnt_range[lim, 1].copy())
        self._ar = np.array([k for k, j in enumerate(lim) if "ankle_roll" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or "")], int)
        self.rec = {k: [] for k in ("t", "qpos", "wp", "cmd", "load", "h", "hold")}
        self.adapter.reset(self.view, rng)
        return r

    def ground(self, x, y):
        return float(self.app.height(x, y)) if self.app is not None else 0.0

    def _cls(self, x, y):
        """Which obstacle a side face at (x, y) belongs to: "4x4", "wedge", "stair" or "other" (door threshold, brace foot)."""
        for b in self.beam_lines:
            a, c = np.asarray(b["a"]), np.asarray(b["b"])
            t = np.clip((np.array([x, y]) - a) @ (c - a) / ((c - a) @ (c - a)), 0, 1)
            if np.linalg.norm(a + t * (c - a) - [x, y]) < BEAM_NEAR:
                return "4x4"
        for z in self.zones:
            r = z["rect"]
            if r[0] <= x <= r[2] and r[1] <= y <= r[3]:
                return z["cls"]
        return "other"

    def _relief(self, x, y):
        if self.app is None:
            return 0.0
        i, j = self.app.cell(x, y)
        g = self.app.grid[max(i - 2, 0):i + 3, max(j - 2, 0):j + 3]
        return float(g.max() - g.min())

    def _sole_world(self, b):
        return self.d.xpos[b] + self.sole_pts @ self.d.xmat[b].reshape(3, 3).T

    def _sole_gap(self, b):
        P = self._sole_world(b)
        g = self.app.height(P[:, 0], P[:, 1]) if self.app is not None else np.zeros(len(P))
        return float((P[:, 2] - g).min())

    # ---- the command, from the waypoints only -------------------------------------------------------------------------
    def command(self) -> Command:
        d, a = self.d, self.adapter
        pos, yaw = d.qpos[:3], X.yaw_of(d.qpos[3:7])
        if not self.wps:                                    # no route: stand where you are
            return Command(pos[:2].copy(), self.ground(pos[0], pos[1]), None, True, np.zeros(3), 0.0, 0.0, self.wp, len(self.wps), np.zeros((0, 2)))
        if self.wp >= len(self.wps):                        # finished: keep standing AT the last waypoint, facing its heading
            # (fixed 2026-11-12 09:30: before, the target followed the pelvis with no heading, so a goal-driven adapter got a
            # zero goal and dpsi = -yaw (towards world heading 0) during the end stand; every result before this time has that)
            w = self.wps[-1]
            tgt = np.array([w["x"], w["y"]])
            th = math.radians(w["heading_deg"]) if w["hold_s"] > 0 and w["heading_deg"] is not None else None
            err = X.wrap(th - yaw) if th is not None else 0.0
            vel = np.array([0.0, 0.0, float(np.clip(1.5 * err, -a.wmax, a.wmax)) if abs(err) > 0.15 else 0.0])
            return Command(tgt, self.ground(*tgt), th, True, vel, float(np.linalg.norm(tgt - pos[:2])), float(err), self.wp, len(self.wps),
                           np.zeros((0, 2)), bool(w["hold_s"] > 0))
        w = self.wps[self.wp]
        tgt = np.array([w["x"], w["y"]])
        vec = tgt - pos[:2]
        dist = float(np.linalg.norm(vec))
        th = math.radians(w["heading_deg"]) if w["hold_s"] > 0 and w["heading_deg"] is not None else None
        holding = w["hold_s"] > 0 and dist < RADIUS
        if w.get("face_deg") is not None and not holding:   # keep this heading while moving there (backward / sideways)
            th = math.radians(w["face_deg"])
            err = X.wrap(th - yaw)
            c_, s_ = math.cos(yaw), math.sin(yaw)
            u = vec / max(dist, 1e-9) * float(np.clip(dist, 0.15, 0.5 * a.vmax))
            vel = np.array([c_ * u[0] + s_ * u[1], -s_ * u[0] + c_ * u[1], float(np.clip(1.5 * err, -a.wmax, a.wmax))])
        elif holding:
            err = X.wrap(th - yaw) if th is not None else 0.0
            vel = np.array([0.0, 0.0, float(np.clip(1.5 * err, -a.wmax, a.wmax)) if abs(err) > 0.15 else 0.0])
        else:
            err = X.wrap(math.atan2(vec[1], vec[0]) - yaw)
            wz = float(np.clip(1.5 * err, -a.wmax, a.wmax))
            vx = 0.0 if abs(err) > math.radians(60) else float(np.clip(1.0 * dist, 0.15, a.vmax) * math.cos(err))
            vel = np.array([vx, 0.0, wz])
        ahead = np.array([[v["x"], v["y"]] for v in self.wps[self.wp + 1:self.wp + 4]]).reshape(-1, 2)
        return Command(tgt, self.ground(*tgt), th, holding, vel, dist, float(err), self.wp, len(self.wps), ahead, bool(w["hold_s"] > 0))

    def _beams(self):
        """The pelvis path of this control step against each 4x4's centre line, within the beam's length: going round
        an end is not a crossing."""
        cur = self.d.qpos[:2].copy()
        for b in self.beams:
            if _seg_cross(self.prev_xy, cur, b["a"], b["b"]) is not None:
                self.beam_cross[b["name"]].append(round(float(self.d.time), 2))
        self.prev_xy = cur

    def _progress(self, dt):
        d = self.d
        if self.wp >= len(self.wps):
            return
        w = self.wps[self.wp]
        inside = (math.hypot(w["x"] - d.qpos[0], w["y"] - d.qpos[1]) < RADIUS
                  and abs(d.qpos[2] - self.ground(w["x"], w["y"]) - self.h0) < Z_TOL)
        if inside and w.get("heading_tol_deg") is not None:      # a turn test: the station counts only when faced
            inside = abs(X.wrap(math.radians(w["heading_deg"]) - X.yaw_of(d.qpos[3:7]))) < math.radians(w["heading_tol_deg"])
        smp = self._stand_sample()
        if inside and STAND_RULE == "strict" and w["hold_s"] > 0 and (smp[0] < STRICT_LOAD or smp[1] >= STRICT_SPEED):
            inside = False                                     # strict rule: the 1 s counts only while standing
        self.hold_t = self.hold_t + dt if inside else 0.0
        self.win = self.win + [smp] if inside else []
        if inside and self.hold_t >= w["hold_s"]:
            self.reach_t.append(round(float(d.time), 2))
            if w["hold_s"] > 0:                                # strict stand over the hold window that made the station count
                k = max(1, int(round(w["hold_s"] / dt)))
                bad = self._stand_bad(self.win[-k:])
                if bad and self.strict_fail is None:
                    self.strict_fail = "station %s: %s" % (w["name"], bad)
                self.stations.append(dict(name=w["name"], t=round(float(d.time), 2), strict=not bad, why=bad, xy=d.qpos[:2].copy(), travel=0.0, end=False))
            self.wp += 1
            self.hold_t, self.last_progress, self.win = 0.0, float(d.time), []
            if self.wp == len(self.wps):
                self.done_t = float(d.time)
                if self.wps[-1]["hold_s"] <= 0:
                    self.stations.append(dict(name=w["name"] + " (end)", t=round(float(d.time), 2), strict=True, why=None, xy=d.qpos[:2].copy(), travel=0.0, end=True))

    def _stand_sample(self):
        """(load share of body weight, horizontal pelvis speed, pelvis inside the last waypoint's disc) at this control step."""
        d = self.d
        w = self.wps[min(self.wp, len(self.wps) - 1)]
        inside = math.hypot(w["x"] - d.qpos[0], w["y"] - d.qpos[1]) < RADIUS
        return (sum(self.load[b] for b in self.x.foot_ids) / self.weight, math.hypot(d.qvel[0], d.qvel[1]), inside)

    @staticmethod
    def _stand_bad(win):
        """Why a stand window is not a stand under the strict rule (None if it is)."""
        why = []
        lo = min(s[0] for s in win)
        if lo < STRICT_LOAD:
            why.append("feet carried %.0f %% of the weight at the least" % (100 * lo))
        hi = max(s[1] for s in win)
        if hi >= STRICT_SPEED:
            why.append("pelvis moved at up to %.2f m/s" % hi)
        if not all(s[2] for s in win):
            why.append("left the disc")
        return "; ".join(why) or None

    def _travel(self):
        """Path length of the pelvis after each station (the "does not stop" measure)."""
        d = self.d
        for s in self.stations:
            if d.time - s["t"] <= TRAVEL_AFTER + 1e-9:
                s["travel"] += float(np.linalg.norm(d.qpos[:2] - s["xy"]))
                s["xy"] = d.qpos[:2].copy()

    # ---- contacts, every physics step -------------------------------------------------------------------------------
    def _event(self, kind, side, t, force, pos, extra=None, fh=0.0):
        """One event per contact episode (same kind, side and obstacle class, gaps up to EVENT_GAP). fh: the contact's
        horizontal force (N): its peak and its time integral (impulse, N s) are kept."""
        key = (kind, side, (extra or {}).get("cls"))
        last = self.last_event.get(key)
        if last is not None and t - last["t1"] <= EVENT_GAP:
            last["t1"], last["force"] = t, max(last["force"], force)
            if t - last["t"] <= HIT_WINDOW:                 # the impact: force and impulse of the first HIT_WINDOW seconds only
                last["peak_h"], last["impulse"] = max(last["peak_h"], fh), last["impulse"] + fh * X.SIM_DT
            return
        d = self.d
        e = dict(kind=kind, side=side, t=round(t, 3), t1=t, force=force, peak_h=fh, impulse=fh * X.SIM_DT,
                 xy=[round(float(pos[0]), 3), round(float(pos[1]), 3)], z=round(float(pos[2]), 3), wp=self.wp,
                 _v0=d.qvel[:2].copy(), _tilt0=self._tilt(), dv=0.0, dtilt=0.0, pushed=False)
        e.update(extra or {})
        self.last_event[key] = e
        self.events.append(e)

    def _tilt(self):
        return math.degrees(math.acos(max(-1.0, min(1.0, self.d.xmat[self.pelvis][8]))))

    def _disturbance(self):
        """For every event younger than HIT_WINDOW: the largest change of the pelvis' horizontal velocity and of its tilt
        since the contact began. A push of the randomised protocol inside the window voids the measurement."""
        d = self.d
        tilt = None
        for e in reversed(self.events):
            if d.time - e["t"] > HIT_WINDOW + 0.05:
                break
            if d.time - e["t"] <= HIT_WINDOW:
                tilt = self._tilt() if tilt is None else tilt
                e["dv"] = max(e["dv"], float(np.linalg.norm(d.qvel[:2] - e["_v0"])))
                e["dtilt"] = max(e["dtilt"], tilt - e["_tilt0"])
                if any(e["t"] - 0.02 <= pt <= e["t"] + HIT_WINDOW for pt in self.push_done):
                    e["pushed"] = True

    def _agg(self, agg, kind, side, fw, fn, pos, extra):
        key = (kind, side, extra.get("cls"), extra.get("on"))
        a = agg.get(key)
        if a is None:
            agg[key] = [fw.copy(), fn, pos.copy(), extra]
        else:
            a[0] += fw
            a[1] = max(a[1], fn)

    def _jlim(self):
        qa, lo, hi = self._jl
        if len(qa):
            q = self.d.qpos[qa]
            exv = np.maximum(q - hi, lo - q)
            ex = float(max(0.0, np.max(exv)))
            self.jlim_max = max(self.jlim_max, ex)
            if ex > JLIM_EXCESS:
                self.jlim_t.append(float(self.d.time))
            if len(self._ar) and float(np.max(exv[self._ar])) > ANKLE_ROLL_EXCESS:
                self.ankroll_t.append(float(self.d.time))

    def _flags(self, reason):
        """Auto-flags (audit 2026-11-12): seconds with a trapped foot, a head / trunk / limb clamped in a wall box, a joint past its
        limit; for a fall, the last second before it is left out (impact)."""
        now = float(self.d.time)
        eps = list(self.flag_eps) + [[k[0], k[1], round(t0, 3), round(now, 3), None, None] for k, t0 in self.flag_runs.items() if now - t0 >= TRAP_MIN]
        cut = now - 1.0 if str(reason).startswith("fall") else now + 1.0
        dt_c = self.sub * X.SIM_DT

        def secs(kind):
            return round(sum(max(0.0, min(e[3], cut) - e[2]) for e in eps if e[0] == kind), 2)

        def longest(kind):
            return round(max([e[3] - e[2] for e in eps if e[0] == kind], default=0.0), 2)
        return dict(trapped_foot_s=secs("trapped_foot"), trapped_foot_longest_s=longest("trapped_foot"),
                    limb_clamp_s=secs("limb_clamp"), limb_clamp_parts=sorted({e[1] for e in eps if e[0] == "limb_clamp"}),
                    joint_limit_s=round(sum(dt_c for t_ in self.jlim_t if t_ < cut), 2), joint_limit_max_rad=round(self.jlim_max, 3),
                    ankle_roll_past_s=round(sum(dt_c for t_ in self.ankroll_t if t_ < cut), 2), joint_limit_threshold_rad=JLIM_EXCESS,
                    last_1s_of_fall_excluded=bool(str(reason).startswith("fall")), episodes=eps[:200])

    def _contacts(self):
        m, d, x = self.m, self.d, self.x
        load = {b: 0.0 for b in x.foot_ids}
        fsum = {b: np.zeros(3) for b in x.foot_ids}
        agg = {}                                            # event key -> [summed world force on the robot, max normal, pos, extra]
        deep = set()
        trunk_on_ground = False
        for k in range(d.ncon):
            c = d.contact[k]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if (b1 == 0) == (b2 == 0):                      # robot against itself (or nothing of the robot)
                continue
            env_geom, body = (c.geom1, b2) if b1 == 0 else (c.geom2, b1)
            mujoco.mj_contactForce(m, d, k, self._f6)
            fn = float(self._f6[0])
            fw = c.frame.reshape(3, 3).T @ self._f6[:3]       # the contact force in the world frame (on geom2)
            if b1 != 0:
                fw = -fw                                      # the force on the robot
            fh = float(math.hypot(fw[0], fw[1]))
            nz = abs(float(c.frame[2]))
            tag = x.wall_cls.get(env_geom)                  # a box that is a floor obstacle (slanted 4x4): scored like the floor
            on_wall = env_geom in x.wall_geoms and tag is None
            group = x.body_group[body]
            if group == "foot":                             # v2: every contact of the foot, walls included, as a vector
                fsum[body] += fw
                if -c.dist > TRAP_DEPTH:
                    deep.add(("trapped_foot", m.body(body).name))
            elif on_wall and -c.dist > CLAMP_DEPTH and group in ("head", "trunk", "arm", "thigh", "knee", "shin"):
                deep.add(("limb_clamp", m.body(body).name))
            if fn < F_MIN:
                continue
            name = m.body(body).name
            side = "left" if name.startswith("left") else "right" if name.startswith("right") else "-"
            face = (not on_wall) and (tag is not None or self._relief(c.pos[0], c.pos[1]) > SIDE_RELIEF)   # a face is here
            cls = "wall" if on_wall else (tag or (self._cls(c.pos[0], c.pos[1]) if face else None))
            if group == "foot":
                # the foot against the side of something: a wall box, or a horizontal-dominant contact where the raster
                # really has a face (height-field contacts can report a tilted normal on flat ground at impact). A sole
                # on a tread, a beam top or a ramp surface has a near-vertical normal and is not one.
                if on_wall or (nz < SIDE_NZ and face):
                    v6 = np.zeros(6)
                    mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_XBODY, body, v6, 0)
                    spd = float(math.hypot(v6[3], v6[4]))
                    self._agg(agg, "foot_side", side, fw, fn, c.pos, dict(on="wall" if on_wall else "terrain", cls=cls, nz=round(nz, 2),
                              swing=bool(self.load[body] < STANCE_ON or spd > 0.3), foot_speed=round(spd, 2)))
            elif group in ("shin", "knee", "thigh"):
                self._agg(agg, group, side, fw, fn, c.pos, dict(on="wall" if on_wall else "terrain", cls=cls))
            else:                                           # trunk, head, arm
                self._agg(agg, group, side, fw, fn, c.pos, dict(on="wall" if on_wall else "terrain", cls=cls))
                if group in ("trunk", "head") and not on_wall:
                    trunk_on_ground = True
        for (kind, side, _, _), (F, fnmax, pos, extra) in agg.items():   # one reading per episode key and step: the summed vector
            self._event(kind, side, d.time, float(np.linalg.norm(F)), pos, extra, float(math.hypot(F[0], F[1])))
        for b in x.foot_ids:
            load[b] = max(0.0, float(fsum[b][2]))
        now = float(d.time)
        for key in deep:
            self.flag_runs.setdefault(key, now)
        for key in [k_ for k_ in self.flag_runs if k_ not in deep]:
            t0 = self.flag_runs.pop(key)
            if now - t0 >= TRAP_MIN:
                self.flag_eps.append([key[0], key[1], round(t0, 3), round(now, 3), round(float(d.qpos[0]), 2), round(float(d.qpos[1]), 2)])
        for b in x.foot_ids:                                # landings: the ground under the sole at touchdown
            self.load[b] = load[b]
            if self.stance[b] and load[b] < STANCE_OFF:
                self.stance[b] = False
            elif not self.stance[b] and load[b] > STANCE_ON:
                self.stance[b] = True
                if d.time < 0.5:                              # the robot being set down at the start is not a step
                    continue
                P = self._sole_world(b)
                g = self.app.height(P[:, 0], P[:, 1]) if self.app is not None else np.zeros(len(P))
                A = np.c_[P[:, :2] - P[:, :2].mean(0), np.ones(len(P))]
                res = g - A @ np.linalg.lstsq(A, g, rcond=None)[0]
                rng_ = float(res.max() - res.min())
                self.steps[b].append(dict(t=round(float(d.time), 3), xy=[round(float(v), 3) for v in d.xpos[b][:2]], range=round(rng_, 3),
                                          edge=bool(rng_ > EDGE_RANGE), wp=self.wp))
        self.contact_steps += 1
        return trunk_on_ground

    # ---- one trial ----------------------------------------------------------------------------------------------------
    def run(self, seed: int, protocol: dict, seconds: float = TIME_LIMIT, record: bool = True, stand_only: bool = False, prestand: float = 0.0):
        """prestand (s): station legs (eval/legs.py): the robot first gets this long a stand command at its spawn pose (target = where
        it stands, heading = its spawn heading, hold), no waypoint progress; the stall timer starts after it."""
        m, d, x = self.m, self.d, self.x
        rand = self.reset(seed, protocol)
        reason, pushes = None, list(self.pushes)
        dt_c = self.sub * X.SIM_DT
        if prestand > 0:
            p0, yaw0 = d.qpos[:2].copy(), X.yaw_of(d.qpos[3:7])
            self.last_progress = prestand
        while reason is None:
            pre = prestand > 0 and d.time < prestand
            if not stand_only and not pre:
                self._progress(dt_c)
            if pre:
                cmd = Command(p0.copy(), self.ground(*p0), yaw0, True, np.zeros(3), float(np.linalg.norm(d.qpos[:2] - p0)), 0.0, 0,
                              len(self.wps), np.zeros((0, 2)), True)
            else:
                cmd = self.command()
            self.last_cmd = cmd
            if stand_only:
                cmd.vel[:] = 0.0
                cmd.hold = True
            target = np.asarray(self.adapter.act(self.view, cmd), float)
            assert target.shape == (31,) and np.isfinite(target).all(), "adapter must return 31 finite joint targets"
            d.ctrl[:] = target
            trunk = False
            for _ in range(self.sub):
                if pushes and d.time >= pushes[0][0]:
                    _, vx, vy = pushes.pop(0)
                    self.push_done.append(float(d.time))
                    d.qvel[0] += vx
                    d.qvel[1] += vy
                mujoco.mj_step(m, d)
                trunk |= self._contacts()
            self._beams()
            self._jlim()
            self._disturbance()
            self._travel()
            end_retry = END_RETRY and STAND_RULE == "strict" and self.wps and self.wps[-1]["hold_s"] > 0
            if self.done_t is not None and self.disc_reason is None and self.wps and self.wps[-1]["hold_s"] > 0:
                smp = self._stand_sample()                     # the end stand, strict rule
                if end_retry and (smp[0] < STRICT_LOAD or smp[1] >= STRICT_SPEED or not smp[2]):
                    self.end_win, self.end_breaks = [], self.end_breaks + 1     # a correction restarts the 1 s
                else:
                    self.end_win.append(smp)
            px, py, pz = d.qpos[:3]
            feet = [d.xpos[b] for b in x.foot_ids]
            low = min([self.ground(px, py)] + [self.ground(f[0], f[1]) for f in feet])
            h = float(pz - low)
            tilt = math.degrees(math.acos(max(-1.0, min(1.0, d.xmat[self.pelvis][8]))))
            if record:
                r = self.rec
                r["t"].append(d.time); r["qpos"].append(d.qpos.astype(np.float32).copy()); r["wp"].append(self.wp)
                r["cmd"].append(cmd.vel.copy()); r["load"].append([self.load[b] for b in x.foot_ids]); r["h"].append(h)
                r["hold"].append(cmd.hold)
            fall = None
            if h < FALL_H:
                fall = "fall: pelvis %.2f m above the ground" % h
            elif tilt > FALL_TILT:
                fall = "fall: pelvis tilted %.0f deg" % tilt
            elif trunk:
                fall = "fall: trunk or head on the ground"
            if self.disc_reason is not None:                   # after the end stand: only the strict rule's "still upright"
                if fall and self.strict_fail is None:
                    self.strict_fail = "%s %.1f s after the end stand" % (fall, d.time - self.end_ok_t)
                if fall or d.time - self.end_ok_t >= STRICT_POST_END:
                    reason = self.disc_reason
                    if STAND_RULE == "strict" and self.strict_fail:
                        reason = "not a stand: %s" % self.strict_fail
            elif fall:
                reason = fall
            elif not stand_only and self.app is not None and (self._off_route(px, py) > OFF_COURSE or not self.app.inside(px, py)):
                reason = "left the course"
            elif self.done_t is not None and (len(self.end_win) * dt_c >= END_STAND - 1e-9 if end_retry else d.time - self.done_t >= END_STAND):
                self.end_ok_t = float(d.time)
                missed = [k for k, v in self.beam_cross.items() if not v]
                self.disc_reason = "success" if not missed else "went round a 4x4: %s not stepped over" % ", ".join(missed)
                if self.end_win:
                    bad = self._stand_bad(self.end_win)
                    if bad and self.strict_fail is None:
                        self.strict_fail = "end stand: %s" % bad
                if self.disc_reason != "success":
                    reason = self.disc_reason
            elif d.time >= seconds:
                reason = "time limit" if not stand_only else "stood"
                if end_retry and self.done_t is not None:
                    reason = "not a stand: end stand never held 1 s (%d corrections) before the time limit" % self.end_breaks
            elif end_retry and self.done_t is not None and d.time - self.done_t > STALL:
                reason = "not a stand: end stand never held 1 s in %.0f s (%d corrections)" % (STALL, self.end_breaks)
            elif not stand_only and self.wps and self.done_t is None and d.time - self.last_progress > STALL + self.wps[self.wp]["hold_s"]:
                reason = "stalled"
        return self._result(seed, rand, reason)

    def _off_route(self, px, py):
        p = np.array([px, py])
        a, b = self.path[:-1], self.path[1:]
        ab = b - a
        t = np.clip(((p - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-12), 0, 1)
        return float(np.linalg.norm(a + ab * t[:, None] - p, axis=1).min())

    def _stop_context(self):
        """Where a trial stopped (coordinator 2026-11-12, for every stall / failure record): distance from the pelvis to the nearest
        wall / post / box (footprint, boxes starting below 1.3 m), to the nearest raster cell more than 0.15 m above the floor under
        the pelvis (within 1.5 m), to the edge of the course's raster, the harness command and the adapter's own command (if any)."""
        d = self.d
        px, py = float(d.qpos[0]), float(d.qpos[1])
        out = {}
        if self.app is not None:
            bd = [math.hypot(max(abs(px - b["center"][0]) - b["half"][0], 0.0), max(abs(py - b["center"][1]) - b["half"][1], 0.0))
                  for b in self.app.boxes if b["center"][2] - b["half"][2] < 1.3]
            out["nearest_box_m"] = round(min(bd), 3) if bd else None
            g = self.app.grid
            xs = self.app.x0 + np.arange(g.shape[0]) * self.app.res
            ys = self.app.y0 + np.arange(g.shape[1]) * self.app.res
            i0, i1 = np.searchsorted(xs, px - 1.5), np.searchsorted(xs, px + 1.5)
            j0, j1 = np.searchsorted(ys, py - 1.5), np.searchsorted(ys, py + 1.5)
            hi = g[i0:i1, j0:j1] > self.ground(px, py) + 0.15
            if hi.any():
                ii, jj = np.nonzero(hi)
                out["nearest_raised_cell_m"] = round(float(np.min(np.hypot(xs[i0 + ii] - px, ys[j0 + jj] - py))), 3)
            else:
                out["nearest_raised_cell_m"] = None
            out["course_edge_m"] = round(float(min(px - xs[0], xs[-1] - px, py - ys[0], ys[-1] - py)), 3)
        c = getattr(self, "last_cmd", None)
        out["harness_cmd"] = [round(float(v), 3) for v in c.vel] if c is not None else None
        ac = getattr(self.adapter, "last_command", None)
        out["adapter_cmd"] = [round(float(v), 3) for v in ac] if ac is not None else None
        return out

    def _result(self, seed, rand, reason):
        d, x = self.d, self.x
        hits = {c: dict(foot=0, foot_in_swing=0, shin_knee=0, scuffs=0, impulse=0.0, peak_impulse=0.0, peak_force=0.0, penalty=0.0) for c in CLASSES}
        penalty = 0.0
        for e in self.events:
            e["dur"] = round(e.pop("t1") - e["t"], 3)
            e["force"] = round(e["force"], 1)
            e.pop("_v0"); e.pop("_tilt0")
            e["peak_h"], e["impulse"], e["dv"], e["dtilt"] = round(e["peak_h"], 1), round(e["impulse"], 3), round(e["dv"], 3), round(max(e["dtilt"], 0.0), 2)
            if e["kind"] in ("foot_side", "shin", "knee") and e.get("cls") in CLASSES:
                h = hits[e["cls"]]
                if e["peak_h"] >= HIT_F or e["impulse"] >= HIT_J:
                    dist = 0.0 if e["pushed"] else min(1.0, max(e["dv"] / HIT_DV_REF, e["dtilt"] / HIT_TILT_REF))
                    e["hit"], e["penalty"] = True, round(HIT_BASE + HIT_J_W * min(1.0, e["impulse"] / HIT_J_REF) + HIT_D_W * dist, 3)
                    if e["kind"] == "foot_side":
                        h["foot"] += 1
                        h["foot_in_swing"] += int(e["swing"])
                    else:
                        h["shin_knee"] += 1
                    h["impulse"] += e["impulse"]
                    h["peak_impulse"], h["peak_force"] = max(h["peak_impulse"], e["impulse"]), max(h["peak_force"], e["peak_h"])
                else:
                    e["hit"], e["penalty"] = False, SCUFF_P
                    h["scuffs"] += 1
                h["penalty"] += e["penalty"]
                penalty += e["penalty"]
        for h in hits.values():
            h["impulse"], h["penalty"] = round(h["impulse"], 3), round(h["penalty"], 3)
        n_hits = sum(h["foot"] + h["shin_knee"] for h in hits.values())
        n = len(self.wps)
        sec_of = [w["section"] for w in self.wps]
        sections = {}
        for s in (self.app.sections if self.app is not None else []):
            idx = [i for i, v in enumerate(sec_of) if v == s]
            sections[s] = dict(entered=bool(self.wp >= idx[0]), completed=bool(self.wp > idx[-1]))
        steps = [s for b in x.foot_ids for s in self.steps[b]]
        kinds = {}
        for e in self.events:
            kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
        stubs = [e for e in self.events if e["kind"] == "foot_side"]
        res = dict(
            seed=seed, success=reason == "success", clean=bool(reason == "success" and n_hits == 0), side_hits=hits, side_hit_count=n_hits,
            stumble_penalty=round(penalty, 3), reason=reason, time=round(float(self.done_t if self.done_t is not None else d.time), 2),
            sim_time=round(float(d.time), 2), waypoints=self.wp, of=n, reach_t=self.reach_t,
            stopped_heading_to=(self.wps[self.wp]["name"] if self.wp < n else None), stopped_section=(sec_of[self.wp] if self.wp < n else None),
            sections=sections, waypoint_sections=sec_of, random=rand, beams_crossed=self.beam_cross,
            contacts=dict(feet_checked=True, physics_steps_checked=self.contact_steps, steps=len(steps), edge_landings=sum(s["edge"] for s in steps),
                          foot_side_contacts=len(stubs), foot_side_in_swing=sum(e["swing"] for e in stubs),
                          shin=kinds.get("shin", 0), knee=kinds.get("knee", 0), thigh=kinds.get("thigh", 0), arm=kinds.get("arm", 0),
                          trunk=kinds.get("trunk", 0), head=kinds.get("head", 0)),
            events=self.events[:2000], events_dropped=max(0, len(self.events) - 2000), edge_steps=[s for s in steps if s["edge"]][:500],
            final=dict(xy=[round(float(d.qpos[0]), 3), round(float(d.qpos[1]), 3)], pelvis_z=round(float(d.qpos[2]), 3)),
            stop=(self._stop_context() if reason != "success" else None),
            stand_rule=STAND_RULE, end_retry=bool(END_RETRY and STAND_RULE == "strict"), end_breaks=self.end_breaks, strict_success=bool(reason == "success" and self.strict_fail is None), strict_fail=self.strict_fail if (reason == "success" or reason.startswith("not a stand")) else None,
            stations=[dict(name=s["name"], t=s["t"], strict=s["strict"], why=s["why"], travel_1s=round(s["travel"], 3)) for s in self.stations])
        res["strict_clean"] = bool(res["strict_success"] and res["clean"])
        res["flags"] = self._flags(reason)
        res["measure"] = MEASURE
        return res

    def save_traj(self, path):
        r = self.rec
        np.savez_compressed(path, t=np.array(r["t"]), qpos=np.array(r["qpos"]), wp=np.array(r["wp"]), cmd=np.array(r["cmd"]),
                            load=np.array(r["load"], np.float32), h=np.array(r["h"], np.float32), hold=np.array(r["hold"]))


def _why_key(f):
    """Short class of a strict-rule failure: where (station name, end stand, after the end stand) and which condition."""
    if f.startswith("fall"):
        return "fell after the end stand"
    where, _, why = f.partition(": ")
    cond = [c for c, k in (("feet unloaded", "weight"), ("moving", "m/s"), ("left the disc", "disc")) if k in why]
    return "%s: %s" % (where, ", ".join(cond))


def _count(xs):
    out = {}
    for v in xs:
        out[v] = out.get(v, 0) + 1
    return out


def summarise(results, meta):
    n = len(results)
    ok = [r for r in results if r["success"]]
    lo, hi = wilson(len(ok), n)
    secs = {}
    for s in meta["sections"]:
        ent = sum(r["sections"][s]["entered"] for r in results)
        com = sum(r["sections"][s]["completed"] for r in results)
        secs[s] = dict(entered=ent, completed=com, of=n)
    reasons, stopped = {}, {}
    for r in results:
        k = r["reason"].split(":")[0]
        reasons[k] = reasons.get(k, 0) + 1
        if not r["success"] and r["stopped_section"]:
            stopped[r["stopped_section"]] = stopped.get(r["stopped_section"], 0) + 1
    by_sec = {s: dict(foot_side=0, shin_knee=0, thigh=0, arm_trunk_head=0, edge_landings=0, side_hits=0, stumble_penalty=0.0)
              for s in meta["sections"]}
    clean = [r for r in results if r["clean"]]
    clo, chi = wilson(len(clean), n)
    hit_tot = {c: {k: 0.0 for k in ("foot", "foot_in_swing", "shin_knee", "scuffs", "impulse", "penalty")} for c in CLASSES}
    peak_j = peak_f = 0.0
    for r in results:
        for c in CLASSES:
            for k in hit_tot[c]:
                hit_tot[c][k] += r["side_hits"][c][k]
            peak_j, peak_f = max(peak_j, r["side_hits"][c]["peak_impulse"]), max(peak_f, r["side_hits"][c]["peak_force"])
    for c in CLASSES:
        hit_tot[c] = {k: (round(v, 3) if k in ("impulse", "penalty") else int(v)) for k, v in hit_tot[c].items()}
    for r in results:                                       # events carry the waypoint the robot was heading to
        sec = r.get("waypoint_sections", [])
        for e in r["events"]:
            if e["wp"] < len(sec):
                k = {"foot_side": "foot_side", "shin": "shin_knee", "knee": "shin_knee", "thigh": "thigh"}.get(e["kind"], "arm_trunk_head")
                by_sec[sec[e["wp"]]][k] += 1
                if "penalty" in e:
                    by_sec[sec[e["wp"]]]["stumble_penalty"] = round(by_sec[sec[e["wp"]]]["stumble_penalty"] + e["penalty"], 3)
                    by_sec[sec[e["wp"]]]["side_hits"] += int(e["hit"])
        for e in r["edge_steps"]:
            if e["wp"] < len(sec):
                by_sec[sec[e["wp"]]]["edge_landings"] += 1
    tot = lambda k: int(sum(r["contacts"][k] for r in results))  # noqa: E731
    steps = max(tot("steps"), 1)
    times = sorted(r["time"] for r in ok)
    strict = [r for r in results if r.get("strict_success")]
    slo, shi = wilson(len(strict), n)
    have_strict = all("strict_success" in r for r in results)
    st = {}
    for r in results:
        for x_ in r.get("stations", []):
            st.setdefault(x_["name"], []).append(x_["travel_1s"])
    travel = {k: dict(reached=len(v), median_m=round(float(np.median(v)), 3), max_m=round(float(np.max(v)), 3),
                      over_0_2m=int(sum(t > 0.2 for t in v))) for k, v in st.items()}
    return dict(meta, trials=n, stand_rule=(results[0].get("stand_rule", "disc") if results else STAND_RULE),
                strict_rule="strict stand (coordinator 2026-11-12): every station and the end stand 1 s in the disc with the feet carrying >= %.0f %% of the weight and pelvis speed < %.2f m/s, upright %.0f s after the end stand" % (100 * STRICT_LOAD, STRICT_SPEED, STRICT_POST_END),
                strict_successes=len(strict) if have_strict else None, strict_success_rate=round(len(strict) / max(n, 1), 4) if have_strict else None,
                strict_ci95=[round(slo, 4), round(shi, 4)] if have_strict else None,
                strict_clean_successes=sum(bool(r.get("strict_clean")) for r in results) if have_strict else None,
                strict_fail_reasons=_count([_why_key(r["strict_fail"]) for r in results if r.get("strict_fail")]),
                station_travel_1s=travel,
                measure=(results[0].get("measure", "v1 (before 2026-11-12 evening): summed magnitudes") if results else MEASURE),
                flags=dict(trials_with_trapped_foot=sum(1 for r in results if (r.get("flags") or {}).get("trapped_foot_s", 0) > 0),
                           trapped_foot_s_total=round(sum((r.get("flags") or {}).get("trapped_foot_s", 0) for r in results), 2),
                           trials_with_limb_clamp=sum(1 for r in results if (r.get("flags") or {}).get("limb_clamp_s", 0) > 0),
                           trials_with_joint_limit=sum(1 for r in results if (r.get("flags") or {}).get("joint_limit_s", 0) > 0),
                           joint_limit_s_total=round(sum((r.get("flags") or {}).get("joint_limit_s", 0) for r in results), 2),
                           joint_limit_threshold_rad=JLIM_EXCESS,
                           trials_with_ankle_roll_past=sum(1 for r in results if (r.get("flags") or {}).get("ankle_roll_past_s", 0) > 0),
                           ankle_roll_past_s_total=round(sum((r.get("flags") or {}).get("ankle_roll_past_s", 0) for r in results), 2)),
                disc_rule="disc only (the rule before 2026-11-12): 1 s inside the disc at each station, still up 1 s after the last waypoint",
                successes=len(ok), success_rate=round(len(ok) / max(n, 1), 4), ci95=[round(lo, 4), round(hi, 4)],
                clean_successes=len(clean), clean_success_rate=round(len(clean) / max(n, 1), 4), clean_ci95=[round(clo, 4), round(chi, 4)],
                mean_stumble_penalty=round(float(np.mean([r["stumble_penalty"] for r in results])) if results else 0.0, 3),
                side_hits=dict(by_class=hit_tot, per_trial=round(sum(r["side_hit_count"] for r in results) / max(n, 1), 2),
                               summed_impulse=round(sum(hit_tot[c]["impulse"] for c in CLASSES), 3), peak_impulse=round(peak_j, 3), peak_force=round(peak_f, 1),
                               by_section={k: dict(side_hits=v["side_hits"], stumble_penalty=v["stumble_penalty"],
                                                   mean_penalty_per_trial=round(v["stumble_penalty"] / max(n, 1), 3)) for k, v in by_sec.items()}),
                sections=secs, outcomes=reasons, failures_by_section=stopped,
                beams_crossed={b: int(sum(bool(r["beams_crossed"].get(b)) for r in results)) for b in (results[0]["beams_crossed"] if results else {})},
                time_success=dict(min=times[0], median=times[len(times) // 2], max=times[-1]) if times else None,
                mean_waypoints=round(float(np.mean([r["waypoints"] for r in results])), 2),
                contacts=dict(feet_checked=all(r["contacts"]["feet_checked"] for r in results), steps=tot("steps"), edge_landings=tot("edge_landings"),
                              edge_landing_rate=round(tot("edge_landings") / steps, 4), foot_side_contacts=tot("foot_side_contacts"),
                              foot_side_in_swing=tot("foot_side_in_swing"), shin=tot("shin"), knee=tot("knee"), thigh=tot("thigh"),
                              arm=tot("arm"), trunk=tot("trunk"), head=tot("head"), by_section=by_sec,
                              per_trial=dict(foot_side=round(tot("foot_side_contacts") / max(n, 1), 2), shin_knee=round((tot("shin") + tot("knee")) / max(n, 1), 2))))


TRIAL_COLS = ["seed", "outcome", "strict_success", "strict_fail", "success", "clean", "time_s", "sim_time_s", "waypoints", "of", "last_waypoint", "stopped_heading_to",
              "stopped_section", "sections_completed", "beams_crossed", "side_hits", "side_hits_4x4", "side_hits_wedge", "side_hits_stair",
              "scuffs", "stumble_penalty", "summed_impulse_Ns", "peak_impulse_Ns", "peak_force_N", "landings", "edge_landings",
              "foot_side_all", "shin", "knee", "thigh", "arm", "trunk", "head", "start_dx", "start_dy", "start_dyaw_deg", "friction",
              "mass_scale", "torso_payload_kg", "final_x", "final_y", "final_pelvis_z", "station_travel_1s_m",
              "stop_nearest_box_m", "stop_nearest_raised_cell_m", "stop_course_edge_m", "stop_harness_cmd", "stop_adapter_cmd",
              "flag_trapped_foot_s", "flag_trapped_foot_longest_s", "flag_limb_clamp_s", "flag_joint_limit_s", "flag_joint_limit_max_rad", "flag_ankle_roll_past_s"]
EVENT_COLS = ["seed", "t_s", "dur_s", "kind", "side", "on", "class", "hit", "penalty", "peak_horizontal_N", "impulse_Ns", "peak_normal_N",
              "normal_z", "in_swing", "foot_speed", "pelvis_dv", "pelvis_dtilt_deg", "push_in_window", "x", "y", "z", "heading_to_waypoint", "section"]


def write_logs(out, results, wps):
    """trials.csv (one row per trial) and events.csv (one row per contact event: every side hit and scuff, every other
    foot-side, shin, knee, thigh, arm, trunk and head contact; edge landings as kind "edge_landing"). Legend: docs/EVAL.md."""
    import csv
    with open(out / "trials.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(TRIAL_COLS)
        for r in results:
            h, c, rd = r["side_hits"], r["contacts"], r["random"]
            w.writerow([r["seed"], r["reason"], int(bool(r.get("strict_success"))) if "strict_success" in r else "", r.get("strict_fail") or "", int(r["success"]), int(r["clean"]), r["time"], r["sim_time"], r["waypoints"], r["of"],
                        wps[r["waypoints"] - 1]["name"] if r["waypoints"] else "", r["stopped_heading_to"] or "", r["stopped_section"] or "",
                        " ".join(k for k, v in r["sections"].items() if v["completed"]),
                        " ".join("%s:%d" % (k.replace(" ", "_"), len(v)) for k, v in r["beams_crossed"].items()), r["side_hit_count"],
                        *[h[k]["foot"] + h[k]["shin_knee"] for k in CLASSES], sum(h[k]["scuffs"] for k in CLASSES), r["stumble_penalty"],
                        round(sum(h[k]["impulse"] for k in CLASSES), 3), max(h[k]["peak_impulse"] for k in CLASSES), max(h[k]["peak_force"] for k in CLASSES),
                        c["steps"], c["edge_landings"], c["foot_side_contacts"], c["shin"], c["knee"], c["thigh"], c["arm"], c["trunk"], c["head"],
                        round(rd["dx"], 3), round(rd["dy"], 3), round(rd["dyaw_deg"], 2), round(rd["friction"], 3), round(rd["mass_scale"], 3),
                        round(rd["torso_payload"], 2), *r["final"]["xy"], r["final"]["pelvis_z"],
                        " ".join("%s:%.2f%s" % (x_["name"].replace(" ", "_"), x_["travel_1s"], "" if x_["strict"] else "!") for x_ in r.get("stations", [])),
                        *[(r.get("stop") or {}).get(k, "") for k in ("nearest_box_m", "nearest_raised_cell_m", "course_edge_m")],
                        " ".join(map(str, (r.get("stop") or {}).get("harness_cmd") or [])), " ".join(map(str, (r.get("stop") or {}).get("adapter_cmd") or [])),
                        *[(r.get("flags") or {}).get(k, "") for k in ("trapped_foot_s", "trapped_foot_longest_s", "limb_clamp_s", "joint_limit_s", "joint_limit_max_rad", "ankle_roll_past_s")]])
    with open(out / "events.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(EVENT_COLS)
        for r in results:
            sec = r["waypoint_sections"]
            rows = [[r["seed"], e["t"], e["dur"], e["kind"], e["side"], e.get("on", ""), e.get("cls") or "", int(e["hit"]) if "hit" in e else "",
                     e.get("penalty", ""), e["peak_h"], e["impulse"], e["force"], e.get("nz", ""), int(e["swing"]) if "swing" in e else "",
                     e.get("foot_speed", ""), e["dv"], e["dtilt"], int(e["pushed"]), e["xy"][0], e["xy"][1], e["z"],
                     wps[e["wp"]]["name"] if e["wp"] < len(wps) else "", sec[e["wp"]] if e["wp"] < len(sec) else ""] for e in r["events"]]
            rows += [[r["seed"], e["t"], "", "edge_landing", "", "terrain", "", "", "", "", "", "", "", "", "", "", "", "", e["xy"][0], e["xy"][1], "",
                      wps[e["wp"]]["name"] if e["wp"] < len(wps) else "", sec[e["wp"]] if e["wp"] < len(sec) else ""] for e in r["edge_steps"]]
            w.writerows(sorted(rows, key=lambda x: x[1]))


def pick_renders(results):
    """Seeds of the best (fastest success), the median success and a failure (the median one by waypoints)."""
    ok = sorted((r for r in results if r["success"]), key=lambda r: r["time"])
    bad = sorted((r for r in results if not r["success"]), key=lambda r: r["waypoints"])
    out = {}
    if ok:
        out["best"], out["median"] = ok[0]["seed"], ok[len(ok) // 2]["seed"]
    if bad:
        out["failed"] = bad[len(bad) // 2]["seed"]
    return out


def kinematic(app: Apparatus, out: pathlib.Path, speed=0.6):
    """No physics: the pelvis is carried along the route at standing height (default pose) to check the waypoint and
    section bookkeeping and the cameras over the whole course. Never a result; the output says so."""
    class Carry(Adapter):
        name, checkpoint, observes = "kinematic", "none (pelvis carried along the route, no physics)", "nothing"

        def act(self, view, cmd):
            return view.default_q.copy()
    x = X.build(app)
    tr = Trial(x, app, Carry(), x.model.body_mass.copy(), x.model.body_inertia.copy())
    tr.reset(0, dict(NOMINAL, start_xy=0.0, start_yaw_deg=0.0))
    d, dt = tr.d, 0.02
    yaw = X.yaw_of(d.qpos[3:7])
    while tr.wp < len(tr.wps) and d.time < 600:
        tr._progress(dt)
        cmd = tr.command()
        if tr.wp >= len(tr.wps):
            break
        if not cmd.hold:
            step = min(speed * dt, cmd.dist)
            v = (cmd.target_xy - d.qpos[:2]) / max(cmd.dist, 1e-9)
            d.qpos[:2] += v * step
            yaw += float(np.clip(X.wrap(math.atan2(v[1], v[0]) - yaw), -3 * dt, 3 * dt))
        elif cmd.target_heading is not None:
            yaw += float(np.clip(X.wrap(cmd.target_heading - yaw), -3 * dt, 3 * dt))
        d.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        d.qpos[2] = app.height_patch(d.qpos[0], d.qpos[1], 0.12, np.max) + tr.h0
        d.time += dt
        mujoco.mj_forward(x.model, d)
        tr._beams()
        r = tr.rec
        r["t"].append(d.time); r["qpos"].append(d.qpos.astype(np.float32).copy()); r["wp"].append(tr.wp); r["cmd"].append(cmd.vel.copy())
        r["load"].append([0.0, 0.0]); r["h"].append(tr.h0); r["hold"].append(cmd.hold)
    res = tr._result(0, tr.rand, "kinematic sweep (not a result)")
    out.mkdir(parents=True, exist_ok=True)
    (out / "traj").mkdir(exist_ok=True)
    tr.save_traj(out / "traj" / "0.npz")
    meta = dict(kind="kinematic sweep, no physics, not a result", apparatus=app.name, simulator="MuJoCo %s (kinematics only)" % mujoco.__version__,
                adapter="kinematic", checkpoint=Carry.checkpoint, sections=app.sections, protocol="none", control_dt=dt)
    json.dump([res], open(out / "results.json", "w"), indent=1)
    json.dump(dict(meta, waypoints=res["waypoints"], of=res["of"], reach_t=res["reach_t"], sections=res["sections"], beams_crossed=res["beams_crossed"]),
              open(out / "summary.json", "w"), indent=1)
    print("kinematic sweep: %d / %d waypoints in %.1f s of route time; sections %s; beam crossings at t = %s" % (res["waypoints"], res["of"], d.time,
          {k: v["completed"] for k, v in res["sections"].items()}, res["beams_crossed"]))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="zero")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--apparatus", default="ours", help="ours | nist | flat")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--seed0", type=int, default=SEED0)
    ap.add_argument("--protocol", choices=["randomised", "nominal"], default="randomised")
    ap.add_argument("--seconds", type=float, default=TIME_LIMIT)
    ap.add_argument("--stand", action="store_true", help="standing check: zero command, no route, ends after --seconds")
    ap.add_argument("--kinematic", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep-traj", choices=["picked", "all"], default="picked")
    ap.add_argument("--wall-limit", type=float, default=0.0, help="s: stop starting trials after this long, keep what is done in "
                    "partial.json and exit with code 3; run the same command again to continue")
    ap.add_argument("--label", default=None, help="a line for the summary and render labels (what this run is)")
    a = ap.parse_args()
    import memcap
    memcap.guard(cap_gb=6, need_free_gb=12, low_priority=True)
    out = pathlib.Path(a.out)
    app = None if a.apparatus == "flat" else Apparatus(a.apparatus)
    if a.kinematic:
        kinematic(app, out)
        return
    adapter = load_adapter(a.adapter, a.checkpoint)
    x = X.build(app, adapter.plant(X.joint_names_of(X.robot_dir() / "mjcf" / "x2_ultra.xml")))
    trial = Trial(x, app, adapter, x.model.body_mass.copy(), x.model.body_inertia.copy())
    protocol = RANDOM if a.protocol == "randomised" else NOMINAL
    (out / "traj").mkdir(parents=True, exist_ok=True)
    results, t0 = [], time.time()
    part = out / "partial.json"
    if part.exists():                                       # continue an interrupted run: same seeds, same definition
        results = json.load(open(part))
        print("continuing: %d trials already done" % len(results), flush=True)
    elif not a.kinematic:                                   # the regression check, once at the start of a run (eval/regression.py)
        import regression
        rg = regression.gate()
        if rg is not None:
            json.dump(rg, open(out / "regression.json", "w"), indent=1)
    have = {r["seed"] for r in results}
    for i in range(a.trials):
        seed = a.seed0 + i
        if seed in have:
            continue
        if a.wall_limit and time.time() - t0 > a.wall_limit:
            json.dump(results, open(str(part) + ".tmp", "w"))
            os.replace(str(part) + ".tmp", part)
            print("wall limit: %d of %d trials done, run again to continue" % (len(results), a.trials), flush=True)
            sys.exit(3)
        r = trial.run(seed, protocol, a.seconds, record=True, stand_only=a.stand)
        trial.save_traj(out / "traj" / ("%d.npz" % seed))
        results.append(r)
        if True:                                            # every trial (2026-11-12: slow SSR trials, over 5 per 590 s slice, were never saved)
            json.dump(results, open(str(part) + ".tmp", "w"))
            os.replace(str(part) + ".tmp", part)
        print("trial %d seed %d: %s, %d / %d waypoints, %.1f s, steps %d (edge %d), foot side %d, shin/knee %d, wall time %.0f s" % (
            i, seed, r["reason"], r["waypoints"], r["of"], r["sim_time"], r["contacts"]["steps"], r["contacts"]["edge_landings"],
            r["contacts"]["foot_side_contacts"], r["contacts"]["shin"] + r["contacts"]["knee"], time.time() - t0), flush=True)
    results.sort(key=lambda r: r["seed"])
    if part.exists():
        os.remove(part)
    rgf = out / "regression.json"
    meta = dict(regression=(json.load(open(rgf)) if rgf.exists() else "not run"), label=a.label, apparatus=a.apparatus, route=(app.route["name"] if app else None), simulator="MuJoCo %s (CPU), dt %.4f s%s" % (mujoco.__version__, X.SIM_DT, X.model_note()), policy_trained_on=getattr(adapter, "trained_model", None),
                adapter=adapter.name, checkpoint=str(adapter.checkpoint), observes=adapter.observes, control_dt=adapter.control_dt,
                plant=adapter.plant_name or "harness default vendor_20260823 (test fixture)",
                protocol=a.protocol if not a.stand else "standing check (zero command)", seeds=[a.seed0, a.seed0 + a.trials - 1],
                time_limit=a.seconds, sections=(app.sections if app else []),
                definition=dict(radius=RADIUS, z_tol=Z_TOL, end_stand=END_STAND, stall=STALL, fall_h=FALL_H, fall_tilt=FALL_TILT,
                                off_course=OFF_COURSE, f_min=F_MIN, side_nz=SIDE_NZ, side_relief=SIDE_RELIEF, edge_range=EDGE_RANGE,
                                hit_f=HIT_F, hit_j=HIT_J, scuff_p=SCUFF_P, hit_base=HIT_BASE, hit_j_w=HIT_J_W, hit_j_ref=HIT_J_REF, hit_d_w=HIT_D_W,
                                hit_dv_ref=HIT_DV_REF, hit_tilt_ref=HIT_TILT_REF, hit_window=HIT_WINDOW, beam_near=BEAM_NEAR))
    s = summarise(results, meta)
    s["renders"] = pick_renders(results) if not a.stand else {"stand": results[0]["seed"]}
    if a.keep_traj == "picked":
        keep = set(s["renders"].values())
        # and up to 5 failing seeds per section and kind (fall / stall / other), so typical failures can be rendered (2026-11-12)
        per = {}
        for r in results:
            if not r["success"]:
                k_ = (r.get("stopped_section"), str(r["reason"]).split(":")[0])
                if len(per.setdefault(k_, [])) < 5:
                    per[k_].append(r["seed"])
        keep |= {sd for v in per.values() for sd in v}
        s["kept_failures"] = {"%s / %s" % k_: v for k_, v in per.items()}
        for r in results:
            if r["seed"] not in keep:
                os.remove(out / "traj" / ("%d.npz" % r["seed"]))
    json.dump(results, open(out / "results.json", "w"), indent=1)
    json.dump(s, open(out / "summary.json", "w"), indent=1)
    write_logs(out, results, app.waypoints if app else [])
    print("%s on %s: success %d / %d (%.1f %%, 95 %% CI %.1f-%.1f %%); clean success %d / %d (%.1f %%); mean stumble penalty %.2f; side hits per trial %.2f; outcomes %s; renders %s" % (
        adapter.name, a.apparatus, s["successes"], s["trials"], 100 * s["success_rate"], 100 * s["ci95"][0], 100 * s["ci95"][1],
        s["clean_successes"], s["trials"], 100 * s["clean_success_rate"], s["mean_stumble_penalty"], s["side_hits"]["per_trial"], s["outcomes"], s["renders"]))


if __name__ == "__main__":
    main()
