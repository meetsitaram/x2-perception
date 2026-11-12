"""Fit a captured stair climb (Bones-SEED, X2 retarget) to the apparatus stairs: every footstep onto its tread.

The capture's stairs are not ours (rise, run, number of steps), so placing the clip is not enough. Instead:

  plants     each foot's stance plateaus (sole centre slower than V_PLANT for MIN_PLANT frames); the capture's rise is
             the median step between plant heights, and each plant's level is its height in rises (0 floor, 1.. treads)
  place      the clip turned so the climb goes +x (heading from the first to the last stair plant), centred across
             the stairs
  targets    a plant on level k (1..len(TOPS)) goes, foot turned straight up the stairs, onto tread k with its toe
             TOE_GAP short of the next riser, at the course's height there; floor plants stay, pulled
             back if the toe would reach the first riser; plants above the last tread go onto the landing at its
             height, at least LANDING_IN past its edge
  offsets    per foot, the target minus the plant (xyz), held through the plant and eased (smoothstep) between plants;
             the root moves by the mean of the feet's plant offsets as they bear weight (a swinging foot counts at
             its last plant), averaged over ROOT_SMOOTH frames
  clearance  any swing whose sole passes closer than CLEAR above the terrain under it is lifted by a bump over
             that swing (sin profile), to CLEAR
  legs       leg IK (stepgen.LegIK) to each ankle's offset position, ankle orientation as captured; arms, waist and
             head as captured

    q, info = fit(load_csv(path), course)

Run (from sim): python -m pipeline.stairfit   -> build/stairs/fit/<clip>.npy + a fit report
"""
from __future__ import annotations

import glob
import json
import math
import os

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from pipeline import clipdb as C
from pipeline import hmap as H
from pipeline import props as P
from pipeline import stepgen as SG

# The stairs per apparatus (use()): riser faces along x in climbing order, tread tops, landing deck top, middle across,
# climbing direction along x (+1 / -1), deck the stairs start from. Measured on the height map.
STAIRS = dict(
    cad=dict(risers=(8.62, 8.84, 9.06, 9.30, 9.52, 9.74), tops=(0.186, 0.373, 0.559, 0.745, 0.931),
             landing=1.118, y=0.572, dir=1, floor=0.0),      # landing 44 in; the last riser's nose strip is 1.099
    nist=dict(risers=(3.668, 3.433, 3.198, 2.953, 2.713), tops=(0.341, 0.545, 0.748, 0.951),
              landing=1.154, y=4.966, dir=-1, floor=0.14, leg_cone=True))   # 20.3 cm risers, 23.5-24.5 cm treads,
#                                                                            climbing -x
RISERS, TOPS, LANDING = STAIRS['cad']['risers'], STAIRS['cad']['tops'], STAIRS['cad']['landing']
STAIR_Y, DIR, FLOOR = STAIRS['cad']['y'], STAIRS['cad']['dir'], STAIRS['cad']['floor']
LEG_CONE = False                                   # root cap by each stance foot's reach cone (root_height; use())
LANDING_IN = 0.12                                  # m: landing plants' sole centre at least this far past the edge
TOE_GAP = 0.02                                     # m: a tread foot's toe this far short of the next riser (the heel
#                                                    overhangs the tread edge instead: a 22 cm foot on a 22.2 cm run)
FLOOR_GAP = 0.05                                   # m: a floor plant's toe at least this short of the first riser (was 1 cm:
#                                                    a step landing a few cm long would hit the riser; edge buffer)
TOE, HEEL = 0.148, 0.072                           # m: sole extent ahead of / behind the ankle (foot shell)
SOLE_X = (TOE - HEEL) / 2                          # m: sole centre ahead of the ankle
V_PLANT, MIN_PLANT = 0.12, 8                       # m/s, frames
MERGE_GAP, MERGE_MOVE = 5, 0.03                    # frames, m: a speed blip this short and small splits no plant
MIN_STOP, V_STOP = 4, 0.05                         # frames, m/s: a shorter plant counts if the foot really stops
LEVEL_GAP = 0.04                                   # m: plant heights further apart than this are different levels
ROOT_SMOOTH = 21                                   # frames: window of the root offset's moving average (0.4 s)
LIFT_EDGE = 0.2                                    # share of a swing over which a clearance lift rises (and falls)
CLEAR_PASSES, IK_ITERS = 3, 40                     # clearance passes (each re-solves the legs), IK iterations
MAX_LIFT = 0.08                                    # m: most a clearance pass lifts a swing
SINK_TOL, SINK_EASE = 0.005, 6                     # m a sole may sink before it is raised; frames the raise eases over
# TCRS swing optimization (Perceptive BFM Eq. 3, sampled as in its Eq. 9 with MPPI): knots of a displacement of the
# swing foot from its eased path; cost = W_REF stay near it + W_SMOOTH smooth + W_CLEAR clearance CLEAR under every
# sole point + W_EDGE clearance EDGE_CLEAR over the highest terrain within EDGE_R (a vertical face under the
# footprint); liftoff and landing fixed
LATCH = False                                      # TCRS stance latching: a planted foot pinned (latch_curve); off: see PERCEPTION_PROGRESS (2026-10-27 pilot)
SWING_KNOTS, MPPI_SAMPLES, MPPI_ITERS, MPPI_SIGMA = 6, 128, 8, 0.03
W_REF, W_SMOOTH, W_CLEAR, W_EDGE = 1.0, 4.0, 400.0, 100.0
# edge inflation (2026-10-27): where terrain within EDGE_R of the sole rises more than EDGE_STEP over the ground under
# it, the sole must pass EDGE_CLEAR over that highest point (was 3 cm / 1 cm on every swing: the references grazed the
# 4x4s by 1 cm, and a tracker a few cm off caught the edge)
EDGE_R, EDGE_CLEAR, EDGE_STEP = 0.08, 0.06, 0.02   # m
# the buffer shrinks with what the leg can do: a taller edge and a shorter step leave less room above it.
# buffer = EDGE_CLEAR * (1 - rise / RISE_FULL) * min(1, stride / STRIDE_FULL), at least EDGE_MIN over the top
# (a 9 cm 4x4 on a 40 cm stride: 3.8 cm; an 18.6 cm riser: 1.5 -> EDGE_MIN 2 cm)
RISE_FULL, STRIDE_FULL, EDGE_MIN = 0.25, 0.40, 0.02
# TCRS root reconstruction (Perceptive BFM Eq. 10): root height = support-weighted terrain under the adapted feet +
# the raw root-to-sole offset, at most LEG_REACH above the support, smoothed
LEG_REACH = 0.72                                   # m: root above the supporting sole at full leg extension (X2)
# LEG_CONE (NIST, 2026-10-28): the cap is per stance foot and shrinks with how far ahead or behind the root it is,
# sqrt(LEG_STRAIGHT^2 - d^2) - LEG_SLACK (d along the heading): over the NIST 4x4 (12.6 cm) the trailing leg reached
# full extension (root 0.671 m above its sole) and its knee snapped from 0 to 0.8 rad in a frame on lift-off.
# LEG_STRAIGHT is measured (_Kin, legs straight, foot flat): LEG_REACH's 0.72 never binds there.
LEG_STRAIGHT = 0.675                               # m: root above the sole bottom, legs straight
LEG_SLACK = 0.02                                   # m
# TCRS multi-point IK (Perceptive BFM Sec. 3.2): every sole point of both feet as a target (stance W_STANCE, swing
# W_SWING) with the root height free but pulled to its reconstructed height (W_ROOT), damped least squares
# over legs + root; the solved root's offset is then eased (raised or lowered where needed, then smoothed) and the legs
# solved again with the root fixed there
W_STANCE, W_SWING, W_ROOT, IK_DAMP = 1.0, 0.5, 0.15, 1e-4
# TCRS collision repair: the shin (knee to ankle, SHIN_R thick, SHIN_PTS points) above the terrain by CLEAR in a
# swing, else that swing lifted like the sole clearance
SHIN_R, SHIN_PTS = 0.045, 6
CLEAR = 0.03                                       # m: least swing clearance above the terrain under the sole
FPS = 50
SEED = os.path.join(C.PROJ, 'bones-seed', 'x2_props_all')
OUT = os.path.join(H.ROOT, 'build', 'stairs', 'fit')


def use(scene):
    """Fit to that apparatus's stairs (STAIRS) from now on."""
    global RISERS, TOPS, LANDING, STAIR_Y, DIR, FLOOR, LEG_CONE
    st = STAIRS[scene]
    RISERS, TOPS, LANDING, STAIR_Y, DIR, FLOOR = st['risers'], st['tops'], st['landing'], st['y'], st['dir'], st['floor']
    LEG_CONE = st.get('leg_cone', False)


class _Kin:
    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(C.X2_XML)
        self.d = mujoco.MjData(self.m)
        self.ankle = [self.m.body('%s_ankle_roll_link' % s).id for s in ('left', 'right')]
        b = self.ankle
        self.sole = [[g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] == b[f]
                      and self.m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE] for f in range(2)]
        self.sole_local = [self.m.geom_pos[self.sole[f]].copy() for f in range(2)]     # sphere centres, ankle frame
        self.sole_r = float(self.m.geom_size[self.sole[0][0], 0])

    def shins(self, q):
        """Shin centreline points (T, 2, SHIN_PTS, 3), knee to ankle."""
        knee = [self.m.body('%s_knee_link' % s).id for s in ('left', 'right')]
        ank = [self.m.body('%s_ankle_pitch_link' % s).id for s in ('left', 'right')]
        u = np.linspace(0, 1, SHIN_PTS)[:, None]
        out = np.zeros((len(q), 2, SHIN_PTS, 3))
        for t in range(len(q)):
            self.d.qpos[:] = q[t]
            mujoco.mj_kinematics(self.m, self.d)
            for f in range(2):
                out[t, f] = (1 - u) * self.d.xpos[knee[f]] + u * self.d.xpos[ank[f]]
        return out

    def sole_at(self, pos, rot, f):
        """Sole sphere bottoms (T, 12, 3) of foot f for ankle poses pos (T, 3), rot (T, 3, 3)."""
        return pos[:, None] + np.einsum('tij,pj->tpi', rot, self.sole_local[f]) - [0.0, 0.0, self.sole_r]

    def feet(self, q):
        """Per frame and foot: ankle position (T, 2, 3), ankle rotation (T, 2, 3, 3), sole sphere bottoms (T, 2, 12, 3)."""
        T = len(q)
        pos, rot = np.zeros((T, 2, 3)), np.zeros((T, 2, 3, 3))
        sole = np.zeros((T, 2, len(self.sole[0]), 3))
        for t in range(T):
            self.d.qpos[:] = q[t]
            mujoco.mj_kinematics(self.m, self.d)
            for f in range(2):
                pos[t, f] = self.d.xpos[self.ankle[f]]
                rot[t, f] = self.d.xmat[self.ankle[f]].reshape(3, 3)
                sole[t, f] = self.d.geom_xpos[self.sole[f]] - [0, 0, self.m.geom_size[self.sole[f][0], 0]]
        return pos, rot, sole


def plants(sole, lift_min=None):
    """Stance plateaus of one foot's sole centre (T, 3): [(start, end)]. Runs slower than V_PLANT, merged across a gap
    of at most MERGE_GAP frames in which the foot moves less than MERGE_MOVE (a speed blip splits no stance); then kept
    if at least MIN_PLANT long, or at least MIN_STOP long with the foot really stopped (below V_STOP: a quick stair
    stance is still a stance). lift_min: also merged across a gap of any length in which the sole never rises more
    than this above the lower of its two plants (a pivot or slide on the floor, not a swing; routepass)."""
    v = np.linalg.norm(np.gradient(sole, 1 / FPS, axis=0), axis=1)
    runs = P._runs(v < V_PLANT, 1)
    merged = []
    for a, e in runs:
        pivot = (lift_min is not None and merged
                 and sole[merged[-1][1]:a, 2].max(initial=-np.inf) - min(sole[merged[-1][1] - 1, 2], sole[a, 2]) < lift_min)
        if merged and (pivot or (a - merged[-1][1] <= MERGE_GAP
                                 and np.linalg.norm(sole[a] - sole[merged[-1][1] - 1]) < MERGE_MOVE)):
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((a, e))
    return [(a, e) for a, e in merged if e - a >= MIN_PLANT or (e - a >= MIN_STOP and v[a:e].min() < V_STOP)]


def _smooth(x):
    return x * x * (3 - 2 * x)


def offset_curve(T, runs, offs):
    """(T, 3): offs[i] held over runs[i], eased between consecutive runs, held before the first / after the last."""
    o = np.zeros((T, 3))
    o[:runs[0][0]] = offs[0]
    for i, (a, e) in enumerate(runs):
        o[a:e] = offs[i]
        if i + 1 < len(runs):
            a1 = runs[i + 1][0]
            s = _smooth(np.linspace(0, 1, a1 - e + 2)[1:-1])
            o[e:a1] = offs[i] + s[:, None] * (offs[i + 1] - offs[i])
    o[runs[-1][1]:] = offs[-1]
    return o


def latch_curve(pos, runs):
    """(T, 3) TCRS stance latching (Perceptive BFM Sec. 3.2) for one foot's ankle path pos (T, 3): through each plant
    the foot is pinned where it stands at mid-plant (pos[mid] - pos[t]), eased between plants (smoothstep), held before
    the first / after the last."""
    T = len(pos)
    L = np.zeros((T, 3))
    if not runs:
        return L
    ends = []
    for a, e in runs:
        m_ = (a + e) // 2
        L[a:e] = pos[m_] - pos[a:e]
        ends.append((L[a].copy(), L[e - 1].copy()))
    L[:runs[0][0]] = ends[0][0]
    for i, ((a, e), (a1, _)) in enumerate(zip(runs, runs[1:])):
        sm = _smooth(np.linspace(0, 1, a1 - e + 2)[1:-1])
        L[e:a1] = ends[i][1] + sm[:, None] * (ends[i + 1][0] - ends[i][1])
    L[runs[-1][1]:] = ends[-1][1]
    return L


def lift_profile(n, min_rise=0):
    """(n,) 0..1..0 over a swing of n frames: up over the first LIFT_EDGE of it (at least min_rise frames, at most
    half the swing), held, down over the last."""
    s = np.linspace(0, 1, n + 2)[1:-1]
    edge = min(max(LIFT_EDGE, min_rise / max(n, 1)), 0.5)
    return _smooth(np.clip(np.minimum(s, 1 - s) / edge, 0, 1))


def hold_curve(T, runs, offs):
    """(T, 3): offs[i] from the start of runs[i] until the next plant starts (a swinging foot keeps its last offset)."""
    o = np.zeros((T, 3))
    o[:] = offs[0]
    for (a, _), off in zip(runs, offs):
        o[a:] = off
    return o


def _turn(q, yaw, pivot):
    R = Rot.from_euler('z', yaw)
    q = q.copy()
    q[:, :2] = R.apply(np.c_[q[:, :2] - pivot, np.zeros(len(q))])[:, :2] + pivot
    q[:, 3:7] = (R * Rot.from_quat(q[:, [4, 5, 6, 3]])).as_quat()[:, [3, 0, 1, 2]]
    return q


def _ground(course, pts):
    """Walkable terrain height under points (..., 3) (obstacle cells - rails, posts, walls - read as the floor: a foot
    is never placed or lifted against those)."""
    xy = pts[..., :2].reshape(-1, 2)
    h = np.where(course.cls(xy) == H.OBSTACLE, 0.0, course.height_min(xy))
    return h.reshape(pts.shape[:-1])


class MultiPointIK:
    """TCRS multi-point IK (module constants): sole points of both feet, legs + root height."""

    def __init__(self, kin):
        self.k, m = kin, kin.m
        self.d = mujoco.MjData(m)
        legs = [[m.joint(j).id for j in SG.LEGS[s]] for s in SG.SIDES]
        self.dofs = [[m.jnt_dofadr[j] for j in js] for js in legs]
        self.adr = [[m.jnt_qposadr[j] for j in js] for js in legs]
        self.lim = [m.jnt_range[js] for js in legs]
        self.cols = [2] + self.dofs[0] + self.dofs[1]                    # root height + legs (TCRS: the root height)

    def solve(self, q, pos, rot, w, root=True, iters=IK_ITERS):
        """q (38,) with the root and upper body set; pos / rot the ankle targets (2, 3) / (2, 3, 3); w (2,) per-foot
        weights. root: root height free (pulled to q's). Returns q, ankle miss (2,), sole-point miss (2,)."""
        m, d, k = self.k.m, self.d, self.k
        d.qpos[:] = q
        root0 = q[:3].copy()
        cols = self.cols if root else self.cols[1:]
        tgt = [pos[f] + k.sole_local[f] @ rot[f].T for f in range(2)]                  # (P, 3) sole sphere centres
        jp = np.zeros((3, m.nv))
        for _ in range(iters):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            E, J = [], []
            for f in range(2):
                for g, t in zip(k.sole[f], tgt[f]):
                    mujoco.mj_jacGeom(m, d, jp, None, g)
                    E.append(w[f] * (t - d.geom_xpos[g]))
                    J.append(w[f] * jp[:, cols])
            if root:
                E.append(W_ROOT * (root0[2:] - d.qpos[2:3]))
                J.append(W_ROOT * np.eye(1, len(cols)))
            E, J = np.concatenate(E), np.vstack(J)
            dq = J.T @ np.linalg.solve(J @ J.T + IK_DAMP * np.eye(len(E)), E) if len(E) < len(cols) else                 np.linalg.solve(J.T @ J + IK_DAMP * np.eye(len(cols)), J.T @ E)
            if root:
                d.qpos[2] += dq[0]
                dq = dq[1:]
            for f in range(2):
                a = self.adr[f]
                d.qpos[a] = np.clip(d.qpos[a] + dq[6 * f:6 * f + 6], *self.lim[f].T)
        mujoco.mj_kinematics(m, d)
        miss = np.array([np.linalg.norm(pos[f] - d.xpos[k.ankle[f]]) for f in range(2)])
        pmiss = np.array([np.linalg.norm(tgt[f] - d.geom_xpos[k.sole[f]], axis=1).max() for f in range(2)])
        return d.qpos.copy(), miss, pmiss


def _ease_root(base, solved):
    """The root offset the multi-point IK chose, eased over ROOT_SMOOTH: lowering (a leg short of reach) and raising
    (a leg folded past the knee) each first widened by half the window (running min / max), so the smoothed change is
    at least what each frame needed."""
    h = ROOT_SMOOTH // 2
    raw = solved[:, 2] - base[:, 2]
    lo, hi = np.pad(np.minimum(raw, 0.0), h, mode='edge'), np.pad(np.maximum(raw, 0.0), h, mode='edge')
    dz = (np.min([lo[s:s + len(base)] for s in range(2 * h + 1)], 0)
          + np.max([hi[s:s + len(base)] for s in range(2 * h + 1)], 0))
    k = np.ones(ROOT_SMOOTH) / ROOT_SMOOTH
    out = base.copy()
    out[:, 2] += np.convolve(np.pad(dz, (h, ROOT_SMOOTH - 1 - h), mode='edge'), k, 'valid')
    return out


def _near_max(course, pts, heading=None):
    """(N,) highest ground within EDGE_R of each point: ahead of it (heading +-45 deg, the edges a toe can catch), or
    all around when heading is None."""
    hmax = _ground(course, pts)
    angles = (np.linspace(0, 2 * np.pi, 8, endpoint=False) if heading is None
              else heading + np.radians([-45.0, -22.5, 0.0, 22.5, 45.0]))
    for a in angles:
        for r in (0.5 * EDGE_R, EDGE_R):
            hmax = np.maximum(hmax, _ground(course, pts + [r * np.cos(a), r * np.sin(a), 0.0]))
    return hmax


def _edge_buffer(rise, stride):
    """Clearance kept over a rising edge of height rise, for a swing of horizontal length stride (module constants)."""
    return np.maximum(EDGE_CLEAR * np.clip(1 - rise / RISE_FULL, 0, 1) * min(1.0, stride / STRIDE_FULL), EDGE_MIN)


def _required_z(course, pts, stride, heading):
    """(...) least sole height at points (..., 3) of a swing of length stride heading `heading` (rad): CLEAR over the
    ground, or the edge buffer over a sharp rising edge ahead."""
    flat = pts.reshape(-1, 3)
    h, hmax = _ground(course, flat), _near_max(course, flat, heading)
    top = hmax + _edge_buffer(hmax - h, stride)
    return np.where(hmax - h > EDGE_STEP, np.maximum(h + CLEAR, top), h + CLEAR).reshape(pts.shape[:-1])


def _swing_cost(D, path, rot, local, r, course, t_knot):
    """MPPI cost of knot displacements D (N, K, 3) for a swing whose eased ankle path is path (L, 3)."""
    N, K, _ = D.shape
    L = len(path)
    tt = np.linspace(0, 1, L)
    knots_t = np.concatenate([[0.0], t_knot, [1.0]])
    Dz = np.concatenate([np.zeros((N, 1, 3)), D, np.zeros((N, 1, 3))], 1)          # endpoints fixed
    disp = np.stack([np.stack([np.interp(tt, knots_t, Dz[n, :, i]) for i in range(3)], -1) for n in range(N)])
    pts = (path[None, :, None] + disp[:, :, None] + np.einsum('lij,pj->lpi', rot, local)[None]
           - np.array([0.0, 0.0, r]))                                              # (N, L, P, 3) sole bottoms
    flat = pts.reshape(-1, 3)
    h = _ground(course, flat).reshape(pts.shape[:3])
    d = path[-1, :2] - path[0, :2]
    stride, heading = float(np.linalg.norm(d)), float(np.arctan2(d[1], d[0]))
    hmax = _near_max(course, flat, heading).reshape(pts.shape[:3])
    clr = np.maximum(h + CLEAR - pts[..., 2], 0.0) ** 2
    edge = np.where(hmax - h > EDGE_STEP,
                    np.maximum(hmax + _edge_buffer(hmax - h, stride) - pts[..., 2], 0.0) ** 2, 0.0)
    ramp = np.minimum(tt, 1 - tt)[None, :, None] * 2                               # no clearance demand at the ends
    smooth = (np.diff(Dz, 2, axis=1) ** 2).sum((1, 2))
    return (W_REF * (D ** 2).sum((1, 2)) + W_SMOOTH * smooth
            + W_CLEAR * (clr * ramp).sum((1, 2)) + W_EDGE * (edge * ramp).sum((1, 2)))


def optimize_swings(pos, rot, o, runs, course, kin, rng):
    """TCRS swing optimization (module constants): adds each swing's optimized displacement to o in place."""
    t_knot = np.linspace(0, 1, SWING_KNOTS + 2)[1:-1]
    for f in range(2):
        local = kin.sole_local[f]
        for (_, e0), (a1, _) in zip(runs[f], runs[f][1:]):
            if a1 - e0 < 4:
                continue
            path, R = pos[e0:a1, f] + o[e0:a1, f], rot[e0:a1, f]
            mean = np.zeros((SWING_KNOTS, 3))
            best, best_c = mean, _swing_cost(mean[None], path, R, local, kin.sole_r, course, t_knot)[0]
            for _ in range(MPPI_ITERS):
                D = mean[None] + MPPI_SIGMA * rng.standard_normal((MPPI_SAMPLES, SWING_KNOTS, 3))
                c = _swing_cost(D, path, R, local, kin.sole_r, course, t_knot)
                if c.min() < best_c:
                    best, best_c = D[c.argmin()], c.min()
                lam = max(1e-9, 0.1 * (c.mean() - c.min()))
                w = np.exp(-(c - c.min()) / lam)
                mean = (w[:, None, None] * D).sum(0) / w.sum()
            tt = np.linspace(0, 1, a1 - e0)
            knots_t = np.concatenate([[0.0], t_knot, [1.0]])
            Dz = np.concatenate([np.zeros((1, 3)), best, np.zeros((1, 3))])
            o[e0:a1, f] += np.stack([np.interp(tt, knots_t, Dz[:, i]) for i in range(3)], -1)


def root_height(q, pos, rot, o, runs, course, kin):
    """TCRS root reconstruction (module constants): (T,) root height."""
    T = len(q)
    w = np.zeros((T, 2))
    for f in range(2):
        for a, e in runs[f]:
            w[a:e, f] = 1.0
    k = np.ones(5) / 5
    w = np.stack([np.convolve(np.pad(w[:, f], 2, mode='edge'), k, 'valid') for f in range(2)], 1)
    z = np.empty(T)
    for t in range(T):
        num = den = 0.0
        lo = np.inf
        for f in range(2):
            raw = kin.sole_at(pos[t:t + 1, f], rot[t:t + 1, f], f)[0]
            new = kin.sole_at(pos[t:t + 1, f] + o[t:t + 1, f], rot[t:t + 1, f], f)[0]
            ground = float(np.median(_ground(course, new)))
            num += w[t, f] * (ground + q[t, 2] - raw[:, 2].min())
            den += w[t, f]
            if w[t, f] > 0.5:
                reach = LEG_REACH
                if LEG_CONE:
                    hd = 2 * math.atan2(q[t, 6], q[t, 3])           # root yaw (quaternion w, z; tilt is small)
                    dd = float(np.dot(new[:, :2].mean(0) - q[t, :2], [math.cos(hd), math.sin(hd)]))
                    reach = math.sqrt(max(LEG_STRAIGHT ** 2 - dd ** 2, 0.0)) - LEG_SLACK
                lo = min(lo, ground + reach - LEG_REACH)
        if den > 0.1:
            z[t] = min(num / den, lo + LEG_REACH)
        else:                                      # flight / weak support: the raw vertical profile from the last frame
            z[t] = (z[t - 1] + q[t, 2] - q[t - 1, 2]) if t else q[t, 2] + o[t].mean(0)[2]
    kk = np.ones(ROOT_SMOOTH) / ROOT_SMOOTH
    return np.convolve(np.pad(z, (ROOT_SMOOTH // 2, ROOT_SMOOTH - 1 - ROOT_SMOOTH // 2), mode='edge'), kk, 'valid')


def apply_offsets(q, pos, rot, runs, offs, turns, course, kin, ik, max_lift=MAX_LIFT, seed=0, lift_cap=False,
                  shuffle=0.0, min_rise=0, two_seed=False):
    """Move each foot plant by its offset and turn it by its rotation, and solve the motion around that. Swings are
    optimized as in TCRS (optimize_swings) and the root height reconstructed from the supporting feet
    (root_height), both from Perceptive BFM (arXiv 2606.08059, Sec. 3.2).

    q (T, 38) the captured motion, pos / rot (T, 2, 3) / (T, 2, 3, 3) its ankle poses, runs[f] the plants of foot f,
    offs[f][i] (3,) the move of plant i, turns[f][i] (3,) its rotation (rotation vector, world frame, applied before
    the captured ankle rotation). Both are held through a plant and eased between plants; the root follows the feet's
    plant offsets as they bear weight (a swinging foot counts at its last plant), smoothed over ROOT_SMOOTH frames.
    Legs by IK; a swing whose sole (as solved) passes closer than CLEAR over the terrain is lifted by the shortfall (at
    most max_lift a pass; fast rise, held, fast fall) and the legs solved again. Before that, once, a target sole
    more than SINK_TOL below the terrain in any frame (a landing or stance on a slope) is raised by its depth, eased
    over SINK_EASE frames. lift_cap: max_lift bounds a swing's lift over all passes, not each pass (a swing the IK
    cannot lift is otherwise lifted again every pass, and the free root sinks toward it: routepass 2026-10-27).
    shuffle: a swing shorter than this (a step in place) keeps plain CLEAR, no edge buffer. min_rise: a lift rises
    over at least this many frames (a short swing's lift otherwise rises in 3-4 frames: knee spikes). two_seed: each
    frame's legs are solved from the previous frame's and from the captured legs, the lower cost kept (a leg warm
    start alone can stay in a crouch the free root sinks into). Returns (qpos, IK miss
    (T, 2))."""
    T = len(q)
    o = np.stack([offset_curve(T, runs[f], offs[f]) for f in range(2)], 1)      # (T, 2, 3)
    if LATCH:
        o += np.stack([latch_curve(pos[:, f], runs[f]) for f in range(2)], 1)
    tv = np.stack([offset_curve(T, runs[f], turns[f]) for f in range(2)], 1)    # (T, 2, 3)
    rot = np.einsum('tfij,tfjk->tfik', Rot.from_rotvec(tv.reshape(-1, 3)).as_matrix().reshape(T, 2, 3, 3), rot)
    hold = np.stack([hold_curve(T, runs[f], offs[f]) for f in range(2)], 1).mean(1)
    k = np.ones(ROOT_SMOOTH) / ROOT_SMOOTH
    pad = np.pad(hold, ((ROOT_SMOOTH // 2, ROOT_SMOOTH - 1 - ROOT_SMOOTH // 2), (0, 0)), mode='edge')
    base = q.copy()
    base[:, :3] += np.stack([np.convolve(pad[:, i], k, 'valid') for i in range(3)], 1)
    optimize_swings(pos, rot, o, runs, course, kin, np.random.default_rng(seed))
    # no target sole below the terrain in any frame (a landing or stance on a slope): raise by the depth, eased so the
    # raise covers every frame that needs it (widened, then smoothed with a window of the same width)
    win = np.hanning(2 * SINK_EASE + 3)[1:-1]
    win /= win.sum()
    for f in range(2):
        sole = kin.sole_at(pos[:, f] + o[:, f], rot[:, f], f)
        depth = -(sole[..., 2] - _ground(course, sole)).min(1)
        need = np.maximum(depth - SINK_TOL, 0.0)
        if need.any():
            need = np.max([np.roll(np.pad(need, SINK_EASE), s)[SINK_EASE:-SINK_EASE]
                           for s in range(-SINK_EASE, SINK_EASE + 1)], 0)
            o[:, f, 2] += np.convolve(np.pad(need, SINK_EASE, mode='edge'), win, 'valid')
    base[:, 2] = root_height(q, pos, rot, o, runs, course, kin)
    stance = np.zeros((T, 2), bool)
    for f in range(2):
        for a, e in runs[f]:
            stance[a:e, f] = True
    wts = np.where(stance, W_STANCE, W_SWING)
    mp = MultiPointIK(kin)

    def solve(prev, t, root):
        """Legs from prev's (and, two_seed, from the capture's); (q, miss) of the better."""
        ref = base if root else fixed
        best = None
        for legs in ((prev[7:19], q[t, 7:19]) if two_seed else (prev[7:19],)):
            sol, m_, pm = mp.solve(np.r_[ref[t, :7], legs, ref[t, 19:]], pos[t] + o[t], rot[t], wts[t], root=root)
            c = float((wts[t] * pm).sum() + (W_ROOT * abs(sol[2] - base[t, 2]) if root else 0.0)
                      + 1e-3 * np.abs(sol[7:19] - q[t, 7:19]).sum())
            if best is None or c < best[0]:
                best = (c, sol, m_)
        return best[1], best[2]

    total = {}                                                                  # (f, e0): lift so far
    for _ in range(CLEAR_PASSES):
        free = base.copy()
        fixed = base
        for t in range(T):
            free[t] = solve(free[t - 1] if t else base[t], t, True)[0]                # legs warm-started
        fixed = _ease_root(base, free)
        out, miss = fixed.copy(), np.zeros((T, 2))
        for t in range(T):
            out[t], miss[t] = solve(out[t - 1] if t else fixed[t], t, False)
        _, _, solved = kin.feet(out)
        shin = kin.shins(out) - [0.0, 0.0, SHIN_R]
        lifted = False
        for f in range(2):
            for (_, e0), (a1, _) in zip(runs[f], runs[f][1:]):
                sp, sh = solved[e0:a1, f], shin[e0:a1, f]
                d = (pos[a1 - 1, f] + o[a1 - 1, f] - pos[e0, f] - o[e0, f])[:2]
                stride, heading = float(np.linalg.norm(d)), float(np.arctan2(d[1], d[0]))
                req = (_required_z(course, sp, stride, heading) if stride >= shuffle
                       else _ground(course, sp.reshape(-1, 3)).reshape(sp.shape[:2]) + CLEAR)
                short = np.concatenate([req - sp[..., 2],                                       # sole: inflated
                                        _ground(course, sh.reshape(-1, 3)).reshape(sh.shape[:2]) + CLEAR - sh[..., 2]], 1)
                ramp = np.minimum(np.linspace(0, 1, a1 - e0), np.linspace(1, 0, a1 - e0)) * 2   # ends land / lift off
                need = min(float((short * np.minimum(ramp * 4, 1.0)[:, None]).max()), max_lift)
                if lift_cap:
                    need = min(need, max_lift - total.get((f, e0), 0.0))
                if need > 0.002:
                    o[e0:a1, f, 2] += need * lift_profile(a1 - e0, min_rise)
                    total[(f, e0)] = total.get((f, e0), 0.0) + need
                    lifted = True
        if not lifted:
            break
    return out, miss


def place(q, kin=None):
    """Captured climb (T, 38) -> (placed qpos, runs, levels, rise) or (None, reason): turned to climb along DIR * x,
    centred across the stairs, its first step-up onto the middle of tread 1. Not yet fitted (fit / tread_targets)."""
    kin = kin or _Kin()
    _, _, sole = kin.feet(q)
    centre = sole.mean(2)                                                   # (T, 2, 3) sole centres
    runs = [plants(centre[:, f]) for f in range(2)]
    pz = [np.array([sole[a:e, f, :, 2].min(-1).mean() for a, e in runs[f]]) for f in range(2)]
    allz = np.sort(np.concatenate(pz))
    groups = np.split(allz, np.nonzero(np.diff(allz) > LEVEL_GAP)[0] + 1)      # plant heights, one group per level
    means = np.array([g.mean() for g in groups])
    rise = float(np.median(np.diff(means))) if len(means) > 1 else 0.0
    if rise < 0.05:
        return None, 'no climb found (capture rise %.3f m)' % rise
    lvl = [np.round((pz[f] - means[0]) / rise).astype(int) for f in range(2)]
    # place: climb along DIR * x, centred across the stairs
    up = [(a, centre[(a + e) // 2, f]) for f in range(2) for (a, e), k in zip(runs[f], lvl[f]) if k >= 1]
    up.sort(key=lambda r: r[0])
    d = up[-1][1][:2] - up[0][1][:2]
    q = _turn(q, (0.0 if DIR > 0 else math.pi) - math.atan2(d[1], d[0]), q[0, :2])
    pos, rot, sole = kin.feet(q)
    centre = sole.mean(2)
    q[:, 1] += STAIR_Y - np.mean([centre[(a + e) // 2, f, 1] for f in range(2) for a, e in runs[f]])
    first = min(((a, centre[(a + e) // 2, f, 0]) for f in range(2) for (a, e), k in zip(runs[f], lvl[f]) if k == 1),
                default=None)
    if first is None:
        return None, 'no plant on the first step'
    q[:, 0] += (RISERS[0] + RISERS[1]) / 2 - first[1]                      # first step-up onto the middle of tread 1
    return q, runs, lvl, rise


def tread_targets(q, runs, lvl, course, kin, pos=None, rot=None, sole=None):
    """Plant targets of a placed climb: (offs, turns) per foot, as apply_offsets takes them. lvl[f][i] the level of
    plant i (0 floor, 1..len(TOPS) treads, above: landing)."""
    if pos is None:
        pos, rot, sole = kin.feet(q)
    centre = sole.mean(2)
    # targets per plant: the foot turned straight up the stairs (a turned-out foot is longer than a tread), its sole
    # centre onto the middle of its tread (or kept on the floor / pushed onto the landing), sole bottom onto its top
    # distances along the climb are in u = DIR * x (u grows up the stairs); up: the heading straight up the stairs
    up = 0.0 if DIR > 0 else math.pi
    riser = [DIR * r for r in RISERS]
    offs, yaws = [], []
    for f in range(2):
        o_f, y_f, last_du = [], [], 0.0
        for (a, e), k in zip(runs[f], lvl[f]):
            m_ = (a + e) // 2
            c = centre[m_, f]
            yaw = math.atan2(rot[m_, f, 1, 0], rot[m_, f, 0, 0])
            turn = (up - yaw + math.pi) % (2 * math.pi) - math.pi
            au = DIR * pos[m_, f, 0]                                        # ankle u; straight foot: sole centre at +SOLE_X
            if 1 <= k <= len(TOPS):                                         # toe TOE_GAP short of the next riser
                du, top = riser[k] - TOE_GAP - TOE - au, TOPS[k - 1]
            elif k > len(TOPS):
                du, top = max(au + last_du, riser[-1] + LANDING_IN - SOLE_X) - au, LANDING
            else:                                                           # floor: keep, toe FLOOR_GAP off the first riser
                du, top = min(0.0, (riser[0] - FLOOR_GAP) - (DIR * c[0] - SOLE_X + TOE)), FLOOR
            last_du = du
            dx = DIR * du
            if k >= 1:                                                      # the surface where the straight foot lands
                s = sole[m_, f, :, :2] - pos[m_, f, :2]
                s = Rot.from_euler('z', turn).apply(np.c_[s, np.zeros(len(s))])[:, :2] + pos[m_, f, :2] + [dx, 0.0]
                top = float(np.median(course.height_min(s)))
            o_f.append(np.array([dx, 0.0, top - sole[a:e, f, :, 2].min(-1).mean()]))
            y_f.append(np.array([0.0, 0.0, turn if k >= 1 else 0.0]))
        offs.append(o_f)
        yaws.append(y_f)
    return offs, yaws


def fit(q, course=None, kin=None, ik=None):
    """Captured climb (T, 38) -> (fitted qpos, info). See the module doc."""
    course = course or H.Course('cad')
    kin, ik = kin or _Kin(), ik or SG.LegIK()
    placed = place(q, kin)
    if placed[0] is None:
        return None, dict(reason=placed[1])
    q, runs, lvl, rise = placed
    pos, rot, sole = kin.feet(q)
    offs, yaws = tread_targets(q, runs, lvl, course, kin, pos, rot, sole)
    out, miss = apply_offsets(q, pos, rot, runs, offs, yaws, course, kin, ik)
    info = dict(capture_rise=round(rise, 3), levels=int(max(l.max() for l in lvl)), plants=sum(len(r) for r in runs),
                ik_miss_max_cm=round(float(miss.max() * 100), 1), ik_miss_p95_cm=round(float(np.percentile(miss, 95) * 100), 1),
                max_shift_cm=round(float(np.abs(np.concatenate([np.array(x) for x in offs])[:, :2]).max() * 100), 1))
    return out, info


def check(q, course, kin=None):
    """Per stair plant: share of sole spheres on the surface (within 1.5 cm), deepest sinking; swing clearance."""
    kin = kin or _Kin()
    _, _, sole = kin.feet(q)
    rows, clear = [], []
    for f in range(2):
        runs = plants(sole[:, f].mean(1))
        for a, e in runs:
            S = sole[(a + e) // 2, f]
            g = course.height_min(S[:, :2])
            gap = S[:, 2] - g
            if g.max() > 0.1:
                rows.append((float((np.abs(gap) < 0.015).mean()), float(-gap.min())))
        for (a0, e0), (a1, _) in zip(runs, runs[1:]):
            S = sole[e0:a1, f]
            gap = S[..., 2] - course.height_min(S[..., :2].reshape(-1, 2)).reshape(S.shape[:2])
            clear.append(float(gap.min()))
    sup = np.array([r[0] for r in rows]) if rows else np.zeros(1)
    return dict(stair_plants=len(rows), sole_on_tread_mean=round(float(sup.mean()), 2), sole_on_tread_min=round(float(sup.min()), 2),
                deepest_cm=round(100 * max([r[1] for r in rows] or [0]), 1),
                swing_clear_min_cm=round(100 * min(clear or [0]), 1))


def main():
    os.makedirs(OUT, exist_ok=True)
    course, kin, ik = H.Course('cad'), _Kin(), SG.LegIK()
    report = {}
    for path in sorted(glob.glob(os.path.join(SEED, 'stairs_climbing_up*.csv'))):
        name = os.path.basename(path)[:-4]
        q, info = fit(P.load_csv(path), course, kin, ik)
        if q is None:
            report[name] = info
            print('%-46s skip: %s' % (name, info['reason']))
            continue
        info.update(check(q, course, kin))
        np.save(os.path.join(OUT, name + '.npy'), q)
        report[name] = info
        print('%-46s %s' % (name, info), flush=True)
    json.dump(report, open(os.path.join(OUT, 'report.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
