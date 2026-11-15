"""Kinematic stepping over the height map: a walk along any route where each step lands on a safe foothold and each
swing lifts the foot clear of the terrain under it. No physics: footsteps -> foot and pelvis trajectories -> leg IK.

  footsteps  alternate feet along the path, STEP apart and +-HALF_WIDTH off the centreline, feet along the path
             heading, steered toward the path LOOKAHEAD ahead and turning at most MAX_TURN per step (corners are walked
             as arcs, never turned in place, feet always side by side about the walker's heading); a foot
             climbs at most MAX_RISE per step, so stairs are taken step-to (each foot onto each tread); each moved to the nearest safe foothold (P2) within a small window (shorter rather than longer)
  swing      lift straight up first, then travel; the sole stays CLEAR above the highest terrain under the whole
             foot (footprint + toe/heel margin) at every point of the swing, then comes straight down
  pelvis     over the feet, swaying toward the stance foot; height PELVIS_H above the lower foot, less STEP_DROP of the
             height between the feet (so the lower leg is not over-stretched on stairs); upright torso; heading = the feet's mean heading
  legs       damped least-squares IK on each leg (6 joints) to the ankle pose; arms and waist from a relaxed walk

    g = generate(H.Course('cad'), H.route_waypoints('mobility'))
    g.qpos (T, 38) at 50 Hz, g.steps (list of footsteps), g.miss (T, 2) IK error per foot
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import joblib
import mujoco
import numpy as np

from pipeline import clipdb as C
from pipeline import footholds as F
from pipeline import hmap as H

FPS = 50
STEP = 0.22                 # m nominal step (one foot to the other along the path)
HALF_WIDTH = 0.10           # m sole centre off the path centreline
MAX_TURN = math.radians(25)   # largest heading change per step: turns are walked as arcs, never in place
CORNER = 0.5                # m: route corners rounded over this distance each side (arcs about this radius)
LOOKAHEAD = 0.25            # m: the walker steers toward the path this far past its nearest point
MIN_STRIDE = 0.3            # share of STEP kept on the sharpest corners: turn while still stepping forward
MAX_RISE = 0.21             # m a foot climbs or drops per step: more and it steps to beside the other foot (stairs)
CLEAR = 0.06                # m sole clearance over the terrain during a swing
SWING_T, DOUBLE_T = 0.55, 0.20      # s per swing (+ RISE_T per 0.1 m of height change), double support
RISE_T = 0.12
PELVIS_H = 0.655            # m pelvis above the lower sole walking on the level (standing: 0.675): knees ~20 deg
STEP_DROP = 0.20            # pelvis lowers by this share of the height between the feet (stairs: the lower leg would
                            # otherwise lock straight)
WAIST_PITCH = 0.05          # rad: upright torso
LEG_REACH = 0.655           # m pelvis-to-sole distance a planted leg may span (straight: ~0.68): the pelvis dips when
                            # the feet are spread so the trailing knee keeps a bend
SWAY = 0.03                 # m toward the stance foot
LEGS = {s: ['%s_%s_joint' % (s, j) for j in ('hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch', 'ankle_roll')]
        for s in ('left', 'right')}
SIDES = ('left', 'right')


@dataclass
class Step:
    side: int               # 0 left, 1 right
    xy: np.ndarray          # sole centre
    z: float                # terrain height under it
    yaw: float
    safe: bool
    normal: np.ndarray = None   # ground plane normal under the foot (the sole is tilted onto it)
    t_lift: float = 0.0
    t_land: float = 0.0


@dataclass
class Gait:
    qpos: np.ndarray
    steps: list
    foot: np.ndarray        # (T, 2, 4) sole centre xyz + yaw targets
    miss: np.ndarray        # (T, 2) IK position error (m)
    terrain: str


def _path_at(poses, s_cum, s):
    i = int(np.clip(np.searchsorted(s_cum, s), 0, len(poses) - 1))
    return poses[i]


def _snap(course, nominal, t, n, yaw, foot, ahead=0.10):
    """The safe foothold nearest the nominal one: back up to 20 cm (a shorter step), forward up to `ahead`, sideways
    +-6 cm on a 5 mm / 1 cm grid, foot yaw within +-11 deg (a straight foot fits a narrow tread on a skewed path).
    Returns the chosen index, candidates, yaw offsets, along-path offsets and whether it is safe."""
    da = np.arange(-0.20, ahead + 1e-4, 0.005)
    db = np.arange(-0.06, 0.0601, 0.01)
    dy = np.radians([0.0, -5.0, 5.0, -11.0, 11.0])
    A, B, Y = [g.ravel() for g in np.meshgrid(da, db, dy, indexing='ij')]
    cand = nominal + A[:, None] * t + B[:, None] * n
    q = F.quality(course, cand, yaw + Y, foot)
    cost = np.hypot(A * np.where(A < 0, 1.0, 2.0), B * 1.5) + 0.2 * np.abs(Y)
    ok = bool(q['safe'].any())
    if not ok and ahead < 0.32:                                      # nothing near: reach further, e.g. over a 4x4
        return _snap(course, nominal, t, n, yaw, foot, ahead=0.32)
    k = int(np.argmin(np.where(q['safe'], cost, np.inf))) if ok else int(np.argmin(cost))
    _snap.normal = q['normal'][k] if ok else np.array([0, 0, 1.0])  # an unsafe spot's fitted plane means nothing
    return k, cand, Y, A, ok


def plan_footsteps(course, waypoints, foot=F.FootSpec()):
    """Alternating footsteps along the path, each snapped to the nearest safe foothold."""
    poses = H.resample_path(waypoints, 0.01, turn_blend=CORNER)
    s_cum = np.arange(len(poses)) * 0.01
    x, y, yaw = poses[0]
    n = np.array([-math.sin(yaw), math.cos(yaw)])
    steps = []
    for side in (0, 1):                                              # stand with both feet beside the start
        xy = np.array([x, y]) + (HALF_WIDTH if side == 0 else -HALF_WIDTH) * n
        steps.append(Step(side, xy, float(course.height_min([xy])[0]), yaw, True, np.array([0, 0, 1.0])))
    side = 0
    wrap = lambda a: (a + math.pi) % (2 * math.pi) - math.pi
    walker, heading = np.array([x, y]), yaw                         # body centre and heading between the feet
    s_near = 0.0
    for _ in range(2000):
        if s_near >= s_cum[-1] - 0.05:
            break
        # pure pursuit: steer toward the path LOOKAHEAD past the nearest point, at most MAX_TURN per step
        lo = int(s_near / 0.01)
        near = lo + int(np.argmin(np.linalg.norm(poses[lo:lo + 80, :2] - walker, axis=1)))
        s_near = near * 0.01
        tgt = _path_at(poses, s_cum, min(s_near + LOOKAHEAD, s_cum[-1]))[:2]
        want = wrap(math.atan2(*(tgt - walker)[::-1]) - heading)
        turn = float(np.clip(want, -MAX_TURN, MAX_TURN))
        heading += turn
        stride = STEP * float(np.clip(1.2 - abs(want) / (2 * MAX_TURN), MIN_STRIDE, 1.0))   # sharp: short strides
        stride = min(stride, max(np.linalg.norm(tgt - walker), 0.05))
        pyaw = heading
        t = np.array([math.cos(pyaw), math.sin(pyaw)])
        n = np.array([-math.sin(pyaw), math.cos(pyaw)])
        centre = walker + stride * t
        nominal = centre + (HALF_WIDTH if side == 0 else -HALF_WIDTH) * n
        own = [st for st in steps if st.side == side][-1]
        k, cand, Y, A, ok = _snap(course, nominal, t, n, pyaw, foot)
        z_new = float(course.height_min([cand[k]])[0])
        if abs(z_new - own.z) > MAX_RISE:                              # too big a climb for one leg: step to beside
            other = [st for st in steps if st.side != side][-1]        # the other foot instead, same level
            nb = np.array([-math.sin(other.yaw), math.cos(other.yaw)])
            nominal = other.xy + (2 * HALF_WIDTH if side == 0 else -2 * HALF_WIDTH) * nb
            t = np.array([math.cos(other.yaw), math.sin(other.yaw)])
            pyaw = other.yaw
            k, cand, Y, A, ok = _snap(course, nominal, t, nb, pyaw, foot, ahead=0.06)
            z_new = float(course.height_min([cand[k]])[0])
            heading -= turn
        else:
            walker = cand[k] - (HALF_WIDTH if side == 0 else -HALF_WIDTH) * n   # re-centre on where the foot went
        steps.append(Step(side, cand[k], z_new, float(pyaw + Y[k]), ok, _snap.normal.copy()))
        side = 1 - side
    other = steps[-1].side
    last = steps[-1]
    n = np.array([-math.sin(last.yaw), math.cos(last.yaw)])
    xy = last.xy + (-2 * HALF_WIDTH if other == 0 else 2 * HALF_WIDTH) * n   # close the stance: feet side by side
    steps.append(Step(1 - other, xy, float(course.height_min([xy])[0]), last.yaw, True, last.normal.copy()))
    return steps


def swing_curve(course, p0, p1, yaw0, yaw1, n, foot=F.FootSpec(), clear=CLEAR):
    """(n, 4) x, y, z (sole), yaw for one swing from p0 to p1 (xyz sole centres)."""
    u = np.linspace(0, 1, n)
    h = np.clip((u - 0.15) / 0.7, 0, 1)                              # horizontal travel after the lift, before the drop
    h = h * h * (3 - 2 * h)
    xy = p0[:2] + (p1[:2] - p0[:2]) * h[:, None]
    yaw = yaw0 + ((yaw1 - yaw0 + math.pi) % (2 * math.pi) - math.pi) * h
    # highest terrain under the whole foot (+2 cm margin) along the way, then a smooth envelope over it
    big = F.FootSpec(length=foot.length + 0.04, width=foot.width + 0.03, heel_overhang=0.0, edge=0.0, n_l=9, n_w=5)
    off = big.offsets()
    top = np.empty(n)
    for i in range(n):
        c, s = math.cos(yaw[i]), math.sin(yaw[i])
        pts = xy[i] + np.column_stack([c * off[:, 0] - s * off[:, 1], s * off[:, 0] + c * off[:, 1]])
        top[i] = course.height(pts).max()
    need = top + clear
    k = max(1, n // 10)                                              # widen each requirement, then smooth it
    need = np.array([need[max(0, i - k):i + k + 1].max() for i in range(n)])
    base = p0[2] + (p1[2] - p0[2]) * h + clear * np.sin(math.pi * u) ** 0.7
    z = np.maximum(base, need * np.sin(math.pi * np.clip(u, 0, 1)) ** 0.15 + base * (1 - np.sin(math.pi * u) ** 0.15))
    z[0], z[-1] = p0[2], p1[2]
    # endpoints: straight up / straight down, never below the end heights
    z[:k] = np.maximum(z[:k], p0[2] + (z[k] - p0[2]) * (u[:k] / u[k]))
    z[-k:] = np.maximum(z[-k:], p1[2] + (z[-k - 1] - p1[2]) * ((1 - u[-k:]) / (1 - u[-k - 1])))
    return np.column_stack([xy, z, yaw])


class LegIK:
    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(C.X2_XML)
        self.d = mujoco.MjData(self.m)
        self.ankle = [self.m.body('%s_ankle_roll_link' % s).id for s in SIDES]
        self.dofs = [[self.m.jnt_dofadr[self.m.joint(j).id] for j in LEGS[s]] for s in SIDES]
        self.adr = [[self.m.jnt_qposadr[self.m.joint(j).id] for j in LEGS[s]] for s in SIDES]
        self.lim = [self.m.jnt_range[[self.m.joint(j).id for j in LEGS[s]]] for s in SIDES]

    def solve(self, q, targets, iters=12):
        """targets: per side (ankle xyz, ankle 3x3); q has the root and upper body set. Returns q, miss (2,)."""
        d, m = self.d, self.m
        d.qpos[:] = q
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for _ in range(iters):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            for f in range(2):
                b = self.ankle[f]
                pos, Rt = targets[f]
                R = d.xmat[b].reshape(3, 3)
                e = np.concatenate([pos - d.xpos[b],
                                    0.6 * 0.5 * sum(np.cross(R[:, i], Rt[:, i]) for i in range(3))])
                mujoco.mj_jacBody(m, d, jp, jr, b)
                J = np.vstack([jp[:, self.dofs[f]], 0.6 * jr[:, self.dofs[f]]])
                dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
                d.qpos[self.adr[f]] = np.clip(d.qpos[self.adr[f]] + dq, *self.lim[f].T)
        mujoco.mj_kinematics(m, d)
        miss = np.array([np.linalg.norm(targets[f][0] - d.xpos[self.ankle[f]]) for f in range(2)])
        return d.qpos.copy(), miss


def _rz(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _tilt(n):
    """Smallest rotation taking +z to the unit normal n."""
    z = np.array([0, 0, 1.0])
    v, c = np.cross(z, n), float(np.dot(z, n))
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def _quat(yaw, pitch):
    cy, sy, cp, sp = math.cos(yaw / 2), math.sin(yaw / 2), math.cos(pitch / 2), math.sin(pitch / 2)
    return np.array([cy * cp, -sy * sp, cy * sp, sy * cp])          # yaw then pitch (z, then local y)


@dataclass
class Timeline:
    """The walk as targets per frame (50 Hz), before IK: what the stitcher reuses."""
    steps: list
    foot: np.ndarray        # (T, 2, 4) sole centre xyz + yaw
    planted: np.ndarray     # (T, 2) height each foot stands at (a swinging foot: between take-off and landing)
    nrm: np.ndarray         # (T, 2, 3) sole normal (ground plane under the foot)
    stance_of: np.ndarray   # (T,) the only foot on the ground, -1 for both
    swings: list            # (foot, first frame, end frame, take-off (4,), landing (4,), Step) per step
    pxy: np.ndarray         # (T, 2) pelvis
    pz: np.ndarray          # (T,)
    pyaw: np.ndarray        # (T,)


def timeline(course, steps):
    """Foot, pelvis and contact targets per frame for a footstep plan (the P3 geometric swings)."""
    foot, planted, nrm, stance_of, swings = [], [], [], [], []
    cur = [np.array([*steps[0].xy, steps[0].z, steps[0].yaw]), np.array([*steps[1].xy, steps[1].z, steps[1].yaw])]
    cur_n = [steps[0].normal, steps[1].normal]

    def hold(n):
        for _ in range(n):
            foot.append(np.stack(cur))
            planted.append([cur[0][2], cur[1][2]])
            nrm.append(np.stack(cur_n))
            stance_of.append(-1)
    hold(int(0.6 * FPS))
    for st in steps[2:]:
        f = st.side
        p0, p1 = cur[f], np.array([*st.xy, st.z, st.yaw])
        dur = SWING_T + RISE_T * abs(p1[2] - p0[2]) / 0.1 + 0.1 * max(0.0, np.linalg.norm(p1[:2] - p0[:2]) - 0.3)
        n = int(round(dur * FPS))
        st.t_lift = len(foot) / FPS
        swings.append((f, len(foot), len(foot) + n, p0.copy(), p1.copy(), st))
        curve = swing_curve(course, p0, p1, p0[3], p1[3], n)
        for i in range(n):
            pair = [cur[0], cur[1]]
            pair[f] = curve[i]
            foot.append(np.stack(pair))
            pz_pair = [cur[0][2], cur[1][2]]
            pz_pair[f] = p0[2] + (p1[2] - p0[2]) * i / max(n - 1, 1)
            planted.append(pz_pair)
            u = i / max(n - 1, 1)                                     # tilt from the take-off plane to the landing one
            n_pair = list(cur_n)
            nv = (1 - u) * cur_n[f] + u * st.normal
            n_pair[f] = nv / np.linalg.norm(nv)
            nrm.append(np.stack(n_pair))
            stance_of.append(1 - f)
        cur[f] = p1
        cur_n[f] = st.normal
        st.t_land = len(foot) / FPS
        hold(int(DOUBLE_T * FPS))
    hold(int(0.8 * FPS))
    planted, nrm, foot, stance_of = np.array(planted), np.array(nrm), np.array(foot), np.array(stance_of)
    T = len(foot)
    # pelvis: xy over the feet, swaying toward the stance foot; z from the feet; yaw from the feet
    mid = foot[:, :, :2].mean(1)
    sway = np.zeros((T, 2))
    for f in range(2):
        sway[stance_of == f] = (foot[stance_of == f, f, :2] - mid[stance_of == f]) * (SWAY / HALF_WIDTH)
    pyaw = np.unwrap(np.arctan2(np.sin(foot[:, :, 3]).sum(1), np.cos(foot[:, :, 3]).sum(1)))   # circular mean
    pyaw = smooth(pyaw[:, None])[:, 0]
    zlo, zhi = planted.min(1), planted.max(1)                       # where the feet stand, not the swing height
    pz = (zlo - STEP_DROP * (zhi - zlo)) + PELVIS_H
    pxy = smooth(mid + sway)
    for f in range(2):                                               # each planted leg keeps within LEG_REACH
        hd = np.stack([np.cos(pyaw), np.sin(pyaw)], 1)
        fwd = np.abs(((foot[:, f, :2] - pxy) * hd).sum(1))
        cap = planted[:, f] + np.sqrt(np.maximum(LEG_REACH ** 2 - fwd ** 2, 0.0))
        pz = np.where(stance_of != 1 - f, np.minimum(pz, cap), pz)  # foot f planted unless the other is the stance
    pz = np.array([pz[max(0, i - 8):i + 9].min() for i in range(len(pz))])   # keep under the caps after smoothing
    pz = smooth(pz[:, None])[:, 0]
    return Timeline(steps, foot, planted, nrm, stance_of, swings, pxy, pz, pyaw)


def smooth(a, width=31):
    ker = np.hanning(width)
    ker /= ker.sum()
    h = width // 2
    return np.stack([np.convolve(np.pad(a[:, k], h, mode='edge'), ker, 'valid') for k in range(a.shape[1])], 1)


def upper_reference():
    """Arms, waist and head: the relaxed walk's average pose, torso upright."""
    v = list(joblib.load(C.SOURCES['walks'][0]).values())[0]
    walk = C._from_pkl_entry(v)
    ref = walk[0].copy()
    ref[7:] = walk[:, 7:].mean(0)
    ref[7 + 13] = WAIST_PITCH                                        # waist_pitch_joint (dof 13)
    return ref


def solve_legs(tl, upper=None, root_quat=None, foot_rot=None, pz=None, foot=None):
    """Leg IK over a timeline. Optional per-frame overrides: upper (T, 38) for the non-leg joints, root_quat (T, 4),
    foot_rot (T, 2, 3, 3) sole rotations, pz (T,) pelvis height, foot (T, 2, 4) foot targets. Returns qpos, miss."""
    T = len(tl.foot)
    ref = upper_reference()
    foot = tl.foot if foot is None else foot
    pz = tl.pz if pz is None else pz
    ik = LegIK()
    legs = sum(ik.adr, [])
    qpos, miss = np.zeros((T, 38)), np.zeros((T, 2))
    q = ref.copy()
    for t in range(T):
        if upper is not None:
            keep = q[legs].copy()
            q[7:] = upper[t, 7:]
            q[legs] = keep
        q[:3] = (tl.pxy[t, 0], tl.pxy[t, 1], pz[t])
        q[3:7] = _quat(tl.pyaw[t], math.radians(2)) if root_quat is None else root_quat[t]
        targets = []
        for f in range(2):
            R = (_tilt(tl.nrm[t, f]) @ _rz(foot[t, f, 3])) if foot_rot is None else foot_rot[t, f]
            targets.append((foot[t, f, :3] - R @ C.SOLE, R))
        q, miss[t] = ik.solve(q, targets)
        if miss[t].max() > 0.01:                                      # stuck from the last frame: restart the legs
            q0 = q.copy()
            q0[legs] = ref[legs]
            q2, m2 = ik.solve(q0, targets, iters=40)
            if m2.max() < miss[t].max():
                q, miss[t] = q2, m2
        qpos[t] = q
    return qpos, miss


def generate(course, waypoints, terrain='cad'):
    tl = timeline(course, plan_footsteps(course, waypoints))
    qpos, miss = solve_legs(tl)
    return Gait(qpos, tl.steps, tl.foot, miss, terrain)


if __name__ == '__main__':
    import argparse
    import time
    ap = argparse.ArgumentParser()
    ap.add_argument('--route', default='mobility')
    ap.add_argument('--scene', default='cad')
    a = ap.parse_args()
    t0 = time.time()
    course = H.Course(a.scene)
    g = generate(course, H.route_waypoints(a.route), a.scene)
    print('%d steps (%d not safe), %.1f s of motion, IK miss p99 %.1f mm max %.1f mm, %.0f s' % (
        len(g.steps), sum(not s.safe for s in g.steps), len(g.qpos) / FPS, np.percentile(g.miss, 99) * 1000,
        g.miss.max() * 1000, time.time() - t0))
