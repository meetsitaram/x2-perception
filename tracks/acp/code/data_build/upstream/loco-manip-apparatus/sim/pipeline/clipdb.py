"""Clip index: every frame of the X2 clip library with its matching features, including contact features.

Library (short term), all X2-native (31 dof, MuJoCo order), resampled to 50 Hz:
  kplanner   build/clips/kplanner_x2.pkl: X2 planner + G1 styles via Phi (clip_gen.py), flat floor
  pico       gwbc-x2 x2_pico_chores_0917.pkl: Pico full-body teleop sessions on the X2 (chores), flat floor
  walks      sonic-x2 relaxed walk and idle stand (Bones-SEED retargets), flat floor
  stairs     stair clips in the coordinates of the stairs they were made for: our IK climb on the CAD stairs
             (stair_clip.py), Bones-SEED stair climbs retargeted (bones_retarget.py) onto the CAD stairs (_ours), the
             low apparatus stairs (_app109), test flights (_flight<n>_<mm>) and single steps (_single<mm>); the ground
             under each foot comes from the P1 height map of that scene

Per frame t, in the heading frame at t (origin at the root's xy, x along the root's heading, z up from ref_z, the
height of the lowest foot in contact, the same reference as the P1 height map):
  traj      root xy at +0.33, +0.67, +1.0 s and heading (cos, sin) there                     12
  feet      left, right sole-centre position                                                    6
  feet_vel  left, right sole-centre velocity                                                    6
  root_vel  root velocity                                                                       3
  contact   left, right in contact (0/1)                                                        2
  next      left, right next foothold: where the foot lands next (if it lands within HORIZON s;
            a planted foot that stays planted gives its own position) and the time until then   8
The first 27 are the Perceptive Humanoid Parkour (PHP) matching features; contact and next are ours, for matching
against the footholds the P5 contact planner proposes.

Contacts: each foot's support point is the one of its 12 sole contact spheres closest to the ground directly under it
(so a heel hanging over a tread edge does not count). The foot is in contact when that point is within Z_CONTACT of the
ground and moves slower than V_CONTACT; runs shorter than MIN_RUN frames (either state) are absorbed. A sole point
rather than the sole centre, so a foot rolling heel to toe still counts as planted.

    db = build()                              # cached in build/clips/clipdb.npz
    db.features[db.valid]                     # (N, 37) rows to match
    db.clip_of(i), db.frame_of(i)             # which clip / frame a row comes from

Run (rl venv, from sim): python -m pipeline.clipdb [--rebuild] [--sources kplanner pico walks stairs]
"""
from __future__ import annotations

import glob
import math
import os
import re
from dataclasses import dataclass, field

import joblib
import mujoco
import numpy as np

from pipeline import hmap as H

FPS = 50
FUTURE = (17, 33, 50)                    # frames: +0.33, +0.67, +1.0 s
HORIZON = 1.5                            # s: next-foothold look-ahead (the P5 contact plan length)
V_CONTACT = 0.5                          # m/s: support-point speed below which a foot can be planted (retargets skate up to ~0.4)
Z_CONTACT = 0.04                         # m: lowest sole point within this of the ground
MIN_RUN = 4                              # frames: shorter contact / swing runs are absorbed
SOLE = np.array([0.03, 0.0, -0.073])     # sole centre in the ankle-roll frame (contact spheres at z -0.068, r 0.005)
FEET = ('left_ankle_roll_link', 'right_ankle_roll_link')

PROJ = os.path.dirname(H.ROOT)
SONIC = os.path.join(PROJ, 'sonic-x2')
X2_XML = os.path.join(SONIC, 'assets', 'mjcf', 'x2_ultra.xml')
CACHE = os.path.join(H.ROOT, 'build', 'clips', 'clipdb.npz')
SOURCES = {
    'kplanner': [os.path.join(H.ROOT, 'build', 'clips', 'kplanner_x2.pkl')],
    'pico': [os.path.join(PROJ, 'gwbc-x2', 'gear_sonic', 'data', 'motions', 'x2_pico_chores', 'x2_pico_chores_0917.pkl')],
    'walks': [os.path.join(SONIC, 'motions', 'x2_relaxed_walk.pkl'), os.path.join(SONIC, 'motions', 'x2_idle_stand.pkl')],
    'stairs': [os.path.join(H.ROOT, 'build', 'stairs', 'climb_x2_frames.npy')]
              + sorted(glob.glob(os.path.join(H.ROOT, 'build', 'stairs', 'bones', 'stairs_climbing_*.npy'))),
}
# stair clip file -> the scene it was retargeted onto; files matching none (raw retargets, tracking runs) are skipped
STAIR_SCENES = [(r'^climb_x2_frames$', 'cad'), (r'_ours$', 'cad'), (r'_app109$', 'low'),
                (r'_(flight\d_\d{3})$', None), (r'_(single_?\d{3})$', None)]


def stair_scene(name):
    for pat, scene in STAIR_SCENES:
        m = re.search(pat, name)
        if m:
            return scene or m.group(1).replace('single', 'single_').replace('__', '_')
    return None
GROUPS = {'traj': 12, 'feet': 6, 'feet_vel': 6, 'root_vel': 3, 'contact': 2, 'next': 8}


def slices():
    out, k = {}, 0
    for g, n in GROUPS.items():
        out[g] = slice(k, k + n)
        k += n
    return out


SL = slices()
DIM = sum(GROUPS.values())


@dataclass
class Clip:
    name: str
    source: str
    qpos: np.ndarray                     # (T, 38) root xyz, root quat wxyz, 31 dof; 50 Hz
    terrain: str | None = None           # None: flat floor at z = 0; else a hmap scene name ('cad', 'low')
    foot: np.ndarray = field(default=None, repr=False)       # (T, 2, 3) sole centres
    foot_low: np.ndarray = field(default=None, repr=False)   # (T, 2) support point z
    low_speed: np.ndarray = field(default=None, repr=False)  # (T, 2) support point speed
    foot_yaw: np.ndarray = field(default=None, repr=False)   # (T, 2)
    ground: np.ndarray = field(default=None, repr=False)     # (T, 2) ground height under the support point
    contact: np.ndarray = field(default=None, repr=False)    # (T, 2) bool
    labels: dict = field(default=None, repr=False)            # nine-body labels (pipeline.contacts.label_clip)


# ------------------------------------------------------------------ loading
def _resample(q, fps):
    """(T, 38) qpos at fps -> 50 Hz (linear for positions and joints, normalised lerp for the quaternion)."""
    if abs(fps - FPS) < 1e-6:
        return q
    t_src = np.arange(len(q)) / fps
    t = np.arange(0.0, t_src[-1] + 1e-9, 1.0 / FPS)
    out = np.stack([np.interp(t, t_src, q[:, k]) for k in range(q.shape[1])], 1)
    quat = q[:, 3:7].copy()
    for i in range(1, len(quat)):                           # keep one hemisphere before interpolating
        if np.dot(quat[i], quat[i - 1]) < 0:
            quat[i] = -quat[i]
    out[:, 3:7] = np.stack([np.interp(t, t_src, quat[:, k]) for k in range(4)], 1)
    out[:, 3:7] /= np.linalg.norm(out[:, 3:7], axis=1, keepdims=True)
    return out


def _from_pkl_entry(v):
    xyzw = np.asarray(v['root_rot'], float)
    q = np.concatenate([np.asarray(v['root_trans_offset'], float), xyzw[:, 3:4], xyzw[:, :3],
                        np.asarray(v['dof'], float)], 1)
    return _resample(q, float(v.get('fps', FPS)))


def load_library(sources=tuple(SOURCES)):
    clips = []
    for src in sources:
        for path in SOURCES[src]:
            if not os.path.exists(path):
                print('missing, skipped:', path)
                continue
            if path.endswith('.pkl'):
                for name, v in joblib.load(path).items():
                    clips.append(Clip(name, src, _from_pkl_entry(v)))
            else:                                            # world qpos of a stair scene; the robot comes first
                name = os.path.splitext(os.path.basename(path))[0]
                scene = stair_scene(name)
                if scene is not None:
                    clips.append(Clip(name, src, np.load(path)[:, :38].astype(float), terrain=scene))
    return clips


# ------------------------------------------------------------------ kinematics and contacts
class X2Kin:
    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(X2_XML)
        self.d = mujoco.MjData(self.m)
        self.feet = [self.m.body(b).id for b in FEET]
        self.spheres = [[g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] == b
                         and self.m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE] for b in self.feet]

    def feet_of(self, qpos, fps=FPS):
        """For (T, 38) qpos: sole centres (T, 2, 3), sole sphere bottoms (T, 2, 12, 3), their speeds (T, 2, 12) and
        foot yaw (T, 2)."""
        T = len(qpos)
        foot, yaw = np.zeros((T, 2, 3)), np.zeros((T, 2))
        sph = np.zeros((T, 2, len(self.spheres[0]), 3))
        for t in range(T):
            self.d.qpos[:] = qpos[t]
            mujoco.mj_kinematics(self.m, self.d)
            for f, b in enumerate(self.feet):
                R = self.d.xmat[b].reshape(3, 3)
                foot[t, f] = self.d.xpos[b] + R @ SOLE
                sph[t, f] = self.d.geom_xpos[self.spheres[f]]
                yaw[t, f] = math.atan2(R[1, 0], R[0, 0])
        sph[..., 2] -= self.m.geom_size[self.spheres[0][0]][0]
        speed = np.linalg.norm(np.gradient(sph, 1.0 / fps, axis=0), axis=-1)
        return foot, sph, speed, yaw


def _absorb_short_runs(c, min_run):
    """Flip runs shorter than min_run (inside the clip) to the surrounding state."""
    c = c.copy()
    T = len(c)
    i = 0
    while i < T:
        j = i
        while j < T and c[j] == c[i]:
            j += 1
        if j - i < min_run and i > 0 and j < T:
            c[i:j] = c[i - 1]
        i = j
    return c


def contact_labels(low_speed, foot_low, ground, v_max=V_CONTACT, z_max=Z_CONTACT, min_run=MIN_RUN):
    """(T, 2) bool: the lowest sole point slow and on the ground; short runs absorbed."""
    raw = (low_speed < v_max) & (foot_low - ground < z_max)
    return np.stack([_absorb_short_runs(raw[:, f], min_run) for f in range(2)], 1)


def touchdowns(c):
    """Frames where a contact run starts (not counting frame 0)."""
    return np.nonzero(c[1:] & ~c[:-1])[0] + 1


SETTLE = 25                              # frames: a foothold is where the foot settles within this of touchdown


def settle_frames(c, gap):
    """For each touchdown of one foot (contact (T,), support gap (T,)), the frame where it settles: the lowest gap in
    the first SETTLE frames of that contact run."""
    out = []
    for td in touchdowns(c):
        e = td
        while e < len(c) and c[e] and e - td < SETTLE:
            e += 1
        out.append(td + int(np.argmin(gap[td:e])))
    return np.array(out, int)


def annotate(clips, kin=None, courses=None, bodies=None):
    from pipeline import contacts as K
    kin = kin or X2Kin()
    bodies = bodies or K.BodyPoints()
    courses = courses if courses is not None else {}
    for c in clips:
        c.foot, sph, speed, c.foot_yaw = kin.feet_of(c.qpos)
        if c.terrain is None:
            g = np.zeros(sph.shape[:3])
        else:
            course = courses.setdefault(c.terrain, H.Course(c.terrain))
            g = course.height(sph[..., :2].reshape(-1, 2)).reshape(sph.shape[:3])
        k = (sph[..., 2] - g).argmin(-1)[..., None]                 # support point: closest to its own ground
        c.foot_low = np.take_along_axis(sph[..., 2], k, -1)[..., 0]
        c.ground = np.take_along_axis(g, k, -1)[..., 0]
        c.low_speed = np.take_along_axis(speed, k, -1)[..., 0]
        c.contact = contact_labels(c.low_speed, c.foot_low, c.ground)
        c.labels = K.label_clip(c, bodies, None if c.terrain is None else courses[c.terrain])
    return clips


# ------------------------------------------------------------------ features
def root_yaw(quat_wxyz):
    w, x, y, z = quat_wxyz.T
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _to_heading(v_xy, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([c * v_xy[..., 0] + s * v_xy[..., 1], -s * v_xy[..., 0] + c * v_xy[..., 1]], -1)


def next_footholds(c: Clip, horizon=HORIZON):
    """(T, 2, 3) next foothold and (T, 2) time to it, (T, 2) known. A foot that touches down within horizon gives where
    it settles after that touchdown, at the time to the touchdown; a planted foot that does not gives where it stands
    (time = horizon); a swinging foot with no touchdown left in the clip is unknown."""
    T = len(c.qpos)
    pos, dt, known = np.zeros((T, 2, 3)), np.zeros((T, 2)), np.zeros((T, 2), bool)
    H_frames = int(round(horizon * FPS))
    for f in range(2):
        td = touchdowns(c.contact[:, f])
        st = settle_frames(c.contact[:, f], c.foot_low[:, f] - c.ground[:, f])
        k = np.searchsorted(td, np.arange(T), side='right')         # first touchdown after t
        for t in range(T):
            if k[t] < len(td) and td[k[t]] - t <= H_frames:
                pos[t, f], dt[t, f], known[t, f] = c.foot[st[k[t]], f], (td[k[t]] - t) / FPS, True
            elif c.contact[t, f]:
                pos[t, f], dt[t, f], known[t, f] = c.foot[t, f], horizon, True
    return pos, dt, known


def clip_features(c: Clip):
    """(T, DIM) features and (T,) valid (a full second ahead, both next footholds known)."""
    q = c.qpos
    T = len(q)
    root, yaw = q[:, :3], root_yaw(q[:, 3:7])
    vel = np.gradient(root, 1.0 / FPS, axis=0)
    fvel = np.gradient(c.foot, 1.0 / FPS, axis=0)
    low_contact = np.where(c.contact, c.foot[:, :, 2], np.inf).min(1)
    ref_z = np.where(np.isfinite(low_contact), low_contact, c.foot[:, :, 2].min(1))
    nxt, ndt, known = next_footholds(c)
    F = np.zeros((T, DIM))
    for k, df in enumerate(FUTURE):
        j = np.minimum(np.arange(T) + df, T - 1)
        F[:, SL['traj'].start + 2 * k: SL['traj'].start + 2 * k + 2] = _to_heading(root[j, :2] - root[:, :2], yaw)
        F[:, SL['traj'].start + 6 + 2 * k] = np.cos(yaw[j] - yaw)
        F[:, SL['traj'].start + 7 + 2 * k] = np.sin(yaw[j] - yaw)
    for f in range(2):
        rel = _to_heading(c.foot[:, f, :2] - root[:, :2], yaw)
        F[:, SL['feet'].start + 3 * f: SL['feet'].start + 3 * f + 2] = rel
        F[:, SL['feet'].start + 3 * f + 2] = c.foot[:, f, 2] - ref_z
        F[:, SL['feet_vel'].start + 3 * f: SL['feet_vel'].start + 3 * f + 2] = _to_heading(fvel[:, f, :2], yaw)
        F[:, SL['feet_vel'].start + 3 * f + 2] = fvel[:, f, 2]
        F[:, SL['contact'].start + f] = c.contact[:, f]
        F[:, SL['next'].start + 3 * f: SL['next'].start + 3 * f + 2] = _to_heading(nxt[:, f, :2] - root[:, :2], yaw)
        F[:, SL['next'].start + 3 * f + 2] = nxt[:, f, 2] - ref_z
        F[:, SL['next'].start + 6 + f] = ndt[:, f]
    F[:, SL['root_vel']] = np.concatenate([_to_heading(vel[:, :2], yaw), vel[:, 2:3]], 1)
    valid = (np.arange(T) + FUTURE[-1] < T) & known.all(1)
    return F, valid


# ------------------------------------------------------------------ database
@dataclass
class ClipDB:
    names: list
    sources: list
    terrain: list
    starts: np.ndarray                   # (C + 1,) row offsets per clip
    features: np.ndarray                 # (N, DIM)
    valid: np.ndarray                    # (N,)
    qpos: np.ndarray                     # (N, 38)
    foot: np.ndarray                     # (N, 2, 3)
    foot_low: np.ndarray                 # (N, 2)
    ground: np.ndarray                   # (N, 2)
    contact: np.ndarray                  # (N, 2)
    contact9: np.ndarray = None          # (N, 9) nine-body contact (pipeline.contacts.BODIES order)
    support9: np.ndarray = None          # (N, 9, 3) support point per body, world
    gap9: np.ndarray = None              # (N, 9) support point height above the terrain
    next9: np.ndarray = None             # (N, 9, 3) terrain point of the next contact within the horizon, world
    next_dt9: np.ndarray = None          # (N, 9) s to it
    occ9: np.ndarray = None              # (N, 9) a new contact within the horizon

    def clip_of(self, i):
        return int(np.searchsorted(self.starts, i, side='right') - 1)

    def frame_of(self, i):
        return int(i - self.starts[self.clip_of(i)])

    def rows(self, name):
        k = self.names.index(name)
        return slice(int(self.starts[k]), int(self.starts[k + 1]))

    def save(self, path=CACHE):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, names=np.array(self.names), sources=np.array(self.sources),
                            terrain=np.array([t or '' for t in self.terrain]), starts=self.starts,
                            features=self.features, valid=self.valid, qpos=self.qpos, foot=self.foot,
                            foot_low=self.foot_low, ground=self.ground, contact=self.contact, contact9=self.contact9,
                            support9=self.support9, gap9=self.gap9, next9=self.next9, next_dt9=self.next_dt9,
                            occ9=self.occ9)

    @staticmethod
    def load(path=CACHE):
        z = np.load(path)
        return ClipDB(list(z['names']), list(z['sources']), [t or None for t in z['terrain']], z['starts'],
                      z['features'], z['valid'], z['qpos'], z['foot'], z['foot_low'], z['ground'], z['contact'],
                      z['contact9'], z['support9'], z['gap9'], z['next9'], z['next_dt9'], z['occ9'])


def from_clips(clips):
    feats, valid = zip(*(clip_features(c) for c in clips))
    starts = np.concatenate([[0], np.cumsum([len(c.qpos) for c in clips])])
    cat = lambda a: np.concatenate([getattr(c, a) for c in clips])
    lab = lambda k: np.concatenate([c.labels[k] for c in clips])
    return ClipDB([c.name for c in clips], [c.source for c in clips], [c.terrain for c in clips], starts,
                  np.concatenate(feats), np.concatenate(valid), cat('qpos'), cat('foot'), cat('foot_low'),
                  cat('ground'), cat('contact'), lab('contact'), lab('support').astype(np.float32),
                  lab('gap').astype(np.float32), lab('next').astype(np.float32), lab('next_dt').astype(np.float32),
                  lab('occ'))


def build(sources=tuple(SOURCES), rebuild=False, path=CACHE):
    if os.path.exists(path) and not rebuild and path == CACHE and sources == tuple(SOURCES):
        return ClipDB.load(path)
    db = from_clips(annotate(load_library(sources)))
    if sources == tuple(SOURCES):
        db.save(path)
    return db


if __name__ == '__main__':
    import argparse
    import time
    ap = argparse.ArgumentParser()
    ap.add_argument('--rebuild', action='store_true')
    ap.add_argument('--sources', nargs='+', default=list(SOURCES))
    a = ap.parse_args()
    t0 = time.time()
    db = build(tuple(a.sources), a.rebuild)
    print('%d clips, %d frames (%.1f min), %d valid rows, %d features, %.1f s' % (
        len(db.names), len(db.valid), len(db.valid) / FPS / 60, db.valid.sum(), db.features.shape[1], time.time() - t0))
