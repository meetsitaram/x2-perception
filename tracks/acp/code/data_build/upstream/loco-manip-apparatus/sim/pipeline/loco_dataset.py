"""Training data: flat-ground locomotion with distractor objects and nine-body contacts, for contact-driven SuperMimic.

The motion is captured locomotion on a flat floor (Bones-SEED subsets). Objects are placed where the motion never
touches them, of two kinds:

* step blocks, in the stride: at the middle of a foot's swing (between where it lifts off and where it lands next),
  turned along the stride, as high as the swing allows: the lowest body point over the block's footprint (+ STEP_SIDE
  around it) minus STEP_GAP, snapped down to STEP_HEIGHTS. The foot passes over them, so the scan shows an obstacle
  right where the contacts step across it;
* clutter, beside the path: no body point over the whole clip within CLEARANCE of the footprint below top + CLEARANCE.

Sizes come from small fixed sets (STEP_HALF x STEP_HEIGHTS, CLUTTER_HALF x CLUTTER_HEIGHTS): the trainer gives each
size its own mocap boxes in every world, so a few sizes keep the model small. Contacts are the nine-body hindsight
labels (pipeline.contacts) on the flat floor; the feet never touch the objects, so the labels are unchanged.

Clips are kept only if they are flat-ground motion: content type not in SKIP_TYPES and no weight-bearing support found
by the prop detector (pipeline.props): a foot or knee resting above the floor (box, stairs) or the pelvis on a seat
at least SEAT_MIN high (a chair; a deep crouch reads as a low seat). Resting hands are ignored: on flat ground a hand
that rested on a table or on another person just stays in the air.

Writes build/train/x2_loco.pkl (SONIC motion format, 50 Hz), x2_loco_props.json (clip -> distractor boxes, the
chains' prop format) and x2_loco_contacts.pkl (the chains' contact format), plus x2_loco_report.json.

Run (from sim): python -m pipeline.loco_dataset [--subset loco_2k] [--max 0] [--seed 0]
"""
import argparse
import json
import math
import os

import joblib
import numpy as np

from pipeline import clipdb as C
from pipeline import contacts as K
from pipeline import hmap as H
from pipeline import props as P

import pandas as pd  # noqa: E402  (after mujoco: importing pandas first breaks mujoco's DLL init on Windows)

SEED_DIR = os.path.join(C.PROJ, 'bones-seed')
CSV_DIRS = ('x2_loco30k', 'x2_walk_2k')
OUT = os.path.join(H.ROOT, 'build', 'train')
SKIP_TYPES = ('climbing', 'dancing')        # content types that need props or are not locomotion
CLEARANCE = 0.08                            # m: clutter: no body point this close (sideways and above its top)
CLUTTER_HALF = ((0.05, 0.05), (0.10, 0.06), (0.15, 0.10))   # m half footprints
CLUTTER_HEIGHTS = (0.05, 0.10, 0.15)        # m
N_CLUTTER, N_TRY = 4, 40                    # clutter kept per clip, candidates
NEAR = (0.15, 0.7)                          # m: clutter centre distance from a foot-path point
STEP_HALF = ((0.04, 0.06), (0.05, 0.10))    # m half footprints (along the stride, across)
STEP_HEIGHTS = (0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12)   # m
STEP_GAP = 0.02                             # m: least clearance of the swing over a step block
STEP_SIDE = 0.03                            # m: around the footprint, body points must also clear it
MIN_STRIDE = 0.25                           # m: swings shorter than this (shuffles, turns on the spot) get no block
N_STEP = 6                                  # step blocks kept per clip
SEAT_MIN = 0.4                              # m: a pelvis support at least this high is a chair, not a crouch
LEGS = ('left_foot', 'right_foot', 'left_knee', 'right_knee')


def needs_prop(sup):
    """The weight-bearing supports among the prop detector's (see module doc)."""
    return [p for p in sup if any(b in LEGS for b in p.bodies) or ('pelvis' in p.bodies and p.top >= SEAT_MIN)]


def find_csv(name):
    for d in CSV_DIRS:
        p = os.path.join(SEED_DIR, d, name + '.csv')
        if os.path.exists(p):
            return p
    return None


def _local(flat, c, yaw):
    """Body points (rows of flat) in a box frame (centre c, yaw): along, across."""
    cy, sy = math.cos(yaw), math.sin(yaw)
    d = flat[:, :2] - c
    return cy * d[:, 0] + sy * d[:, 1], -sy * d[:, 0] + cy * d[:, 1]


def _box(kind, c, yaw, hx, hy, h):
    return dict(kind=kind, center=[float(c[0]), float(c[1]), h / 2], half=[hx, hy, h / 2], yaw=float(yaw))


def _overlaps(c, hx, hy, boxes):
    r = math.hypot(hx, hy)
    return any(math.hypot(c[0] - b['center'][0], c[1] - b['center'][1]) < r + math.hypot(*b['half'][:2]) + 0.02
               for b in boxes)


def step_blocks(rng, flat, lab):
    """Blocks in the stride (module doc): one candidate per foot swing, as high as the swing clears."""
    cands = []
    for f in range(2):
        runs = P._runs(lab['contact'][:, f], 1)
        for (_, e0), (a1, _) in zip(runs, runs[1:]):
            lift, land = lab['support'][e0 - 1, f, :2], lab['support'][a1, f, :2]
            d = land - lift
            if np.hypot(*d) >= MIN_STRIDE:
                cands.append(((lift + land) / 2, math.atan2(d[1], d[0])))
    out = []
    for i in rng.permutation(len(cands)):
        if len(out) >= N_STEP:
            break
        c, yaw = cands[i]
        hx, hy = STEP_HALF[rng.integers(len(STEP_HALF))]
        u, v = _local(flat, c, yaw)
        near = (np.abs(u) < hx + STEP_SIDE) & (np.abs(v) < hy + STEP_SIDE)
        room = flat[near, 2].min() - STEP_GAP if near.any() else STEP_HEIGHTS[-1]
        fits = [h for h in STEP_HEIGHTS if h <= room]
        if fits and not _overlaps(c, hx, hy, out):
            out.append(_box('step', c, yaw, hx, hy, fits[-1]))
    return out


def clutter(rng, flat, pts, taken):
    """Blocks beside the path, clear of every body point by CLEARANCE (module doc)."""
    feet = pts[:, :2].reshape(-1, 3)
    feet = feet[:: max(1, len(feet) // 400), :2]
    out = []
    for _ in range(N_TRY):
        if len(out) >= N_CLUTTER:
            break
        r, th = rng.uniform(*NEAR), rng.uniform(-math.pi, math.pi)
        c = feet[rng.integers(len(feet))] + r * np.array([math.cos(th), math.sin(th)])
        hx, hy = CLUTTER_HALF[rng.integers(len(CLUTTER_HALF))]
        h = CLUTTER_HEIGHTS[rng.integers(len(CLUTTER_HEIGHTS))]
        yaw = rng.uniform(-math.pi, math.pi)
        u, v = _local(flat, c, yaw)
        if ((np.abs(u) < hx + CLEARANCE) & (np.abs(v) < hy + CLEARANCE) & (flat[:, 2] < h + CLEARANCE)).any():
            continue
        if not _overlaps(c, hx, hy, taken + out):
            out.append(_box('clutter', c, yaw, hx, hy, h))
    return out


def distractors(rng, pts, lab):
    flat = pts.reshape(-1, 3)
    flat = flat[flat[:, 2] < max(STEP_HEIGHTS[-1], CLUTTER_HEIGHTS[-1]) + CLEARANCE + STEP_GAP]   # low points only
    steps = step_blocks(rng, flat, lab)
    return steps + clutter(rng, flat, pts, steps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subset', default='loco_2k')
    ap.add_argument('--max', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    meta = pd.read_csv(os.path.join(SEED_DIR, 'subsets', a.subset + '.csv'))
    meta = meta[~meta.content_type_of_movement.fillna('').str.contains('|'.join(SKIP_TYPES))]
    bp = K.BodyPoints()
    clips, props, cons, report = {}, {}, {}, []
    for name, kind in zip(meta.move_name, meta.content_type_of_movement):
        if a.max and len(clips) >= a.max:
            break
        path = find_csv(name)
        if path is None:
            continue
        q = P.load_csv(path)
        sup, _ = P.build_props(q, bp)
        if needs_prop(sup):
            report.append(dict(clip=name, kept=False, why='rests on a support (prop capture)'))
            continue
        pts = bp.world(q)
        lab = K.label_clip(C.Clip(name, 'bones', q), bp, None)
        boxes = distractors(rng, pts, lab)
        xyzw = np.concatenate([q[:, 4:7], q[:, 3:4]], 1)
        clips[name] = dict(root_trans_offset=q[:, :3].astype(np.float32), root_rot=xyzw.astype(np.float32),
                           dof=q[:, 7:].astype(np.float32), fps=50)
        props[name] = boxes
        cons[name] = dict(next=lab['next'].astype(np.float32), next_dt=lab['next_dt'].astype(np.float32),
                          occ=lab['occ'], contact=lab['contact'])
        report.append(dict(clip=name, kept=True, type=kind, seconds=round(len(q) / 50, 1), distractors=len(boxes),
                           steps=[round(2 * b['half'][2], 2) for b in boxes if b['kind'] == 'step']))
        if len(clips) % 50 == 0:
            print('%d clips' % len(clips), flush=True)
    tag = 'x2_loco' if a.subset == 'loco_2k' else 'x2_' + a.subset
    joblib.dump(clips, os.path.join(OUT, tag + '.pkl'))
    json.dump(props, open(os.path.join(OUT, tag + '_props.json'), 'w'))
    joblib.dump(cons, os.path.join(OUT, tag + '_contacts.pkl'))
    json.dump(report, open(os.path.join(OUT, tag + '_report.json'), 'w'), indent=1)
    kept = [r for r in report if r['kept']]
    hs = np.array([h for r in kept for h in r['steps']])
    print('kept %d clips (%.1f min), %d dropped as prop captures, %.1f objects per clip'
          % (len(kept), sum(r['seconds'] for r in kept) / 60, len(report) - len(kept),
             np.mean([r['distractors'] for r in kept]) if kept else 0))
    print('step blocks: %d in %d clips; heights (m): %s'
          % (len(hs), sum(bool(r['steps']) for r in kept),
             {float(h): int(n) for h, n in zip(*np.unique(hs, return_counts=True))}))


if __name__ == '__main__':
    main()
