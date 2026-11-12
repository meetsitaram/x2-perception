"""Training data: chained motions (walking by motion matching, prop skills on placed props) with their props.

Each scene is a random sequence of prop skills placed at random spots and headings: box (onto it and off it), stairs
(up a 6 x 10.9 cm flight), box then stairs, two boxes. The walk between them is motion matching over the walking
database (pipeline.mm). Every motion is scored (pipeline.motion_eval, on its own props); only motions with no teleport,
no rest more than GATE_AIR above a surface and nothing deeper than GATE_INSIDE into a prop are kept.

Writes build/train/x2_chains.pkl (SONIC motion format: clip -> root_trans_offset (T,3) world, root_rot (T,4) xyzw,
dof (T,31), fps 50) and x2_chains_props.json (clip -> [box: kind, center xyz, half xyz, yaw]) for the trainer.

Run (rl venv, from sim): python -m pipeline.chain_dataset [--n 64] [--seed 0]
"""
import argparse
import json
import math
import os

import joblib
import numpy as np

from pipeline import contacts as K
from pipeline import hmap as H
from pipeline import mm as MM
from pipeline import motion_eval as ME
from pipeline import poc_run as PR
from pipeline import poc_scene as PS
from pipeline import skills as SK

OUT = os.path.join(H.ROOT, 'build', 'train')
GATE_INSIDE = 0.04
GATE_AIR = 0.04             # m: the stair clip's own top-tread rest sits 3.6 cm high (retarget); physics settles it
SCENES = ('box', 'stairs', 'box_stairs', 'box_box')


def place(rng, start_xy, heading, dist):
    a = heading + rng.uniform(-0.5, 0.5)
    return (start_xy[0] + dist * math.cos(a), start_xy[1] + dist * math.sin(a)), a + rng.uniform(-0.3, 0.3)


def scene(rng, kind, bases):
    up, dn, st = bases
    start = (0.0, 0.0, float(rng.uniform(-math.pi, math.pi)))
    seq, xy, hd = [], start[:2], start[2]
    for item in kind.split('_'):
        c, yaw = place(rng, xy, hd, rng.uniform(2.0, 3.5))
        if item == 'box':
            seq += [up.place(c, yaw), dn.place(c, yaw)]
            end = seq[-1].qpos[-1]
        else:
            seq += [st.place(c, yaw)]
            end = seq[-1].qpos[-1]
        xy, hd = (end[0], end[1]), float(MM._yaw(end))
    return start, seq


def props_of(seq):
    out, seen = [], set()
    for sk in seq:
        for p in sk.props:
            key = (round(p.center[0], 3), round(p.center[1], 3), round(p.top, 3))
            if key in seen:
                continue
            seen.add(key)
            out.append(dict(kind=p.kind, center=[p.center[0], p.center[1], p.top / 2],
                            half=[p.half[0], p.half[1], p.top / 2], yaw=p.yaw))
    return out


def terrain_of(props):
    t = PS.Terrain()
    t.boxes = [(p['center'][0], p['center'][1], p['half'][0], p['half'][1], p['yaw'], 2 * p['half'][2]) for p in props]
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=64)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    db = PR.walking_db()
    bp = K.BodyPoints()
    up = SK.build_skill(os.path.join(PR.SKILLS, 'come_up_50cm_box_R_001__A045.csv'), bp=bp)
    dn = SK.build_skill(os.path.join(PR.SKILLS, 'come_down_50cm_box_R_001__A045.csv'), bp=bp)
    st = SK.build_skill(os.path.join(H.ROOT, 'build', 'stairs', 'bones',
                                     'stairs_climbing_up_start_R_001__A300_flight6_109.npy'), bp=bp, scene='flight6_109')
    bars = None
    clips, props_all, report = {}, {}, []
    tries = 0
    while len(clips) < a.n and tries < 4 * a.n:
        tries += 1
        kind = SCENES[tries % len(SCENES)]
        start, seq = scene(rng, kind, (up, dn, st))
        q, log = MM.run(db, seq, start)
        props = props_of(seq)
        terr = terrain_of(props)
        if bars is None:
            from pipeline import review_eval as RE
            bars = ME.calibrate(RE.real_clips())
        card, bad, f = ME.score(q, terr, bars)
        o = card['overall']
        ok = o['teleport']['passed'] and f['air'].max() <= GATE_AIR and f['inside'].max() <= GATE_INSIDE
        report.append(dict(scene=kind, seconds=round(len(q) / 50, 1), kept=bool(ok),
                           inside_cm=round(float(f['inside'].max()) * 100, 1),
                           air_frames=int(bad['air'].sum()), air_worst_cm=round(float(f['air'].max()) * 100, 1),
                           teleport_frames=int(bad['teleport'].sum())))
        if not ok:
            continue
        name = 'chain_%03d_%s' % (len(clips), kind)
        xyzw = np.concatenate([q[:, 4:7], q[:, 3:4]], 1)
        clips[name] = dict(root_trans_offset=q[:, :3].astype(np.float32), root_rot=xyzw.astype(np.float32),
                           dof=q[:, 7:].astype(np.float32), fps=50)
        props_all[name] = props
        print('%s kept (%.1f s, inside %.1f cm)' % (name, len(q) / 50, f['inside'].max() * 100), flush=True)
    joblib.dump(clips, os.path.join(OUT, 'x2_chains.pkl'))
    json.dump(props_all, open(os.path.join(OUT, 'x2_chains_props.json'), 'w'), indent=1)
    json.dump(report, open(os.path.join(OUT, 'x2_chains_report.json'), 'w'), indent=1)
    kept = [r for r in report if r['kept']]
    print('kept %d of %d tries, %.1f min of motion' % (len(kept), len(report), sum(r['seconds'] for r in kept) / 60))


if __name__ == '__main__':
    main()
