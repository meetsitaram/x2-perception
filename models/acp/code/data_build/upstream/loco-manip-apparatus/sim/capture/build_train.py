"""Pico crossings -> SuperMimic training clips + props (the joblib pkl / props JSON the trainer loads).

    python sim/capture/build_train.py C:\\Users\\PC\\projects\\pico-captures\\2026-10-29

Takes every crossing retarget_x2.py placed a prop for (<session>/x2 = the captured 4x4, x2_3.8cm = a 2x2 where the
human crossed the 4x4) where the X2's legs clear the prop and both feet end on the far side, and cuts it to WINDOW_S
either side of the crossing (inside the crossing's own window). Rejected: a stream gap longer than GAP_S in the cut
(Wi-Fi dropouts: the skeleton jumps), or a joint faster than QD_MAX. Resampled to 50 Hz (the trainer's clips) and
levelled (lower toe at its standing height: Pico's floor drifts +-2 cm between takes).

Writes build/train/pico_<session>.pkl and _props.json (clip -> [box: kind, center, half, yaw], as
x2_chains_far_props.json), and _split.json (train / test clip names: every TEST_EVERY-th take is held out).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as sRot, Slerp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_props import X2REC  # noqa: E402,F401  (puts the X2 recorder on sys.path)
from retarget_x2 import check_model  # noqa: E402
from x2_recorder.pico_mocap.x2_ik import FOOT_TARGET_Z, build_mocap_model  # noqa: E402

FPS = 50
WINDOW_S = 3.0
GAP_S = 0.15
QD_MAX = 25.0      # rad/s
SMOOTH_SIGMA_S = 0.04   # s: pico_mocap.SMOOTH_SIGMA_S
TEST_EVERY = 4
# Joints Pico does not track, zeroed: the retargeted head sits at its yaw limit and the wrists hold the rest-pose
# calibration offsets (wrist yaw -1.0, pitch 0.55 rad) through every take. A tracker that never saw those poses spun
# the X2 round within 0.2 s on every clip; with them at 0 the same checkpoint completed 9 of 10.
UNTRACKED = ('left_wrist_yaw_joint', 'left_wrist_pitch_joint', 'left_wrist_roll_joint', 'right_wrist_yaw_joint',
             'right_wrist_pitch_joint', 'right_wrist_roll_joint', 'head_yaw_joint', 'head_pitch_joint')
OUT = Path(__file__).resolve().parents[2] / 'build' / 'train'
PROPS = {'x2': '4x4', 'x2_3.8cm': '2x2'}


def resample(t: np.ndarray, q: np.ndarray, sigma_s: float = SMOOTH_SIGMA_S) -> np.ndarray:
    """Irregular device-rate qpos -> 50 Hz, Gaussian-smoothed: the X2 recorder's resample_smooth for saved takes
    (scripts/pico_mocap.py; Pico body updates arrive every 12-45 ms with pelvis jitter, which left the root
    acceleration 2.7x the Bones chains' and the tracker falling within 1 s)."""
    tu = np.arange(t[0], t[-1], 1.0 / FPS)
    out = np.stack([np.interp(tu, t, q[:, j]) for j in range(q.shape[1])], 1)
    wxyz = q[:, 3:7]
    rot = sRot.from_quat(wxyz[:, [1, 2, 3, 0]])
    out[:, 3:7] = Slerp(t, rot)(tu).as_quat()[:, [3, 0, 1, 2]]
    sig = sigma_s * FPS
    k = np.arange(-int(3 * sig), int(3 * sig) + 1)
    w = np.exp(-0.5 * (k / sig) ** 2)
    w /= w.sum()
    out[:, 3:7] *= np.sign((out[:, 3:7] * out[:1, 3:7]).sum(1, keepdims=True))   # one quaternion hemisphere
    pad = np.pad(out, ((len(k) // 2, len(k) // 2), (0, 0)), mode='edge')
    out = np.stack([np.convolve(pad[:, j], w, mode='valid') for j in range(out.shape[1])], 1)
    out[:, 3:7] /= np.linalg.norm(out[:, 3:7], axis=1, keepdims=True)
    return out


def level(m, d, qs: np.ndarray) -> float:
    """Root height shift that puts the lower toe at its standing-flat height (p10 over the clip): Pico's floor drifts
    +-2 cm between takes and the retarget carries it."""
    z = []
    for q in qs[::2]:
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        z.append(min(d.xpos[m.body(b).id][2] for b in ('left_toe_link', 'right_toe_link')))
    return FOOT_TARGET_Z - float(np.percentile(z, 10))


def clearance(qs: np.ndarray, prop: dict, cx: float, cy: float, yaw: float) -> float:
    """Least distance between the X2's leg meshes and the prop over the clip (retarget_x2's check)."""
    cm, gp, legs = check_model(prop)
    cm.geom_pos[gp, :2] = cx, cy
    cm.geom_quat[gp] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    cd, ft, best = mujoco.MjData(cm), np.zeros(6), 9.0
    for q in qs:
        cd.qpos[:] = q
        mujoco.mj_kinematics(cm, cd)
        best = min(best, min(mujoco.mj_geomDistance(cm, cd, g, gp, 0.5, ft) for g in legs))
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('session', type=Path)
    ap.add_argument('--window', type=float, default=WINDOW_S, help='s either side of the crossing')
    ap.add_argument('--gap', type=float, default=GAP_S, help='longest stream gap allowed in a clip (s); shorter ones are '
                    'interpolated, and the clearance check runs on the interpolated clip')
    ap.add_argument('--logs', nargs='+', default=list(PROPS.values()), choices=list(PROPS.values()),
                    help='which logs to build clips for (default: both)')
    a = ap.parse_args()
    prop = json.load(open(a.session / 'props.json'))['props'][0]
    takes = sorted(p.stem for p in (a.session / 'x2').glob('*.npz'))
    test_takes = set(takes[TEST_EVERY - 1::TEST_EVERY])
    clips, props, split, rej = {}, {}, {'train': [], 'test': []}, {}
    mm = build_mocap_model()
    md = mujoco.MjData(mm)
    shifts = []
    for sub, kind in PROPS.items():
        if kind not in a.logs:
            continue
        for f in sorted((a.session / sub).glob('*.npz')):
            z = np.load(f)
            t, q = z['t'], z['qpos']
            for k, (start, end, cx, cy, yaw, clear, pen, shift, *_) in enumerate(z['events']):
                if clear < 0:
                    rej['collides'] = rej.get('collides', 0) + 1
                    continue
                w = np.where((t >= max(start, 0.0)) & (t < end))[0]
                mid = 0.5 * (t[w[0]] + t[w[-1]])
                sel = w[(t[w] >= mid - a.window) & (t[w] <= mid + a.window)]
                if len(sel) < 10 or np.diff(t[sel]).max() > a.gap:
                    rej['stream gap'] = rej.get('stream gap', 0) + 1
                    continue
                qs = resample(t[sel], q[sel])
                qd = np.abs(np.diff(qs[:, 7:], axis=0)) * FPS
                if qd.max() > QD_MAX:
                    rej['joint speed'] = rej.get('joint speed', 0) + 1
                    continue
                n = np.array([-np.sin(yaw), np.cos(yaw)])
                if np.sign((qs[0, :2] - (cx, cy)) @ n) == np.sign((qs[-1, :2] - (cx, cy)) @ n):
                    rej['does not cross'] = rej.get('does not cross', 0) + 1
                    continue
                dz = level(mm, md, qs)
                qs[:, 2] += dz
                h = prop['size_m'][2] if kind == '4x4' else 0.038
                if clearance(qs, dict(prop, size_m=[prop['size_m'][0], h, h]), cx, cy, yaw) < 0:
                    rej['collides after levelling'] = rej.get('collides after levelling', 0) + 1
                    continue
                shifts.append(dz)
                for j in UNTRACKED:
                    qs[:, mm.joint(j).qposadr[0]] = 0.0
                name = f'pico_{f.stem[7:]}_{k}_{kind}'
                xyzw = qs[:, [4, 5, 6, 3]]
                clips[name] = dict(root_trans_offset=qs[:, :3].astype(np.float32), root_rot=xyzw.astype(np.float32),
                                   dof=qs[:, 7:].astype(np.float32), fps=FPS)
                props[name] = [dict(kind='step', center=[float(cx), float(cy), h / 2],
                                    half=[prop['size_m'][0] / 2, h / 2, h / 2], yaw=float(yaw))]
                split['test' if f.stem in test_takes else 'train'].append(name)
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / f"pico_{a.session.name}_{'_'.join(sorted(a.logs))}"
    joblib.dump({n: clips[n] for n in split['train']}, f'{stem}_train.pkl')
    joblib.dump({n: clips[n] for n in split['test']}, f'{stem}_test.pkl')
    json.dump(props, open(f'{stem}_props.json', 'w'), indent=1)
    json.dump(split, open(f'{stem}_split.json', 'w'), indent=1)
    for part in ('train', 'test'):
        names = split[part]
        secs = sum(len(clips[n]['dof']) for n in names) / FPS
        print('%s: %d clips (%d 4x4, %d 2x2), %.0f s' % (part, len(names), sum(n.endswith('4x4') for n in names),
                                                       sum(n.endswith('2x2') for n in names), secs))
    print('rejected:', rej)
    print('floor levelling: root shifted %.1f .. %.1f cm (median %.1f)' % tuple(100 * np.percentile(shifts, [0, 100, 50])))
    print('->', f'{stem}_train.pkl', f'{stem}_test.pkl', f'{stem}_props.json')


if __name__ == '__main__':
    main()
