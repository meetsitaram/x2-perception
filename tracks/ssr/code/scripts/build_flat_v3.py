"""Flat motion prior v3 (coordinator, 2026-11-12): the flat set rebuilt from the Hiking track's cleaned walking file
(data/motions/hiking_x2/x2_parkour_motion_without_run_v2_retargetted.npz, one concatenated sequence, clips chained in xy/yaw).
Clip seams are found from discontinuities (joint jump > 0.25 rad in one 50 Hz frame, pelvis height jump > 2 cm, pelvis
acceleration > 40 m/s^2) and windows never cross them; pieces whose pelvis height range exceeds 8 cm (the stair takes) are left
out of the flat set. Writes data/amp_v3.npz = data/amp_v2.npz with "flat" replaced, and prints the speed tables before/after.
CPU only. Run from tracks/ssr/code:  CUDA_VISIBLE_DEVICES= ../env/venv/bin/python scripts/build_flat_v3.py
Motion data: BONES-SEED by Bones Studio (bones.studio); Pico captures by the user."""
import pathlib, sys, json
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import mujoco, numpy as np
import build_amp as B
from ssr import x2 as X
SRC = "/mnt/filesystem-g4/data/motions/hiking_x2/x2_parkour_motion_without_run_v2_retargetted.npz"
x2 = X.build(foot_boxes=False)
m = x2.model
data = mujoco.MjData(m)
ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in X.HAND_BODIES + X.FOOT_BODIES]
names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
d = np.load(SRC, allow_pickle=True)
assert float(d["framerate"]) == 50.0
assert list(d["joint_names"]) == names, "joint order differs from the SSR model"
P, Q, J = d["base_pos_w"].astype(np.float64), d["base_quat_w"].astype(np.float64), d["joint_pos"].astype(np.float64)   # quat wxyz
n = len(P)
jump = np.abs(np.diff(J, axis=0)).max(1) > 0.25
zj = np.abs(np.diff(P[:, 2])) > 0.02
acc = np.r_[0.0, np.linalg.norm(np.diff(P, 2, axis=0), axis=1) * 2500, 0.0][:-1] > 40.0
seam = jump | zj | acc                                     # seam between frame i and i+1
cuts = np.r_[0, np.nonzero(seam)[0] + 1, n]
pieces, dropped_short, dropped_stairs = [], 0, 0
for a, b in zip(cuts[:-1], cuts[1:]):
    a2, b2 = a + 2, b - 2                                  # stay off the seam frames
    if b2 - a2 < 50:                                       # under 1 s
        dropped_short += 1
        continue
    if np.ptp(P[a2:b2, 2]) > 0.08:
        dropped_stairs += 1
        continue
    pieces.append((a2, b2))
fr, starts, tot = [], [], 0
for a, b in pieces:
    f = B.frames(x2, data, ids, P[a:b], Q[a:b] / np.linalg.norm(Q[a:b], axis=1, keepdims=True), J[a:b])
    if not np.isfinite(f).all() or np.abs(f[:, 30:51]).max() > 25:
        continue
    fr.append(f); starts.append(tot); tot += len(f)
z = dict(np.load(ROOT / "data/amp_v2.npz"))
old, old_st = z["flat"], z["flat_starts"]
z["flat"], z["flat_starts"] = np.concatenate(fr).astype(np.float32), np.array(starts)
np.savez(ROOT / "data/amp_v3.npz", **z)


def win_speed(a, st, k=5):
    """5-frame window mean planar speed (as fed to the discriminator), windows inside one clip only."""
    ends = list(st[1:]) + [len(a)]
    out = []
    for s0, e0 in zip(st, ends):
        sp = np.linalg.norm(a[s0:e0, 3:5], axis=1)
        if len(sp) >= k:
            out.append(np.convolve(sp, np.ones(k) / k, mode="valid"))
    return np.concatenate(out)


rep = {"seams_found": int(seam.sum()), "pieces_kept": len(fr), "dropped_short": dropped_short, "dropped_stairs_or_height": dropped_stairs}
for tag, a, st in (("v2_flat (r5)", old, old_st), ("v3_flat (Hiking file)", z["flat"], z["flat_starts"])):
    w = win_speed(a, st)
    mv = w[w >= 0.2]
    rep[tag] = dict(seconds=round(len(a) / 50, 1), all_windows_p10_p50_p90=np.percentile(w, [10, 50, 90]).round(2).tolist(),
                    share_under_0p2=round(float((w < 0.2).mean()), 3),
                    moving_windows_p10_p50_p90=np.percentile(mv, [10, 50, 90]).round(2).tolist(),
                    moving_share_over_0p8=round(float((mv > 0.8).mean()), 3), moving_share_over_1p0=round(float((mv > 1.0).mean()), 3),
                    seconds_by_speed_lt0p2_0p4_0p6_0p8_1p0_1p2_gt=(np.histogram(w, [0, .2, .4, .6, .8, 1.0, 1.2, 9])[0] / 50).round(1).tolist())
print(json.dumps(rep, indent=1))
(ROOT.parent / "eval" / "flat_prior_v3.json").write_text(json.dumps(rep, indent=1))
