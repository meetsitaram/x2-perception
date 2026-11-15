"""Clip-quality check of the reference sets (part of the data build): joint-limit dwell, per-frame joint jumps, leg-branch
flips (hip yaw far from neutral), foot and shin penetration against the clip's own props.

    CUDA_VISIBLE_DEVICES= python tools/clip_quality.py cfg/stage1_sets.json OUT.json [--no-cache]

Per clip and per set. A clip is flagged when: a leg joint dwells at a limit for more than 10 % of its frames; any joint
jumps more than 0.35 rad between 50 Hz frames (17.5 rad/s); |hip yaw| exceeds 2.0 rad (the leg solved on the flipped
branch); a sole corner is more than 3 cm inside a prop or the floor on more than 5 % of frames; a shin point is more
than 2 cm inside a prop box on more than 2 % of frames.
"""
import json
import sys

import torch

sys.path.insert(0, "/mnt/filesystem-g4/tracks/acp/code")
from acp.data import Dataset  # noqa: E402
from gear_sonic.supermimic.kinematics import mv3  # noqa: E402

LIMIT_EPS, JUMP, YAW_FLIP, SOLE_PEN, SHIN_PEN = 0.02, 0.35, 2.0, 0.03, 0.02


def main():
    cfg, out = sys.argv[1], sys.argv[2]
    dev = torch.device("cpu")
    data = Dataset(cfg, dev, cache="" if "--no-cache" in sys.argv else cfg.replace(".json", ".json.pt").replace("cfg/", "cfg/labels_"))
    m, T = data.motion, data.terrain
    names = data.x2.joint_names
    lo, hi = torch.tensor(data.x2.q_lo, dtype=torch.float32), torch.tensor(data.x2.q_hi, dtype=torch.float32)
    legs = [i for i, n in enumerate(names) if any(k in n for k in ("hip", "knee", "ankle"))]
    yaw = [i for i, n in enumerate(names) if "hip_yaw" in n]
    F = m.num_frames
    q = m.q
    at_lim = ((q - lo).abs() < LIMIT_EPS) | ((hi - q).abs() < LIMIT_EPS) | (q < lo) | (q > hi)
    jump = torch.zeros(F, len(names))
    jump[1:] = (q[1:] - q[:-1]).abs()
    jump[m.clip_start] = 0.0
    link = data.kin.body_index(["left_ankle_roll_link", "right_ankle_roll_link", "left_knee_link", "right_knee_link"])
    sole_pen = torch.zeros(F)
    shin_pen = torch.zeros(F)
    for a in range(0, F, 20000):
        f = torch.arange(a, min(a + 20000, F))
        P, Rb, _, _ = data.kin.forward(*m.ref_state(f))
        clip = m.clip_id[f]
        c = P[:, link[:2], None, :] + mv3(Rb[:, link[:2], None, :, :], data.corners)        # (n, 2, 4, 3)
        h = T.height(c[..., 0].reshape(len(f), 8), c[..., 1].reshape(len(f), 8), clip)
        sole_pen[f] = (h - c[..., 2].reshape(len(f), 8)).amax(-1)                            # > 0: below the surface
        w = torch.linspace(0.15, 0.85, 5)[None, None, :, None]
        shin = P[:, link[2:], None, :] * (1 - w) + P[:, link[:2], None, :] * w               # knee -> ankle, 5 points per leg
        shin_pen[f] = T.penetration(shin.reshape(len(f), 10, 3), clip).amax(-1) if T.num_slots else 0.0
    rows, per_set = [], {}
    for c, name in enumerate(m.names):
        s, e = int(m.clip_start[c]), int(m.clip_end[c]) + 1
        n = e - s
        r = dict(clip=c, set=data.set_names[int(data.clip_set[c])], frames=n,
                 limit_dwell_leg=round(float(at_lim[s:e][:, legs].any(-1).float().mean()), 3),
                 max_jump_rad=round(float(jump[s:e].max()), 3), jump_frames=int((jump[s:e].amax(-1) > JUMP).sum()),
                 max_abs_hip_yaw=round(float(q[s:e][:, yaw].abs().max()), 2),
                 sole_pen_share=round(float((sole_pen[s:e] > SOLE_PEN).float().mean()), 3), sole_pen_max_m=round(float(sole_pen[s:e].max()), 3),
                 shin_pen_share=round(float((shin_pen[s:e] > SHIN_PEN).float().mean()), 3), shin_pen_max_m=round(float(shin_pen[s:e].max()), 3))
        r["flags"] = [k for k, v in (("limit_dwell", r["limit_dwell_leg"] > 0.10), ("jump", r["jump_frames"] > 0), ("leg_flip", r["max_abs_hip_yaw"] > YAW_FLIP),
                                      ("sole_penetration", r["sole_pen_share"] > 0.05), ("shin_penetration", r["shin_pen_share"] > 0.02)) if v]
        rows.append(r)
    for sname in data.set_names:
        rs = [r for r in rows if r["set"] == sname]
        n = len(rs)
        per_set[sname] = dict(clips=n, flagged=sum(bool(r["flags"]) for r in rs),
                              **{k: sum(k in r["flags"] for r in rs) for k in ("limit_dwell", "jump", "leg_flip", "sole_penetration", "shin_penetration")},
                              max_jump_rad=max(r["max_jump_rad"] for r in rs), max_abs_hip_yaw=max(r["max_abs_hip_yaw"] for r in rs),
                              worst_sole_pen_m=max(r["sole_pen_max_m"] for r in rs), worst_shin_pen_m=max(r["shin_pen_max_m"] for r in rs))
    json.dump(dict(thresholds=dict(limit_eps=LIMIT_EPS, jump=JUMP, yaw_flip=YAW_FLIP, sole_pen=SOLE_PEN, shin_pen=SHIN_PEN), per_set=per_set, clips=rows),
              open(out, "w"), indent=1)
    print(f"{'set':30s} clips flagged | limit jump flip sole shin | max jump  max |hip yaw|  worst sole pen  worst shin pen")
    for k, v in per_set.items():
        print(f"{k:30s} {v['clips']:5d} {v['flagged']:7d} | {v['limit_dwell']:5d} {v['jump']:4d} {v['leg_flip']:4d} {v['sole_penetration']:4d} {v['shin_penetration']:4d} | "
              f"{v['max_jump_rad']:8.2f} {v['max_abs_hip_yaw']:14.2f} {v['worst_sole_pen_m']:15.3f} {v['worst_shin_pen_m']:15.3f}")


if __name__ == "__main__":
    main()
