"""Re-solve the leg joints of reference clips whose leg IK took the flipped branch (hip yaw near +-pi, at its limit) or
jumps between branches: keep the root, the upper body and both FOOT poses (ankle-roll link position and orientation) of
the original clip, and find leg joints on the normal branch that are continuous in time.

    CUDA_VISIBLE_DEVICES= python tools/repair_legs.py IN.pkl OUT.pkl [--only-flagged]

Per clip and leg: minimise  |p_foot - p*|^2 / (1 cm)^2 + |R_foot - R*|_F^2 / (0.05)^2 + smoothness |q_t - q_t-1|^2
+ a weak pull of hip yaw / roll to neutral + a joint-limit barrier. Start: the default leg pose for a flipped leg, the
original joints (median-filtered over 5 frames) otherwise. Reports the remaining foot error and the largest per-frame jump.
"""
import sys

import joblib
import numpy as np
import torch

sys.path.insert(0, "/mnt/filesystem-g4/tracks/acp/code")
from gear_sonic.supermimic.assets import build_x2  # noqa: E402
from gear_sonic.supermimic.kinematics import X2Kinematics, quat_wxyz_to_mat  # noqa: E402

YAW_FLIP, JUMP = 2.0, 0.35


def main():
    src, out = sys.argv[1], sys.argv[2]
    only = "--only-flagged" in sys.argv
    torch.set_num_threads(8)
    x2 = build_x2()
    kin = X2Kinematics(x2.model, "cpu")
    names = x2.joint_names
    L = [i for i, n in enumerate(names) if n.startswith("left_") and any(k in n for k in ("hip", "knee", "ankle"))]
    R = [i for i, n in enumerate(names) if n.startswith("right_") and any(k in n for k in ("hip", "knee", "ankle"))]
    feet = kin.body_index(["left_ankle_roll_link", "right_ankle_roll_link"])
    lo, hi = torch.tensor(x2.q_lo, dtype=torch.float32), torch.tensor(x2.q_hi, dtype=torch.float32)
    q0 = torch.tensor(x2.default_q, dtype=torch.float32)
    yaw_i = [i for i, n in enumerate(names) if "hip_yaw" in n]
    roll_i = [i for i, n in enumerate(names) if "hip_roll" in n]
    clips = joblib.load(src)
    res, done = {}, 0
    stats = []
    for name, c in clips.items():
        q = torch.tensor(np.asarray(c["dof"]), dtype=torch.float32)
        T = q.shape[0]
        scale = float(c["fps"]) / 50.0
        jump = float((q[1:] - q[:-1]).abs().max()) / scale if T > 1 else 0.0
        flip = [bool((q[:, [i for i in leg if i in yaw_i]].abs() > YAW_FLIP).any()) for leg in (L, R)]
        if only and not (any(flip) or jump > JUMP):
            res[name] = c
            continue
        p = torch.tensor(np.asarray(c["root_trans_offset"]), dtype=torch.float32)
        xyzw = torch.tensor(np.asarray(c["root_rot"]), dtype=torch.float32)
        Rr = quat_wxyz_to_mat(torch.cat((xyzw[:, 3:4], xyzw[:, :3]), -1))
        z3, zq = torch.zeros(T, 3), torch.zeros(T, len(names))
        with torch.no_grad():
            P, Rb, _, _ = kin.forward(p, Rr, z3, z3, q, zq)
            pt, Rt = P[:, feet].clone(), Rb[:, feet].clone()
        init = q.clone()
        for leg, fl in zip((L, R), flip):
            if fl:
                init[:, leg] = q0[leg]
            else:                                           # median over 5 frames removes single-frame branch jumps
                pad = torch.cat((init[:2, leg], init[:, leg], init[-2:, leg]))
                init[:, leg] = pad.unfold(0, 5, 1).median(-1).values
        legs = L + R
        x = init[:, legs].clone().requires_grad_(True)
        opt = torch.optim.Adam([x], lr=0.03)
        for it in range(700):
            qq = q.clone()
            qq[:, legs] = x
            P, Rb, _, _ = kin.forward(p, Rr, z3, z3, qq, zq)
            e_p = ((P[:, feet] - pt) ** 2).sum((-1, -2)) / 0.01**2
            e_R = ((Rb[:, feet] - Rt) ** 2).sum((-1, -2, -3)) / 0.05**2
            smooth = ((x[1:] - x[:-1]) ** 2).sum(-1).mean() * (50.0 * scale) ** 2 * 2e-3 if T > 1 else 0.0
            accel = ((x[2:] - 2 * x[1:-1] + x[:-2]) ** 2).sum(-1).mean() * (50.0 * scale) ** 4 * 2e-7 if T > 2 else 0.0
            neutral = (qq[:, yaw_i] ** 2).sum(-1).mean() * 0.05 + (qq[:, roll_i] ** 2).sum(-1).mean() * 0.02
            lim = (torch.relu(lo[legs] + 0.02 - x) ** 2 + torch.relu(x - hi[legs] + 0.02) ** 2).sum(-1).mean() * 1e3
            loss = (e_p + e_R).mean() + smooth + accel + neutral + lim
            opt.zero_grad(); loss.backward(); opt.step()
            if it == 450:
                for g in opt.param_groups:
                    g["lr"] = 0.008
        with torch.no_grad():
            qn = q.clone()
            qn[:, legs] = torch.maximum(torch.minimum(x, hi[legs]), lo[legs])
            P, Rb, _, _ = kin.forward(p, Rr, z3, z3, qn, zq)
            ep = (P[:, feet] - pt).norm(dim=-1)
            tr = (Rb[:, feet] * Rt).sum((-1, -2))
            ang = torch.acos(((tr - 1) / 2).clamp(-1, 1))
            jn = float((qn[1:] - qn[:-1]).abs().max()) / scale if T > 1 else 0.0
        stats.append((name, any(flip), jump, jn, float(ep.mean()), float(ep.max()), float(ang.mean()), float(ang.max()), float(qn[:, yaw_i].abs().max())))
        res[name] = dict(c, dof=qn.numpy().astype(np.asarray(c["dof"]).dtype))
        done += 1
    joblib.dump(res, out)
    a = np.array([s[2:] for s in stats]) if stats else np.zeros((0, 7))
    print(f"{src} -> {out}: {done} of {len(clips)} clips re-solved ({sum(s[1] for s in stats)} with a flipped leg)")
    if len(a):
        print(f"  largest per-frame jump (rad at 50 Hz): before median {np.median(a[:, 0]):.2f} max {a[:, 0].max():.2f}; after median {np.median(a[:, 1]):.2f} max {a[:, 1].max():.2f}")
        print(f"  foot position error vs the original: mean {a[:, 2].mean() * 100:.1f} cm, worst frame {a[:, 3].max() * 100:.1f} cm; "
              f"foot orientation error: mean {np.degrees(a[:, 4].mean()):.1f} deg, worst frame {np.degrees(a[:, 5].max()):.1f} deg; max |hip yaw| after {a[:, 6].max():.2f} rad")
        bad = [s[0] for s in stats if s[5] > 0.05 or s[3] > JUMP]
        print(f"  clips still with a foot error over 5 cm or a jump over {JUMP} rad: {len(bad)}")


if __name__ == "__main__":
    main()
