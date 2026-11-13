"""Gait posture statistics from recorded flat-walk trajectories (eval monitor, 2026-11-12; asked after the user's visual review of
Hiking 4125: "the torso seems to lean backward while walking").

    python eval/gait_stats.py LADDER_TEST_DIR [LADDER_TEST_DIR ...]     (e.g. eval/out/hiking/x2_parkour_v4_4125/ladder_nohip/step1/flat_walk)

From traj/*.npz (qpos at 50 Hz, foot loads), frames while walking (pelvis speed over 0.2 m/s):
- torso and pelvis pitch from vertical, in degrees, FORWARD LEAN POSITIVE (backward lean negative): the body's z axis
  projected on the heading frame's x-z plane;
- stance width: lateral distance (heading frame) between the two feet at each touchdown (foot load rising past 40 N after < 10 N);
- crossing: share of touchdowns where the landing foot is on the other side of the stance foot (left foot right of the right foot
  or the reverse) in the heading frame.
MuJoCo forward kinematics on the eval's X2 model (the joint angles are the recorded ones; model differences do not matter here).
"""
import glob
import json
import math
import os
import pathlib
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import x2 as X  # noqa: E402

x = X.build(None)
m, d = x.model, mujoco.MjData(x.model)
torso, pelvis = m.body("torso_link").id, m.body("pelvis").id
feet = list(x.foot_ids)


def lean(R, yaw):
    zb = R[:, 2]
    fwd = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    return math.degrees(math.atan2(float(zb @ fwd), float(zb[2])))


rows = []
for td in sys.argv[1:]:
    tp, pp, widths, cross, n_td = [], [], [], 0, 0
    for f in sorted(glob.glob(os.path.join(td, "traj", "*.npz"))):
        z = np.load(f)
        q, L = z["qpos"], z["load"]
        prev = np.array([1e9, 1e9])
        for k in range(0, len(q)):
            d.qpos[:] = q[k]
            mujoco.mj_kinematics(m, d)
            yaw = X.yaw_of(q[k][3:7])
            v = np.linalg.norm(q[min(k + 5, len(q) - 1), :2] - q[max(k - 5, 0), :2]) / (0.02 * (min(k + 5, len(q) - 1) - max(k - 5, 0)) + 1e-9)
            if v > 0.2 and k % 2 == 0:
                tp.append(lean(d.xmat[torso].reshape(3, 3), yaw))
                pp.append(lean(d.xmat[pelvis].reshape(3, 3), yaw))
            for s in (0, 1):
                if L[k][s] > 40 and prev[s] < 10 and v > 0.2:
                    c, sn = math.cos(yaw), math.sin(yaw)
                    rel = d.xpos[feet[s]][:2] - d.xpos[feet[1 - s]][:2]
                    lat = -sn * rel[0] + c * rel[1]                # + = landing foot to the left of the other foot
                    widths.append(abs(lat))
                    n_td += 1
                    # feet[0] is the left foot in the eval model (x2.foot_ids order: left, right)
                    if (s == 0 and lat < 0) or (s == 1 and lat > 0):
                        cross += 1
                prev[s] = L[k][s] if L[k][s] > 40 or L[k][s] < 10 else prev[s]
    r = dict(run=td, frames=len(tp), torso_pitch_mean_deg=round(float(np.mean(tp)), 1) if tp else None,
             torso_pitch_p5_p95=[round(float(np.percentile(tp, 5)), 1), round(float(np.percentile(tp, 95)), 1)] if tp else None,
             pelvis_pitch_mean_deg=round(float(np.mean(pp)), 1) if pp else None,
             pelvis_pitch_p5_p95=[round(float(np.percentile(pp, 5)), 1), round(float(np.percentile(pp, 95)), 1)] if pp else None,
             touchdowns=n_td, stance_width_mean_m=round(float(np.mean(widths)), 3) if widths else None,
             crossing_share=round(cross / max(n_td, 1), 3), sign="forward lean positive, backward lean negative")
    rows.append(r)
    print(json.dumps(r))
