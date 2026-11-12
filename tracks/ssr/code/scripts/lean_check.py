"""Why the X2 settles leaning back with zero actions: CPU MuJoCo, flat floor, the same model build as training.

X2_PLANT=mc_20261024 python scripts/lean_check.py
Prints: centre of mass over the feet at the default pose, the settled pose under PD targets = default pose (zero action),
joint deflections under load, and the same with gravity-sag compensation added to the targets.
"""
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np

from ssr import x2 as X

x2 = X.build()
m = x2.model
names = x2.joint_names
bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
feet = [bid(b) for b in X.FOOT_BODIES]
out = {"plant": os.environ.get("X2_PLANT", "default file")}


def pitch_deg(d, body):
    R = d.xmat[body].reshape(3, 3)
    return float(np.degrees(np.arctan2(R[0, 2], R[2, 2])))          # + = leaning forward (body z axis tipped towards +x)


def place(d, q):
    d.qpos[:] = 0
    d.qpos[3] = 1.0
    d.qpos[7:] = q
    d.qpos[2] = 1.0
    mujoco.mj_kinematics(m, d)
    low = min(d.geom_xpos[g, 2] - m.geom_size[g, 0] for g in range(m.ngeom) if m.geom_bodyid[g] in feet and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE and m.geom_contype[g])
    d.qpos[2] -= low - 0.001
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)


def describe(d, tag):
    com = d.subtree_com[bid("pelvis")] if False else (m.body_mass[:, None] * d.xipos).sum(0) / m.body_mass.sum()
    ankle = d.xpos[feet].mean(0)
    sole_c = ankle[0] + X.SOLE_OFFSET_X
    q = d.qpos[7:]
    r = dict(com_ahead_of_sole_centre_m=round(float(com[0] - sole_c), 4), com_ahead_of_ankle_m=round(float(com[0] - ankle[0]), 4),
             sole_patch_m=[round(-X.SOLE_HALF[0], 4), round(X.SOLE_HALF[0], 4)], pelvis_pitch_deg=round(pitch_deg(d, bid("pelvis")), 2),
             torso_pitch_deg=round(pitch_deg(d, bid("torso_link")), 2), pelvis_height_m=round(float(d.qpos[2]), 4))
    out[tag] = r
    return r


d = mujoco.MjData(m)
q0 = x2.default_q.copy()
place(d, q0)
print("default pose, placed on the floor:", describe(d, "default_pose_kinematic"))
out["joints_at_or_past_limit_in_default_pose"] = [(n, round(float(q0[i]), 3), round(float(x2.q_lo[i]), 3), round(float(x2.q_hi[i]), 3)) for i, n in enumerate(names)
                                                    if q0[i] <= x2.q_lo[i] + 0.02 or q0[i] >= x2.q_hi[i] - 0.02]
print("joints within 0.02 rad of a limit at the default pose:", out["joints_at_or_past_limit_in_default_pose"])


def settle(target, tag, seconds=4.0):
    place(d, q0)
    d.ctrl[:] = target
    hist = []
    for k in range(int(seconds / m.opt.timestep)):
        mujoco.mj_step(m, d)
        if k % 100 == 0:
            hist.append(round(pitch_deg(d, bid("pelvis")), 2))
        if d.qpos[2] < 0.4:
            break
    r = describe(d, tag)
    r["fell"] = bool(d.qpos[2] < 0.4)
    r["pelvis_pitch_every_0p5s"] = hist[:9]
    defl = d.qpos[7:] - target
    tau = d.actuator_force
    r["deflection_deg"] = {n: round(float(np.degrees(defl[i])), 2) for i, n in enumerate(names) if abs(defl[i]) > np.radians(0.5)}
    r["torque_nm"] = {n: round(float(tau[i]), 1) for i, n in enumerate(names) if abs(tau[i]) > 3.0}
    r["past_limit"] = [n for i, n in enumerate(names) if d.qpos[7 + i] < x2.q_lo[i] or d.qpos[7 + i] > x2.q_hi[i]]
    print(tag, json.dumps(r))
    return defl.copy(), tau.copy()


defl, tau = settle(q0, "zero_action_settled")
# gravity-sag compensation: shift each target by the torque it carried at rest divided by its stiffness, iterated twice
tgt = q0.copy()
for it in range(3):
    place(d, q0)
    d.ctrl[:] = tgt
    for _ in range(int(3.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    tgt = tgt + (q0 - d.qpos[7:]) * 0.8
out["sag_compensation_offsets_deg"] = {n: round(float(np.degrees(tgt[i] - q0[i])), 2) for i, n in enumerate(names) if abs(tgt[i] - q0[i]) > np.radians(0.3)}
print("offsets added to the default targets:", out["sag_compensation_offsets_deg"])
settle(tgt, "sag_compensated_settled")
p = ROOT.parent / "eval" / ("lean_check_%s.json" % out["plant"])
p.write_text(json.dumps(out, indent=1))
print("wrote", p)
