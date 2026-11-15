"""Motion reference v3 for v4j (CPU): v2 (x2_parkour_motion_without_run_v2_retargetted.npz) plus the user's Pico takes:
the 9 sideways clips of tp_t4x4u_y20.pkl (body-frame lateral share of travel > 0.6) and the turning stair takes of
tp_stairs_all_v1.pkl (in_set and checks ok, class one of: down then turn, side approach then up, C (up/down), S (up/down)).
Same conversion as build_x2_motion.py: pelvis base, root_rot xyzw -> wxyz, head and wrist joints zeroed, chained in xy / yaw at
the seam, joints clipped to the MJCF limits. Whole clips (no standing cut: sideways steps are 0.15-0.3 m/s).
usage: append_motion_v3.py <mjcf> <v2 npz> <terrain_park_train_sets dir> <out npz>"""
import json, sys
import joblib, mujoco, numpy as np

MJCF, V2, SRC, OUT = sys.argv[1:5]
m = mujoco.MjModel.from_xml_path(MJCF)
names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
v2 = np.load(V2)
assert list(v2["joint_names"]) == names, "joint order differs from the v2 file"
zero = [i for i, n in enumerate(names) if "head" in n or "wrist" in n]
TURN = {"down then turn", "side approach then up", "C (up)", "C (down)", "S (up)", "S (down)"}


def yaw_xyzw(q):
    x, y, z, w = q.T
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def yaw_of(q):  # wxyz
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def qmul(a, b):
    w1, x1, y1, z1 = a.T; w2, x2, y2, z2 = b.T
    return np.stack([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2, w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2], -1)


def lateral_share(v):
    p = np.asarray(v["root_trans_offset"]); ya = yaw_xyzw(np.asarray(v["root_rot"]))
    dp = np.diff(p[:, :2], axis=0); c, s = np.cos(ya[:-1]), np.sin(ya[:-1])
    fwd, lat = dp[:, 0] * c + dp[:, 1] * s, -dp[:, 0] * s + dp[:, 1] * c
    return np.abs(lat).sum() / (np.abs(lat).sum() + np.abs(fwd).sum() + 1e-9)


side = joblib.load(f"{SRC}/tp_t4x4u_y20.pkl")
side_keys = sorted(k for k, v in side.items() if lateral_share(v) > 0.6)
stairs = joblib.load(f"{SRC}/tp_stairs_all_v1.pkl")
meta = json.load(open(f"{SRC}/tp_stairs_all_v1_meta.json"))
import os
EXCL = set(os.environ.get("V3_EXCLUDE", "tp_stairs_113941").split(","))  # 2026-11-12: 113941 retarget artefact (frozen right hip yaw 2.38 rad, pelvis pitched 46-52 deg)
stair_keys = sorted(k for k in stairs if k not in EXCL and k in meta and meta[k].get("in_set") and meta[k].get("checks", {}).get("ok") and meta[k].get("cls") in TURN)
clips = [side[k] for k in side_keys] + [stairs[k] for k in stair_keys]
for v in clips:
    assert int(v["fps"]) == 50 and np.asarray(v["dof"]).shape[1] == 31

P, Q, J = [v2["base_pos_w"].astype(np.float64)], [v2["base_quat_w"].astype(np.float64)], [v2["joint_pos"].astype(np.float64)]
end_xy, end_yaw = P[0][-1, :2].copy(), yaw_of(Q[0][-1])
for v in clips:
    p = np.asarray(v["root_trans_offset"], np.float64).copy()
    q = np.asarray(v["root_rot"], np.float64)[:, [3, 0, 1, 2]]
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    dyaw = end_yaw - yaw_of(q[0])
    c, s = np.cos(dyaw), np.sin(dyaw)
    xy = (p[:, :2] - p[0, :2]) @ np.array([[c, s], [-s, c]]) + end_xy
    p[:, :2] = xy
    q = qmul(np.tile([np.cos(dyaw / 2), 0, 0, np.sin(dyaw / 2)], (len(q), 1)), q)
    j = np.asarray(v["dof"], np.float64).copy(); j[:, zero] = 0.0
    P.append(p); Q.append(q); J.append(j)
    end_xy, end_yaw = xy[-1], yaw_of(q[-1])
P, Q, J = np.concatenate(P), np.concatenate(Q), np.concatenate(J)
for k in range(1, len(Q)):
    if (Q[k] * Q[k - 1]).sum() < 0:
        Q[k] *= -1
lo, hi = m.jnt_range[m.jnt_type == mujoco.mjtJoint.mjJNT_HINGE].T
n2 = len(v2["joint_pos"])
viol = int(((J[n2:] < lo - 1e-3) | (J[n2:] > hi + 1e-3)).any(1).sum())
J = np.clip(J, lo, hi)
np.savez(OUT, framerate=np.float64(50.0), joint_names=np.array(names), joint_pos=J.astype(np.float32), base_pos_w=P.astype(np.float32), base_quat_w=Q.astype(np.float32))
# checks on the appended part
new = slice(n2, None)
pz2 = np.percentile(v2["base_pos_w"][:, 2], [5, 50, 95]).round(3).tolist()
pzn = np.percentile(P[new, 2] - 0, [5, 50, 95]).round(3).tolist()
ya = np.array([yaw_of(q) for q in Q[new]])
dp = np.diff(P[new, :2], axis=0); c, s = np.cos(ya[:-1]), np.sin(ya[:-1])
lat = np.abs(-dp[:, 0] * s + dp[:, 1] * c).sum() / (np.abs(dp).sum(1).sum() + 1e-9)
seam = np.abs(np.diff(P[:, :2], axis=0)).max(1) * 50
print(json.dumps(dict(v2_frames=n2, added_frames=len(J) - n2, added_s=(len(J) - n2) / 50, sideways_clips=len(side_keys),
                      sideways_s=sum(len(side[k]["dof"]) for k in side_keys) / 50, stair_turn_clips=len(stair_keys),
                      stair_classes=sorted({meta[k]["cls"] for k in stair_keys}), joint_limit_frames_added=viol,
                      pelvis_z_p5_50_95_v2=pz2, pelvis_z_p5_50_95_added=pzn, lateral_share_sideways_part=None,
                      max_planar_step_speed=float(seam.max()), total_s=len(J) / 50)))
# lateral share of the sideways part alone (in the chained file)
ns = sum(len(side[k]["dof"]) for k in side_keys)
sl = slice(n2, n2 + ns)
ya = np.array([yaw_of(q) for q in Q[sl]]); dp = np.diff(P[sl, :2], axis=0); c, s = np.cos(ya[:-1]), np.sin(ya[:-1])
print("lateral share of travel, sideways part as chained: %.2f" % (np.abs(-dp[:, 0] * s + dp[:, 1] * c).sum() /
      (np.abs(-dp[:, 0] * s + dp[:, 1] * c).sum() + np.abs(dp[:, 0] * c + dp[:, 1] * s).sum())))
