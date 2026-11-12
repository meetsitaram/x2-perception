"""X2 motion reference for the Hiking parkour task, in the released file's format (one concatenated retargetted.npz).
Flat walking/turning from the BONES-SEED X2 retargets plus Pico stair takes (terrain-park), about the paper's 380 s.
Base = pelvis (the X2 model's root). Head and wrist joints zeroed (not tracked). Clips chained in xy/yaw at the seams.
v2 (2026-11-11): standing and near-standing stretches are cut out of every clip (0.5 s mean planar speed under MIN_SPEED), because
v1 was 21% standing against 5% in the paper's G1 file and the X2 policy learned to stand. Each kept piece is at least MIN_PIECE_S.
Motion data: BONES-SEED by Bones Studio (bones.studio); Pico captures by the user.
"""
import re, sys, json
import joblib, mujoco, numpy as np
MJCF, FLAT, STAIRS, OUT = sys.argv[1:5]
FLAT_S, STAIRS_S = 300.0, 80.0
MIN_SPEED, MIN_PIECE_S = 0.2, 1.5
STAND_S = 15.0  # standing kept on purpose (4%; the paper's G1 file has 5%), so a zero command still has a style to match
STAND = []
m = mujoco.MjModel.from_xml_path(MJCF)
names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
assert len(names) == 31, names
zero = [i for i, n in enumerate(names) if "head" in n or "wrist" in n]
KEEP = re.compile(r"walk|turn|stride", re.I)
DROP = re.compile(r"injured|crouch|jump|zombie|drunk|hop|crawl|tiptoe|limp|bump|sneak|jog|run|back", re.I)
rng = np.random.default_rng(0)

def moving_pieces(v):
    """Split a clip into the stretches where the 0.5 s mean planar speed is at least MIN_SPEED."""
    p = np.asarray(v["root_trans_offset"], np.float64)
    sp = np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1) * 50
    sp = np.convolve(np.r_[sp, sp[-1]], np.ones(25) / 25, mode="same")
    ok, out, a = sp >= MIN_SPEED, [], None
    cut = lambda a, i: {k: (np.asarray(v[k])[a:i] if k != "fps" else v[k]) for k in ("root_trans_offset", "root_rot", "dof", "fps")}
    b = None
    for i, o in enumerate(np.r_[sp < 0.05, False]):  # standing stretches of 2 to 5 s go to the standing pool
        if o and b is None:
            b = i
        elif not o and b is not None:
            if 100 <= i - b:
                STAND.append(cut(b, min(i, b + 250)))
            b = None
    for i, o in enumerate(np.r_[ok, False]):
        if o and a is None:
            a = i
        elif not o and a is not None:
            if i - a >= MIN_PIECE_S * 50:
                out.append({k: (np.asarray(v[k])[a:i] if k != "fps" else v[k]) for k in ("root_trans_offset", "root_rot", "dof", "fps")})
            a = None
    return out


def pick(path, filt, budget):
    d = joblib.load(path)
    keys = sorted(k for k in d if filt(k))
    rng.shuffle(keys)
    out, tot, raw = [], 0.0, 0.0
    for k in keys:
        v = d[k]
        assert int(v["fps"]) == 50 and v["dof"].shape[1] == 31
        if len(v["dof"]) < 100:
            continue
        raw += len(v["dof"]) / 50
        for piece in moving_pieces(v):
            out.append(piece); tot += len(piece["dof"]) / 50
        if tot >= budget:
            break
    print("%s: %.0f s of clips read, %.0f s kept as moving pieces (%d pieces)" % (path.split("/")[-1], raw, tot, len(out)))
    return out, tot

def yaw_of(q):  # wxyz
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

def qmul(a, b):
    w1, x1, y1, z1 = a.T; w2, x2, y2, z2 = b.T
    return np.stack([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2, w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2], -1)

flat, tf = pick(FLAT, lambda k: KEEP.search(k) and not DROP.search(k), FLAT_S)
stairs, ts = pick(STAIRS, lambda k: True, STAIRS_S)
stand, tstand = [], 0.0
for piece in STAND:
    if tstand >= STAND_S:
        break
    stand.append(piece); tstand += len(piece["dof"]) / 50
print("standing kept on purpose: %.1f s in %d pieces" % (tstand, len(stand)))
clips = flat + stairs + stand
order = rng.permutation(len(clips))
P, Q, J = [], [], []
end_xy, end_yaw = np.zeros(2), 0.0
for i in order:
    v = clips[i]
    p = np.asarray(v["root_trans_offset"], np.float64).copy()
    q = np.asarray(v["root_rot"], np.float64)[:, [3, 0, 1, 2]]
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    dyaw = end_yaw - yaw_of(q[0])
    c, s = np.cos(dyaw), np.sin(dyaw)
    xy = (p[:, :2] - p[0, :2]) @ np.array([[c, s], [-s, c]]) + end_xy
    p[:, :2] = xy
    rz = np.tile([np.cos(dyaw / 2), 0, 0, np.sin(dyaw / 2)], (len(q), 1))
    q = qmul(rz, q)
    j = np.asarray(v["dof"], np.float64).copy(); j[:, zero] = 0.0
    P.append(p); Q.append(q); J.append(j)
    end_xy, end_yaw = xy[-1], yaw_of(q[-1])
P, Q, J = np.concatenate(P), np.concatenate(Q), np.concatenate(J)
for k in range(1, len(Q)):
    if (Q[k] * Q[k - 1]).sum() < 0:
        Q[k] *= -1
lo, hi = m.jnt_range[m.jnt_type == mujoco.mjtJoint.mjJNT_HINGE].T
print("joint limit violations (frames):", int(((J < lo - 1e-3) | (J > hi + 1e-3)).any(1).sum()), "of", len(J))
J = np.clip(J, lo, hi)
np.savez(OUT, framerate=np.float64(50.0), joint_names=np.array(names), joint_pos=J.astype(np.float32), base_pos_w=P.astype(np.float32), base_quat_w=Q.astype(np.float32))
v = np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1) * 50
print(json.dumps(dict(frames=len(J), seconds=len(J) / 50, flat_clips=len(flat), flat_s=tf, stair_clips=len(stairs), stair_s=ts,
    z_pct=np.percentile(P[:, 2], [0, 5, 50, 95, 100]).round(3).tolist(), speed_pct=np.percentile(v, [5, 50, 95, 100]).round(2).tolist())))
print(names)
