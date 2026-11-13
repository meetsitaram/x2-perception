"""Arm clearance data fix (coordinator, 2026-11-12): the X2 references put an arm inside its thigh or the torso on 16-91 %
of frames, and the simulator collides the robot with itself. Per frame and side, the smallest outward shoulder roll
(then, if the roll is capped, elbow extension and outward shoulder yaw) that gives at least CLEAR between that arm's geoms and the thigh /
torso geoms the simulator actually collides, with the training collision geometry; then smoothed in time (max filter +
Hann window over +-W frames, so the smoothed offset covers the need) and rate-limited to MAX_STEP rad per 50 Hz frame.
Writes cfg/armfix_<cfg>.pt (per-frame offsets for the four joints) that acp/data.py adds to the reference when the cfg
has "armfix"; prints the before / after table per set.
    CUDA_VISIBLE_DEVICES= python tools/arm_clear.py cfg/stage2d_sets.json [WORKERS=8]"""
import json, multiprocessing as mp, os, sys, time
import numpy as np, mujoco, torch
sys.path.insert(0, "/mnt/filesystem-g4/tracks/acp/code")
from acp.data import Dataset
from gear_sonic.supermimic.assets import build_x2
from scipy.spatial.transform import Rotation as Rot

CLEAR, ROLL_MAX, ELBOW_MAX, W, MAX_STEP = float(os.environ.get("ARM_CLEAR", "0.01")), 0.6, 0.6, 10, 0.10
cfg = sys.argv[1]; workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
x2 = build_x2("mc_20261024"); M = x2.model
JN = ["left_shoulder_roll_joint", "left_elbow_joint", "right_shoulder_roll_joint", "right_elbow_joint", "left_shoulder_yaw_joint", "right_shoulder_yaw_joint"]
JCOL = [x2.joint_names.index(n) for n in JN]                       # columns in the 31-joint q
SIGN = [+1.0, +1.0, -1.0, +1.0, +1.0, -1.0]                        # outward roll (left +, right -), elbow extension (+), outward shoulder yaw (left +, right -)
LO = np.array([M.joint(n).range[0] for n in JN]); HI = np.array([M.joint(n).range[1] for n in JN])
bname = lambda g: mujoco.mj_id2name(M, mujoco.mjtObj.mjOBJ_BODY, int(M.geom_bodyid[g])) or ""


def collides(a, b):
    if not ((M.geom_contype[a] & M.geom_conaffinity[b]) or (M.geom_contype[b] & M.geom_conaffinity[a])):
        return False
    ba, bb = int(M.geom_bodyid[a]), int(M.geom_bodyid[b])
    pa, pb = int(M.body_parentid[ba]), int(M.body_parentid[bb])
    for i in range(M.nexclude):
        sig = int(M.exclude_signature[i])
        if {sig >> 16, sig & 0xFFFF} == {ba, bb}:
            return False
    return ba != bb and not (pa == bb or pb == ba)


robot = [g for g in range(M.ngeom) if M.geom_bodyid[g] != 0 and (M.geom_contype[g] | M.geom_conaffinity[g])]
arm = {s: [g for g in robot if bname(g).startswith(s) and any(k in bname(g) for k in ("shoulder", "elbow", "wrist", "hand"))] for s in ("left", "right")}
body = {s: [g for g in robot if (bname(g).startswith(s) and any(k in bname(g) for k in ("hip", "knee"))) or any(k in bname(g) for k in ("torso", "pelvis", "waist"))] for s in ("left", "right")}
PAIRS = {s: [(a, b) for a in arm[s] for b in body[s] if collides(a, b)] for s in ("left", "right")}
# the shoulder links sit 0-7 mm from the torso hull at every pose (they surround the shoulder joint): for them only
# "not inside" (>= 0) is asked, 1 cm for every other pair
THR = {s: np.array([0.0 if ("torso" in bname(b) and any(k in bname(a) for k in ("shoulder_pitch", "shoulder_roll"))) else CLEAR for a, b in PAIRS[s]]) for s in PAIRS}
os.environ["ACP_NO_ARMFIX"] = "1"                                   # build the offsets from the unmodified references
data = Dataset(cfg, torch.device("cpu"), cache="")                 # no label cache: the cfg's cache must be built with the fix applied
mo = data.motion
P_ = mo.root_p.numpy(); Q_ = Rot.from_matrix(mo.root_R[torch.arange(mo.num_frames)].numpy()).as_quat()[:, [3, 0, 1, 2]]; J_ = mo.q.numpy()


def clip_job(c):
    d = mujoco.MjData(M); ft = np.zeros(6)
    a, b = int(mo.clip_start[c]), int(mo.clip_end[c]) + 1
    T = b - a

    def clearance(t, off, s):
        d.qpos[:3] = P_[a + t]; d.qpos[3:7] = Q_[a + t]; q = J_[a + t].copy()
        for k in range(6):
            q[JCOL[k]] = np.clip(q[JCOL[k]] + off[k], LO[k], HI[k])
        d.qpos[7:] = q
        mujoco.mj_kinematics(M, d)
        return min(mujoco.mj_geomDistance(M, d, g1, g2, 0.1, ft) - th for (g1, g2), th in zip(PAIRS[s], THR[s]))   # margin over each pair's threshold

    need = np.zeros((T, 6))
    before = np.zeros((T, 2))
    for t in range(T):
        for si, s in enumerate(("left", "right")):
            kr, ke, ky = 2 * si, 2 * si + 1, 4 + si
            c0 = clearance(t, np.zeros(6), s); before[t, si] = c0
            if c0 >= 0.0:
                continue
            off = np.zeros(6)
            ok = lambda r, e, y=0.0: (off.__setitem__(kr, SIGN[kr] * r), off.__setitem__(ke, SIGN[ke] * e), off.__setitem__(ky, SIGN[ky] * y), clearance(t, off, s) >= 0.0)[-1]
            if ok(ROLL_MAX, 0.0):                                  # bisection on the roll (clearance grows with outward roll)
                lo_, hi_ = 0.0, ROLL_MAX
                for _ in range(6):
                    mid = 0.5 * (lo_ + hi_)
                    lo_, hi_ = (lo_, mid) if ok(mid, 0.0) else (mid, hi_)
                need[t, kr] = hi_
            else:                                                  # roll capped (hand across the body): extend the elbow, then turn the upper arm out
                done = False
                for y in (0.0, 0.2, 0.4, 0.6, 0.8):
                    for e in np.linspace(0.1, ELBOW_MAX, 6):
                        if ok(ROLL_MAX, e, y):
                            need[t, kr], need[t, ke], need[t, ky] = ROLL_MAX, e, y; done = True; break
                    if done:
                        break
                if not done:
                    need[t, kr], need[t, ke], need[t, ky] = ROLL_MAX, ELBOW_MAX, 0.8
    # smooth: widen (max filter) then Hann average of the same width, then rate limit forward and backward
    win = np.hanning(2 * W + 3)[1:-1]; win /= win.sum()
    sm = np.zeros_like(need)
    for k in range(6):
        x = need[:, k]
        if not x.any():
            continue
        wide = np.max([np.roll(np.pad(x, W), s_)[W:-W] for s_ in range(-W, W + 1)], 0)
        y = np.convolve(np.pad(wide, W, mode="edge"), win, "valid")
        y = np.maximum(y, x)                                       # never less than the need
        for t in range(1, T):
            y[t] = max(y[t], y[t - 1] - MAX_STEP)
        for t in range(T - 2, -1, -1):
            y[t] = max(y[t], y[t + 1] - MAX_STEP)
        sm[:, k] = y
    off_signed = sm * np.array(SIGN)
    after = np.zeros((T, 2))
    for t in range(T):
        for si, s in enumerate(("left", "right")):
            after[t, si] = clearance(t, off_signed[t], s) if (sm[t] > 0).any() else before[t, si]
    # effective (limit-clipped) offset
    eff = np.clip(J_[a:b][:, JCOL] + off_signed, LO, HI) - J_[a:b][:, JCOL]
    return c, eff.astype(np.float32), before.astype(np.float32), after.astype(np.float32)


if __name__ == "__main__":
    t0 = time.time()
    print("pairs per side", {s: len(v) for s, v in PAIRS.items()}, "frames", mo.num_frames, flush=True)
    dq = np.zeros((mo.num_frames, 6), np.float32); bef = np.zeros((mo.num_frames, 2), np.float32); aft = np.zeros_like(bef)
    with mp.get_context("fork").Pool(workers) as pool:
        for c, eff, b_, a_ in pool.imap_unordered(clip_job, range(len(mo.names)), chunksize=4):
            a = int(mo.clip_start[c]); dq[a:a + len(eff)] = eff; bef[a:a + len(eff)] = b_; aft[a:a + len(eff)] = a_
    out = os.path.join(os.path.dirname(cfg), "armfix_" + os.path.basename(cfg).replace(".json", "") + ".pt")
    torch.save(dict(names=mo.names, num_frames=mo.num_frames, joints=JN, dq=torch.from_numpy(dq), clear=CLEAR, roll_max=ROLL_MAX,
                    elbow_max=ELBOW_MAX, window=W, max_step=MAX_STEP), out)
    rows = {}
    st = np.zeros(len(dq), bool); st[mo.clip_start.numpy()] = True
    qn = J_[:, JCOL] + dq
    step = np.abs(np.diff(qn, axis=0)) - np.abs(np.diff(J_[:, JCOL], axis=0)); step[st[1:]] = 0   # added joint speed per frame
    print(f"{'set':28s} frames | short of clearance by > 1 cm (either arm): before -> after | worst cm before -> after | below 1 cm clearance after | mean |roll offset| rad, max | elbow / yaw used frames | max added step rad/frame")
    for s, name in enumerate(data.set_names):
        f = (data.clip_set[mo.clip_id] == s).numpy()
        pb, pa = (-bef[f]).max(1), (-aft[f]).max(1)                  # shortfall below each pair's clearance (1 cm, or 0 at the shoulder)
        roll = np.abs(dq[f][:, [0, 2]]).max(1)
        yawu = int((np.abs(dq[f][:, [4, 5]]).max(1) > 1e-4).sum())
        rows[name] = dict(frames=int(f.sum()), inside_1cm_before=round(float((pb > 0.01).mean()), 3), inside_1cm_after=round(float((pa > 0.01).mean()), 3),
                          worst_cm_before=round(100 * float(pb.max()), 1), worst_cm_after=round(100 * float(pa.max()), 1),
                          below_clear_after=round(float((aft[f].min(1) < -1e-3).mean()), 3), roll_mean=round(float(roll.mean()), 3), roll_max=round(float(roll.max()), 3),
                          elbow_frames=int((np.abs(dq[f][:, [1, 3]]).max(1) > 1e-4).sum()), yaw_frames=yawu, max_step=round(float(step[f[1:]].max()) if f[1:].any() else 0.0, 3))
        r = rows[name]
        print(f"{name:28s} {r['frames']:6d} | {r['inside_1cm_before']:5.3f} -> {r['inside_1cm_after']:5.3f} | {r['worst_cm_before']:5.1f} -> {r['worst_cm_after']:5.1f} | "
              f"{r['below_clear_after']:5.3f} | {r['roll_mean']:.3f}, {r['roll_max']:.3f} | {r['elbow_frames']} / {r['yaw_frames']} | {r['max_step']:.3f}", flush=True)
    json.dump(dict(rows=rows, file=out, minutes=round((time.time() - t0) / 60, 1)), open(out.replace(".pt", "_report.json"), "w"), indent=1)
    print("wrote", out, f"{(time.time() - t0) / 60:.1f} min", flush=True)
