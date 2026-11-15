"""SSR check of the five Hiking port-audit points (coordinator, 2026-11-12 03:20). CPU, read-only. Run from tracks/ssr/code:
CUDA_VISIBLE_DEVICES= ../env/venv/bin/python scripts/port_audit.py > ../eval/port_audit.json"""
import pathlib, sys, json, re
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np, mujoco, joblib
import build_amp as B
from ssr import x2 as X

x2 = X.build(foot_boxes=False)          # the training model's robot (terrain guards off; they are terrain-only spheres)
m = x2.model
d = mujoco.MjData(m)
hn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
n21 = [hn[i] for i in x2.idx21]
out = {"policy_and_discriminator_joints": n21,
       "not_in_discriminator": [n for n in hn if n not in n21],
       "discriminator_also_sees": "base gravity, base lin/ang velocity, 21 joint positions and velocities, wrist and foot positions in the base frame (63-d)",
       "terminate_bodies": list(getattr(X, "TERMINATE_BODIES", []))}
z = np.load(ROOT / "data/amp_v2.npz")
q0 = x2.default_q[x2.idx21]
scale = np.asarray(x2.action_scale)[x2.idx21]
lo, hi = x2.q_lo[x2.idx21], x2.q_hi[x2.idx21]

# (2) prior mean pose vs default, (3) ranges in action units and the action-rate cost of following the reference, (5) limits
stats = {}
for c in ("flat", "ramp", "stairs", "beam", "platform"):
    a = z[c]
    mv = np.linalg.norm(a[:, 3:5], axis=1) >= 0.2
    a = a[mv]
    q = a[:, 9:30] + q0
    qd = a[:, 30:51]
    row = {}
    for j, n in enumerate(n21):
        r = dict(default=round(float(q0[j]), 3), mean=round(float(q[:, j].mean()), 3), p5=round(float(np.percentile(q[:, j], 5)), 3),
                 p95=round(float(np.percentile(q[:, j], 95)), 3), lo=round(float(lo[j]), 3), hi=round(float(hi[j]), 3),
                 frames_outside_limits=round(float(((q[:, j] < lo[j]) | (q[:, j] > hi[j])).mean()), 4),
                 frames_within_5pct_of_limit=round(float(((q[:, j] < lo[j] + 0.05 * (hi[j] - lo[j])) | (q[:, j] > hi[j] - 0.05 * (hi[j] - lo[j]))).mean()), 4))
        if scale is not None:
            r["p5_p95_range_in_action_units"] = round(float((r["p95"] - r["p5"]) / scale[j]), 2)
        row[n] = r
    if scale is not None:   # action change per 50 Hz step needed to follow the reference with PD targets on the reference: da = qd * dt / scale
        da = qd * 0.02 / scale
        row["_action_rate_cost_per_step_following_reference"] = round(float(0.12 * (da ** 2).sum(-1).mean() / 21), 4)
    stats[c] = row
out["reference_vs_default"] = stats
out["action_scale"] = {n: round(float(s), 4) for n, s in zip(n21, scale)} if scale is not None else "not found"

# (1) and (4): self-contacts on full reference poses (BONES-SEED flat clips, all 31 joints) and at the default pose
src = joblib.load("/mnt/filesystem-g4/data/motions/ssr_prior_sources/apparatus_build_train/x2_loco_flat_train.pkl")
keys = sorted(k for k in src if B.FLAT_KEEP.search(k) and not B.FLAT_DROP.search(k))
body = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g]))


def self_contacts(qpos):
    d.qpos[:] = qpos
    mujoco.mj_forward(m, d)
    res = []
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
        if b1 > 0 and b2 > 0:
            res.append((tuple(sorted((body(c.geom1), body(c.geom2)))), float(c.dist)))
    return res


qd0 = np.zeros(m.nq); qd0[2] = 1.0; qd0[3] = 1.0; qd0[7:] = x2.default_q
out["self_contacts_default_pose"] = [(p, round(dd * 1000, 1)) for p, dd in self_contacts(qd0)]
pairs, nfr, nfr_hit, hand_leg, legs_only_hit = {}, 0, 0, 0, 0
arm_names = [i for i, n in enumerate(hn) if re.search("shoulder|elbow|wrist", n)]
for k in keys[::3]:
    root, quat, dof = B.resample(src[k])
    for t in range(0, len(root), 10):
        qp = np.zeros(m.nq); qp[:3], qp[3:7], qp[7:] = root[t], quat[t], dof[t]
        cs = self_contacts(qp)
        nfr += 1
        if cs:
            nfr_hit += 1
        hl = False
        for p, dd in cs:
            e = pairs.setdefault(" / ".join(p), [0, 0.0])
            e[0] += 1; e[1] = min(e[1], dd)
            if any("wrist" in b or "elbow" in b for b in p) and any("hip" in b or "knee" in b or "pelvis" in b for b in p):
                hl = True
        hand_leg += hl
        qa = qp.copy(); qa[7 + np.array(arm_names)] = x2.default_q[arm_names]
        if self_contacts(qa):
            legs_only_hit += 1
out["reference_self_contacts"] = dict(frames=nfr, frames_with_any=round(nfr_hit / nfr, 3), frames_hand_or_elbow_vs_leg_or_pelvis=round(hand_leg / nfr, 3),
                                      frames_with_any_when_arms_set_to_default=round(legs_only_hit / nfr, 3),
                                      pairs={p: dict(share=round(v[0] / nfr, 3), deepest_mm=round(-v[1] * 1000, 1)) for p, v in sorted(pairs.items(), key=lambda x: -x[1][0])[:12]})
# torso lean in the references: waist pitch (held at 0 in SSR's sim, not a policy joint) and pelvis pitch
wp = [i for i, n in enumerate(hn) if "waist_pitch" in n]
allq = np.concatenate([B.resample(src[k])[2] for k in keys[::3]])
out["reference_waist_pitch_rad"] = dict(mean=round(float(allq[:, wp].mean()), 3), p95=round(float(np.percentile(allq[:, wp], 95)), 3)) if wp else None
g = z["flat"][:, :3]
out["reference_pelvis_tilt_deg_flat_moving"] = round(float(np.degrees(np.arcsin(np.clip(np.linalg.norm(g[:, :2], axis=1), 0, 1))).mean()), 1)
print(json.dumps(out, indent=1))
