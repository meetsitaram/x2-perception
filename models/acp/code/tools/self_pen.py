"""Self-penetration of the X2 reference clips with the training model's collision geometry (coordinator, Hiking port
audit 2026-11-12). (1) arm (shoulder..hand) geoms vs thigh / shin and vs torso / pelvis, per clip set, every 5th frame;
(2) leg geoms against each other and the pelvis at the default pose and at hip yaw +-0.2 / +-0.4. Also which robot-robot
geom pairs the simulator actually collides (contype / conaffinity / parent filter).
    CUDA_VISIBLE_DEVICES= python self_pen.py cfg/stage2c_sets.json OUT.json"""
import json, os, sys
import numpy as np, mujoco, torch
sys.path.insert(0, "/mnt/filesystem-g4/tracks/acp/code")
from acp.data import Dataset
from gear_sonic.supermimic.assets import build_x2

cfg, out = sys.argv[1], sys.argv[2]
x2 = build_x2("mc_20261024")
m = x2.model
d = mujoco.MjData(m)
bname = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or ""
robot = [g for g in range(m.ngeom) if m.geom_bodyid[g] != 0 and m.geom_conaffinity[g] | m.geom_contype[g]]
kind = lambda g: next((k for k, keys in (("arm", ("shoulder", "elbow", "wrist", "hand")), ("thigh", ("hip", "knee")),
                                          ("foot", ("ankle",)), ("torso", ("torso", "waist", "pelvis", "head", "base"))) if any(s in bname(g) for s in keys)), "other")
G = {k: [g for g in robot if kind(g) == k] for k in ("arm", "thigh", "foot", "torso")}
side = lambda g: "L" if bname(g).startswith("left") else "R" if bname(g).startswith("right") else "C"


def collides(a, b):
    """would MuJoCo generate a contact for this pair (bitmask and parent / same-body filters, explicit excludes)?"""
    if not ((m.geom_contype[a] & m.geom_conaffinity[b]) or (m.geom_contype[b] & m.geom_conaffinity[a])):
        return False
    ba, bb = int(m.geom_bodyid[a]), int(m.geom_bodyid[b])
    if ba == bb:
        return False
    pa, pb = int(m.body_parentid[ba]), int(m.body_parentid[bb])
    filt = not (m.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_FILTERPARENT)
    if filt and (pa == bb or pb == ba) and pa != 0 and pb != 0:
        return False
    for i in range(m.nexclude):
        sig = int(m.exclude_signature[i]); x, y = sig >> 16, sig & 0xFFFF
        if {x, y} == {ba, bb}:
            return False
    return True


pairs_arm = [(a, b) for a in G["arm"] for b in G["thigh"] + G["torso"]]
pairs_leg = [(a, b) for a in G["thigh"] + G["foot"] for b in G["thigh"] + G["foot"] + G["torso"] if a < b and not (kind(a) == kind(b) == "torso")
             and int(m.body_parentid[m.geom_bodyid[a]]) != int(m.geom_bodyid[b]) and int(m.body_parentid[m.geom_bodyid[b]]) != int(m.geom_bodyid[a])
             and m.geom_bodyid[a] != m.geom_bodyid[b]]
fromto = np.zeros(6)
dist = lambda a, b: mujoco.mj_geomDistance(m, d, a, b, 0.05, fromto)
rep = dict(robot_geoms=len(robot), arm=len(G["arm"]), thigh=len(G["thigh"]), torso=len(G["torso"]), foot=len(G["foot"]),
           arm_pairs_collided_in_sim=sum(collides(a, b) for a, b in pairs_arm), arm_pairs=len(pairs_arm),
           leg_pairs_collided_in_sim=sum(collides(a, b) for a, b in pairs_leg), leg_pairs=len(pairs_leg))
print(rep, flush=True)
# (2) leg overlaps at default pose and hip yaw
q0 = m.qpos0.copy(); q0[2] = 1.0
dq = dict(zip(x2.joint_names, range(7, 7 + len(x2.joint_names))))
default = getattr(x2, "default_q", None)
rep["leg_static"] = {}
for yaw in (0.0, 0.2, 0.4, -0.2, -0.4):
    q = q0.copy()
    if default is not None:
        q[7:7 + len(default)] = np.asarray(default)
    for s, sg in (("left", 1), ("right", -1)):
        q[dq[f"{s}_hip_yaw_joint"]] = sg * yaw                  # toes outward for +yaw
    d.qpos[:] = q; mujoco.mj_kinematics(m, d); mujoco.mj_collision(m, d)
    worst = sorted(((dist(a, b), bname(a), bname(b), collides(a, b)) for a, b in pairs_leg), key=lambda t: t[0])[:4]
    rep["leg_static"][f"hip_yaw_{yaw:+.1f}"] = [(round(100 * w, 1), x, y, c) for w, x, y, c in worst]
    print("hip yaw", yaw, rep["leg_static"][f"hip_yaw_{yaw:+.1f}"], flush=True)
# (1) arms in the reference clips
data = Dataset(cfg, torch.device("cpu"), cache=os.path.join(os.path.dirname(cfg), "labels_" + os.path.basename(cfg) + ".pt"))
mo = data.motion
from scipy.spatial.transform import Rotation as Rot
res = {}
for s, name in enumerate(data.set_names):
    cl = (data.clip_set == s).nonzero().squeeze(-1).tolist()
    fr = torch.cat([torch.arange(int(mo.clip_start[c]), int(mo.clip_end[c]) + 1, 5) for c in cl])
    p, R, _, _, q, _ = mo.ref_state(fr)
    quat = Rot.from_matrix(R.numpy()).as_quat()[:, [3, 0, 1, 2]]
    pen_thigh = np.zeros(len(fr)); pen_torso = np.zeros(len(fr))
    for i in range(len(fr)):
        d.qpos[:3] = p[i].numpy(); d.qpos[3:7] = quat[i]; d.qpos[7:] = q[i].numpy()
        mujoco.mj_kinematics(m, d)
        for a, b in pairs_arm:
            if side(b) != "C" and side(a) != side(b):
                continue                                          # left arm vs left thigh only
            v = dist(a, b)
            if v < 0:
                if kind(b) == "thigh": pen_thigh[i] = max(pen_thigh[i], -v)
                else: pen_torso[i] = max(pen_torso[i], -v)
    res[name] = dict(frames=len(fr), arm_thigh_share_over_1cm=round(float((pen_thigh > 0.01).mean()), 3), arm_thigh_share_over_2cm=round(float((pen_thigh > 0.02).mean()), 3),
                     arm_thigh_p95_cm=round(100 * float(np.percentile(pen_thigh, 95)), 1), arm_thigh_max_cm=round(100 * float(pen_thigh.max()), 1),
                     arm_torso_share_over_1cm=round(float((pen_torso > 0.01).mean()), 3), arm_torso_max_cm=round(100 * float(pen_torso.max()), 1))
    print(name, res[name], flush=True)
rep["arms"] = res
json.dump(rep, open(out, "w"), indent=1)
