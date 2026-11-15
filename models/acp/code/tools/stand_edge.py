"""Stand-at-the-edge set (ACP agent, 2026-11-12, after the eval's button-5 finding): the policy must hold a stand 0.15-0.45 m
from a ~1.1 m drop that is ahead (and for two variants also to one side), facing it, with the drop in its scan. Source: the
plain idle (standing) clips of the flat set that drift under 0.10 m; each is lifted onto a 1.5 x 1.5 x 0.3 m landing slab
(a box size the gs2 sets already use, so no new prop slot) whose top is at TOP m, placed relative to the clip's start
pose: front edge EDGE ahead of the pelvis (0.25-0.45 m, i.e. about 0.13-0.33 m ahead of the toes), and in the corner
variants the left or right edge 0.30-0.40 m to the side. Writes stand_edge_v1.pkl and stand_edge_v1_props.json."""
import json, math, os, sys, joblib, numpy as np, torch
sys.path.insert(0, "/tmp/c2e")
from acp.data import Dataset
D = "/mnt/filesystem-g4/tracks/acp/data/clips"; TOP = 1.1; HALF = [0.75, 0.75, 0.15]
rng = np.random.default_rng(7)
d = Dataset("cfg/stage2e_sets.json", torch.device("cpu"), cache="cfg/labels_stage2e_sets.json.pt")
m = d.motion
src = joblib.load(os.path.join(D, "x2_loco_flat_train.pkl"))
first = {}
for c, nm in enumerate(m.names):
    si, rest = nm.split("/", 1)
    if si != "s0": continue
    key, seg = (rest.split("#") + ["0"])[:2]
    if "idle" in key.lower() and int(seg) == 0: first[key] = c
out, props, skipped = {}, {}, 0
for key, c in sorted(first.items()):
    clip = src[key]; rt = np.asarray(clip["root_trans_offset"])
    if np.linalg.norm(rt[:, :2] - rt[:1, :2], axis=1).max() > 0.10: skipped += 1; continue
    a = int(m.clip_start[c]); p0 = m.root_p[a].numpy(); R0 = m.root_R[a].numpy(); yaw = math.atan2(R0[1, 0], R0[0, 0])
    fwd, left = np.array([math.cos(yaw), math.sin(yaw)]), np.array([-math.sin(yaw), math.cos(yaw)])
    for var in ("front", "corner_left", "corner_right"):
        edge = rng.uniform(0.25, 0.45); side = rng.uniform(0.30, 0.40)
        lat = 0.0 if var == "front" else (HALF[1] - side) * (-1 if var == "corner_left" else 1)
        cxy = p0[:2] + fwd * (edge - HALF[0]) + left * lat
        name = f"{key}_edge_{var}"
        new = dict(clip); new["root_trans_offset"] = rt + np.array([0.0, 0.0, TOP])
        out[name] = new
        props[name] = [dict(kind="step", center=[float(cxy[0]), float(cxy[1]), TOP - HALF[2]], half=HALF, yaw=float(yaw))]
joblib.dump(out, os.path.join(D, "stand_edge_v1.pkl")); json.dump(props, open(os.path.join(D, "stand_edge_v1_props.json"), "w"), indent=1)
print(f"idle clips {len(first)}, drifting > 0.10 m skipped {skipped}, stand-edge clips written {len(out)}")
