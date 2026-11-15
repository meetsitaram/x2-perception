"""Data set 2g (ACP agent, 2026-11-12, after the eval's three traced gaps):
(1) turn_place_v1: clips of the kept flat styles whose heading turns > 2.0 rad while the pelvis travels < 0.6 m (turn in
    place / turn around), as their own set, so goals behind the robot with |dpsi| near pi are trained often;
(2) start-from-a-stand: 1.0 s of the first frame prepended to every clip of the descending / ramp / chains / stairs sets
    whose first frame is in double support, so a step (down) is started from a standstill at an edge (the ramps' trough).
Writes <set>_hs.pkl (prepend on top of the _hold end stands) and turn_place_v1.pkl, and cfg/stage2g_sets.json."""
import json, math, os, sys, joblib, numpy as np, torch
sys.path.insert(0, "/tmp/c2e")
from acp.data import Dataset
D = "/mnt/filesystem-g4/tracks/acp/data/clips"; C = "/mnt/filesystem-g4/tracks/acp/code/cfg/"
cfg = json.load(open(C + "stage2f_sets.json"))
d = Dataset("cfg/stage2f_sets.json", torch.device("cpu"), cache="cfg/labels_stage2f_sets.json.pt")
m = d.motion; R = m.root_R[torch.arange(m.num_frames)]; yaw = torch.atan2(R[:, 1, 0], R[:, 0, 0])
seg0 = {}
for c, nm in enumerate(m.names):
    si, rest = nm.split("/", 1); key, seg = (rest.split("#") + ["0"])[:2]
    k = (int(si[1:]), key)
    if k not in seg0 or int(seg) < seg0[k][0]: seg0[k] = (int(seg), c)
# (1) turn in place, from the flat set (index 0) after its filters
fs = cfg["sets"][0]; flat = joblib.load(os.path.join(D, fs["pkl"])); turn = {}
for c, nm in enumerate(m.names):
    if not nm.startswith("s0/"): continue
    a, e = int(m.clip_start[c]), int(m.clip_end[c])
    dy = torch.remainder(yaw[a + 1:e + 1] - yaw[a:e] + math.pi, 2 * math.pi) - math.pi
    tot = float(dy.sum().abs()); travel = float((m.root_p[a:e + 1, :2] - m.root_p[a, :2]).norm(dim=-1).max())
    if tot > 2.0 and travel < 0.6:
        key = nm.split("/", 1)[1].split("#")[0]; turn[key] = flat[key]
joblib.dump(turn, os.path.join(D, "turn_place_v1.pkl"))
print("turn_place_v1:", len(turn), "clips")
# (2) prepend 1.0 s stands
PRE = {"x2_chains_far.pkl", "tp_stairs_x10_sh2_hold.pkl", "gs2dn_v2_hold.pkl", "gs2lo_v1_hold.pkl", "tp_ramp_sh2_hold.pkl", "tp_t4x4u_y20_hold.pkl", "gst_v1.pkl"}
for si, s in enumerate(cfg["sets"]):
    if s["pkl"] not in PRE: continue
    src = joblib.load(os.path.join(D, s["pkl"])); new = {}; n = 0
    for key, clip in src.items():
        e = seg0.get((si, key))
        if e is not None and bool(d.contact[int(m.clip_start[e[1]]), 0]) and bool(d.contact[int(m.clip_start[e[1]]), 1]):
            T = len(clip["dof"]); k = int(round(1.0 * float(clip["fps"])))
            clip = {k2: (np.concatenate([np.repeat(np.asarray(v)[:1], k, 0), np.asarray(v)]) if hasattr(v, "__len__") and not isinstance(v, str) and len(v) == T else v) for k2, v in clip.items()}
            n += 1
        new[key] = clip
    out = s["pkl"].replace("_hold.pkl", ".pkl").replace(".pkl", "_hs.pkl")
    joblib.dump(new, os.path.join(D, out)); print(f"{s['pkl']}: stand prepended to {n} of {len(src)} -> {out}"); s["pkl"] = out
for s in cfg["sets"]:
    if s["pkl"].startswith("x2_loco_flat_train"): s["weight"] = 0.12
cfg["sets"].append(dict(pkl="turn_place_v1.pkl", props="", weight=0.03))
cfg["armfix"] = "armfix_stage2g_sets.pt"
cfg["note"] = ("stage 2g (2026-11-12 14:50, ACP agent): as 2f plus turn_place_v1 (weight 0.03; flat 0.15 -> 0.12) and 1.0 s stands prepended "
               "(*_hs.pkl) to the stairs, gs2, ramp, 4x4, chains and gst clips that start in double support. Reason: the eval's traces at "
               "6,000 (no turn-around step for a target behind; no step down from a stand at the ramps' trough). " + cfg["note"])
json.dump(cfg, open(C + "stage2g_sets.json", "w"), indent=1)
