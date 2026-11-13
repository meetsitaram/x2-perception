"""Station data for stage 2 (ACP agent, 2026-11-12): copy each clip set with a 1.5 s stand appended to every clip whose
last frame has both feet in labelled contact (double support), so the tracker learns to stop and hold at its goal.
Writes <set>_hold.pkl next to the originals in the track's clip folder and reports counts."""
import json, os, sys, joblib, numpy as np, torch
sys.path.insert(0, "/tmp/c2e")
from acp.data import Dataset
CFG = "cfg/stage2f_sets.json"; HOLD_S = 1.5
cfg = json.load(open(CFG)); d = Dataset(CFG, torch.device("cpu"), cache="cfg/labels_stage2f_sets.json.pt")
m = d.motion; C = d.contact
last_seg = {}
for c, nm in enumerate(m.names):
    si, rest = nm.split("/", 1); key, seg = (rest.split("#") + ["0"])[:2]
    k = (int(si[1:]), key)
    if k not in last_seg or int(seg) > last_seg[k][0]: last_seg[k] = (int(seg), c)
out = {}
for si, s in enumerate(cfg["sets"]):
    src = os.path.join(cfg["dir"], s["pkl"]); data = joblib.load(src); new = {}; n_hold = 0
    for key, clip in data.items():
        e = last_seg.get((si, key))
        ok = e is not None and bool(C[int(m.clip_end[e[1]]), 0]) and bool(C[int(m.clip_end[e[1]]), 1])
        if ok:
            T = len(clip["dof"]); n = int(round(HOLD_S * float(clip["fps"])))
            clip = {k2: (np.concatenate([np.asarray(v), np.repeat(np.asarray(v)[-1:], n, 0)]) if hasattr(v, "__len__") and not isinstance(v, str) and len(v) == T else v) for k2, v in clip.items()}
            n_hold += 1
        new[key] = clip
    dst = src.replace(".pkl", "_hold.pkl"); joblib.dump(new, dst)
    out[s["pkl"]] = (n_hold, len(data)); print(f"{s['pkl']}: stand appended to {n_hold} of {len(data)} clips -> {os.path.basename(dst)}")
