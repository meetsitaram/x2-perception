"""Where does the 'sole inside the terrain' of clip_quality come from? For every frame and foot: deepest corner
(h - z, > 0 inside), lowest-corner gap, whether the foot is labelled in stance, and the terrain height spread under the
foot's four corners (> 3 cm = the foot straddles an edge: riser, box or raster step). Per set: share of clips flagged by
clip_quality's rule, and of the flagged frames how many are (a) straddling an edge, (b) flat-under-foot in stance,
(c) flat-under-foot in swing.
    CUDA_VISIBLE_DEVICES= python pen_cause.py cfg/stage2_sets.json"""
import os, sys, json
import torch
sys.path.insert(0, "/mnt/filesystem-g4/tracks/acp/code_next")
from acp.data import Dataset
from gear_sonic.supermimic.kinematics import mv3
cfg = sys.argv[1]
data = Dataset(cfg, torch.device("cpu"), cache=os.path.join(os.path.dirname(cfg), "labels_" + os.path.basename(cfg) + ".pt"))
m, T = data.motion, data.terrain
link = data.kin.body_index(["left_ankle_roll_link", "right_ankle_roll_link"])
F = m.num_frames
deep = torch.zeros(F, 2); low = torch.zeros(F, 2); spread = torch.zeros(F, 2)
for a in range(0, F, 20000):
    f = torch.arange(a, min(a + 20000, F))
    P, Rb, _, _ = data.kin.forward(*m.ref_state(f))
    c = P[:, link, None, :] + mv3(Rb[:, link, None, :, :], data.corners)
    h = T.height(c[..., 0].reshape(len(f), 8), c[..., 1].reshape(len(f), 8), m.clip_id[f]).reshape(len(f), 2, 4)
    d = h - c[..., 2]
    deep[f] = d.amax(-1); low[f] = (-d).amin(-1); spread[f] = h.amax(-1) - h.amin(-1)
stance = data.contact[:, :2]
res = {}
print(f"{'set':26s} clips flag | worst cm | flagged frames (deepest corner > 3 cm): edge-straddle / flat stance / flat swing | worst flat-stance cm | worst lowest-corner-in-stance cm")
for s, name in enumerate(data.set_names):
    cl = (data.clip_set == s).nonzero().squeeze(-1).tolist()
    nflag = 0; worst = 0.0; ce = cs = cw = 0; wfs = 0.0; wlow = 0.0
    for c in cl:
        a, b = int(m.clip_start[c]), int(m.clip_end[c]) + 1
        dd, sp, st, lo = deep[a:b], spread[a:b], stance[a:b], low[a:b]
        fr = dd.amax(-1) > 0.03
        if float(fr.float().mean()) > 0.05: nflag += 1
        worst = max(worst, float(dd.max()))
        pen = dd > 0.03
        edge = pen & (sp > 0.03); flat = pen & ~(sp > 0.03)
        ce += int(edge.sum()); cs += int((flat & st).sum()); cw += int((flat & ~st).sum())
        if (flat & st).any(): wfs = max(wfs, float(dd[flat & st].max()))
        if st.any(): wlow = min(wlow, float(lo[st].min()))
    tot = max(ce + cs + cw, 1)
    res[name] = dict(clips=len(cl), flagged=nflag, worst_cm=round(100 * worst, 1), edge=ce, flat_stance=cs, flat_swing=cw, worst_flat_stance_cm=round(100 * wfs, 1), worst_low_stance_cm=round(100 * wlow, 1))
    print(f"{name:26s} {len(cl):5d} {nflag:4d} | {100*worst:7.1f} | {ce/tot:5.2f} / {cs/tot:5.2f} / {cw/tot:5.2f} ({tot} foot-frames) | {100*wfs:6.1f} | {100*wlow:6.1f}")
json.dump(res, open(sys.argv[2], "w"), indent=1)
