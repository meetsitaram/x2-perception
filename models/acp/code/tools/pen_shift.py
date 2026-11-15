"""Prop offset or retarget? For the sets with sole penetration (terrain-park stairs, ramps, chains, 4x4): shift the
terrain under each clip horizontally by (dx, dy) in a grid of +-8 cm and count the foot-frames with a sole corner more than
3 cm inside plus the stance foot-frames floating more than 3 cm (so moving the terrain away is not rewarded). A prop placed off its capture shows up as a clear best shift away from (0, 0) that removes most of it; a
foot that does not fit (retarget: foot length, ankle range) stays penetrating at every shift.
    CUDA_VISIBLE_DEVICES= python pen_shift.py cfg/stage2_sets.json OUT.json"""
import os, sys, json, collections
import torch
sys.path.insert(0, "/mnt/filesystem-g4/tracks/acp/code")
from acp.data import Dataset
from gear_sonic.supermimic.kinematics import mv3
cfg = sys.argv[1]
data = Dataset(cfg, torch.device("cpu"), cache=os.path.join(os.path.dirname(cfg), "labels_" + os.path.basename(cfg) + ".pt"))
m, T = data.motion, data.terrain
link = data.kin.body_index(["left_ankle_roll_link", "right_ankle_roll_link"])
G = [x / 100 for x in (-16, -8, -4, -2, 0, 2, 4, 8, 16)]
import re
st_all = data.contact[:, :2]
res = {}
for s, name in enumerate(data.set_names):
    if not any(k in name for k in ("tp_stairs", "tp_ramp", "chains", "4x4")):
        continue
    best_hist, base_tot, best_tot, rows = collections.Counter(), 0, 0, []
    for c in (data.clip_set == s).nonzero().squeeze(-1).tolist():
        a, b = int(m.clip_start[c]), int(m.clip_end[c]) + 1
        f = torch.arange(a, b)
        P, Rb, _, _ = data.kin.forward(*m.ref_state(f))
        cp = (P[:, link, None, :] + mv3(Rb[:, link, None, :, :], data.corners)).reshape(len(f), 8, 3)
        cnt = {}
        for dx in G:
            for dy in G:
                h = T.height(cp[..., 0] - dx, cp[..., 1] - dy, m.clip_id[f])
                d = (h - cp[..., 2]).reshape(len(f), 2, 4)
                deep = d.amax(-1) > 0.03                                   # a corner more than 3 cm inside
                floating = st_all[a:b] & ((-d).amin(-1) > 0.03)            # labelled stance, every corner more than 3 cm above
                cnt[(dx, dy)] = int(deep.sum()) + int(floating.sum())
        base = cnt[(0.0, 0.0)]
        if base == 0:
            continue
        bk = min(cnt, key=lambda k: (cnt[k], abs(k[0]) + abs(k[1])))
        best_hist[bk] += 1; base_tot += base; best_tot += cnt[bk]
        var = 'shifted' if re.search(r'_sh[+-]\d+', m.names[c]) else 'base'
        rows.append(dict(var=var, clip=m.names[c], base=base, best=list(bk), best_count=cnt[bk]))
    res[name] = dict(clips_with_pen=len(rows), foot_frames_at_0=base_tot, foot_frames_at_best=best_tot,
                     by_variant={v: (sum(r['var'] == v for r in rows), sum(r['base'] for r in rows if r['var'] == v), sum(r['best_count'] for r in rows if r['var'] == v)) for v in ('base', 'shifted')},
                     best_shift_hist={f"{100*k[0]:+.0f},{100*k[1]:+.0f}": v for k, v in best_hist.most_common(8)}, clips=rows)
    print(f"{name:24s} clips with penetration {len(rows)}: foot-frames at (0,0) {base_tot}, at each clip's best shift {best_tot} "
          f"({100*(1-best_tot/max(base_tot,1)):.0f}% less); most common best shifts (cm): {res[name]['best_shift_hist']}; by variant (clips, at 0, at best): {res[name]['by_variant']}", flush=True)
json.dump(res, open(sys.argv[2], "w"), indent=1)
