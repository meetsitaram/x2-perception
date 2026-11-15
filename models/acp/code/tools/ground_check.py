"""Where do soles sit relative to their terrain in the reference clips? Per clip: the gap of the lowest sole corner of a
foot in labelled stance (0 = on the surface, negative = inside), its median and spread over the clip; per set, how much is a
constant offset per clip and whether it goes with the shifted capture variants.
    CUDA_VISIBLE_DEVICES= python tools/ground_check.py cfg/stage2_sets.json"""
import os
import re
import sys

import torch

sys.path.insert(0, "/mnt/filesystem-g4/tracks/acp/code")
from acp.data import Dataset  # noqa: E402
from gear_sonic.supermimic.kinematics import mv3  # noqa: E402

cfg = sys.argv[1]
data = Dataset(cfg, torch.device("cpu"), cache=os.path.join(os.path.dirname(cfg), "labels_" + os.path.basename(cfg) + ".pt"))
m, T = data.motion, data.terrain
link = data.kin.body_index(["left_ankle_roll_link", "right_ankle_roll_link"])
F = m.num_frames
gap = torch.zeros(F, 2)
for a in range(0, F, 20000):
    f = torch.arange(a, min(a + 20000, F))
    P, Rb, _, _ = data.kin.forward(*m.ref_state(f))
    c = P[:, link, None, :] + mv3(Rb[:, link, None, :, :], data.corners)
    h = T.height(c[..., 0].reshape(len(f), 8), c[..., 1].reshape(len(f), 8), m.clip_id[f]).reshape(len(f), 2, 4)
    gap[f] = (c[..., 2] - h).amin(-1)
stance = data.contact[:, :2]
print(f"{'set':28s} clips | stance gap of the lowest sole corner: median over clips of the per-clip median (cm), 10th pct, share of clips whose "
      f"median is under -3 cm, median within-clip spread (cm) | by variant")
for s, name in enumerate(data.set_names):
    med, spr, var = [], [], {}
    for c in (data.clip_set == s).nonzero().squeeze(-1).tolist():
        a, b = int(m.clip_start[c]), int(m.clip_end[c]) + 1
        g = gap[a:b][stance[a:b]]
        # frames where the label says no stance but a sole is deep inside also count as penetration
        deep = gap[a:b].amin(-1)
        if len(g) < 5:
            continue
        md = float(g.median())
        med.append(md); spr.append(float(g.quantile(0.9) - g.quantile(0.1)))
        mt = re.search(r"_sh([+-]\d+)", m.names[c])
        var.setdefault("shifted" if mt else "base", []).append(md)
    if not med:
        continue
    t = torch.tensor(med)
    byv = {k: f"{len(v)} clips, median {100 * float(torch.tensor(v).median()):+.1f} cm, worst {100 * float(torch.tensor(v).min()):+.1f} cm" for k, v in var.items()}
    print(f"{name:28s} {len(med):5d} | {100 * float(t.median()):+6.1f} {100 * float(t.quantile(0.1)):+6.1f} {float((t < -0.03).float().mean()):5.2f} "
          f"{100 * float(torch.tensor(spr).median()):5.1f} | {byv}")
