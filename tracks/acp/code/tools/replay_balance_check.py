"""CPU check of the planner replay's balanced sampling (ACP agent, 2026-11-13, state/acp.md 17j).

    python tools/replay_balance_check.py cfg/stage2h_sets.json OUT.json

Fills a Replay the way the recorder does in training: episodes are drawn with the training clip weights (clip_w), each
successful episode adds about min(clip length, 500) / 2 samples of its clip, ring capacity 1,500,000 (Table S5 buffer of
the run). Then it compares, per data set, the share of the buffer content, of 10 x 4,096 uniform draws (old rule) and
of 10 x 4,096 balanced draws (data sets -> clips -> frames). A second phase floods the buffer with the most common set
only (the planner share rising on common frames, 16q) and repeats the comparison.
"""
import json
import sys

import torch

sys.path.insert(0, ".")
from acp.data import Dataset          # noqa: E402
from acp.planner import Replay        # noqa: E402

torch.manual_seed(0)
cfg, out = sys.argv[1], sys.argv[2]
data = Dataset(cfg, "cpu", cache="cfg/labels_" + cfg.split("/")[-1] + ".pt")
names, cs, w = data.set_names, data.clip_set.long(), data.clip_w.float()
m = getattr(data, "motion", data)
ln = None
for a in ("clip_len", "clip_lens"):
    if hasattr(m, a):
        ln = getattr(m, a).long().cpu(); break
if ln is None:
    ln = torch.full((len(cs),), 500, dtype=torch.long)
per_ep = (ln.clamp(max=500) // 2).clamp(min=1)


class Tiny(Replay):                   # same sampling code, without the 1.5 M x 495 history tensors
    def __init__(self, cap, clip_set, balance):
        self.cap, self.size, self.ptr, self.balance = cap, 0, 0, balance
        self.clip = torch.full((cap,), -1, dtype=torch.long)
        self.clip_set = clip_set; self.n_sets = int(clip_set.max()) + 1

    def push_clips(self, clips):
        n = clips.numel(); idx = (self.ptr + torch.arange(n)) % self.cap
        self.clip[idx] = clips; self.ptr = int((self.ptr + n) % self.cap); self.size = min(self.cap, self.size + n)


def fill(rb, n_samples, wv):
    while n_samples > 0:
        ep = torch.multinomial(wv, 100, replacement=True)
        c = torch.repeat_interleave(ep, per_ep[ep]); rb.push_clips(c); n_samples -= c.numel()


def table(rb):
    rows = {}
    content = rb.set_share()
    rb.balance = "none"; uni = rb.set_share(rb.indices(40960))
    rb.balance = "set_clip"; bal = rb.set_share(rb.indices(40960))
    for i, nm in enumerate(names):
        if i < len(content) and float(content[i]) > 0:
            rows[nm] = dict(content=round(float(content[i]), 4), uniform=round(float(uni[i]), 4), balanced=round(float(bal[i]), 4))
    return rows


rb = Tiny(1500000, cs, "set_clip")
fill(rb, 1500000, w)
res = {"phase1_training_weights": table(rb)}
big = int(torch.bincount(cs, weights=w).argmax())
wf = torch.where(cs == big, w, torch.zeros_like(w)); fill(rb, 1200000, wf)                    # 80 % of the ring overwritten
res["phase2_flooded_with_" + names[big]] = table(rb)
json.dump(res, open(out, "w"), indent=1)
for ph, rows in res.items():
    print(ph)
    print(f"  {'set':44s} content uniform balanced")
    for nm, r in sorted(rows.items(), key=lambda kv: kv[1]["content"]):
        print(f"  {nm:44s} {r['content']:.4f}  {r['uniform']:.4f}  {r['balanced']:.4f}")
