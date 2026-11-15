"""Per-clip failure analysis of a paper-eval rollout dump (CPU): what kind of episodes end early, and how much of the gap
to 98.5 % is held by a few clips.  CUDA_VISIBLE_DEVICES= python tools/fail_clips.py ROLLOUTS.pt cfg/stage1_sets.json"""
import re, sys, collections
import torch
sys.path.insert(0, "/mnt/filesystem-g4/tracks/acp/code")
from acp.data import Dataset
d = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
data = Dataset(sys.argv[2], torch.device("cpu"), cache=sys.argv[2].replace("stage1_sets.json", "labels_stage1_sets.json.pt"))
m = data.motion
alive, root, clip, succ, set_id = d["alive"], d["root"], d["clip"], d["success"].bool(), d["set_id"]
T, N = alive.shape
last = alive.long().sum(0).clamp(max=T - 1); clen = m.clip_len[clip]
phase = last.float() / (clen - 1).float(); early = phase < 0.97; idx = torch.arange(N)
rob = root[(last - 1).clamp(min=0), idx]
f_end = (m.clip_start[clip] + (last - 1).clamp(min=0)).clamp(max=m.num_frames - 1); ref = m.root_p[f_end]
v = m.root_v[f_end][:, :2]; along = ((rob[:, :2] - ref[:, :2]) * v).sum(-1) / v.norm(dim=-1).clamp(min=0.05)
ez = rob[:, 2] - ref[:, 2]
# per-clip features: mean speed, peak yaw rate, style words of the clip name (tallied, names not printed)
C = len(m.names); spd = torch.zeros(C); yawr = torch.zeros(C)
for c in clip.unique().tolist():
    a, b = int(m.clip_start[c]), int(m.clip_end[c])
    spd[c] = m.root_v[a:b][:, :2].norm(dim=-1).mean()
    R = m.root_R[a:b]; yw = torch.atan2(R[:, 1, 0], R[:, 0, 0]); dy = (yw[1:] - yw[:-1] + torch.pi) % (2 * torch.pi) - torch.pi
    yawr[c] = (dy.abs() * 50).quantile(0.95) if len(dy) > 1 else 0
STYLE = ["injur", "limp", "bump", "drunk", "zombie", "old", "tired", "sneak", "jog", "run", "turn", "back", "side", "stairs", "jump",
         "start", "stop", "fast", "slow", "dance", "crouch", "hurry", "trip", "stumbl", "push"]
def words(c): return [w for w in STYLE if w in m.names[c].lower()]
for s, name in enumerate(d["set_names"]):
    k = set_id == s; n = int(k.sum()); e = k & early; f = k & ~succ; nf = int(f.sum())
    if nf == 0: continue
    cl = clip[k]; uc, tot = cl.unique(return_counts=True)
    fc, fcnt = clip[f].unique(return_counts=True)
    rate = {int(u): int(c) / int(tot[(uc == u).nonzero()[0, 0]]) for u, c in zip(fc, fcnt)}
    rep = [u for u in rate if int(tot[(uc == u).nonzero()[0, 0]]) >= 2]
    always = [u for u in rep if rate[u] == 1.0]
    order = sorted(fc.tolist(), key=lambda u: -int(fcnt[fc == u]))
    cum, top = 0, []
    for u in order:
        cum += int(fcnt[fc == u]); top.append(cum)
    gap_ep = nf - int(round(0.015 * n))   # failures above the 98.5 % mark
    n_for_gap = next((i + 1 for i, c in enumerate(top) if c >= gap_ep), None) if gap_ep > 0 else 0
    lag = e & (along < -0.3); low = e & (ez < -0.2)
    print(f"\n{name}: {n} ep, fail {nf} ({nf / n:.1%}), early {int(e.sum())}, not-early-but-off-goal {nf - int((f & e).sum())}; "
          f"early ends: behind ref >0.3 m {int(lag.sum())}, root >0.2 m low {int(low.sum())}, end phase median {float(phase[e].median()) if e.any() else 0:.2f}")
    print(f"  clips evaluated {len(uc)} (seen >=2x: {int((tot >= 2).sum())}); failing clips {len(fc)}; of those seen >=2x: {len(rep)}, failing every time {len(always)}; "
          f"clips needed to hold the failures above 98.5 % ({gap_ep} ep): {n_for_gap} of {len(fc)}")
    for lab, msk in (("success", k & succ), ("failure", f)):
        cs = clip[msk]
        print(f"  {lab}: ref speed median {float(spd[cs].median()):.2f} m/s, p90 {float(spd[cs].quantile(0.9)):.2f}; peak yaw rate median "
              f"{float(yawr[cs].median()):.2f} rad/s, p90 {float(yawr[cs].quantile(0.9)):.2f}")
    for lo, hi in ((0, .3), (.3, .6), (.6, .9), (.9, 9)):
        b = k & (spd[clip] >= lo) & (spd[clip] < hi)
        if b.any(): print(f"    speed {lo}-{hi}: {int((b & ~succ).sum())}/{int(b.sum())} fail")
    for lo, hi in ((0, 1), (1, 2), (2, 9)):
        b = k & (yawr[clip] >= lo) & (yawr[clip] < hi)
        if b.any(): print(f"    yaw rate {lo}-{hi}: {int((b & ~succ).sum())}/{int(b.sum())} fail")
    wf = collections.Counter(w for c in clip[f].tolist() for w in words(c)); wa = collections.Counter(w for c in cl.tolist() for w in words(c))
    print("  style words, failures/episodes:", ", ".join(f"{w} {wf[w]}/{wa[w]}" for w in sorted(wa, key=lambda w: -wf[w]) if wa[w] >= 3))
