"""Where do the weak sets' evaluation episodes end? From a paper-eval rollout dump (CPU).
    CUDA_VISIBLE_DEVICES= python tools/weak_sets.py ROLLOUTS.pt cfg/stage1_sets.json"""
import sys

import torch

sys.path.insert(0, "/mnt/filesystem-g4/tracks/acp/code")
from acp.data import Dataset  # noqa: E402

d = torch.load(sys.argv[1], map_location="cpu")
data = Dataset(sys.argv[2], torch.device("cpu"), cache=sys.argv[2].replace("stage1_sets.json", "labels_stage1_sets.json.pt"))
m = data.motion
alive, root, clip, succ, set_id = d["alive"], d["root"], d["clip"], d["success"], d["set_id"]
T, N = alive.shape
last = alive.long().sum(0).clamp(max=T - 1)                     # first step at which the episode is over
clen = m.clip_len[clip]
phase = last.float() / (clen - 1).float()
ended_early = phase < 0.97
idx = torch.arange(N)
rob = root[(last - 1).clamp(min=0), idx]                        # last recorded state before the end
f_end = (m.clip_start[clip] + (last - 1).clamp(min=0)).clamp(max=m.num_frames - 1)
ref = m.root_p[f_end]
exy = (rob[:, :2] - ref[:, :2]).norm(dim=-1)
ez = rob[:, 2] - ref[:, 2]
ryaw = torch.atan2(m.root_R[f_end][:, 1, 0], m.root_R[f_end][:, 0, 0])
eyaw = (rob[:, 3] - ryaw + torch.pi) % (2 * torch.pi) - torch.pi
# along-track error: positive = robot ahead of the reference in the reference's direction of travel
v = m.root_v[f_end][:, :2]
along = ((rob[:, :2] - ref[:, :2]) * v).sum(-1) / v.norm(dim=-1).clamp(min=0.05)
for s, name in enumerate(d["set_names"]):
    k = set_id == s
    e = k & ended_early
    n, ne = int(k.sum()), int(e.sum())
    if ne == 0:
        continue
    ph = phase[e]
    hist = torch.histc(ph, bins=5, min=0, max=1) / ne
    clips_e = clip[e]
    uniq, cnt = clips_e.unique(return_counts=True)
    tot = torch.stack([(clip[k] == u).sum() for u in uniq])
    rate = cnt.float() / tot.float()
    worst = rate.argsort(descending=True)[:4]
    nclip = len(clip[k].unique())
    print(f"{name}: {n} episodes, {ne} ended early ({ne / n:.1%}); end phase quintiles share {[round(float(x), 2) for x in hist]}; "
          f"at the end: xy error median {float(exy[e].median()):.2f} m, along-track median {float(along[e].median()):+.2f} m "
          f"(share behind the reference {float((along[e] < -0.2).float().mean()):.0%}, ahead {float((along[e] > 0.2).float().mean()):.0%}), "
          f"root height error median {float(ez[e].median()):+.2f} m (share more than 0.2 m low {float((ez[e] < -0.2).float().mean()):.0%}), "
          f"|yaw error| median {float(eyaw[e].abs().median()):.2f} rad; clips evaluated {nclip}, clips with a failure {len(uniq)}, "
          f"clips failing more than half the time {int((rate > 0.5).sum())}")
    for w in worst:
        c = int(uniq[w])
        boxes = data.terrain.boxes[c]
        b = boxes[boxes[:, 4] > 0]
        halves = data.terrain.half[boxes[:, 4] > 0]
        tops = sorted(set(round(float(z + h[2]), 3) for z, h in zip(b[:, 2], halves)))
        rises = [round(tops[i + 1] - tops[i], 3) for i in range(len(tops) - 1)]
        depth = sorted(set(round(float(2 * h[0]), 3) for h in halves))
        dur = float(clen[clip == c][0]) / 50
        spd = float(m.root_v[m.clip_start[c]:m.clip_end[c]][:, :2].norm(dim=-1).mean())
        print(f"    clip #{c}: fails {int(cnt[w])}/{int(tot[w])}, {dur:.1f} s, mean reference speed {spd:.2f} m/s, box tops {tops[:9]}, "
              f"rises {rises[:8]}, box lengths (2 x half-x) {depth[:6]}")
