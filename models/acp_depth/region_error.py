"""Per-region error of a trained student on the validation episodes (same split as train.py): by distance band along the
heading (behind / under the feet / ahead) and on terrain cells (true height > 3 cm off the local floor) vs flat cells.
    python3 models/acp_depth/region_error.py --data .../data --run .../student_vN
"""
import argparse, glob, os, sys, pathlib, numpy as np, torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from student import Student
from train import load, d435
ap = argparse.ArgumentParser(); ap.add_argument("--data", nargs="+"); ap.add_argument("--run"); a = ap.parse_args()
files = sorted(f for d in a.data for f in glob.glob(os.path.join(d, "*.npz"))); rng = np.random.default_rng(0); rng.shuffle(files)
val, _ = load(files[:max(1, len(files) // 10)], 8)
m = Student(val[0]["prop"].shape[1]); m.load_state_dict(torch.load(os.path.join(a.run, "best_state.pt"))); m = m.cuda().eval()
X = {"plan": -0.8 + 0.1 * np.arange(17), "track": -0.2 + 0.1 * np.arange(9)}     # x of each grid row (heading frame), x-major
acc = {}
nrng = np.random.default_rng(1)
with torch.no_grad():
    for e in val:
        d = torch.from_numpy(d435(e["depth"], nrng))[None, :, None].cuda()
        p = torch.from_numpy(e["prop"])[None].cuda()
        pp, pt = m.sequence(d, p)
        for name, pred, true, ny in (("plan", pp[0].cpu().numpy(), e["plan"], 11), ("track", pt[0].cpu().numpy(), e["track"], 7)):
            pred, true = pred[10:], true[10:]
            err = np.abs(pred - true).reshape(len(true), -1, ny)
            t = true.reshape(len(true), -1, ny)
            floor = np.median(t.reshape(len(t), -1), 1)[:, None, None]
            terr = np.abs(t - floor) > 0.03
            x = X[name][None, :, None] * np.ones_like(t)
            for band, sel in (("behind (x < -0.2)", x < -0.2), ("under feet (-0.2..0.2)", (x >= -0.2) & (x <= 0.2)), ("ahead (> 0.2)", x > 0.2)):
                for kind, s2 in (("flat", ~terr), ("terrain", terr)):
                    k = (name, band, kind); v = err[sel & s2]
                    acc.setdefault(k, []).append(v)
print("validation episodes: %d (D435-style noise on)" % len(val))
print("%-6s %-24s %-8s %8s %8s %8s %8s" % ("grid", "region", "cells", "MAE cm", "p90 cm", "p99 cm", "share"))
for k in sorted(acc):
    v = np.concatenate(acc[k])
    if len(v):
        print("%-6s %-24s %-8s %8.2f %8.2f %8.2f %7.1f%%" % (k[0], k[1], k[2], 100 * v.mean(), 100 * np.percentile(v, 90), 100 * np.percentile(v, 99),
              100 * len(v) / sum(len(np.concatenate(acc[j])) for j in acc if j[0] == k[0])))
