"""Train the depth -> height-scan student on recorded ACP teacher data (collect.py) on the local GPU.

    python3 models/acp_depth/train.py --data DIR [DIR ...] --out models/acp_depth/runs/student_vN [--epochs 40] [--init best_state.pt]

Supervised (stage 1 of the plan; DAgger with the student in the loop comes next): L1 on ACP's two scans, sequences of 32 camera
frames (1.3 s) so the GRU learns to remember terrain that has left the view. The sim depth is clean; each sample is made to
look like a D435 (axial noise growing with range, holes at depth edges and below the 0.3 m minimum range, the invalid strip at
the left edge, random holes, a repeated frame now and then). Split by episode (10 % validation). Output: <out>/student.pt
(TorchScript, CPU, for ACPD_MODE=student), <out>/log.jsonl, <out>/best.json.
Memory: the dataset is loaded into RAM (refuses more than --max-gb); run under a cgroup cap, e.g.
systemd-run --user --scope -p MemoryMax=24G python3 ...
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from student import Student  # noqa: E402

SEQ = 32


def load(files, max_gb):
    eps, total = [], 0
    for f in files:
        z = np.load(f)
        d = z["depth"]
        total += d.nbytes
        if total > max_gb * 1e9:
            raise SystemExit("dataset over %.0f GB in RAM; raise --max-gb or use fewer files" % max_gb)
        if len(d) < SEQ + 1:
            continue
        eps.append(dict(depth=d, prop=z["prop"].astype(np.float32), plan=z["scan_plan"].astype(np.float32),
                        track=z["scan_track"].astype(np.float32)))
    return eps, total


def d435(depth_mm, rng):
    """Clean sim z-depth (T, H, W) uint16 mm -> D435-like float metres with holes (0)."""
    z = depth_mm.astype(np.float32) / 1000.0
    T, H, W = z.shape
    z = z + rng.normal(size=z.shape).astype(np.float32) * (0.002 + 0.004 * z * z)        # axial noise ~ z^2
    gx = np.abs(np.diff(z, axis=2, prepend=z[:, :, :1]))
    gy = np.abs(np.diff(z, axis=1, prepend=z[:, :1, :]))
    edge = (np.maximum(gx, gy) > 0.06) & (rng.random(z.shape) < 0.6)                        # stereo fails at depth edges
    drop = edge | (rng.random(z.shape) < 0.01) | (z < 0.3)                                   # speckle, below min range
    band = rng.integers(0, 4)
    if band:
        drop[:, :, :band] = True                                                             # left invalid strip
    for _ in range(rng.integers(0, 3)):                                                      # a few holes (shiny / dark patches)
        h, w = rng.integers(3, 12), rng.integers(3, 16)
        y0, x0 = rng.integers(0, H - h), rng.integers(0, W - w)
        drop[:, y0:y0 + h, x0:x0 + w] = True
    z[drop] = 0.0
    rep = rng.random(T) < 0.05                                                               # a late frame: the previous one again
    rep[0] = False
    for t in np.where(rep)[0]:
        z[t] = z[t - 1]
    return z


def batches(eps, bs, rng, noise=True):
    while True:
        idx = rng.integers(0, len(eps), bs)
        D, P, YP, YT = [], [], [], []
        for i in idx:
            e = eps[i]
            t0 = rng.integers(0, len(e["depth"]) - SEQ)
            sl = slice(t0, t0 + SEQ)
            D.append(d435(e["depth"][sl], rng) if noise else e["depth"][sl].astype(np.float32) / 1000.0)
            p = e["prop"][sl]
            P.append(p + rng.normal(size=p.shape).astype(np.float32) * 0.01 if noise else p)
            YP.append(e["plan"][sl]); YT.append(e["track"][sl])
        yield (torch.from_numpy(np.stack(D))[:, :, None], torch.from_numpy(np.stack(P)),
               torch.from_numpy(np.stack(YP)), torch.from_numpy(np.stack(YT)))


def wl1(pred, y, nx, ny, edge_w):
    """Smooth L1, with cells on raised / lowered terrain (> 3 cm from the grid's median) or at a height step (> 3 cm to a
    neighbour) weighted 1 + edge_w."""
    l = F.smooth_l1_loss(pred, y, beta=0.02, reduction="none")
    if edge_w <= 0:
        return l.mean()
    g = y.reshape(*y.shape[:-1], nx, ny)
    terr = (g - g.flatten(-2).median(-1).values[..., None, None]).abs() > 0.03
    dx = (g[..., 1:, :] - g[..., :-1, :]).abs() > 0.03
    dy = (g[..., :, 1:] - g[..., :, :-1]).abs() > 0.03
    edge = torch.zeros_like(terr)
    edge[..., 1:, :] |= dx; edge[..., :-1, :] |= dx; edge[..., :, 1:] |= dy; edge[..., :, :-1] |= dy
    wgt = 1.0 + edge_w * (terr | edge).flatten(-2).float()
    return (l * wgt).sum() / wgt.sum()


def evaluate(model, eps, dev, rng, n=200):
    model.eval()
    ep, et, gen = [], [], batches(eps, 16, rng)
    with torch.no_grad():
        for _ in range(max(1, n // 16)):
            d, p, yp, yt = (x.to(dev) for x in next(gen))
            pp, pt = model.sequence(d, p)
            w = slice(8, None)                         # skip the GRU's first frames (it starts with no memory)
            ep.append((pp[:, w] - yp[:, w]).abs().mean().item())
            et.append((pt[:, w] - yt[:, w]).abs().mean().item())
    model.train()
    return float(np.mean(ep)), float(np.mean(et))


def export(model, out, prop_dim, H, W):
    m = Student(prop_dim, H, W)
    m.load_state_dict(model.state_dict())
    m = m.cpu().eval()
    ts = torch.jit.script(m)
    ts.save(str(out / "student.pt"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, nargs="+", help="one or more folders of episode files (DAgger: aggregate all)")
    ap.add_argument("--init", default=None, help="start from this best_state.pt (DAgger rounds)")
    ap.add_argument("--edge-w", type=float, default=0.0, help="extra loss weight on terrain / edge cells (2026-11-16: the 4x4 came "
                    "out at 4-6 cm of its 8.9 cm: an even L1 lets the net blur steps); 0 = off")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--steps", type=int, default=500, help="optimiser steps per epoch")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-gb", type=float, default=16.0)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(f for d in a.data for f in glob.glob(os.path.join(d, "*.npz")))
    rng = np.random.default_rng(0)
    rng.shuffle(files)
    nv = max(1, len(files) // 10)
    val, tr = files[:nv], files[nv:]
    t0 = time.time()
    tr_eps, gb = load(tr, a.max_gb)
    val_eps, _ = load(val, a.max_gb)
    H, W = tr_eps[0]["depth"].shape[1:]
    P = tr_eps[0]["prop"].shape[1]
    nfr = sum(len(e["depth"]) for e in tr_eps)
    print("train %d episodes (%d frames, %.2f GB), val %d; depth %dx%d, prop %d; loaded in %.0f s"
          % (len(tr_eps), nfr, gb / 1e9, len(val_eps), W, H, P, time.time() - t0), flush=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Student(P, H, W).to(dev)
    allp = np.concatenate([e["prop"] for e in tr_eps])
    model.prop_mean.copy_(torch.from_numpy(allp.mean(0)))
    model.prop_std.copy_(torch.from_numpy(allp.std(0) + 1e-3))
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location=dev))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs * a.steps)
    gen = batches(tr_eps, a.bs, rng)
    best = (1e9, None)
    log = open(out / "log.jsonl", "a")
    for ep in range(1, a.epochs + 1):
        t1, tl = time.time(), []
        for _ in range(a.steps):
            d, p, yp, yt = (x.to(dev, non_blocking=True) for x in next(gen))
            pp, pt = model.sequence(d, p)
            w = slice(4, None)
            loss = wl1(pp[:, w], yp[:, w], 17, 11, a.edge_w) + 2.0 * wl1(pt[:, w], yt[:, w], 9, 7, a.edge_w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tl.append(loss.item())
        vp, vt = evaluate(model, val_eps, dev, np.random.default_rng(1))
        rec = dict(epoch=ep, loss=float(np.mean(tl)), val_mae_plan_cm=100 * vp, val_mae_track_cm=100 * vt, s=round(time.time() - t1, 1))
        log.write(json.dumps(rec) + "\n"); log.flush()
        print("epoch %3d  loss %.4f  val MAE plan %.2f cm  track %.2f cm  (%.0f s)" % (ep, rec["loss"], 100 * vp, 100 * vt, rec["s"]), flush=True)
        score = vp + 2 * vt
        if score < best[0]:
            best = (score, rec)
            torch.save(model.state_dict(), out / "best_state.pt")
            export(model, out, P, H, W)
            json.dump(dict(best=rec, prop_dim=P, res=[W, H], train_files=len(tr), val_files=len(val)), open(out / "best.json", "w"), indent=1)
    print("DONE best:", best[1], flush=True)


if __name__ == "__main__":
    main()
