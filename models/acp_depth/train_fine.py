"""Train the fine-map depth student (fine_student.py) on the local GPU.

    systemd-run --user --scope -p MemoryMax=28G python3 models/acp_depth/train_fine.py --data DIR [DIR ...] --out RUN [--epochs 30]

Each sample: the camera frame t and the frames t-4, t-8, t-12 (clean sim depth, each given D435-like noise with train.d435,
then projected with its own camera pose), ACP's proprioception at t; labels: the true fine map at t (2.5 cm) and ACP's two
scans. Loss: L1 on the fine map, cells on raised / lowered terrain or at a height step (> 2 cm) weighted 1 + --edge-w, plus L1
on ACP's points read off the map. Validation (by episode) reports the fine map's error, the 4x4 / step cells' height error
and ACP's points' error. Output: RUN/student.pt (TorchScript, CPU, ACPD_STUDENT_KIND=fine), RUN/best.json, RUN/log.jsonl.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fine_student import FineStudent, LAGS, NX, NY  # noqa: E402
from train import d435  # noqa: E402


def load(files, max_gb):
    eps, total = [], 0
    for f in files:
        z = np.load(f)
        if "fine" not in z.files or len(z["depth"]) < 20:
            continue
        total += z["depth"].nbytes + z["fine"].nbytes
        if total > max_gb * 1e9:
            raise SystemExit("dataset over %.0f GB" % max_gb)
        eps.append(dict(depth=z["depth"], prop=z["prop"].astype(np.float32), cam=z["cam_T"].astype(np.float32),
                        fine=z["fine"], plan=z["scan_plan"].astype(np.float32), track=z["scan_track"].astype(np.float32)))
    return eps, total


def batch(eps, bs, rng, noise=True):
    lags = (0,) + tuple(LAGS)
    D, C, P, Y, YP, YT = [], [], [], [], [], []
    for i in rng.integers(0, len(eps), bs):
        e = eps[i]
        t = int(rng.integers(0, len(e["depth"])))
        ts = [max(t - l, 0) for l in lags]
        frames = e["depth"][ts]
        D.append(d435(frames, rng) if noise else frames.astype(np.float32) / 1000.0)
        C.append(e["cam"][ts])
        p = e["prop"][t]
        P.append(p + rng.normal(size=p.shape).astype(np.float32) * 0.01 if noise else p)
        Y.append(e["fine"][t].astype(np.float32)); YP.append(e["plan"][t]); YT.append(e["track"][t])
    f = lambda a: torch.from_numpy(np.stack(a))  # noqa: E731
    return f(D), f(C), f(P), f(Y), f(YP), f(YT)


def run(model, d, c, p):
    """d (B, 4, H, W), c (B, 4, 12), p (B, P) -> fine, plan, track (the lagged grids projected independently)."""
    B, L, H, W = d.shape
    g = model.project(d.reshape(B * L, 1, H, W), c.reshape(B * L, 12)).reshape(B, L * 4, NX, NY)
    fine = model.decode(g, p)
    plan, track = model.sample(fine)
    return fine, plan, track


def masks(y):
    floor = y.flatten(1).median(1).values[:, None, None]
    terr = (y - floor).abs() > 0.02
    dx = (y[:, 1:] - y[:, :-1]).abs() > 0.02
    dy = (y[:, :, 1:] - y[:, :, :-1]).abs() > 0.02
    edge = torch.zeros_like(terr)
    edge[:, 1:] |= dx; edge[:, :-1] |= dx; edge[:, :, 1:] |= dy; edge[:, :, :-1] |= dy
    return terr, edge, floor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--edge-w", type=float, default=4.0)
    ap.add_argument("--max-gb", type=float, default=20.0)
    ap.add_argument("--init", default=None)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(f for d in a.data for f in glob.glob(os.path.join(d, "*.npz")))
    rng = np.random.default_rng(0)
    rng.shuffle(files)
    nv = max(1, len(files) // 10)
    t0 = time.time()
    tr, gb = load(files[nv:], a.max_gb)
    va, _ = load(files[:nv], a.max_gb)
    H, W = tr[0]["depth"].shape[1:]
    P = tr[0]["prop"].shape[1]
    print("train %d episodes (%d frames, %.1f GB), val %d; loaded in %.0f s" % (len(tr), sum(len(e["depth"]) for e in tr), gb / 1e9,
                                                                             len(va), time.time() - t0), flush=True)
    dev = torch.device("cuda")
    model = FineStudent(P, H, W).to(dev)
    allp = np.concatenate([e["prop"] for e in tr])
    model.prop_mean.copy_(torch.from_numpy(allp.mean(0)))
    model.prop_std.copy_(torch.from_numpy(allp.std(0) + 1e-3))
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location=dev))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * a.steps, pct_start=max(0.05, 4.0 / (a.epochs * a.steps)))
    log = open(out / "log.jsonl", "a")
    best = 1e9
    for ep in range(1, a.epochs + 1):
        t1, tl = time.time(), []
        for _ in range(a.steps):
            d, c, p, y, yp, yt = (x.to(dev, non_blocking=True) for x in batch(tr, a.bs, rng))
            fine, plan, track = run(model, d, c, p)
            terr, edge, _ = masks(y)
            wgt = 1.0 + a.edge_w * (terr | edge).float()
            l_fine = (F.smooth_l1_loss(fine, y, beta=0.01, reduction="none") * wgt).sum() / wgt.sum()
            loss = l_fine + 0.5 * F.smooth_l1_loss(plan, yp, beta=0.01) + 1.0 * F.smooth_l1_loss(track, yt, beta=0.01)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tl.append(loss.item())
        # validation
        model.eval()
        vr = np.random.default_rng(1)
        s = dict(fine=[], terr=[], bump=[], plan=[], track=[])
        with torch.no_grad():
            for _ in range(20):
                d, c, p, y, yp, yt = (x.to(dev) for x in batch(va, 32, vr))
                fine, plan, track = run(model, d, c, p)
                terr, edge, floor = masks(y)
                err = (fine - y).abs()
                s["fine"].append(err.mean().item())
                if terr.any():
                    s["terr"].append(err[terr].mean().item())
                bumps = (y - floor) > 0.05                     # raised obstacles (4x4s, blocks, steps up): signed height error
                if bumps.any():
                    s["bump"].append((fine - y)[bumps].mean().item())
                s["plan"].append((plan - yp).abs().mean().item())
                s["track"].append((track - yt).abs().mean().item())
        model.train()
        rec = {k: round(100 * float(np.mean(v)), 2) for k, v in s.items()}
        rec.update(epoch=ep, loss=round(float(np.mean(tl)), 4), s=round(time.time() - t1))
        log.write(json.dumps(rec) + "\n"); log.flush()
        print("epoch %2d loss %.4f | val cm: fine %.2f terrain %.2f raised-obstacle bias %+.2f | ACP plan %.2f track %.2f (%.0f s)"
              % (ep, rec["loss"], rec["fine"], rec["terr"], rec["bump"], rec["plan"], rec["track"], rec["s"]), flush=True)
        score = rec["terr"] + rec["track"] + abs(rec["bump"])
        if score < best:
            best = score
            torch.save(model.state_dict(), out / "best_state.pt")
            m = FineStudent(P, H, W)
            m.load_state_dict(model.state_dict())
            torch.jit.script(m.cpu().eval()).save(str(out / "student.pt"))
            json.dump(dict(best=rec, prop_dim=P, res=[W, H], kind="fine"), open(out / "best.json", "w"), indent=1)
    print("DONE", json.load(open(out / "best.json"))["best"], flush=True)


if __name__ == "__main__":
    main()
