"""Spikes in the walk raster (the physics floor) of our apparatus, against the CAD boxes (eval monitor, 2026-11-12).

    python eval/tools/raster_spikes.py OUT_DIR [RESULTS_JSON ...]

A spike cell: its height is over 5 cm above the median of its 7 x 7 neighbourhood and its connected spike patch is at most
0.10 m across (a thin post, blade or brace foot, not a step, ramp or 4x4). Draws top-down pictures of the 4x4 and ramp
sections: raster heights, spike cells (red), CAD wall boxes (outlines; braces and boards included), the route and the end
points of failed trials (x fall, o stall). Counts failures that end within 0.20 m of a spike.
"""
import json
import pathlib
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from apparatus import Apparatus, quat_mat  # noqa: E402

out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
app = Apparatus("ours")
g = app.grid.astype(float)
med = ndimage.median_filter(g, size=7)
cand = (g - med) > 0.05
lab, n = ndimage.label(cand)
spike = np.zeros_like(cand)
sizes = []
for i, sl in enumerate(ndimage.find_objects(lab), 1):
    ext = max(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start) * app.res
    m_ = lab[sl] == i
    if ext <= 0.10 and (g - med)[sl][m_].max() > 0.10:     # thin and over 10 cm proud (4x4 ends at 9 cm and the deck rim are not spikes)
        spike[sl] |= m_
        sizes.append(ext)
xs = app.x0 + app.res * np.arange(app.nx)
ys = app.y0 + app.res * np.arange(app.ny)
ii, jj = np.nonzero(spike)
pts = np.c_[xs[ii], ys[jj], g[ii, jj]]
on_deck = (pts[:, 0] > 0) & (pts[:, 0] < 11) & (pts[:, 1] > 0) & (pts[:, 1] < 2.35)
print("spike cells: %d in %d patches (on the deck: %d cells); heights %.2f-%.2f m above their surroundings" % (
    len(pts), len(sizes), on_deck.sum(), float((g - med)[spike].min()), float((g - med)[spike].max())))
fails = []
for f in sys.argv[2:]:
    for r in json.load(open(f)):
        if not r["success"]:
            fails.append((r["final"]["xy"][0], r["final"]["xy"][1], str(r["reason"]).split(":")[0], r.get("stopped_section")))
near = [f for f in fails if len(pts) and np.min(np.hypot(pts[:, 0] - f[0], pts[:, 1] - f[1])) < 0.20]
if fails:
    print("failed trials: %d; ending within 0.20 m of a spike: %d (falls %d, stalls %d)" % (
        len(fails), len(near), sum(f[2] == "fall" for f in near), sum(f[2] == "stalled" for f in near)))
    by = {}
    for f in near:
        by[f[3]] = by.get(f[3], 0) + 1
    print("   by section:", by)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
walls = json.load(open(app.dir / "cad_walls.json"))
walls = walls["boxes"] if isinstance(walls, dict) else walls
P = app.path()
for name, (xa, xb) in {"4x4s": (2.2, 3.9), "ramps": (3.5, 6.4)}.items():
    fig, ax = plt.subplots(1, 2, figsize=(16, 7.5))
    m = (xs >= xa) & (xs <= xb)
    for k, a in enumerate(ax):
        im = a.imshow(g[m].T, origin="lower", extent=(xs[m][0], xs[m][-1], ys[0], ys[-1]), cmap="viridis", vmin=-0.05, vmax=0.4, aspect="equal")
        a.set_xlim(xa, xb); a.set_ylim(-0.1, 2.45)
        if k == 0:
            a.set_title("walk raster (physics floor), 2 cm; red: thin spikes > 10 cm proud", fontsize=10)
            s = (pts[:, 0] >= xa) & (pts[:, 0] <= xb)
            a.scatter(pts[s, 0], pts[s, 1], s=6, c="red")
        else:
            a.set_title("CAD boxes over the raster (white: reach below 0.5 m, orange: above)", fontsize=10)
            for b in walls:
                R = quat_mat(b["quat"]); h = np.array(b["half"]); c = np.array(b["center"])
                corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * h @ R.T + c
                if corners[:, 0].max() < xa or corners[:, 0].min() > xb:
                    continue
                from scipy.spatial import ConvexHull
                try:
                    hull = ConvexHull(corners[:, :2])
                    a.add_patch(Polygon(corners[hull.vertices, :2], closed=True, fill=False, ec="white" if corners[:, 2].min() < 0.5 else "orange", lw=0.8))
                except Exception:
                    pass
        a.plot(P[:, 0], P[:, 1], "w--", lw=1)
        for f in fails:
            if xa <= f[0] <= xb:
                a.plot(f[0], f[1], "rx" if f[2] == "fall" else "co", ms=4, mfc="none")
    fig.colorbar(im, ax=ax, shrink=0.7, label="height (m)")
    fig.suptitle("Our apparatus, %s section: top-down; route dashed; failed trial end points (x fall, o stall) of the runs given" % name)
    fig.savefig(out / ("raster_vs_cad_%s.png" % name), dpi=110, bbox_inches="tight")
    print("wrote", out / ("raster_vs_cad_%s.png" % name))
