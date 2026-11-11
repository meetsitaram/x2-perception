"""Top-down pictures of ladder courses, for review before a list is run: python eval/ladder_maps.py STEP OUT_DIR
Each picture: the tile's top-surface raster, the boxes that are not in the floor raster (magenta), the 4x4 centre lines
(orange), the route with its waypoints and arrival discs, the start (star) and the stations (filled). Also prints the route
check of each course (corridor 0.25 m, step 0.25 m)."""
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import check_routes as C  # noqa: E402
import ladder as L  # noqa: E402

step, out = int(sys.argv[1]), pathlib.Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.patches as mp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

rows = []
for name, make in L.STEPS[step].items():
    c = make()
    P = c.path()
    length = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
    segs, blocked = C.check(c)
    clear = min(r["clearance"] for r in segs)
    top = c.grid
    if c.route.get("tile"):
        t, z = L._tile(name_type := [k for k in ("single_beams", "beam_fields", "blocks", "wedges", "tight_spots", "log_corridors", "homes", "street_front", "stairs")
                                     if c.route["tile"].startswith(k)][0], c.route["tile"])
        top = z["height_top"][t["index"]].astype(float)
    fig, ax = plt.subplots(figsize=(8, 8))
    ext = [c.x0 - c.res / 2, c.x0 + c.nx * c.res, c.y0 - c.res / 2, c.y0 + c.ny * c.res]
    im = ax.imshow(np.clip(top, -0.2, 1.5).T, origin="lower", extent=ext, cmap="terrain")
    plt.colorbar(im, ax=ax, shrink=0.7, label="top surface height (m), clipped at 1.5")
    for b in c.boxes:
        ax.add_patch(mp.Rectangle((b["lo"][0], b["lo"][1]), b["hi"][0] - b["lo"][0], b["hi"][1] - b["lo"][1], fill=False, ec="m", lw=0.6))
    for bm in c.route.get("beam_lines", []):
        ax.plot([bm["a"][0], bm["b"][0]], [bm["a"][1], bm["b"][1]], color="orange", lw=2)
    ax.plot(P[:, 0], P[:, 1], "r-", lw=1.6)
    ax.plot(P[0, 0], P[0, 1], "r*", ms=16)
    for i, w in enumerate(c.waypoints):
        ax.add_patch(mp.Circle((w["x"], w["y"]), 0.2, fill=w["hold_s"] > 0, fc="red", alpha=0.35 if w["hold_s"] > 0 else 1, ec="r", lw=0.8))
        ax.text(w["x"], w["y"], str(i), fontsize=8, color="k", bbox=dict(fc="w", pad=0.3, alpha=0.7))
    stations = sum(w["hold_s"] > 0 for w in c.waypoints)
    ax.set_title("step %d PROPOSAL: %s\n%s route %s, level %s | %.1f m, %d waypoints, %d stations | tightest wall %.2f m | %s" % (
        step, name, c.route.get("tile"), c.route.get("tile_route"), c.route.get("level"), length, len(c.waypoints), stations, clear,
        "route check PASS" if not blocked else "BLOCKED: " + "; ".join(b[2] for b in blocked[:2])), fontsize=9)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out / ("%s.png" % name), dpi=90)
    plt.close(fig)
    rows.append(dict(name=name, tile=c.route.get("tile"), route=c.route.get("tile_route"), level=c.route.get("level"), length_m=round(length, 1), waypoints=len(c.waypoints),
                     stations=stations, beams_to_cross=len(c.route.get("beams", [])), tightest_wall_m=round(clear, 2), blocked=[b[2] for b in blocked],
                     floor_range_m=[round(float(c.grid.min()), 2), round(float(c.grid.max()), 2)]))
    print(rows[-1])
json.dump(rows, open(out / "list.json", "w"), indent=1)
