"""Check each eval route against the apparatus it runs through, and draw it.

For every 2 cm of the route (spawn -> waypoints, straight segments):
  * step: along the centre line and FOOT_OFF to either side, the walk raster must not jump more than MAX_STEP between
    samples (a wall or deck edge the robot cannot step);
  * corridor: no wall box that overlaps the body's height band (ground + BODY_LO .. ground + BODY_HI) may be closer to the
    centre line than HALF_W (robot half width). Reported: the tightest clearance of every segment and at each waypoint.
  * arrival: the ground heights inside each waypoint's arrival disc, so a disc that spans an obstacle is seen.

    python eval/check_routes.py [ours nist] [--png DIR]       (CPU, seconds)
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from apparatus import Apparatus

MAX_STEP = 0.25      # m (apparatus riser 0.186)
HALF_W = 0.25        # m: X2 half width, arms at its sides (model: 0.48 m across the elbows at the default pose)
FOOT_OFF = 0.10      # m: foot lanes either side of the centre line
BODY_LO, BODY_HI = 0.30, 1.30   # m above the ground under the robot: the band walls are checked in
DS = 0.02
RADIUS = 0.20        # m: arrival radius of the eval (docs/EVAL.md)
BEAM_HALF = 0.045    # m: half width of a 4x4
END_CLEAR = 0.25     # m: a route leg must cross a beam at least this far from the beam's ends (the robot's half width)
DOOR_MIN = 2 * HALF_W + 2 * 0.05     # m: least clear width of the doorway: the robot's width plus 5 cm either side


def seg_cross(p, q, a, b):
    """Where segment p-q crosses segment a-b: distance along a-b from a (m), or None."""
    p, q, a, b = (np.asarray(v, float) for v in (p, q, a, b))
    r, s = q - p, b - a
    den = r[0] * s[1] - r[1] * s[0]
    if abs(den) < 1e-12:
        return None
    t = ((a[0] - p[0]) * s[1] - (a[1] - p[1]) * s[0]) / den
    u = ((a[0] - p[0]) * r[1] - (a[1] - p[1]) * r[0]) / den
    return float(u * np.linalg.norm(s)) if 0 <= t <= 1 and 0 <= u <= 1 else None


def seg_dist(p, a, b):
    p, a, b = (np.asarray(v, float) for v in (p, a, b))
    t = np.clip((p - a) @ (b - a) / ((b - a) @ (b - a)), 0, 1)
    return float(np.linalg.norm(a + t * (b - a) - p))


def check_beams(app):
    """Per beam: the route legs that cross it (leg index, distance from the nearer end, crossing angle) and the
    waypoint whose arrival disc comes closest to it. Problems as text."""
    P, out, bad = app.path(), [], []
    for bm in app.route.get("beams", []):
        a, b = np.array(bm["a"]), np.array(bm["b"])
        L = float(np.linalg.norm(b - a))
        cross = []
        for s, (p, q) in enumerate(zip(P[:-1], P[1:])):
            u = seg_cross(p, q, a, b)
            if u is not None:
                d = (q - p) / np.linalg.norm(q - p)
                ang = float(np.degrees(np.arcsin(min(1.0, abs(d[0] * (b - a)[1] - d[1] * (b - a)[0]) / L))))
                cross.append(dict(leg=s, end_clear=round(min(u, L - u), 3), angle=round(ang)))
        gaps = [seg_dist([w["x"], w["y"]], a, b) - BEAM_HALF for w in app.waypoints]
        k = int(np.argmin(gaps))
        out.append(dict(name=bm["name"], length=round(L, 2), crossings=cross, nearest_wp=k, nearest_gap=round(gaps[k], 3)))
        if not cross:
            bad.append("%s: no route leg crosses it" % bm["name"])
        for c in cross:
            if c["end_clear"] < END_CLEAR:
                bad.append("%s: leg %d crosses %.2f m from its end" % (bm["name"], c["leg"], c["end_clear"]))
        if gaps[k] < RADIUS:
            bad.append("%s: the arrival disc of waypoint %d reaches it (%.2f m from the beam)" % (bm["name"], k, gaps[k]))
    return out, bad


def check_door(app):
    """Clear width of the doorway: the longest run across the door plane (and 10 cm either side of it) with no wall
    box at body height. The open leaf is one of the boxes. Returns (dict, problems)."""
    dr = app.route.get("door")
    if not dr:
        return None, ["no door described in route.json"]
    ys = np.arange(dr["span_y"][0], dr["span_y"][1] + 1e-9, 0.01)
    free = np.ones(len(ys), bool)
    for x in np.arange(dr["plane_x"] - 0.10, dr["plane_x"] + 0.101, 0.02):
        for i, y in enumerate(ys):
            if free[i]:
                g = app.height_patch(x, y, 0.06, np.max)
                free[i] = app.wall_clearance(x, y, g + BODY_LO, g + BODY_HI) > 0.005
    P = app.path()
    through = [float(p[1] + (dr["plane_x"] - p[0]) * (q[1] - p[1]) / (q[0] - p[0])) for p, q in zip(P[:-1], P[1:])
               if (p[0] - dr["plane_x"]) * (q[0] - dr["plane_x"]) < 0]
    runs, i = [], 0                                         # free runs (from_y, to_y); the doorway is the one the route is in
    while i < len(ys):
        if free[i]:
            j = i
            while j + 1 < len(ys) and free[j + 1]:
                j += 1
            runs.append((float(ys[i]), float(ys[j])))
            i = j + 1
        else:
            i += 1
    yr = through[0] if through else None
    mine = [r for r in runs if yr is not None and r[0] <= yr <= r[1]]
    lo, hi = mine[0] if mine else (max(runs, key=lambda r: r[1] - r[0]) if runs else (float("nan"), float("nan")))
    width = hi - lo
    leaf = [seg_dist(c, dr["leaf"][0], dr["leaf"][1]) for c in np.concatenate([np.linspace(p, q, 50) for p, q in zip(P[:-1], P[1:])])]
    out = dict(clear_width=round(width, 2), from_y=round(lo, 2), to_y=round(hi, 2), open_deg=dr["open_deg"],
               route_crosses_at_y=[round(v, 2) for v in through], leaf_to_route=round(float(min(leaf)), 2))
    bad = []
    if not mine:
        bad.append("the route does not pass through a free part of the door plane")
    if not width >= DOOR_MIN:
        bad.append("doorway clear width %.2f m is under %.2f m" % (width, DOOR_MIN))
    if mine and min(yr - lo, hi - yr) < HALF_W:
        bad.append("the route passes the doorway %.2f m from a jamb" % min(yr - lo, hi - yr))
    if min(leaf) < HALF_W + 0.05:
        bad.append("the open leaf is %.2f m from the route" % min(leaf))
    return out, bad


def check(app: Apparatus, half_w=HALF_W):
    P = app.path()
    rows, blocked = [], []
    for s, (a, b) in enumerate(zip(P[:-1], P[1:])):
        L = float(np.linalg.norm(b - a))
        n = max(2, int(L / DS) + 1)
        c = a + (b - a) * np.linspace(0, 1, n)[:, None]
        u = (b - a) / max(L, 1e-9)
        nrm = np.array([-u[1], u[0]])
        worst_step = 0.0
        for off in (0.0, -FOOT_OFF, FOOT_OFF):
            h = app.height(c[:, 0] + nrm[0] * off, c[:, 1] + nrm[1] * off)
            worst_step = max(worst_step, float(np.abs(np.diff(h)).max()))
        clear, where = np.inf, None
        for k in range(n):
            g = app.height_patch(c[k, 0], c[k, 1], 0.10, np.max)
            d = app.wall_clearance(c[k, 0], c[k, 1], g + BODY_LO, g + BODY_HI)
            if d < clear:
                clear, where = d, c[k]
        w = app.waypoints[s]
        rows.append(dict(seg=s, to=w["name"], length=round(L, 2), max_step=round(worst_step, 3), clearance=round(float(clear), 3),
                         at=[round(float(v), 2) for v in where] if where is not None else None))
        if worst_step > MAX_STEP:
            blocked.append((s, w["name"], "step of %.2f m" % worst_step))
        if clear < half_w:
            blocked.append((s, w["name"], "wall %.2f m from the centre line at (%.2f, %.2f)" % (clear, where[0], where[1])))
    return rows, blocked


def arrival(app: Apparatus, radius=RADIUS):
    out = []
    k = int(round(radius / app.res))
    ii, jj = np.mgrid[-k:k + 1, -k:k + 1]
    m = (ii ** 2 + jj ** 2) <= k * k
    for i, w in enumerate(app.waypoints):
        ci, cj = app.cell(w["x"], w["y"])
        z = app.grid[np.clip(ci + ii[m], 0, app.nx - 1), np.clip(cj + jj[m], 0, app.ny - 1)]
        g = app.height_patch(w["x"], w["y"], 0.10, np.max)
        out.append(dict(i=i, name=w["name"], ground=round(app.height_patch(w["x"], w["y"]), 3), disc_min=round(float(z.min()), 3),
                        disc_max=round(float(z.max()), 3), wall=round(app.wall_clearance(w["x"], w["y"], g + BODY_LO, g + BODY_HI), 3)))
    return out


def draw(app: Apparatus, path, rows, radius=RADIUS, door=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mp
    import matplotlib.pyplot as plt
    full = np.load(app.dir / app.route["raster_full"])["grid"]
    ext = [app.x0, app.x0 + app.nx * app.res, app.y0, app.y0 + app.ny * app.res]
    fig, ax = plt.subplots(2, 1, figsize=(24, 19))
    for a, g, t in ((ax[0], app.grid, "walk raster (physics floor) + wall boxes (magenta)"), (ax[1], full, "full raster (tall cells kept)")):
        im = a.imshow(np.clip(g, app.floor - 0.2, app.floor + 1.3).T, origin="lower", extent=ext, cmap="terrain")
        plt.colorbar(im, ax=a, shrink=0.6, label="height (m)")
        P = app.path()
        a.plot(P[:, 0], P[:, 1], "r-", lw=1.4)
        a.plot(P[0, 0], P[0, 1], "r*", ms=14)
        for i, w in enumerate(app.waypoints):
            a.add_patch(mp.Circle((w["x"], w["y"]), radius, fill=False, ec="r", lw=0.8))
            a.text(w["x"], w["y"], str(i), color="k", fontsize=9, bbox=dict(fc="w", pad=0.4, alpha=0.75))
        a.set_title("%s: %s  (orange: 4x4 centre lines; green: open door leaf and hinge; cyan: clear doorway)" % (app.name, t))
        a.set_aspect("equal")
    for a in ax:
        for bm in app.route.get("beams", []):
            a.plot([bm["a"][0], bm["b"][0]], [bm["a"][1], bm["b"][1]], color="orange", lw=2.5, solid_capstyle="butt")
        dr = app.route.get("door")
        if dr:
            a.plot([dr["leaf"][0][0], dr["leaf"][1][0]], [dr["leaf"][0][1], dr["leaf"][1][1]], color="lime", lw=3)
            a.plot(dr["hinge"][0], dr["hinge"][1], "o", color="lime", ms=6)
            if door:
                a.plot([dr["plane_x"]] * 2, [door["from_y"], door["to_y"]], color="cyan", lw=3)
                a.text(dr["plane_x"], app.y0 + app.ny * app.res - 0.9, "doorway %.2f m clear, leaf open %.0f deg" % (door["clear_width"], dr["open_deg"]),
                       color="w", fontsize=10, ha="center")
    for b in app.boxes:
        ax[0].add_patch(mp.Rectangle((b["lo"][0], b["lo"][1]), b["hi"][0] - b["lo"][0], b["hi"][1] - b["lo"][1], fill=False, ec="m", lw=0.4))
    for r in rows:
        if r["at"] and r["clearance"] < 0.35:
            ax[0].plot(r["at"][0], r["at"][1], "kx", ms=9)
    plt.tight_layout()
    plt.savefig(path, dpi=70)
    plt.close()


if __name__ == "__main__":
    names = [a for a in sys.argv[1:] if not a.startswith("--") and a in ("ours", "nist")] or ["ours", "nist"]
    png = sys.argv[sys.argv.index("--png") + 1] if "--png" in sys.argv else None
    for name in names:
        app = Apparatus(name)
        rows, blocked = check(app)
        print("== %s: %d waypoints, %.1f m, %d wall boxes" % (name, len(app.waypoints), sum(r["length"] for r in rows), len(app.boxes)))
        for r in rows:
            print("  seg %2d -> %-26s %5.2f m  max step %.3f  wall clearance %.3f at %s" % (r["seg"], r["to"][:26], r["length"], r["max_step"], r["clearance"], r["at"]))
        print("  arrival discs (r = %.2f m):" % RADIUS)
        for a in arrival(app):
            print("   %2d %-26s ground %.3f  disc %.3f..%.3f  wall %.3f" % (a["i"], a["name"][:26], a["ground"], a["disc_min"], a["disc_max"], a["wall"]))
        beams, bad_b = check_beams(app)
        for b in beams:
            print("  beam %-10s %.2f m long; crossed by legs %s; nearest arrival disc: waypoint %d, %.2f m from the beam" % (
                b["name"], b["length"], [(c["leg"], "%.2f m from its end" % c["end_clear"], "%d deg" % c["angle"]) for c in b["crossings"]],
                b["nearest_wp"], b["nearest_gap"]))
        door, bad_d = check_door(app)
        if door:
            print("  door: leaf open %.1f deg; doorway clear %.2f m (y %.2f..%.2f, needs %.2f); route passes at y %s; leaf %.2f m from the route" % (
                door["open_deg"], door["clear_width"], door["from_y"], door["to_y"], DOOR_MIN, door["route_crosses_at_y"], door["leaf_to_route"]))
        problems = ["seg %d -> %s: %s" % b for b in blocked] + bad_b + bad_d
        if problems:
            print("  FAILED:")
            for b in problems:
                print("   " + b)
        else:
            print("  PASS: no segment blocked (half width %.2f m, max step %.2f m), both beams crossed clear of their ends, doorway open" % (HALF_W, MAX_STEP))
        if png:
            draw(app, str(pathlib.Path(png) / ("route_%s.png" % name)), rows, door=door)
