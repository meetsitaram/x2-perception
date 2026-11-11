"""The eval ladder below the apparatuses (docs/EVAL.md, "Eval ladder"): skill tests on single tiles of the tracks' training
map and on flat ground (step 1), and a fixed list of reviewed routes on that map (step 2). Everything here is "on the
training map": never an apparatus result.

    python eval/ladder.py --adapter acp_baseline:ACPBaseline --step 1 --trials 50 --out eval/out/<run> [--tests a b] [--label "..."]

Courses come from the simulator-neutral export `data/terrains/extended_v1` (README there): floor raster + the boxes and
meshes that are not in the raster. A course looks like an apparatus to the harness (same scoring, same logs).
Per test: summary.json, results.json, trials.csv, events.csv, traj/ in <out>/step<k>/<test>/; the table in <out>/step<k>.json / .txt.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from apparatus import Apparatus, data_dir, quat_mat  # noqa: E402

TIME_LIMIT = 60.0        # s per ladder test (the routes are 2 to 9 m)
T0, WALL = time.time(), float(os.environ.get("LADDER_WALL", "520"))   # s: a run stops itself after this (a test not finished is redone)


def terrain_dir():
    p = os.environ.get("PA_TERRAIN_DIR")
    return pathlib.Path(p) if p else data_dir().parent / "terrains" / "extended_v1"


class Course(Apparatus):
    """A test course with the attributes the harness, the model builder and the renderer use from an Apparatus."""

    def __init__(self, name, grid, x0, y0, res, boxes, spawn, waypoints, section, extra=None, floor=0.0):
        self.name, self.dir = name, None
        self.grid = np.asarray(grid, np.float64)
        self.x0, self.y0, self.res = x0, y0, res
        self.nx, self.ny = self.grid.shape
        self.floor, self.boxes, self.spawn = floor, boxes, spawn
        self.waypoints = [dict(w, section=section) for w in waypoints]
        self.sections = [section]
        self.route = dict(name=name, sections=self.sections, **(extra or {}))
        for b in self.boxes:
            e = np.abs(quat_mat(b["quat"])) @ np.array(b["half"])
            b["lo"], b["hi"] = np.array(b["center"]) - e, np.array(b["center"]) + e
        self.box_lo = np.array([b["lo"] for b in self.boxes]).reshape(-1, 3)
        self.box_hi = np.array([b["hi"] for b in self.boxes]).reshape(-1, 3)


def wp(x, y, hold=0.0, heading=None, face=None, name=None, tol=None):
    """face (deg): the heading to keep while moving to this waypoint (walking backward or sideways)."""
    d = dict(name=name or "(%.2f, %.2f)" % (x, y), x=float(x), y=float(y), heading_deg=heading, hold_s=hold)
    if face is not None:
        d["face_deg"] = face
    if tol is not None:
        d["heading_tol_deg"] = tol                          # the station counts only when the robot faces `heading` within this
    return d


def flat(name):
    g = np.zeros((400, 400))
    X0 = -2.0                                              # the tests start 2 m behind the tile centre, facing +x
    routes = {
        "flat_walk": [wp(X0 + 2.0, 0, name="2 m ahead"), wp(X0 + 4.0, 0, name="4 m ahead")],
        "flat_stop_stand": [wp(X0 + 2.0, 0, hold=1.0, heading=0, name="stop and stand 1 s"), wp(X0 + 4.0, 0, hold=1.0, heading=0, name="stop and stand 1 s again")],
        "flat_turn_in_place": [wp(X0, 0.0, hold=1.0, heading=180, tol=20, name="turn to face back"), wp(X0, 0.0, hold=1.0, heading=0, tol=20, name="turn to face ahead")],
        "flat_turn_and_return": [wp(X0 + 1.5, 0, name="1.5 m ahead"), wp(X0, 0, name="back to the start (a 180 degree turn)")],
        "flat_backward": [wp(X0 - 1.5, 0, face=0, name="1.5 m backward, facing ahead")],
        "flat_sideways": [wp(X0, 1.0, face=0, name="1 m to the left, facing ahead"), wp(X0, -1.0, face=0, name="2 m to the right, facing ahead")],
    }
    return Course("ladder:" + name, g, -3.99, -3.99, 0.02, [], [X0, 0.0, 0.0], routes[name], name)


def _tile(ttype, tile_id):
    D = terrain_dir()
    J = json.load(open(D / (ttype + ".json")))
    meta = [t for t in (J if isinstance(J, list) else J["tiles"]) if t["tile_id"] == tile_id]
    assert meta, "no tile %s in %s" % (tile_id, ttype)
    return meta[0], np.load(D / (ttype + ".npz"), allow_pickle=False)


def tile(name, ttype, tile_id, route=0, upto=None, reverse=False):
    """A course on one tile of the export: its route `route`, optionally cut `upto` metres after the start."""
    t, z = _tile(ttype, tile_id)
    i = t["index"]
    labels = [str(s) for s in z["labels"]]
    boxes, beam_lines, stair_pts = [], [], []
    for k in range(int(z["box_start"][i]), int(z["box_start"][i + 1])):
        lab = labels[int(z["box_label"][k])]
        c, h, q = z["box_center"][k].astype(float), z["box_half"][k].astype(float), z["box_quat"][k].astype(float)
        R = quat_mat(q)
        if lab == "beam":
            ax = int(np.argmax(h))
            a, b = c + R[:, ax] * h[ax], c - R[:, ax] * h[ax]
            beam_lines.append(dict(name="beam %d" % len(beam_lines), a=[float(a[0]), float(a[1])], b=[float(b[0]), float(b[1])],
                                   slanted=not bool(z["box_in_raster"][k]), top=float(c[2] + abs(R[2]) @ h)))
        if lab == "stair_step":
            e = np.abs(R) @ h
            stair_pts += [c - e, c + e]
        if not bool(z["box_in_raster"][k]) and int(z["box_role"][k]) > 0:
            boxes.append(dict(center=c.tolist(), half=h.tolist(), quat=q.tolist(), geom=k, label=lab, cls="4x4" if lab == "beam" else None))
    wedge_pts = []
    mlab = [labels[int(v)] for v in z["mesh_label"]]
    for mth in range(int(z["mesh_start"][i]), int(z["mesh_start"][i + 1])):
        c, h = z["mesh_aabb_center"][mth].astype(float), z["mesh_aabb_half"][mth].astype(float)
        if mlab[mth].startswith("wedge"):
            wedge_pts += [c - h, c + h]
        if not bool(z["mesh_in_raster"][mth]) and int(z["mesh_role"][mth]) > 0:     # a cylinder, stone or bush: its box
            boxes.append(dict(center=c.tolist(), half=np.maximum(h, 1e-3).tolist(), quat=[1.0, 0, 0, 0], geom=-1, label=mlab[mth] + " (mesh as its box)", cls=None))
    zones = []
    for cls, pts in (("stair", stair_pts), ("wedge", wedge_pts)):
        if pts:
            P = np.array(pts)
            zones.append(dict(cls=cls, rect=[float(P[:, 0].min() - 0.05), float(P[:, 1].min() - 0.05), float(P[:, 0].max() + 0.05), float(P[:, 1].max() + 0.05)]))
    r = t["routes"][route]
    pts = [list(r["start"][:2])] + [list(w[:2]) if isinstance(w, (list, tuple)) else [w["x"], w["y"]] for w in r["waypoints"]]
    yaw = r["start"][2]
    if reverse:
        pts = pts[::-1]
        yaw = math.atan2(pts[1][1] - pts[0][1], pts[1][0] - pts[0][0])
    P = np.array(pts, float)
    if upto is not None:                                    # cut the polyline `upto` metres from the start
        seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
        cum = np.r_[0, np.cumsum(seg)]
        k = int(np.searchsorted(cum, upto))
        if k < len(P):
            u = (upto - cum[k - 1]) / max(seg[k - 1], 1e-9)
            P = np.vstack((P[:k], P[k - 1] + u * (P[k] - P[k - 1])))
    # waypoints at most 1.2 m apart, so progress is seen and a stall is caught on long legs
    way = []
    for a, b in zip(P[:-1], P[1:]):
        n = max(1, int(math.ceil(np.linalg.norm(b - a) / 1.2)))
        way += [a + (b - a) * (j + 1) / n for j in range(n)]
    from harness import _seg_cross
    need = [dict(name=b["name"], a=b["a"], b=b["b"]) for b in beam_lines
            if any(_seg_cross(p, q, b["a"], b["b"]) is not None for p, q in zip(P[:-1], P[1:]))]
    extra = dict(beams=need, beam_lines=beam_lines, zones=zones, tile=tile_id, tile_route=route, level=t["level"], realised=t["realised"])
    g = z["height"][i].astype(float)
    c0 = z["cell0_center"] if "cell0_center" in z.files else np.array([-3.99, -3.99])
    return Course("ladder:" + name, g, float(c0[0]), float(c0[1]), float(z["resolution"]) if "resolution" in z.files else 0.02, boxes,
                  [float(P[0, 0]), float(P[0, 1]), math.degrees(yaw)], [wp(float(p[0]), float(p[1])) for p in way], name, extra)


def beam_course(name, ttype, tile_id, slanted, before=0.9, after=0.8):
    """One log, crossed square at its middle: start `before` m short of it, end `after` m past it. The first beam of the tile
    (slanted or not, as asked) whose approach and far side are level floor clear of other obstacles."""
    c = tile("probe", ttype, tile_id, 0)
    for bm in c.route["beam_lines"]:
        if bm["slanted"] != slanted:
            continue
        a, b = np.array(bm["a"]), np.array(bm["b"])
        mid, u = (a + b) / 2, (b - a) / np.linalg.norm(b - a)
        n = np.array([-u[1], u[0]])
        for sgn in (1.0, -1.0):
            p0, p1 = mid - sgn * n * before, mid + sgn * n * after
            ok = all(abs(v) < 3.6 for v in (*p0, *p1))
            for q in (p0, p1, mid - sgn * n * 0.45, mid + sgn * n * 0.45):
                i, j = c.cell(q[0], q[1])
                g = c.grid[max(i - 12, 0):i + 13, max(j - 12, 0):j + 13]
                ok = ok and float(g.max() - g.min()) < 0.02 and c.wall_clearance(q[0], q[1], 0.05, 1.3) > 0.30
            others = [o for o in c.route["beam_lines"] if o is not bm]
            from harness import _seg_cross
            ok = ok and not any(_seg_cross(p0, p1, o["a"], o["b"]) is not None for o in others)
            if ok:
                yaw = math.degrees(math.atan2(sgn * n[1], sgn * n[0]))
                extra = dict(c.route, beams=[dict(name=bm["name"], a=bm["a"], b=bm["b"])])
                extra.pop("name"); extra.pop("sections")
                return Course("ladder:" + name, c.grid, c.x0, c.y0, c.res, c.boxes, [float(p0[0]), float(p0[1]), yaw],
                              [wp(float(p1[0]), float(p1[1]), name="%.1f m past the %s log (%.0f cm tall)" % (after, "slanted" if slanted else "level", 100 * bm["top"]))], name, extra)
    raise RuntimeError("no %s beam with a clear square crossing on %s" % ("slanted" if slanted else "level", tile_id))


# ---- step 1: one skill per test. Tiles are fixed by name (level 3 to 4 of 9: mid difficulty).
STEP1 = {
    "flat_walk": lambda: flat("flat_walk"),
    "flat_stop_stand": lambda: flat("flat_stop_stand"),
    "flat_turn_in_place": lambda: flat("flat_turn_in_place"),
    "flat_turn_and_return": lambda: flat("flat_turn_and_return"),
    "flat_backward": lambda: flat("flat_backward"),
    "flat_sideways": lambda: flat("flat_sideways"),
    "log_single": lambda: beam_course("log_single", "single_beams", "single_beams_L4_v1", False),
    "log_slanted": lambda: beam_course("log_slanted", "single_beams", "single_beams_L3_v0", True),
    "wedges": lambda: tile("wedges", "wedges", "wedges_L3_v1", 0),
    "stairs_up": lambda: tile("stairs_up", "stairs", "stairs_L3_v0", 0),
    "stairs_down": lambda: tile("stairs_down", "stairs", "stairs_L3_v0", 1),
    "log_corridor_walls": lambda: tile("log_corridor_walls", "log_corridors", "log_corridors_L3_v0", 0),
    "log_corridor_rails": lambda: tile("log_corridor_rails", "log_corridors", "log_corridors_L3_v1", 0),
}
STEPS = {1: STEP1}


def load_course(name):
    """'ours' / 'nist' -> the apparatus; 'ladder:<test>' -> that test's course."""
    if not name.startswith("ladder:"):
        return Apparatus(name)
    t = name.split(":", 1)[1]
    for tests in STEPS.values():
        if t in tests:
            return tests[t]()
    raise KeyError(name)


def run(adapter_name, checkpoint, step, tests, trials, out, protocol_name, label, seed0=1000):
    import mujoco
    import harness as Hn
    import x2 as X
    adapter = Hn.load_adapter(adapter_name, checkpoint)
    protocol = Hn.RANDOM if protocol_name == "randomised" else Hn.NOMINAL
    names = X.joint_names_of(X.robot_dir() / "mjcf" / "x2_ultra.xml")
    table = []
    for test in tests:
        course = STEPS[step][test]()
        x = X.build(course, adapter.plant(names))
        tr = Hn.Trial(x, course, adapter, x.model.body_mass.copy(), x.model.body_inertia.copy())
        d = out / ("step%d" % step) / test
        (d / "traj").mkdir(parents=True, exist_ok=True)
        results, t0 = [], time.time()
        if (d / "summary.json").exists() and not os.environ.get("LADDER_REDO"):
            print("%-20s already done, kept" % test, flush=True)        # a group cut by its wall limit continues here
            continue
        for i in range(trials):
            if time.time() - T0 > WALL:                 # never unbounded: stop cleanly, the same command continues later
                print("wall limit reached in %s after %d trials; run the same command again" % (test, i), flush=True)
                sys.exit(3)
            r = tr.run(seed0 + i, protocol, TIME_LIMIT, record=True)
            tr.save_traj(d / "traj" / ("%d.npz" % (seed0 + i)))
            results.append(r)
        meta = dict(label=label, ladder_step=step, test=test, on="the training map (not an apparatus result)", apparatus=course.name, route=course.name,
                    tile=course.route.get("tile"), simulator="MuJoCo %s (CPU), dt %.4f s" % (mujoco.__version__, X.SIM_DT), adapter=adapter.name,
                    checkpoint=str(adapter.checkpoint), observes=adapter.observes, control_dt=adapter.control_dt, protocol=protocol_name,
                    seeds=[seed0, seed0 + trials - 1], time_limit=TIME_LIMIT, sections=course.sections,
                    definition=dict(radius=Hn.RADIUS, end_stand=Hn.END_STAND, stall=Hn.STALL, hit_f=Hn.HIT_F, hit_j=Hn.HIT_J))
        s = Hn.summarise(results, meta)
        s["renders"] = Hn.pick_renders(results)
        keep = set(s["renders"].values())
        for r in results:
            if r["seed"] not in keep:
                os.remove(d / "traj" / ("%d.npz" % r["seed"]))
        json.dump(results, open(d / "results.json", "w"), indent=1)
        json.dump(s, open(d / "summary.json", "w"), indent=1)
        Hn.write_logs(d, results, course.waypoints)
        row = dict(test=test, trials=s["trials"], success=s["successes"], ci95=s["ci95"], clean=s["clean_successes"], mean_penalty=s["mean_stumble_penalty"],
                   side_hits_per_trial=s["side_hits"]["per_trial"], mean_waypoints=s["mean_waypoints"], of=len(course.waypoints), outcomes=s["outcomes"],
                   time_median=(s["time_success"] or {}).get("median"), length_m=round(float(np.linalg.norm(np.diff(course.path(), axis=0), axis=1).sum()), 1),
                   tile=course.route.get("tile"), wall_s=round(time.time() - t0))
        table.append(row)
        print("%-20s %3d / %d success (%.0f-%.0f %%), clean %3d, penalty %.2f, waypoints %.1f / %d, %s, %d s" % (
            test, row["success"], row["trials"], 100 * row["ci95"][0], 100 * row["ci95"][1], row["clean"], row["mean_penalty"], row["mean_waypoints"], row["of"],
            row["outcomes"], row["wall_s"]), flush=True)
        json.dump(dict(label=label, step=step, adapter=adapter.name, checkpoint=str(adapter.checkpoint), protocol=protocol_name, rows=table),
                  open(out / ("step%d.json" % step), "w"), indent=1)
    return table


def collect(out, step):
    """The step's table from the per-test summaries on disk (tests run in several groups), as json and text."""
    rows, head = [], None
    for test in STEPS[step]:
        f = out / ("step%d" % step) / test / "summary.json"
        if not f.exists():
            continue
        s = json.load(open(f))
        head = head or s
        course = STEPS[step][test]()
        rows.append(dict(test=test, trials=s["trials"], success=s["successes"], ci95=s["ci95"], clean=s["clean_successes"], mean_penalty=s["mean_stumble_penalty"],
                         side_hits_per_trial=s["side_hits"]["per_trial"], mean_waypoints=s["mean_waypoints"], of=len(course.waypoints), outcomes=s["outcomes"],
                         time_median=(s["time_success"] or {}).get("median"), length_m=round(float(np.linalg.norm(np.diff(course.path(), axis=0), axis=1).sum()), 1),
                         tile=s.get("tile"), renders=s.get("renders")))
    if not rows:
        return rows
    json.dump(dict(label=head.get("label"), step=step, on="the training map (not an apparatus result)", adapter=head["adapter"], checkpoint=head["checkpoint"],
                   protocol=head["protocol"], rows=rows), open(out / ("step%d.json" % step), "w"), indent=1)
    lines = ["%s | step %d, ON THE TRAINING MAP (not an apparatus result) | %s | protocol %s" % (head.get("label") or head["adapter"], step, head["checkpoint"], head["protocol"]),
             "%-22s %-10s %-9s %-12s %-6s %-8s %-10s %-7s %s" % ("test", "success", "rate", "95 % interval", "clean", "penalty", "waypoints", "time", "outcomes / tile")]
    for r in rows:
        lines.append("%-22s %3d / %-4d %5.0f %%   %3.0f - %3.0f %%   %3d    %6.2f   %4.1f / %-3d %5s   %s%s" % (
            r["test"], r["success"], r["trials"], 100.0 * r["success"] / r["trials"], 100 * r["ci95"][0], 100 * r["ci95"][1], r["clean"], r["mean_penalty"],
            r["mean_waypoints"], r["of"], ("%.0f s" % r["time_median"]) if r["time_median"] else "-",
            ", ".join("%s %d" % kv for kv in sorted(r["outcomes"].items(), key=lambda kv: -kv[1])), (" | " + r["tile"]) if r["tile"] else ""))
    open(out / ("step%d.txt" % step), "w").write(chr(10).join(lines) + chr(10))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true", help="only rebuild the step's table from the summaries on disk")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--tests", nargs="*", default=None)
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--protocol", choices=["nominal", "randomised"], default="nominal")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default=None)
    a = ap.parse_args()
    import memcap
    memcap.guard(cap_gb=6, need_free_gb=12, low_priority=True)
    if not a.collect:
        run(a.adapter, a.checkpoint, a.step, a.tests or list(STEPS[a.step]), a.trials, pathlib.Path(a.out), a.protocol, a.label)
    collect(pathlib.Path(a.out), a.step)
    print(open(pathlib.Path(a.out) / ("step%d.txt" % a.step)).read())
