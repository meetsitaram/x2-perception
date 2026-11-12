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
TIME_LIMIT_STEP2 = 180.0  # s per step-2 route (7 to 15.5 m with stations and turns on the spot; the apparatus limit), eval monitor 2026-11-12
EDGE_START = 3.7         # m: the same for the start pose (stair tiles have only 0.5 m of flat run-up)
EDGE = 3.4               # m: route points are kept this far inside the 8 m tile (the raster ends at 3.99; export routes end as
#                          close as 0.09 m to it, and a robot that arrived and took one more step was "off the course")
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


def tile(name, ttype, tile_id, route=0, upto=None, reverse=False, skip=None, back=False, via=None, scan_boxes=False):
    """A course on one tile of the export: its route `route`, optionally cut `upto` metres after the start. `via`: {export
    waypoint index: [(x, y), ...]} replaces that route point by these (a detour). `scan_boxes`: the height scan sees every
    physics box that stands on the floor (the walk raster used for the physics leaves them out)."""
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
    rw = [list(w[:2]) if isinstance(w, (list, tuple)) else [w["x"], w["y"]] for w in r["waypoints"]]
    if via:
        rw = [q for k, w in enumerate(rw) for q in ([list(v) for v in via[k]] if k in via else [w])]
    pts = [list(r["start"][:2])] + rw
    yaw = r["start"][2]
    if reverse:
        pts = pts[::-1]
        yaw = math.atan2(pts[1][1] - pts[0][1], pts[1][0] - pts[0][0])
    P = np.array(pts, float)
    P[0] = np.clip(P[0], -EDGE_START, EDGE_START)           # keep the start and the end off the tile's rim (see EDGE)
    P[1:] = np.clip(P[1:], -EDGE, EDGE)
    if skip is not None:                                    # start `skip` metres along the route (on a landing)
        seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
        cum = np.r_[0, np.cumsum(seg)]
        k = int(np.searchsorted(cum, skip))
        u = (skip - cum[k - 1]) / max(seg[k - 1], 1e-9)
        P = np.vstack((P[k - 1] + u * (P[k] - P[k - 1]), P[k:]))
        yaw = math.atan2(P[1][1] - P[0][1], P[1][0] - P[0][0])
        if upto is not None:
            upto -= skip
    if upto is not None:                                    # cut the polyline `upto` metres from the start
        seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
        cum = np.r_[0, np.cumsum(seg)]
        k = int(np.searchsorted(cum, upto))
        if k < len(P):
            u = (upto - cum[k - 1]) / max(seg[k - 1], 1e-9)
            P = np.vstack((P[:k], P[k - 1] + u * (P[k] - P[k - 1])))
    far = None
    if back:                                                # out and back: stand 1 s at the far end, turn on the spot, return
        far = len(P) - 1
        P = np.vstack((P, P[-2::-1]))
    # waypoints at most 1.2 m apart, so progress is seen and a stall is caught on long legs
    way, far_wp = [], None
    for k, (a, b) in enumerate(zip(P[:-1], P[1:])):
        n = max(1, int(math.ceil(np.linalg.norm(b - a) / 1.2)))
        way += [a + (b - a) * (j + 1) / n for j in range(n)]
        if far is not None and k + 1 == far:
            far_wp = len(way) - 1
    from harness import _seg_cross
    need = [dict(name=b["name"], a=b["a"], b=b["b"]) for b in beam_lines
            if any(_seg_cross(p, q, b["a"], b["b"]) is not None for p, q in zip(P[:-1], P[1:]))]
    extra = dict(beams=need, beam_lines=beam_lines, zones=zones, tile=tile_id, tile_route=route, level=t["level"], realised=t["realised"])
    g = z["height"][i].astype(float)
    c0 = z["cell0_center"] if "cell0_center" in z.files else np.array([-3.99, -3.99])
    if scan_boxes:
        extra["scan"] = "walk raster plus every physics box standing on the floor (eval monitor 2026-11-12)"
    else:
        extra["scan"] = "walk raster only"
    course = Course("ladder:" + name, g, float(c0[0]), float(c0[1]), float(z["resolution"]) if "resolution" in z.files else 0.02, boxes,
                  [float(P[0, 0]), float(P[0, 1]), math.degrees(yaw)],
                  [wp(float(p[0]), float(p[1]), hold=(1.0 if k in (len(way) - 1, far_wp) else 0.0),
                      name=("far end: stand 1 s, then turn back" if k == far_wp else None)) for k, p in enumerate(way)], name, extra)   # ends at a station
    if scan_boxes:
        add_scan_raster(course)
    return course


def add_scan_raster(course):
    """course.scan_height: the walk raster raised to the top of every physics box that stands on the floor (blocks, rocks,
    plants, furniture, posts, walls; not 4x4s, which are in the physics as boxes and handled as beams, and not overhead
    pieces). Only the height scan reads it (adapters/base.py); the physics height field stays the walk raster."""
    import check_routes as C
    sg = course.grid.copy()
    xs = course.x0 + course.res * np.arange(course.nx)
    ys = course.y0 + course.res * np.arange(course.ny)
    for k in C.floor_boxes(course):
        b = course.boxes[k]
        i = np.where((xs >= b["lo"][0] - course.res) & (xs <= b["hi"][0] + course.res))[0]
        j = np.where((ys >= b["lo"][1] - course.res) & (ys <= b["hi"][1] + course.res))[0]
        for a in i:
            for c in j:
                if C._foot_dist(b, xs[a], ys[c]) <= 0.5 * course.res:
                    sg[a, c] = max(sg[a, c], float(b["hi"][2]))
    course.scan_grid = sg

    def scan_height(x, y):
        i, j = course.cell(x, y)
        return sg[i, j]
    course.scan_height = scan_height
    return course


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
    # stairs_L3_v0: flat to x -3.4, six steps up to a 2.3 m landing at 0.84 m, six steps down, flat. Route 0 runs x -3.85 -> 2.42.
    "stairs_up": lambda: tile("stairs_up", "stairs", "stairs_L3_v0", 0, upto=3.0),             # from the floor to the middle of the landing
    "stairs_down": lambda: tile("stairs_down", "stairs", "stairs_L3_v0", 0, skip=3.0),         # from there down to the floor
    "log_corridor_walls": lambda: tile("log_corridor_walls", "log_corridors", "log_corridors_L3_v0", 0),
    "log_corridor_rails": lambda: tile("log_corridor_rails", "log_corridors", "log_corridors_L3_v1", 0),
}
# ---- step 2: PROPOSED routes (not run until the coordinator confirms the list). Export routes are 5 to 11 m; corridors and
# dead ends are walked out and back, which gives the turn on the spot and brings them to 13 to 15 m.
STEP2 = {
    "street_doorway_out_and_back": lambda: tile("street_doorway_out_and_back", "street_front", "street_front_L4_v0", 0, back=True, scan_boxes=True,
                                                  via={0: [(-1.6, 1.0), (-0.6, 1.0)]}),   # round the block on the export route (2026-11-12)
    "street_out_and_back": lambda: tile("street_out_and_back", "street_front", "street_front_L3_v0", 0, back=True, scan_boxes=True),
    "corridor_walls_out_and_back": lambda: tile("corridor_walls_out_and_back", "log_corridors", "log_corridors_L3_v0", 0, back=True, scan_boxes=True),
    "corridor_rails_out_and_back": lambda: tile("corridor_rails_out_and_back", "log_corridors", "log_corridors_L3_v1", 0, back=True, scan_boxes=True),
    "wedge_field_loop": lambda: tile("wedge_field_loop", "wedges", "wedges_L4_v1", 3, scan_boxes=True),
    "stairs_U_turn": lambda: tile("stairs_U_turn", "stairs", "stairs_L4_v4", 0, scan_boxes=True),
    "amphitheatre_loop": lambda: tile("amphitheatre_loop", "stairs", "stairs_L3_v6", 0, scan_boxes=True),
    "home_dead_end_and_back": lambda: tile("home_dead_end_and_back", "homes", "homes_L3_v0", 2, back=True, scan_boxes=True),
}
# what each step-2 route needs: skill -> level that has to be passed in step 1 before the route is scored for a checkpoint
# (the baseline pair runs every route once, as the reference). From the tiles' own dimensions.
STEP2_NEEDS = {
    "street_out_and_back": {"wedge": 1},                                   # gentle ramps up to 0.34 m, open space, a kerb
    "street_doorway_out_and_back": {"wedge": 1, "corridor": 3},            # the same street tile, through a doorway
    "corridor_walls_out_and_back": {"log": 2, "corridor": 1},              # 1.33 m corridor, four 9 cm logs each way
    "corridor_rails_out_and_back": {"log": 2, "corridor": 1},
    "wedge_field_loop": {"wedge": 3},                                      # pinwheel cells with crests of 0.21 to 0.27 m
    "stairs_U_turn": {"stairs_up": 2},                                     # rise 0.15 m, run 0.30 m, a U turn on the landing
    "amphitheatre_loop": {"stairs_up": 2, "stairs_down": 2},               # curved steps, rise 0.125 m
    "home_dead_end_and_back": {"corridor": 3},                             # a doorway into a furnished room and back
}


# the route's own size per needed skill, beside the levels' sizes (stairs: riser height in m), to flag a route whose obstacle is
# smaller than the lowest level passed while a lower level failed (coordinator 2026-11-12)
STEP2_SIZE = {"stairs_U_turn": {"stairs_up": 0.15}, "amphitheatre_loop": {"stairs_up": 0.125, "stairs_down": 0.125}}
LEVEL_SIZE = {"stairs_up": {1: 0.10, 2: 0.14, 3: 0.186}, "stairs_down": {1: 0.10, 2: 0.14, 3: 0.186}}
# routes whose course is wrong (eval monitor 2026-11-12): their rows are printed but are NOT results
STEP2_DEFECT = {
    "street_doorway_out_and_back": "COURSE DEFECT in runs before the fix, not a result: a 0.33 m block, 0.27 m tall, stands on the "
                                   "route's centre line (the export's own route) and is not in the walk raster that the height scan "
                                   "and the route check used; every baseline trial fell at it (x -1.45 to -1.28 m). Fixed 2026-11-12: "
                                   "detour south of it, scan sees floor boxes, route check tests them",
}


def eligible_routes(profile, with_notes=False):
    """Step-2 routes a checkpoint may run, from its step-1 profile: every needed skill passed at the needed level or a higher one
    (levels actually passed; a failed lower level does not block). A route whose obstacle is smaller than the lowest level
    passed, where a lower level failed, is held back and named in the notes."""
    ok, notes = [], {}
    for r, need in STEP2_NEEDS.items():
        good, why = True, []
        for k, lv in need.items():
            p = profile.get(k, {})
            passed = p.get("levels_passed", list(range(1, p.get("highest_level_passed", 0) + 1)))
            if not any(x >= lv for x in passed):
                good = False
                why.append("%s needs L%d, passed %s" % (k, lv, passed or "none"))
                continue
            size, low = STEP2_SIZE.get(r, {}).get(k), min(x for x in passed if x >= lv)
            failed_below = [x for x in p.get("levels_failed", []) if x < low]
            if size is not None and failed_below and size < LEVEL_SIZE[k][low]:
                good = False
                why.append("%s %.3f m is below the lowest level passed (L%d, %.3f m) and L%s failed" % (k, size, low, LEVEL_SIZE[k][low], ",".join(map(str, failed_below))))
        if good:
            ok.append(r)
        else:
            notes[r] = "; ".join(why)
    return (ok, notes) if with_notes else ok


FLAT = ("flat_walk", "flat_stop_stand", "flat_turn_in_place", "flat_turn_and_return", "flat_backward", "flat_sideways")
FLAT_GATE = FLAT[:4]                 # the only hard gate: 50 of 50 on each of these
STEP1_V0 = STEP1                     # the first version: one tile of the training export per skill (kept for continuity)


def _step1():
    import ladder_synth as LS
    d = {k: STEP1_V0[k] for k in FLAT}
    d.update(LS.TESTS)               # five obstacle skills x three levels on held-out built courses
    return d


STEPS = {1: _step1(), 2: STEP2, 10: STEP1_V0} if __name__ != "__main__" else {1: {}, 2: STEP2, 10: STEP1_V0}
STEP_DIR = {1: "step1", 2: "step2", 10: "step1_v0"}
SKILLS = ("log", "wedge", "stairs_up", "stairs_down", "corridor")
PASS = 45                            # of 50: a skill level is passed


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
        d = out / STEP_DIR[step] / test
        (d / "traj").mkdir(parents=True, exist_ok=True)
        results, t0 = [], time.time()
        if (d / "summary.json").exists() and not os.environ.get("LADDER_REDO"):
            results = json.load(open(d / "results.json"))               # done before: keep, or add trials up to the count asked
            if len(results) >= trials:
                print("%-20s already done (%d trials), kept" % (test, len(results)), flush=True)
                continue
        part = d / "partial.json"
        if part.exists():
            results = json.load(open(part))
        have = {r["seed"] for r in results}
        for i in range(trials):
            if seed0 + i in have:
                continue
            if time.time() - T0 > WALL:                 # never unbounded: stop cleanly, the same command continues later
                json.dump(results, open(part, "w"))
                print("wall limit reached in %s after %d trials; run the same command again" % (test, len(results)), flush=True)
                sys.exit(3)
            r = tr.run(seed0 + i, protocol, TIME_LIMIT_STEP2 if step == 2 else TIME_LIMIT, record=True)
            tr.save_traj(d / "traj" / ("%d.npz" % (seed0 + i)))
            results.append(r)
        results.sort(key=lambda r: r["seed"])
        if part.exists():
            os.remove(part)
        meta = dict(label=label, ladder_step=step, test=test, on="a ladder course below the apparatus (not an apparatus result)", apparatus=course.name, route=course.name,
                    tile=course.route.get("tile"), scan=course.route.get("scan", "walk raster only"), simulator="MuJoCo %s (CPU), dt %.4f s" % (mujoco.__version__, X.SIM_DT), adapter=adapter.name,
                    checkpoint=str(adapter.checkpoint), observes=adapter.observes, control_dt=adapter.control_dt, protocol=protocol_name,
                    plant=adapter.plant_name or "harness default vendor_20260823 (test fixture)",
                    seeds=[seed0, seed0 + trials - 1], time_limit=TIME_LIMIT_STEP2 if step == 2 else TIME_LIMIT, sections=course.sections,
                    definition=dict(radius=Hn.RADIUS, end_stand=Hn.END_STAND, stall=Hn.STALL, hit_f=Hn.HIT_F, hit_j=Hn.HIT_J))
        s = Hn.summarise(results, meta)
        s["renders"] = Hn.pick_renders(results)
        # every trajectory is kept (a test is 50 short trials): a later extension of the test may pick other trials to render
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
    return table


def collect(out, step):
    """The step's table from the per-test summaries on disk (tests run in several groups), as json and text."""
    rows, head = [], None
    for test in STEPS[step]:
        f = out / STEP_DIR[step] / test / "summary.json"
        if not f.exists():
            continue
        s = json.load(open(f))
        head = head or s
        course = STEPS[step][test]()
        rows.append(dict(test=test, trials=s["trials"], success=s["successes"], ci95=s["ci95"], clean=s["clean_successes"], mean_penalty=s["mean_stumble_penalty"],
                         side_hits_per_trial=s["side_hits"]["per_trial"], mean_waypoints=s["mean_waypoints"], of=len(course.waypoints), outcomes=s["outcomes"],
                         time_median=(s["time_success"] or {}).get("median"), length_m=round(float(np.linalg.norm(np.diff(course.path(), axis=0), axis=1).sum()), 1),
                         tile=s.get("tile"), renders=s.get("renders"), defect=STEP2_DEFECT.get(test) if step == 2 and "floor" not in str(s.get("scan")) else None, scan=s.get("scan", "walk raster only")))
    if not rows:
        return rows
    plants = sorted({str(json.load(open(out / STEP_DIR[step] / r["test"] / "summary.json")).get("plant")) for r in rows})
    head = dict(head, plant=" / ".join(plants) + ("  (MIXED PLANTS)" if len(plants) > 1 else ""),
                label="%s | %s" % (head.get("label") or head["adapter"], head.get("simulator", "MuJoCo (CPU)")))
    json.dump(dict(label=head.get("label"), plant=head.get("plant"), step=step, on="the training map (not an apparatus result)", adapter=head["adapter"], checkpoint=head["checkpoint"],
                   protocol=head["protocol"], rows=rows), open(out / (STEP_DIR[step] + ".json"), "w"), indent=1)
    lines = ["%s | ladder step %d, BELOW THE APPARATUS: skill courses and training-map tiles (not an apparatus result) | %s | plant %s | protocol %s" % (
        head.get("label") or head["adapter"], step, head["checkpoint"], head.get("plant"), head["protocol"]),
             "%-22s %-10s %-9s %-12s %-6s %-8s %-10s %-7s %s" % ("test", "success", "rate", "95 % interval", "clean", "penalty", "waypoints", "time", "outcomes / tile")]
    for r in rows:
        lines.append("%-22s %3d / %-4d %5.0f %%   %3.0f - %3.0f %%   %3d    %6.2f   %4.1f / %-3d %5s   %s%s" % (
            r["test"], r["success"], r["trials"], 100.0 * r["success"] / r["trials"], 100 * r["ci95"][0], 100 * r["ci95"][1], r["clean"], r["mean_penalty"],
            r["mean_waypoints"], r["of"], ("%.0f s" % r["time_median"]) if r["time_median"] else "-",
            ", ".join("%s %d" % kv for kv in sorted(r["outcomes"].items(), key=lambda kv: -kv[1])), (" | " + r["tile"]) if r["tile"] else ""))
        if r.get("defect"):
            lines.append("    ^ %s" % r["defect"])
    if step == 1:                                          # the skill profile: highest level passed per skill
        by = {r["test"]: r for r in rows}
        flat_ok = all(by.get(t, {}).get("success") == by.get(t, {}).get("trials") and by.get(t) for t in FLAT_GATE)
        lines += ["", "FLAT GATE (50 of 50 on %s): %s" % (", ".join(FLAT_GATE), "PASSED" if flat_ok else "NOT PASSED"), "",
                  "SKILL PROFILE (a level is passed at %d of 50; highest level passed, with any lower level that failed)" % PASS,
                  "%-12s %-14s %-22s %-22s %-22s" % ("skill", "highest passed", "level 1", "level 2", "level 3 (apparatus size)")]
        prof = {}
        for sk in SKILLS:
            # coordinator 2026-11-12: the highest level passed, whatever the lower ones did, plus any lower level that failed
            cells, best, passed, failed = [], 0, [], []
            for lv in (1, 2, 3):
                r = by.get("%s_L%d" % (sk, lv))
                cells.append("%d / %d (clean %d)" % (r["success"], r["trials"], r["clean"]) if r else "not run")
                if r and r["success"] * 50 >= PASS * r["trials"]:
                    best = lv
                    passed.append(lv)
                elif r:
                    failed.append(lv)
            below = [lv for lv in failed if lv < best]
            prof[sk] = dict(highest_level_passed=best, levels_passed=passed, levels_failed=failed, failed_below_highest=below, levels=cells)
            tag = ("L%d" % best if best else "none") + (" (L%s failed)" % ",".join(map(str, below)) if below else "")
            lines.append("%-12s %-14s %-22s %-22s %-22s" % (sk, tag, *cells))
        weakest = min(prof, key=lambda k: (prof[k]["highest_level_passed"] - 0.5 * len(prof[k]["failed_below_highest"]), sum(int(c.split(" ")[0]) for c in prof[k]["levels"] if c != "not run")))
        lines.append("weakest skill: %s" % weakest)
        elig, why = eligible_routes(prof, with_notes=True)
        lines += ["", "STEP 2 ROUTES ELIGIBLE: %s" % (", ".join(elig) or "none")] + ["  not eligible: %s (%s)" % kv for kv in why.items()]
        j = json.load(open(out / (STEP_DIR[step] + ".json")))
        j.update(flat_gate_passed=bool(flat_ok), profile=prof, weakest_skill=weakest, step2_eligible=elig, step2_not_eligible=why, pass_mark="%d of 50" % PASS)
        json.dump(j, open(out / (STEP_DIR[step] + ".json"), "w"), indent=1)
    open(out / (STEP_DIR[step] + ".txt"), "w").write(chr(10).join(lines) + chr(10))
    return rows


def main():
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
    print(open(pathlib.Path(a.out) / (STEP_DIR[a.step] + ".txt")).read())


if __name__ == "__main__":      # run through the imported module, so ladder_synth and this file share one `ladder`
    import ladder as _ladder
    _ladder.main()
