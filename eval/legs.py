"""Station legs (user, 2026-11-12): "instead of just the full route success rate, we should spawn, with some randomness, the robot at
each manipulation and see if it can make up to the next station successfully."

    python eval/legs.py --adapter acp:ACPHandover --apparatus ours --out DIR [--legs 1 2 3 4 5] [--trials 50] [--wall-limit 520]

Leg k runs from station k-1 to station k (leg 1 from the route's start) over the full route's own waypoints between them.
- Spawn: leg 1 as a full trial (route start, nominal jitter). Legs 2..: standing at station k-1, position uniform inside its 0.20 m
  arrival disc, heading = the station's heading +- 20 deg (per-seed draw); a draw is rejected and redrawn (up to 200 times) if the
  ground under the two soles is not one plane (residual spread over the 0.26 x 0.4 m sole area > 2 cm: the ramp stations stand on a
  15 deg slope, so "level" means no edge or step under the feet) or a wall / post box comes within 0.25 m of the
  pelvis footprint. The rest of the nominal protocol is unchanged. The robot first gets PRESTAND (1 s) of a stand command at its
  spawn pose, so every adapter's history (and ACP's planner input) fills with real standing frames; adapters reset as at a route start.
- Success: the leg's last station counts under the strict rule (1 s stand, corrective steps allowed); for the last leg the end stand
  as in a full trial. A 4x4 crossed by the leg's own path must be stepped over. Falls, stalls (45 s) and leaving the course as in a
  full trial. Time limit LEG_TIME_S per leg (scaled: 180 s x leg length / route length, at least 60 s).
Writes DIR/legK/{summary.json, results.json, trials.csv, events.csv} and DIR/legs.json / legs.txt (per leg, and the product).
"""
import argparse
import copy
import json
import math
import os
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import harness as Hn  # noqa: E402
import x2 as X  # noqa: E402
from apparatus import Apparatus  # noqa: E402
import ladder as L  # noqa: E402  (chunk helpers: chunk_seeds, merge_chunks, run_chunk)

PRESTAND = 1.0           # s of stand command at the spawn pose before the first waypoint
SPAWN_R = 0.20           # m: uniform in the station's arrival disc
SPAWN_YAW = 20.0         # deg either side of the station heading
LEVEL_TOL = 0.02         # m: raster spread allowed under the soles
WALL_CLEAR = 0.25        # m: no wall / post box this close to the pelvis footprint
MIN_LEG_TIME = 60.0      # s


def seg_cross(p, q, a, b):
    d1, d2 = q - p, b - a
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-12:
        return False
    t = ((a[0] - p[0]) * d2[1] - (a[1] - p[1]) * d2[0]) / den
    u = ((a[0] - p[0]) * d1[1] - (a[1] - p[1]) * d1[0]) / den
    return 0 <= t <= 1 and 0 <= u <= 1


def legs_of(app):
    st = [i for i, w in enumerate(app.waypoints) if w["hold_s"] > 0 and not w.get("test_stop")]   # test_stop: a TEST-course stop inside a leg
    out, prev = [], None
    for k, i in enumerate(st):
        lo = 0 if prev is None else prev + 1
        out.append(dict(k=k + 1, start=prev, idx=list(range(lo, i + 1)), name="%s -> %s" % ("start" if prev is None else app.waypoints[prev]["name"], app.waypoints[i]["name"])))
        prev = i
    return out


def leg_course(app, leg):
    c = copy.copy(app)
    c.route = dict(app.route)
    c.waypoints = [dict(app.waypoints[i]) for i in leg["idx"]]
    c.sections = sorted({w["section"] for w in c.waypoints}, key=[w["section"] for w in app.waypoints].index)
    sp = app.spawn if leg["start"] is None else [app.waypoints[leg["start"]]["x"], app.waypoints[leg["start"]]["y"], app.waypoints[leg["start"]]["heading_deg"] or 0.0]
    c.spawn = list(sp)
    P = np.array([sp[:2]] + [[w["x"], w["y"]] for w in c.waypoints], float)
    beams = [b for b in app.route.get("beams", []) if any(seg_cross(P[j], P[j + 1], np.array(b["a"]), np.array(b["b"])) for j in range(len(P) - 1))]
    c.route["beams"] = beams
    c.route.setdefault("beam_lines", app.route.get("beams", []))
    c.length = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
    return c


def spawn_ok(app, x, y):
    pts = np.array([(x + dx, y + dy) for dx in np.linspace(-0.13, 0.13, 6) for dy in np.linspace(-0.2, 0.2, 9)])
    h = np.array([app.height(px, py) for px, py in pts])
    A = np.c_[pts - pts.mean(0), np.ones(len(pts))]          # "level" = on one plane: the stations on the ramps stand on a 15 deg slope
    res = h - A @ np.linalg.lstsq(A, h, rcond=None)[0]
    if res.max() - res.min() > LEVEL_TOL:
        return False
    for b in app.boxes:
        if b["center"][2] - b["half"][2] > 1.3:
            continue
        if math.hypot(max(abs(x - b["center"][0]) - b["half"][0], 0.0), max(abs(y - b["center"][1]) - b["half"][1], 0.0)) < WALL_CLEAR:
            return False
    return True


def draw_spawn(app, leg, base, seed):
    if leg["start"] is None:
        return list(base), 0
    rng = np.random.default_rng(10_000 + seed)
    for n in range(200):
        r, a = SPAWN_R * math.sqrt(rng.uniform()), rng.uniform(0, 2 * math.pi)
        x, y = base[0] + r * math.cos(a), base[1] + r * math.sin(a)
        if spawn_ok(app, x, y):
            return [x, y, base[2] + float(rng.uniform(-SPAWN_YAW, SPAWN_YAW))], n
    return None, 200


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--apparatus", default="ours")
    ap.add_argument("--out", required=True)
    ap.add_argument("--legs", type=int, nargs="*", default=None)
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--seed0", type=int, default=Hn.SEED0)
    ap.add_argument("--wall-limit", type=float, default=0)
    ap.add_argument("--label", default="")
    ap.add_argument("--chunk", type=int, default=None, help="run only chunk k (0-based) of the seeds of ONE leg into legK/chunks/c<k>.json; "
                    "the same command without --chunk merges the chunks into summary.json (automatic scorer, 2026-11-14)")
    ap.add_argument("--chunk-size", type=int, default=5)
    a = ap.parse_args()
    if a.chunk is not None and (not a.legs or len(a.legs) != 1):
        sys.exit("--chunk needs exactly one leg (--legs K)")
    import memcap
    memcap.guard(cap_gb=6, need_free_gb=12, low_priority=True)
    out = pathlib.Path(a.out)
    app = Apparatus(a.apparatus)
    adapter = Hn.load_adapter(a.adapter, a.checkpoint)
    route_len = float(np.linalg.norm(np.diff(app.path(), axis=0), axis=1).sum())
    proto = dict(Hn.NOMINAL, start_xy=0.0, start_yaw_deg=0.0)
    t0 = time.time()
    for leg in legs_of(app):
        if a.legs and leg["k"] not in a.legs:
            continue
        d = out / ("leg%d" % leg["k"])
        if (d / "summary.json").exists():
            continue
        (d / "traj").mkdir(parents=True, exist_ok=True)
        c = leg_course(app, leg)
        limit = max(MIN_LEG_TIME, Hn.TIME_LIMIT * c.length / route_len) + PRESTAND
        x = X.build(c, adapter.plant(X.joint_names_of(X.robot_dir() / "mjcf" / "x2_ultra.xml")))
        base = list(c.spawn)
        part = d / "partial.json"
        results = json.load(open(part)) if part.exists() else []
        results = L.merge_chunks(d, results)            # chunk files of a chunked run (automatic scorer, 2026-11-14)
        have = {r["seed"] for r in results}

        def one(seed):
            sp, tries = draw_spawn(c, leg, base, seed)
            if sp is None:
                print("leg %d seed %d: no clean spawn in 200 draws" % (leg["k"], seed), flush=True)
                return None
            c.spawn = sp
            tr = Hn.Trial(x, c, adapter, x.model.body_mass.copy(), x.model.body_inertia.copy())
            r = tr.run(seed, proto if leg["start"] is not None else Hn.NOMINAL, limit, record=True, prestand=(PRESTAND if leg["start"] is not None else 0.0))
            r["leg"], r["spawn"], r["spawn_redraws"] = leg["k"], [round(v, 3) for v in sp], tries
            tr.save_traj(d / "traj" / ("%d.npz" % seed))
            print("leg %d seed %d: %s, %d/%d, %.1f s" % (leg["k"], seed, r["reason"], r["waypoints"], r["of"], r["sim_time"]), flush=True)
            return r
        if a.chunk is not None:
            L.T0, L.WALL = t0, (a.wall_limit or 1e12)
            L.run_chunk(d, a.chunk, a.chunk_size, a.trials, a.seed0, have, one)
            return
        for i in range(a.trials):
            seed = a.seed0 + i
            if seed in have:
                continue
            if a.wall_limit and time.time() - t0 > a.wall_limit:
                print("wall limit in leg %d after %d trials; run again" % (leg["k"], len(results)), flush=True)
                sys.exit(3)
            r = one(seed)
            if r is None:
                continue
            results.append(r)
            json.dump(results, open(str(part) + ".tmp", "w")); os.replace(str(part) + ".tmp", part)
        results.sort(key=lambda r: r["seed"])
        meta = dict(label=a.label, leg=leg["k"], leg_name=leg["name"], apparatus=a.apparatus, route="%s leg %d" % (app.route.get("name"), leg["k"]),
                    simulator="MuJoCo %s (CPU)%s" % (__import__("mujoco").__version__, X.model_note()), policy_trained_on=getattr(adapter, "trained_model", None),
                    adapter=adapter.name, checkpoint=str(adapter.checkpoint), observes=adapter.observes, control_dt=adapter.control_dt,
                    plant=adapter.plant_name, protocol="nominal; legs 2-5 spawn uniform in the station disc, heading +-20 deg", seeds=[a.seed0, a.seed0 + a.trials - 1],
                    time_limit=limit, prestand_s=PRESTAND if leg["start"] is not None else 0.0, leg_length_m=round(c.length, 2), sections=c.sections,
                    definition=dict(radius=Hn.RADIUS, end_stand=Hn.END_STAND, stall=Hn.STALL))
        s = Hn.summarise(results, meta)
        s["renders"] = Hn.pick_renders(results)
        json.dump(results, open(d / "results.json", "w"), indent=1)
        json.dump(s, open(d / "summary.json", "w"), indent=1)
        Hn.write_logs(d, results, c.waypoints)
        if part.exists():
            os.remove(part)
        if (d / "chunks").exists():
            __import__("shutil").rmtree(d / "chunks", ignore_errors=True)
    # the table
    rows, prod, prod_s = [], 1.0, 1.0
    for leg in legs_of(app):
        f = out / ("leg%d" % leg["k"]) / "summary.json"
        if not f.exists():
            continue
        s = json.load(open(f))
        # final-station stop rule (user 2026-11-13): successes = headline (strict 0.2 m + stopped short); strict beside it
        n, k, ks = s["trials"], Hn.main_count(s), Hn.strict_count(s)
        prod *= k / max(n, 1)
        prod_s *= ks / max(n, 1)
        rows.append(dict(leg=leg["k"], name=leg["name"], trials=n, successes=k, strict=ks, stopped_short=s.get("stopped_short"),
                         ci95=s.get("headline_ci95") or s.get("strict_ci95") or s["ci95"], outcomes=s["outcomes"], failures_by_section=s["failures_by_section"]))
    json.dump(dict(rows=rows, product=round(prod, 4), product_strict=round(prod_s, 4)), open(out / "legs.json", "w"), indent=1)
    lines = ["STATION LEGS (user 2026-11-12) | %s | %s | headline = strict 0.2 m + stopped short (within 0.4 m, user 2026-11-13)" % (a.label, adapter.checkpoint)]
    for r in rows:
        lines.append("leg %d  %-40s %2d / %d  (%.0f-%.0f %%)  strict %2d, stopped short %s  %s" % (r["leg"], r["name"], r["successes"], r["trials"], 100 * r["ci95"][0], 100 * r["ci95"][1],
                                                                    r["strict"], "-" if r["stopped_short"] is None else r["stopped_short"],
                                                                    ", ".join("%s %d" % kv for kv in sorted(r["outcomes"].items(), key=lambda kv: -kv[1]))))
    lines.append("product of the leg rates: %.3f (strict 0.2 m: %.3f)" % (prod, prod_s))
    open(out / "legs.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
