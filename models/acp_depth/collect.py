"""Teacher data for the depth -> height-scan student: ACP (perfect scans) walks terrain-park's training tiles (extended_v4k,
the map the policies trained on; never an eval course: the apparatus and the eval ladder's courses stay held out), recorded
by eval/adapters/acp_depth.py (ACPD_MODE=record).

    python3 models/acp_depth/collect.py --out models/acp_depth/runs/data_v1 [--levels 0-6] [--per-type 30] [--seeds 2] [--workers 20]

Per course (one tile route) and seed: one episode file <out>/data/<pid>_<n>.npz. Ladder results per course in <out>/ladder/.
Each worker caps its memory at 8 GB. Env defaults point at this PC's paths (PA_ROBOT_DIR, ACP_HANDOVER_DIR, PA_TERRAIN_DIR).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import resource
import sys
import time
from multiprocessing import Pool

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
HOME = pathlib.Path.home()
ENV = dict(EVAL_REGRESSION="0", ACPD_MODE="record", ACP_GOAL_R="0.2", ACP_LOOKAHEAD="0.3", X2_EXCLUDE_PAIRS="hip",
           ACP_HANDOVER_DIR=str(next((d / "release/ckpt/acp" for d in (REPO, *REPO.parents) if (d / "release/ckpt/acp").exists()),
                                     HOME / "projects/x2-perception/release/ckpt/acp")),
           PA_ROBOT_DIR=str(HOME / "Projects/gwbc-x2/gear_sonic/data/assets/robot_description"),
           PA_TERRAIN_DIR=str(HOME / "Projects/terrain-park/terrains/extended_v4k"))
TYPES = ("single_beams", "beam_fields", "blocks", "wedges", "log_corridors", "stairs", "tight_spots", "homes", "street_front")


def courses(levels, per_type, rng, extra=None):
    D = pathlib.Path(os.environ.get("PA_TERRAIN_DIR", ENV["PA_TERRAIN_DIR"]))
    out = []
    for t in TYPES:
        J = json.load(open(D / (t + ".json")))
        tiles = [x for x in (J if isinstance(J, list) else J["tiles"]) if x["level"] in levels]
        pool = [(t, x["tile_id"], r) for x in tiles for r in range(len(x.get("routes", [])))]
        rng.shuffle(pool)
        out += pool[:per_type + (extra or {}).get(t, 0)]
    return out


def work(job):
    (ttype, tile_id, route), seeds, seed0, out = job
    resource.setrlimit(resource.RLIMIT_AS, (8 << 30, 8 << 30))
    for k, v in ENV.items():
        os.environ.setdefault(k, v)
    os.environ["ACPD_OUT"] = str(out / "data")
    sys.path.insert(0, str(REPO / "eval"))
    import ladder as L
    name = "tp_%s_%s_r%d" % (ttype, tile_id, route)
    L.STEPS[30] = {name: (lambda: L.tile(name, ttype, tile_id, route))}
    L.STEP_DIR[30] = "tp"
    t0 = time.time()
    try:
        L.run("acp_depth:ACPDepth", None, 30, [name], seeds, out / "ladder", "nominal", "acp_depth teacher data", seed0=seed0)
        import gc
        from adapters.acp_depth import ACPDepth
        for o in gc.get_objects():          # pool workers skip atexit: write the last episode now
            if isinstance(o, ACPDepth):
                o._flush()
        return name, "ok", time.time() - t0
    except BaseException as e:     # a course that does not build or crashes is skipped, the others go on
        return name, "fail: %s" % str(e)[:120], time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--levels", default="0-6")
    ap.add_argument("--per-type", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--extra", default="", help="more routes for some types, e.g. single_beams:35,beam_fields:35 (the 4x4s)")
    a = ap.parse_args()
    lo, hi = (int(v) for v in a.levels.split("-"))
    out = pathlib.Path(a.out)
    (out / "data").mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    extra = {k: int(v) for k, v in (x.split(":") for x in a.extra.split(",") if x)}
    cs = courses(set(range(lo, hi + 1)), a.per_type, rng, extra)
    json.dump(dict(courses=cs, args=vars(a), env=ENV), open(out / "plan.json", "w"), indent=1)
    print("%d courses x %d seeds, %d workers -> %s" % (len(cs), a.seeds, a.workers, out), flush=True)
    jobs = [(c, a.seeds, 5000 + 10 * i, out) for i, c in enumerate(cs)]
    with Pool(a.workers, maxtasksperchild=4) as pool:
        for n, (name, status, dt) in enumerate(pool.imap_unordered(work, jobs), 1):
            print("[%d/%d] %-40s %s (%.0f s)" % (n, len(jobs), name, status, dt), flush=True)
    files = list((out / "data").glob("*.npz"))
    print("DONE: %d episode files, %.2f GB" % (len(files), sum(f.stat().st_size for f in files) / 1e9), flush=True)


if __name__ == "__main__":
    main()
