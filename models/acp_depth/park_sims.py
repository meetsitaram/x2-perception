"""Episodes for the multi-robot park render (render_park.py): ACP on the fine depth student (D435-like noise in the loop)
walks terrain-park extended_v4k tiles through the eval ladder (eval physics), one process per episode, recording the
trajectory and the maps ACP was given (ACPD_EXTRA).

    python3 models/acp_depth/park_sims.py --student runs/fine_v3/student.pt --out runs/park [--level 4] [--workers 12]

Slots (feature -> tile type, variant, route kinds): see SLOTS. Per slot several candidate routes are run; plan.json lists them,
results land in <out>/<name>/ (ladder run dir with traj/) and <out>/extra_<name>/ep0001.npz.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import resource
import sys
import time
from multiprocessing import Pool

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from collect import ENV  # noqa: E402

# feature: (tile type, variant_kind or None, route kinds in preference order, candidates)
SLOTS = {        # 2026-11-17, user on the first cut: "too few x2s": 3-4 robots per tile
    "4x4s": ("single_beams", None, None, 5),
    "4x4_field": ("beam_fields", None, None, 4),
    "blocks": ("blocks", None, None, 5),
    "ramps": ("wedges", "long_ramp", None, 5),
    "wedges": ("wedges", "pinwheel", None, 5),
    "stairs_straight_up": ("stairs", "straight", ("straight", "side_left", "oblique"), 3),
    "stairs_straight_down": ("stairs", "straight", ("descend", "side_down"), 2),
    "stairs_L_up": ("stairs", "L", ("straight", "C", "oblique"), 3),
    "stairs_L_down": ("stairs", "L", ("descend", "side_down"), 2),
    "stairs_U_up": ("stairs", "U", ("straight", "oblique", "side_right"), 3),
    "stairs_U_down": ("stairs", "U", ("descend", "side_down"), 2),
    "homes": ("homes", None, None, 4),                  # user: "Where are the home sections with walls, furniture ..."
    "homes_L6": ("homes", None, None, 4, 6),            # a second room layout (level 6)
    # user, 2026-11-17: "more coverage stepping on the big concrete block and slanting wedges, also walking through hallways
    # stepping on 4x4s"
    "blocks_big": ("blocks", None, None, 5, 9),         # blocks up to 0.25 m
    "wedges_steep": ("wedges", "pinwheel", None, 5, 8), # slopes 14-24 deg
    "hallway": ("log_corridors", None, None, 2, 4),     # house-height corridor walls, five 4x4s
    "hallway_L6": ("log_corridors", None, None, 2, 6),
    "hallway_L8": ("log_corridors", None, None, 2, 8),
}


def plan(level):
    D = pathlib.Path(ENV["PA_TERRAIN_DIR"])
    jobs = []
    for slot, (tt, vk, kinds, n, *lv) in SLOTS.items():
        lev = lv[0] if lv else level
        tiles = [t for t in json.load(open(D / (tt + ".json")))["tiles"] if t["level"] == lev and (vk is None or t.get("variant_kind") == vk)]
        t = tiles[0]
        rs = list(range(len(t["routes"])))
        if kinds:
            rs = [r for k in kinds for r in rs if t["routes"][r].get("kind") == k]
        for r in rs[:n]:
            jobs.append(dict(slot=slot, ttype=tt, tile_id=t["tile_id"], route=r, kind=t["routes"][r].get("kind"),
                             name="%s_%s_r%d" % (slot, t["tile_id"], r)))
    return jobs


def work(arg):
    job, out, student = arg
    resource.setrlimit(resource.RLIMIT_AS, (8 << 30, 8 << 30))
    for k, v in ENV.items():
        os.environ.setdefault(k, v)
    os.environ.update(ACPD_MODE="student", ACPD_STUDENT=student, ACPD_DEPTH_NOISE="1",
                      ACPD_EXTRA=str(out / ("extra_" + job["name"])))
    sys.path.insert(0, str(REPO / "eval"))
    import ladder as L
    name = job["name"]
    L.STEPS[30] = {name: (lambda: L.tile(name, job["ttype"], job["tile_id"], job["route"]))}
    L.STEP_DIR[30] = "park"
    t0 = time.time()
    try:
        L.run("acp_depth:ACPDepth", None, 30, [name], 1, out / "ladder", "nominal", "park render", seed0=3000)
        import gc
        from adapters.acp_depth import ACPDepth
        for o in gc.get_objects():
            if isinstance(o, ACPDepth):              # pool workers skip atexit: write the episode and its overlay data now
                o._flush()
                o._flush_extra()
        r = json.load(open(out / "ladder" / "park" / name / "results.json"))[0]
        return name, "%s %s/%s" % ("ok" if r["success"] else "FAIL", r.get("waypoints"), r.get("of")), time.time() - t0
    except BaseException as e:
        return name, "crash: %s" % str(e)[:150], time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--level", type=int, default=4)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    out = pathlib.Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    jobs = plan(a.level)
    done = {j["name"] for j in jobs if (out / "ladder" / "park" / j["name"] / "results.json").exists()
            and (out / ("extra_" + j["name"]) / "ep0001.npz").exists()}
    json.dump(dict(jobs=jobs, level=a.level, student=a.student), open(out / "plan.json", "w"), indent=1)
    print("%d episodes (%d done before), %d workers -> %s" % (len(jobs), len(done), a.workers, out), flush=True)
    with Pool(a.workers, maxtasksperchild=1) as pool:
        for n, (name, status, dt) in enumerate(pool.imap_unordered(work, [(j, out, str(pathlib.Path(a.student).resolve())) for j in jobs if j["name"] not in done]), 1):
            print("[%d/%d] %-44s %s (%.0f s)" % (n, len(jobs), name, status, dt), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
