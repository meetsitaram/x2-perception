"""ACP stage 3 goal-reaching tiles (arXiv 2610.31608, Methods "Stage 3"; our families, see state/acp.md section 15).

The simulator gives every distinct prop-box size a slot in every world, so stage-3 tiles are built ONLY from box sizes
(and at most as many of each) that the stage-2 reference props already use: tread slabs 0.3 x {0.9, 1.2, 1.5} x 0.3 m
(7 each), landing slabs 1.5 x {0.9, 1.2, 1.5} x 0.3 m (1 each), two walls 4.6 x 0.05 x 2.76 m, one 4x4 log
(2.44 x 0.089 x 0.089 m), one 2x2 log, and the small blocks of the block set. Box heights are set by sinking a box
into the floor (only its top is walked on), so risers and log heights vary continuously with no new slot. Result: no new
prop slots, no extra GPU cost per world.

Families (6) x levels (10) x seeds; each tile has a spawn, a route of waypoints (x, y, floor z, heading, station flag) and
v_max. Routes follow the user's eval rule (docs/EVAL.md): arrival within 0.2 m, stations = 1 s stand, sharp heading turns
right after a waypoint (+-70-90 degrees at the higher levels), backward / sideways segments on flat routes.

    python -m acp.goals OUT.json [SEEDS_PER_LEVEL=12]      (CPU)
"""
from __future__ import annotations

import json
import math
import sys

import numpy as np

TREAD = {0.45: (0.15, 0.45, 0.15), 0.6: (0.15, 0.6, 0.15), 0.75: (0.15, 0.75, 0.15)}       # half extents, by half width
LANDING = {0.45: (0.75, 0.45, 0.15), 0.6: (0.75, 0.6, 0.15), 0.75: (0.75, 0.75, 0.15)}
WALL = (2.3, 0.025, 1.38)
LOG4 = (1.219, 0.044, 0.044)
LOG2 = (1.219, 0.019, 0.019)
BLOCKS = [((0.05, 0.05, 0.025), 4), ((0.05, 0.05, 0.05), 4), ((0.1, 0.06, 0.05), 3), ((0.1, 0.06, 0.025), 3),
          ((0.15, 0.1, 0.05), 3), ((0.1, 0.06, 0.075), 3), ((0.15, 0.1, 0.025), 3), ((0.05, 0.05, 0.075), 3),
          ((0.15, 0.1, 0.075), 2)]
MAX_TREADS = 7
FAMILIES = ["stairs_up", "stairs_down", "logs", "corridor", "blocks", "flat_turns"]
V_MAX_STAIRS, V_MAX = 0.8, 2.0
X_OFF, Y_OFF = -40.0, 0.0          # tiles live here in the world (the shared ramp raster is at x = +30)
APP = ((0.186, 0.222), (0.203, 0.244))   # apparatus (riser, tread): never drawn within 6 mm of both


def box(half, cx, cy, top, yaw=0.0):
    """A box of `half` whose top is at height `top` (sunk into the floor when top < 2 * half z)."""
    return dict(center=[cx + X_OFF, cy + Y_OFF, top - half[2]], half=list(half), yaw=yaw)


def wp(x, y, z, heading, station=False):
    return [x + X_OFF, y + Y_OFF, z, heading, 1.0 if station else 0.0]


def turn_after(level, rng):
    """Heading change right after a waypoint: grows with the level to +-70-90 degrees."""
    mag = math.radians(rng.uniform(20, 45) + level * rng.uniform(4.0, 5.0))
    return min(mag, math.radians(90)) * (1 if rng.random() < 0.5 else -1)


def leg_after(x, y, heading, dist):
    return x + dist * math.cos(heading), y + dist * math.sin(heading)


def stairs(level, rng, down):
    t = level / 9
    while True:
        rise = rng.uniform(0.08, 0.11) + t * rng.uniform(0.08, 0.11)
        run = rng.uniform(0.25, 0.30)
        if not any(abs(rise - r) < 0.006 and abs(run - tr) < 0.006 for r, tr in APP):
            break
    n = int(rng.integers(3, MAX_TREADS + 1))
    hw = float(rng.choice(list(TREAD)))
    walls = rng.choice(["none", "left", "right", "both"], p=[0.4 - 0.2 * t, 0.2, 0.2, 0.2 + 0.2 * t])
    x0 = 0.0
    boxes = []
    for k in range(1, n + 1):                                   # tread k: top at k * rise, front edge at x0 + (k - 1) * run
        xa = x0 + (k - 1) * run
        boxes.append(box(TREAD[hw], xa + 0.15, 0.0, k * rise))
    xl = x0 + n * run                                           # landing after the last tread
    boxes.append(box(LANDING[hw], xl + 0.75, 0.0, (n + 1) * rise))
    top = (n + 1) * rise
    for side in {"left": (1,), "right": (-1,), "both": (1, -1)}.get(str(walls), ()):
        boxes.append(box(WALL, x0 - 1.0 + 2.3, side * (hw + 0.025 + rng.uniform(0.0, 0.05)), 2.76))
    land_c = xl + 0.75
    if not down:
        spawn = [x0 - rng.uniform(1.0, 1.6), 0.0, 0.0, 0.0]
        route = [wp(land_c - 0.25, 0.0, top, 0.0)]
        h = turn_after(level, rng)
        x, y = leg_after(land_c - 0.25, 0.0, h, 0.5)
        y = float(np.clip(y, -hw + 0.25, hw - 0.25))
        route.append(wp(x, y, top, h, station=True))
    else:
        spawn = [land_c + 0.3, 0.0, top, math.pi]
        route = [wp(x0 - 0.6, 0.0, 0.0, math.pi)]
        h = math.pi + turn_after(level, rng)
        x, y = leg_after(x0 - 0.6, 0.0, h, rng.uniform(1.0, 1.5))
        route.append(wp(x, y, 0.0, h, station=True))
    return boxes, spawn, route, V_MAX_STAIRS, dict(rise=round(rise, 3), run=round(run, 3), steps=n, width=2 * hw, walls=str(walls))


def logs(level, rng):
    t = level / 9
    h4 = 0.03 + t * (2 * LOG4[2] - 0.03) * rng.uniform(0.85, 1.0)
    yaw = math.pi / 2 + rng.uniform(-0.3, 0.3) * t
    xl = rng.uniform(1.2, 1.8)
    boxes = [box(LOG4, xl, 0.0, h4, yaw)]
    extra = {}
    if level >= 4 and rng.random() < 0.6:                       # a 2x2 log further on
        x2 = xl + rng.uniform(0.7, 1.2)
        boxes.append(box(LOG2, x2, 0.0, 2 * LOG2[2], math.pi / 2 + rng.uniform(-0.3, 0.3)))
        extra["log2_x"] = round(x2, 2)
    xe = max(b["center"][0] - X_OFF for b in boxes) + 0.6
    spawn = [0.0, 0.0, 0.0, 0.0]
    route = [wp(xe, 0.0, 0.0, 0.0)]
    h = turn_after(level, rng)
    x, y = leg_after(xe, 0.0, h, rng.uniform(1.0, 1.6))
    route.append(wp(x, y, 0.0, h, station=True))
    return boxes, spawn, route, V_MAX, dict(log_height=round(h4, 3), log_yaw=round(yaw, 2), **extra)


def corridor(level, rng):
    t = level / 9
    gap = 1.3 - t * rng.uniform(0.45, 0.55)                     # 1.3 m -> about 0.8 m between the walls
    x0 = rng.uniform(0.8, 1.2)
    boxes = [box(WALL, x0 + 2.3, side * (gap / 2 + 0.025), 2.76) for side in (1, -1)]
    extra = {}
    if level >= 3:
        xl = x0 + float(rng.choice([rng.uniform(1.0, 1.8), rng.uniform(2.8, 3.6)]))   # not within 0.5 m of the mid-corridor waypoint
        hl = 0.03 + t * (2 * LOG4[2] - 0.03)
        boxes.append(box(LOG4, xl, 0.0, hl, math.pi / 2))
        extra.update(log_x=round(xl, 2), log_height=round(hl, 3))
    spawn = [0.0, 0.0, 0.0, 0.0]
    xe = x0 + 4.6 + 0.5
    route = [wp(x0 + 2.3, 0.0, 0.0, 0.0), wp(xe, 0.0, 0.0, 0.0)]
    h = turn_after(level, rng)
    x, y = leg_after(xe, 0.0, h, rng.uniform(1.0, 1.5))
    route.append(wp(x, y, 0.0, h, station=True))
    return boxes, spawn, route, V_MAX, dict(gap=round(gap, 3), **extra)


def blocks(level, rng):
    t = level / 9
    boxes = []
    for half, cnt in BLOCKS:
        for _ in range(int(round(cnt * (0.3 + 0.7 * t)))):
            boxes.append(box(half, rng.uniform(0.6, 3.6), rng.uniform(-0.8, 0.8), 2 * half[2], rng.uniform(-math.pi, math.pi)))
    spawn = [0.0, 0.0, 0.0, 0.0]
    route, x, y, h = [], 0.0, 0.0, 0.0
    for i in range(3):
        x, y = leg_after(x, y, h, rng.uniform(1.2, 1.6))
        y = float(np.clip(y, -1.5, 1.5))
        h = h + (turn_after(level, rng) if i < 2 else 0.0)
        route.append(wp(x, y, 0.0, h, station=(i == 2)))
    keep = [p[:2] for p in route] + [[spawn[0] + X_OFF, spawn[1] + Y_OFF]]       # no block on or next to a waypoint or the spawn
    boxes = [bx for bx in boxes if min(math.hypot(bx["center"][0] - k[0], bx["center"][1] - k[1]) for k in keep) > 0.35]
    return boxes, spawn, route, V_MAX, dict(blocks=len(boxes))


def flat_turns(level, rng):
    """Flat ground: sharp turns after waypoints, turn in place, backward and sideways segments (the robot's heading at a
    waypoint differs from the direction it walked in)."""
    spawn = [0.0, 0.0, 0.0, 0.0]
    route, x, y, face = [], 0.0, 0.0, 0.0
    for i in range(4):
        kind = rng.choice(["forward", "backward", "sideways", "turn_in_place"],
                          p=[0.55 - 0.3 * level / 9, 0.15 + 0.1 * level / 9, 0.15 + 0.1 * level / 9, 0.15 + 0.1 * level / 9])
        if kind == "turn_in_place":
            face = face + turn_after(max(level, 5), rng)
            route.append(wp(x, y, 0.0, face, station=True))
            continue
        travel = face + {"forward": 0.0, "backward": math.pi, "sideways": math.pi / 2 * (1 if rng.random() < 0.5 else -1)}[str(kind)]
        d = rng.uniform(0.8, 1.6) if kind == "forward" else rng.uniform(0.5, 1.0)
        x, y = leg_after(x, y, travel, d)
        route.append(wp(x, y, 0.0, face, station=(i == 3)))
        face = face + (turn_after(level, rng) if kind == "forward" else 0.0)
    route[-1][4] = 1.0
    return [], spawn, route, V_MAX, dict(kinds="mixed")


def standing_pose():
    """Default joint pose and the root height that puts the lowest sole point on the floor (MuJoCo CPU)."""
    import mujoco
    sys.path.insert(0, "/mnt/filesystem-g4/tracks/acp/code")
    from gear_sonic.supermimic.assets import build_x2
    x2 = build_x2("mc_20261024")
    m, d = x2.model, mujoco.MjData(x2.model)
    d.qpos[:] = m.qpos0; d.qpos[2] = 1.0; d.qpos[3:7] = [1, 0, 0, 0]; d.qpos[7:] = x2.default_q
    mujoco.mj_kinematics(m, d)
    low = min(float(d.geom_xpos[g][2] - (m.geom_size[g][0] if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE else 0.0))
              for g in range(m.ngeom) if m.geom_bodyid[g] > 0 and "ankle" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or "")
              and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE)
    return [float(v) for v in x2.default_q], round(1.0 - low + 0.005, 4)


def build(seeds):
    q0, h0 = standing_pose()
    tiles = []
    for fam in FAMILIES:
        for level in range(10):
            for s in range(seeds):
                rng = np.random.default_rng(1000 * FAMILIES.index(fam) + 100 * level + s)
                if fam in ("stairs_up", "stairs_down"):
                    b, sp, rt, vm, info = stairs(level, rng, fam == "stairs_down")
                else:
                    b, sp, rt, vm, info = {"logs": logs, "corridor": corridor, "blocks": blocks, "flat_turns": flat_turns}[fam](level, rng)
                sp = [sp[0] + X_OFF, sp[1] + Y_OFF, sp[2], sp[3]]
                tiles.append(dict(name=f"goal_{fam}_L{level}_s{s}", family=fam, level=level, boxes=b, spawn=sp, route=rt, v_max=vm, info=info))
    return dict(note="ACP stage-3 goal tiles v1 (acp/goals.py); boxes only of sizes the stage-2 props already use",
                default_q=q0, stand_root_h=h0, families=FAMILIES, x_off=X_OFF, y_off=Y_OFF, tiles=tiles)


def check_slots(out, ref_props_files):
    """Every goal tile within the reference slot library (sizes and counts)."""
    import collections
    lib = collections.Counter()
    for f in ref_props_files:
        for bs in json.load(open(f)).values():
            c = collections.Counter(tuple(round(x, 3) for x in b["half"]) for b in bs)
            for k, n in c.items():
                lib[k] = max(lib[k], n)
    bad = []
    for t in out["tiles"]:
        c = collections.Counter(tuple(round(x, 3) for x in b["half"]) for b in t["boxes"])
        for k, n in c.items():
            if n > lib.get(k, 0):
                bad.append((t["name"], k, n, lib.get(k, 0)))
    return bad


if __name__ == "__main__":
    out = build(int(sys.argv[2]) if len(sys.argv) > 2 else 12)
    D = "/mnt/filesystem-g4/tracks/acp/data/clips/"
    bad = check_slots(out, [D + f for f in ("x2_loco_blocks_props.json", "x2_chains_far_props.json", "tp_stairs_x10_sh2_props.json", "gst_v1_props.json",
                                            "gs2dn_v2_props.json", "gs2lo_v1_props.json", "tp_t4x4u_y20_props.json", "pico_2026-10-29_2x2_props.json")])
    assert not bad, bad[:5]
    json.dump(out, open(sys.argv[1], "w"))
    print(len(out["tiles"]), "tiles ->", sys.argv[1], "stand root h", out["stand_root_h"])
