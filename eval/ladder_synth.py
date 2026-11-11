"""Step 1 of the eval ladder, version 1: every obstacle skill at three levels on courses built here (held out by
construction: they are not tiles of the training export). Level 1 easy, level 2 medium, level 3 at our apparatus's own
dimensions, re-arranged (docs/EVAL.md, "Eval ladder").

    skill          level 1                    level 2                         level 3 (apparatus dimensions)
    log            5 cm beam, square          8.9 cm beam, 20 deg skew        8.9 cm 4x4 reached from a 0.51 m slot behind a wall, a post beside the far side
    wedge          8 deg hill, crest 0.10 m   12 deg hill, crest 0.15 m       15 deg pinwheel cells (1.22 m, crest 0.16 m) crossed over their 16 cm faces
    stairs_up      4 x 10 cm, run 30 cm       5 x 14 cm, run 28 cm            6 x 18.6 cm, run 22 cm, 1.1 m wide between walls
    stairs_down    the same flights, from the landing down
    corridor       1.3 m wide, 5 m            1.0 m wide, two 5 cm logs       0.87 m wide (the doorway's width), two 8.9 cm logs, 15 deg skew

Every course is a 2 cm raster on an 8 x 8 m tile plus wall boxes; routes end at a 1 s station; a log on the route has to be
stepped over. `python eval/ladder_synth.py` prints the route check of every course.
"""
from __future__ import annotations

import math

import numpy as np

from ladder import Course, wp

RES, N = 0.02, 400
XS = -3.99 + RES * np.arange(N)
GX, GY = np.meshgrid(XS, XS, indexing="ij")
Q0 = [1.0, 0.0, 0.0, 0.0]


def _beam(g, cx, cy, yaw, length, h, width=0.089):
    c, s = math.cos(yaw), math.sin(yaw)
    u = (GX - cx) * c + (GY - cy) * s          # along the beam
    v = -(GX - cx) * s + (GY - cy) * c
    g[(np.abs(u) <= length / 2) & (np.abs(v) <= width / 2)] = h
    a = [cx - c * length / 2, cy - s * length / 2]
    b = [cx + c * length / 2, cy + s * length / 2]
    return dict(name="log at (%.1f, %.1f)" % (cx, cy), a=a, b=b, slanted=False, top=h)


def _wall(x0, x1, y0, y1, h=1.2, label="wall"):
    return dict(center=[(x0 + x1) / 2, (y0 + y1) / 2, h / 2], half=[abs(x1 - x0) / 2, abs(y1 - y0) / 2, h / 2], quat=Q0, geom=-1, label=label, cls=None)


def _course(name, g, boxes, spawn, way, beams=(), zones=(), need=None):
    extra = dict(beams=[dict(name=b["name"], a=b["a"], b=b["b"]) for b in (beams if need is None else need)], beam_lines=list(beams), zones=list(zones),
                 tile="built in eval/ladder_synth.py (held out)", level=int(name[-1]))
    return Course("ladder:" + name, g, -3.99, -3.99, RES, boxes, spawn, way, name, extra)


def log(level):
    g = np.zeros((N, N))
    name = "log_L%d" % level
    if level < 3:
        b = _beam(g, -1.2, 0.0, math.radians(90 + (0 if level == 1 else 20)), 2.4, 0.05 if level == 1 else 0.089)
        return _course(name, g, [], [-2.3, 0.0, 0.0], [wp(-0.2, 0.0, hold=1.0, name="0.9 m past the log, stand 1 s")], [b])
    # level 3: the 4x4 is met from a slot 0.51 m deep (wall behind), walked into sideways-on, then a 90 degree turn to cross it
    b = _beam(g, -1.2, 0.2, math.radians(90), 1.6, 0.089)
    face = -1.2 - 0.0445 - 0.51
    boxes = [_wall(face - 0.09, face, -0.9, 1.6, 2.0), _wall(-0.80, -0.71, 0.75, 0.84, 1.0, "post")]
    x_slot = face + 0.255
    return _course(name, g, boxes, [x_slot, -1.9, 90.0],
                   [wp(x_slot, 0.2, name="in the slot, beside the 4x4"), wp(-0.55, 0.2, hold=1.0, name="over the 4x4, stand 1 s")], [b])


def wedge(level):
    g = np.zeros((N, N))
    name = "wedge_L%d" % level
    if level < 3:
        ang, crest = (8.0, 0.10) if level == 1 else (12.0, 0.15)
        run = crest / math.tan(math.radians(ang))
        up = np.clip((GX + 1.5) / run, 0, 1) * crest
        down = np.clip((1.5 - GX) / run, 0, 1) * crest
        g[:] = np.where(np.abs(GY) <= 1.5, np.minimum(up, down), 0.0)
        z = [dict(cls="wedge", rect=[-1.6, -1.6, 1.6, 1.6])]
        return _course(name, g, [], [-2.6, 0.0, 0.0], [wp(-0.0, 0.0, name="on the hill"), wp(2.4, 0.0, hold=1.0, name="past the hill, stand 1 s")], zones=z)
    # level 3: a 2 x 2 pinwheel of 1.22 m cells, each rising 15 degrees to a 0.16 m crest along its own direction and ending in a 16 cm face
    cell, crest = 1.22, 0.16
    run = crest / math.tan(math.radians(15.0))
    for (ix, iy, d) in ((0, 0, 0), (1, 0, 1), (1, 1, 2), (0, 1, 3)):
        x0, y0 = -cell + ix * cell, -cell + iy * cell
        m = (GX >= x0) & (GX < x0 + cell) & (GY >= y0) & (GY < y0 + cell)
        u = [(GX - x0), (GY - y0), (x0 + cell - GX), (y0 + cell - GY)][d]
        g[m] = (np.clip((u - (cell - run)) / run, 0, 1) * crest)[m]
    z = [dict(cls="wedge", rect=[-cell - 0.05, -cell - 0.05, cell + 0.05, cell + 0.05])]
    return _course(name, g, [], [-2.4, -0.61, 0.0],
                   [wp(-0.61, -0.61, name="on the first cell"), wp(0.61, -0.61, name="second cell"), wp(0.61, 0.61, name="third cell"),
                    wp(-0.61, 0.61, name="fourth cell"), wp(-2.4, 0.61, hold=1.0, name="off the ramps, stand 1 s")], zones=z)


STAIRS = {1: (4, 0.10, 0.30, 2.0), 2: (5, 0.14, 0.28, 2.0), 3: (6, 0.186, 0.22, 1.1)}     # steps, rise, run, width


def _stairs(level):
    n, rise, run, width = STAIRS[level]
    g = np.zeros((N, N))
    x0 = -2.0
    top = n * rise
    land0, land1 = x0 + n * run, x0 + n * run + 1.6
    h = np.zeros_like(GX)
    for k in range(n):
        h = np.where(GX >= x0 + k * run, (k + 1) * rise, h)                       # up flight
    for k in range(n):
        h = np.where(GX >= land1 + k * run, top - (k + 1) * rise, h)              # down flight
    h = np.where(GX >= land1 + n * run, 0.0, h)
    g[:] = np.where(np.abs(GY) <= width / 2, h, 0.0)
    boxes = []
    if level == 3:                                                                # walls either side, as between the apparatus' stair panels
        for sgn in (-1, 1):
            boxes.append(_wall(x0 - 0.3, land1 + n * run + 0.3, sgn * (width / 2 + 0.005), sgn * (width / 2 + 0.085), top + 1.0))
    zone = [dict(cls="stair", rect=[x0 - 0.05, -width / 2 - 0.05, land1 + n * run + 0.05, width / 2 + 0.05])]
    return g, boxes, zone, x0, land0, land1, n * run


def stairs_up(level):
    g, boxes, zone, x0, land0, land1, flight = _stairs(level)
    return _course("stairs_up_L%d" % level, g, boxes, [x0 - 1.0, 0.0, 0.0],
                   [wp(x0 - 0.3, 0.0, name="foot of the stairs"), wp((land0 + land1) / 2, 0.0, hold=1.0, name="middle of the landing, stand 1 s")], zones=zone)


def stairs_down(level):
    g, boxes, zone, x0, land0, land1, flight = _stairs(level)
    return _course("stairs_down_L%d" % level, g, boxes, [(land0 + land1) / 2, 0.0, 0.0],
                   [wp(land1 - 0.25, 0.0, name="top of the down flight"), wp(land1 + flight + 0.9, 0.0, hold=1.0, name="on the floor past the stairs, stand 1 s")], zones=zone)


def corridor(level):
    g = np.zeros((N, N))
    half = {1: 0.65, 2: 0.50, 3: 0.435}[level]
    boxes = [_wall(-2.5, 2.5, sgn * half if sgn > 0 else -half - 0.09, sgn * half + 0.09 if sgn > 0 else -half, 1.4) for sgn in (-1, 1)]
    beams = []
    if level > 1:
        for cx, skew in ((-0.8, 15.0), (0.9, -15.0)):
            beams.append(_beam(g, cx, 0.0, math.radians(90 + (skew if level == 3 else 0.0)), 2 * half - 0.02, 0.05 if level == 2 else 0.089))
    return _course("corridor_L%d" % level, g, boxes, [-3.2, 0.0, 0.0],
                   [wp(-2.0, 0.0, name="corridor entrance"), wp(0.0, 0.0, name="middle"), wp(2.0, 0.0, name="corridor exit"), wp(3.1, 0.0, hold=1.0, name="past the corridor, stand 1 s")], beams)


SKILLS = dict(log=log, wedge=wedge, stairs_up=stairs_up, stairs_down=stairs_down, corridor=corridor)
TESTS = {"%s_L%d" % (k, lv): (lambda k=k, lv=lv: SKILLS[k](lv)) for k in SKILLS for lv in (1, 2, 3)}

if __name__ == "__main__":
    import check_routes as C
    for name, make in TESTS.items():
        c = make()
        segs, blocked = C.check(c)
        P = c.path()
        print("%-16s %.1f m, %d waypoints, floor %.2f..%.2f m, %d boxes, tightest wall %.2f m, largest floor step on the route %.3f m: %s" % (
            name, float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()), len(c.waypoints), c.grid.min(), c.grid.max(), len(c.boxes),
            min(r["clearance"] for r in segs), max(r["max_step"] for r in segs), "route check PASS" if not blocked else "BLOCKED " + str([b[2] for b in blocked])))
