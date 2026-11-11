"""Desktop-only: collect both apparatuses into eval/data/apparatus/{ours,nist} (then rsync to /mnt/filesystem-g4/data/apparatus).

Ours: copies of ssr-x2/assets/route (identical to loco-manip-apparatus/build/hmap). NIST: run collect_nist.py first.
Writes route.json for each (metres, apparatus frame) and nothing else; READMEs are written by hand next to them.
    python eval/collect_apparatus.py
"""
import json, os, shutil
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'data', 'apparatus')
IN = 0.0254
SRC = 'C:/Users/PC/projects/ssr-x2/assets/route/'
for f in ('cad_20mm.npz', 'cad_20mm_walk.npz', 'cad_walls.json'):
    shutil.copy2(SRC + f, os.path.join(OUT, 'ours', f))

# ---- ours: ssr-x2/ssr/route.py MOBILITY == loco-manip-apparatus sim/x2_route.py ROUTES['mobility'] (inches) -------
# name, x in, y in, heading deg, hold s (station), section
OURS = [
    ("to the door", 84, 47, 0, 0.0, "door"),
    ("through the door", 107, 46, 0, 0.0, "door"),
    ("clear of the door wall", 109, 27, -70, 0.0, "door"),
    ("over the front 4x4", 130, 27, 0, 0.0, "4x4s"),
    ("button 1", 130, 13, -90, 1.0, "4x4s"),
    ("square to the front 4x4", 130, 27, 90, 0.0, "4x4s"),
    ("back over the front 4x4", 109, 27, 180, 0.0, "4x4s"),
    ("door side", 110.24, 63.0, 90, 0.0, "4x4s"),
    ("over the back 4x4", 133.07, 70.08, 0, 0.0, "4x4s"),
    ("button 2", 129, 79, 90, 1.0, "4x4s"),
    ("ramps, back side", 150, 72, 0, 0.0, "ramps"),
    ("over the back hill", 196, 72, 0, 0.0, "ramps"),
    ("button 3", 226, 83, 90, 1.0, "ramps"),
    ("ramps, diagonal", 196, 46, -130, 0.0, "ramps"),
    ("button 4", 162, 14, -90, 1.0, "ramps"),
    ("ramps, front side", 214, 24, 0, 0.0, "ramps"),
    ("around the wall", 278, 54, 0, 0.0, "wall"),
    ("past the wall", 312, 54, 0, 0.0, "wall"),
    ("foot of the stairs", 326, 21, 0, 0.0, "stairs"),
    ("up the stairs", 380, 21, 0, 0.0, "stairs"),
    ("landing, turn left", 404, 34, 90, 0.0, "landing"),
    ("button 5", 410, 83, 90, 1.0, "landing"),
]
ours = dict(
    name="ours_mobility", apparatus="ours", units="m", frame="apparatus frame of the CAD (x along the lane from the start end, y across, z up, deck top z = 0)",
    source="loco-manip-apparatus sim/x2_route.py ROUTES['mobility'] (inches), copied in ssr-x2/ssr/route.py; holds 1 s as in the 2026-11-02 ACP 50-run eval. Changed here: waypoint 1 moved from x = 104 in to 107 in, and waypoints 2 and 6 from x = 107 / 108 in to 109 in, the middle of the 0.51 m slot between the door wall and the front 4x4 (0.20 / 0.23 m from the wall before, 0.25 m now); waypoints 7 and 8 (either side of the back 4x4) from (108, 60) and (129, 64) in to (2.80, 1.60) and (3.38, 1.78) m, so the straight leg between them crosses the beam 0.27 m from its end instead of 0.14 m, and stays 0.25 m from the sloping board west of it (user, 2026-11-11: it has to step over both 4x4s, not round their ends)",
    raster="cad_20mm_walk.npz", raster_full="cad_20mm.npz", walls="cad_walls.json", floor=0.0,
    spawn=[30 * IN, 30 * IN, 0.0], sections=["door", "4x4s", "ramps", "wall", "stairs", "landing"],
    # centre lines of the two 4x4 stepovers, fitted to the raster (8.9 cm tall, 9 cm wide); a trial must cross both
    beams=[dict(name="front 4x4", a=[2.985, 0.257], b=[3.098, 1.041]), dict(name="back 4x4", a=[3.097, 1.416], b=[2.984, 2.183])],
    # where a side face belongs to which obstacle class (x0, y0, x1, y1): the crossing ramps and the stair flight; the 4x4s
    # are the beams above (within 0.12 m of a centre line)
    zones=[dict(cls="wedge", rect=[3.44, 0.00, 6.32, 2.35]), dict(cls="stair", rect=[8.55, -0.05, 9.80, 1.15])],
    # the door: hinge at (2.374, 1.651), leaf 0.92 m, standing open at 90 deg along -x (toward the start), in the wall boxes
    door=dict(name="door", plane_x=2.45, span_y=[0.30, 2.00], hinge=[2.374, 1.651], open_deg=90.0,
              leaf=[[2.374, 1.651], [1.452, 1.63]], leaf_in_walls=True),
    waypoints=[dict(name=n, x=round(x * IN, 4), y=round(y * IN, 4), heading_deg=h, hold_s=hold, section=s) for n, x, y, h, hold, s in OURS])
json.dump(ours, open(os.path.join(OUT, 'ours', 'route.json'), 'w'), indent=1)

# ---- NIST: loco-manip-apparatus sim/pipeline/routepass.py NIST_ROUTE (metres, NIST CAD frame) -----------------------
NIST = (('start', 11.90, 4.21, None), ('1 keypad', 10.08, 4.76, 135), ('2 door: open', 10.17, 4.23, 135),
        ('', 9.70, 4.30, None), ('2 door: close', 9.27, 4.13, 45), ('', 8.80, 3.80, None),
        ('', 8.80, 3.60, None), ('3 bulbs', 8.88, 3.45, -12), ('', 8.35, 3.75, None), ('', 7.72, 3.95, None), ('', 7.72, 4.35, None),
        ('5 buttons', 7.775, 4.625, 0), ('', 7.70, 4.30, None), ('', 7.72, 3.90, None), ('', 7.00, 3.75, None),
        ('4 valves', 6.72, 3.42, -128), ('', 6.75, 3.95, None), ('', 6.45, 4.80, None), ('', 5.85, 4.75, None),
        ('C counter R', 5.40, 4.05, -100), ('', 5.00, 4.30, None), ('C counter L', 4.64, 4.05, -120),
        ('', 4.20, 4.30, None), ('', 3.55, 4.05, None), ('crusher', 3.03, 3.52, 180), ('', 3.30, 4.05, None),
        ('', 4.25, 4.10, None), ('', 4.55, 4.60, None), ('', 4.40, 4.965, None), ('stair foot', 3.80, 4.965, None),
        ('stair top', 2.45, 4.965, None), ('landing', 1.80, 4.965, None))


# The eval's NIST route = NIST_ROUTE plus what the user asked for on 2026-11-11: the pocket behind the upper 4x4 (the
# mobility trial's first button: "walk through the open doorway and then turn right to press the first button, turn
# around and proceed to press the second ... step over two lengths of 4x4"), both 4x4s crossed square and away from
# their ends, and the landing walked to its far end ("turn left, walk to and press the button").
# name, x, y, facing deg or None, hold s, section
NIST_EVAL = [
    ("1 keypad", 10.08, 4.76, 135, 1.0, "door"), ("2 door: open", 10.17, 4.23, 135, 1.0, "door"),
    ("through the door", 9.70, 4.30, None, 0.0, "door"), ("2 door: close", 9.27, 4.13, 45, 1.0, "door"),
    ("square to the upper 4x4", 9.10, 4.35, None, 0.0, "4x4s"),
    ("mobility button 1 (pocket behind the upper 4x4)", 8.88, 5.00, None, 1.0, "4x4s"),
    ("back over the upper 4x4", 9.10, 4.35, None, 0.0, "4x4s"),
    ("square to the lower 4x4", 9.246, 4.050, None, 0.0, "4x4s"),
    ("over the lower 4x4", 8.750, 3.584, None, 0.0, "4x4s"),
    ("3 bulbs", 8.88, 3.45, -12, 1.0, "4x4s"), ("to the ramps", 8.35, 3.75, None, 0.0, "4x4s"),
]
for i, (n, x, y, f) in enumerate(NIST):
    if i >= 9:                                           # from the first ramp vertex (7.72, 3.95) on: NIST_ROUTE as it is
        sec = "ramps" if x > 6.2 else "counter" if i < 28 else "stairs" if x > 2.2 else "landing"
        NIST_EVAL.append((n or "via %d" % i, x, y, f, 1.0 if f is not None else 0.0, sec))
NIST_EVAL[-1] = ("landing, turn left", 1.80, 4.965, None, 0.0, "landing")
NIST_EVAL.append(("landing end (mobility button 5)", 1.84, 3.35, -90, 1.0, "landing"))
wps = [dict(name=n, x=x, y=y, heading_deg=f, hold_s=h, section=s) for n, x, y, f, h, s in NIST_EVAL]
nist = dict(
    name="nist_stations_mobility", apparatus="nist", units="m", frame="frame of the organisers' CAD export (LocoManipulationTestbed.dae in metres; the lane runs along -x from the start platform, z up, deck top z = 0.14)",
    source="loco-manip-apparatus sim/pipeline/routepass.py NIST_ROUTE (2026-10-28, the manipulation-station route), plus on 2026-11-11 (user): the pocket behind the upper 4x4, both 4x4s crossed square, and the landing walked to its far end; station holds 1 s",
    raster="nist_20mm_walk.npz", raster_full="nist_20mm.npz", walls="nist_walls.json", floor=0.14,
    spawn=[11.90, 4.21, 180.0], sections=["door", "4x4s", "ramps", "counter", "stairs", "landing"],
    beams=[dict(name="upper 4x4", a=[8.711, 4.333], b=[9.616, 5.309]), dict(name="lower 4x4", a=[8.690, 4.145], b=[9.622, 3.154])],
    zones=[dict(cls="wedge", rect=[6.05, 3.05, 8.50, 5.42]), dict(cls="stair", rect=[2.40, 4.45, 3.70, 5.42])],
    # the door: hinge at (9.756, 3.765), leaf 0.94 m, open at 94.5 deg (joint -1.65 rad) along +x (toward the start)
    door=dict(name="door", plane_x=9.745, span_y=[3.30, 5.20], hinge=[9.756, 3.765], open_deg=94.5,
              leaf=[[9.756, 3.765], [10.693, 3.69]], leaf_in_walls=True),
    waypoints=wps)
# the open door leaf as one box (its geoms at the open pose span x 9.755..10.693, y 3.600..3.775, z 0.252..2.235, handle
# included). collect_nist.py leaves it out because the leaf is a jointed body.
wj = os.path.join(OUT, 'nist', 'nist_walls.json')
W = json.load(open(wj))
W['boxes'] = [b for b in W['boxes'] if b.get('type') != 'door_leaf']
W['boxes'].append(dict(center=[10.224, 3.6875, 1.2435], half=[0.469, 0.0875, 0.9915], quat=[1.0, 0.0, 0.0, 0.0], geom=-1, type='door_leaf'))
json.dump(W, open(wj, 'w'), indent=0)
json.dump(nist, open(os.path.join(OUT, 'nist', 'route.json'), 'w'), indent=1)
print('ours', len(ours['waypoints']), 'waypoints; nist', len(wps))
