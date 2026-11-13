"""SSR 2026-11-12 18:50: extended routes must fit the X2 along their whole length: --ext_min_clear M drops routes whose
min_clearance_along_route_m < M (X2 limb centres reach 0.30-0.39 m from the centre line; overlap ends the episode)."""
import shutil
C = "/mnt/filesystem-g4/tracks/ssr/code/ssr/"
for f in ("terrain.py", "train.py"):
    shutil.copy(C + f, C + f + ".pre_extclear")
t = open(C + "terrain.py").read()
a = 'variants=2, gap=2.0, start_clear=0.35) -> dict:'
assert t.count(a) == 1; t = t.replace(a, 'variants=2, gap=2.0, start_clear=0.35, min_along=0.0) -> dict:')
a = 'any(ro["clearance_m"][0] >= start_clear for ro in m["routes"])]'
assert t.count(a) == 1; t = t.replace(a, 'any(ro["clearance_m"][0] >= start_clear and ro["min_clearance_along_route_m"] >= min_along for ro in m["routes"])]')
a = '                    if ro["clearance_m"][0] < start_clear:\n'
assert t.count(a) == 1; t = t.replace(a, '                    if ro["clearance_m"][0] < start_clear or ro["min_clearance_along_route_m"] < min_along:\n')
open(C + "terrain.py", "w").write(t)
r = open(C + "train.py").read()
a = 'variants=a.ext_variants)'
assert r.count(a) == 1; r = r.replace(a, 'variants=a.ext_variants, min_along=a.ext_min_clear)')
a = '    ap.add_argument("--ext_variants", type=int, default=2)\n'
assert r.count(a) == 1
r = r.replace(a, a + '    ap.add_argument("--ext_min_clear", type=float, default=0.0, help="extended map: drop routes whose least obstacle clearance along the route is under this (m); 0.39 = the X2 fits (SSR 2026-11-12)")\n')
open(C + "train.py", "w").write(r)
print("patched")
