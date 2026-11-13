"""Route-start filter for the extended map (2026-11-12 17:30): starts closer than 0.35 m to an obstacle (the X2's arms are
0.28 m from its centre line) or on uneven ground (floor range > 3 cm within 0.27 m) are left out; tiles need one such route.
Backup ../_bak_20261112/terrain.py.pre_extfilter"""
import pathlib
import shutil

C = pathlib.Path(__file__).resolve().parents[1]
p = C / "ssr/terrain.py"
s = p.read_text()
shutil.copy(p, C.parent / "_bak_20261112/terrain.py.pre_extfilter")
rep = [
    ('''                for ro in m["routes"]:
                    wp = [(gcx + a, gcy + b) for a, b in ro["waypoints"]]''',
     '''                for ro in m["routes"]:
                    sx, sy = ro["start"][0], ro["start"][1]                 # start must fit the X2 (arms 0.28 m from its centre line) and be level
                    if ro["clearance_m"][0] < start_clear:
                        continue
                    ci, cj = int(round((sx + 3.99) / 0.02)), int(round((sy + 3.99) / 0.02))
                    patch = h[max(ci - 13, 0):ci + 14, max(cj - 13, 0):cj + 14]
                    if float(patch.max() - patch.min()) > 0.03:
                        continue
                    wp = [(gcx + a, gcy + b) for a, b in ro["waypoints"]]'''),
    ('variants=2, gap=2.0) -> dict:', 'variants=2, gap=2.0, start_clear=0.35) -> dict:'),
    ('            cand = [m for m in meta if m["level"] == lvl and m["routes"]]',
     '            cand = [m for m in meta if m["level"] == lvl and any(ro["clearance_m"][0] >= start_clear for ro in m["routes"])]'),
]
for o, n in rep:
    assert s.count(o) == 1, o
    s = s.replace(o, n)
p.write_text(s)
print("patched")
