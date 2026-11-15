"""v13 (SSR 2026-11-12 21:00): stairs-inset tiles, extra prop routes, more ext types, wall stations. Backups *.pre_v13. Run from code/."""
import pathlib, shutil
R = pathlib.Path(__file__).resolve().parents[1]


def sub(path, old, new, count=1):
    p = R / path
    s = p.read_text()
    assert s.count(old) == count, (path, old[:60], s.count(old))
    p.write_text(s.replace(old, new))


for f in ("ssr/terrain.py", "ssr/env.py", "ssr/train.py"):
    b = R / (f + ".pre_v13")
    if not b.exists():
        shutil.copy(R / f, b)

# ---------------- terrain.py
sub("ssr/terrain.py", 'EXT_DIR = "/mnt/filesystem-g4/data/terrains/extended_v1"\n',
'''EXT_DIR = "/mnt/filesystem-g4/data/terrains/extended_v1"
EXT_STAIRS_INSET = "/mnt/filesystem-g4/data/terrains/extended_v1_2_stairs_inset"   # v13: stair foot 2.5-3 m inside the tile, 700 routes
EXT_PROP_ROUTES = "/mnt/filesystem-g4/data/terrains/extended_v1_1_prop_routes"     # v13: 1,116 approach-shape routes on prop tiles


def _box_near(p, B):
    """2D distance from p to each oriented box (world [x,y,z,hx,hy,hz,qw,qx,qy,qz]) and the closest point on it."""
    c, h = B[:, :2], np.abs(B[:, 3:5])
    yaw = 2.0 * np.arctan2(B[:, 9], B[:, 6])
    cs, sn = np.cos(yaw), np.sin(yaw)
    d = p[None] - c
    lx, ly = cs * d[:, 0] + sn * d[:, 1], -sn * d[:, 0] + cs * d[:, 1]
    qx, qy = np.clip(lx, -h[:, 0], h[:, 0]), np.clip(ly, -h[:, 1], h[:, 1])
    q = c + np.stack((cs * qx - sn * qy, sn * qx + cs * qy), -1)
    return np.linalg.norm(p[None] - q, axis=1), q


def _stations(rts, pb, h, gcx, gcy, rng, per_tile):
    """Wall stations (SSR v13, from the apparatus legs 2026-11-12): the buttons sit 0.19-0.33 m from a wall face, so the robot must
    stop with its pelvis about 0.35-0.40 m from a wall, turn in place there and leave along it. For a route whose target is
    0.35-1.5 m from a tall wall box: an approach route = the route + a last segment straight at the wall to a point 0.15-0.22 m
    from the face (the hold radius 0.20 then stops the pelvis 0.35-0.42 m from it), and a departure route = start at that point
    facing the wall, back along the route to its start. Checks: other boxes >= 0.30 m from the new segment, level floor."""
    if not pb or per_tile <= 0 or not rts:
        return []
    B = np.array(pb, np.float64)
    wall = (B[:, 2] - np.abs(B[:, 5]) < 0.30) & (B[:, 2] + np.abs(B[:, 5]) > 0.80)
    if not wall.any():
        return []

    def hz(x, y):
        i = int(np.clip(round((x - gcx + 3.99) / 0.02), 0, 399)); j = int(np.clip(round((y - gcy + 3.99) / 0.02), 0, 399))
        return h[max(i - 13, 0):i + 14, max(j - 13, 0):j + 14]
    out, used = [], []
    for ro in rng.permutation(len(rts)):
        r = rts[ro]
        if len(out) >= 2 * per_tile:
            break
        T = np.array(r["wp"][-1], np.float64)
        if any(np.linalg.norm(T - u) < 1.0 for u in used):
            continue
        dist, q = _box_near(T, B)
        dist = np.where(wall, dist, 9e9)
        k = int(np.argmin(dist))
        if not 0.35 <= dist[k] <= 1.5:
            continue
        n = (T - q[k]) / dist[k]
        S = q[k] + n * rng.uniform(0.15, 0.22)
        ok = True
        for s_ in np.linspace(0.0, 1.0, 12):
            dP, _ = _box_near(T + s_ * (S - T), B)
            dP[k] = 9e9
            if dP.min() < 0.30:
                ok = False
                break
        if not ok:
            continue
        pS, pT = hz(*S), hz(*T)
        if float(pS.max() - pS.min()) > 0.03 or abs(float(pS.mean()) - float(pT.mean())) > 0.03:
            continue
        L = float(np.linalg.norm(S - T))
        used.append(T)
        yaw_in = float(np.arctan2(-n[1], -n[0]))
        out.append(dict(tile=r["tile"], start=r["start"], wp=list(r["wp"]) + [(float(S[0]), float(S[1]))], length=r["length"] + L, station=1))
        back = [tuple(w) for w in r["wp"][:-1]][::-1] + [(float(r["start"][0]), float(r["start"][1]))]
        out.append(dict(tile=r["tile"], start=(float(S[0]), float(S[1]), yaw_in), wp=back, length=r["length"] + L, station=2))
    return out
''')
sub("ssr/terrain.py", 'variants=2, gap=2.0, start_clear=0.35, min_along=0.0) -> dict:',
    'variants=2, gap=2.0, start_clear=0.35, min_along=0.0, stairs_inset=False, prop_routes=False, stations=0) -> dict:')
sub("ssr/terrain.py", '''        dat = np.load(f"{EXT_DIR}/{t}.npz")
        meta = json.load(open(f"{EXT_DIR}/{t}.json"))["tiles"]
''', '''        _dir = EXT_STAIRS_INSET if (stairs_inset and t == "stairs") else EXT_DIR
        dat = np.load(f"{_dir}/{t}.npz")
        meta = json.load(open(f"{_dir}/{t}.json"))["tiles"]
        for m_ in meta:
            m_["level"] = int(m_["level"]); m_["index"] = int(m_["index"])
        _xf = f"{EXT_PROP_ROUTES}/{t}_routes_extra.json"
        if prop_routes and t != "stairs" and os.path.exists(_xf):
            _x = json.load(open(_xf))["tiles"]
            for m_ in meta:
                m_["routes"] = list(m_["routes"]) + list(_x.get(m_["tile_id"], {}).get("routes", []))
        _rng = np.random.default_rng(1000 + ti)
''')
sub("ssr/terrain.py", '''                for ro in m["routes"]:
                    sx, sy = ro["start"][0], ro["start"][1]''', '''                _n0 = len(routes)
                for ro in m["routes"]:
                    sx, sy = ro["start"][0], ro["start"][1]''')
sub("ssr/terrain.py", '''                    routes.append(dict(tile=tid, start=(gcx + ro["start"][0], gcy + ro["start"][1], ro["start"][2]), wp=wp, length=ro["length_m"]))
''', '''                    routes.append(dict(tile=tid, start=(gcx + ro["start"][0], gcy + ro["start"][1], ro["start"][2]), wp=wp, length=ro["length_m"]))
                if stations:
                    routes.extend(_stations(routes[_n0:], pb, h, gcx, gcy, _rng, stations))
''')
s = (R / "ssr/terrain.py").read_text()
if "\nimport os\n" not in s:
    (R / "ssr/terrain.py").write_text(s.replace("import numpy as np", "import os\nimport numpy as np", 1))

# ---------------- env.py
sub("ssr/env.py", '''        self.ext_beam = torch.tensor([nm in ("tight_spots", "homes") for nm in e["types"]], device=dev)''',
'''        self.ext_beam = torch.tensor([nm in ("tight_spots", "homes", "single_beams", "beam_fields") for nm in e["types"]], device=dev)
        self.ext_slope = torch.tensor([nm == "wedges" for nm in e["types"]], device=dev)          # v13: wedges -> ramp discriminator
        self.ext_plat = torch.tensor([nm == "blocks" for nm in e["types"]], device=dev)           # v13: blocks -> platform discriminator''')
sub("ssr/env.py", '''torch.where(self.ext_beam[self.ext_type[eid]], torch.full_like(self.type[eid], T.BEAM), torch.full_like(self.type[eid], T.FLAT)))''',
'''torch.where(self.ext_beam[self.ext_type[eid]], torch.full_like(self.type[eid], T.BEAM), torch.where(self.ext_slope[self.ext_type[eid]], torch.full_like(self.type[eid], T.SLOPE), torch.where(self.ext_plat[self.ext_type[eid]], torch.full_like(self.type[eid], T.PLATFORM), torch.full_like(self.type[eid], T.FLAT)))))''')
sub("ssr/env.py", '''            segc[i, len(pts) - 1:] = segc[i, len(pts) - 2]
''', '''            segc[i, len(pts) - 1:] = segc[i, len(pts) - 2]
            if r.get("station") == 1:                                  # v13 wall station: the last segment goes at the wall on purpose (no sidestep)
                segc[i, len(pts) - 2:] = 9.9
            elif r.get("station") == 2:                                # departure from a station: first segment starts at the wall
                segc[i, 0] = 9.9
''')

# ---------------- train.py
sub("ssr/train.py", 'variants=a.ext_variants, min_along=a.ext_min_clear)',
    'variants=a.ext_variants, min_along=a.ext_min_clear, stairs_inset=a.ext_stairs_inset, prop_routes=a.ext_prop_routes, stations=a.ext_stations)')
sub("ssr/train.py", '''    ap.add_argument("--ext_variants", type=int, default=2)
''', '''    ap.add_argument("--ext_variants", type=int, default=2)
    ap.add_argument("--ext_stairs_inset", action="store_true", help="v13: stair tiles from extended_v1_2_stairs_inset (foot inside the tile, 700 routes)")
    ap.add_argument("--ext_prop_routes", action="store_true", help="v13: add extended_v1_1_prop_routes (approach shapes on beams, beam fields, blocks, logs, wedges)")
    ap.add_argument("--ext_stations", type=int, default=0, help="v13: wall stations per tile (approach to 0.15-0.22 m of a wall face + departure); 0 = off")
''')
print("patched")
