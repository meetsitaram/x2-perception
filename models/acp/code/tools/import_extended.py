"""Extended-map import v2 for ACP stage 3 (ACP agent, 2026-11-12; v2b 22:40 UTC). The user's extended training map
(/mnt/filesystem-g4/data/terrains/extended_v1) goes in as it is: each tile's floor raster plus every box that is not in the
floor raster (walls, posts, rails, chairs, tables, raised or slanted 4x4s, door leaves) is written into ONE height raster,
so no new prop slots are needed. Door headers are left out (a raster cannot show the opening under them, so the doorway
stays open). Meshes not in the raster (stones, bushes, cylinders) go in by their axis-aligned box top. 2 cm -> 4 cm by max.
v2b additions (user's new assets, 2026-11-12):
- stairs: the tiles of extended_v1_2_stairs_inset (stair foot 2.5-3 m inside, 700 routes with `kind`) replace the extended_v1
  stair tiles of the same id; per level one straight, one L, one U (first variant of each with routes) and the extended_v1
  amphitheatre (not in the inset set), so 4 stair rows.
- single_beams, beam_fields, log_corridors, wedges, blocks: the tile's own routes plus extended_v1_1_prop_routes (approach shapes).
- every route keeps its `kind` (base routes: "base"); per tile at most KIND_CAP routes of each kind (seeded draw), so kinds are
  sampled about evenly. Routes flagged by the generator's checker defect (start above 1 m read as blocked; `check_note`) are kept.
Layout: row r (one per type and pick), tile at level L centred at x = X0 + 8 L + 4, y = 8 r + 4. The ramp raster is pasted in.
Routes: each tile route -> goal-env waypoints [x, y, z, heading, station]; heading = travel direction of the leg into the
waypoint (the last keeps the previous); only the final waypoint is a station (1 s stand). Spawn = route start facing leg 1.
    python tools/import_extended.py [KIND_CAP=4]
Writes data/clips/hmap_ext_v2_40mm.npz and data/goals/goal_tiles_v2.json (= the v1 tiles + the extended tiles)."""
import json, math, sys, collections, numpy as np
TR = "/mnt/filesystem-g4/data/terrains"; E = f"{TR}/extended_v1"; D = "/mnt/filesystem-g4/tracks/acp/data"
INSET = f"{TR}/extended_v1_2_stairs_inset"; PROP = f"{TR}/extended_v1_1_prop_routes"
TYPES = ["single_beams", "beam_fields", "blocks", "wedges", "tight_spots", "log_corridors", "homes", "street_front", "stairs"]
PROP_TYPES = ["single_beams", "beam_fields", "log_corridors", "wedges", "blocks"]
X0, T, RES = 40.0, 8.0, 0.04
KIND_CAP = int(sys.argv[1]) if len(sys.argv) > 1 else 4
rng = np.random.default_rng(20261112)
# rows: (type, source npz, source json meta, picker) ; picker(tile) -> True if this row takes the tile
def kind_is(k): return lambda t: t.get("variant_kind") == k
ROWS = [(t, f"{E}/{t}", None) for t in TYPES if t != "stairs"]
ROWS += [("stairs", f"{INSET}/stairs", kind_is(k)) for k in ("straight", "L", "U")] + [("stairs", f"{E}/stairs", kind_is("amphitheatre"))]
ramp = np.load(f"{D}/clips/tp_ramp_hmap_x30_40mm.npz")
rx0, ry0 = float(ramp["x0"]), float(ramp["y0"])
gx0, gy0 = min(rx0, X0), min(ry0, 0.0)
gx1, gy1 = X0 + 10 * T, len(ROWS) * T
nx, ny = int(round((gx1 - gx0) / RES)) + 1, int(round((gy1 - gy0) / RES)) + 1
G = np.zeros((nx, ny), np.float32)
ri, rj = int(round((rx0 - gx0) / RES)), int(round((ry0 - gy0) / RES))
G[ri:ri + ramp["grid"].shape[0], rj:rj + ramp["grid"].shape[1]] = np.maximum(np.nan_to_num(ramp["grid"]), 0)
def quat_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
extra = {t: json.load(open(f"{PROP}/{t}_routes_extra.json"))["tiles"] for t in PROP_TYPES}
v1 = json.load(open(f"{D}/goals/goal_tiles_v1.json"))
tiles, report = [], {}
xs = -3.99 + 0.02 * np.arange(400)
XX, YY = np.meshgrid(xs, xs, indexing="ij")
for row, (typ, src, pick) in enumerate(ROWS):
    z = np.load(f"{src}.npz"); meta = json.load(open(f"{src}.json")); lab = meta["labels"]
    rep = report.setdefault(typ, dict(rows=0, tiles=0, routes_available=0, routes=0, kinds=collections.Counter(), boxes_rasterised=0,
                                      headers_left_out=0, meshes_rasterised=0, max_top_m=0.0, sources=[]))
    rep["rows"] += 1; rep["sources"].append(src.split("/")[-2] + (f":{pick.__name__}" if pick else ""))
    used = set()
    for t in sorted(meta["tiles"], key=lambda t: (int(t["level"]), int(t.get("variant", 0) or 0))):
        L = int(t["level"])
        if L in used or not t["routes"] or (pick and not pick(t)): continue     # per level the first matching variant with routes
        used.add(L)
        i = t["index"]; h = z["height"][i].copy()
        b0, b1 = z["box_start"][i], z["box_start"][i + 1]
        for k in range(b0, b1):
            if z["box_in_raster"][k]: continue
            if lab[z["box_label"][k]] == "door_header": rep["headers_left_out"] += 1; continue
            c, hh, R = z["box_center"][k], z["box_half"][k], quat_R(z["box_quat"][k])
            corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * hh @ R.T + c
            top = float(corners[:, 2].max())
            dx, dy = XX - c[0], YY - c[1]
            u = R[0, 0] * dx + R[1, 0] * dy; v = R[0, 1] * dx + R[1, 1] * dy      # footprint in the box's own x / y (upright or near)
            ins = (np.abs(u) <= hh[0] + 0.01) & (np.abs(v) <= hh[1] + 0.01)
            if not z["box_upright"][k]:                                           # tilted: axis-aligned footprint of its corners
                ins = (XX >= corners[:, 0].min()) & (XX <= corners[:, 0].max()) & (YY >= corners[:, 1].min()) & (YY <= corners[:, 1].max())
            h[ins] = np.maximum(h[ins], top); rep["boxes_rasterised"] += 1
        m0, m1 = z["mesh_start"][i], z["mesh_start"][i + 1]
        for k in range(m0, m1):
            if z["mesh_in_raster"][k]: continue
            c, hh = z["mesh_aabb_center"][k], z["mesh_aabb_half"][k]
            ins = (np.abs(XX - c[0]) <= hh[0]) & (np.abs(YY - c[1]) <= hh[1])
            h[ins] = np.maximum(h[ins], c[2] + hh[2]); rep["meshes_rasterised"] += 1
        h4 = h.reshape(200, 2, 200, 2).max(axis=(1, 3))
        cx, cy = X0 + T * L + T / 2, T * row + T / 2
        ii, jj = int(round((cx - 4 + 0.02 - gx0) / RES)), int(round((cy - 4 + 0.02 - gy0) / RES))
        G[ii:ii + 200, jj:jj + 200] = np.maximum(G[ii:ii + 200, jj:jj + 200], h4)
        rep["max_top_m"] = max(rep["max_top_m"], float(h4.max())); rep["tiles"] += 1
        routes = [dict(r, kind=r.get("kind", "base")) for r in t["routes"]]
        if typ in extra and t["tile_id"] in extra[typ]:
            routes += [dict(r, kind="prop_" + r["kind"]) for r in extra[typ][t["tile_id"]]["routes"]]
        rep["routes_available"] += len(routes)
        byk = collections.defaultdict(list)
        for r_i, r in enumerate(routes): byk[r["kind"]].append(r_i)
        keep = sorted(j for k, ids in byk.items() for j in (ids if len(ids) <= KIND_CAP else rng.choice(ids, KIND_CAP, replace=False).tolist()))
        for r_i in keep:
            r = routes[r_i]
            pts = [r["start"][:2]] + r["waypoints"]
            route = []
            for q in range(1, len(pts)):
                hd = math.atan2(pts[q][1] - pts[q - 1][1], pts[q][0] - pts[q - 1][0])
                route.append([pts[q][0] + cx, pts[q][1] + cy, 0.0, hd, 1.0 if q == len(pts) - 1 else 0.0])
            h0 = math.atan2(pts[1][1] - pts[0][1], pts[1][0] - pts[0][0])
            tiles.append(dict(name=f"ext_{t['tile_id']}_r{r_i}", family=f"ext_{typ}", level=L, boxes=[], spawn=[pts[0][0] + cx, pts[0][1] + cy, 0.0, h0],
                              route=route, v_max=0.8, info=dict(source=src.split("/")[-2], tile_id=t["tile_id"], variant_kind=t.get("variant_kind"),
                                                                 kind=r["kind"], check_note=r.get("check_note"),
                                                                 min_clearance_m=r.get("min_clearance_along_route_m"))))
            rep["routes"] += 1; rep["kinds"][r["kind"]] += 1
# waypoint and spawn heights from the raster itself (what the robot stands on)
for t in tiles:
    for w in t["route"] + [t["spawn"]]:
        w[2] = float(G[int(round((w[0] - gx0) / RES)), int(round((w[1] - gy0) / RES))])
out = dict(v1)
out["tiles"] = v1["tiles"] + tiles
out["families"] = list(v1["families"]) + [f"ext_{t}" for t in TYPES]
out["note"] = v1["note"] + (" | v2b (2026-11-12): + extended_v1 tiles as one height raster (tools/import_extended.py), families ext_<type>;"
                            " stairs from extended_v1_2_stairs_inset (straight, L, U) + extended_v1 amphitheatre; prop routes from"
                            f" extended_v1_1_prop_routes; at most {KIND_CAP} routes per kind per tile")
json.dump(out, open(f"{D}/goals/goal_tiles_v2.json", "w"))
np.savez_compressed(f"{D}/clips/hmap_ext_v2_40mm.npz", grid=G, x0=np.float64(gx0), y0=np.float64(gy0), res=np.float64(RES))
print(f"raster {G.shape} = {G.size / 1e6:.2f} M cells, x {gx0:.2f}..{gx0 + (nx - 1) * RES:.2f}, y {gy0:.2f}..{gy0 + (ny - 1) * RES:.2f}, max {G.max():.2f} m")
for k, v in report.items(): print(k, {a: (dict(b) if isinstance(b, collections.Counter) else b) for a, b in v.items()})
print("goal tiles v1", len(v1["tiles"]), "+ extended", len(tiles))
