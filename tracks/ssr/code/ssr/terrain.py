"""Training terrain (App. A.1): five types x 20 levels of 8 x 4 m tiles in one height raster, shared by every world.

Layout: type c along x (8 m each), level r along y (4 m each), 2 m flat border. A robot walks +x through its tile.
Height queries (`HeightMap`) are nearest-cell lookups on the GPU; `depth` ray-marches the same raster.
"""
from __future__ import annotations

import os
import numpy as np
import torch

# The route's own terrain categories (the paper's were slope, stairs, discrete, gap, platform). One discriminator and one gate expert each.
TYPES = ("ramp", "stairs", "beam", "platform", "flat")
SLOPE, STAIRS, BEAM, PLATFORM, FLAT = range(5)
NLEVEL, TILE_X, TILE_Y, RES, BORDER = 20, 8.0, 4.0, 0.025, 2.0
FLAT_RUN = 1.25                  # flat approach at the start of each tile (and 0.75 m flat at the end)


def _i(v):
    return int(round(v / RES))


def _slope(z, s, rng):
    t = np.tan(np.deg2rad(20.0 * s)) * rng.choice((-1, 1))
    L = (TILE_X - FLAT_RUN - 0.75) / 2
    x = np.arange(z.shape[0]) * RES - FLAT_RUN
    h = np.where(x < 0, 0.0, np.where(x < L, t * x, np.where(x < 2 * L, t * (2 * L - x), 0.0)))
    z[:] = h[:, None]


def _stairs(z, s, rng):
    run, rise = 0.40 - 0.15 * s, 0.05 + 0.15 * s
    avail = TILE_X - FLAT_RUN - 0.75 - 0.9
    n = min(6, int((avail / 2) / run))
    sign = rng.choice((-1, 1))                                      # up-then-down or down-then-up
    x, h = FLAT_RUN, 0.0
    for _ in range(n):
        h += rise * sign
        z[_i(x):] = h
        x += run
    x += max(0.9, avail - 2 * n * run + 0.9)
    for _ in range(n):
        h -= rise * sign
        z[_i(x):] = h
        x += run
    z[_i(x):] = 0.0


def _beam(z, s, rng):
    """Beams laid across the lane, like the apparatus' 4x4s and the captured 2x2 / 4x4 crossings: height and width grow with the level."""
    H, w = 0.03 + 0.09 * s, 0.05 + 0.09 * s
    nx, ny = z.shape
    xs, ys = np.meshgrid(np.arange(nx) * RES, np.arange(ny) * RES, indexing="ij")
    x = FLAT_RUN + 0.3
    while x < TILE_X - 1.1:
        yaw = np.deg2rad(rng.uniform(-25.0, 25.0))
        u = (xs - x) * np.cos(yaw) + (ys - TILE_Y / 2) * np.sin(yaw)
        z[(np.abs(u) < w / 2) & (xs < TILE_X - 0.75)] = H
        x += rng.uniform(1.1, 1.8)


def _flat(z, s, rng):
    pass


def _platform(z, s, rng):
    H, L, x = 0.05 + 0.35 * s, 1.3, FLAT_RUN + 0.3
    while x + L < TILE_X - 0.75:
        z[_i(x):_i(x + L)] = H
        x += L + 1.0


BUILD = (_slope, _stairs, _beam, _platform, _flat)


WALL_H = 1.3
NO_WALL = 2.0                     # lane half width of a tile without side walls


def _side_obstacles(t, s, rng, p_kind=(0.3, 0.3, 0.2, 0.2)):
    """Walls, rails, posts and door frames along the walking lane, like the apparatus' (door 0.97 m, rails, posts). Returns (obstacle mask, lane half
    width). They are vision and contact-check geometry only: physics stays the walkable raster, as in the ACP evaluation."""
    o = np.zeros(t.shape, bool)
    kind = rng.choice(4, p=p_kind)                         # none, lane walls, rails on posts, door frames
    cy = t.shape[1] // 2
    x0, x1 = _i(FLAT_RUN - 0.4), t.shape[0] - _i(0.4)
    if kind == 0:
        return o, NO_WALL
    half = rng.uniform(0.5 + (1 - s) * 0.4, 0.75 + (1 - s) * 0.85)
    thick = max(_i(0.08), 1)
    j0, j1 = cy - _i(half), cy + _i(half)
    if kind == 1:
        o[x0:x1, j0 - thick:j0] = True
        o[x0:x1, j1:j1 + thick] = True
    elif kind == 2:
        for x in np.arange(x0, x1, _i(1.2)):
            xi = int(x)
            o[xi:xi + thick, j0 - thick:j0] = True
            o[xi:xi + thick, j1:j1 + thick] = True
    else:
        half = rng.uniform(0.48, 0.62 + (1 - s) * 0.4)
        j0, j1 = cy - _i(half), cy + _i(half)
        for x in rng.uniform(x0 + _i(0.5), x1 - _i(0.5), 2):
            xi = int(x)
            o[xi:xi + _i(0.12), j0 - _i(0.5):j0] = True            # door jambs with a stub of wall on each side
            o[xi:xi + _i(0.12), j1:j1 + _i(0.5)] = True
    return o, float(half)


def generate(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    bn, tx, ty = _i(BORDER), _i(TILE_X), _i(TILE_Y)
    nx, ny = bn * 2 + tx * len(TYPES) + 1, bn * 2 + ty * NLEVEL + 1
    z = np.zeros((nx, ny), np.float32)
    v = np.zeros((nx, ny), np.float32)
    lane = np.full((len(TYPES), NLEVEL), NO_WALL, np.float32)
    for c, build in enumerate(BUILD):
        for r in range(NLEVEL):
            t = np.zeros((tx, ty), np.float32)
            build(t, r / (NLEVEL - 1), rng)
            kinds = (0.0, 0.4, 0.25, 0.35) if c == FLAT else (0.3, 0.3, 0.2, 0.2)
            o, h = _side_obstacles(t, r / (NLEVEL - 1), rng, kinds) if not (c == FLAT and r == 0) else (np.zeros(t.shape, bool), NO_WALL)  # the omni-command tile stays open
            z[bn + c * tx: bn + (c + 1) * tx, bn + r * ty: bn + (r + 1) * ty] = t
            v[bn + c * tx: bn + (c + 1) * tx, bn + r * ty: bn + (r + 1) * ty] = t + WALL_H * o
            lane[c, r] = h
    vision = dict(z=v, x0=-BORDER, y0=-BORDER, res=RES)
    return dict(z=z, x0=-BORDER, y0=-BORDER, res=RES, base=0.1, vision=vision, lane=lane)


EXT_DIR = "/mnt/filesystem-g4/data/terrains/extended_v1"
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


def _slope_turns(rts, h, gcx, gcy, rng, per_tile, lo=8.0, hi=22.0):
    """Slope turn stations (SSR v14, from apparatus leg 5 on 2026-11-12: a turn in place on a 16 deg side-sloped wedge): a route
    goes from its start along an existing route to the first point (>= 1.0 m along) whose 0.5 m patch is a plane sloped lo-hi deg,
    then back to its start. The waypoint law turns the robot in place there (heading error > 60 deg: vx 0, |wz| up to 1)."""
    if per_tile <= 0 or not rts:
        return []
    ii_ = np.arange(-12, 13) * 0.02
    GX, GY = np.meshgrid(ii_, ii_, indexing="ij")
    A = np.stack((GX.ravel(), GY.ravel(), np.ones(GX.size)), -1)
    out = []
    for ro in rng.permutation(len(rts)):
        if len(out) >= per_tile:
            break
        r = rts[ro]
        pts = [np.array(r["start"][:2], np.float64)] + [np.array(w, np.float64) for w in r["wp"]]
        along, hit = 0.0, None
        for j in range(len(pts) - 1):
            a, b = pts[j], pts[j + 1]
            L = float(np.linalg.norm(b - a))
            for s_ in np.arange(0.0, L, 0.1):
                P = a + (b - a) * (s_ / max(L, 1e-6))
                if along + s_ < 1.0:
                    continue
                i = int(round((P[0] - gcx + 3.99) / 0.02)); k = int(round((P[1] - gcy + 3.99) / 0.02))
                if not (12 <= i < 388 and 12 <= k < 388):
                    continue
                z = h[i - 12:i + 13, k - 12:k + 13].astype(np.float64).ravel()
                c, *_ = np.linalg.lstsq(A, z, rcond=None)
                res = z - A @ c
                ang = float(np.degrees(np.arctan(np.hypot(c[0], c[1]))))
                if lo <= ang <= hi and float(np.abs(res).max()) < 0.015:
                    hit = (j, P)
                    break
            if hit:
                break
            along += L
        if not hit:
            continue
        j, P = hit
        fwd = [tuple(map(float, w)) for w in pts[1:j + 1]] + [(float(P[0]), float(P[1]))]
        back = [tuple(map(float, w)) for w in pts[1:j + 1]][::-1] + [(float(pts[0][0]), float(pts[0][1]))]
        L = float(sum(np.linalg.norm(np.subtract(q2, q1)) for q1, q2 in zip([tuple(pts[0])] + fwd[:-1], fwd)))
        out.append(dict(tile=r["tile"], start=r["start"], wp=fwd + back, length=2.0 * L, station=3))
    return out


def add_extended(r: dict, types=("tight_spots", "log_corridors", "homes", "street_front", "stairs"), variants=2, gap=2.0, start_clear=0.35, min_along=0.0, stairs_inset=False, prop_routes=False, slope_turns=0, stations=0) -> dict:
    """Append extended_v1 tiles to a raster made by generate(): see scripts/patch_ext.py. Returns r with r["ext"] metadata."""
    import json
    z, v = r["z"], r["vision"]["z"]
    x_base = len(TYPES) * TILE_X + gap                                # local x where the extended block starts (raster x0 = -BORDER)
    ncol = len(types) * variants
    nx_new = _i(BORDER) * 2 + _i(x_base + ncol * 8.0) + 1
    if nx_new > z.shape[0]:
        pad = nx_new - z.shape[0]
        z = np.concatenate((z, np.zeros((pad, z.shape[1]), np.float32)), 0)
        v = np.concatenate((v, np.zeros((pad, v.shape[1]), np.float32)), 0)
    tiles, routes = [], []
    for ti, t in enumerate(types):
        _dir = EXT_STAIRS_INSET if (stairs_inset and t == "stairs") else EXT_DIR
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
        for lvl in range(10):
            cand = [m for m in meta if m["level"] == lvl and any(ro["clearance_m"][0] >= start_clear and ro["min_clearance_along_route_m"] >= min_along for ro in m["routes"])]
            for k in range(variants):
                if not cand:
                    continue
                m = cand[k % len(cand)]
                col, row = ti * variants + k, lvl
                cx, cy = x_base + col * 8.0 + 4.0, row * 8.0 + 4.0          # tile centre in local raster coordinates
                h, top, mask = dat["height"][m["index"]], dat["height_top"][m["index"]], dat["obstacle_mask"][m["index"]]
                floor = np.where(~mask & (top - h <= 0.30), top, h).astype(np.float32)
                vis = (floor + 1.3 * mask).astype(np.float32)
                n8 = _i(8.0)
                gx = (np.arange(n8) + 0.5) * RES - 4.0                      # SSR cell centres in tile coordinates
                ii = np.clip(np.round((gx + 3.99) / 0.02).astype(int), 0, 399)
                i0, j0 = _i(BORDER + cx - 4.0), _i(BORDER + cy - 4.0)
                z[i0:i0 + n8, j0:j0 + n8] = floor[np.ix_(ii, ii)]
                v[i0:i0 + n8, j0:j0 + n8] = vis[np.ix_(ii, ii)]
                gcx, gcy = cx, cy                                           # tile centre in world coordinates: raster index _i(BORDER + cx) is world x = cx (fixed 2026-11-12 17:20: was cx - BORDER, routes 2 m off their tiles)
                tid = len(tiles)
                tiles.append(dict(type=ti, name=t, level=lvl, tile_id=m["tile_id"], centre=(gcx, gcy)))
                pb, i_ = [], m["index"]                                     # physical tall boxes (role 2) and tall meshes as boxes, world coords
                bs = dat["box_start"]
                for bi in range(int(bs[i_]), int(bs[i_ + 1])):
                    c_, h_ = dat["box_center"][bi], dat["box_half"][bi]
                    if int(dat["box_role"][bi]) == 2 and c_[2] - abs(h_).max() < 1.6:
                        pb.append([gcx + float(c_[0]), gcy + float(c_[1]), float(c_[2])] + [float(x) for x in h_] + [float(x) for x in dat["box_quat"][bi]])
                if "mesh_start" in dat.files:
                    ms = dat["mesh_start"]
                    for mi in range(int(ms[i_]), int(ms[i_ + 1])):
                        c_, h_ = dat["mesh_aabb_center"][mi], dat["mesh_aabb_half"][mi]
                        if int(dat["mesh_role"][mi]) == 2 and c_[2] - h_[2] < 1.6:
                            pb.append([gcx + float(c_[0]), gcy + float(c_[1]), float(c_[2])] + [float(x) for x in h_] + [1.0, 0.0, 0.0, 0.0])
                tiles[-1]["phys"] = pb
                _n0 = len(routes)
                for ro in m["routes"]:
                    sx, sy = ro["start"][0], ro["start"][1]                 # start must fit the X2 (arms 0.28 m from its centre line) and be level
                    if ro["clearance_m"][0] < start_clear or ro["min_clearance_along_route_m"] < min_along:
                        continue
                    ci, cj = int(round((sx + 3.99) / 0.02)), int(round((sy + 3.99) / 0.02))
                    patch = h[max(ci - 13, 0):ci + 14, max(cj - 13, 0):cj + 14]
                    if float(patch.max() - patch.min()) > 0.03:
                        continue
                    wp = [(gcx + a, gcy + b) for a, b in ro["waypoints"]]
                    routes.append(dict(tile=tid, start=(gcx + ro["start"][0], gcy + ro["start"][1], ro["start"][2]), wp=wp, length=ro["length_m"]))
                if stations:
                    routes.extend(_stations(routes[_n0:], pb, h, gcx, gcy, _rng, stations))
                if slope_turns:
                    _base = [ro for ro in routes[_n0:] if not ro.get("station")]
                    routes.extend(_slope_turns(_base, h, gcx, gcy, _rng, slope_turns))
    r["z"], r["vision"]["z"] = z, v
    r["ext"] = dict(types=list(types), variants=variants, tiles=tiles, routes=routes)
    return r


class HeightMap:
    """GPU height lookups on a raster; optionally a separate raster for vision (tall walls kept)."""

    def __init__(self, raster: dict, device, vision: dict | None = None):
        self.dev = torch.device(device)
        self.z = torch.as_tensor(raster["z"], device=self.dev, dtype=torch.float32)
        self.x0, self.y0, self.res = raster["x0"], raster["y0"], raster["res"]
        self.nx, self.ny = self.z.shape
        v = vision or raster
        self.vz = torch.as_tensor(v["z"], device=self.dev, dtype=torch.float32)
        self.vx0, self.vy0, self.vres = v["x0"], v["y0"], v["res"]
        self.vnx, self.vny = self.vz.shape

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        i = torch.round((x - self.x0) / self.res).long().clamp(0, self.nx - 1)
        j = torch.round((y - self.y0) / self.res).long().clamp(0, self.ny - 1)
        return self.z[i, j]

    def vision(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        i = torch.round((x - self.vx0) / self.vres).long().clamp(0, self.vnx - 1)
        j = torch.round((y - self.vy0) / self.vres).long().clamp(0, self.vny - 1)
        return self.vz[i, j]

    @torch.no_grad()
    def depth(self, o: torch.Tensor, d: torch.Tensor, tmin=0.08, tmax=2.0, iters=26, step_cap=0.10) -> torch.Tensor:
        """Ray length to the raster for origins o (N, 3) and unit rays d (N, R, 3): (N, R), tmax if no hit.
        Marches with a step bounded by the height above the surface along the ray, then bisects."""
        p0 = o[:, None, :]
        t = torch.full(d.shape[:2], tmin, device=d.device)
        tp = t.clone()
        for _ in range(iters):
            p = p0 + t[..., None] * d
            above = p[..., 2] - self.vision(p[..., 0], p[..., 1])
            hit = above <= 0
            if bool(hit.all()):
                break
            tp = torch.where(hit, tp, t)
            dz = (-d[..., 2]).clamp(min=1e-3)
            t = torch.where(hit, t, (t + (above / dz).clamp(min=0.015, max=step_cap)).clamp(max=tmax + 0.2))
        p = p0 + t[..., None] * d
        hit = p[..., 2] - self.vision(p[..., 0], p[..., 1]) <= 0
        lo, hi = tp, t
        for _ in range(4):
            m = (lo + hi) / 2
            p = p0 + m[..., None] * d
            below = p[..., 2] - self.vision(p[..., 0], p[..., 1]) <= 0
            hi, lo = torch.where(below, m, hi), torch.where(below, lo, m)
        return torch.where(hit, hi, torch.full_like(hi, tmax)).clamp(tmin, tmax)


def camera_rays(n: int, fov: float, device) -> torch.Tensor:
    """Unit rays (n*n, 3) in a camera frame (x forward, y left, z up); row 0 = image top, column 0 = image left."""
    fh, fv = fov if isinstance(fov, (tuple, list)) else (fov, fov)          # horizontal, vertical field of view
    half_h, half = float(np.tan(fh / 2)), float(np.tan(fv / 2))
    c = (torch.arange(n, device=device, dtype=torch.float32) + 0.5) / n * 2 - 1
    v, u = torch.meshgrid(c, c, indexing="ij")
    r = torch.stack((torch.ones_like(u), -u * half_h, -v * half), -1).reshape(-1, 3)
    return r / r.norm(dim=-1, keepdim=True)
