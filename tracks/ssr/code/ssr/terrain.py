"""Training terrain (App. A.1): five types x 20 levels of 8 x 4 m tiles in one height raster, shared by every world.

Layout: type c along x (8 m each), level r along y (4 m each), 2 m flat border. A robot walks +x through its tile.
Height queries (`HeightMap`) are nearest-cell lookups on the GPU; `depth` ray-marches the same raster.
"""
from __future__ import annotations

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
