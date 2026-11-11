"""An apparatus for the common eval: a 2 cm walk raster (physics floor), the tall geometry as boxes, and a route.

Data folder (desktop: eval/data/apparatus, nodes: /mnt/filesystem-g4/data/apparatus; override with PA_APPARATUS_DIR):
    <name>/route.json   route, spawn, sections, names of the raster and wall files (metres, apparatus frame)
    <name>/*_walk.npz   grid[ix, iy] heights (m), x0, y0 (centre of cell 0, 0), res; tall obstacle cells set to the floor
    <name>/*walls.json  boxes: center, half, quat (wxyz) of every tall geom (walls, jambs, posts, rails, braces)
"""
from __future__ import annotations

import json
import os
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent


def data_dir() -> pathlib.Path:
    p = os.environ.get("PA_APPARATUS_DIR")
    if p:
        return pathlib.Path(p)
    local = HERE / "data" / "apparatus"
    return local if local.exists() else pathlib.Path("/mnt/filesystem-g4/data/apparatus")


def quat_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class Apparatus:
    def __init__(self, name: str):
        self.name = name
        self.dir = data_dir() / name
        self.route = json.load(open(self.dir / "route.json"))
        z = np.load(self.dir / self.route["raster"])
        self.grid = np.asarray(z["grid"], np.float64)            # [ix, iy]
        self.x0, self.y0, self.res = float(z["x0"]), float(z["y0"]), float(z["res"])
        self.nx, self.ny = self.grid.shape
        self.floor = float(self.route.get("floor", 0.0))
        self.boxes = json.load(open(self.dir / self.route["walls"]))["boxes"]
        self.waypoints = self.route["waypoints"]
        self.sections = self.route["sections"]
        self.spawn = self.route["spawn"]                         # x, y (m), yaw (deg)
        for b in self.boxes:                                     # world extents, for quick tests
            R = quat_mat(b["quat"])
            e = np.abs(R) @ np.array(b["half"])
            b["lo"], b["hi"] = (np.array(b["center"]) - e), (np.array(b["center"]) + e)
        self.box_lo = np.array([b["lo"] for b in self.boxes])
        self.box_hi = np.array([b["hi"] for b in self.boxes])

    # nearest-cell ground height of the walk raster
    def cell(self, x, y):
        i = np.clip(np.round((np.asarray(x) - self.x0) / self.res).astype(int), 0, self.nx - 1)
        j = np.clip(np.round((np.asarray(y) - self.y0) / self.res).astype(int), 0, self.ny - 1)
        return i, j

    def height(self, x, y):
        i, j = self.cell(x, y)
        return self.grid[i, j]

    def height_patch(self, x, y, r=0.08, fn=np.median):
        i, j = self.cell(x, y)
        k = max(1, int(round(r / self.res)))
        return float(fn(self.grid[max(i - k, 0):i + k + 1, max(j - k, 0):j + k + 1]))

    def inside(self, x, y, margin=0.0):
        return (self.x0 + margin <= x <= self.x0 + (self.nx - 1) * self.res - margin
                and self.y0 + margin <= y <= self.y0 + (self.ny - 1) * self.res - margin)

    def path(self):
        """(N, 2) spawn then every waypoint."""
        return np.array([self.spawn[:2]] + [[w["x"], w["y"]] for w in self.waypoints], float)

    def wall_clearance(self, x, y, z_lo, z_hi):
        """Horizontal distance (m) from the point to the nearest wall box that overlaps heights z_lo..z_hi. Exact for
        boxes with one axis vertical; a tilted box (a brace) is sampled every 2 cm inside the height band."""
        best = np.inf
        near = np.where((self.box_hi[:, 2] > z_lo) & (self.box_lo[:, 2] < z_hi)
                        & (self.box_lo[:, 0] - 1.0 < x) & (self.box_hi[:, 0] + 1.0 > x)
                        & (self.box_lo[:, 1] - 1.0 < y) & (self.box_hi[:, 1] + 1.0 > y))[0]
        for k in near:
            b = self.boxes[k]
            R = quat_mat(b["quat"])
            h = np.array(b["half"])
            if abs(np.abs(R[2]).max() - 1.0) < 1e-6:
                p = R.T @ (np.array([x, y, 0.0]) - np.array([b["center"][0], b["center"][1], 0.0]))
                d = np.maximum(np.abs(p) - h, 0.0)
                d[int(np.argmax(np.abs(R[2])))] = 0.0
                best = min(best, float(np.linalg.norm(d)))
            else:
                n = [max(2, int(2 * hh / 0.02) + 1) for hh in h]
                gx, gy, gz = np.meshgrid(*[np.linspace(-hh, hh, nn) for hh, nn in zip(h, n)], indexing="ij")
                P = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], 1) @ R.T + np.array(b["center"])
                P = P[(P[:, 2] > z_lo) & (P[:, 2] < z_hi)]
                if len(P):
                    best = min(best, float(np.hypot(P[:, 0] - x, P[:, 1] - y).min()))
        return best
