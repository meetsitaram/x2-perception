"""Props and the height scan for terrain-aware SuperMimic (ACP's contact-informed tracker sees a 9 x 7 height scan).

Props: per clip, a list of static boxes (center xyz, half extents xyz, yaw) from a JSON sidecar keyed by clip name
(``x2_chains_props.json``: the props each chained motion was generated on). Every clip's boxes are assigned to a fixed
set of SLOTS (one MuJoCo mocap box each, so all worlds share one model): a slot per distinct box size, as many of each
as the clip that needs most. A clip leaves unused slots parked far below the floor. Clips with no entry have no props.

Height scan: SCAN_NX x SCAN_NY points on a grid in the robot's heading frame (x forward, y left; SCAN_X / SCAN_Y
spacing, from SCAN_X0 behind to ahead), height of the highest box top under each point (the floor is 0), relative to
the root height (ACP: relative to the base height). Exact and cheap (analytic boxes), so it is the same in the simulator,
in imagined rollouts and in the critic. Not differentiable in the root position (the scan is an input, like the goal).

Course (2026-10-27): a height raster (loco-manip-apparatus ``build/hmap/<scene>_20mm.npz``: x0, y0, res, grid) is the
static terrain every world shares: a MuJoCo heightfield in the physics (add_hfield) and, in the scan, the raster
height of the cell under each point (0 outside it); the scan takes the higher of the boxes and the raster.

Walls (2026-10-28): the course raster has its tall obstacles removed (a heightfield cannot leave the space under a
rail or a door header open), so walls, door jambs, posts and rails come in as static boxes instead (loco-manip-apparatus
``pipeline.walls``: ``build/hmap/cad_walls.json``, each tall CAD geom near the route as its oriented box). They collide
with the whole robot, so an arm can no longer pass through a door post. Not in the height scan.

Collision layers: the floor and the robot use bit 1 as before; props and the heightfield also carry bit 2 in their
conaffinity, so a geom with contype 2 only (the terrain foot boxes, assets.build_x2(foot_terrain=True)) collides with
them and with nothing else.
"""

from __future__ import annotations

import json

import torch

SCAN_NX, SCAN_NY = 9, 7
SCAN_X0, SCAN_X, SCAN_Y = -0.2, 0.1, 0.1          # m: first row 0.2 m behind, 0.1 m apart, 0.6 m wide
SCAN_DIM = SCAN_NX * SCAN_NY
PARK_Z = -20.0                                    # m: where an unused slot sits
TERRAIN_BIT = 2                                   # conaffinity bit of props and the heightfield (see the module doc)


def _grid(device):
    xs = SCAN_X0 + SCAN_X * torch.arange(SCAN_NX, device=device, dtype=torch.float32)
    ys = SCAN_Y * (torch.arange(SCAN_NY, device=device, dtype=torch.float32) - (SCAN_NY - 1) / 2)
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    return torch.stack((gx.flatten(), gy.flatten()), -1)    # (63, 2)


class Terrain:
    """Per-clip prop boxes as slot assignments, the slot sizes, and the height scan."""

    def __init__(self, names: list[str], props: dict | None, device, hmap: str = ""):
        self.device = torch.device(device)
        self.raster = None
        if hmap:
            import numpy as np
            z = np.load(hmap)
            grid = np.nan_to_num(np.maximum(z["grid"], 0.0)).astype(np.float32)
            self.raster = dict(x0=float(z["x0"]), y0=float(z["y0"]), res=float(z["res"]), grid=grid,
                               t=torch.as_tensor(grid, device=self.device))
        props = props or {}
        sizes, need = [], []                                          # distinct box sizes and how many of each
        for boxes in props.values():
            count = {}
            for b in boxes:
                key = tuple(round(float(x), 3) for x in b["half"])
                if key not in sizes:
                    sizes.append(key)
                count[key] = count.get(key, 0) + 1
            need.append(count)
        n_of = {k: max(c.get(k, 0) for c in need) for k in sizes}
        self.slot_half = [k for k in sizes for _ in range(n_of[k])]   # slot -> half extents
        S = len(self.slot_half)
        # per clip and slot: center xyz, yaw, present
        self.boxes = torch.zeros((len(names), max(S, 1), 5), device=self.device)
        self.boxes[..., 2] = PARK_Z
        for c, name in enumerate(names):
            used = set()
            for b in props.get(name, []):
                key = tuple(round(float(x), 3) for x in b["half"])
                s = next(i for i, k in enumerate(self.slot_half) if k == key and i not in used)
                used.add(s)
                self.boxes[c, s] = torch.tensor([*b["center"], b["yaw"], 1.0], device=self.device)
        self.half = torch.tensor(self.slot_half if S else [(0.0, 0.0, 0.0)], device=self.device, dtype=torch.float32)
        self.grid = _grid(self.device)

    @staticmethod
    def load(names: list[str], paths: str, device, hmap: str = "") -> "Terrain":
        """Props from one sidecar or several (comma-separated), merged by clip name; optional course raster."""
        props = {}
        for path in filter(None, paths.split(",")):
            props.update(json.load(open(path)))
        return Terrain(names, props or None, device, hmap)

    @property
    def num_slots(self) -> int:
        return len(self.slot_half)

    def mocap(self, clip: torch.Tensor):
        """(n, S, 3) positions and (n, S, 4) wxyz quaternions of each slot's box for clips `clip`."""
        b = self.boxes[clip]
        yaw = b[..., 3]
        q = torch.stack((torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2)), -1)
        return b[..., :3], q

    def height(self, wx: torch.Tensor, wy: torch.Tensor, clip: torch.Tensor) -> torch.Tensor:
        """(B, N) ground height at world points (wx, wy) (B, N) for clips `clip` (B,): the highest box top or raster
        cell under each point (the floor is 0). Row chunks keep the (B, S, N) temporaries ~<= 0.5 GB each: one call on
        the 65k imagined states x 187 planner points spiked 38.5 GB and spilled the 32 GB card into system RAM."""
        step = max(1, (2**27) // max(1, self.boxes.shape[1] * wx.shape[1]))
        if wx.shape[0] > step:
            return torch.cat([self._height(wx[i:i + step], wy[i:i + step], clip[i:i + step])
                              for i in range(0, wx.shape[0], step)])
        return self._height(wx, wy, clip)

    def _height(self, wx: torch.Tensor, wy: torch.Tensor, clip: torch.Tensor) -> torch.Tensor:
        b = self.boxes[clip]                                              # (B, S, 5)
        cy, sy = torch.cos(b[..., 3]), torch.sin(b[..., 3])
        dx = wx[:, None, :] - b[..., 0:1]                                 # (B, S, N)
        dy = wy[:, None, :] - b[..., 1:2]
        u = cy[..., None] * dx + sy[..., None] * dy
        v = -sy[..., None] * dx + cy[..., None] * dy
        hx, hy, hz = self.half[None, :, 0:1], self.half[None, :, 1:2], self.half[None, :, 2:3]
        inside = (u.abs() <= hx) & (v.abs() <= hy) & (b[..., 4:5] > 0)
        top = torch.where(inside, b[..., 2:3] + hz, torch.zeros_like(u)).amax(1)    # (B, N); floor 0
        if self.raster is not None:
            r = self.raster
            i = torch.round((wx - r["x0"]) / r["res"]).long()
            j = torch.round((wy - r["y0"]) / r["res"]).long()
            nx, ny = r["t"].shape
            ok = (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
            h = r["t"][i.clamp(0, nx - 1), j.clamp(0, ny - 1)]
            top = torch.maximum(top, torch.where(ok, h, torch.zeros_like(h)))
        return top

    def penetration(self, pts: torch.Tensor, clip: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
        """(B, N) depth of world points pts (B, N, 3) inside the deepest prop box of clips `clip` (B,) (0 outside every
        box): distance to the box's nearest face. Differentiable in pts (edge-penetration reward). Boxes only, not
        the raster. margin: the points are spheres of this radius (depth + margin, 0 when the sphere is clear)."""
        # the deepest box per point without autograd, then its depth recomputed with gradient: same value and gradient
        # as amax over all boxes, but autograd keeps (B, N) instead of (B, S, N) tensors (stage-2 box slots x 65k
        # imagined states x 18 foot points held ~5 GB through the backward pass)
        with torch.no_grad():
            k = self._box_depth(pts.detach(), self.boxes[clip][:, :, None, :], self.half[None, :, None, :]).argmax(1)
        b = self.boxes[clip].gather(1, k[..., None].expand(-1, -1, 5))  # (B, N, 5): each point's deepest box
        h = self.half[k]                                                  # (B, N, 3)
        return torch.relu(self._box_depth(pts, b, h) + margin)

    @staticmethod
    def _box_depth(pts, b, h):
        """Signed depth of points pts (B, N, 3) in boxes b (..., 5) with half sizes h (..., 3), broadcast over a box
        axis if b has one (B, S, 1, 5); -1 for parked (absent) boxes."""
        cy, sy = torch.cos(b[..., 3]), torch.sin(b[..., 3])
        p = pts[:, None] if b.dim() == 4 else pts
        dx, dy, dz = p[..., 0] - b[..., 0], p[..., 1] - b[..., 1], p[..., 2] - b[..., 2]
        u = cy * dx + sy * dy
        v = -sy * dx + cy * dy
        d = torch.minimum(torch.minimum(h[..., 0] - u.abs(), h[..., 1] - v.abs()), h[..., 2] - dz.abs())
        return torch.where(b[..., 4] > 0, d, torch.full_like(d, -1.0))

    def scan(self, p: torch.Tensor, R: torch.Tensor, clip: torch.Tensor) -> torch.Tensor:
        """(..., 63) height of the ground under each scan point minus the root height, for roots p (..., 3) with
        rotations R (..., 3, 3) tracking clips `clip` (...). Detached."""
        lead = p.shape[:-1]
        p, R, clip = p.detach().reshape(-1, 3), R.detach().reshape(-1, 3, 3), clip.reshape(-1)
        heading = torch.atan2(R[:, 1, 0], R[:, 0, 0])
        c, s = torch.cos(heading), torch.sin(heading)
        gx, gy = self.grid[:, 0], self.grid[:, 1]
        wx = p[:, :1] + c[:, None] * gx - s[:, None] * gy                 # (B, 63) world points
        wy = p[:, 1:2] + s[:, None] * gx + c[:, None] * gy
        return (self.height(wx, wy, clip) - p[:, 2:3]).reshape(*lead, SCAN_DIM)


def add_slots(spec, slot_half, park_z: float = PARK_Z):
    """One mocap body with a box geom per slot (collides with everything, like the floor, and with the terrain foot
    boxes)."""
    import mujoco
    for i, half in enumerate(slot_half):
        body = spec.worldbody.add_body(name=f"prop_slot_{i}", mocap=True, pos=[0.0, 0.0, park_z])
        g = body.add_geom(name=f"prop_slot_{i}", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(half),
                          rgba=[0.76, 0.6, 0.42, 1.0])
        g.conaffinity = 1 | TERRAIN_BIT
    return spec


def load_walls(path: str) -> list:
    """Wall boxes (center, half, quat wxyz) from a loco-manip-apparatus walls JSON; [] for ""."""
    return json.load(open(path))["boxes"] if path else []


def add_walls(spec, boxes: list):
    """Static boxes on the world body (collide with everything, like the floor, and with the terrain foot boxes)."""
    import mujoco
    for i, b in enumerate(boxes):
        g = spec.worldbody.add_geom(name=f"wall_{i}", type=mujoco.mjtGeom.mjGEOM_BOX, size=b["half"], pos=b["center"],
                                    quat=b["quat"], rgba=[0.7, 0.62, 0.5, 1.0])
        g.conaffinity = 1 | TERRAIN_BIT
    return spec


HF_CHUNK = int(__import__("os").environ.get("ACP_HF_CHUNK", "400"))   # vertices per side of one heightfield geom (ACP agent 2026-11-12, see add_hfield)


def add_hfield(spec, raster: dict, base: float = 0.05):
    """The course raster as static heightfield geoms (cell centres on the raster's points).
    Chunked (ACP agent 2026-11-12, state/acp.md 17c): MuJoCo Warp mis-collides on one 2,260 x 1,819 field (a foot dropped on a
    flat 1.66 m platform got a 9 m deep contact and was thrown 240 m; MuJoCo C and any crop up to 681 x 763 are correct), so the
    raster is split into pieces of at most HF_CHUNK vertices per side that share their border vertices (same surface). A raster
    that fits in one piece gives one geom named "course", as before.
    ACP_HF_FIX=1 (fixes A + B of state/acp.md 16y, measured with tools/drop_test.py): the field collides only with conaffinity
    bit 1 (sole spheres and other robot geoms; the terrain foot boxes no longer sink into it, they still meet props), and its
    contact is stiffer (solref 0.01 s, priority 1; 0.005 clamps to 2 dt). Default 0 keeps the running stage-2 physics."""
    import os
    import mujoco
    import numpy as np
    fix = os.environ.get("ACP_HF_FIX", "0") == "1"
    grid = raster["grid"]
    NX, NY = grid.shape
    res = raster["res"]
    step = HF_CHUNK - 1
    xs = list(range(0, max(NX - 1, 1), step))
    ys = list(range(0, max(NY - 1, 1), step))
    one = len(xs) == 1 and len(ys) == 1
    for i0 in xs:
        for j0 in ys:
            sub = grid[i0:min(i0 + HF_CHUNK, NX), j0:min(j0 + HF_CHUNK, NY)]
            nx, ny = sub.shape
            if nx < 2 or ny < 2:
                continue
            name = "course" if one else f"course_{i0}_{j0}"
            top = max(float(sub.max()), 1e-3)
            hf = spec.add_hfield(name=name, nrow=ny, ncol=nx, size=[(nx - 1) * res / 2, (ny - 1) * res / 2, top, base])
            hf.userdata = (sub.T / top).astype(np.float32).ravel().tolist()        # rows along y, columns along x
            g = spec.worldbody.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname=name,
                                        pos=[raster["x0"] + (i0 + (nx - 1) / 2) * res, raster["y0"] + (j0 + (ny - 1) / 2) * res, 0.0],
                                        rgba=[0.55, 0.5, 0.45, 1.0])
            g.conaffinity = 1 if fix else (1 | TERRAIN_BIT)
            if fix:
                g.solref = [0.01, 1.0]
                g.priority = 1
    return spec
