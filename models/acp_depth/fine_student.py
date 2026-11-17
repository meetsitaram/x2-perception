"""Fine-map depth student for ACP (2026-11-16, after the 10 cm-grid student flattened the 4x4s: an 8.9 cm beam on 10 cm sample
points flickers in the labels, and the net hedges at half height). The camera's points are projected into a bird's-eye grid
around the robot (2.5 cm cells, the camera pose from the IMU attitude + leg kinematics), a small U-Net turns the current grid
and the grids of 4 / 8 / 12 frames before (0.5 s of memory: the beam under the feet) into a fine height map, and ACP's two
scans are read off it at their exact points. ACP is unchanged.

forward(depth (B, 1, H, W) metres, 0 = no return; prop (B, P); cam_T (B, 12) camera rotation (camera -> heading frame, 3 x 3)
and position (heading frame, relative to the root); h (B, 13, 4, NX, NY) the last 13 grids (now + 12 before), or empty)
  -> plan (B, 187), track (B, 63), h, fine (B, NX, NY). Heights: minus the root height, metres, as ACP reads them.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

RES, X0, NX, NY = 0.025, -0.40, 52, 44          # eval/adapters/acp_depth.py FINE_*
HFOV, VFOV = 87.0, 58.0
LAGS = (4, 8, 12)                               # previous grids (camera frames at 25 Hz)


def acp_points():
    """ACP's planner (17 x 11 from x -0.8) and tracker (9 x 7 from x -0.2) points, 0.1 m, heading frame, x-major."""
    def grid(nx, ny, x0):
        xs = x0 + 0.1 * torch.arange(nx, dtype=torch.float32)
        ys = 0.1 * (torch.arange(ny, dtype=torch.float32) - (ny - 1) / 2)
        gx, gy = torch.meshgrid(xs, ys, indexing="ij")
        return torch.stack((gx.flatten(), gy.flatten()), 1)
    return grid(17, 11, -0.8), grid(9, 7, -0.2)


def to_sample_grid(xy):
    """Heading-frame points -> grid_sample coordinates on the (NX, NY) map (align_corners=True); x beyond the map's rear
    edge is clamped onto it (ACP does not need the rear rows)."""
    fi = ((xy[:, 0] - X0) / RES).clamp(0, NX - 1)
    fj = (xy[:, 1] / RES + (NY - 1) / 2).clamp(0, NY - 1)
    return torch.stack((fj / (NY - 1) * 2 - 1, fi / (NX - 1) * 2 - 1), 1)       # (x = column, y = row)


def block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, 1, 1), nn.GroupNorm(8, cout), nn.ELU(),
                         nn.Conv2d(cout, cout, 3, 1, 1), nn.GroupNorm(8, cout), nn.ELU())


class FineStudent(nn.Module):
    def __init__(self, prop_dim: int, H: int = 45, W: int = 80):
        super().__init__()
        self.x0, self.res, self.nx, self.ny = X0, RES, NX, NY     # TorchScript: no module-level floats in methods
        fx, fy = W / 2 / math.tan(math.radians(HFOV / 2)), H / 2 / math.tan(math.radians(VFOV / 2))
        v, u = torch.meshgrid(torch.arange(H, dtype=torch.float32) + 0.5, torch.arange(W, dtype=torch.float32) + 0.5, indexing="ij")
        r = torch.stack((torch.ones_like(u), -(u - W / 2) / fx, -(v - H / 2) / fy), -1).reshape(-1, 3)
        self.register_buffer("rays", r / r.norm(dim=1, keepdim=True))          # camera frame unit rays (x fwd, y left, z up)
        p, t = acp_points()
        self.register_buffer("g_plan", to_sample_grid(p))
        self.register_buffer("g_track", to_sample_grid(t))
        self.register_buffer("prop_mean", torch.zeros(prop_dim))
        self.register_buffer("prop_std", torch.ones(prop_dim))
        self.nlag = len(LAGS)
        self.lags = list(LAGS)
        cin = 4 * (1 + len(LAGS)) + 16
        self.pmap = nn.Sequential(nn.Linear(prop_dim, 64), nn.ELU(), nn.Linear(64, 16))
        self.e1, self.e2, self.e3 = block(cin, 32), block(32, 64), block(64, 128)
        self.d2, self.d1 = block(128 + 64, 64), block(64 + 32, 32)
        self.out = nn.Conv2d(32, 1, 1)

    def project(self, depth, cam_T):
        """-> (B, 4, NX, NY): max height, mean height (minus root, scaled), point count (log), observed mask."""
        B = depth.shape[0]
        z = depth.reshape(B, -1)
        R, t = cam_T[:, :9].reshape(B, 3, 3), cam_T[:, 9:]
        P = self.rays[None] * (z / self.rays[:, 0][None])[..., None]           # camera frame points (z-depth -> range)
        P = torch.einsum("bij,bnj->bni", R, P) + t[:, None]                     # heading frame
        i = torch.round((P[..., 0] - self.x0) / self.res).long()
        j = torch.round(P[..., 1] / self.res + (self.ny - 1) / 2).long()
        ok = (z > 0) & (i >= 0) & (i < self.nx) & (j >= 0) & (j < self.ny)
        idx = torch.where(ok, i * self.ny + j, torch.zeros_like(i)) + torch.arange(B, device=depth.device)[:, None] * (self.nx * self.ny)
        hz = torch.where(ok, P[..., 2], torch.full_like(z, -10.0))
        w = ok.float()
        mx = torch.full((B * self.nx * self.ny,), -10.0, device=depth.device).scatter_reduce(0, idx.flatten(), hz.flatten(), "amax")
        sm = torch.zeros(B * self.nx * self.ny, device=depth.device).index_add(0, idx.flatten(), (P[..., 2] * w).flatten())
        cn = torch.zeros(B * self.nx * self.ny, device=depth.device).index_add(0, idx.flatten(), w.flatten())
        seen = cn > 0
        mx = torch.where(seen, mx, torch.zeros_like(mx))
        mean = torch.where(seen, sm / cn.clamp(min=1.0), torch.zeros_like(sm))
        g = torch.stack(((mx + 0.7) / 0.3 * seen.float(), (mean + 0.7) / 0.3 * seen.float(), torch.log1p(cn), seen.float()), 0)
        return g.reshape(4, B, self.nx, self.ny).permute(1, 0, 2, 3)

    def decode(self, grids, prop):
        """grids (B, 4 * (1 + nlag), NX, NY), prop (B, P) -> fine (B, NX, NY)."""
        B = grids.shape[0]
        pm = self.pmap((prop - self.prop_mean) / self.prop_std)[:, :, None, None].expand(B, 16, self.nx, self.ny)
        x1 = self.e1(torch.cat((grids, pm), 1))
        x2 = self.e2(F.max_pool2d(x1, 2))
        x3 = self.e3(F.max_pool2d(x2, 2))
        y2 = self.d2(torch.cat((F.interpolate(x3, size=x2.shape[-2:], mode="bilinear", align_corners=False), x2), 1))
        y1 = self.d1(torch.cat((F.interpolate(y2, size=x1.shape[-2:], mode="bilinear", align_corners=False), x1), 1))
        return self.out(y1)[:, 0] * 0.3 - 0.7

    def sample(self, fine):
        B = fine.shape[0]
        m = fine[:, None]
        plan = F.grid_sample(m, self.g_plan[None, None].expand(B, 1, -1, 2), align_corners=True)[:, 0, 0]
        track = F.grid_sample(m, self.g_track[None, None].expand(B, 1, -1, 2), align_corners=True)[:, 0, 0]
        return plan, track

    def forward(self, depth, prop, cam_T, h):
        g = self.project(depth, cam_T)
        if h.numel() == 0:
            h = g[:, None].expand(g.shape[0], 13, 4, self.nx, self.ny).contiguous()     # the current grid + 12 before
        h = torch.cat((h[:, 1:], g[:, None]), 1)                               # newest last: h[:, 12] = now
        past = [h[:, 12 - lag] for lag in self.lags]
        fine = self.decode(torch.cat([g] + past, 1), prop)
        plan, track = self.sample(fine)
        return plan, track, h, fine
