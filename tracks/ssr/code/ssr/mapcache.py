"""Rolling terrain memory around the robot: depth points written into a robot-centred height grid, and the grid moved with odometry.

The grid is 3.2 x 3.2 m at 5 cm (64 x 64), x forward / y left in the heading frame, two channels: height relative to the base (clipped, scaled)
and a known mask. Each control step the old content is resampled by the estimated motion, then the new depth points are written over it, so
ground the camera saw earlier stays available under and beside the feet.

Odometry in training is true motion plus a per-episode velocity and yaw-rate bias and step noise, standing in for the leg-odometry drift of
the real robot. The same class runs on the robot (deploy/ssr_core.py) with the policy's velocity estimate and the gyro.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

N_CELL, RES = 64, 0.05
HALF = N_CELL * RES / 2
SCALE = 1.5                  # height / SCALE, clipped to +-1


class MapCache:
    def __init__(self, n: int, device, bias_v=0.05, bias_w=0.02, noise_v=0.02):
        self.n, self.dev = n, torch.device(device)
        self.h = torch.zeros(n, N_CELL, N_CELL, device=self.dev)
        self.known = torch.zeros(n, N_CELL, N_CELL, device=self.dev)
        self.bias_v, self.bias_w, self.noise_v = bias_v, bias_w, noise_v
        self.bv = torch.zeros(n, 2, device=self.dev)
        self.bw = torch.zeros(n, device=self.dev)

    def reset(self, ids):
        self.h[ids], self.known[ids] = 0.0, 0.0
        k = len(ids)
        if k:
            self.bv[ids] = (torch.rand(k, 2, device=self.dev) * 2 - 1) * self.bias_v
            self.bw[ids] = (torch.rand(k, device=self.dev) * 2 - 1) * self.bias_w

    def shift(self, dx, dy, dyaw, dz, noisy=True):
        """Move the content by the base's motion this step: dx, dy in the previous heading frame (m), dyaw (rad), dz (m, base height change)."""
        if noisy:
            dx = dx * (1 + self.bv[:, 0]) + torch.randn_like(dx) * self.noise_v * 0.02
            dy = dy * (1 + self.bv[:, 1]) + torch.randn_like(dy) * self.noise_v * 0.02
            dyaw = dyaw * (1 + self.bw)
        c, s = torch.cos(dyaw), torch.sin(dyaw)
        # grid_sample coordinates: a sample at new-frame (x, y) reads old-frame R(dyaw) (x, y) + (dx, dy); x is along rows, y along columns
        th = torch.zeros(self.n, 2, 3, device=self.dev)
        th[:, 0, 0], th[:, 0, 1], th[:, 1, 0], th[:, 1, 1] = c, s, -s, c
        th[:, 0, 2], th[:, 1, 2] = dy / HALF, dx / HALF
        x = torch.stack((self.h, self.known), 1)
        g = F.affine_grid(th, x.shape, align_corners=False)
        y = F.grid_sample(x, g, mode="nearest", padding_mode="zeros", align_corners=False)
        self.h, self.known = y[:, 0] - dz[:, None, None] / SCALE * (y[:, 1] > 0.5).float(), y[:, 1]

    def write(self, pts_x, pts_y, rel_z, valid):
        """Depth points in the heading frame: x, y (m) and height relative to the base (m), shapes (N, R); valid (N, R) bool."""
        i = ((pts_x + HALF) / RES).long()
        j = ((pts_y + HALF) / RES).long()
        ok = valid & (i >= 0) & (i < N_CELL) & (j >= 0) & (j < N_CELL)
        env = torch.arange(self.n, device=self.dev)[:, None].expand_as(i)
        e, a, b, v = env[ok], i[ok], j[ok], (rel_z[ok] / SCALE).clamp(-1.0, 1.0)
        self.h[e, a, b] = v
        self.known[e, a, b] = 1.0

    def image(self):
        return torch.stack((self.h * self.known, self.known), 1)
