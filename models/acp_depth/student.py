"""Depth -> height-scan student for ACP (plan: docs/TODO_depth_to_heightmap.md). DPL-style but small enough for the robot's
Orin CPU: a depth CNN and a proprioception MLP feed a GRU (memory: terrain that has left the forward-only view, under and
behind the feet), and two heads give ACP's planner scan (17 x 11 = 187) and tracker scan (9 x 7 = 63), height minus root
height in metres, the values ACP reads (eval/adapters/acp.py PLAN_SCAN, SCAN).

forward(depth (B, 1, H, W) metres with 0 = no return, prop (B, P) ACP's proprioception, h (1, B, HID) or empty) ->
(plan (B, 187), track (B, 63), h). One call per camera frame (25 Hz). Exported as TorchScript for
eval/adapters/acp_depth.py (ACPD_MODE=student).
"""
from __future__ import annotations

import torch
import torch.nn as nn

FAR = 3.0          # m: depth normalisation and the value given to no return
HID = 256


class Student(nn.Module):
    def __init__(self, prop_dim: int, H: int = 45, W: int = 80, hid: int = HID):
        super().__init__()
        self.hid = hid
        self.far = FAR
        c = [16, 32, 64, 64]
        self.cnn = nn.Sequential(
            nn.Conv2d(2, c[0], 5, 2, 2), nn.ELU(),
            nn.Conv2d(c[0], c[1], 3, 2, 1), nn.ELU(),
            nn.Conv2d(c[1], c[2], 3, 2, 1), nn.ELU(),
            nn.Conv2d(c[2], c[3], 3, 2, 1), nn.ELU(),
        )
        with torch.no_grad():
            n = self.cnn(torch.zeros(1, 2, H, W)).numel()
        self.dproj = nn.Sequential(nn.Flatten(), nn.Linear(n, 256), nn.ELU())
        self.pproj = nn.Sequential(nn.Linear(prop_dim, 128), nn.ELU(), nn.Linear(128, 64), nn.ELU())
        self.gru = nn.GRU(256 + 64, hid, batch_first=True)
        self.plan = nn.Sequential(nn.Linear(hid + 64, 256), nn.ELU(), nn.Linear(256, 187))
        self.track = nn.Sequential(nn.Linear(hid + 64, 128), nn.ELU(), nn.Linear(128, 63))
        self.register_buffer("prop_mean", torch.zeros(prop_dim))
        self.register_buffer("prop_std", torch.ones(prop_dim))

    def encode(self, depth, prop):
        valid = (depth > 0).float()
        d = torch.where(depth > 0, depth.clamp(max=self.far), torch.full_like(depth, self.far)) / self.far
        f = self.dproj(self.cnn(torch.cat((d, valid), 1)))
        p = self.pproj((prop - self.prop_mean) / self.prop_std)
        return f, p

    def forward(self, depth, prop, h):
        f, p = self.encode(depth, prop)
        if h.numel() == 0:
            h = torch.zeros(1, depth.shape[0], self.hid, device=depth.device)
        y, h = self.gru(torch.cat((f, p), -1)[:, None], h)
        z = torch.cat((y[:, 0], p), -1)
        return self.plan(z), self.track(z), h

    def sequence(self, depth, prop, h0=None):
        """Training: depth (B, T, 1, H, W), prop (B, T, P) -> plan (B, T, 187), track (B, T, 63)."""
        B, T = depth.shape[:2]
        f, p = self.encode(depth.flatten(0, 1), prop.flatten(0, 1))
        f, p = f.view(B, T, -1), p.view(B, T, -1)
        y, _ = self.gru(torch.cat((f, p), -1), h0)
        z = torch.cat((y, p), -1)
        return self.plan(z), self.track(z)
