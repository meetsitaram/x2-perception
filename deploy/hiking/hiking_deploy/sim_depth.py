"""Chest depth camera in MuJoCo: the RAW 64 x 36 z-depth image (metres) the RealSense publisher gives on the robot.

Same ray cast as eval/adapters/hiking.py (HikingHandover.depth_frame): camera on the card's body at its pos / quat,
pinhole from the card's apertures, rays against the chosen geom groups (terrain, walls and the robot's own visual meshes,
as in training), hits nearer than the card's min distance re-cast from it. No blur / crop / normalisation here: that is
core.preprocess_depth, which runs on the deploy side for sim and robot alike.
"""
from __future__ import annotations

import mujoco
import numpy as np

from .core import camera_rays, quat_wxyz_to_mat


class ChestDepthCaster:
    def __init__(self, model, depth_card, geom_groups=(1, 4, 5)):
        self.m = model
        self.dc = depth_card
        self.body = model.body(depth_card["body"]).id
        self.R_off = quat_wxyz_to_mat(depth_card["quat_wxyz"])
        self.p_off = np.array(depth_card["pos_in_body"], float)
        self.min_dist = float(depth_card["min_distance_m"])
        self.dirs = camera_rays(depth_card)                       # (36, 64, 3)
        self.flat = self.dirs.reshape(-1, 3)
        self.gg = np.zeros(6, np.uint8)
        for g in geom_groups:
            self.gg[g] = 1
        n = len(self.flat)
        self._gid, self._dist = np.zeros(n, np.int32), np.zeros(n)

    def pose(self, d):
        R_b = d.xmat[self.body].reshape(3, 3)
        return d.xpos[self.body] + R_b @ self.p_off, R_b @ self.R_off

    def render(self, d):
        """(36, 64) float32 z-depth in metres; inf where no hit."""
        p, R = self.pose(d)
        dw = np.ascontiguousarray(self.flat @ R.T)
        n = len(dw)
        mujoco.mj_multiRay(self.m, d, p, dw.ravel(), self.gg, 1, -1, self._gid, self._dist, None, n, 1e6)
        dist = self._dist.copy()
        close = (dist >= 0) & (dist < self.min_dist)
        if close.any():
            g1 = np.zeros(1, np.int32)
            for k in np.nonzero(close)[0]:
                r = mujoco.mj_ray(self.m, d, p + self.min_dist * dw[k], dw[k], self.gg, 1, -1, g1)
                dist[k] = r + self.min_dist if r >= 0 else -1
        dist[dist < 0] = np.inf
        return (dist * self.flat[:, 0]).reshape(self.dirs.shape[:2]).astype(np.float32)
