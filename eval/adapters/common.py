"""Pieces every track adapter needs, in numpy only (the eval machine has no torch): an MLP forward pass from exported
weights, observation normalisation, a history buffer, the heading-frame height-scan grid and the bounded goal command.

A track hands over a policy as an `.npz` (plain MLPs: tools/export_actor.py) or an `.onnx` (anything with a CNN or a
recurrent part; run with onnxruntime) in `/mnt/filesystem-g4/tracks/<track>/eval/`, with an adapter card (see
state/SHARED_LESSONS.md) that states what goes in and what comes out.
"""
from __future__ import annotations

import math

import numpy as np


def elu(x):
    return np.where(x > 0, x, np.expm1(np.minimum(x, 0.0)))


class NpMLP:
    """Linear layers `w0, b0, w1, b1, ...` (torch layout: w is (out, in)) with ELU between them, none after the last."""

    def __init__(self, z, prefix="actor"):
        self.layers = []
        i = 0
        while f"{prefix}_w{i}" in z:
            self.layers.append((np.asarray(z[f"{prefix}_w{i}"], np.float64), np.asarray(z[f"{prefix}_b{i}"], np.float64)))
            i += 1
        assert self.layers, "no layers named %s_w0.. in the file" % prefix
        self.in_dim, self.out_dim = self.layers[0][0].shape[1], self.layers[-1][0].shape[0]

    def __call__(self, x):
        for k, (w, b) in enumerate(self.layers):
            x = w @ x + b
            if k < len(self.layers) - 1:
                x = elu(x)
        return x


class Normaliser:
    """(x - mean) / (sqrt(var) + eps), clipped: rsl_rl-style empirical normalisation frozen at export."""

    def __init__(self, mean, var, eps=1e-5, clip=10.0):
        self.mean, self.std, self.clip = np.asarray(mean, np.float64), np.sqrt(np.asarray(var, np.float64)) + eps, clip

    def __call__(self, x):
        return np.clip((x - self.mean) / self.std, -self.clip, self.clip)


class History:
    """The last n vectors, oldest first when flattened; filled with the first vector at reset."""

    def __init__(self, n):
        self.n, self.buf = n, None

    def reset(self, x):
        self.buf = np.tile(np.asarray(x, np.float64), (self.n, 1))

    def push(self, x):
        self.buf = np.concatenate((self.buf[1:], np.asarray(x, np.float64)[None]), 0)

    def flat(self):
        return self.buf.ravel()


def heading_grid(nx, ny, x0, step=0.1):
    """(nx * ny, 2) points in the robot's heading frame: x from x0 forward in `step`, y centred; x-major order
    (torch.meshgrid(xs, ys, indexing="ij").flatten())."""
    xs = x0 + step * np.arange(nx)
    ys = step * (np.arange(ny) - (ny - 1) / 2)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    return np.stack((gx.ravel(), gy.ravel()), 1)


def to_world(grid, pos_xy, heading):
    c, s = math.cos(heading), math.sin(heading)
    return np.stack((pos_xy[0] + c * grid[:, 0] - s * grid[:, 1], pos_xy[1] + s * grid[:, 0] + c * grid[:, 1]), 1)


def heading_of(R):
    """Yaw of the body's x axis."""
    return math.atan2(R[1, 0], R[0, 0])


def bounded_goal(target_xy, pos_xy, heading, r):
    """Direction to the target in the heading frame, d / max(|d|, r): unit length far away, shrinking inside r."""
    d = np.asarray(target_xy, float) - np.asarray(pos_xy, float)
    c, s = math.cos(heading), math.sin(heading)
    b = np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])
    return b / max(float(np.linalg.norm(b)), r)
