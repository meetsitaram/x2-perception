"""Sanity adapters with no learned policy.

zero   the zero action: every joint held at the plant's default pose by the PD, whatever the command. It is not expected
       to walk; it checks that the model loads, stands on the apparatus, and that the scoring sees a robot that goes
       nowhere (0 waypoints, a timeout or a fall, feet on the ground and nothing else). With the deploy plant's gains
       the X2 tips forward from the default pose in about 1.3 s, so this one ends in a fall: the fall check's test.
stand  the default pose plus a hand-written ankle balance (the ankles follow the centre of mass over the feet, read from
       the simulator: privileged, a test fixture, not a policy). It stands still whatever the command, for the standing
       check (--stand) and for testing time limits and stalls on a route.
"""
from __future__ import annotations

import numpy as np

import mujoco

from .base import Adapter, Command, View


class ZeroPolicy(Adapter):
    name = "zero"
    checkpoint = "none (default pose held by the PD)"
    observes = "nothing (constant action)"

    def act(self, view: View, cmd: Command) -> np.ndarray:
        return view.default_q.copy()


class Stand(Adapter):
    name = "stand"
    checkpoint = "none (default pose + hand-written ankle balance on the simulator's centre of mass)"
    observes = "privileged: centre of mass and its velocity over the feet (test fixture)"
    KX, DX, KY, DY = 8.0, 2.0, -8.0, -2.0        # rad/m and rad/(m/s) on ankle pitch (x) and ankle roll (y)

    def reset(self, view: View, rng):
        n = view.joint_names
        self.pitch = [n.index(s + "_ankle_pitch_joint") for s in ("left", "right")]
        self.roll = [n.index(s + "_ankle_roll_joint") for s in ("left", "right")]

    def act(self, view: View, cmd: Command) -> np.ndarray:
        m, d, x = view.m, view.d, view.x2
        mujoco.mj_subtreeVel(m, d)
        feet = np.mean([d.xpos[b] for b in x.foot_ids], 0)
        yaw = np.arctan2(d.xmat[1][3], d.xmat[1][0])
        c, s = np.cos(yaw), np.sin(yaw)
        e, v = d.subtree_com[1] - feet, d.subtree_linvel[1]
        ex, ey = c * e[0] + s * e[1] - 0.038, -s * e[0] + c * e[1]       # 0.038: the sole centre is ahead of the ankle
        vx, vy = c * v[0] + s * v[1], -s * v[0] + c * v[1]
        t = view.default_q.copy()
        t[self.pitch] += np.clip(self.KX * ex + self.DX * vx, -0.6, 0.6)
        t[self.roll] += np.clip(self.KY * ey + self.DY * vy, -0.4, 0.4)
        return t


ADAPTERS = {"zero": ZeroPolicy, "stand": Stand}
