"""The adapter interface: how a track's policy plugs into the common eval.

The harness owns the physics, the route, the command and the scoring. An adapter owns everything policy-specific:
what the policy observes (built from the simulator through `View`), the network, and the mapping from its action to
joint position targets. One adapter per track (adapters/acp.py, ssr.py, hiking.py), written when the track has a
checkpoint; the track documents its observation and action in its state file and the adapter must match the training
code term by term (order, scales, history, joint order, default pose, action scale, PD gains, control rate).

    class MyAdapter(Adapter):
        name = "ssr"
        control_dt = 0.02
        def plant(self, joint_names): return dict(kp=..., kd=...)        # optional: the gains the policy trained with
        def reset(self, view, rng): ...                                  # clear histories
        def act(self, view, cmd) -> np.ndarray                           # (31,) joint position targets, model joint order

What a policy may use is what its deployed version would have on the real robot: proprioception, the command, and its
own sensing of the terrain (depth from the head camera, or a height scan if the track declares a map on the robot).
`View` offers both; the adapter declares which it uses in `observes` and that goes into every result file.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass
class Command:
    """Produced by the harness from the route's waypoints only, once per control step."""
    target_xy: np.ndarray          # current waypoint, world (m)
    target_z: float                # ground height at the waypoint (m)
    target_heading: float | None   # rad, world; set at stations (the way the station is faced), else None
    hold: bool                     # True while the robot is inside a station's radius and must stand
    vel: np.ndarray                # (vx, vy, wz) in the robot's heading frame: the harness's standard velocity command
    dist: float                    # horizontal distance to the waypoint (m)
    heading_err: float             # rad: direction to the waypoint (or the station heading while holding) minus yaw
    index: int                     # waypoint index
    total: int                     # number of waypoints
    path_ahead: np.ndarray         # (k, 2) the next waypoints after the current one, world (m); may be empty
    station: bool = False          # the current waypoint is a station (the robot has to stop and stand there)


class View:
    """Read-only access to the simulator for building observations."""

    def __init__(self, x2, data, app):
        self.x2, self.m, self.d, self.app = x2, x2.model, data, app
        self.joint_names = x2.joint_names
        self.default_q = x2.plant["default_q"]
        self._pelvis = self.m.body("pelvis").id
        self._gg = np.zeros(6, np.uint8)
        self._gg[4] = self._gg[5] = 1                       # terrain and walls: what a depth sensor sees

    @property
    def t(self):
        return float(self.d.time)

    def q(self):
        return self.d.qpos[self.x2.qadr].copy()

    def qd(self):
        return self.d.qvel[self.x2.dadr].copy()

    def body_pose(self, body="pelvis"):
        b = self.m.body(body).id
        return self.d.xpos[b].copy(), self.d.xquat[b].copy(), self.d.xmat[b].reshape(3, 3).copy()

    def body_velocity(self, body="pelvis"):
        """(linear, angular) velocity in the body's own frame."""
        v6 = np.zeros(6)
        mujoco.mj_objectVelocity(self.m, self.d, mujoco.mjtObj.mjOBJ_BODY, self.m.body(body).id, v6, 1)
        return v6[3:].copy(), v6[:3].copy()

    def projected_gravity(self, body="pelvis"):
        return self.body_pose(body)[2].T @ np.array([0.0, 0.0, -1.0])

    def height_scan(self, xy_world, with_walls=False):
        """Ground height (m, world z) of the walk raster at (N, 2) world points; with_walls adds the tall geometry by
        casting rays down instead (what an elevation map built on the robot would hold)."""
        xy = np.atleast_2d(np.asarray(xy_world, float))
        if self.app is None:
            return np.zeros(len(xy))
        if not with_walls:
            return self.app.height(xy[:, 0], xy[:, 1])
        top = 3.0
        o = np.c_[xy, np.full(len(xy), top)]
        return top - self.rays(o, np.tile([0.0, 0.0, -1.0], (len(xy), 1)), far=top + 1.0)

    def rays(self, origins, dirs, far=5.0):
        """Ray lengths (N,) from world origins along unit world directions to the terrain and walls; `far` if nothing is hit.
        The robot's own body is not seen (add self-occlusion in the adapter if the policy trained with it)."""
        origins, dirs = np.asarray(origins, float), np.asarray(dirs, float)
        out = np.full(len(origins), far)
        gid = np.zeros(1, np.int32)
        for i in range(len(origins)):
            r = mujoco.mj_ray(self.m, self.d, origins[i], dirs[i], self._gg, 1, -1, gid)
            if 0 <= r < far:
                out[i] = r
        return out

    def depth(self, body, pos, pitch_down, fov, n, far=5.0):
        """(n, n) depth image (ray length along each pixel's ray, m) of a pinhole camera fixed on `body` at local `pos`,
        looking forward and `pitch_down` rad below the body's horizontal; row 0 is the top, column 0 the left."""
        p, _, R = self.body_pose(body)
        cp, sp = np.cos(pitch_down), np.sin(pitch_down)
        Rc = R @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])      # camera frame: x forward, y left, z up
        half = np.tan(fov / 2)
        c = (np.arange(n) + 0.5) / n * 2 - 1
        v, u = np.meshgrid(c, c, indexing="ij")
        r = np.stack([np.ones_like(u), -u * half, -v * half], -1).reshape(-1, 3)
        r /= np.linalg.norm(r, axis=1, keepdims=True)
        o = p + R @ np.asarray(pos, float)
        return self.rays(np.tile(o, (len(r), 1)), r @ Rc.T, far).reshape(n, n)


class Adapter:
    name = "base"
    checkpoint = None              # path or label, written into results and render labels
    control_dt = 0.02              # s; a multiple of the harness's SIM_DT
    vmax = 0.6                     # m/s: forward speed limit of the standard velocity command
    wmax = 1.0                     # rad/s: yaw rate limit of the standard velocity command
    observes = "nothing"           # one line: what the policy observes (goes into the result files)

    def plant(self, joint_names):
        """Optional dict with any of kp, kd, effort, armature, default_q as (31,) arrays in `joint_names` order."""
        return None

    # optional: def start_q(self, joint_names) -> (31,) joint positions the trial starts in (default: the plant's default pose)

    def reset(self, view: View, rng: np.random.Generator):
        pass

    def act(self, view: View, cmd: Command) -> np.ndarray:
        raise NotImplementedError
