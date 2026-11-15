"""Option w (2026-11-12): the extended map's checked routes as start-to-target sequences.

On the extended tile columns, at each reset a share HIKING_X2_ROUTE_SHARE (0.7) of the robots get one of their tile's checked
routes (the set's `routes`, after `check_routes.py`): the robot is placed at the route start (its env origin moves there, so
the height terminations use the route's floor), facing the first waypoint (+-0.3 rad), and follows the waypoints in order with
the eval harness's waypoint law (as SSR, 2026-11-12):
  yaw rate   = clip(1.5 x bearing error, -1, 1) rad/s
  forward    = 0 if |bearing error| > 60 deg, else clip(distance, 0.15, vmax) x cos(bearing error)   (vmax: the cell's sampled
               walking speed, 0.45-0.8 m/s)
  pass-through waypoints: the next one is taken inside 0.20 m, without a stop; the last one is a station: zero command inside
  0.20 m, held to the end of the episode.
Route robots are never turners (option y) and keep their route for the whole episode (no mid-episode resample). Other robots
keep the paper's / run 4's commands. Which tile each terrain cell got is recorded by `extended_tile_terrain` (extended._CALLS)
in the generator's order (columns ascending, rows ascending within a column).
"""
from __future__ import annotations

import math
import os

import torch

from isaaclab.utils import configclass

from .keep import TurnFarTargetPoseVelocityCommand, TurnFarTargetPoseVelocityCommandCfg

MAX_WP = 12
# v4j (2026-11-12, user: "there is option of side stepping in really tight spots"): on tight-spot and home tiles a share of the
# route robots walk the route sideways (facing +-90 deg off the bearing, |v| <= SIDE_V); and every route robot gets a lateral
# command for a waypoint off to the side within 1 m (vy = clip(lateral offset, +-ROUTE_VY)), so small side errors are stepped out.
SIDE_SHARE = float(os.environ.get("HIKING_X2_SIDESTEP_SHARE", "0"))
SIDE_V = float(os.environ.get("HIKING_X2_SIDESTEP_V", "0.3"))
ROUTE_VY = float(os.environ.get("HIKING_X2_ROUTE_VY", "0"))
SIDE_TYPES = ("tight_spots", "homes")


def side_law(d_xy, heading, side, is_last, pass_radius=0.2, vside=0.3):
    """Sideways walking: side (n,) +1 = waypoint on the robot's left (walks left), -1 = right. Returns vx, vy, wz, arrived."""
    dist = torch.norm(d_xy, dim=1)
    err = torch.atan2(d_xy[:, 1], d_xy[:, 0]) - heading
    err = torch.atan2(torch.sin(err), torch.cos(err))
    off = err - side * (math.pi / 2)
    off = torch.atan2(torch.sin(off), torch.cos(off))
    wz = torch.clamp(1.5 * off, -1.0, 1.0)
    s = torch.minimum(torch.clamp(dist, min=0.1), torch.full_like(dist, vside))
    s = torch.where(off.abs() > math.pi / 3, torch.zeros_like(s), s)
    vx, vy = s * torch.cos(err), s * torch.sin(err)
    arrived = dist < pass_radius
    stop = arrived & is_last
    z = torch.zeros_like(vx)
    return torch.where(stop, z, vx), torch.where(stop, z, vy), torch.where(stop, z, wz), arrived


def lateral_vy(d_xy, heading, vmax_y):
    """Body-frame lateral offset of a waypoint within 1 m, clipped to +-vmax_y (0 farther away or behind)."""
    dist = torch.norm(d_xy, dim=1)
    c, s = torch.cos(heading), torch.sin(heading)
    lx = c * d_xy[:, 0] + s * d_xy[:, 1]
    ly = -s * d_xy[:, 0] + c * d_xy[:, 1]
    vy = torch.clamp(ly, -vmax_y, vmax_y)
    return torch.where((dist < 1.0) & (lx > -0.1), vy, torch.zeros_like(vy))


def build_route_table(calls, tile_meta, columns_of_type, num_rows, terrain_origins):
    """-> dict (row, col) -> list of routes, each (start (3,), yaw0, waypoints (k, 3)) in world coordinates.
    calls: tile_type -> [(tile index in the set, spawn_centred (3,))] in generator order; tile_meta: tile_type -> list of tile
    dicts (the set's json `tiles`, indexable by `index`); columns_of_type: tile_type -> ascending columns of that type."""
    table = {}
    for ttype, lst in calls.items():
        cols = columns_of_type.get(ttype, [])
        assert len(lst) == len(cols) * num_rows, (ttype, len(lst), len(cols), num_rows)
        by_index = {t["index"]: t for t in tile_meta[ttype]}
        for j, (i, spawn) in enumerate(lst):
            col, row = cols[j // num_rows], j % num_rows
            o = terrain_origins[row][col]
            tile = by_index[i]
            routes = []
            for r in tile.get("routes", []):
                wps = r["waypoints"]
                zs = r.get("z", [0.0] * (len(wps) + 1))
                if len(wps) == 0 or len(wps) > MAX_WP:
                    continue
                to_w = lambda x, y, z: [o[0] + x - spawn[0], o[1] + y - spawn[1], o[2] + z - spawn[2]]
                start = to_w(r["start"][0], r["start"][1], zs[0])
                pts = [to_w(p[0], p[1], zs[min(k + 1, len(zs) - 1)]) for k, p in enumerate(wps)]
                yaw0 = math.atan2(pts[0][1] - start[1], pts[0][0] - start[0])
                routes.append((start, yaw0, pts))
            if routes:
                table[(row, col)] = routes
    return table


def waypoint_law(d_xy, heading, vmax, is_last, pass_radius=0.2):
    """Eval harness law. d_xy (n, 2) world vector robot -> waypoint; heading (n,); vmax (n,); is_last (n,) bool.
    Returns vx, wz (n,) and arrived (n,) bool (inside pass_radius)."""
    dist = torch.norm(d_xy, dim=1)
    err = torch.atan2(d_xy[:, 1], d_xy[:, 0]) - heading
    err = torch.atan2(torch.sin(err), torch.cos(err))
    wz = torch.clamp(1.5 * err, -1.0, 1.0)
    vx = torch.minimum(torch.clamp(dist, min=0.15), vmax) * torch.cos(err)   # clip(distance, 0.15, vmax) x cos(error)
    vx = torch.where(err.abs() > math.pi / 3, torch.zeros_like(vx), vx)
    arrived = dist < pass_radius
    stop = arrived & is_last
    vx = torch.where(stop, torch.zeros_like(vx), vx)
    wz = torch.where(stop, torch.zeros_like(wz), wz)
    return vx, wz, arrived


class RouteFarTargetPoseVelocityCommand(TurnFarTargetPoseVelocityCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        n = self.num_envs
        self.on_route = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.route_wp = torch.zeros(n, MAX_WP, 3, device=self.device)
        self.route_n = torch.ones(n, dtype=torch.long, device=self.device)
        self.route_i = torch.zeros(n, dtype=torch.long, device=self.device)
        self.route_done = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.side = torch.zeros(n, device=self.device)
        self.side_cells = set()
        self._in_reset = False
        self.table = {}
        try:
            from . import extended as ext
            from .monitor import _terrain_names
            if ext._CALLS:
                names, col_type = _terrain_names(env)
                col_type = col_type.tolist()
                type_of_name = {nm: v[0] for nm, v in ext.EXTENDED_COLUMNS.items()}
                cols = {}
                for c, k in enumerate(col_type):
                    tt = type_of_name.get(names[k])
                    if tt is not None:
                        cols.setdefault(tt, []).append(c)
                meta = {tt: ext._load(tt, ext.EXTENDED_DIR)[1] for tt in ext._CALLS}
                gen = self.terrain.cfg.terrain_generator
                self.table = build_route_table(ext._CALLS, meta, cols, gen.num_rows, self.terrain.terrain_origins.tolist())
                self.side_cells = {(r_, c_) for tt in SIDE_TYPES for c_ in cols.get(tt, []) for r_ in range(gen.num_rows)}
        except Exception as e:  # without routes the task is v4h's
            print("[X2 routes] no route table: %r" % (e,), flush=True)
        self.n_routes = sum(len(v) for v in self.table.values())
        print("[X2 routes] %d routes on %d terrain cells, share %.2f" % (self.n_routes, len(self.table), self.cfg.route_share), flush=True)

    def reset(self, env_ids=None):
        self._in_reset = True
        try:
            return super().reset(env_ids)
        finally:
            self._in_reset = False

    def _resample(self, env_ids):
        super()._resample(env_ids)
        if len(env_ids) == 0:
            return
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if not self._in_reset:
            # route robots keep their route through the episode (their time_left is set beyond the episode)
            return
        self.on_route[ids] = False
        self.route_done[ids] = False
        self.side[ids] = 0.0
        if not self.table:
            return
        lv = self.terrain.terrain_levels[ids].tolist()
        ty = self.terrain.terrain_types[ids].tolist()
        pick = torch.rand(len(ids), device=self.device) < self.cfg.route_share
        sel, poses = [], []
        for k, (e, row, col) in enumerate(zip(ids.tolist(), lv, ty)):
            routes = self.table.get((row, col))
            if not routes or not bool(pick[k]):
                continue
            start, yaw0, pts = routes[int(torch.randint(len(routes), (1,)))]
            m = len(pts)
            self.route_wp[e, :m] = torch.tensor(pts, device=self.device)
            self.route_wp[e, m:] = self.route_wp[e, m - 1]
            self.route_n[e], self.route_i[e] = m, 0
            yaw = yaw0 + (torch.rand(1).item() - 0.5) * 0.6
            if SIDE_SHARE > 0 and (row, col) in self.side_cells and torch.rand(1).item() < SIDE_SHARE:
                sd = 1.0 if torch.rand(1).item() < 0.5 else -1.0
                self.side[e] = sd
                yaw -= sd * math.pi / 2   # first waypoint at +-90 deg in the body frame
            sel.append(e)
            poses.append(start + [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
        if not sel:
            return
        sid = torch.tensor(sel, device=self.device, dtype=torch.long)
        pose = torch.tensor(poses, device=self.device, dtype=torch.float)
        # the env origin moves to the route start, so `root_height` and the reset height use the route's floor
        self.terrain.env_origins[sid] = pose[:, :3]
        pose[:, 2] += self.robot.data.default_root_state[sid, 2]
        self.robot.write_root_pose_to_sim(pose, env_ids=sid)
        self.on_route[sid] = True
        self.is_turn_env[sid] = False
        self.time_left[sid] = 1.0e6
        self.pos_command_w[sid] = self.route_wp[sid, 0]

    def _update_command(self):
        super()._update_command()
        r = self.on_route.nonzero().flatten()
        if len(r) == 0:
            return
        pos = self.robot.data.root_pos_w[r, :2]
        for _ in range(2):  # at most two pass-throughs per step
            cur = self.route_wp[r, self.route_i[r]]
            is_last = self.route_i[r] >= self.route_n[r] - 1
            arrived = torch.norm(cur[:, :2] - pos, dim=1) < self.cfg.pass_radius
            adv = arrived & ~is_last
            if not adv.any():
                break
            self.route_i[r[adv]] += 1
        cur = self.route_wp[r, self.route_i[r]]
        is_last = self.route_i[r] >= self.route_n[r] - 1
        vmax = torch.clamp(self.max_command_b[r, 0], min=0.45)
        d_xy, hd = cur[:, :2] - pos, self.robot.data.heading_w[r]
        vx, wz, arrived = waypoint_law(d_xy, hd, vmax, is_last, self.cfg.pass_radius)
        vy = lateral_vy(d_xy, hd, ROUTE_VY) if ROUTE_VY > 0 else torch.zeros_like(vx)
        sd = self.side[r]
        if (sd != 0).any():
            svx, svy, swz, _ = side_law(d_xy, hd, sd, is_last, self.cfg.pass_radius, SIDE_V)
            m = sd != 0
            vx, vy, wz = torch.where(m, svx, vx), torch.where(m, svy, vy), torch.where(m, swz, wz)
        self.route_done[r] |= arrived & is_last
        done = self.route_done[r] | (arrived & is_last)
        vx = torch.where(done, torch.zeros_like(vx), vx)
        vy = torch.where(done, torch.zeros_like(vy), vy)
        wz = torch.where(done, torch.zeros_like(wz), wz)
        self.vel_command_b[r, 0] = vx
        self.vel_command_b[r, 1] = vy
        self.vel_command_b[r, 2] = wz
        self.pos_command_w[r] = cur


@configclass
class RouteFarTargetPoseVelocityCommandCfg(TurnFarTargetPoseVelocityCommandCfg):
    class_type: type = RouteFarTargetPoseVelocityCommand
    route_share: float = 0.7
    pass_radius: float = 0.2
