"""Height maps of the apparatus: a cached course raster, robot-centred maps, and maps along any path.

Heights come from straight-down raycasts against the apparatus geometry (geom group 0: decks, lumber, ramps, stairs,
walls, posts and rails; the robot is parked out of the way). The top surface wins, as an onboard elevation map would
report it, so walls and rails show as tall cells. Heights are metres above the lane deck (z = 0).

Obstacles: a cell is walkable when it can be reached from the start of the lane by steps of at most MAX_STEP between
neighbouring cells over surfaces no steeper than MAX_SLOPE (deck, 4x4s, ramps, every stair tread, the landing).
The slope test is one-sided (the flatter side of each cell), so a stair edge counts as flat but a brace does not. Everything else (rails, posts, walls,
cabinets, crates) is an obstacle the robot must not touch, and a MARGIN band around obstacles is kept clear too.

    course = Course()                                   # raster cached under build/hmap/
    h = course.height([[x, y], ...])                    # bilinear lookup, world metres
    local = local_map(course, x, y, yaw, ref_z=foot_z)  # (nx, ny) robot-frame grid, relative to the lowest stance foot
    cls = local_classes(course, x, y, yaw)              # same grid: FREE, MARGIN or OBSTACLE
    poses, maps = path_maps(course, route_waypoints('mobility'))

Units: world metres; route waypoints in x2_route are inches in the same frame (x_in * 0.0254).
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass

import mujoco
import numpy as np
from collections import deque
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.dirname(HERE)
ROOT = os.path.dirname(SIM)
sys.path.insert(0, SIM)
IN = 0.0254
SCENES = {'cad': os.path.join(SIM, 'scene_x2_env_mobility.xml'),          # 7-1/3 in stair risers
          'low': os.path.join(SIM, 'scene_x2_env_mobility_low.xml'),      # 10.9 cm risers (e2e.py)
          'nist': os.path.join(ROOT, 'build', 'nist', 'scene_x2_nist_norobot.xml')}   # official testbed (sim/nist)
# Per scene: geom group the rays hit, a walkable seed point (m), joints set before casting (name -> qpos), the deck
# height (m) and the 4x4 height band above it (routefit.lumber), and a scene whose X2 is attached when the scene file has
# none. Ours: the apparatus geometry is group 0, the seed is SEED_IN, the deck is at 0. NIST (sim/nist/scene_nist.py):
# collision geometry is group 3, the seed is the start platform, the door stands open as in the mobility trials (rules),
# the deck is at 0.14 m; its door threshold stands 8 cm, its 4x4s 12.6 cm above the deck, so the band starts at 10 cm.
# thin: geometry thinner than a cell is drawn in at its top (the NIST lane's 1 cm partitions, 0.8 m tall, fall between
# ray columns at 2 cm: the raster showed a 2 cm lip where a wall stands).
SCENE_CFG = {'nist': dict(group=3, seed=(12.0, 4.21), joints={'door': -1.65}, floor=0.14, lumber_h=(0.10, 0.15),
                          robot='cad', thin=True)}
CACHE = os.path.join(ROOT, 'build', 'hmap')
RAY_TOP = 3.5            # m: rays start above everything on the course
OVERHEAD = 1.4           # m: geometry whose underside is higher than this is walked under (X2 ~1.3 m tall)
THIN_TALL = 0.10         # m: thin geometry this tall or more is drawn in (Course thin)
TERRAIN_GROUP = np.array([1, 0, 0, 0, 0, 0], np.uint8)
# Defaults below follow Watch Your Step (arXiv 2610.29612, Sec. III) where it gives a number; every one can be
# overridden per Course / MapSpec / call, since the right values come out of running the pipeline.
MAX_STEP = 0.25          # m: largest step between neighbouring cells (paper's tallest training stair; CAD riser 0.186)
MAX_SLOPE = math.tan(math.radians(30))   # steepest walkable incline (ours: ramps 15 deg, braces steeper; paper: none)
MARGIN = 0.05            # m: safety band kept clear around obstacles (paper: none; one map cell)
CLIP = (-1.0, 1.0)       # m: relative heights clipped to this range (paper clips its height input to [-1, 1])
PATCH_K = 7              # foothold patch: k x k cells at the map resolution (paper: 7 x 7 at 5 cm, 0.35 m square)
SEED_IN = (30.0, 30.0)   # in: a walkable point, the route spawn on the lane deck
FREE, NEAR, OBSTACLE = 0, 1, 2   # cell classes; NEAR is the safety margin


def scene_path(scene: str):
    """A named apparatus scene, or any sim/scene_x2_<scene>.xml (the stair test scenes: flight6_109, single_186)."""
    return SCENES.get(scene) or os.path.join(SIM, 'scene_x2_%s.xml' % scene)


def load_scene(scene: str):
    """The scene's MjModel; with SCENE_CFG 'robot', that scene's X2 (body 'pelvis' and below) attached, parked."""
    robot = SCENE_CFG.get(scene, {}).get('robot')
    if not robot:
        return mujoco.MjModel.from_xml_path(scene_path(scene))
    s = mujoco.MjSpec.from_file(scene_path(scene))
    r = mujoco.MjSpec.from_file(scene_path(robot))
    r.option.timestep = s.option.timestep        # the scene's is kept (no attach-conflict warning)
    s.worldbody.add_frame(pos=[500.0, 500.0, 1.0]).attach_body(r.body('pelvis'), '', '')
    return s.compile()


class Course:
    """The apparatus as a height raster (res metres per cell), cached to build/hmap/<scene>_<res>.npz."""

    def __init__(self, scene: str = 'cad', res: float = 0.02, rebuild: bool = False,
                 max_step: float = MAX_STEP, max_slope: float = MAX_SLOPE, margin: float = MARGIN,
                 model: mujoco.MjModel | None = None, cfg: dict | None = None):
        """model / cfg: a generated scene (pipeline.terrain_park) instead of a named one; scene is then its cache name."""
        self.scene, self.res = scene, res
        self.max_step, self.max_slope, self.margin = max_step, max_slope, margin
        cfg = cfg if cfg is not None else SCENE_CFG.get(scene, {})
        self.m = model if model is not None else load_scene(scene)
        self.d = mujoco.MjData(self.m)
        self.group = cfg.get('group', 0)
        self.seed = cfg.get('seed')
        self.floor = cfg.get('floor', 0.0)
        self.lumber_h = cfg.get('lumber_h')
        self.thin = cfg.get('thin', False)
        self.flood = cfg.get('flood', True)
        for name, q in cfg.get('joints', {}).items():
            self.d.qpos[self.m.jnt_qposadr[self.m.joint(name).id]] = q
        self._park_robot()
        path = os.path.join(CACHE, '%s_%dmm.npz' % (scene, round(res * 1000)))
        if os.path.exists(path) and not rebuild:
            z = np.load(path)
            self.x0, self.y0, self.grid = float(z['x0']), float(z['y0']), z['grid']
        else:
            self._build()
            os.makedirs(CACHE, exist_ok=True)
            np.savez_compressed(path, x0=self.x0, y0=self.y0, grid=self.grid, res=res)
        self.nx, self.ny = self.grid.shape
        self.walkable = self._walkable()
        r = max(1, int(math.ceil(margin / res)))
        disk = np.hypot(*np.mgrid[-r:r + 1, -r:r + 1]) * res <= margin + 1e-9
        near = ndimage.binary_dilation(~self.walkable, structure=disk) if margin > 0 else ~self.walkable
        self.classes = np.where(~self.walkable, OBSTACLE, np.where(near, NEAR, FREE)).astype(np.int8)

    def _park_robot(self):
        """Move the robot far off the course so no ray hits it."""
        if mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, 'pelvis') < 0:   # a scene without the robot
            mujoco.mj_forward(self.m, self.d)
            self.robot_bodies = set()
            self._geom_bottoms()
            return
        pelvis = self.m.body('pelvis').id
        root = self.m.body_rootid[pelvis]
        jnt = self.m.body_jntadr[root]
        if jnt >= 0 and self.m.jnt_type[jnt] == mujoco.mjtJoint.mjJNT_FREE:
            a = self.m.jnt_qposadr[jnt]
            self.d.qpos[a:a + 3] = (500.0, 500.0, 1.0)
        mujoco.mj_forward(self.m, self.d)
        self.robot_bodies = {b for b in range(self.m.nbody) if self.m.body_rootid[b] == root}
        self._geom_bottoms()

    def _geom_bottoms(self):
        # world z of each geom's underside (from its bounding box): geometry whose underside is above OVERHEAD is
        # something the robot walks under (the door header), so the height map looks past it
        R = self.d.geom_xmat.reshape(-1, 3, 3)
        c, half = self.m.geom_aabb[:, :3], self.m.geom_aabb[:, 3:]
        self.geom_bottom = (self.d.geom_xpos[:, 2] + np.einsum('gk,gk->g', R[:, 2, :], c)
                            - np.einsum('gk,gk->g', np.abs(R[:, 2, :]), half))

    def _walkable(self):
        """Cells reachable from SEED_IN by 4-neighbour steps of at most max_step across cells no steeper than max_slope."""
        g = self.grid
        p = np.pad(g, 2, mode='edge')                # slope over 2 cells either side, so an edge chamfer reads flat
        sx = np.minimum(np.abs(p[4:, 2:-2] - g), np.abs(g - p[:-4, 2:-2]))
        sy = np.minimum(np.abs(p[2:-2, 4:] - g), np.abs(g - p[2:-2, :-4]))
        flat = np.hypot(sx, sy) <= self.max_slope * 2 * self.res
        if not self.flood:                           # generated terrain: every flat cell (stepping-pillar tops have no
            return flat                              # step-sized path from the floor)
        seed = self.seed or (SEED_IN[0] * IN, SEED_IN[1] * IN)
        i0, j0 = (round((seed[0] - self.x0) / self.res), round((seed[1] - self.y0) / self.res))
        if self.scene not in SCENES:                 # test scenes (stairs beside the course): start on the floor corner
            i0, j0 = 0, 0
        seen = np.zeros(g.shape, bool)
        seen[i0, j0] = True
        q = deque([(i0, j0)])
        while q:
            i, j = q.popleft()
            for a, b in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                if (0 <= a < self.nx and 0 <= b < self.ny and not seen[a, b] and flat[a, b]
                        and abs(g[a, b] - g[i, j]) <= self.max_step):
                    seen[a, b] = True
                    q.append((a, b))
        return seen

    def cls(self, xy):
        """Nearest-cell class (FREE, NEAR, OBSTACLE) at (N, 2) world points; FREE outside the raster."""
        xy = np.atleast_2d(np.asarray(xy, float))
        i = np.round((xy[:, 0] - self.x0) / self.res).astype(int)
        j = np.round((xy[:, 1] - self.y0) / self.res).astype(int)
        inside = (i >= 0) & (j >= 0) & (i < self.nx) & (j < self.ny)
        return np.where(inside, self.classes[np.clip(i, 0, self.nx - 1), np.clip(j, 0, self.ny - 1)], FREE)

    def bounds(self):
        """x0, x1, y0, y1 (m) of the apparatus geometry (no floor plane, no robot), padded 0.3 m."""
        lo, hi = np.full(2, np.inf), np.full(2, -np.inf)
        for g in range(self.m.ngeom):
            if self.m.geom_bodyid[g] in self.robot_bodies or self.m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
                continue
            if self.m.geom_group[g] != self.group:
                continue
            c, r = self.d.geom_xpos[g][:2], self.m.geom_rbound[g]
            lo, hi = np.minimum(lo, c - r), np.maximum(hi, c + r)
        return lo[0] - 0.3, hi[0] + 0.3, lo[1] - 0.3, hi[1] + 0.3

    def _ray_group(self):
        g = np.zeros(6, np.uint8)
        g[self.group] = 1
        return g

    def ray_heights(self, xy):
        """Exact heights at (N, 2) world points by one downward ray each."""
        xy = np.atleast_2d(np.asarray(xy, float))
        down = np.array([0.0, 0.0, -1.0])
        gid = np.zeros(1, np.int32)
        out = np.zeros(len(xy))
        p = np.array([0.0, 0.0, RAY_TOP])
        for k, (x, y) in enumerate(xy):              # mj_multiRay shares one origin, so one ray per point
            p[0], p[1], p[2] = x, y, RAY_TOP
            out[k] = 0.0
            for _ in range(4):                       # look past overhead geometry (door header) to the surface below
                dist = mujoco.mj_ray(self.m, self.d, p, down, self._ray_group(), 1, -1, gid)
                if dist < 0:
                    break
                g = int(gid[0])
                if self.geom_bottom[g] > OVERHEAD:
                    p[2] = self.geom_bottom[g] - 0.005
                    continue
                out[k] = p[2] - dist
                break
        return out

    def _build(self):
        x0, x1, y0, y1 = self.bounds()
        self.x0, self.y0 = x0, y0
        xs = x0 + self.res * np.arange(int(math.ceil((x1 - x0) / self.res)) + 1)
        ys = y0 + self.res * np.arange(int(math.ceil((y1 - y0) / self.res)) + 1)
        gx, gy = np.meshgrid(xs, ys, indexing='ij')
        self.grid = self.ray_heights(np.column_stack([gx.ravel(), gy.ravel()])).reshape(gx.shape).astype(np.float32)
        if self.thin:
            self._thin_walls()

    def _thin_walls(self):
        """Draw in geometry thinner than a cell across that the rays can miss (exact world extent: mesh vertices, box
        corners): rays down its centre line, the nearest cell to each gets at least that height. Left out: overhead
        geometry (underside above OVERHEAD, as for rays) and anything under THIN_TALL tall (deck plate edges)."""
        m, d = self.m, self.d
        unit = np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).reshape(3, -1).T
        for g in range(m.ngeom):
            if (m.geom_group[g] != self.group or m.geom_bodyid[g] in self.robot_bodies or
                    self.geom_bottom[g] > OVERHEAD):
                continue
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                k = m.geom_dataid[g]
                v = m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k] + m.mesh_vertnum[k]]
            elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
                v = unit * m.geom_size[g]
            else:
                continue
            w = d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T
            lo, hi = w.min(0), w.max(0)
            ext = hi[:2] - lo[:2]
            if ext.min() >= self.res or hi[2] - lo[2] < THIN_TALL:
                continue
            # rays down its centre line, every half cell along it: its real top (a wedge's side board slopes)
            a = int(np.argmax(ext))
            n = int(math.ceil(ext[a] / (self.res / 2))) + 1
            pts = np.tile((lo[:2] + hi[:2]) / 2, (n, 1))
            pts[:, a] = np.linspace(lo[a], hi[a], n)
            h = self.ray_heights(pts)
            i = np.round((pts[:, 0] - self.x0) / self.res).astype(int)
            j = np.round((pts[:, 1] - self.y0) / self.res).astype(int)
            ok = (i >= 0) & (j >= 0) & (i < self.grid.shape[0]) & (j < self.grid.shape[1])
            np.maximum.at(self.grid, (i[ok], j[ok]), h[ok].astype(np.float32))

    def height(self, xy):
        """Bilinear raster lookup at (N, 2) world points; 0 outside the raster (the venue floor)."""
        xy = np.atleast_2d(np.asarray(xy, float))
        fx = (xy[:, 0] - self.x0) / self.res
        fy = (xy[:, 1] - self.y0) / self.res
        inside = (fx >= 0) & (fy >= 0) & (fx <= self.nx - 1) & (fy <= self.ny - 1)
        i = np.clip(np.floor(fx).astype(int), 0, self.nx - 2)
        j = np.clip(np.floor(fy).astype(int), 0, self.ny - 2)
        u, v = np.clip(fx - i, 0, 1), np.clip(fy - j, 0, 1)
        g = self.grid
        h = (g[i, j] * (1 - u) * (1 - v) + g[i + 1, j] * u * (1 - v) + g[i, j + 1] * (1 - u) * v + g[i + 1, j + 1] * u * v)
        return np.where(inside, h, 0.0)

    def height_min(self, xy):
        """Lowest of the four raster cells around each (N, 2) point: no blending across an edge, so a point beside a
        riser reads the lower tread, not a ramp between the two. 0 outside the raster."""
        xy = np.atleast_2d(np.asarray(xy, float))
        fx = (xy[:, 0] - self.x0) / self.res
        fy = (xy[:, 1] - self.y0) / self.res
        inside = (fx >= 0) & (fy >= 0) & (fx <= self.nx - 1) & (fy <= self.ny - 1)
        i = np.clip(np.floor(fx).astype(int), 0, self.nx - 2)
        j = np.clip(np.floor(fy).astype(int), 0, self.ny - 2)
        g = self.grid
        h = np.minimum(np.minimum(g[i, j], g[i + 1, j]), np.minimum(g[i, j + 1], g[i + 1, j + 1]))
        return np.where(inside, h, 0.0)

    def extent(self):
        """x0, x1, y0, y1 of the raster (m)."""
        return self.x0, self.x0 + (self.nx - 1) * self.res, self.y0, self.y0 + (self.ny - 1) * self.res


@dataclass
class MapSpec:
    """Robot-centred grid: x forward, y left, in the robot's heading frame. Default 1.6 x 1.0 m at 5 cm (33 x 21), the
    paper's map; the 0.4 m back / 1.2 m front split is ours (the paper does not give one)."""
    back: float = 0.4       # m behind the robot
    front: float = 1.2      # m ahead
    half_width: float = 0.5  # m to each side
    res: float = 0.05       # m per cell

    def axes(self):
        xs = np.arange(-self.back, self.front + 1e-9, self.res)
        ys = np.arange(-self.half_width, self.half_width + 1e-9, self.res)
        return xs, ys


def map_points(x, y, yaw, spec: MapSpec = MapSpec()):
    """World (nx, ny, 2) points of the robot-centred grid."""
    xs, ys = spec.axes()
    gx, gy = np.meshgrid(xs, ys, indexing='ij')
    c, s = math.cos(yaw), math.sin(yaw)
    return np.stack([x + c * gx - s * gy, y + s * gx + c * gy], -1)


def local_map(course: Course, x, y, yaw, spec: MapSpec = MapSpec(), relative: bool = True, ref_z=None, clip=CLIP):
    """(nx, ny) heights around a robot at (x, y) facing yaw. Relative (default): metres above ref_z, the height of the
    robot's lowest foot in contact, so steps up are positive and steps down negative, clipped to clip (None: off).
    Without a robot, ref_z defaults to the ground under (x, y). relative=False gives world heights, unclipped."""
    pts = map_points(x, y, yaw, spec)
    h = course.height(pts.reshape(-1, 2)).reshape(pts.shape[:2])
    if relative:
        h = h - (course.height([[x, y]])[0] if ref_z is None else ref_z)
        if clip is not None:
            h = np.clip(h, *clip)
    return h


def local_classes(course: Course, x, y, yaw, spec: MapSpec = MapSpec()):
    """(nx, ny) int8 cell classes on the same grid as local_map: FREE, NEAR (safety margin) or OBSTACLE."""
    pts = map_points(x, y, yaw, spec)
    return course.cls(pts.reshape(-1, 2)).reshape(pts.shape[:2]).astype(np.int8)


def patch_points(x, y, yaw, foot_xy, spec: MapSpec = MapSpec(), k: int = PATCH_K):
    """World (k, k, 2) points of a k x k patch at the map resolution, centred on a foothold given in the robot frame
    (forward, left) and clamped so the patch stays inside the map, as the paper does."""
    xs, ys = spec.axes()
    half = (k - 1) / 2 * spec.res
    fx = float(np.clip(foot_xy[0], xs[0] + half, xs[-1] - half))
    fy = float(np.clip(foot_xy[1], ys[0] + half, ys[-1] - half))
    off = (np.arange(k) - (k - 1) / 2) * spec.res
    gx, gy = np.meshgrid(fx + off, fy + off, indexing='ij')
    c, s = math.cos(yaw), math.sin(yaw)
    return np.stack([x + c * gx - s * gy, y + s * gx + c * gy], -1)


def foothold_patch(course: Course, x, y, yaw, foot_xy, spec: MapSpec = MapSpec(), k: int = PATCH_K, ref_z=None,
                   clip=CLIP):
    """Heights (k, k), relative like local_map, and cell classes (k, k) around one foothold (robot frame)."""
    pts = patch_points(x, y, yaw, foot_xy, spec, k).reshape(-1, 2)
    h = course.height(pts) - (course.height([[x, y]])[0] if ref_z is None else ref_z)
    if clip is not None:
        h = np.clip(h, *clip)
    return h.reshape(k, k), course.cls(pts).reshape(k, k).astype(np.int8)


def route_waypoints(name: str):
    """(N, 2) metres from x2_route.ROUTES[name] (spawn, then every waypoint)."""
    import x2_route as XR
    r = XR.ROUTES[name]
    pts = [r['spawn'][:2]] + [(w[1], w[2]) for w in r['wps']]
    return np.array(pts, float) * IN


def resample_path(waypoints, ds: float = 0.1, turn_blend: float = 0.3):
    """(N, 3) x, y, yaw at ds spacing along the polyline. Heading follows the path tangent, blended over turn_blend m
    on each side of a corner so the robot turns into the next leg instead of snapping."""
    w = np.asarray(waypoints, float)
    seg = np.diff(w, axis=0)
    keep = np.linalg.norm(seg, axis=1) > 1e-6
    w = np.vstack([w[0], w[1:][keep]])
    seg = np.diff(w, axis=0)
    L = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(L)])
    s = np.arange(0.0, cum[-1] + 1e-9, ds)
    k = np.clip(np.searchsorted(cum, s, side='right') - 1, 0, len(L) - 1)
    t = (s - cum[k]) / L[k]
    xy = w[k] + seg[k] * t[:, None]
    ang = np.unwrap(np.arctan2(seg[:, 1], seg[:, 0]))
    yaw = ang[k].copy()
    for c in range(1, len(L)):                                   # blend headings around each corner
        near = np.abs(s - cum[c]) < turn_blend
        f = np.clip((s[near] - (cum[c] - turn_blend)) / (2 * turn_blend), 0, 1)
        yaw[near] = ang[c - 1] + (ang[c] - ang[c - 1]) * f
    return np.column_stack([xy, yaw])


def path_maps(course: Course, waypoints, ds: float = 0.1, spec: MapSpec = MapSpec()):
    """Poses (N, 3) along the path, the robot-centred height map at each (N, nx, ny) and its cell classes."""
    poses = resample_path(waypoints, ds)
    return (poses, np.stack([local_map(course, x, y, yaw, spec) for x, y, yaw in poses]),
            np.stack([local_classes(course, x, y, yaw, spec) for x, y, yaw in poses]))
