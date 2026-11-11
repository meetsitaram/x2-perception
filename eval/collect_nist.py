"""Desktop-only, one-off: derive the NIST walk raster and wall boxes the same way ours were made.

Ours (loco-manip-apparatus sim/pipeline): cad_20mm.npz = top-surface raster; cad_20mm_walk.npz = the same with tall
obstacle cells (class OBSTACLE and taller than TALL_Z) set to the floor; cad_walls.json = the tall geoms as boxes
(pipeline.walls.tall_boxes). This script first checks that rule reproduces ours, then applies it to the NIST scene
(build/nist/scene_x2_nist_norobot.xml, door open at -1.65 rad as in the mobility trials).

    python eval/collect_nist.py OUT_DIR        (CPU, about a minute, memory-capped)
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import memcap
memcap.guard(cap_gb=8, need_free_gb=16)
import numpy as np
LMA = 'C:/Users/PC/projects/loco-manip-apparatus'
sys.path.insert(0, LMA + '/sim')
os.chdir(LMA + '/sim')
from pipeline import hmap as H
from pipeline import walls as W

out = sys.argv[1]
NEAR = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0

def walk_raster(c):
    tall = (c.classes == H.OBSTACLE) & (c.grid > c.floor + W.TALL_Z)
    g = c.grid.copy(); g[tall] = c.floor
    return g, tall

if '--skip-cad' not in sys.argv:
    c = H.Course('cad')
    g, tall = walk_raster(c)
    ref = np.load(LMA + '/build/hmap/cad_20mm_walk.npz')['grid']
    d = np.abs(g - ref)
    print('cad check: tall cells %d, cells differing from cad_20mm_walk %d (max %.3f m)' % (tall.sum(), (d > 1e-6).sum(), d.max()))
    print('  ref values at differing cells:', np.unique(np.round(ref[d > 1e-6], 3))[:20], ' mine:', np.unique(np.round(g[d > 1e-6], 3))[:20])

c = H.Course('nist')
print('nist: grid', c.grid.shape, 'floor', c.floor, 'x0 y0', c.x0, c.y0)
g, tall = walk_raster(c)
route = np.array(json.load(open(sys.argv[3]))) if len(sys.argv) > 3 and not sys.argv[3].startswith('--') else None
assert route is not None
# pipeline.walls.tall_boxes takes each geom's box in MuJoCo's geom frame. That is exact for box geoms, but the NIST
# collision pieces are mostly meshes (thin triangles of panels), whose frame is the triangle's inertia frame: its box is
# rotated in the panel's plane and covers far more than the panel (it closed the doorway and put the landing deck over the
# floor beside the stairs). So for meshes the box is fitted to the vertices in the world: the smallest box over rotations
# about one world axis (z, x or y; 1 deg steps), which is the panel itself for panels, rails and braces.
import mujoco
m, d = c.m, c.d
TZ = c.floor + W.TALL_Z


def rot(axis, a):
    ca, sa = np.cos(a), np.sin(a)
    R = np.eye(3); i, j = [(0, 1), (1, 2), (2, 0)][axis]
    R[i, i], R[i, j], R[j, i], R[j, j] = ca, -sa, sa, ca
    return R


CAND = [rot(ax, np.radians(a)) for ax in (0, 1, 2) for a in range(0, 90)]     # box axes repeat every 90 deg


def fit_box(V):
    best = None
    for R in CAND:
        P = V @ R                       # coordinates in the candidate frame (columns of R = box axes)
        lo, hi = P.min(0), P.max(0)
        vol = np.prod(np.maximum(hi - lo, 2e-3))
        if best is None or vol < best[0] - 1e-12:
            best = (vol, R, lo, hi)
    _, R, lo, hi = best
    return R @ ((lo + hi) / 2), np.maximum((hi - lo) / 2, 1e-3), R


walk = g
tall = (c.classes == H.OBSTACLE) & (c.grid > TZ)
path = np.concatenate([np.linspace(a, b, max(2, int(np.linalg.norm(b - a) / 0.02))) for a, b in zip(route[:-1], route[1:])])
boxes, types, seen = [], {}, set()
for g in range(m.ngeom):
    root = m.body_rootid[m.geom_bodyid[g]]
    if (m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE or root in c.robot_bodies or m.body_jntnum[root]
            or not (m.geom_contype[g] or m.geom_conaffinity[g]) or m.geom_group[g] != c.group):
        continue
    Rg = d.geom_xmat[g].reshape(3, 3)
    if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
        mid = m.geom_dataid[g]
        V = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]] @ Rg.T + d.geom_xpos[g]
        center, half, R = fit_box(V)
    else:
        center, half, R = d.geom_xpos[g] + Rg @ m.geom_aabb[g, :3], m.geom_aabb[g, 3:].copy(), Rg
    ext = np.abs(R) @ half
    lo, hi = center - ext, center + ext
    if hi[2] < TZ or lo[2] > c.floor + H.OVERHEAD:
        continue
    i0, i1 = (int(np.clip(round((v - c.x0) / c.res), 0, c.nx - 1)) for v in (lo[0], hi[0]))
    j0, j1 = (int(np.clip(round((v - c.y0) / c.res), 0, c.ny - 1)) for v in (lo[1], hi[1]))
    if not tall[i0:i1 + 1, j0:j1 + 1].any():
        continue
    gap = np.hypot(np.maximum(0, np.maximum(lo[0] - path[:, 0], path[:, 0] - hi[0])),
                   np.maximum(0, np.maximum(lo[1] - path[:, 1], path[:, 1] - hi[1]))).min()
    if gap > NEAR:
        continue
    key = tuple(np.round(np.r_[lo, hi], 2))             # the CAD has both faces of a sheet as separate pieces
    if key in seen:
        continue
    seen.add(key)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, R.ravel())
    t = mujoco.mjtGeom(int(m.geom_type[g])).name
    types[t] = types.get(t, 0) + 1
    boxes.append(dict(center=[round(float(x), 5) for x in center], half=[round(float(x), 5) for x in half],
                      quat=[round(float(x), 6) for x in quat], geom=int(g), type=t))
print('nist: tall cells', int(tall.sum()), 'boxes', len(boxes), 'by geom type', types)
np.savez_compressed(os.path.join(out, 'nist_20mm.npz'), x0=c.x0, y0=c.y0, grid=c.grid, res=c.res)
np.savez_compressed(os.path.join(out, 'nist_20mm_walk.npz'), x0=c.x0, y0=c.y0, grid=walk.astype(np.float32), res=c.res)
np.savez_compressed(os.path.join(out, 'nist_20mm_classes.npz'), x0=c.x0, y0=c.y0, classes=c.classes, res=c.res)
json.dump(dict(source='build/nist/scene_x2_nist_norobot.xml (door joint -1.65 rad = open)', tall_z=TZ, near=NEAR, floor=c.floor, boxes=boxes),
          open(os.path.join(out, 'nist_walls.json'), 'w'), indent=0)
print('written to', out)
