"""Solid physics floor (eval monitor, 2026-11-12, after the sim-artefact audit). DEFAULT since 2026-11-12 19:00 UTC (coordinator decision).

The old floor (feet on the 2 cm height field) is selected with PHYS_FLOOR=heightfield (controls only; results marked "old floor").
Solid floor: the feet collide with boxes instead of the 2 cm height field. The height field stays in the
scene for drawing, depth images and height scans (rays), but has no contact. Course geometry is unchanged:
- flat areas (deck, troughs, ground around the deck, stair treads, landing, ramp tops): the raster's cells merged into the fewest
  axis-aligned boxes of equal height (1 mm), each cell centred on its raster vertex, boxes reaching down below the lowest cell;
  vertical faces (risers, the deck edge) therefore sit half a cell from the raster's vertex, where its 2 cm ramp is half-way;
- ramps: one tilted box per ramp face, its top the least-squares plane of the face's cells; the cells under it are filled as flat
  floor at the face's low height so nothing is open below;
- 4x4s: boxes of the real section (0.089 m square) along the route's beam lines, their raster cells flattened first;
- the brace wedge near (3.4, 1.1) (a diagonal brace rasterised as a solid wedge, open in reality): its cells set to the floor;
- the landing deck plate (wall box 202 on our apparatus; generally: a thin wall box whose top equals the raster under it) is taken
  out of contact, since the raster's landing already carries the feet.
"""
import os
import math
import numpy as np

GROUP_PHYS = 0          # not drawn by eval/render.py or depth_views.py, not hit by the adapters' rays (they use terrain + wall groups)
BRACE_FIX = [(3.36, 3.46, 1.07, 1.17)]      # x0, x1, y0, y1 (m): the brace wedge cells (audit section 5, class d)
BEAM_HALF = 0.0445
TRIM = os.environ.get("PHYS_FLOOR_TRIM", "1") != "0"   # PHYS_FLOOR_TRIM=0: the untrimmed solid floor of 2026-11-12 19:00 (controls only)


def enabled():
    return os.environ.get("PHYS_FLOOR", "solid") != "heightfield"


def note():
    return (" | solid floor (default since 2026-11-12): boxes for the floor, steps, ramps and 4x4s; height field drawn and scanned only;"
            " brace wedge removed; landing plate out of contact") if enabled() else " | OLD FLOOR (PHYS_FLOOR=heightfield): feet on the 2 cm height field"


def _ring_median(z, mask, XX, YY, x0, x1, y0, y1, pad=0.08):
    ring = (XX > x0 - pad) & (XX < x1 + pad) & (YY > y0 - pad) & (YY < y1 + pad) & ~mask
    return float(np.median(z[ring]))


def plan(app):
    """-> dict(boxes=[(center, half, quat)], ramps=[...], plate_boxes=[wall indices], stats). Pure numpy, no MuJoCo."""
    from scipy import ndimage
    z = app.grid.astype(float).copy()
    res = app.res
    xs, ys = app.x0 + res * np.arange(app.nx), app.y0 + res * np.arange(app.ny)
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    # 4x4s: flatten their cells
    beams = app.route.get("beam_lines") or app.route.get("beams") or []
    beam_boxes = []
    for b in beams:
        a_, b_ = np.array(b["a"], float), np.array(b["b"], float)
        ab = b_ - a_
        tt = np.clip(((XX - a_[0]) * ab[0] + (YY - a_[1]) * ab[1]) / (ab @ ab), 0, 1)
        dist = np.hypot(XX - (a_[0] + tt * ab[0]), YY - (a_[1] + tt * ab[1]))
        near = dist < 0.07
        base = float(np.median(z[(dist > 0.10) & (dist < 0.2)]))
        z[near] = base
        c = (a_ + b_) / 2
        L = float(np.linalg.norm(ab))
        ang = math.atan2(ab[1], ab[0])
        beam_boxes.append(([c[0], c[1], base + BEAM_HALF], [L / 2, BEAM_HALF, BEAM_HALF], [math.cos(ang / 2), 0, 0, math.sin(ang / 2)]))
    # brace wedge
    for x0, x1, y0, y1 in BRACE_FIX:
        m = (XX >= x0) & (XX <= x1) & (YY >= y0) & (YY <= y1)
        if m.any():
            z[m] = _ring_median(z, m, XX, YY, x0, x1, y0, y1)
    # ramp faces: slope 10-20 deg, grouped by the direction of the slope
    gx, gy = np.gradient(z, res)
    s = np.degrees(np.arctan(np.hypot(gx, gy)))
    ang = np.degrees(np.arctan2(gy, gx))
    sloped = (s > 10) & (s < 20)
    ramps, ramp_cells = [], np.zeros_like(sloped)
    for lo in range(-180, 180, 90):
        dm = sloped & (((ang - lo + 45) % 360) < 90)
        lab, n = ndimage.label(dm)
        for k in range(1, n + 1):
            m = lab == k
            if m.sum() < 300:
                continue
            core = ndimage.binary_erosion(m)
            A = np.c_[XX[core], YY[core], np.ones(core.sum())]
            p = np.linalg.lstsq(A, z[core], rcond=None)[0]                 # z = p0 x + p1 y + p2
            full = ndimage.binary_dilation(m, iterations=1)
            # the face's extent: cells of the dilated face that lie within 6 mm of its plane, and inside the face's slope band
            zp = p[0] * XX + p[1] * YY + p[2]
            face = full & (np.abs(z - zp) < 0.006)
            u = np.array([p[0], p[1]]) / math.hypot(p[0], p[1])           # uphill, horizontal
            v = np.array([-u[1], u[0]])
            pu, pv = XX[face] * u[0] + YY[face] * u[1], XX[face] * v[0] + YY[face] * v[1]
            u0, u1 = pu.min() - res / 2, pu.max() + res / 2
            v0, v1 = pv.min() - res / 2, pv.max() + res / 2
            # sides fitted (coordinator 2026-11-12): +-res/2 past the face cells put some slab sides 1-2 cm outside the raster's edge;
            # each side now sits where the raster crosses half-way between the slab's plane and the floor beside it
            if TRIM:
                v0 = _fit_side(app, p, u, v, (u0, u1), v0, -1)
                v1 = _fit_side(app, p, u, v, (u0, u1), v1, +1)
                u0 = _fit_side(app, p, v, u, (v0, v1), u0, -1)              # the ends the same way (a drop beyond an end)
                u1 = _fit_side(app, p, v, u, (v0, v1), u1, +1)
            th = math.atan(math.hypot(p[0], p[1]))
            cu, cv = (u0 + u1) / 2, (v0 + v1) / 2
            cxy = cu * u + cv * v
            ztop = p[0] * cxy[0] + p[1] * cxy[1] + p[2]
            ex = np.array([u[0] * math.cos(th), u[1] * math.cos(th), math.sin(th)])
            ey = np.array([v[0], v[1], 0.0])
            ez = np.cross(ex, ey)
            if ez[2] < 0:
                ey, ez = -ey, -ez
            T = 0.03                                                        # half thickness of the slab
            ctr = np.array([cxy[0], cxy[1], ztop]) - ez * T
            R = np.stack([ex, ey, ez], 1)
            q = _mat2quat(R)
            ramps.append(([*ctr], [(u1 - u0) / 2 / math.cos(th), (v1 - v0) / 2, T], q, dict(cells=int(face.sum()), slope_deg=round(math.degrees(th), 2),
                          z_low=round(float(z[face].min()), 3), z_high=round(float(z[face].max()), 3))))
            low = float(z[face].min())
            z[face] = low                                                  # filled as flat floor under the slab
            ramp_cells |= face
    # flat boxes: greedy rectangles of equal height (1 mm)
    q = np.round(z, 3)
    zb = float(q.min()) - 0.05
    used = np.zeros(q.shape, bool)
    boxes = []
    nx, ny = q.shape
    for i in range(nx):
        j = 0
        while j < ny:
            if used[i, j]:
                j += 1
                continue
            h = q[i, j]
            j1 = j
            while j1 + 1 < ny and not used[i, j1 + 1] and q[i, j1 + 1] == h:
                j1 += 1
            i1 = i
            while i1 + 1 < nx and not used[i1 + 1, j:j1 + 1].any() and (q[i1 + 1, j:j1 + 1] == h).all():
                i1 += 1
            used[i:i1 + 1, j:j1 + 1] = True
            cx = (xs[i] + xs[i1]) / 2
            cy = (ys[j] + ys[j1]) / 2
            hx = (xs[i1] - xs[i]) / 2 + res / 2
            hy = (ys[j1] - ys[j]) / 2 + res / 2
            if h - zb > 1e-4:
                boxes.append(([cx, cy, (h + zb) / 2], [hx, hy, (h - zb) / 2], [1.0, 0, 0, 0]))
            j = j1 + 1
    plates = []
    for k, b in enumerate(app.boxes):
        top = b["center"][2] + b["half"][2]
        if b["half"][2] < 0.05 and b["quat"][0] > 0.999:
            g = app.height(b["center"][0], b["center"][1])
            if abs(float(g) - top) < 0.01 and b["half"][0] > 0.2 and b["half"][1] > 0.2:
                plates.append(k)
    return dict(boxes=boxes, ramps=ramps, beams=beam_boxes, plates=plates,
                stats=dict(flat_boxes=len(boxes), ramp_faces=len(ramps), beams=len(beam_boxes), plates=plates,
                           ramps=[r[3] for r in ramps]))


def _raster(app, px, py):
    from scipy import ndimage
    return ndimage.map_coordinates(app.grid.astype(float), [(np.asarray(px) - app.x0) / app.res, (np.asarray(py) - app.y0) / app.res],
                                   order=1, mode="nearest")


def _fit_side(app, p, u, v, urange, vside, sg):
    """The slab side's v (outward sign sg): the median, over 20 points along the side, of where the raster crosses half-way between
    the slab's plane and the floor 1.5 cells outside. Unchanged where there is no step (< 2 cm) or fewer than 5 crossings."""
    res = app.res
    us = np.linspace(urange[0] + 0.05, urange[1] - 0.05, 20)
    offs = np.arange(-3 * res, res + 1e-9, 0.001)                 # up to 3 cells inward (the face mask can leak into a neighbour)
    found = []
    for uu in us:
        pts = uu * u[None, :] + (vside + sg * offs)[:, None] * v[None, :]
        zr = _raster(app, pts[:, 0], pts[:, 1])
        q = uu * u + (vside - sg * res) * v
        hp = p[0] * q[0] + p[1] * q[1] + p[2]
        o = uu * u + (vside + sg * 1.5 * res) * v
        ho = float(_raster(app, [o[0]], [o[1]])[0])
        if abs(hp - ho) < 0.02:
            continue
        sgn = np.sign(zr - (hp + ho) / 2)
        k = np.where(sgn[:-1] != sgn[1:])[0]
        if len(k):
            found.append(offs[k[-1]])                              # the outermost crossing
    if len(found) < 5:
        return vside
    return vside + sg * float(np.median(found))


def _mat2quat(R):
    import mujoco
    qq = np.zeros(4)
    mujoco.mju_mat2Quat(qq, np.asarray(R, float).ravel())
    return [float(v) for v in qq]


def add_to_spec(spec, app, friction, material="deck"):
    """Adds the boxes to an MjSpec; returns (plan, set of wall indices to take out of contact)."""
    import mujoco
    P = plan(app)
    n = 0
    for kind, items in (("flat", P["boxes"]), ("ramp", [r[:3] for r in P["ramps"]]), ("beam", P["beams"])):
        for c, h, q in items:
            g = spec.worldbody.add_geom(name="phys_%s_%d" % (kind, n), type=mujoco.mjtGeom.mjGEOM_BOX, size=list(h), pos=list(c), quat=list(q),
                                        material=material)
            g.group, g.contype, g.conaffinity = GROUP_PHYS, 1, 1
            g.friction = [friction, 0.005, 0.0001]
            n += 1
    return P, set(P["plates"])
