"""Chest depth-camera mount for the AgiBot X2 Ultra (Hiking and SSR policies). Builds the 3D-printable brackets, checks them,
and draws the pictures in this folder.

    python build_mount.py [--mesh path/to/torso_plus_link.STL] [--mjcf path/to/x2_ultra.xml] [--no-render]

Inputs
  * camera poses: the trained poses, copied from the released adapter cards (see POSES below);
  * chest shell: the X2 Ultra torso_link visual mesh (torso_plus_link.STL, md5 f080a9b2f4eec4aa46f2a5cb400d7733, the mesh the
    SSR MJCF uses). It is ray-cast once into shell_heightmap.npz (kept in this folder), so the brackets can be rebuilt
    without the mesh;
  * camera: Intel RealSense D435 / D435i, values from Intel's D400 Series datasheet 337029-005 (see README.md).

Everything is built in the torso_link frame (x forward, y left, z up), in millimetres. Libraries: numpy, trimesh, manifold3d,
matplotlib, mujoco (renders only).
"""
import argparse, json, os
import numpy as np
import trimesh
import manifold3d as mf

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- trained camera poses (torso_link frame)
POSES = {
    # release/hf/x2-hiking-in-the-wild/adapter_card.json "depth" and env.yaml scene.camera.offset (Isaac, convention world):
    # pos (0.131, 0, 0.259) m, quat wxyz (0.85896, 0, 0.51204, 0) = 61.6 deg pitch down about y.
    "hiking": dict(p=(131.0, 0.0, 259.0), pitch=61.6, src="x2-hiking-in-the-wild adapter_card.json depth.pos_in_body / quat_wxyz"),
    # release/hf/x2-ssr/adapter_card.json observation.depth and models/ssr/code/ssr/x2.py CAMERAS["chest_rs"]:
    # pos (0.126, 0, 0.16) m, 59.5 deg pitch down.
    # At exactly this pose the camera's back-bottom edge would sit 3.1 mm inside the chest shell (the trained pose was never
    # checked against the camera body), so the mount puts the camera 5 mm further forward (x 131 mm): inside the +-20 mm
    # forward mount randomisation SSR trained with (dr_pos 0.02 m).
    # The camera sits in front of the detachable lower chest panel, so the bracket reaches up to the fixed shell above the panel
    # seam with two rails that stay clear of the panel and of the sensor window (2026-11-15).
    "ssr": dict(p=(126.0, 0.0, 160.0), pitch=59.5, src="x2-ssr adapter_card.json observation.depth / ssr/x2.py chest_rs",
                build_shift=(5.0, 0.0, 0.0), cradle_y=(-50.0, 40.0), rails=((-50.0, -23.0), (23.0, 40.0))),
}
# Both pads bond only to the fixed upper chest shell above the seam of the detachable lower chest panel (user, 2026-11-15:
# "that part of the x2 body is detachable so the part might not stick well"). The seam is found on the mesh (panel_seam()).
PAD_Y = (-60.0, 45.0)    # pad width (robot y)
PAD_TOP = 292.0          # pad upper edge (the shell rounds over to the shoulder top at z ~ 300 on the centreline)
SEAM_MARGIN = 4.0        # pad lower edge this far above the seam's upper lip (the gap itself is ~3.5 mm on the centreline)
CLEAR_PANEL = 2.5        # every bracket surface off the bond area stays this far from the shell (>= 2 mm from the panel)
RAIL_T = 8.0             # SSR rail thickness (normal to the chest)

# ---------------------------------------------------------------- Intel RealSense D435 / D435i (datasheet 337029-005)
CAM_W, CAM_H, CAM_D = 90.0, 25.0, 25.0   # Table 3-44: width 90, height 25, depth 25 mm (drawing: height 25.05)
ORIGIN_BEHIND_GLASS = 4.2                # Table 4-13: depth start point 4.2 mm behind the front cover glass
ORIGIN_FROM_BACK = CAM_D - ORIGIN_BEHIND_GLASS   # 20.8 mm in front of the back face
IMAGER_Y = 17.5                          # Table 4-15: left imager 17.5 mm from the 1/4-20 centreline (= case centre)
M3_SPACING = 45.0                        # Figure 10-8: 2x M3 mounting points on the back, 45 mm apart, max insertion 3 mm
HFOV, VFOV = 87.0, 58.0                  # depth field of view used in training (datasheet: 87 x 58 deg +-3)

# ---------------------------------------------------------------- bracket parameters (mm)
TAPE = 1.1          # 3M VHB 5952 thickness
PAD_T = 3.0         # minimum pad thickness above the tape
CRADLE_Y = (-47.5, 22.5)   # cradle width (robot y), both ends of the camera stay free (case spans -62.5 .. 27.5)
CLEAR = 0.3         # clearance around the camera body
WALL = 3.2          # back wall thickness (M3x6 screw: 3.2 wall + 2.8 into the camera, <= 3 mm allowed)
FLOOR = 4.0         # floor (under the camera) thickness
FLOOR_FRONT = -1.5  # floor ends 1.5 mm behind the optical centre (5.7 mm behind the glass): nothing in the field of view
WALL_TOP = 10.0     # back wall reaches 10 mm above the camera's mid-height
M3_HOLE, M3_HEAD = 3.4, 6.5
# sensor window on the chest (measured on the mesh): capsule |y| <= 20, z 139 .. 225 mm; the pad bridges it with a pocket
WIN_HALF_W, WIN_Z = 20.0, (139.0, 225.0)
WIN_MARGIN, WIN_BRIDGE = 1.5, 108.5   # pocket = window + 1.5 mm; bridge level x = 108.5 mm (window rim is 106.5 .. 108.1)
NOTCH = 3.0

# ---------------------------------------------------------------- fall-protection cage (2026-11-15, user: "add some protect on
# top of the camera ... the robot will fall to the ground"). A box frame around the camera, open at the front: roof over the
# camera's top face, a cheek at each end, the cradle floor and wall widened to tie them together, and a front rim (the front
# ends of roof, cheeks and floor) standing proud of the cover glass. The cage touches the camera only where the cradle always
# did (floor and back wall); the roof and cheeks keep an air gap, so a blow on them goes into the wall, the web and the pad,
# not into the camera body or its two M3 screws.
TOP_GAP, ROOF_T = 3.0, 5.5     # air gap over the camera top, roof thickness: roof stands 8.5 mm proud of the top face
END_GAP, CHEEK_T = 2.0, 7.0    # air gap at each camera end, cheek thickness: cheeks stand 9 mm proud of the end faces
FRONT_PROUD = 8.5              # front rim plane this far ahead of the cover glass
CAGE_FLOOR_EXTRA = 1.5         # the cage's floor is 1.5 mm thicker than the cradle floor, so its rounded front edge still stands 8 mm proud
USB_WIN = (16.0, 12.0)         # window in each cheek for the USB-C plug (along the axis x camera height), centred on the end face
EAR_L, EAR_H, EAR_OUT, EAR_HOLE = 12.0, 11.0, 45.0, 4.0   # tether ear on the left cheek's back-top corner, out to y = +45 mm
TPU_T, TPU_SKIRT = 3.0, 6.0    # optional TPU bumper cap over the front rim: 3 mm thick, skirts 6 mm back over the outer faces
CUT_EXTRA = 1.0                # the view cones are cut out of the parts 1 deg wider than the checked cones below
# view cones that no bracket point may enter: name, apex span in camera y (mm, left imager = 0, right imager = -50),
# aperture half-size (mm), horizontal and vertical half-angles (deg). Apexes at the depth origin plane (4.2 mm behind the glass).
D435_IMAGER_FOV = (91.2, 65.5)   # D430 module IR imager FOV, datasheet 337029-005 Table 4-4 (as read; +-3 deg tolerance added)
D435_PROJ_FOV = (90.0, 63.0)     # IR projector FOV (as read from the same datasheet; +-3 deg added; position not dimensioned)
D435_RGB_FOV = (69.4, 42.5)      # RGB sensor FOV (not used by the policies; checked so the rim does not vignette it)
CONES = [
    ("depth frustum (87 x 58) + 4 deg training margin", (0.0, 0.0), 0.0, HFOV / 2 + 4, VFOV / 2 + 4),
    ("left IR imager", (0.0, 0.0), 1.0, D435_IMAGER_FOV[0] / 2 + 3, D435_IMAGER_FOV[1] / 2 + 3),
    ("right IR imager", (-50.0, -50.0), 1.0, D435_IMAGER_FOV[0] / 2 + 3, D435_IMAGER_FOV[1] / 2 + 3),
    ("IR projector (anywhere between the imagers)", (-50.0, 0.0), 3.0, D435_PROJ_FOV[0] / 2 + 3, D435_PROJ_FOV[1] / 2 + 3),
    ("RGB (at +15 mm; at -15 mm it is inside the imager cones)", (15.0, 15.0), 1.0, D435_RGB_FOV[0] / 2 + 3, D435_RGB_FOV[1] / 2 + 3),
]


def cam_axes(pitch_deg):
    t = np.deg2rad(pitch_deg)
    a = np.array([np.cos(t), 0.0, -np.sin(t)])      # optical axis (camera x, forward)
    ey = np.array([0.0, 1.0, 0.0])                  # camera y (left)
    u = np.array([np.sin(t), 0.0, np.cos(t)])       # camera z (up)
    return np.stack([a, ey, u], 1)                  # columns: local (da, dy, du) -> torso


# ---------------------------------------------------------------- chest shell height map x(y, z)
def shell_heightmap(mesh_path):
    cache = os.path.join(HERE, "shell_heightmap.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        return d["ys"], d["zs"], d["X"]
    m = trimesh.load(mesh_path)
    m.apply_scale(1000.0)
    ys = np.arange(-80.0, 80.01, 1.0); zs = np.arange(80.0, 320.01, 1.0)
    Y, Z = np.meshgrid(ys, zs)
    O = np.stack([np.full(Y.size, 500.0), Y.ravel(), Z.ravel()], 1)
    loc, ir, _ = m.ray.intersects_location(O, np.tile([-1.0, 0, 0], (Y.size, 1)), multiple_hits=False)
    X = np.full(Y.size, np.nan); X[ir] = loc[:, 0]; X = X.reshape(Y.shape)
    # the panel seam at z ~ 248 mm lets rays through to the back shell: interpolate those cells along z
    for j in range(X.shape[1]):
        col = X[:, j]; bad = ~(col > 40.0); good = ~bad
        if good.sum() > 2:
            inside = (zs > zs[good].min()) & (zs < zs[good].max())
            fix = bad & inside
            col[fix] = np.interp(zs[fix], zs[good], col[good])
    np.savez_compressed(cache, ys=ys, zs=zs, X=X.astype(np.float32),
                        note="front surface x (mm) of torso_plus_link.STL at (y, z), torso_link frame; NaN = no shell")
    return ys, zs, X


def panel_seam(mesh_path, ys):
    """Seam between the detachable lower chest panel and the fixed upper shell, per y (mm): (seam_lo, seam_hi) = upper edge
    of the lower panel's front face and lower edge of the upper shell's front face. In torso_plus_link.STL the two are
    separate connected pieces (lower: z 38-247 mm, upper: z 139-374 mm) with a gap between them that lets rays through.
    Cached in shell_heightmap.npz (seam_lo, seam_hi) so the brackets can be rebuilt without the mesh."""
    cache = os.path.join(HERE, "shell_heightmap.npz")
    d = dict(np.load(cache))
    if "seam_hi" in d:
        return d["seam_lo"], d["seam_hi"]
    from trimesh.graph import connected_component_labels
    m = trimesh.load(mesh_path)
    m.apply_scale(1000.0)
    lab = connected_component_labels(m.face_adjacency, node_count=len(m.faces))
    zz = np.arange(100.0, 300.01, 0.25)
    Y, Z = np.meshgrid(ys, zz)
    O = np.stack([np.full(Y.size, 500.0), Y.ravel(), Z.ravel()], 1)
    loc, ir, it = m.ray.intersects_location(O, np.tile([-1.0, 0, 0], (Y.size, 1)), multiple_hits=False)
    Xs = np.full(Y.size, np.nan); Xs[ir] = loc[:, 0]; Xs = Xs.reshape(Y.shape)
    C = np.full(Y.size, -1); C[ir] = lab[it]; C = C.reshape(Y.shape)
    j0 = int(np.argmin(abs(ys)))
    low = C[np.argmin(abs(zz - 120.0)), j0]; up = C[np.argmin(abs(zz - 275.0)), j0]   # lower panel / upper shell pieces
    front = Xs > 40.0
    lo = np.full(len(ys), np.nan); hi = np.full(len(ys), np.nan)
    for j in range(len(ys)):
        a = zz[(C[:, j] == low) & front[:, j]]
        if len(a):
            lo[j] = a.max()
        b = zz[(C[:, j] == up) & front[:, j] & (zz > (lo[j] if len(a) else 0.0))]
        if len(b):
            hi[j] = b.min()
    d.update(seam_lo=lo.astype(np.float32), seam_hi=hi.astype(np.float32),
             seam_note="per ys: top of the lower (detachable) chest panel's front face, bottom of the fixed upper shell's front face, mm")
    np.savez_compressed(cache, **d)
    return lo, hi


def interp_shell(ys, zs, X, y, z):
    from scipy.interpolate import RegularGridInterpolator
    f = RegularGridInterpolator((zs, ys), np.nan_to_num(X, nan=-50.0), bounds_error=False, fill_value=-50.0)
    return f(np.stack([z, y], -1))


def in_window(y, z, margin=0.0):
    r = WIN_HALF_W + margin
    zc = np.clip(z, WIN_Z[0] + WIN_HALF_W, WIN_Z[1] - WIN_HALF_W)
    return (y ** 2 + (z - zc) ** 2) <= r ** 2


# ---------------------------------------------------------------- mesh helpers
def to_manifold(tm):
    tm = trimesh.Trimesh(tm.vertices, tm.faces, process=True)
    tm.fix_normals()
    assert tm.is_watertight, "input solid not watertight"
    return mf.Manifold(mf.Mesh(vert_properties=np.asarray(tm.vertices, np.float32), tri_verts=np.asarray(tm.faces, np.uint32)))


def to_trimesh(man):
    m = man.to_mesh()
    return trimesh.Trimesh(np.asarray(m.vert_properties)[:, :3], np.asarray(m.tri_verts), process=True)


def heightfield_solid(ys, zs, xlo, xhi):
    """Closed solid between x = xlo(z, y) and x = xhi(z, y) over the grid (arrays shaped (nz, ny))."""
    nz, ny = xlo.shape
    Y, Z = np.meshgrid(ys, zs)
    V = np.concatenate([np.stack([xhi.ravel(), Y.ravel(), Z.ravel()], 1), np.stack([xlo.ravel(), Y.ravel(), Z.ravel()], 1)])
    off = nz * ny
    idx = np.arange(off).reshape(nz, ny)
    F = []
    a, b, c, d = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel(), idx[1:, :-1].ravel()
    F += [np.stack([a, b, c], 1), np.stack([a, c, d], 1)]                       # top
    F += [np.stack([a + off, c + off, b + off], 1), np.stack([a + off, d + off, c + off], 1)]   # bottom
    for e in (idx[0, :], idx[:, -1], idx[-1, ::-1], idx[::-1, 0]):            # side walls along the boundary
        p, q = e[:-1], e[1:]
        F += [np.stack([p, p + off, q], 1), np.stack([q, p + off, q + off], 1)]
    return to_manifold(trimesh.Trimesh(V, np.concatenate(F)))


def box_local(R, p, lo, hi):
    c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    return p + c @ R.T


ROUND = True            # round the outer edges (user, 2026-11-15: "can you smoothen the edges and corners?"); --sharp turns it off
R_OUT, R_RIM, R_SMALL, R_HULL, R_PAD = 2.5, 4.0, 1.0, 2.0, 1.0   # outer edges, cage front rim, small features, webs, pad perimeter


def sphere_pts(r, n=24):
    m = mf.Manifold.sphere(r, n).to_mesh()
    return np.asarray(m.vert_properties)[:, :3].astype(float)


def rounded_box(R, p, lo, hi, r_back, r_front):
    """Box in camera-local coordinates with all edges rounded: hull of spheres in its corners, radius r_front at the four
    corners of the front face (local +da) and r_back at the others. Exact outer dimensions."""
    if not ROUND:
        return mf.Manifold.hull_points(box_local(R, p, lo, hi).tolist())
    P = []
    for x in (lo[0], hi[0]):
        r = r_front if x == hi[0] else r_back
        for y in (lo[1], hi[1]):
            for z in (lo[2], hi[2]):
                c = np.array([x + (r if x == lo[0] else -r), y + (r if y == lo[1] else -r), z + (r if z == lo[2] else -r)])
                P.append(p + R @ c + sphere_pts(r, 32))
    return mf.Manifold.hull_points(np.concatenate(P).tolist())


def grown_hull(pts, r=R_HULL):
    """Convex hull of the points grown by a ball of radius r: the hull's edges and corners come out rounded with radius r."""
    pts = np.asarray(pts, float)
    if ROUND:
        pts = (pts[:, None, :] + sphere_pts(r, 12)[None]).reshape(-1, 3)
    return mf.Manifold.hull_points(pts.tolist())


def edge_drop(d, r):
    """How far to lower a surface at distance d inside an edge to round that edge with radius r (0 beyond r)."""
    if not ROUND:
        return np.zeros_like(d)
    d = np.clip(d, 0.0, r)
    return r - np.sqrt(np.maximum(r * r - (r - d) ** 2, 0.0))


def opening_cut(R, p, y0, y1, u0, u1, da_front, rf, nseg=5):
    """Rectangular opening (local dy y0..y1, du u0..u1) through a front face at da = da_front, its edge rounded with radius rf
    (stack of frusta approximating the fillet), plus everything in front of the face."""
    def rect(da, g):
        return [[da, y, u] for y in (y0 - g, y1 + g) for u in (u0 - g, u1 + g)]
    if not ROUND:
        return mf.Manifold.hull_points(box_local(R, p, [da_front - 30.0, y0, u0], [da_front + 80.0, y1, u1]).tolist())
    out = mf.Manifold.hull_points(box_local(R, p, [da_front - 0.01, y0 - rf, u0 - rf], [da_front + 80.0, y1 + rf, u1 + rf]).tolist())
    z = np.linspace(0.0, rf, nseg + 1)                              # depth behind the face
    g = rf - np.sqrt(np.maximum(rf * rf - (rf - z) ** 2, 0.0))      # growth of the opening at that depth (quarter circle)
    for i in range(nseg):
        P = np.array(rect(da_front - z[i], g[i]) + rect(da_front - z[i + 1], g[i + 1]))
        out = out + mf.Manifold.hull_points((p + P @ R.T).tolist())
    return out


def cone_solids(R, p, extra, L=150.0):
    """Union of the view cones of CONES (each widened by `extra` deg), as solids in torso_link coordinates."""
    out = None
    for _, (ylo, yhi), r, hh, hv in CONES:
        th, tv = np.tan(np.deg2rad(hh + extra)), np.tan(np.deg2rad(hv + extra))
        P = [[d, y, u] for d, sy, su in ((0.0, r, r), (L, r + L * th, r + L * tv)) for y in (ylo - sy, yhi + sy) for u in (-su, su)]
        s = mf.Manifold.hull_points((p + np.array(P) @ R.T).tolist())
        out = s if out is None else out + s
    return out


def fov_margins(P, R, p):
    """Smallest angular distance (deg) of the points P (torso_link, mm) outside each view cone of CONES (negative = inside)."""
    L = (np.asarray(P) - p) @ R
    res = {}
    for nm, (ylo, yhi), r, hh, hv in CONES:
        da = L[:, 0]; ok = da > 1e-6
        ly = np.maximum(0.0, np.maximum(ylo - r - L[:, 1], L[:, 1] - yhi - r))
        lu = np.maximum(0.0, np.abs(L[:, 2]) - r)
        m = np.maximum(np.degrees(np.arctan2(ly, da)) - hh, np.degrees(np.arctan2(lu, da)) - hv)
        res[nm] = round(float(m[ok].min()), 2) if ok.any() else None
    return res


def cyl_local(R, p, start_local, axis_local, length, radius, radius2=None):
    """Cylinder (or frustum, radius -> radius2) from start along axis (both in camera-local coordinates)."""
    cyl = mf.Manifold.cylinder(length, radius, radius if radius2 is None else radius2, 40)
    z = np.asarray(axis_local, float); z /= np.linalg.norm(z)
    x = np.cross(z, [0, 1, 0]) if abs(z[1]) < 0.9 else np.cross(z, [1, 0, 0]); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    Rl = np.stack([x, y, z], 1)
    Rw = R @ Rl
    t = p + R @ np.asarray(start_local, float)
    return cyl.transform(np.concatenate([Rw, t[:, None]], 1).tolist())


# ---------------------------------------------------------------- build one variant
def normal_factor(F, h):
    """x-offset per mm of normal offset for a height field x = F(z, y) on a grid of spacing h (clipped at 3, i.e. 70 deg)."""
    gz, gy = np.gradient(F, h)
    return np.clip(np.sqrt(1 + gy ** 2 + gz ** 2), 1.0, 3.0)


def pad_lower_edge(ys, seam_hi, y):
    """Pad lower edge z(y): the seam's upper lip (running max over +-2 mm) plus SEAM_MARGIN."""
    s = np.array([np.nanmax(seam_hi[max(0, j - 2):j + 3]) for j in range(len(seam_hi))])
    return np.interp(y, ys, s) + SEAM_MARGIN


def build(name, cfg, ys, zs, X, seam_hi):
    R = cam_axes(cfg["pitch"])
    p = np.array(cfg["p"], float) + np.array(cfg.get("build_shift", (0.0, 0.0, 0.0)))
    a, u = R[:, 0], R[:, 2]
    cy = -IMAGER_Y        # case centre in camera-local dy: left imager (depth origin) on the robot centreline, case 17.5 mm to the right
    # camera body in local coordinates
    cam_lo = np.array([-ORIGIN_FROM_BACK, cy - CAM_W / 2, -CAM_H / 2])
    cam_hi = np.array([ORIGIN_BEHIND_GLASS, cy + CAM_W / 2, CAM_H / 2])

    # check that the camera body itself clears the shell at the trained pose
    g = np.linspace(0, 1, 25)
    S = []
    for i in g:
        for j in g:
            for face in range(6):
                q = cam_lo + (cam_hi - cam_lo) * np.array([[i, j, 0], [i, j, 1], [i, 0, j], [i, 1, j], [0, i, j], [1, i, j]][face])
                S.append(q)
    Sw = p + np.array(S) @ R.T
    cam_clear = float(np.min(Sw[:, 0] - interp_shell(ys, zs, X, Sw[:, 1], Sw[:, 2])))

    # pad (conformal, above the tape): only on the fixed upper shell, lower edge SEAM_MARGIN above the panel seam
    pys = np.arange(PAD_Y[0], PAD_Y[1] + 0.01, 0.5)
    zlo = pad_lower_edge(ys, seam_hi, pys)
    pzs = np.arange(np.floor(zlo.min() * 2) / 2, PAD_TOP + 0.01, 0.5)
    PY, PZ = np.meshgrid(pys, pzs)
    shell = interp_shell(ys, zs, X, PY, PZ)
    f = normal_factor(shell, 0.5)      # offsets are normal to the shell (it tilts back up to ~45 deg near the pad's top)
    # the pad's outer face is rounded down by R_PAD along its perimeter; the taped face stays the shell offset by the tape
    dedge = np.minimum.reduce([PY - PAD_Y[0], PAD_Y[1] - PY, PAD_TOP - PZ, PZ - np.interp(PY, pys, zlo)])
    slab = heightfield_solid(pys, pzs, shell + TAPE * f, shell + (TAPE + PAD_T - edge_drop(dedge, R_PAD)) * f)
    outline = np.concatenate([np.stack([pys, zlo], 1), [[PAD_Y[1], PAD_TOP], [PAD_Y[0], PAD_TOP]]])
    prism = mf.CrossSection([outline]).extrude(300.0).transform([[0, 0, 1, -50.0], [1, 0, 0, 0], [0, 1, 0, 0]])
    pad = slab ^ prism
    bond = (PZ >= np.interp(PY, pys, zlo)) & (PZ <= PAD_TOP)
    bond_area = float((0.25 * f * bond).sum()) / 100.0     # cm2 on the curved shell (cell-centred, +-1 %)

    def in_bond(y, z, dil):
        return (y >= PAD_Y[0] - dil) & (y <= PAD_Y[1] + dil) & (z >= pad_lower_edge(ys, seam_hi, y) - dil) & (z <= PAD_TOP + dil)

    # chest keep-out (window bridged): shell + tape (normal) on the bond area; everywhere else the shell grown by a ball of
    # radius CLEAR_PANEL, so edges such as the panel's lip at the seam keep the clearance too
    h = 0.5
    kys = np.arange(-80.0, 80.01, h); kzs = np.arange(zs[0], zs[-1] + 0.01, h)
    KY, KZ = np.meshgrid(kys, kzs)
    ks = np.maximum(interp_shell(ys, zs, X, KY, KZ), -40.0)
    ks = np.where(in_window(KY, KZ, WIN_MARGIN), np.maximum(ks, WIN_BRIDGE), ks)
    r = CLEAR_PANEL; n = int(np.ceil(r / h)); kp = np.pad(ks, n, mode="edge"); ball = np.full_like(ks, -1e9)
    for di in range(-n, n + 1):
        for dj in range(-n, n + 1):
            dd = h * np.hypot(di, dj)
            if dd <= r:
                ball = np.maximum(ball, kp[n + di:n + di + ks.shape[0], n + dj:n + dj + ks.shape[1]] + np.sqrt(r * r - dd * dd))
    kx = np.where(in_bond(KY, KZ, 1.5), ks + TAPE * normal_factor(ks, h), ball)
    keep = heightfield_solid(kys, kzs, np.full_like(ks, -120.0), kx)

    # cradle: floor under the camera and back wall behind it
    y0, y1 = cfg.get("cradle_y", CRADLE_Y)
    # cage envelope (camera-local): wall at the back, floor below, roof above, a cheek at each end, front rim FRONT_PROUD ahead
    c_lo = np.array([-ORIGIN_FROM_BACK - CLEAR - WALL, cam_lo[1] - END_GAP - CHEEK_T, -CAM_H / 2 - CLEAR - FLOOR - CAGE_FLOOR_EXTRA])
    c_hi = np.array([ORIGIN_BEHIND_GLASS + FRONT_PROUD, cam_hi[1] + END_GAP + CHEEK_T, CAM_H / 2 + TOP_GAP + ROOF_T])
    floor_pts = box_local(R, p, [-ORIGIN_FROM_BACK - WALL, y0, -CAM_H / 2 - CLEAR - FLOOR], [FLOOR_FRONT, y1, -CAM_H / 2 - CLEAR])
    wall_lo = [-ORIGIN_FROM_BACK - CLEAR - WALL, y0, -CAM_H / 2 - CLEAR - FLOOR]
    wall_hi = [-ORIGIN_FROM_BACK - CLEAR, y1, c_hi[2]]          # the wall (and the web behind it) now reaches the roof
    wall_pts = box_local(R, p, wall_lo, wall_hi)
    top = np.stack([shell + (TAPE + PAD_T) * f, PY, PZ], -1)
    if "rails" not in cfg:
        # camera close under the pad (Hiking): one solid hull from the cradle to the pad's central part
        zc = float(pad_lower_edge(ys, seam_hi, np.arange(y0, y1 + 0.01, 0.5)).max())
        sel = (PY >= y0) & (PY <= y1) & (PZ >= zc)
        edge = sel & ~(np.roll(sel, 1, 0) & np.roll(sel, -1, 0) & np.roll(sel, 1, 1) & np.roll(sel, -1, 1))
        pad_top = np.concatenate([top[sel][::7], top[edge]])
        part = grown_hull(np.concatenate([floor_pts, wall_pts, pad_top])) + pad
    else:
        # camera far below the pad (SSR): cradle hull, two rails off the panel up to the pad, a gusset from each rail to the wall
        part = grown_hull(np.concatenate([floor_pts, wall_pts])) + pad
        wz = wall_pts[:, 2]
        for ry0, ry1 in cfg["rails"]:
            rys = np.arange(ry0, ry1 + 0.01, 0.5); rzs = np.arange(np.floor(wz.mean()), PAD_TOP + 0.01, 0.5)
            RY, RZ = np.meshgrid(rys, rzs)
            rs = interp_shell(ys, zs, X, RY, RZ)
            rs = np.where(in_window(RY, RZ, WIN_MARGIN), np.maximum(rs, WIN_BRIDGE), rs)
            rt = RAIL_T - edge_drop(np.minimum(RY - ry0, ry1 - RY), R_OUT)      # rail's two outer long edges rounded
            part = part + heightfield_solid(rys, rzs, rs + CLEAR_PANEL, rs + CLEAR_PANEL + rt)
            band = (RZ >= wz.max() - 5) & (RZ <= wz.max() + 30)
            gp = np.concatenate([np.stack([rs + CLEAR_PANEL + RAIL_T - R_HULL * ROUND, RY, RZ], -1)[band],
                                 np.stack([rs + CLEAR_PANEL, RY, RZ], -1)[band],
                                 box_local(R, p, [wall_lo[0], max(ry0, y0), wall_lo[2]], [wall_hi[0], min(ry1, y1), wall_hi[2]])])
            part = part + grown_hull(gp)
    # fall-protection cage: one rounded box (rim corners R_RIM, other edges R_OUT), hollowed by the camera pocket below
    part = part + rounded_box(R, p, c_lo, c_hi, R_OUT, R_RIM)
    # tether ear on the outside of the left cheek, at its back-top corner, with a hole along the camera's up axis
    ear_lo = [c_lo[0], c_hi[1] - 1.0, c_hi[2] - EAR_H]; ear_hi = [c_lo[0] + EAR_L, EAR_OUT, c_hi[2]]
    part = part + rounded_box(R, p, ear_lo, ear_hi, R_SMALL * 1.5, R_SMALL * 1.5)
    part = part - keep
    # camera pocket: 0.3 mm on the floor and the wall (the seat faces), END_GAP at the ends, TOP_GAP under the roof, open forward
    pk_lo = np.array([-ORIGIN_FROM_BACK - CLEAR, cam_lo[1] - END_GAP, -CAM_H / 2 - CLEAR])
    pk_hi = np.array([80.0, cam_hi[1] + END_GAP, CAM_H / 2 + TOP_GAP])
    part = part - mf.Manifold.hull_points(box_local(R, p, pk_lo, pk_hi).tolist())
    part = part - opening_cut(R, p, pk_lo[1], pk_hi[1], pk_lo[2], pk_hi[2], c_hi[0], R_SMALL * 1.5)   # rim's inner edge
    # USB-C windows through both cheeks (the port's end is not confirmed), rounded corners, 1 mm round on the outer face
    wc = -ORIGIN_FROM_BACK + CAM_D / 2; ww, wh = USB_WIN
    def rrect(yv, g, rr=2.0):     # rounded rectangle (corner radius rr) grown by g, in the plane local dy = yv
        th = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        return [[dd + (rr + g) * np.cos(t), yv, uu + (rr + g) * np.sin(t)]
                for dd in (wc - ww / 2 + rr, wc + ww / 2 - rr) for uu in (-wh / 2 + rr, wh / 2 - rr) for t in th]
    for s, yin, yout in ((-1, cam_lo[1] - END_GAP, c_lo[1]), (1, cam_hi[1] + END_GAP, c_hi[1])):
        pieces = [rrect(yin - s, 0.0) + rrect(yout + s * 2.0, 0.0)]
        if ROUND:   # 1 mm x 45 deg chamfer on the window's outer edge
            pieces.append(rrect(yout - s * R_SMALL, 0.0) + rrect(yout + s * 0.01, R_SMALL) + rrect(yout + s * 2.0, R_SMALL))
        for P in pieces:
            part = part - mf.Manifold.hull_points((p + np.array(P) @ R.T).tolist())
    # every view cone, 1 deg wider than checked, is cut out (this bevels the rim where a cone reaches it)
    cones = cone_solids(R, p, CUT_EXTRA)
    part = part - cones
    # M3 holes (1 mm chamfer at the seat face) and screw-head channels (screw from behind the wall)
    holes_y = [cy - M3_SPACING / 2, cy + M3_SPACING / 2]
    for hy in holes_y:
        part = part - cyl_local(R, p, [-ORIGIN_FROM_BACK + 1.0, hy, 0.0], [-1, 0, 0], WALL + CLEAR + 2.0, M3_HOLE / 2)
        if ROUND:
            part = part - cyl_local(R, p, [-ORIGIN_FROM_BACK - CLEAR + 0.01, hy, 0.0], [-1, 0, 0], R_SMALL, M3_HOLE / 2 + R_SMALL, M3_HOLE / 2)
        part = part - cyl_local(R, p, [-ORIGIN_FROM_BACK - CLEAR - WALL, hy, 0.0], [-1, 0, 0], 80.0, M3_HEAD / 2)
    # tether hole through the ear (4 mm), 0.5 mm chamfer at both ends
    ec = [c_lo[0] + EAR_L / 2, (c_hi[1] + EAR_OUT) / 2]
    part = part - cyl_local(R, p, [ec[0], ec[1], c_hi[2] - EAR_H - 5], [0, 0, 1], EAR_H + 10, EAR_HOLE / 2)
    if ROUND:
        for u0, sgn in ((c_hi[2] - EAR_H, 1), (c_hi[2], -1)):
            part = part - cyl_local(R, p, [ec[0], ec[1], u0 - sgn * 0.01], [0, 0, sgn], 0.6, EAR_HOLE / 2 + 0.6, EAR_HOLE / 2)
    # optional TPU bumper cap over the front rim: shell TPU_T thick around the rim, skirts TPU_SKIRT back over the outer faces
    tpu = None
    if True:
        o_lo = c_lo - [0.0, TPU_T, TPU_T]; o_hi = c_hi + [TPU_T, TPU_T, TPU_T]
        outer = rounded_box(R, p, o_lo, o_hi, R_OUT + TPU_T, R_RIM + TPU_T)
        cut_back = mf.Manifold.hull_points(box_local(R, p, [-200.0, -200.0, -200.0], [c_hi[0] - TPU_SKIRT, 200.0, 200.0]).tolist())
        tpu = outer - cut_back - rounded_box(R, p, c_lo - [5.0, 0, 0], c_hi, R_OUT, R_RIM)
        tpu = tpu - mf.Manifold.hull_points(box_local(R, p, pk_lo, pk_hi).tolist())
        tpu = tpu - opening_cut(R, p, pk_lo[1], pk_hi[1], pk_lo[2], pk_hi[2], o_hi[0], R_SMALL * 1.5) - cones
        tpu = tpu - keep
    cage = dict(c_lo=c_lo, c_hi=c_hi, pk_lo=pk_lo, pk_hi=pk_hi)
    # alignment notches on the robot centreline (left imager / sensor window centreline), lower and upper pad edges
    pz = (float(pad_lower_edge(ys, seam_hi, np.array([0.0]))[0]), PAD_TOP)
    for zc, sgn in ((pz[0], 1), (pz[1], -1)):
        tri = np.array([[-NOTCH, zc - sgn], [NOTCH, zc - sgn], [0.0, zc + sgn * NOTCH]])
        pts = [[x, yy, zz] for x in (60.0, 200.0) for yy, zz in tri]
        part = part - mf.Manifold.hull_points(pts)
    part = part.as_original()
    tm = to_trimesh(part)
    comps = tm.split(only_watertight=False)
    dropped = [[round(c.volume / 1000, 2), np.round(c.centroid, 0).tolist()] for c in sorted(comps, key=lambda c: -c.volume)[1:]
               if c.volume > 10.0]   # pieces > 0.01 cm3 cut off by the screw channels (degenerate zero-volume bits not listed)
    if len(comps) > 1:     # drop slivers, keep the main body
        tm = max(comps, key=lambda c: c.volume)

    tt = to_trimesh(tpu.as_original())
    tc = tt.split(only_watertight=False)
    tpu_dropped = [round(c.volume / 1000, 3) for c in sorted(tc, key=lambda c: -c.volume)[1:]]
    if len(tc) > 1:
        tt = max(tc, key=lambda c: c.volume)

    # checks: field of view (with the training's +-4 deg mount randomisation margin) and shell clearance
    L = (tm.vertices - p) @ R
    vis = L[:, 0] > 0
    ang_v = np.degrees(np.arctan2(np.abs(L[:, 2]), np.maximum(L[:, 0], 1e-9)))
    ang_h = np.degrees(np.arctan2(np.abs(L[:, 1]), np.maximum(L[:, 0], 1e-9)))
    in_fov = vis & (ang_v < VFOV / 2 + 4) & (ang_h < HFOV / 2 + 4)
    V = np.concatenate([tm.vertices, tm.sample(40000)])
    shell_gap = V[:, 0] - interp_shell(ys, zs, X, V[:, 1], V[:, 2])
    near = shell_gap < 30.0
    below = near & (V[:, 2] < np.interp(V[:, 1], ys, seam_hi))       # in front of the detachable panel (or its seam)
    offb = near & ~in_bond(V[:, 1], V[:, 2], 2.0)
    info = dict(
        variant=name, source=cfg["src"], trained_optical_centre_mm=list(cfg["p"]),
        mount_shift_from_trained_mm=list(cfg.get("build_shift", (0.0, 0.0, 0.0))),
        optical_centre_mm=list(p), pitch_down_deg=cfg["pitch"],
        case_centre_mm=list(np.round(p + R @ np.array([-(ORIGIN_FROM_BACK - CAM_D / 2), cy, 0.0]), 1)),
        back_face_centre_mm=list(np.round(p + R @ np.array([-ORIGIN_FROM_BACK, cy, 0.0]), 1)),
        m3_holes_mm=[list(np.round(p + R @ np.array([-ORIGIN_FROM_BACK, hy, 0.0]), 1)) for hy in holes_y],
        camera_min_clearance_to_shell_mm=round(cam_clear, 1),
        bracket_vertices_in_fov=int(in_fov.sum()),
        bracket_min_gap_to_shell_mm=round(float(shell_gap[near].min()), 2),
        bracket_min_gap_off_bond_area_mm=round(float(shell_gap[offb].min()), 2),
        bracket_min_gap_below_seam_mm=round(float(shell_gap[below].min()), 2) if below.any() else None,
        pad_y_mm=list(PAD_Y), pad_z_centreline_mm=[round(pz[0], 1), pz[1]],
        pad_lower_edge_z_mm={"y%+.0f" % yy: round(float(pad_lower_edge(ys, seam_hi, np.array([yy]))[0]), 1)
                              for yy in (PAD_Y[0], -40.0, -20.0, 0.0, 20.0, PAD_Y[1])},
        bonded_area_cm2=round(bond_area, 1), cradle_y_mm=[y0, y1], rails_y_mm=[list(r) for r in cfg.get("rails", [])],
        dropped_fragments_cm3=dropped,
        volume_cm3=round(tm.volume / 1000, 1), mass_g_petg=round(tm.volume / 1000 * 1.27, 0),
        bounds_mm=np.round(tm.bounds, 1).tolist(), watertight=bool(tm.is_watertight),
    )
    # shell point and normal under the optical centre (horizontal ray backwards) and along the optical axis backwards
    sx = float(interp_shell(ys, zs, X, np.array([0.0]), np.array([p[2]]))[0])
    h = 1.0
    dxdy = (interp_shell(ys, zs, X, np.array([h]), np.array([p[2]]))[0] - interp_shell(ys, zs, X, np.array([-h]), np.array([p[2]]))[0]) / (2 * h)
    dxdz = (interp_shell(ys, zs, X, np.array([0.0]), np.array([p[2] + h]))[0] - interp_shell(ys, zs, X, np.array([0.0]), np.array([p[2] - h]))[0]) / (2 * h)
    n = np.array([1.0, -dxdy, -dxdz]); n /= np.linalg.norm(n)
    yy = np.array([-30.0, 0.0, 30.0]); xs = interp_shell(ys, zs, X, yy, np.full(3, p[2]))
    sag = float(xs[1] - (xs[0] + xs[2]) / 2); rad = 30.0 ** 2 / (2 * sag) if sag > 0.05 else float("inf")
    info.update(shell_point_behind_centre_mm=[round(sx, 1), 0.0, p[2]], centre_ahead_of_shell_mm=round(p[0] - sx, 1),
                shell_normal=list(np.round(n, 3)), shell_tilt_back_deg=round(float(np.degrees(np.arctan2(n[2], n[0]))), 1),
                shell_lateral_sag_over_60mm=round(sag, 1), shell_lateral_radius_mm=round(rad, 0))
    info["cage"] = cage_checks(tm, tt, R, p, c_lo, c_hi, pk_lo, pk_hi, tpu_dropped)
    return tm, tt, info, R, p


def cage_checks(tm, tt, R, p, c_lo, c_hi, pk_lo, pk_hi, tpu_dropped):
    """Geometry of the cage: how far bracket (and bracket + TPU cap) stand proud of the camera in given directions (a flat
    ground approaching from that direction touches the bracket first and has that much travel before it reaches the camera),
    the view-cone margins, and the air gaps between the camera and the cage."""
    from scipy.spatial import cKDTree
    a, ey, u = R[:, 0], R[:, 1], R[:, 2]
    Pc = cam_surface(R, p, 40)
    Pb = np.concatenate([tm.vertices, tm.sample(80000)]); Pt = np.concatenate([tt.vertices, tt.sample(40000)])
    dirs = {"glass face (optical axis)": a, "camera top": u, "left end (+y)": ey, "right end (-y)": -ey,
            "front corner top-left": a + u + ey, "front corner top-right": a + u - ey,
            "front corner bottom-left": a - u + ey, "front corner bottom-right": a - u - ey,
            "glass top edge": a + u, "glass bottom edge": a - u, "torso +x (lying face down)": np.array([1.0, 0, 0])}
    def proud(P, n):
        n = n / np.linalg.norm(n)
        return round(float((P @ n).max() - (Pc @ n).max()), 1)
    # worst direction over the half sphere facing the camera's front and top (n . (a + u) >= 0); the other half faces the chest
    k = np.arange(6000) + 0.5; ph = np.arccos(1 - 2 * k / 6000); th = np.pi * (1 + 5 ** 0.5) * k
    N = np.stack([np.cos(th) * np.sin(ph), np.sin(th) * np.sin(ph), np.cos(ph)], 1)
    N = N[N @ ((a + u) / np.sqrt(2)) >= 0]
    from scipy.spatial import ConvexHull
    hb = Pb[ConvexHull(Pb).vertices]; hc = Pc[ConvexHull(Pc).vertices]; hbt = np.concatenate([hb, Pt[ConvexHull(Pt).vertices]])
    sb = (hb @ N.T).max(0) - (hc @ N.T).max(0); st = (hbt @ N.T).max(0) - (hc @ N.T).max(0)
    worst = N[np.argmin(sb)] @ R
    # air gaps: camera top face to the roof, end faces to the cheeks (edges where the floor and wall seat the camera excluded)
    cy = -IMAGER_Y; Lc = (Pc - p) @ R
    tree = cKDTree(Pb)
    topf = (Lc[:, 2] > CAM_H / 2 - 0.01) & (Lc[:, 0] > -ORIGIN_FROM_BACK + 2.5) & (np.abs(Lc[:, 1] - cy) < CAM_W / 2 - 2.5)
    endf = (np.abs(Lc[:, 1] - cy) > CAM_W / 2 - 0.01) & (Lc[:, 2] > -CAM_H / 2 + 2.5) & (Lc[:, 0] > -ORIGIN_FROM_BACK + 2.5)
    return dict(
        note="fall-protection cage; camera-local axes: optical axis, camera left (= robot +y), camera up",
        envelope_local_mm=dict(axis=[round(c_lo[0], 1), round(c_hi[0], 1)], left=[round(c_lo[1], 1), round(c_hi[1], 1)],
                               up=[round(c_lo[2], 1), round(c_hi[2], 1)]),
        opening_local_mm=dict(left=[round(pk_lo[1], 1), round(pk_hi[1], 1)], up=[round(pk_lo[2], 1), round(pk_hi[2], 1)]),
        roof_mm=dict(gap=TOP_GAP, thickness=ROOF_T), cheeks_mm=dict(gap=END_GAP, thickness=CHEEK_T), floor_mm=FLOOR + CAGE_FLOOR_EXTRA, wall_mm=WALL,
        front_rim_ahead_of_glass_mm=FRONT_PROUD,
        proud_of_camera_mm={k: proud(Pb, n) for k, n in dirs.items()},
        proud_of_camera_with_tpu_mm={k: proud(np.concatenate([Pb, Pt]), n) for k, n in dirs.items()},
        proud_min_over_front_top_half_sphere_mm=round(float(sb.min()), 1),
        proud_min_over_front_top_half_sphere_with_tpu_mm=round(float(st.min()), 1),
        proud_worst_direction_local=list(np.round(worst, 2)),
        fov_margin_deg_bracket=fov_margins(Pb, R, p), fov_margin_deg_tpu=fov_margins(Pt, R, p),
        cone_cut_extra_deg=CUT_EXTRA,
        camera_top_to_roof_mm=round(float(tree.query(Pc[topf])[0].min()), 2),
        camera_ends_to_cheeks_mm=round(float(tree.query(Pc[endf])[0].min()), 2),
        tpu_volume_cm3=round(tt.volume / 1000, 1), tpu_mass_g=round(tt.volume / 1000 * 1.21, 0), tpu_watertight=bool(tt.is_watertight),
        tpu_dropped_fragments_cm3=tpu_dropped,
        rounding_mm=dict(enabled=ROUND, outer_edges=R_OUT, rim_corners=R_RIM, small_features=R_SMALL, webs=R_HULL, pad_perimeter=R_PAD),
    )


def cam_surface(R, p, n=25):
    """Points on the D435 body surface (torso_link, mm)."""
    cy = -IMAGER_Y
    lo = np.array([-ORIGIN_FROM_BACK, cy - CAM_W / 2, -CAM_H / 2]); hi = np.array([ORIGIN_BEHIND_GLASS, cy + CAM_W / 2, CAM_H / 2])
    g = np.linspace(0, 1, n); S = []
    for i in g:
        for j in g:
            for f in ([i, j, 0], [i, j, 1], [i, 0, j], [i, 1, j], [0, i, j], [1, i, j]):
                S.append(lo + (hi - lo) * np.array(f))
    return p + np.array(S) @ R.T


def mesh_clearances(tm, R, p, mesh_path):
    """Exact distances to the chest mesh: to the detachable lower panel (its piece of torso_plus_link.STL plus the sensor-window
    inserts in it) and to the rest of the torso."""
    from trimesh.graph import connected_component_labels
    m = trimesh.load(mesh_path); m.apply_scale(1000.0)
    lab = connected_component_labels(m.face_adjacency, node_count=len(m.faces))
    _, _, it = m.ray.intersects_location([[500.0, 0.0, 120.0]], [[-1.0, 0, 0]], multiple_hits=False)
    low = lab[it[0]]
    fz = m.vertices[m.faces][:, :, 2].max(1); fx = m.vertices[m.faces][:, :, 0].max(1)
    cmax_z = np.zeros(lab.max() + 1); cmax_x = np.zeros(lab.max() + 1)
    np.maximum.at(cmax_z, lab, fz); np.maximum.at(cmax_x, lab, fx)
    panel = (lab == low) | ((cmax_z[lab] <= 248.0) & (cmax_x[lab] > 90.0))
    Bp = np.concatenate([tm.vertices, tm.sample(60000)]); Cp = cam_surface(R, p, 40)
    out = {}
    for key, fmask in (("panel", panel), ("rest_of_torso", ~panel)):
        cen = m.triangles_center
        box = np.concatenate([Bp, Cp])
        lo, hi = box.min(0) - 15, box.max(0) + 15
        sel = fmask & np.all((cen > lo) & (cen < hi), 1)
        if not sel.any():
            out["bracket_min_dist_to_%s_mm" % key] = None; out["camera_min_dist_to_%s_mm" % key] = None; continue
        sub = m.submesh([np.where(sel)[0]], append=True)
        for nm, P in (("bracket", Bp), ("camera", Cp)):
            _, dist, _ = trimesh.proximity.closest_point(sub, P)
            out["%s_min_dist_to_%s_mm" % (nm, key)] = round(float(dist.min()), 2)
    return out


def head_clearance(P, mjcf, mesh_dir):
    """Smallest distance from the head (head_yaw_link + head_pitch_link visual meshes) to the bracket and camera, at the default
    head pose and over the head joints' full range (corners and centre of the yaw x pitch range)."""
    import mujoco
    from scipy.spatial import cKDTree
    spec = mujoco.MjSpec.from_file(mjcf); spec.meshdir = mesh_dir
    m = spec.compile(); d = mujoco.MjData(m)
    tree = cKDTree(P / 1000.0)
    hb = {m.body(n).id for n in ("head_yaw_link", "head_pitch_link")}
    tb = m.body("torso_link").id
    jy, jp = m.joint("head_yaw_joint"), m.joint("head_pitch_joint")
    res = {}
    for qy in (jy.range[0], 0.0, jy.range[1]):
        for qp in (jp.range[0], 0.0, jp.range[1]):
            d.qpos[:] = m.qpos0; d.qpos[jy.qposadr[0]] = qy; d.qpos[jp.qposadr[0]] = qp
            mujoco.mj_forward(m, d)
            Pt, Rt = d.xpos[tb], d.xmat[tb].reshape(3, 3)
            best = np.inf
            for gi in range(m.ngeom):
                if m.geom_bodyid[gi] in hb and m.geom_type[gi] == mujoco.mjtGeom.mjGEOM_MESH:
                    mi = m.geom_dataid[gi]; vv = m.mesh_vert[m.mesh_vertadr[mi]:m.mesh_vertadr[mi] + m.mesh_vertnum[mi]]
                    w = d.geom_xpos[gi] + vv @ d.geom_xmat[gi].reshape(3, 3).T
                    best = min(best, float(tree.query((w - Pt) @ Rt)[0].min()))
            res[(round(float(qy), 3), round(float(qp), 3))] = best * 1000.0
    return dict(head_min_dist_default_pose_mm=round(res[(0.0, 0.0)], 1),
                head_min_dist_over_range_mm=round(min(res.values()), 1),
                head_worst_pose_yaw_pitch_rad=list(min(res, key=res.get)))


ARM_BODIES = [s + n for s in ("left_", "right_") for n in ("shoulder_pitch_link", "shoulder_roll_link", "shoulder_yaw_link",
                                                           "elbow_link", "wrist_yaw_link", "wrist_pitch_link", "wrist_roll_link")]
UPPER_BODIES = ["torso_link", "head_yaw_link", "head_pitch_link"] + ARM_BODIES


def stand(m, d):
    """Hiking default joint pose, base pitched so the soles are flat, feet on the floor. Returns the base pitch (rad)."""
    import mujoco
    dp = {"hip_pitch": -0.136, "knee": 0.379, "ankle_pitch": -0.114, "shoulder_pitch": 0.2, "elbow": -0.6}
    for j in range(m.njnt):
        nm = m.joint(j).name
        for k, v in dp.items():
            if k + "_joint" in nm: d.qpos[m.jnt_qposadr[j]] = v
        if nm == "left_shoulder_roll_joint": d.qpos[m.jnt_qposadr[j]] = 0.2
        if nm == "right_shoulder_roll_joint": d.qpos[m.jnt_qposadr[j]] = -0.2
    mujoco.mj_forward(m, d)
    phi = 0.0      # pitch the base so the soles are flat (the default pose's hip + knee + ankle pitch is not zero)
    for _ in range(6):
        Rf = d.xmat[m.body("left_ankle_roll_link").id].reshape(3, 3)
        phi -= np.arctan2(Rf[0, 2], Rf[2, 2])
        d.qpos[3:7] = [np.cos(phi / 2), 0.0, np.sin(phi / 2), 0.0]
        mujoco.mj_forward(m, d)
    za = min(d.xpos[m.body(n).id][2] for n in ("left_ankle_roll_link", "right_ankle_roll_link"))
    d.qpos[2] += 0.073 - za; mujoco.mj_forward(m, d)
    return phi


def body_points(m, d, bodies):
    """Visual-mesh vertices of the named bodies in the torso_link frame (mm), at the current pose."""
    import mujoco
    tb = m.body("torso_link").id; Pt, Rt = d.xpos[tb], d.xmat[tb].reshape(3, 3)
    ids = {m.body(n).id for n in bodies}; out = []
    for gi in range(m.ngeom):
        if m.geom_bodyid[gi] in ids and m.geom_type[gi] == mujoco.mjtGeom.mjGEOM_MESH:
            mi = m.geom_dataid[gi]; vv = m.mesh_vert[m.mesh_vertadr[mi]:m.mesh_vertadr[mi] + m.mesh_vertnum[mi]]
            out.append(((d.geom_xpos[gi] + vv @ d.geom_xmat[gi].reshape(3, 3).T) - Pt) @ Rt * 1000.0)
    return np.concatenate(out)


def robot_checks(name, P_mount, P_bracket, P_cam, mjcf, mesh_dir):
    """At the standing default pose: arm-to-mount clearance, and which part of the upper body (torso, head, arms; legs left
    out) a flat ground meets first for each fall direction, and how far that is ahead of the camera. Purely geometric: a rigid
    robot at the default pose meeting a plane; no dynamics, no joint motion during the fall."""
    import mujoco
    from scipy.spatial import cKDTree, ConvexHull
    m = mujoco.MjSpec.from_file(mjcf); m.meshdir = mesh_dir; m = m.compile(); d = mujoco.MjData(m)
    stand(m, d)
    tree = cKDTree(P_mount)
    arms = {}
    for side in ("left", "right"):
        Pa = body_points(m, d, [b for b in ARM_BODIES if b.startswith(side)])
        arms["%s_arm_min_dist_mm" % side] = round(float(tree.query(Pa)[0].min()), 1)
    Pu = body_points(m, d, UPPER_BODIES)
    hu, hb, hc = (P[ConvexHull(P).vertices] for P in (Pu, P_bracket, P_cam))
    def lead(n):
        n = np.asarray(n, float); n = n / np.linalg.norm(n, axis=-1, keepdims=True)
        c = (hc @ n.T).max(0); return (hu @ n.T).max(0) - c, (hb @ n.T).max(0) - c
    # named falls: the ground's direction seen from the torso when the chest, side or back touches; beta = torso tilt
    named = {}
    for lab, (s, sy) in {"forward": (1, 0), "left side": (0, 1), "right side": (0, -1), "backward": (-1, 0),
                         "forward-left 45": (0.7071, 0.7071), "forward-right 45": (0.7071, -0.7071)}.items():
        for beta in (90, 60, 45):
            b = np.deg2rad(beta)
            n = np.array([[s * np.sin(b), sy * np.sin(b), -np.cos(b)]])
            lb, lc = lead(n)
            first = "body" if lb[0] >= lc[0] else "cage"
            named["%s, torso %d deg from upright" % (lab, beta)] = dict(first=first, camera_behind_first_contact_mm=round(float(max(lb[0], lc[0])), 1),
                                                                        cage_ahead_of_camera_mm=round(float(lc[0]), 1))
    # map over all directions (torso frame): azimuth about z from +x, elevation
    az = np.deg2rad(np.arange(-180, 181, 2.0)); el = np.deg2rad(np.arange(-90, 91, 2.0))
    AZ, EL = np.meshgrid(az, el)
    N = np.stack([np.cos(EL) * np.cos(AZ), np.cos(EL) * np.sin(AZ), np.sin(EL)], -1).reshape(-1, 3)
    lb, lc = lead(N); best = np.maximum(lb, lc)
    w = np.cos(EL).ravel()     # area weight
    frac = lambda msk: round(float((w * msk).sum() / w.sum()), 3)
    cover = dict(share_of_directions_body_first=frac((lb >= lc) & (best >= 8)), share_cage_first=frac((lc > lb) & (best >= 8)),
                 share_camera_less_than_8mm_behind=frac(best < 8), share_camera_first=frac(best <= 0))
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    fig, A = plt.subplots(1, 2, figsize=(15, 5.2))
    im = A[0].pcolormesh(np.degrees(AZ), np.degrees(EL), np.clip(best, -30, 60).reshape(AZ.shape), cmap="RdYlGn", vmin=-30, vmax=60, shading="auto")
    A[0].contour(np.degrees(AZ), np.degrees(EL), best.reshape(AZ.shape), levels=[0, 8], colors=["k", "k"], linestyles=["-", "--"])
    plt.colorbar(im, ax=A[0], label="mm the camera is behind the first contact")
    A[0].set_title("%s: how far behind the first contact the camera is (dashed: 8 mm, solid: 0)" % name, fontsize=10)
    cat = np.where(best < 8, 2, np.where(lb >= lc, 0, 1)).reshape(AZ.shape)
    A[1].pcolormesh(np.degrees(AZ), np.degrees(EL), cat, cmap=ListedColormap(["#9ab", "#e8893a", "#d22"]), vmin=-0.5, vmax=2.5, shading="auto")
    A[1].set_title("first contact: grey = robot body (torso, head, arms), orange = cage, red = camera < 8 mm behind", fontsize=10)
    for B in A:
        for lab, (x, y) in {"forward\n(face down)": (0, 0), "left side": (90, 0), "right side": (-90, 0), "back": (180, 0),
                            "chest 45 deg": (0, -45), "head first": (0, 85), "feet first": (0, -85)}.items():
            B.plot(x, y, "k+", ms=10); B.text(x + 3, y + 3, lab, fontsize=8)
        B.set_xlabel("ground direction seen from the torso: azimuth (deg, 0 = forward +x, 90 = robot's left)")
        B.set_ylabel("elevation (deg, -90 = down)"); B.set_xlim(-180, 180); B.set_ylim(-90, 90)
    fig.suptitle("%s mount, default standing pose, legs left out: which part a flat ground meets first" % name, fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "fall_coverage_%s.png" % name), dpi=90); plt.close(fig)
    return dict(arms_default_pose=arms, fall_named=named, fall_coverage=cover)


def print_orientation(tm, R):
    """Pick the print orientation with the least overhang (faces steeper than 45 deg facing down, not on the bed) among the
    robot axes and the camera axes; returns the placed mesh, the direction (torso_link) that points down, the overhang area."""
    cands = {"+x": [1, 0, 0], "-x": [-1, 0, 0], "+y": [0, 1, 0], "-y": [0, -1, 0], "+z": [0, 0, 1], "-z": [0, 0, -1],
             "+optical axis": R[:, 0], "-optical axis": -R[:, 0], "+camera up": R[:, 2], "-camera up": -R[:, 2]}
    best = None
    for k, dvec in cands.items():
        t = tm.copy()
        t.apply_transform(trimesh.geometry.align_vectors(np.asarray(dvec, float), [0, 0, -1]))
        t.apply_translation(-t.bounds[0])
        fz = t.vertices[t.faces][:, :, 2].min(1)
        ov = float(t.area_faces[(t.face_normals[:, 2] < -0.7071) & (fz > 0.3)].sum()) / 100.0
        if best is None or ov < best[2]:
            best = (t, k, ov)
    return best


# ---------------------------------------------------------------- drawing
def drawing(name, tm, info, R, p, ys, zs, X, seam, tt=None):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Circle
    fig, ax = plt.subplots(1, 2, figsize=(15, 8.5), gridspec_kw=dict(width_ratios=[1.1, 1]))
    # side section through the case centre
    yc = -IMAGER_Y
    sec = tm.section(plane_origin=[0, yc, 0], plane_normal=[0, 1, 0])
    A = ax[0]
    iy = np.argmin(abs(ys - yc)); A.plot(X[:, iy], zs, color="0.5", lw=1.5, label="chest shell (y = %.1f)" % yc)
    iy0 = np.argmin(abs(ys)); A.plot(X[:, iy0], zs, color="0.75", lw=1, ls="--", label="chest shell (y = 0)")
    if sec is not None:
        for e in sec.discrete:
            A.fill(e[:, 0], e[:, 2], color="#e8893a", alpha=0.85, lw=0.8, ec="k")
    st = tt.section(plane_origin=[0, yc, 0], plane_normal=[0, 1, 0]) if tt is not None else None
    if st is not None:
        for e in st.discrete:
            A.fill(e[:, 0], e[:, 2], color="#3373bf", alpha=0.85, lw=0.6, ec="k")
        A.fill([], [], color="#3373bf", label="optional TPU bumper cap")
    a, u = R[:, 0], R[:, 2]
    cl = np.array([[-ORIGIN_FROM_BACK, -CAM_H / 2], [ORIGIN_BEHIND_GLASS, -CAM_H / 2], [ORIGIN_BEHIND_GLASS, CAM_H / 2], [-ORIGIN_FROM_BACK, CAM_H / 2]])
    cw = np.array([p + c[0] * a + c[1] * u for c in cl])
    A.add_patch(Polygon(cw[:, [0, 2]], closed=True, fc="0.25", ec="k", alpha=0.85, label="D435 body"))
    A.plot(p[0], p[2], "r+", ms=14, mew=2)
    for ang, ls in ((0, "-"), (VFOV / 2, "--"), (-VFOV / 2, "--")):
        t = np.deg2rad(info["pitch_down_deg"] + ang)
        d = np.array([np.cos(t), -np.sin(t)])
        A.plot([p[0], p[0] + 120 * d[0]], [p[2], p[2] + 120 * d[1]], "r", ls=ls, lw=1)
    A.annotate("depth origin (%.0f, %.0f) mm\n%.1f deg below horizontal" % (p[0], p[2], info["pitch_down_deg"]),
               xy=(p[0], p[2]), xytext=(p[0] + 25, p[2] + 30), arrowprops=dict(arrowstyle="->"), fontsize=10)
    A.annotate("", xy=(p[0], p[2] - 4), xytext=(info["shell_point_behind_centre_mm"][0], p[2] - 4),
               arrowprops=dict(arrowstyle="<->", color="b"))
    A.text((p[0] + info["shell_point_behind_centre_mm"][0]) / 2, p[2] - 9, "%.1f mm" % info["centre_ahead_of_shell_mm"], color="b",
           ha="center", fontsize=9)
    zl, zt = float(np.interp(yc, ys, seam[1])) + SEAM_MARGIN, PAD_TOP
    A.axhline(zl, color="g", lw=0.6, ls=":"); A.axhline(zt, color="g", lw=0.6, ls=":")
    A.text(40, zl + 1, "pad lower edge z = %.0f (here)" % zl, color="g", fontsize=8)
    A.text(40, zt + 1, "pad upper edge z = %.0f" % zt, color="g", fontsize=8)
    s0, s1 = float(np.interp(yc, ys, seam[0])), float(np.interp(yc, ys, seam[1]))
    A.axhspan(s0, s1, color="m", alpha=0.5); A.axhspan(70, s0, xmin=0, xmax=0.02, color="m", alpha=0.5)
    A.text(41, s0 - 7, "panel seam z %.0f-%.0f\nbelow: detachable panel" % (s0, s1), color="m", fontsize=8)
    A.axhspan(WIN_Z[0], WIN_Z[1], xmin=0, xmax=0.04, color="c")
    A.text(45, (WIN_Z[0] + WIN_Z[1]) / 2, "sensor window\nz %.0f-%.0f" % WIN_Z, color="c", fontsize=8, va="center")
    A.set_aspect("equal"); A.set_xlim(35, 215); A.set_ylim(min(p[2] - 60, 150), PAD_TOP + 25); A.grid(alpha=0.3)
    A.set_xlabel("x forward (mm, torso_link)"); A.set_ylabel("z up (mm, torso_link)")
    A.set_title("%s: side section through the camera case centre (y = %.1f mm)" % (name, yc)); A.legend(loc="lower right", fontsize=8)
    # front view (looking back at the chest): pad, window, camera, holes
    B = ax[1]
    Y, Z = np.meshgrid(ys, zs)
    B.contour(Y, Z, X, levels=np.arange(60, 112, 4), colors="0.8", linewidths=0.6)
    th = np.linspace(0, 2 * np.pi, 100)
    zc0, zc1 = WIN_Z[0] + WIN_HALF_W, WIN_Z[1] - WIN_HALF_W
    capsule = np.concatenate([np.stack([WIN_HALF_W * np.cos(th[:50]), zc1 + WIN_HALF_W * np.sin(th[:50])], 1),
                              np.stack([WIN_HALF_W * np.cos(th[50:]), zc0 + WIN_HALF_W * np.sin(th[50:])], 1)])
    B.add_patch(Polygon(capsule, closed=True, fc="c", alpha=0.3, ec="c", label="chest sensor window"))
    py, pz = info["pad_y_mm"], info["pad_z_centreline_mm"]
    pys = np.arange(py[0], py[1] + 0.01, 0.5)
    outline = np.concatenate([np.stack([pys, pad_lower_edge(ys, seam[1], pys)], 1), [[py[1], PAD_TOP], [py[0], PAD_TOP]]])
    B.add_patch(Polygon(outline, closed=True, fc="#e8893a", alpha=0.35, ec="#e8893a", lw=2,
                        label="VHB pad (%.0f cm2), %.0f mm wide" % (info["bonded_area_cm2"], py[1] - py[0])))
    ok = np.isfinite(seam[0])
    B.fill_between(ys[ok], seam[0][ok], seam[1][ok], color="m", alpha=0.6, label="panel seam (gap in the mesh)")
    B.fill_between(ys[ok], np.full(ok.sum(), 60.0), seam[0][ok], color="m", alpha=0.07, label="detachable lower panel")
    for ry0, ry1 in info["rails_y_mm"]:
        B.add_patch(Polygon([[ry0, p[2] + 15], [ry1, p[2] + 15], [ry1, PAD_TOP], [ry0, PAD_TOP]], closed=True, fc="none", ec="k",
                            ls="--", lw=1.2))
    if info["rails_y_mm"]:
        B.plot([], [], "k--", lw=1.2, label="rails, %.1f mm off the panel" % CLEAR_PANEL)
    cc = np.array(info["case_centre_mm"])
    B.add_patch(Polygon([[cc[1] - CAM_W / 2, cc[2] - 6], [cc[1] + CAM_W / 2, cc[2] - 6], [cc[1] + CAM_W / 2, cc[2] + 6],
                         [cc[1] - CAM_W / 2, cc[2] + 6]], closed=True, fc="0.25", alpha=0.5, label="D435 (projected, approx.)"))
    for h in info["m3_holes_mm"]:
        B.add_patch(Circle((h[1], h[2]), 1.7, fc="w", ec="k"))
    hz = info["m3_holes_mm"][0][2]
    B.annotate("", xy=(info["m3_holes_mm"][0][1], hz + 9), xytext=(info["m3_holes_mm"][1][1], hz + 9), arrowprops=dict(arrowstyle="<->"))
    B.text(np.mean([h[1] for h in info["m3_holes_mm"]]), hz + 11, "M3 x 6, 45 mm", ha="center", fontsize=9)
    B.plot([0, 0], [pz[0] - 15, pz[1] + 15], "r-.", lw=1); B.text(1, pz[1] + 10, "robot centreline = left imager", color="r", fontsize=8)
    B.plot(0, p[2], "r+", ms=14, mew=2)
    for zz in pz:
        B.plot([-NOTCH, 0, NOTCH], [zz, zz + (NOTCH if zz == pz[0] else -NOTCH), zz], "k", lw=1.5)
    B.invert_xaxis()   # looking at the robot from the front: robot's left on the right of the picture
    B.set_aspect("equal"); B.set_xlim(80, -80); B.set_ylim(min(p[2] - 40, WIN_Z[0] - 10), PAD_TOP + 25); B.grid(alpha=0.3)
    B.set_xlabel("y (mm; robot's left is +y, shown on the right as seen from the front)"); B.set_ylabel("z (mm)")
    B.set_title("front view (looking at the robot)"); B.legend(loc="lower left", fontsize=8)
    txt = ("optical centre (left imager, 4.2 mm behind glass): x %.0f, y 0, z %.0f mm in torso_link; pitch %.1f deg down; "
           "case centre y = -17.5 mm;  shell under the centre: x %.1f mm (centre %.1f mm ahead), surface normal tilted %.0f deg up;  "
           "camera clearance to shell %.1f mm;  pad gap for VHB %.1f mm, bonded area %.1f cm2, all above the panel seam;  "
           "bracket to detachable panel >= %s mm;  part %.1f cm3" %
           (p[0], p[2], info["pitch_down_deg"], info["shell_point_behind_centre_mm"][0], info["centre_ahead_of_shell_mm"],
            info["shell_tilt_back_deg"], info["camera_min_clearance_to_shell_mm"], TAPE, info["bonded_area_cm2"],
            info.get("bracket_min_dist_to_panel_mm", info["bracket_min_gap_below_seam_mm"]), info["volume_cm3"]))
    c = info.get("cage")
    if c:
        pr = c["proud_of_camera_mm"]
        txt += (";  fall cage stands proud of the camera: glass %.1f, top %.1f, ends %.1f / %.1f, front corners >= %.1f mm "
                "(+3 mm with the TPU cap); air gap %.1f mm over the top, %.1f mm at the ends" %
                (pr["glass face (optical axis)"], pr["camera top"], pr["left end (+y)"], pr["right end (-y)"],
                 min(v for k, v in pr.items() if k.startswith("front corner")), c["camera_top_to_roof_mm"], c["camera_ends_to_cheeks_mm"]))
    fig.text(0.01, 0.01, txt, fontsize=8, wrap=True)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = os.path.join(HERE, "drawing_%s.png" % name)
    fig.savefig(out, dpi=110); plt.close(fig)
    return out


# ---------------------------------------------------------------- MuJoCo renders
def render(name, stl_path, tpu_path, info, R, p, mjcf, mesh_dir):
    import mujoco
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    spec = mujoco.MjSpec.from_file(mjcf)
    spec.meshdir = mesh_dir
    spec.visual.global_.offwidth = 1200; spec.visual.global_.offheight = 1200
    spec.add_mesh(name="mount", file=os.path.abspath(stl_path), scale=[0.001] * 3)
    body = spec.body("torso_link")
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="mount", rgba=[0.95, 0.55, 0.15, 1], contype=0, conaffinity=0, group=2)
    spec.add_mesh(name="tpu", file=os.path.abspath(tpu_path), scale=[0.001] * 3)
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="tpu", rgba=[0.2, 0.45, 0.75, 1], contype=0, conaffinity=0, group=2)
    a, ey, u = R[:, 0], R[:, 1], R[:, 2]
    Mb = np.stack([ey, a, -u], 1); q = np.zeros(4); mujoco.mju_mat2Quat(q, Mb.ravel())
    cc = p + R @ np.array([-(ORIGIN_FROM_BACK - CAM_D / 2), -IMAGER_Y, 0.0])
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[CAM_W / 2000, CAM_D / 2000, CAM_H / 2000], pos=(cc / 1000).tolist(), quat=q.tolist(),
                  rgba=[0.12, 0.12, 0.13, 1], contype=0, conaffinity=0, group=2)
    glass = p + a * ORIGIN_BEHIND_GLASS
    for dy, rr in ((0.0, 0.0045), (-50.0, 0.0045), (-15.0, 0.003)):   # left imager, right imager, RGB (approx.)
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[rr, 0, 0], pos=((glass + ey * dy) / 1000).tolist(), rgba=[0.3, 0.5, 0.9, 1],
                      contype=0, conaffinity=0, group=2)
    m = spec.compile(); d = mujoco.MjData(m)
    phi = stand(m, d)      # standing pose (Hiking default pose), soles flat, feet on the floor
    info["render_base_pitch_deg"] = round(float(np.degrees(phi)), 1)
    tb = m.body("torso_link").id
    P, Rt = d.xpos[tb], d.xmat[tb].reshape(3, 3)
    o = P + Rt @ (p / 1000); Rc = Rt @ R
    info["render_camera_height_m"] = round(float(o[2]), 3)
    toe = -1.0
    for gi in range(m.ngeom):
        if m.geom_bodyid[gi] == m.body("left_ankle_roll_link").id and m.geom_type[gi] == mujoco.mjtGeom.mjGEOM_MESH:
            mi = m.geom_dataid[gi]; vv = m.mesh_vert[m.mesh_vertadr[mi]:m.mesh_vertadr[mi] + m.mesh_vertnum[mi]]
            toe = max(toe, float(np.max(d.geom_xpos[gi][0] + vv @ d.geom_xmat[gi].reshape(3, 3).T[:, 0])))
    info["render_toe_tip_ahead_of_lens_m"] = round(toe - float(o[0]), 3)
    info["render_axis_pitch_world_deg"] = round(float(np.degrees(np.arcsin(-Rc[2, 0]))), 1)
    rays = []
    for sy in (1, -1):
        for sz in (1, -1):
            v = Rc @ np.array([1.0, sy * np.tan(np.deg2rad(HFOV / 2)), sz * np.tan(np.deg2rad(VFOV / 2))])
            t = -o[2] / v[2] if v[2] < 0 else 2.5
            rays.append(o + v * min(t, 2.5 / np.linalg.norm(v)))
    # expected depth image on flat ground at this standing pose (z depth along the optical axis, 0..3 m; robot's own body
    # included, mount and camera excluded): row 0 = image top, column 0 = image left (robot's left), pixel centres
    W, H = (64, 36) if name == "hiking" else (36, 36)
    th, tv = np.tan(np.deg2rad(HFOV / 2)), np.tan(np.deg2rad(VFOV / 2))
    depth = np.zeros((H, W))
    gg = np.array([1, 1, 0, 0, 0, 0], np.uint8); gid = np.zeros(1, np.int32)
    for i in range(H):
        for j in range(W):
            v = Rc @ np.array([1.0, th * (1 - 2 * (j + 0.5) / W), tv * (1 - 2 * (i + 0.5) / H)])
            t = mujoco.mj_ray(m, d, o, v, gg, 1, -1, gid)
            depth[i, j] = 3.0 if t < 0 else min(t, 3.0)    # t is in units of |v|, and v has unit length along the axis
    np.save(os.path.join(HERE, "expected_depth_%s.npy" % name), depth.astype(np.float32))
    fig, A = plt.subplots(figsize=(7, 4.6))
    im = A.imshow(depth, cmap="viridis", vmin=0, vmax=3); plt.colorbar(im, ax=A, label="z depth (m)")
    if name == "hiking":
        A.axvline(15.5, color="w", ls="--"); A.axvline(47.5, color="w", ls="--")
        A.set_title("expected depth, Hiking: raw 64 x 36 (policy uses columns 16-47)", fontsize=10)
    else:
        A.set_title("expected depth, SSR: 36 x 36 (87 x 58 deg resampled)", fontsize=10)
    A.set_xlabel("column (0 = robot's left)"); A.set_ylabel("row (0 = top)")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "expected_depth_%s.png" % name), dpi=90); plt.close(fig)
    r = mujoco.Renderer(m, 900, 900)
    views = [("front", 180, -10, 1.1, P + [0, 0, 0.0]), ("side", -90, -8, 1.6, P + [0.45, 0, -0.35]),
             ("three-quarter", 215, -18, 1.4, P + [0.25, 0, -0.25]), ("close-up", 235, -12, 0.42, o),
             ("cage from the front, below (arms, head hidden; blue = TPU cap)", 200, 28, 0.36, o),
             ("cage from the robot's left, above (arms, head hidden)", 120, -30, 0.36, o)]
    hide = {m.body(n).id for n in ARM_BODIES + ["head_yaw_link", "head_pitch_link"]}
    for gi in range(m.ngeom):
        if m.geom_bodyid[gi] in hide: m.geom_group[gi] = 5
    imgs = []
    for title, az, el, dist, look in views:
        cam = mujoco.MjvCamera(); cam.azimuth, cam.elevation, cam.distance = az, el, dist; cam.lookat[:] = look
        opt = mujoco.MjvOption(); opt.geomgroup[5] = 0 if title.startswith("cage") else 1   # groups 0-2 as default
        r.update_scene(d, camera=cam, scene_option=opt)
        scn = r.scene
        def line(p0, p1, rgba, w=0.0025):
            if scn.ngeom >= scn.maxgeom: return
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.array(rgba, np.float32))
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, w, np.asarray(p0, float), np.asarray(p1, float))
            scn.ngeom += 1
        w = 0.0025 if not title.startswith("cage") else 0.0008
        for e in rays: line(o, e, [1, 0.1, 0.1, 0.9], w)
        for i, j in ((0, 1), (1, 3), (3, 2), (2, 0)): line(rays[i], rays[j], [1, 0.1, 0.1, 0.9], w)
        line(o, o + Rc[:, 0] * 0.35, [1, 1, 0, 1], 0.003 if not title.startswith("cage") else 0.001)
        imgs.append((title, r.render().copy()))
    fig, axs = plt.subplots(2, 3, figsize=(18, 12.5))
    for A, (title, im) in zip(axs.ravel(), imgs):
        A.imshow(im); A.set_title(title, fontsize=10); A.axis("off")
    fig.suptitle("X2 Ultra with the %s chest mount and RealSense D435: depth origin (%.0f, 0, %.0f) mm on torso_link, %.1f deg down; "
                 "\nred = 87 x 58 deg depth frustum to the floor, yellow = optical axis" % (name, p[0], p[2], info["pitch_down_deg"]), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(HERE, "render_%s.png" % name)
    fig.savefig(out, dpi=90); plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", default=os.path.join(os.environ.get("X2_ASSETS", "assets/robot"), r"urdf/x2_ultra/meshes/torso_plus_link.STL"))
    ap.add_argument("--mjcf", default=os.path.join(os.environ.get("X2_ASSETS", "assets/robot"), r"mjcf/x2_ultra.xml"))
    ap.add_argument("--meshdir", default=os.path.join(os.environ.get("X2_ASSETS", "assets/robot"), r"urdf/x2_ultra/meshes"))
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--sharp", action="store_true", help="no edge rounding (for comparison); writes nothing but prints volumes")
    args = ap.parse_args()
    global ROUND
    if args.sharp:
        ROUND = False
        ys, zs, X = shell_heightmap(args.mesh); seam = panel_seam(args.mesh, ys)
        for name, cfg in POSES.items():
            tm, tt, info, R, p = build(name, cfg, ys, zs, X, seam[1])
            print(name, "sharp: bracket %.1f cm3, TPU %.1f cm3" % (tm.volume / 1000, tt.volume / 1000))
        return
    ys, zs, X = shell_heightmap(args.mesh)
    seam = panel_seam(args.mesh, ys)
    summary = {"panel_seam": {"note": "z (mm, torso_link) of the gap between the detachable lower chest panel (below) and the fixed "
                                      "upper shell (above), from torso_plus_link.STL",
                              **{"y%+.0f" % yy: [round(float(np.interp(yy, ys, seam[0])), 1), round(float(np.interp(yy, ys, seam[1])), 1)]
                                 for yy in (-80, -70, -60, -50, -40, -30, -20, 0, 20, 30, 40, 50, 60, 70, 80)}}}
    for name, cfg in POSES.items():
        tm, tt, info, R, p = build(name, cfg, ys, zs, X, seam[1])
        f_robot = os.path.join(HERE, "chest_mount_%s_robot_frame.stl" % name)
        f_print = os.path.join(HERE, "chest_mount_%s_print.stl" % name)
        t_robot = os.path.join(HERE, "chest_mount_%s_tpu_bumper_robot_frame.stl" % name)
        t_print = os.path.join(HERE, "chest_mount_%s_tpu_bumper_print.stl" % name)
        tp, down, ov = print_orientation(tm, R)
        tm.export(f_robot); tp.export(f_print)
        info.update(print_side_down=down, print_overhang_cm2=round(ov, 1))
        tq, tdown, tov = print_orientation(tt, R)
        tt.export(t_robot); tq.export(t_print)
        info["cage"].update(tpu_print_side_down=tdown, tpu_print_overhang_cm2=round(tov, 1))
        both = trimesh.util.concatenate([tm, tt])
        Pc = cam_surface(R, p, 40)
        P_mount = np.concatenate([tm.vertices, tm.sample(60000), tt.vertices, tt.sample(20000), Pc])
        if os.path.exists(args.mesh):
            info.update(mesh_clearances(both, R, p, args.mesh))
        if os.path.exists(args.mjcf):
            info.update(head_clearance(P_mount, args.mjcf, args.meshdir))
            info.update(robot_checks(name, P_mount, np.concatenate([tm.vertices, tm.sample(60000)]), Pc, args.mjcf, args.meshdir))
        drawing(name, tm, info, R, p, ys, zs, X, seam, tt)
        if not args.no_render:
            render(name, f_robot, t_robot, info, R, p, args.mjcf, args.meshdir)
        summary[name] = info
        print(json.dumps(info))
    json.dump(summary, open(os.path.join(HERE, "mount_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
