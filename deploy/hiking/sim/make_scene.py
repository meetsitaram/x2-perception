"""Build a MuJoCo scene for the gwbc-x2 bridge: the X2 MJCF the bridge already uses (its motor actuators kept) on terrain from
this repo's eval course builders. Run inside the gr00t-x2sim container (its MuJoCo writes the XML the bridge then loads).

    python3 make_scene.py --scene park --out DIR        # DIR/scene.xml + DIR/terrain.bin + DIR/scene.json

Scenes:
  flat      the plain floor (z = 0)
  park      terrain-park's extended map (extended_v4k): the nine tile types along +x at levels 2 / 5 / 8 (lanes y = 0 / 8 / 16),
            and a lane at y = -8 of Hiking's paper types and our apparatus; spawn on a flat tile at the origin.
  hiking_park  the first park: three lanes walked +x from a flat 8 x 8 m spawn tile at the origin,
            8 m tiles laid edge to edge (each tile shifted so its -x edge sits at z = 0):
              y =  0  Hiking's training types, curriculum row 2: slope_up, stairs_up, platform_up, gap
              y = +8  Hiking's training types, row 2: boxes, small_boxes, rough, stairs_down
              y = -8  extended-map tiles: single 4x4, wedges, stairs up, and our apparatus
  <course>  any single eval course: hik_<type>_L<row>, ext_* (eval/train_terrains.py), a ladder course, or apparatus:<name>
Terrain data: PA_APPARATUS_DIR (terrain-park/apparatus/ours: our apparatus is its "eval" folder) and PA_TERRAIN_DIR (terrain-park/terrains/extended_v1).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys

os.environ.setdefault("LADDER_PAD_M", "0")       # tiles edge to edge; the park's canvas is the floor around them
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "eval"))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

import x2 as X  # noqa: E402

TILE = 8.0
# Surfaces (looks only; physics and depth unchanged): per terrain piece by the tile it lies in (user, 2026-11-15: "realistic
# textures - concrete blocks, wooden stairs and 4x4s, grassy slopes"). Grass / concrete / dirt / wood are generated here
# (tileable noise); plywood is terrain-park's own (apparatus/ours/cad/fusion/textures). Floors are never wood (user: "everything
# is brown - the floor and the stairs"): wood is for boxes (box-built stairs, walls) and the 4x4s, drawn as visual-only boxes
# over their height-field ridges (no collision, outside the depth camera's groups 1/4/5).
SURFACE = [("hik_slope", "grass"), ("hik_stairs", "concrete"), ("hik_platform", "concrete"), ("hik_gap", "concrete"),
           ("hik_boxes", "concrete"), ("hik_small_boxes", "concrete"), ("hik_rough", "dirt"), ("ext_log", "grass"),
           ("ext_wedges", "grass"), ("ext_stairs", "concrete"), ("ext_log_corridor", "grass"), ("apparatus", "concrete"),
           ("tp:single_beams", "grass"), ("tp:beam_fields", "grass"), ("tp:blocks", "concrete"), ("tp:wedges", "grass"),
           ("tp:tight_spots", "concrete"), ("tp:log_corridors", "grass"), ("tp:homes", "floor"), ("tp:street_front", "asphalt"),
           ("tp:stairs", "concrete")]
PROP_TILES = {"ext_wedges": "wood"}              # tiles whose raised height-field cells are props, drawn in this material
BEAMS = []                                        # 4x4 centre lines (world xy) of the pasted courses, for their wood overlay
CHUNK = 100                                       # height-field piece, cells (2 m at 2 cm)
TP_TYPES = ("single_beams", "beam_fields", "blocks", "wedges", "tight_spots", "log_corridors", "homes", "street_front", "stairs")
PARKS = {
    # the terrain park (default): terrain-park's extended map (extended_v4k, what the policies trained on), all nine types along
    # x in three lanes by level, plus a lane of Hiking's paper types and our apparatus
    "park": [(0.0, ["tp:%s:2%s" % (t, ":straight" if t == "stairs" else "") for t in TP_TYPES]),
             (TILE, ["tp:%s:5%s" % (t, ":L" if t == "stairs" else "") for t in TP_TYPES]),
             (2 * TILE, ["tp:%s:8%s" % (t, ":U" if t == "stairs" else "") for t in TP_TYPES]),
             (-TILE, ["hik_slope_up_L2", "hik_stairs_up_L2", "hik_platform_up_L2", "hik_gap_L2", "hik_boxes_L2", "hik_rough_L2",
                      "hik_stairs_down_L2", "apparatus:eval"])],
    # the render showcase (models/acp_depth/render_park.py): one level-4 tile per feature, the stairs, and our apparatus
    "showcase": [(0.0, ["tp:single_beams:4", "tp:beam_fields:4", "tp:blocks:4", "tp:wedges:4:long_ramp", "tp:wedges:4:pinwheel"]),
                 (TILE, ["tp:stairs:4:straight", "tp:stairs:4:L", "tp:stairs:4:U", "apparatus:eval"]),
                 (-TILE, ["tp:homes:4", "tp:homes:6"]),
                 (-2 * TILE, ["tp:blocks:9", "tp:wedges:8:pinwheel", "tp:log_corridors:4", "tp:log_corridors:6", "tp:log_corridors:8"])],
    # the first park (2026-11-15 morning): Hiking's paper types and three extended_v1 tiles
    "hiking_park": [(0.0, ["hik_slope_up_L2", "hik_stairs_up_L2", "hik_platform_up_L2", "hik_gap_L2"]),
                    (TILE, ["hik_boxes_L2", "hik_small_boxes_L2", "hik_rough_L2", "hik_stairs_down_L2"]),
                    (-TILE, ["ext_log_single", "ext_wedges", "ext_stairs_up", "apparatus:eval"])],
}
APPARATUS_OPEN = {"start_gate": math.pi / 2, "side_gate": math.pi / 2, "door_leaf": math.pi / 2}   # hinge -> drawn angle
APPARATUS = []                                    # world shifts of pasted apparatus tiles
# One apparatus model per scene, never both stacked (user, 2026-11-17: "It's ok to have either map, just don't have both
# overlapped together"). eval (default): the eval's raster + wall boxes, which are the current CAD (docs/FINDINGS.md B-app1).
# cad: a CAD export as physics and drawing, the eval's walls left out. terrain-park's export of 2026-11-15 is a stale CAD
# revision (4x4s 3 in off): use one regenerated from the current CAD (PA_APPARATUS_DIR; 2026-11-17: 98 % of the deck within
# 1 cm of the eval raster, models/acp_depth/runs/park_app).
APPARATUS_MODEL = os.environ.get("APPARATUS_MODEL", "eval")
assert APPARATUS_MODEL in ("eval", "cad"), APPARATUS_MODEL
VIS, MESHES = [], []                              # visual-only boxes; meshes (collide or not), world frame, from tp tiles


class TPTile:
    """One extended-map tile (terrain-park terrains/<set>/<type>.npz, README there): the floor raster for physics, the boxes and
    meshes NOT in the raster as collision geometry, the ones in it drawn visual-only (crisp shapes and materials over the raster)."""

    def __init__(self, ttype, level, variant=None):
        d = pathlib.Path(os.environ.get("PA_TERRAIN_V4K", pathlib.Path(os.environ.get("PA_TERRAIN_DIR", ".")).parent / "extended_v4k"))
        Z, meta = np.load(d / (ttype + ".npz")), json.load(open(d / (ttype + ".json")))["tiles"]
        cand = [m for m in meta if m["level"] == level] or meta
        pick = next((m for m in cand if variant and m.get("variant_kind") == variant), cand[0])
        i = pick["index"]
        self.name, self.meta = "tp:%s:%d %s" % (ttype, pick["level"], pick.get("variant_kind") or ""), pick
        self.grid = np.asarray(Z["height"][i], float)
        self.res = float(Z["resolution"])
        self.x0, self.y0 = (float(v) for v in Z["cell0_center"][:2])
        labels = Z["labels"]
        b0, b1 = Z["box_start"][i], Z["box_start"][i + 1]
        self.boxes, self.vis = [], []
        for k in range(b0, b1):
            lab = str(labels[Z["box_label"][k]])
            b = dict(center=Z["box_center"][k].tolist(), half=Z["box_half"][k].tolist(), quat=Z["box_quat"][k].tolist(),
                     mat=label_material(lab), label=lab)
            if not Z["box_in_raster"][k]:
                self.boxes.append(b)
            elif lab != "ground":
                self.vis.append(b)
        m0, m1 = Z["mesh_start"][i], Z["mesh_start"][i + 1]
        self.meshes = []
        for m in range(m0, m1):
            lab = str(labels[Z["mesh_label"][m]])
            self.meshes.append(dict(verts=Z["mesh_vertices"][Z["mesh_vert_start"][m]:Z["mesh_vert_start"][m + 1]].astype(float),
                                    faces=Z["mesh_faces"][Z["mesh_face_start"][m]:Z["mesh_face_start"][m + 1]].astype(int),
                                    collide=not bool(Z["mesh_in_raster"][m]), mat=label_material(lab), label=lab))
        self.route = {}


def label_material(lab):
    for keys, mat in (((("beam", "brace", "board", "stair_step", "handrail", "rail_fence", "corridor_side:post", "corridor_side:rail",
                         "door_leaf", "table", "chair", "wedge", "box", "post")), "wood"),
                      (("wall", "door_header"), "plaster"), (("lawn",), "grass"), (("paving", "slab", "kerb", "ring_sector"), "paving"),
                      (("car",), "car"), (("tyre",), "rubber"), (("rock", "stone"), "stone"), (("plant", "bush"), "plant"),
                      (("cylinder",), "metal"), (("block",), "concrete")):
        if any(lab.startswith(k) or (":" + k) in lab for k in keys):
            return mat
    return "concrete"


def course(name):
    if name.startswith("tp:"):
        parts = name.split(":")
        return TPTile(parts[1], int(parts[2]), parts[3] if len(parts) > 3 else None)
    if name.startswith("apparatus:"):
        from apparatus import Apparatus
        return Apparatus(name.split(":", 1)[1])
    import train_terrains as T
    if name.startswith("hik_"):
        t, k = name[4:].rsplit("_L", 1)
        return T.hiking(t, int(k))
    if name in T.EXT:
        return T._ext(name)
    import ladder as L
    return L.load_course(name)


def paste(canvas, c0, res, crs, x_edge, y_mid, boxes):
    """Paste course crs into canvas (c0 = world xy of cell 0, 0), its grid's -x edge at x_edge and centred on y_mid, heights
    shifted so the median of its -x edge row is z = 0. Returns the course's world shift (dx, dy, dz)."""
    g = np.asarray(crs.grid, float)
    assert abs(crs.res - res) < 1e-9, (crs.res, res)
    nx, ny = g.shape
    dz = -float(np.median(g[0]))
    dx, dy = x_edge - crs.x0, y_mid - (crs.y0 + (ny - 1) * res / 2)
    i0, j0 = int(round((crs.x0 + dx - c0[0]) / res)), int(round((crs.y0 + dy - c0[1]) / res))
    ci = slice(max(i0, 0), min(i0 + nx, canvas.shape[0]))
    cj = slice(max(j0, 0), min(j0 + ny, canvas.shape[1]))
    canvas[ci, cj] = g[ci.start - i0:ci.stop - i0, cj.start - j0:cj.stop - j0] + dz
    for b in getattr(crs, "boxes", []) or []:
        boxes.append(dict(b, center=[b["center"][0] + dx, b["center"][1] + dy, b["center"][2] + dz]))
    for v in getattr(crs, "vis", []):
        VIS.append(dict(v, center=[v["center"][0] + dx, v["center"][1] + dy, v["center"][2] + dz]))
    for mm in getattr(crs, "meshes", []):
        MESHES.append(dict(mm, verts=mm["verts"] + np.array([dx, dy, dz])))
    for bl in (getattr(crs, "route", None) or {}).get("beam_lines", []) or []:
        BEAMS.append(dict(a=[bl["a"][0] + dx, bl["a"][1] + dy], b=[bl["b"][0] + dx, bl["b"][1] + dy]))
    return dx, dy, dz


def terrain(scene):
    """-> (grid [ix, iy] or None, x0, y0, res, boxes, info)"""
    res = 0.02
    if scene == "flat":
        return None, 0.0, 0.0, res, [], {}
    if scene.startswith("beam"):    # flat floor + one 4x4 (8.9 cm) across the path, 'beam' at x 0.8 m, 'beam:X' at x X m
        bx = float(scene.split(":")[1]) if ":" in scene else 0.8
        grid = np.zeros((int(4.0 / res) + 1, int(4.0 / res) + 1))
        return grid, -1.0, -2.0, res, [dict(center=[bx, 0.0, 0.0445], half=[0.0445, 0.6, 0.0445], quat=[1, 0, 0, 0], mat="cedar")], {}
    if scene not in PARKS:
        crs = course(scene)
        return (np.asarray(crs.grid, float), crs.x0, crs.y0, crs.res, list(getattr(crs, "boxes", []) or []),
                dict(course=scene, spawn=getattr(crs, "spawn", None)))
    PARK = PARKS[scene]
    x_lo, x_hi = -TILE / 2, TILE / 2 + TILE * max(len(t) for _, t in PARK)
    ys = [y for y, _ in PARK]
    y_lo, y_hi = min(ys) - TILE / 2, max(ys) + TILE / 2
    canvas = np.zeros((int(round((x_hi - x_lo) / res)) + 1, int(round((y_hi - y_lo) / res)) + 1))
    boxes, placed = [], []
    for y, names in PARK:
        for k, n in enumerate(names):
            xe = TILE / 2 + TILE * k
            crs = course(n)
            sh = paste(canvas, (x_lo, y_lo), res, crs, xe, y, boxes)
            n = crs.name if isinstance(crs, TPTile) else n
            placed.append(dict(course=n, lane_y=y, x_from=xe, x_to=xe + TILE, shift=list(sh)))
            if n.startswith("apparatus:"):
                APPARATUS.append(list(sh))
            if n in PROP_TILES:     # canvas cells of this tile, for the prop overlay
                i0, i1 = int(round((xe - x_lo) / res)), int(round((xe + TILE - x_lo) / res))
                j0, j1 = int(round((y - TILE / 2 - y_lo) / res)), int(round((y + TILE / 2 - y_lo) / res))
                placed[-1]["cells"] = [i0, i1, j0, j1]
            print("  %-22s lane y %+3.0f  x %4.0f..%4.0f  shift %s" % (n, y, xe, xe + TILE, np.round(sh, 2).tolist()), flush=True)
    return canvas, x_lo, y_lo, res, boxes, dict(tiles=placed)


def in_apparatus(info, x, y):
    return any(t["course"].startswith("apparatus:") and t["x_from"] - 1.0 <= x < t["x_to"] + 6.0 and abs(y - t["lane_y"]) < TILE / 2
               for t in info.get("tiles", []))


def surface_at(info, scene, x, y):
    name = scene
    for t in info.get("tiles", []):
        if t["x_from"] <= x < t["x_to"] and abs(y - t["lane_y"]) < TILE / 2:
            name = t["course"]
            break
    for prefix, mat in SURFACE:
        if name.startswith(prefix):
            return mat
    return "concrete"


def _noise(n, beta, rng):
    """Tileable fractal noise (n x n), 1/f^beta spectrum, normalised to 0..1."""
    f = np.fft.fftfreq(n)
    r = np.hypot(*np.meshgrid(f, f, indexing="ij"))
    r[0, 0] = 1.0
    spec = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) / r ** beta
    spec[0, 0] = 0
    z = np.real(np.fft.ifft2(spec))
    return (z - z.min()) / (z.max() - z.min())


def make_texture(kind, path, n=512, seed=0):
    rng = np.random.default_rng(seed)
    a, b, c = _noise(n, 1.4, rng), _noise(n, 0.9, rng), rng.random((n, n))
    if kind == "grass":       # green with patchy hue and fine blade speckle
        base = np.array([0.30, 0.45, 0.17])
        img = base * (0.70 + 0.45 * a[..., None]) + np.array([0.06, 0.05, -0.02]) * (b[..., None] - 0.5) * 2
        img *= (0.85 + 0.25 * c[..., None])
    elif kind == "concrete":  # grey, faint blotches, fine aggregate, 1 m slab joints (texture = 1 m)
        img = np.full((n, n, 3), 0.62) * (0.88 + 0.16 * a[..., None]) * (0.94 + 0.08 * c[..., None])
        img[:3] *= 0.6
        img[:, :3] *= 0.6
    elif kind == "wood":      # light pine planks along x, 0.25 m wide (texture = 1 m), stretched grain
        rows = np.arange(n)[:, None]
        grain = _noise(n, 1.1, rng)[:, ::1]
        grain = np.repeat(grain[:, :n // 16], 16, axis=1)            # streaks along the board
        img = np.array([0.74, 0.58, 0.40]) * (0.80 + 0.30 * grain[..., None]) * (0.93 + 0.1 * c[..., None])
        img[np.broadcast_to((rows % (n // 4)) < 3, (n, n))] *= 0.55                            # plank joints
    elif kind == "plaster":   # off-white painted wall
        img = np.full((n, n, 3), [0.88, 0.86, 0.80]) * (0.95 + 0.06 * a[..., None]) * (0.97 + 0.04 * c[..., None])
    elif kind == "asphalt":   # dark grey, fine aggregate
        img = np.full((n, n, 3), 0.30) * (0.88 + 0.2 * a[..., None]) * (0.94 + 0.12 * c[..., None])
    elif kind == "floor":     # indoor floor tiles, 0.3 m (texture = 0.6 m), beige
        img = np.array([0.80, 0.74, 0.62]) * (0.93 + 0.1 * a[..., None]) * (0.97 + 0.05 * c[..., None])
        for k in (0, n // 2):
            img[k:k + 2] *= 0.7
            img[:, k:k + 2] *= 0.7
    else:                     # dirt: brown, clumpy, pebbles
        img = np.array([0.42, 0.33, 0.22]) * (0.65 + 0.5 * a[..., None]) * (0.85 + 0.2 * b[..., None])
        img[c > 0.995] = [0.55, 0.53, 0.50]
    from PIL import Image
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(path)


def add_surfaces(spec, out):
    tex = out / "textures"
    tex.mkdir(exist_ok=True)
    for i, kind in enumerate(("grass", "concrete", "dirt", "wood", "plaster", "asphalt", "floor")):
        make_texture(kind, tex / (kind + ".png"), seed=i)
    src = pathlib.Path(os.environ.get("PA_APPARATUS_DIR", "")) / "cad" / "fusion" / "textures"
    spec.add_material(name="hidden", rgba=[0, 0, 0, 0])      # collision walls the apparatus CAD model draws
    for name, rgba in dict(paving=[0.70, 0.68, 0.64, 1], car=[0.62, 0.08, 0.08, 1], rubber=[0.08, 0.08, 0.08, 1],
                           stone=[0.52, 0.50, 0.47, 1], plant=[0.20, 0.42, 0.15, 1], metal=[0.55, 0.57, 0.60, 1]).items():
        spec.add_material(name=name, rgba=rgba, specular=0.2)
    files = dict(grass=tex / "grass.png", concrete=tex / "concrete.png", dirt=tex / "dirt.png", plaster=tex / "plaster.png",
                 asphalt=tex / "asphalt.png", floor=tex / "floor.png",
                 wood=tex / "wood.png", cedar=tex / "wood.png", plywood=src / "plywood.png")
    size_m = dict(grass=2.0, concrete=1.0, dirt=1.5, wood=1.0, cedar=1.0, plywood=1.2, plaster=2.0, asphalt=2.0, floor=0.6)   # metres per texture repeat
    for name, f in files.items():
        if not f.exists():          # no terrain-park textures: a plain wood colour
            spec.add_material(name=name, rgba=[0.72, 0.52, 0.32, 1])
            continue
        spec.add_texture(name="tex_" + name, type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(f))
        tint = dict(cedar=[0.85, 0.80, 0.75, 1]).get(name, [1, 1, 1, 1])           # walls a shade darker than treads
        spec.add_material(name=name, textures=["", "tex_" + name], texuniform=True, rgba=tint,
                          texrepeat=[1 / size_m[name], 1 / size_m[name]], specular=0.1, shininess=0.1)


def add_apparatus_model(spec, shift, k):
    """terrain-park apparatus/ours/sim/apparatus.xml (same frame as the eval raster: lane along x, deck top z = 0), its real
    materials (plywood, cedar, OSB, galvanized, paint), at the tile's shift, 3 mm proud of the raster. Its boxes, cylinders and
    spheres are the apparatus's physics (group 5: the depth camera sees them); its meshes collide when nearly convex (volume / hull >= 0.85:
    panels, boards) and are drawn only otherwise (MuJoCo collides a mesh as its convex hull, which would fill notches). The eval's wall boxes for this tile come from another export
    (loco-manip-apparatus scene_x2_env_mobility.xml) and do not match this model (377 of 437 tall parts had no box within 8 cm:
    the robot walked through drawn rails, user 2026-11-15), so they are left out here. Joints removed: no new DOFs."""
    sim = pathlib.Path(os.environ.get("PA_APPARATUS_DIR", "")) / "sim"
    src = sim / "apparatus.xml"
    if not src.exists():
        print("  (no %s: apparatus drawn from the raster only)" % src)
        return
    a = mujoco.MjSpec.from_file(str(src))
    for t in a.textures:
        if t.file:
            t.file = str((sim / a.texturedir / t.file).resolve()) if not os.path.isabs(t.file) else t.file
    for m in a.meshes:
        if m.file:
            m.file = str((sim / a.meshdir / m.file).resolve()) if not os.path.isabs(m.file) else m.file
    a.texturedir = a.meshdir = ""
    # open the entrance gates and the door before freezing (user: "keep them open"); 90 deg = the eval's open door
    # (route.json door open_deg 90, its leaf in the physics walls); the gates are not in the physics walls at all
    m = a.compile()
    d = mujoco.MjData(m)
    for jn, ang in APPARATUS_OPEN.items():
        d.qpos[m.jnt_qposadr[m.joint(jn).id]] = ang
    mujoco.mj_kinematics(m, d)
    for jn in APPARATUS_OPEN:
        bid = m.jnt_bodyid[m.joint(jn).id]
        pid = m.body_parentid[bid]
        Rp = d.xmat[pid].reshape(3, 3)
        qp_inv, q = np.zeros(4), np.zeros(4)
        mujoco.mju_negQuat(qp_inv, d.xquat[pid])
        mujoco.mju_mulQuat(q, qp_inv, d.xquat[bid])
        body = a.body(m.body(bid).name)
        body.pos = (Rp.T @ (d.xpos[bid] - d.xpos[pid])).tolist()
        body.quat = q.tolist()
    convexity = {}                  # mesh name -> volume / convex-hull volume (1 = convex: panels, boards, posts)
    from scipy.spatial import ConvexHull
    for mi in range(m.nmesh):
        v0, nv = m.mesh_vertadr[mi], m.mesh_vertnum[mi]
        f0, nf = m.mesh_faceadr[mi], m.mesh_facenum[mi]
        V, F = m.mesh_vert[v0:v0 + nv].astype(float), m.mesh_face[f0:f0 + nf]
        vol = abs(np.einsum("ij,ij->i", V[F[:, 0]], np.cross(V[F[:, 1]], V[F[:, 2]])).sum()) / 6.0
        try:
            hull = ConvexHull(V).volume
        except Exception:
            hull = 0.0
        convexity[m.mesh(mi).name] = vol / hull if hull > 1e-12 else 1.0
    print("  apparatus meshes: %d, colliding (convexity >= 0.85): %d" % (len(convexity), sum(c >= 0.85 for c in convexity.values())))
    for j in list(a.joints):
        a.delete(j)
    for g in list(a.geoms):
        if g.type == mujoco.mjtGeom.mjGEOM_PLANE:
            a.delete(g)
            continue
        if g.type == mujoco.mjtGeom.mjGEOM_MESH and convexity.get(g.meshname, 0.0) < 0.85:
            g.contype, g.conaffinity, g.group = 0, 0, 0      # notched (stair stringer, bevelled ramp): its hull would fill it
        else:
            g.contype, g.conaffinity, g.group = 1, 1, X.GROUP_WALL
    for b in a.bodies:
        if hasattr(b, "gravcomp"):
            b.gravcomp = 0
    # the export's z = 0 is the venue floor, the raster's the deck top (deck 4.25 in): lower it by the deck, 3 mm proud
    frame = spec.worldbody.add_frame(pos=[shift[0], shift[1], shift[2] - 4.25 * 0.0254 + 0.003])
    spec.attach(a, prefix="app%d_" % k, frame=frame)


def write_hfield_bin(path, grid01):
    """MuJoCo custom hfield file: int32 nrow, int32 ncol, then float32 row-major (row = y, col = x), values 0..1."""
    data = np.ascontiguousarray(grid01.T, np.float32)
    with open(path, "wb") as f:
        np.array(data.shape, np.int32).tofile(f)
        data.tofile(f)


def build(scene, out: pathlib.Path):
    out.mkdir(parents=True, exist_ok=True)
    rd = X.robot_dir()
    spec = mujoco.MjSpec.from_file(str(rd / "mjcf" / "x2_ultra.xml"))
    for cand in (rd / "urdf" / "x2_ultra" / "meshes", rd / "urdf" / "meshes"):
        if cand.exists():
            spec.meshdir = str(cand)
            break
    for g in spec.geoms:            # the soles as the eval scored Hiking (eval/x2.py build)
        if g.type == mujoco.mjtGeom.mjGEOM_SPHERE and g.parent.name.endswith("ankle_roll_link"):
            g.pos = [g.pos[0], g.pos[1], g.pos[2] + X.SOLE_R - g.size[0]]
            g.size = [X.SOLE_R, 0, 0]
    planes = [g for g in spec.geoms if g.type == mujoco.mjtGeom.mjGEOM_PLANE]
    grid, x0, y0, res, boxes, info = terrain(scene)
    for g in planes:
        g.group = X.GROUP_TERRAIN   # the depth caster sees geom groups 1 (robot), 4 (terrain), 5 (walls)
    if grid is None:
        add_surfaces(spec, out)
        for g in planes:
            g.material = "concrete"
    if grid is not None:
        zmin = float(grid.min())
        for g in planes:            # the MJCF's floor becomes a catch floor 5 cm under the lowest cell
            g.pos = [0, 0, zmin - 0.05]
        # The height field goes in CHUNK x CHUNK cell pieces (sharing their edge rows): one large field made each depth frame
        # cost 80-230 ms of ray casting (MuJoCo tests every cell under a ray's box); 2 m pieces are culled by their bounding boxes.
        add_surfaces(spec, out)
        nx, ny = grid.shape
        k = 0
        for i0 in range(0, nx - 1, CHUNK):
            for j0 in range(0, ny - 1, CHUNK):
                sub = grid[i0:i0 + CHUNK + 1, j0:j0 + CHUNK + 1]
                lo, hi = float(sub.min()), float(sub.max())
                top = max(hi - lo, 1e-3)
                write_hfield_bin(out / ("terrain_%03d.bin" % k), (sub - lo) / top)
                sx, sy = sub.shape
                spec.add_hfield(name="terrain_%03d" % k, file=str(out / ("terrain_%03d.bin" % k)),
                                size=[(sx - 1) * res / 2, (sy - 1) * res / 2, top, 0.05 + lo - zmin])
                t = spec.worldbody.add_geom(name="terrain_%03d" % k, type=mujoco.mjtGeom.mjGEOM_HFIELD,
                                            hfieldname="terrain_%03d" % k,
                                            material=surface_at(info, scene, x0 + (i0 + (sx - 1) / 2) * res, y0 + (j0 + (sy - 1) / 2) * res),
                                            pos=[x0 + (i0 + (sx - 1) / 2) * res, y0 + (j0 + (sy - 1) / 2) * res, lo])
                t.group = X.GROUP_TERRAIN
                k += 1
        info["hfield_chunks"] = k
        for i, b in enumerate(boxes):
            if APPARATUS_MODEL == "cad" and APPARATUS and "mat" not in b and in_apparatus(info, *b["center"][:2]):
                continue                    # the apparatus CAD model is the physics there (add_apparatus_model)
            w = spec.worldbody.add_geom(name="wall_%d" % i, type=mujoco.mjtGeom.mjGEOM_BOX, size=list(b["half"]),
                                        pos=list(b["center"]), quat=list(b["quat"]),
                                        material=b.get("mat") or ("hidden" if APPARATUS_MODEL == "cad" and APPARATUS and surface_at(info, scene, *b["center"][:2]) == "concrete"
                                                                  and in_apparatus(info, *b["center"][:2]) else "cedar"))
            w.group = X.GROUP_WALL
    spec.visual.headlight.ambient = [0.35, 0.35, 0.35]      # lighting as the eval renders (eval/x2.py): a sun with shadows
    spec.visual.headlight.diffuse = [0.45, 0.45, 0.45]
    spec.worldbody.add_light(name="sun", pos=[0, 0, 10], dir=[0.3, 0.2, -1], diffuse=[0.6, 0.6, 0.6], castshadow=True,
                             type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    if grid is not None:            # wooden 4x4s over their height-field ridges (visual only)
        for k, bm in enumerate(BEAMS):
            a_, b_ = np.array(bm["a"], float), np.array(bm["b"], float)
            c_, L = (a_ + b_) / 2, float(np.linalg.norm(b_ - a_))
            ang = math.atan2(b_[1] - a_[1], b_[0] - a_[0])
            i, j = int(round((c_[0] - x0) / res)), int(round((c_[1] - y0) / res))
            top = float(grid[max(i - 3, 0):i + 4, max(j - 3, 0):j + 4].max())
            v = spec.worldbody.add_geom(name="beam_vis_%d" % k, type=mujoco.mjtGeom.mjGEOM_BOX, size=[L / 2, 0.0455, 0.0455],
                                        pos=[c_[0], c_[1], top - 0.0445], quat=[math.cos(ang / 2), 0, 0, math.sin(ang / 2)],
                                        material="wood")
            v.contype, v.conaffinity, v.group = 0, 0, 0
        info["beams_drawn"] = len(BEAMS)
        for k, v in enumerate(VIS):          # extended-map props already in the raster: visual only, 2 mm proud
            g = spec.worldbody.add_geom(name="vis_%d" % k, type=mujoco.mjtGeom.mjGEOM_BOX, size=[h + 0.002 for h in v["half"]],
                                        pos=v["center"], quat=v["quat"], material=v["mat"])
            g.contype, g.conaffinity, g.group = 0, 0, 0
        for k, mm in enumerate(MESHES):      # meshes: collision (group 5, the depth camera sees them) if not in the raster
            spec.add_mesh(name="tpmesh_%d" % k, uservert=mm["verts"].ravel().tolist(), userface=mm["faces"].ravel().tolist())
            g = spec.worldbody.add_geom(name="tpmesh_%d" % k, type=mujoco.mjtGeom.mjGEOM_MESH, meshname="tpmesh_%d" % k,
                                        material=mm["mat"])
            if mm["collide"]:
                g.group = X.GROUP_WALL
            else:
                g.contype, g.conaffinity, g.group = 0, 0, 0
        info["tp_visual_boxes"], info["tp_meshes"] = len(VIS), len(MESHES)
        if APPARATUS_MODEL == "cad":         # terrain-park's CAD export in place of the eval's walls (see APPARATUS_MODEL)
            for k, sh in enumerate(APPARATUS):
                add_apparatus_model(spec, sh, k)
        info["apparatus_model"] = APPARATUS_MODEL
        info["apparatus_models"] = len(APPARATUS) if APPARATUS_MODEL == "cad" else 0
        for t in info.get("tiles", []):  # props baked into the height field (wedges): a visual copy of the raised cells only
            if "cells" not in t:
                continue
            i0, i1, j0, j1 = t["cells"]
            sub = grid[i0:i1 + 1, j0:j1 + 1]
            ground = float(np.median(sub))
            ov = np.where(sub > ground + 0.01, sub + 0.003, ground - 0.05)
            lo, top = float(ov.min()), max(float(ov.max() - ov.min()), 1e-3)
            name = "props_%s" % t["course"]
            write_hfield_bin(out / (name + ".bin"), (ov - lo) / top)
            sx, sy = ov.shape
            spec.add_hfield(name=name, file=str(out / (name + ".bin")), size=[(sx - 1) * res / 2, (sy - 1) * res / 2, top, 0.001])
            g = spec.worldbody.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname=name, material=PROP_TILES[t["course"]],
                                        pos=[x0 + (i0 + (sx - 1) / 2) * res, y0 + (j0 + (sy - 1) / 2) * res, lo])
            g.contype, g.conaffinity, g.group = 0, 0, 0
            t.pop("cells")
    (out / "scene.xml").write_text(spec.to_xml())
    m = mujoco.MjModel.from_xml_path(str(out / "scene.xml"))      # round trip: what the bridge will load
    info.update(scene=scene, ngeom=m.ngeom, boxes=len(boxes), mujoco=mujoco.__version__,
                grid=None if grid is None else list(grid.shape), x0=x0, y0=y0, res=res)
    (out / "scene.json").write_text(json.dumps(info, indent=1, default=float))
    print("scene %s -> %s (%d geoms, %d boxes, mujoco %s)" % (scene, out / "scene.xml", m.ngeom, len(boxes), mujoco.__version__))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="park")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    a = ap.parse_args()
    build(a.scene, a.out)
