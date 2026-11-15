"""Scene dressing for renders only (eval monitor, 2026-11-14, user: renders "not the true color of the apparatus or extended map
- too much saturation and single colored").

    x = X.build(app, wall_alpha=0.30, render_size=size, dress=dress.for_course(app))

Adds visual-only geoms (contype 0, conaffinity 0, group 0) before compile; the physics height field and wall boxes stay in the
model unchanged (render.py hides their groups 4 and 5 when a scene is dressed, so the depth rays of the adapters, which use
groups 1, 4 and 5, see exactly what they saw before).
  apparatus 'ours': every visible geom of the apparatus's own MJCF (loco-manip-apparatus sim/scene_x2_env_mobility.xml, the
          source of the eval raster and of cad_walls.json, same frame) with its own material (pine, cedar stain, birch, OSB,
          plywood textures, paint, galvanised steel ...). Not drawn: the robot, the loose manipulation props (totes, crates,
          bulbs, drill: free bodies, not part of the eval physics) (the venue floor is redrawn with its own dark blue checker).
  extended-map tile (ladder:...): every box and mesh of the tile's export (the physics primitives) in a muted colour per
          primitive class (stair treads, stair walls, posts / rails, wedges, beams / logs, blocks, homes furniture, walls).
Floor everywhere: the venue floor of the apparatus MJCF (dark blue checker, as in the user's earlier SuperMimic renders).
Lighting (both, and the flat plane): headlight ambient 0.30 / diffuse 0.32, one sun 0.42 with soft shadows, low specular.
"""
from __future__ import annotations

import json
import os
import pathlib
import re

import mujoco
import numpy as np

APP_SCENE = pathlib.Path(os.environ.get("APPARATUS_SCENE", r"C:\Users\PC\projects\loco-manip-apparatus\sim\scene_x2_env_mobility.xml"))
GRID = dict(rgb1=[0.2, 0.3, 0.4], rgb2=[0.1, 0.2, 0.3], markrgb=[0.8, 0.8, 0.8])   # the venue floor of the apparatus MJCF (user's earlier renders)

# extended map: label (prefix match, first hit wins) -> rgba; muted, realistic material colours
TILE_RGBA = [
    ("ground", None),                                    # neutral grid material
    ("stair_step", [0.66, 0.53, 0.38, 1]),               # oak treads
    ("stair_side:wall", [0.80, 0.78, 0.73, 1]),          # painted stair wall
    ("stair_side:handrail", [0.60, 0.61, 0.62, 1]),      # brushed steel
    ("stair_side:post", [0.40, 0.30, 0.21, 1]),          # walnut posts
    ("ring_sector", [0.66, 0.65, 0.62, 1]),              # concrete amphitheatre seats
    ("wedge", [0.76, 0.64, 0.47, 1]),                    # plywood ramps
    ("rail_fence:post", [0.45, 0.37, 0.28, 1]),          # weathered fence
    ("rail_fence:rail", [0.52, 0.43, 0.32, 1]),
    ("brace_stand:board", [0.79, 0.67, 0.48, 1]),        # pine
    ("brace_stand", [0.58, 0.47, 0.34, 1]),
    ("beam_support", [0.36, 0.28, 0.21, 1]),
    ("beam", [0.55, 0.40, 0.27, 1]),                     # 4x4 / log
    ("corridor_side:wall", [0.76, 0.73, 0.66, 1]),
    ("corridor_side", [0.42, 0.33, 0.25, 1]),
    ("chair", [0.44, 0.31, 0.22, 1]),                    # homes furniture
    ("table", [0.63, 0.49, 0.34, 1]),
    ("door_leaf", [0.58, 0.44, 0.31, 1]),
    ("door_header", [0.86, 0.85, 0.82, 1]),
    ("wall", [0.84, 0.82, 0.77, 1]),                     # painted interior wall
    ("cylinder", [0.48, 0.50, 0.52, 1]),
    ("box", [0.63, 0.50, 0.34, 1]),                      # cardboard
]
BLOCK_RGBA = {"homes": [0.43, 0.47, 0.54, 1], "street_front": [0.60, 0.59, 0.56, 1]}   # sofas / kerbs; elsewhere concrete
BLOCK_DEFAULT = [0.62, 0.61, 0.58, 1]
WALL_WORDS = ("wall", "post", "rail", "fence", "brace", "door", "header", "corridor_side")


def lighting(spec):
    spec.visual.headlight.ambient = [0.25, 0.25, 0.25]       # the apparatus MJCF's headlight, specular down
    spec.visual.headlight.diffuse = [0.38, 0.38, 0.38]
    spec.visual.headlight.specular = [0.02, 0.02, 0.02]
    spec.visual.map.znear = 0.002                        # near clip (fraction of the extent) small enough for the chest camera
    for li in spec.lights:
        if li.name == "sun":
            li.diffuse = [0.42, 0.42, 0.41]
            li.specular = [0.04, 0.04, 0.04]
            li.ambient = [0.0, 0.0, 0.0]
        if li.name == "tracking":                            # the robot MJCF's spot over the robot: it blew out the floor under it
            li.diffuse = [0.18, 0.18, 0.18]
            li.specular = [0.0, 0.0, 0.0]
    spec.worldbody.add_light(name="fill", pos=[12, 8, 10], dir=[-0.4, -0.3, -1], diffuse=[0.22, 0.22, 0.22], specular=[0, 0, 0],
                             castshadow=False, type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)


def _grid_material(spec):
    spec.add_texture(name="dress_grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     mark=mujoco.mjtMark.mjMARK_EDGE, width=300, height=300, **GRID)
    m = spec.add_material(name="dress_grid_mat", texuniform=True, texrepeat=[5, 5], specular=0.05, shininess=0.1, reflectance=0.12)
    m.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "dress_grid"
    return "dress_grid_mat"


def _geom(spec, name, gtype, size, pos, quat, rgba=None, material=None, mesh=None):
    g = spec.worldbody.add_geom(name=name, type=gtype, size=list(size), pos=list(pos), quat=list(quat))
    if mesh is not None:
        g.meshname = mesh
    if material:
        g.material = material
    elif rgba is not None:
        g.rgba = list(rgba)
    g.contype, g.conaffinity, g.group = 0, 0, 0
    g.density = 0.0
    return g


class Info:
    def __init__(self, kind, walls, note):
        self.kind, self.wall_names, self.note = kind, walls, note

    def finish(self, m):
        self.walls = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in self.wall_names}
        self.ids = {g for g in range(m.ngeom) if m.geom(g).name.startswith("dress_")}
        return self


def apparatus(spec, app):
    lighting(spec)
    src = mujoco.MjModel.from_xml_path(str(APP_SCENE))
    ssp = mujoco.MjSpec.from_file(str(APP_SCENE))
    d = mujoco.MjData(src)
    mujoco.mj_forward(src, d)
    walls_ids = {int(b["geom"]) for b in app.boxes if "geom" in b}
    free = {src.jnt_bodyid[j] for j in range(src.njnt) if src.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE}
    root = src.body_rootid[src.geom_bodyid]
    tex = {t.name: t.file for t in ssp.textures if t.file}
    mats = {}
    for mt in ssp.materials:
        if not mt.name.startswith("m_"):
            continue
        t = mt.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB]
        kw = dict(rgba=list(mt.rgba), specular=0.08, shininess=0.15, reflectance=0.0)
        nm = spec.add_material(name="dress_" + mt.name, **kw)
        if t and t in tex:
            spec.add_texture(name="dress_" + t, type=mujoco.mjtTexture.mjTEXTURE_2D, file=tex[t])
            nm.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "dress_" + t
            nm.texuniform, nm.texrepeat = bool(mt.texuniform), list(mt.texrepeat)
        mats[mt.name] = "dress_" + mt.name
    grid = _grid_material(spec)
    walls, n = [], 0
    for g in range(src.ngeom):
        if src.geom_group[g] > 2 or root[g] in free:
            continue
        mid = src.geom_matid[g]
        mname = src.material(mid).name if mid >= 0 else None
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, d.geom_xmat[g])
        if src.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:          # the venue floor: neutral grid at its height
            _geom(spec, "dress_floor", mujoco.mjtGeom.mjGEOM_PLANE, [30, 30, 0.1], d.geom_xpos[g], [1, 0, 0, 0], material=grid)
            continue
        rgba = src.geom_rgba[g] if mname is None else None
        if rgba is not None and rgba[3] < 0.05:
            continue
        name = ("dress_w_%d" if g in walls_ids else "dress_%d") % g
        mesh = None
        if src.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            k = src.geom_dataid[g]
            va, vn = src.mesh_vertadr[k], src.mesh_vertnum[k]
            fa, fn = src.mesh_faceadr[k], src.mesh_facenum[k]
            mesh = "dress_mesh_%d" % g
            spec.add_mesh(name=mesh, uservert=src.mesh_vert[va:va + vn].ravel().tolist(), userface=src.mesh_face[fa:fa + fn].ravel().tolist())
        _geom(spec, name, src.geom_type[g], src.geom_size[g], d.geom_xpos[g], q, rgba=rgba, material=mats.get(mname), mesh=mesh)
        if g in walls_ids:
            walls.append(name)
        n += 1
    return Info("apparatus", walls, "apparatus drawn from its own MJCF and materials (%d geoms; robot, loose props and the venue floor colour left out; floor = the venue's dark blue checker)" % n)


def _tile_of(course):
    tid = course.route.get("tile")
    m = re.match(r"^(.*)_L\d+_v\d+$", str(tid))
    return (m.group(1), tid) if m else (None, tid)


def tile(spec, course):
    import ladder
    lighting(spec)
    ttype, tid = _tile_of(course)
    D = ladder.terrain_dir()
    J = json.load(open(D / (ttype + ".json")))
    t = [e for e in (J if isinstance(J, list) else J["tiles"]) if e["tile_id"] == tid][0]
    z = np.load(D / (ttype + ".npz"), allow_pickle=False)
    i = int(t["index"])
    labels = [str(s) for s in z["labels"]]
    grid = _grid_material(spec)

    def rgba_of(lab):
        if lab.startswith("block") or lab == "block":
            return BLOCK_RGBA.get(ttype, BLOCK_DEFAULT)
        for k, c in TILE_RGBA:
            if lab.startswith(k) or (":" in lab and lab.split(":")[0].startswith(k) and c is not None and k == lab.split(":")[0]):
                return c
        return [0.60, 0.58, 0.54, 1]

    walls, n = [], 0
    g0 = float(np.asarray(z["height"][i]).min())
    _geom(spec, "dress_floor", mujoco.mjtGeom.mjGEOM_PLANE, [40, 40, 0.1], [0, 0, g0 - 0.003], [1, 0, 0, 0], material=grid)
    for k in range(int(z["box_start"][i]), int(z["box_start"][i + 1])):
        lab = labels[int(z["box_label"][k])]
        wall = any(w in lab for w in WALL_WORDS)
        name = ("dress_w_b%d" if wall else "dress_b%d") % k
        if lab == "ground":
            _geom(spec, name, mujoco.mjtGeom.mjGEOM_BOX, z["box_half"][k], z["box_center"][k], z["box_quat"][k], material=grid)
        else:
            _geom(spec, name, mujoco.mjtGeom.mjGEOM_BOX, z["box_half"][k], z["box_center"][k], z["box_quat"][k], rgba=rgba_of(lab))
        if wall:
            walls.append(name)
        n += 1
    for mth in range(int(z["mesh_start"][i]), int(z["mesh_start"][i + 1])):
        lab = labels[int(z["mesh_label"][mth])]
        v = z["mesh_vertices"][int(z["mesh_vert_start"][mth]):int(z["mesh_vert_start"][mth + 1])].astype(float)
        f = z["mesh_faces"][int(z["mesh_face_start"][mth]):int(z["mesh_face_start"][mth + 1])].astype(int)
        if len(v) < 4 or len(f) < 1:
            continue
        if f.max() >= len(v):                               # faces stored with global vertex indices
            f = f - int(z["mesh_vert_start"][mth])
        mesh = "dress_mesh_%d" % mth
        spec.add_mesh(name=mesh, uservert=v.ravel().tolist(), userface=f.ravel().tolist())
        _geom(spec, "dress_m%d" % mth, mujoco.mjtGeom.mjGEOM_MESH, [1, 1, 1], [0, 0, 0], [1, 0, 0, 0], rgba=rgba_of(lab), mesh=mesh)
        n += 1
    return Info("tile", walls, "extended-map tile %s drawn from its export primitives (%d), colour per primitive class; floor = the venue's dark blue checker" % (tid, n))


def flat(spec):
    lighting(spec)
    for g in spec.geoms:
        if g.name == "terrain":
            g.material = _grid_material(spec)
    return Info("flat", [], "flat plane, neutral grid")


def for_course(app):
    """The dressing callable for a course (None: nothing known to draw, keep the plain render)."""
    if app is None:
        return flat
    if getattr(app, "name", "") == "ours" and APP_SCENE.exists():
        return lambda spec: apparatus(spec, app)
    if isinstance(getattr(app, "route", None), dict) and _tile_of(app)[0]:
        return lambda spec: tile(spec, app)
    return None
