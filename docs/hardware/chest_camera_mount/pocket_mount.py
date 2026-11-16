"""C-pocket chest mount for the Hiking camera pose (user sketch, 2026-11-16): a straight prism along the camera's length
that wraps the D435's back face, most of its top face (lip) and the rear part of its bottom face, and fills down to the
chest. Open at the front (lenses) and at both ends (the camera's side vents). The camera slides in snugly and is held by
VHB tape on its back face. Printed upside down: the flat top face on the bed, so nothing overhangs.

    python pocket_mount.py   # writes pocket_mount_hiking_print.stl, pocket_mount_hiking_robot_frame.stl, pocket_mount.png,
                             # pocket_mount_summary.json
"""
import json, math, os
import numpy as np
import trimesh
import manifold3d as mf
from build_mount import heightfield_solid, to_trimesh, interp_shell, TAPE, HERE

PITCH = 61.6                         # optical axis, deg below horizontal (Hiking training pose)
O = np.array([131.0, 0.0, 259.0])    # depth origin (left imager) on torso_link, mm
GLASS = 4.2                          # depth origin behind the front glass
CAM_W, CAM_H, CAM_D = 90.0, 25.0, 25.0
CASE_Y = -17.5                       # case centre (left imager on the robot centreline)
LENGTH = 70.0                        # holder length along y, centred on the camera (camera ends stick out 10 mm each side)
FIT = 0.3                            # clearance on the camera's top and bottom faces (snug slide fit)
WALL = 4.0                           # back wall and lip thickness
LIP = 18.0                           # lip covers the top face this far from the back edge (7 mm short of the glass)
BOTTOM = 15.0                        # bottom support covers the bottom face this far from the back edge
Z_TOP = 290.0                        # flat top face (the print bed face)
Z_LOW, X_BACK = 255.0, 60.0          # bottom face runs from the bottom support's tip back to (X_BACK, Z_LOW) behind the chest

t = math.radians(PITCH)
a = np.array([math.cos(t), -math.sin(t)])            # x-z: optical axis
u = np.array([math.sin(t), math.cos(t)])             # x-z: camera up
C = np.array([O[0], O[2]]) + (GLASS - CAM_D / 2) * a  # case centre (x, z)


def cam(pa, pu):                                     # camera-local (along axis, along up) -> robot (x, z)
    return C + pa * a + pu * u


hb, hh = CAM_D / 2, CAM_H / 2
back_out = -hb - TAPE - WALL                          # outer back of the back wall
lip_out = hh + FIT + WALL
pts = [cam(back_out, lip_out), cam(-hb + LIP, lip_out), cam(-hb + LIP, hh + FIT),            # lip
       (cam(back_out, lip_out)[0], Z_TOP),                                                   # top face exactly flat (bed face)
       cam(back_out, -hh - FIT), cam(-hb + BOTTOM, -hh - FIT),                                 # back wall bottom, bottom tip
       (X_BACK, Z_TOP), (X_BACK, Z_LOW)]
outer = mf.CrossSection.hull_points([tuple(map(float, p)) for p in pts])
top_clip = mf.CrossSection.square([400.0, 400.0]).translate([-100.0, Z_TOP - 400.0])          # keep z <= Z_TOP
cavity = mf.CrossSection.hull_points([tuple(map(float, p)) for p in (
    cam(-hb - TAPE, -hh - FIT), cam(-hb - TAPE, hh + FIT), cam(80.0, hh + FIT), cam(80.0, -hh - FIT))])   # open to the front
prof2d = (outer ^ top_clip) - cavity
# extrude along +z of the cross-section, then map (X, Y, Z) -> (x = X, y = Z, z = Y) with the y range centred
solid = prof2d.extrude(LENGTH)
y0 = CASE_Y - LENGTH / 2
Mxyz = [[1, 0, 0, 0], [0, 0, 1, y0], [0, 1, 0, 0]]                     # rows: x = X ; y = Z + y0 ; z = Y
solid = solid.transform(Mxyz)
if solid.volume() < 0:
    raise SystemExit("bad orientation")

d = np.load(os.path.join(HERE, "shell_heightmap.npz"))
ys, zs, X = d["ys"], d["zs"], d["X"]; seam_hi = d["seam_hi"]
gy = np.arange(y0 - 3, y0 + LENGTH + 3.01, 0.5); gz = np.arange(Z_LOW - 30, Z_TOP + 3.01, 0.5)
GY, GZ = np.meshgrid(gy, gz)
shell = interp_shell(ys, zs, X, GY, GZ)
assert np.all(shell > 40), "shell heightmap does not cover the holder"
robot = heightfield_solid(gy, gz, np.full_like(shell, 0.0), shell + TAPE)
tm = to_trimesh(solid - robot)
tm = max(tm.split(only_watertight=False), key=lambda c: c.volume)

# checks
V = tm.vertices
sh = interp_shell(ys, zs, X, V[:, 1], V[:, 2]) + TAPE
contact = V[np.abs(V[:, 0] - sh) < 0.3]
margin = float((contact[:, 2] - np.interp(contact[:, 1], ys, seam_hi)).min())
a3 = np.array([a[0], 0.0, a[1]])
ahead = float(np.max((V - O) @ a3))                   # most forward point along the optical axis vs the depth origin
pr = tm.copy(); pr.apply_transform(trimesh.transformations.rotation_matrix(math.pi, [1, 0, 0]))   # top face on the bed
pr.apply_translation(-pr.bounds[0])
nz = pr.face_normals[:, 2]; cz = pr.triangles_center[:, 2]
down = (nz < -1e-3) & (cz > 0.3)
worst = float(np.degrees(np.arcsin(np.clip(-nz[down], 0, 1))).max()) if down.any() else 0.0   # 0 = vertical wall, 90 = ceiling
bed_area = float(pr.area_faces[(nz < -0.999) & (cz < 0.05)].sum()) / 100.0
foot = float(pr.convex_hull.area_faces[(pr.convex_hull.face_normals[:, 2] < -0.999)].sum()) / 100.0
over45 = float(pr.area_faces[(nz < -math.cos(math.radians(45))) & (cz > 0.3)].sum()) / 100.0
back_area = LENGTH * CAM_H / 100.0
info = dict(volume_cm3=round(tm.volume / 1000, 1), mass_g_pla_100pct=round(tm.volume / 1000 * 1.24),
            watertight=bool(tm.is_watertight), print_size_mm=np.round(pr.extents, 1).tolist(),
            worst_downward_face_deg_from_vertical=round(worst, 1), bed_contact_cm2=round(bed_area, 1), overhang_over_45deg_cm2=round(over45, 2),
            chest_contact_z_mm=[round(float(contact[:, 2].min()), 1), round(float(contact[:, 2].max()), 1)],
            chest_contact_min_above_seam_mm=round(margin, 1), camera_back_tape_area_cm2=round(back_area, 1),
            lip_covers_top_face_mm=LIP, bottom_covers_bottom_face_mm=BOTTOM, fit_clearance_mm=FIT, tape_gap_back_mm=TAPE,
            most_forward_point_vs_depth_origin_mm=round(ahead, 1), bounds_robot_mm=np.round(tm.bounds, 1).tolist())
tm.export(os.path.join(HERE, "pocket_mount_hiking_robot_frame.stl"))
pr.export(os.path.join(HERE, "pocket_mount_hiking_print.stl"))
json.dump(info, open(os.path.join(HERE, "pocket_mount_summary.json"), "w"), indent=1)
print(json.dumps(info, indent=1))

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(13, 6))
A = ax[0]
for e in tm.section(plane_origin=[0, CASE_Y, 0], plane_normal=[0, 1, 0]).discrete:
    A.fill(e[:, 0], e[:, 2], color="#e8913a", alpha=0.85)
zz = np.arange(200, 305, 0.5)
A.plot(interp_shell(ys, zs, X, np.full_like(zz, CASE_Y), zz), zz, "k-", lw=1.2, label="chest shell")
box = [cam(sa * hb, sb * hh) for sa, sb in ((1, 1), (1, -1), (-1, -1), (-1, 1), (1, 1))]
A.plot([p[0] for p in box], [p[1] for p in box], "-", color="#333", lw=2, label="D435 (lenses at lower right)")
A.axhline(float(np.interp(CASE_Y, ys, seam_hi)), color="m", ls="--", lw=1, label="panel seam (below = detachable)")
A.plot([O[0], O[0] + 60 * a[0]], [O[2], O[2] + 60 * a[1]], "r--", lw=1, label="optical axis")
A.annotate("print bed side", xy=(110, Z_TOP), xytext=(95, 298), arrowprops=dict(arrowstyle="->", color="b"), color="b")
A.set_aspect("equal"); A.set_xlim(70, 170); A.set_ylim(230, 305); A.grid(alpha=0.3); A.legend(loc="lower right", fontsize=8)
A.set_title("side section at the camera centre (y = %.1f mm)" % CASE_Y); A.set_xlabel("x forward (mm)"); A.set_ylabel("z up (mm)")
B = ax[1]
for e in pr.section(plane_origin=[0, pr.extents[1] / 2, 0], plane_normal=[0, 1, 0]).discrete:
    B.fill(e[:, 0], e[:, 2], color="#e8913a", alpha=0.85)
B.axhline(0, color="b", lw=3)
B.set_aspect("equal"); B.grid(alpha=0.3)
B.set_title("print section: flat top face on the bed (blue), %.0f x %.0f mm, %.0f mm tall" % tuple(pr.extents[[0, 1, 2]]))
fig.suptitle("Pocket holder: camera slides in, tape on its back; ends and lenses open; volume %.1f cm3" % info["volume_cm3"])
fig.tight_layout(); fig.savefig(os.path.join(HERE, "pocket_mount.png"), dpi=100)
