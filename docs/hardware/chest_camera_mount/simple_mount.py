"""Simplest chest mount for the Hiking camera pose: one solid block, no cage, no screws, no cut-outs, prints without supports (lying on the camera face).

- Front-lower face: flat, the D435's underside is taped to it (VHB, 1.1 mm).
- Back face: follows the chest shell (offset by the tape), only on the fixed upper shell, >= 4 mm above the panel seam.
- The block is a straight prism along y (the camera's length) except that its back face follows the chest, so printed
  lying on the camera face, the side walls lean at most ~25 deg and the chest face is the top surface: no supports.
Camera pose = Hiking training pose: depth origin (131, 0, 259) mm on torso_link, 61.6 deg down (as build_mount.py).

    python simple_mount.py      # writes simple_mount_hiking_print.stl, simple_mount_hiking_robot_frame.stl, simple_mount.png
"""
import json, math, os
import numpy as np
import trimesh
from build_mount import heightfield_solid, to_trimesh, interp_shell, TAPE, HERE
import manifold3d as mf

PITCH = 61.6                         # deg below horizontal (optical axis)
O = np.array([131.0, 0.0, 259.0])    # depth origin (left imager), torso_link, mm
GLASS = 4.2                          # depth origin is this far behind the front glass
CAM = (90.0, 25.0, 25.0)             # D435 width (y), height, depth
CASE_Y = -17.5                       # case centre: left imager sits on the robot centreline
LENGTH = 70.0                        # block length along y, centred under the camera
Z_LOW, Z_TOP = 255.0, 276.0          # where the block's lower / upper faces meet the back (behind the shell)
SEAT = 15.0                          # length of the camera seat along the optical axis (from the camera's back edge)
X_BACK = 80.0                        # profile corners behind the chest (cut away by the shell)

t = math.radians(PITCH)
a = np.array([math.cos(t), 0.0, -math.sin(t)])       # optical axis (forward, down)
u = np.array([math.sin(t), 0.0, math.cos(t)])        # camera up
C = O + (GLASS - CAM[2] / 2) * a + np.array([0.0, CASE_Y, 0.0])  # case centre
Cb = C - (CAM[1] / 2 + TAPE) * u                     # camera underside, offset by the tape
B2 = Cb + (CAM[2] / 2) * a                           # front edge (glass plane)
B1 = Cb - (CAM[2] / 2) * a                           # back edge
S = B1 + SEAT * a                                    # front end of the seat: no thin wedge tip at the glass
prof = [(S[0], S[2]), (B1[0], B1[2]), (X_BACK, Z_TOP), (X_BACK, Z_LOW)]   # x-z profile, orientation fixed below

y0, y1 = CASE_Y - LENGTH / 2, CASE_Y + LENGTH / 2
# prism: profile in x-z extruded along y
P = np.array(prof)
e1, e2 = P[1] - P[0], P[2] - P[1]
if e1[0] * e2[1] - e1[1] * e2[0] < 0:
    P = P[::-1]
V = np.array([[x, y, z] for y in (y0, y1) for x, z in P])
n = len(P)
F = [[0, i, i + 1] for i in range(1, n - 1)] + [[n, n + i + 1, n + i] for i in range(1, n - 1)]
for i in range(n):
    j = (i + 1) % n
    F += [[i, n + i, j], [j, n + i, n + j]]
prism = trimesh.Trimesh(V, F, process=True); prism.fix_normals()
assert prism.is_watertight
prism = mf.Manifold(mf.Mesh(vert_properties=prism.vertices.astype(np.float32), tri_verts=prism.faces.astype(np.uint32)))

d = np.load(os.path.join(HERE, "shell_heightmap.npz"))
ys, zs, X = d["ys"], d["zs"], d["X"]
seam_hi = d["seam_hi"]
gy = np.arange(y0 - 3, y1 + 3.01, 0.5); gz = np.arange(Z_LOW - 30, Z_TOP + 10.01, 0.5)
GY, GZ = np.meshgrid(gy, gz)
shell = interp_shell(ys, zs, X, GY, GZ)
assert np.all(shell > 40), "shell heightmap does not cover the block"
robot = heightfield_solid(gy, gz, np.full_like(shell, 0.0), shell + TAPE)    # chest + tape layer
block = prism - robot
tm = to_trimesh(block)
parts = tm.split(only_watertight=False)
tm = max(parts, key=lambda c: c.volume)

# checks
contact = tm.vertices[np.abs(tm.vertices[:, 0] - (interp_shell(ys, zs, X, tm.vertices[:, 1], tm.vertices[:, 2]) + TAPE)) < 0.3]
seam_at = np.interp(contact[:, 1], ys, seam_hi)
margin = float((contact[:, 2] - seam_at).min())
# area of the chest-side faces (normals pointing back, -x)
fn = tm.face_normals; back = fn[:, 0] < -0.5
chest_area = float(tm.area_faces[back].sum()) / 100.0
cam_area = LENGTH * SEAT / 100.0
# nothing ahead of the glass plane (outside the depth view)
tglass = float(np.max((tm.vertices - O) @ a))
# print orientation: lie on the camera seat (flat, smooth face on the bed): the seat's outward normal +u -> -z
Rp = trimesh.geometry.align_vectors(u, [0.0, 0.0, -1.0])
pr = tm.copy(); pr.apply_transform(Rp); pr.apply_translation(-pr.bounds[0])
# overhang: faces pointing down by more than 45 deg (excluding the bed face)
nz = pr.face_normals[:, 2]; cz = pr.triangles_center[:, 2]
overhang = float(pr.area_faces[(nz < -math.cos(math.radians(45))) & (cz > 0.3)].sum()) / 100.0
info = dict(volume_cm3=round(tm.volume / 1000, 1), mass_g_pla_100pct=round(tm.volume / 1000 * 1.24, 0),
            watertight=bool(tm.is_watertight), bounds_robot_mm=np.round(tm.bounds, 1).tolist(),
            print_size_mm=np.round(pr.extents, 1).tolist(), overhang_cm2=round(overhang, 2),
            chest_tape_area_cm2=round(chest_area, 1), camera_tape_area_cm2=round(cam_area, 1),
            chest_contact_z_mm=[round(float(contact[:, 2].min()), 1), round(float(contact[:, 2].max()), 1)],
            contact_min_above_seam_mm=round(margin, 1), max_ahead_of_depth_origin_along_axis_mm=round(tglass, 1),
            camera_seat_front_edge_mm=np.round(S, 1).tolist(), camera_seat_back_edge_mm=np.round(B1, 1).tolist())
tm.export(os.path.join(HERE, "simple_mount_hiking_robot_frame.stl"))
pr.export(os.path.join(HERE, "simple_mount_hiking_print.stl"))
json.dump(info, open(os.path.join(HERE, "simple_mount_summary.json"), "w"), indent=1)
print(json.dumps(info, indent=1))

# picture: side section at the camera centre and the print view
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
sec = tm.section(plane_origin=[0, CASE_Y, 0], plane_normal=[0, 1, 0])
A = ax[0]
for e in sec.discrete:
    A.fill(e[:, 0], e[:, 2], color="#e8913a", alpha=0.85)
zz = np.arange(200, 300, 0.5)
A.plot(interp_shell(ys, zs, X, np.full_like(zz, CASE_Y), zz), zz, "k-", lw=1.2, label="chest shell")
cam = [C + sa * a * CAM[2] / 2 + sb * u * CAM[1] / 2 for sa, sb in ((1, 1), (1, -1), (-1, -1), (-1, 1), (1, 1))]
A.plot([p[0] for p in cam], [p[2] for p in cam], "-", color="#333", lw=2, label="D435")
A.axhline(float(np.interp(CASE_Y, ys, seam_hi)), color="m", ls="--", lw=1, label="panel seam (below = detachable)")
A.plot([O[0], O[0] + 60 * a[0]], [O[2], O[2] + 60 * a[2]], "r--", lw=1, label="optical axis")
A.set_aspect("equal"); A.set_xlim(70, 170); A.set_ylim(220, 300); A.grid(alpha=0.3); A.legend(loc="lower right", fontsize=8)
A.set_title("side section at the camera centre (y = %.1f mm)" % CASE_Y); A.set_xlabel("x forward (mm)"); A.set_ylabel("z up (mm)")
B = ax[1]
sp = pr.section(plane_origin=[0, pr.extents[1] / 2, 0], plane_normal=[0, 1, 0])
for e in sp.discrete:
    B.fill(e[:, 0], e[:, 2], color="#e8913a", alpha=0.85)
B.axhline(0, color='k', lw=2)
B.set_aspect("equal"); B.grid(alpha=0.3)
B.set_title("print: camera face down on the bed, %.0f x %.0f mm, %.0f mm tall, no supports" % (pr.extents[0], pr.extents[1], pr.extents[2]))
fig.suptitle("Simple Hiking chest mount: one block, tape both faces; volume %.1f cm3" % info["volume_cm3"])
fig.tight_layout(); fig.savefig(os.path.join(HERE, "simple_mount.png"), dpi=110)
