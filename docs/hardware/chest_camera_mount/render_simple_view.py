"""Show how the chest camera (simple mount, Hiking training pose) looks down: side view with the view frustum, the camera's
own view (RGB render) with a 4x4 on the floor 0.5 m ahead, and the policy's depth input (64 x 36, cropped to 32 columns).

    X2_ASSETS=.../ssr-x2/assets/robot python render_simple_view.py      # writes simple_mount_view.png
"""
import os
import numpy as np
import mujoco
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from build_mount import stand, HERE

ASSETS = os.environ.get("X2_ASSETS", "assets/robot")
PITCH = 61.6
O = np.array([131.0, 0.0, 259.0]) / 1000              # depth origin on torso_link (m)
HFOV, VFOV = 87.0, 58.0
BEAM_X = 0.50                                         # 4x4 (89 mm) this far ahead of the toes... set below from the feet

t = np.deg2rad(PITCH)
a = np.array([np.cos(t), 0, -np.sin(t)]); ey = np.array([0.0, 1, 0]); u = np.array([np.sin(t), 0, np.cos(t)])

spec = mujoco.MjSpec.from_file(os.path.join(ASSETS, "mjcf", "x2_ultra.xml"))
spec.meshdir = os.path.join(ASSETS, "urdf", "x2_ultra", "meshes")
spec.visual.global_.offwidth = 1280; spec.visual.global_.offheight = 1280
tb = spec.body("torso_link")
spec.add_mesh(name="mount", file=os.path.join(HERE, os.environ.get("MOUNT_STL", "simple_mount_hiking_robot_frame.stl")), scale=[0.001] * 3)
tb.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="mount", rgba=[0.95, 0.55, 0.15, 1], contype=0, conaffinity=0, group=2)
# D435 body: 90 x 25 x 25 mm, case centre 8.3 mm behind the depth origin along the axis, 17.5 mm to the robot's right
cc = O + (4.2 - 12.5) / 1000 * a - ey * 0.0175
Mb = np.stack([ey, a, -u], 1); q = np.zeros(4); mujoco.mju_mat2Quat(q, Mb.ravel())   # right-handed: ey x a = -u
tb.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.045, 0.0125, 0.0125], pos=cc.tolist(), quat=q.tolist(), rgba=[0.12, 0.12, 0.13, 1],
            contype=0, conaffinity=0, group=2)
# the depth camera itself (MuJoCo cameras look along -z, +y up): x = camera right = -ey, y = up = u
tb.add_camera(name="d435", pos=O.tolist(), xyaxes=(-ey).tolist() + u.tolist(), fovy=VFOV)
wb = spec.worldbody
wb.add_light(pos=[0.5, 0.5, 3.0], dir=[-0.2, -0.2, -1], diffuse=[0.6, 0.6, 0.6])
m = spec.compile(); d = mujoco.MjData(m)
stand(m, d)
toe = max(d.xpos[m.body(n).id][0] for n in ("left_ankle_roll_link", "right_ankle_roll_link")) + 0.12
# add a 4x4 on the floor 0.5 m ahead of the toes (recompile with it, keep the pose)
qpos = d.qpos.copy()
wb.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.0445, 0.6, 0.0445], pos=[toe + BEAM_X, 0, 0.0445], rgba=[0.76, 0.6, 0.38, 1])
m = spec.compile(); d = mujoco.MjData(m); d.qpos[:] = qpos; mujoco.mj_forward(m, d)

cid = m.camera("d435").id
o = d.cam_xpos[cid]; Rc = d.cam_xmat[cid].reshape(3, 3)      # columns: right, up, back
fwd = -Rc[:, 2]
pitch_world = float(np.degrees(np.arcsin(-fwd[2])))
th, tv = np.tan(np.deg2rad(HFOV / 2)), np.tan(np.deg2rad(VFOV / 2))
corners = []
for sx in (1, -1):
    for sy in (1, -1):
        v = fwd + sx * th * Rc[:, 0] + sy * tv * Rc[:, 1]
        corners.append(o + v * (-o[2] / v[2]))
def floor_dist(v):
    return float(np.linalg.norm((o + v * (-o[2] / v[2]))[:2] - o[:2])) if v[2] < 0 else float("inf")
near = floor_dist(fwd - tv * Rc[:, 1])           # bottom edge of the image, straight ahead
far = floor_dist(fwd + tv * Rc[:, 1])            # top edge of the image, straight ahead
toe_ahead = toe - o[0]

def add_line(scn, p0, p1, rgba, w=0.004):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9), np.array(rgba, np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, w, np.asarray(p0, float), np.asarray(p1, float))
    scn.ngeom += 1

r = mujoco.Renderer(m, 900, 900)
views = []
for title, az, el, dist, look in (("side view: camera view cone to the floor", 270, -6, 2.1, [toe + 0.25, 0, 0.55]),
                                  ("three-quarter", 215, -20, 2.0, [toe + 0.2, 0, 0.45]),
                                  ("close-up of the mount", 235, -10, 0.45, o)):
    cam = mujoco.MjvCamera(); cam.azimuth, cam.elevation, cam.distance = az, el, dist; cam.lookat[:] = look
    r.update_scene(d, camera=cam)
    for c in corners:
        add_line(r.scene, o, c, [0.9, 0.1, 0.1, 1])
    for i, j in ((0, 1), (1, 3), (3, 2), (2, 0)):
        add_line(r.scene, corners[i], corners[j], [0.9, 0.1, 0.1, 1], 0.003)
    add_line(r.scene, o, o + fwd * (-o[2] / fwd[2]), [1.0, 0.85, 0.1, 1], 0.003)
    views.append((title, r.render().copy()))

# the camera's own view (D435 depth field of view, 87 x 58 deg), RGB render
rc = mujoco.Renderer(m, 374, 640)
opt = mujoco.MjvOption(); opt.geomgroup[2] = 0          # hide the mount and the camera case (the lens looks out through the glass)
rc.update_scene(d, camera="d435", scene_option=opt); rgb = rc.render().copy()
rc.enable_depth_rendering(); rc.update_scene(d, camera="d435", scene_option=opt); dep = rc.render().copy(); rc.disable_depth_rendering()
# policy input: 64 x 36 z-depth, crop columns 16..48, clip 0..3 m (MuJoCo depth = distance along -z = z-depth)
H, W = 36, 64
ii = (np.arange(H) + 0.5) * dep.shape[0] / H; jj = (np.arange(W) + 0.5) * dep.shape[1] / W
small = dep[ii.astype(int)][:, jj.astype(int)]
small = np.clip(small, 0, 3.0)
crop = small[:, 16:48]

fig = plt.figure(figsize=(16, 10))
for k, (title, img) in enumerate(views):
    A = fig.add_subplot(2, 3, k + 1); A.imshow(img); A.set_title(title); A.axis("off")
A = fig.add_subplot(2, 3, 4); A.imshow(rgb); A.set_title("what the camera sees (87 x 58 deg); 4x4 at %.1f m ahead of the toes" % BEAM_X); A.axis("off")
A = fig.add_subplot(2, 3, 5); im = A.imshow(small, cmap="viridis", vmin=0, vmax=3)
A.axvline(15.5, color="w", ls="--"); A.axvline(47.5, color="w", ls="--"); A.set_title("depth 64 x 36 (policy uses columns 16-47)")
plt.colorbar(im, ax=A, fraction=0.03, label="m")
A = fig.add_subplot(2, 3, 6); im = A.imshow(crop, cmap="viridis", vmin=0, vmax=3); A.set_title("policy input frame: 36 x 32, 0-3 m")
fig.suptitle("Chest D435 at the Hiking training pose: axis %.1f deg below horizontal when standing (61.6 deg on the torso), lens %.2f m up" % (pitch_world, o[2]) + "\nstraight ahead the image spans the floor from %.2f to %.2f m in front of the lens (toe tips %.2f m ahead)" % (near, far, toe_ahead), fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "simple_mount_view.png"), dpi=80)
print("pitch_world_deg %.1f  lens_height_m %.3f  floor_near_far_m %.2f %.2f  toe_ahead_m %.2f" % (pitch_world, o[2], near, far, toe_ahead))
