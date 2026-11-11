"""What the X2's chest depth camera sees, standing at points of an apparatus route. MuJoCo's own renderer (CPU).

    python eval/depth_views.py [--apparatus ours] [--out eval/out/depth_views_ours]

Camera: the training camera of docs/camera/README.md in the Hiking run-2 pose: on `torso_link` at (0.131, 0, 0.259), optical
axis 61.6 degrees below horizontal, no roll, depth field of view 87 x 58 degrees, range 0.1 to 3 m. The depth image is
MuJoCo's depth buffer from that camera (distance along the optical axis), so the robot's own legs, feet and hands are in it,
with the apparatus floor and every wall box. The robot is PLACED in the plant's default standing pose with its soles on the
surface (no policy, no physics step). First a flat-ground check against the reference's acceptance rule.
Per point one PNG (outside view with the camera's footprint, depth at full resolution, depth as the policy gets it, the
camera's plain image), a contact sheet and a README.
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import x2 as X  # noqa: E402
from apparatus import Apparatus  # noqa: E402

CAM = dict(body="torso_link", pos=[0.131, 0.0, 0.259], pitch_deg=61.6, fov=(87.0, 58.0), near=0.1, far=3.0)
W, H = 856, 500                      # 87 x 58 degrees: tan(43.5) / tan(29) = 1.712
POLICY_RAW, CROP = (64, 36), 16      # the policy's raw image (width, height) and the columns cut from each side -> 36 rows x 32 columns
TOE = 0.139 + X.SOLE_R               # toe tip ahead of the ankle

# name, x, y, yaw (deg), waypoint it belongs to. Positions are pelvis positions on the eval route of our apparatus.
POINTS_OURS = [
    ("01 start", 0.762, 0.762, 16.7, "start pose, heading to waypoint 0 (section door)"),
    ("02 one metre before the open door", 1.45, 1.18, 0.0, "between waypoints 0 and 1 (door)"),
    ("03 in the doorway", 2.45, 1.18, 0.0, "waypoint 1 (door), in the door plane"),
    ("04 in the slot, facing the front 4x4", 2.769, 0.686, 0.0, "waypoint 2 (door -> 4x4s): the slot is 0.5 m deep, so the toes are close to the beam"),
    ("04a one metre before the front 4x4, seen through the doorway", 2.17, 1.42, -41.4,
     "off the route: the route reaches this beam from the 0.5 m slot, so 1.0 m of approach only exists diagonally through the doorway"),
    ("04b half a metre before the front 4x4, past the doorway", 2.545, 1.085, -41.4,
     "off the route: on the same diagonal, just inside the door wall"),
    ("05 between the 4x4s, facing the back one", 2.80, 1.60, 17.2, "waypoint 7 (4x4s), heading to waypoint 8"),
    ("06 at the foot of the ramps", 3.34, 1.83, 0.0, "between waypoints 9 and 10 (ramps)"),
    ("07 on the ramps", 4.98, 1.83, 0.0, "waypoint 11 (ramps), on the back hill"),
    ("08 beside the wall section", 7.06, 1.37, 0.0, "waypoint 16 (wall)"),
    ("09 one metre before the stairs", 8.615 - 1.0 - 0.16, 0.533, 0.0, "before waypoint 18 (stairs)"),
    ("10 at the foot of the stairs", 8.615 - 0.30 - 0.16, 0.533, 0.0, "waypoint 18 (stairs), toes 0.3 m from the first riser"),
    ("11 on the third stair", 9.17, 0.533, 0.0, "between waypoints 18 and 19 (stairs)"),
    ("12 on the landing, facing the end station", 10.26, 0.864, 83.0, "waypoint 20 (landing), heading to button 5"),
]


def cam_spec():
    th = math.radians(CAM["pitch_deg"])
    f = np.array([math.cos(th), 0.0, -math.sin(th)])
    xc = np.array([0.0, -1.0, 0.0])
    return dict(name="chest_depth", body=CAM["body"], pos=CAM["pos"], xyaxes=list(xc) + list(np.cross(-f, xc)), fovy=CAM["fov"][1])


def place(x, d, app, px, py, yaw):
    """Default standing pose at (px, py) facing yaw, lowered until the lowest sole point is on the surface."""
    m = x.model
    mujoco.mj_resetData(m, d)
    d.qpos[:3] = [px, py, 2.5]
    d.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    d.qpos[x.qadr] = x.plant["default_q"]
    mujoco.mj_forward(m, d)
    gap = np.inf
    for b in x.foot_ids:
        for g in range(m.ngeom):
            if m.geom_bodyid[g] == b and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE:
                p = d.geom_xpos[g]
                ground = float(app.height(p[0], p[1])) if app is not None else 0.0
                gap = min(gap, p[2] - m.geom_size[g][0] - ground)
    d.qpos[2] -= gap
    mujoco.mj_forward(m, d)


def project(m, d, pts):
    """World points -> (column, row) in the depth image, and depth along the axis."""
    cid = m.camera("chest_depth").id
    R, c = d.cam_xmat[cid].reshape(3, 3), d.cam_xpos[cid]
    pc = (np.atleast_2d(pts) - c) @ R
    z = -pc[:, 2]
    fy = (H / 2) / math.tan(math.radians(CAM["fov"][1]) / 2)
    return np.stack((W / 2 + fy * pc[:, 0] / z, H / 2 - fy * pc[:, 1] / z), 1), z


def toes(x, d):
    """World positions of the two toe tips (on the sole plane) and the heading."""
    out = []
    for b in x.foot_ids:
        R = d.xmat[b].reshape(3, 3)
        out.append(d.xpos[b] + R @ np.array([TOE, 0.0, -X.SOLE_Z]))
    yaw = X.yaw_of(d.qpos[3:7])
    return np.array(out), yaw


def render_all(x, d, ren, vopt):
    ren.disable_depth_rendering()
    ren.update_scene(d, camera="chest_depth", scene_option=vopt)
    rgb = ren.render().copy()
    ren.enable_depth_rendering()
    ren.update_scene(d, camera="chest_depth", scene_option=vopt)
    depth = ren.render().copy()
    ren.disable_depth_rendering()
    return rgb, depth


def policy_view(depth):
    """The depth as a 36 x 32 policy input: sampled to 64 x 36 (the centre of each cell), 16 columns cut from each side."""
    w, h = POLICY_RAW
    ys = ((np.arange(h) + 0.5) * H / h).astype(int)
    xs = ((np.arange(w) + 0.5) * W / w).astype(int)
    return np.clip(depth[np.ix_(ys, xs)], CAM["near"], CAM["far"])[:, CROP:w - CROP]


def footprint(x, d, n=14):
    """Where the edges of the camera's view meet the scene: points along the four image edges, by ray casting."""
    m = x.model
    cid = m.camera("chest_depth").id
    R, c = d.cam_xmat[cid].reshape(3, 3), d.cam_xpos[cid]
    tx, ty = math.tan(math.radians(CAM["fov"][0]) / 2), math.tan(math.radians(CAM["fov"][1]) / 2)
    gg = np.zeros(6, np.uint8)
    gg[X.GROUP_TERRAIN] = gg[X.GROUP_WALL] = 1
    gid = np.zeros(1, np.int32)
    edge = [(-1 + 2 * i / n, -1) for i in range(n + 1)] + [(1, -1 + 2 * i / n) for i in range(n + 1)] + \
           [(1 - 2 * i / n, 1) for i in range(n + 1)] + [(-1, 1 - 2 * i / n) for i in range(n + 1)]
    pts = []
    for u, v in edge:
        dirc = R @ np.array([u * tx, v * ty, -1.0])
        dirc /= np.linalg.norm(dirc)
        r = mujoco.mj_ray(m, d, c, dirc, gg, 1, -1, gid)
        pts.append(c + dirc * (r if 0 <= r < 4.0 else 4.0))
    return np.array(pts)


def nearest_edge(app, toe_xy, yaw, reach=3.0):
    """Distance from the toe line, straight ahead, to the first change of floor height over 3 cm or the first wall box."""
    if app is None:
        return None, None
    u = np.array([math.cos(yaw), math.sin(yaw)])
    h0 = float(app.height(toe_xy[0], toe_xy[1]))
    for s in np.arange(0.0, reach, 0.01):
        p = toe_xy + u * s
        h = float(app.height(p[0], p[1]))
        if abs(h - h0) > 0.03:
            return float(s), "floor steps %+.2f m" % (h - h0)
        if app.wall_clearance(p[0], p[1], h + 0.05, h + 1.3) < 0.01:
            return float(s), "a wall, post or rail"
        h0 = h if abs(h - h0) < 0.01 else h0            # follow a slope, catch a step
    return None, None


def outside_view(x, d, ren, lookat, yaw, fp):
    m = x.model
    cam, opt = mujoco.MjvCamera(), mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[1] = opt.geomgroup[2] = opt.geomgroup[X.GROUP_TERRAIN] = opt.geomgroup[X.GROUP_WALL] = 1
    cam.lookat[:] = [lookat[0] + 0.5 * math.cos(yaw), lookat[1] + 0.5 * math.sin(yaw), lookat[2] - 0.25]
    cam.distance, cam.elevation, cam.azimuth = 3.6, -28, math.degrees(yaw) + 140.0      # from ahead and to the side: the robot's front
    ren.update_scene(d, cam, opt)
    scn = ren.scene
    az, el = math.radians(cam.azimuth), math.radians(cam.elevation)
    eye = np.array(cam.lookat) - cam.distance * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    body = np.array(lookat) - [0, 0, 0.15]
    ab = body - eye
    for i in range(scn.ngeom):                              # walls between this camera and the robot are faded out
        gm = scn.geoms[i]
        if gm.objtype == mujoco.mjtObj.mjOBJ_GEOM and gm.objid in x.wall_geoms:
            c = np.array(gm.pos)
            tt = float(np.clip((c - eye) @ ab / (ab @ ab), 0.0, 1.0))
            gm.rgba[3] = 0.05 if (tt < 0.97 and np.linalg.norm(eye + tt * ab - c) < np.linalg.norm(m.geom_size[gm.objid]) + 0.45) else 0.45
    col = np.array([0.0, 0.9, 1.0, 1.0], np.float32)
    for a, b in zip(fp, np.roll(fp, -1, axis=0)):           # the footprint: a thin outline lifted 1 cm off the surface
        if scn.ngeom >= scn.maxgeom:
            break
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).ravel(), col)
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.006, a + [0, 0, 0.01], b + [0, 0, 0.01])
        scn.ngeom += 1
    return ren.render().copy()


def panel(path, title, sub, outside, depth, rgb, note):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.7), gridspec_kw=dict(width_ratios=[1.45, 1.45, 0.8, 1.45]))
    cmap = plt.get_cmap("turbo").copy()
    cmap.set_over("#555555")                               # beyond 3 m
    cmap.set_under("#ff00ff")                              # nearer than 0.1 m
    ax[0].imshow(outside)
    ax[0].set_title("(a) outside view, cyan = edge of the camera's view", fontsize=9)
    im = ax[1].imshow(depth, cmap=cmap, vmin=CAM["near"], vmax=CAM["far"])
    ax[1].set_title("(b) depth, %d x %d, metres (grey: beyond 3 m; magenta: under 0.1 m)" % (W, H), fontsize=9)
    cb = plt.colorbar(im, ax=ax[1], fraction=0.035, pad=0.02, extend="both")
    cb.set_label("m")
    pv = policy_view(depth)
    ax[2].imshow(pv, cmap=cmap, vmin=CAM["near"], vmax=CAM["far"])
    ax[2].set_title("(c) as the policy gets it:\n%d rows x %d columns" % pv.shape, fontsize=9)
    ax[3].imshow(rgb)
    ax[3].set_title("(d) the camera's plain image (orientation only)", fontsize=9)
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("MuJoCo | X2 chest depth camera | %s | standing pose placed, not a policy" % title, fontsize=11, y=0.995)
    import textwrap
    fig.text(0.5, 0.005, chr(10).join(textwrap.wrap(sub, 215) + textwrap.wrap(note, 215)), ha="center", va="bottom", fontsize=8)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.14, wspace=0.05)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def flat_floor_profile(m, d, floor_z, rows):
    """Depth (along the optical axis) a flat floor at floor_z would give in the centre column, per image row."""
    cid = m.camera("chest_depth").id
    R, c = d.cam_xmat[cid].reshape(3, 3), d.cam_xpos[cid]
    ty = math.tan(math.radians(CAM["fov"][1]) / 2)
    out = []
    for r in rows:
        v = 1.0 - 2.0 * (r + 0.5) / H
        dw = R @ np.array([0.0, v * ty, -1.0])
        out.append((floor_z - c[2]) / dw[2] if dw[2] < -1e-6 else np.nan)
    return np.array(out)


def detail(path, title, depth, flat, note):
    """The same depth with the colour scale stretched to what is in the image, the policy-size view likewise, and the
    centre-column profile against image row with the flat-floor curve."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import textwrap
    pv = policy_view(depth)
    lo, hi = float(np.percentile(depth, 1)), float(min(np.percentile(depth, 99), CAM["far"]))
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.9), gridspec_kw=dict(width_ratios=[1.45, 0.8, 1.2, 1.2]))
    im = ax[0].imshow(np.clip(depth, lo, hi), cmap="turbo", vmin=lo, vmax=hi)
    ax[0].set_title("(b') depth, scale stretched to this image: %.2f to %.2f m" % (lo, hi), fontsize=9)
    plt.colorbar(im, ax=ax[0], fraction=0.035, pad=0.02).set_label("m")
    ax[0].axvline(W / 2, color="w", lw=0.6, ls=":")
    ax[1].imshow(np.clip(pv, lo, hi), cmap="turbo", vmin=lo, vmax=hi)
    ax[1].set_title("(c') %d rows x %d columns, same scale" % pv.shape, fontsize=9)
    for a in ax[:2]:
        a.set_xticks([]); a.set_yticks([])
    rows = np.arange(H)
    col = depth[:, W // 2 - 4:W // 2 + 4].mean(1)
    ax[2].plot(col, rows, "b-", lw=1.4, label="this scene, centre column")
    ax[2].plot(flat, rows, "k--", lw=1.0, label="flat floor at the toes' height")
    ax[2].set_ylim(H, 0); ax[2].set_xlim(0.2, min(3.0, np.nanmax(flat[np.isfinite(flat)]) + 0.2))
    ax[2].set_xlabel("depth (m)"); ax[2].set_ylabel("image row (of %d)" % H); ax[2].legend(fontsize=7); ax[2].grid(alpha=0.3)
    ax[2].set_title("(e) centre column at full resolution", fontsize=9)
    prow = ((np.arange(POLICY_RAW[1]) + 0.5) * H / POLICY_RAW[1]).astype(int)
    pc = pv[:, pv.shape[1] // 2 - 1:pv.shape[1] // 2 + 1].mean(1)
    pf = np.clip(flat[prow], CAM["near"], CAM["far"])
    diff = (pf - pc) * 100.0                                 # cm nearer than the flat floor
    ax[3].barh(np.arange(len(diff)), diff, color=["#d62728" if abs(v) > 2 else "#999999" for v in diff])
    ax[3].set_ylim(len(diff) - 0.5, -0.5); ax[3].axvline(0, color="k", lw=0.6)
    ax[3].set_xlabel("cm nearer than a flat floor (centre 2 columns)"); ax[3].set_ylabel("policy image row (of %d)" % len(diff)); ax[3].grid(alpha=0.3)
    nrow, big = int((np.abs(diff) > 2).sum()), float(np.nanmax(np.abs(diff)))
    ax[3].set_title("(f) policy size: %d of %d rows off flat by > 2 cm, max %.0f cm" % (nrow, len(diff), big), fontsize=9)
    fig.suptitle("MuJoCo | X2 chest depth camera, detail | %s | standing pose placed, not a policy" % title, fontsize=11, y=0.995)
    fig.text(0.5, 0.005, chr(10).join(textwrap.wrap(note, 215)), ha="center", va="bottom", fontsize=8)
    fig.subplots_adjust(left=0.02, right=0.99, top=0.88, bottom=0.17, wspace=0.22)
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return nrow, big


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apparatus", default="ours")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import memcap
    memcap.guard(cap_gb=6, need_free_gb=12, low_priority=True)
    out = pathlib.Path(a.out or pathlib.Path(__file__).resolve().parent / "out" / ("depth_views_" + a.apparatus))
    out.mkdir(parents=True, exist_ok=True)
    vopt = mujoco.MjvOption()
    vopt.geomgroup[:] = 0
    vopt.geomgroup[1] = vopt.geomgroup[2] = vopt.geomgroup[X.GROUP_TERRAIN] = vopt.geomgroup[X.GROUP_WALL] = 1
    lines = []

    # ---- flat-ground check against docs/camera/README.md
    x = X.build(None, cameras=[cam_spec()], render_size=(W, H))
    d = mujoco.MjData(x.model)
    place(x, d, None, 0.0, 0.0, 0.0)
    ren = mujoco.Renderer(x.model, H, W)
    tp, yaw = toes(x, d)
    (uv, _) = project(x.model, d, tp)
    line1, _ = project(x.model, d, np.array([[tp[:, 0].mean() + 1.0, 0.0, 0.0]]))
    cid = x.model.camera("chest_depth").id
    camz, lens_behind = float(d.cam_xpos[cid][2]), float(tp[:, 0].mean() - d.cam_xpos[cid][0])
    toe_row, line_row = float(uv[:, 1].mean()) / H, float(line1[0, 1]) / H
    rgb, depth = render_all(x, d, ren, vopt)
    top_ground, _ = project(x.model, d, np.array([[tp[:, 0].mean() + s, 0.0, 0.0] for s in np.arange(0.2, 6.0, 0.01)]))
    seen = [s for s, r in zip(np.arange(0.2, 6.0, 0.01), top_ground[:, 1]) if r >= 0]
    ok_toe, ok_line = 0.90 <= toe_row <= 1.0, abs(line_row - 0.27) <= 0.03
    check = ("flat-ground check: camera %.2f m above the floor, lens %.2f m behind the toe tips; toe tips at %.0f %% of the image height from the top "
             "(rule: just inside the bottom edge) -> %s; floor line 1.0 m ahead of the toes at %.1f %% from the top (rule: 27 %% +- 3 %%) -> %s; "
             "top edge of the image is the floor %.2f m ahead of the toes" % (camz, lens_behind, 100 * toe_row, "OK" if ok_toe else "NOT MET",
                                                                               100 * line_row, "OK" if ok_line else "NOT MET", max(seen) if seen else float("nan")))
    print(check)
    fp = footprint(x, d)
    import matplotlib
    matplotlib.use("Agg")
    rgb2 = rgb.copy()
    for row, colr in ((int(line1[0, 1]), (255, 255, 0)), (int(0.27 * H), (0, 255, 0))):
        if 0 <= row < H:
            rgb2[max(row - 1, 0):row + 1, :] = colr
    panel(out / "00_flat_ground_check.png", "flat ground, the acceptance check", check,
          outside_view(x, d, ren, d.qpos[:3].copy(), 0.0, fp), depth, rgb2, "in (d): yellow = the floor line 1.0 m ahead of the toes, green = 27 % from the top")
    lines.append("00_flat_ground_check.png: " + check)
    ren.close()

    # ---- the apparatus
    app = Apparatus(a.apparatus)
    x = X.build(app, cameras=[cam_spec()], render_size=(W, H))
    d = mujoco.MjData(x.model)
    ren = mujoco.Renderer(x.model, H, W, max_geom=20000)
    files = ["00_flat_ground_check.png"]
    detail_lines = []
    for name, px, py, yawd, where in POINTS_OURS:
        yaw = math.radians(yawd)
        place(x, d, app, px, py, yaw)
        tp, _ = toes(x, d)
        rgb, depth = render_all(x, d, ren, vopt)
        fp = footprint(x, d)
        dist, what = nearest_edge(app, tp[:, :2].mean(0), yaw)
        # check against the scene: the depth at the image point of that obstacle edge
        chk = ""
        if dist is not None:
            u = np.array([math.cos(yaw), math.sin(yaw)])
            q = tp[:, :2].mean(0) + u * (dist + 0.02)
            pw = np.array([[q[0], q[1], float(app.height(q[0], q[1]))]])
            (uv, zc) = project(x.model, d, pw)
            c_, r_ = int(round(uv[0, 0])), int(round(uv[0, 1]))
            if 0 <= c_ < W and 0 <= r_ < H:
                chk = "; check: that edge is %.2f m along the camera axis by geometry, the depth image reads %.2f m there" % (zc[0], depth[r_, c_])
            else:
                chk = "; that edge is outside the image"
        own = float((depth < 0.9 * depth.max()).mean())
        note = ("toes to the nearest obstacle edge straight ahead: %.2f m (%s)" % (dist, what) if dist is not None else "no obstacle edge within 3 m straight ahead") + chk
        fn = name.split(" ", 1)[0] + "_" + name.split(" ", 1)[1].replace(" ", "_").replace(",", "") + ".png"
        panel(out / fn, "our apparatus, point %s" % name, where + " | pelvis (%.2f, %.2f), facing %.0f deg" % (px, py, yawd),
              outside_view(x, d, ren, d.qpos[:3].copy(), yaw, fp), depth, rgb, note)
        floor_z = float(app.height(*tp[:, :2].mean(0)))
        nrow, big = detail(out / fn.replace(".png", "_detail.png"), "our apparatus, point %s" % name, depth,
                           flat_floor_profile(x.model, d, floor_z, np.arange(H)), note)
        detail_lines.append("%s: at 36 x 32, %d of 36 rows of the centre columns differ from a flat floor by more than 2 cm; the largest difference is %.0f cm." % (
            fn.replace(".png", "_detail.png"), nrow, big))
        near = float(depth.min())
        lines.append("%s: %s. Pelvis (%.2f, %.2f, %.2f), facing %.0f deg. %s. Depth range in the image %.2f to %.2f m; %.0f %% of the pixels beyond 3 m." % (
            fn, where, px, py, d.qpos[2], yawd, note, near, float(depth.max()), 100 * float((depth > CAM["far"]).mean())))
        print(lines[-1], flush=True)
        files.append(fn)
    ren.close()
    # contact sheet, 1600 px wide
    from PIL import Image
    ims = [Image.open(out / f) for f in files]
    w = 1600
    hs = [int(im.size[1] * w / im.size[0]) for im in ims]
    sheet = Image.new("RGB", (w, sum(hs)), "white")
    y = 0
    for im, h in zip(ims, hs):
        sheet.paste(im.resize((w, h)), (0, y))
        y += h
    sheet.save(out / "contact_sheet.png")
    open(out / "README.md", "w").write("""# Depth camera views on %s apparatus (MuJoCo)

Made by `eval/depth_views.py`. Every image is a MuJoCo render; the robot is placed in the plant's default standing pose with its
soles on the surface (no policy, no physics step), facing the route direction.

**Camera:** on `torso_link` at (0.131, 0, 0.259) m, optical axis 61.6 degrees below horizontal, no roll, 87 x 58 degrees, shown
range 0.1 to 3 m (the Hiking run-2 pose of the chest RealSense, `docs/camera/README.md`). Depth = distance along the optical axis,
from MuJoCo's depth buffer, %d x %d pixels; it includes the robot's own body, the floor raster and every wall box (the door leaf
stands open). Panel (c) is that image sampled to 64 x 36 and cut by 16 columns on each side: 36 rows x 32 columns.

**%s**

**Not modelled:** sensor noise, holes and invalid pixels, the RealSense's real minimum range (0.1 to 0.3 m depending on mode),
the infrared dot pattern, lens distortion, exposure, motion blur, the camera's latency. The floor is a 2 cm height field, so
vertical faces lean by one cell and thin things (rails, posts) are boxes. Panel (c) uses nearest sampling; the Hiking track's own
preprocessing (blur, normalisation, frame stack) is not applied here.

## Files

%s
""" % ("our" if a.apparatus == "ours" else a.apparatus, W, H, check, "\n".join("- " + ln for ln in lines) + "\n- contact_sheet.png: all of the above, 1600 px wide\n\n## Detail files (`_detail`)\n\n"
       "Per point: (b') the depth with the colour scale stretched to that image, (c') the 36 x 32 view on the same scale, (e) depth along the centre "
       "column against image row with the curve a flat floor at the toes' height would give, (f) per policy row, how many centimetres nearer than that "
       "flat floor the centre two columns read (red: over 2 cm).\n\n" + "\n".join("- " + ln for ln in detail_lines) + "\n"))
    print("written to", out)


if __name__ == "__main__":
    main()
