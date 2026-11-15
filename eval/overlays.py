"""Per-frame overlays for eval/render.py (eval monitor, 2026-11-14), in the style of the user's earlier SuperMimic / ACP renders
(gwbc-v6 scripts/acp/eval/render_full_plan.py, ported here; that repo is not changed).

ACPOverlay(extra)    extra = <seed>.extra.npz written by the capture run (the live trial, not a recomputation):
    height map    the planner's own 17 x 11 scan (1.6 x 1.0 m, 0.1 m) as tiles at the scanned heights, coloured by height
                  relative to the stance foot (green level, orange high, blue below)
    policy scan   the tracker's 9 x 7 scan (0.1 m) as small black dots at the scanned heights
    side obstacle walls / posts / rails within 1.5 m as low red blocks (capped at 0.18 m, so they never hide the robot), the
                  0.10 m margin around them pink
    contacts      the planner's next foot landing (occupancy > 0.5) as a ring, left blue / right orange; the physics stance
                  (foot load > 20 N in the simulator) as a square under that foot
DepthOverlay(extra, track)   Hiking / SSR, bottom-right inset:
    a RealSense-style depth image (424 x 240, D435-like 87 x 58 deg field of view) from the chest camera at the mount pose the
    adapter uses (body, position and rotation from the hand-over card), rendered by MuJoCo from the drawn scene including the
    robot's own links; colour map near warm / far cool, 0.3 - 3.5 m, black = no return; a metres scale bar. Beside it, small,
    the policy's actual input frame (the exact array the adapter fed the policy at that step, normalised, upscaled).
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

BOX, CYL, SPH = mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_SPHERE
FOOT = (np.array([0.2, 0.55, 1.0, 1.0], np.float32), np.array([1.0, 0.5, 0.1, 1.0], np.float32))
OBST_R, OBST_RES, OBST_CAP, MARGIN, TALL = 1.5, 0.06, 0.18, 0.10, 0.25


def _add(scn, kind, size, pos, rgba, mat=None):
    if scn.ngeom >= scn.maxgeom:
        return
    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], kind, np.asarray(size, float), np.asarray(pos, float),
                        (np.eye(3) if mat is None else mat).ravel(), np.asarray(rgba, np.float32))
    scn.ngeom += 1


def height_rgba(h, a=0.55):
    if h < -0.03:
        return np.array([0.25, 0.45, 1.0, a], np.float32)
    u = min(max(h / 0.3, 0.0), 1.0)
    return np.array([0.2 + 0.8 * u, 0.85 - 0.35 * u, 0.25 - 0.15 * u, a], np.float32)


def _quat_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def obstacle_classes(app):
    """(nx, ny) int8 on the course raster: 2 = a tall box (top more than 0.25 m over the walk raster) stands there, 1 = within
    the 0.10 m margin of one, 0 = free. From the course's wall boxes (the physics geometry outside the walk raster)."""
    from scipy import ndimage
    cls = np.zeros((app.nx, app.ny), np.int8)
    xs, ys = app.x0 + app.res * np.arange(app.nx), app.y0 + app.res * np.arange(app.ny)
    for b in app.boxes:
        R = _quat_mat(b["quat"])
        h = np.asarray(b["half"], float)
        c = np.asarray(b["center"], float)
        e = np.abs(R) @ h
        i0, i1 = np.searchsorted(xs, c[0] - e[0]), np.searchsorted(xs, c[0] + e[0])
        j0, j1 = np.searchsorted(ys, c[1] - e[1]), np.searchsorted(ys, c[1] + e[1])
        if i1 <= i0 or j1 <= j0:
            continue
        X, Y = np.meshgrid(xs[i0:i1], ys[j0:j1], indexing="ij")
        loc = (np.stack((X - c[0], Y - c[1], np.zeros_like(X)), -1) @ R)       # into the box frame (z = 0 plane)
        inside = (np.abs(loc[..., 0]) <= h[0] + 1e-3) & (np.abs(loc[..., 1]) <= h[1] + 1e-3) if abs(R[2, 2]) > 0.9 else np.ones_like(X, bool)
        top = c[2] + e[2]
        tall = top - app.grid[i0:i1, j0:j1] > TALL
        cls[i0:i1, j0:j1][inside & tall] = 2
    r = int(math.ceil(MARGIN / app.res))
    disk = np.hypot(*np.mgrid[-r:r + 1, -r:r + 1]) * app.res <= MARGIN + 1e-9
    near = ndimage.binary_dilation(cls == 2, structure=disk)
    cls[(cls == 0) & near] = 1
    return cls


class ACPOverlay:
    legend_text = ("height map: planner's 17 x 11 scan as tiles (green level, orange high, blue below the stance foot) | black dots: "
                   "tracker's 9 x 7 scan | red: side obstacle, pink: 0.10 m margin | ring: planner's next foot landing, square: "
                   "physics stance (load > 20 N); blue left, orange right")

    def __init__(self, extra_path):
        self.e = dict(np.load(extra_path))
        self.cls = None

    def _i(self, t):
        return int(np.clip(np.searchsorted(self.e["t"], t - 0.021), 0, len(self.e["t"]) - 1))

    def scene(self, scn, x, d, app, t, z, k):
        if self.cls is None:
            self.cls = obstacle_classes(app)
        e, i = self.e, self._i(t)
        load = z["load"][k]
        feet = [d.xpos[b] for b in x.foot_ids]
        st = int(np.argmax(load))
        ref_z = float(app.height(feet[st][0], feet[st][1]))
        p = d.qpos[:3]
        # side obstacles (low blocks, capped)
        g = np.arange(-OBST_R, OBST_R + 1e-9, OBST_RES)
        gx, gy = np.meshgrid(p[0] + g, p[1] + g, indexing="ij")
        xy = np.c_[gx.ravel(), gy.ravel()]
        xy = xy[np.hypot(xy[:, 0] - p[0], xy[:, 1] - p[1]) <= OBST_R]
        ii, jj = app.cell(xy[:, 0], xy[:, 1])
        inside = (xy[:, 0] >= app.x0) & (xy[:, 0] <= app.x0 + app.res * (app.nx - 1)) & (xy[:, 1] >= app.y0) & (xy[:, 1] <= app.y0 + app.res * (app.ny - 1))
        c = np.where(inside, self.cls[ii, jj], 0)
        for (x_, y_), cc in zip(xy, c):
            if cc == 2:
                _add(scn, BOX, [OBST_RES * 0.45] * 2 + [OBST_CAP / 2], [x_, y_, ref_z + OBST_CAP / 2], [0.95, 0.1, 0.1, 0.45])
            elif cc == 1:
                _add(scn, BOX, [OBST_RES * 0.45] * 2 + [0.003], [x_, y_, ref_z + 0.008], [0.9, 0.2, 0.9, 0.35])
        # the planner's map: tiles at the scanned heights
        for (x_, y_), h in zip(e["plan_xy"][i], e["plan_h"][i]):
            _add(scn, BOX, [0.045, 0.045, 0.003], [x_, y_, h + 0.006], height_rgba(h - ref_z))
        # the tracker's scan: dots
        for (x_, y_), h in zip(e["scan_xy"][i], e["scan_h"][i]):
            _add(scn, SPH, [0.011, 0, 0], [x_, y_, h + 0.012], [0.08, 0.08, 0.1, 1])
        # contacts: planned next landing (ring) and physics stance (square)
        for f in range(2):
            if e["occ"][i][f] > 0.5:
                px, py, pz = e["tgt_w"][i][f]
                for a in np.linspace(0, 2 * np.pi, 14, endpoint=False):
                    _add(scn, SPH, [0.012, 0, 0], [px + 0.07 * math.cos(a), py + 0.07 * math.sin(a), pz + 0.02], FOOT[f])
            if load[f] > 20.0:
                q = feet[f]
                _add(scn, BOX, [0.025, 0.025, 0.012], [q[0], q[1], float(app.height(q[0], q[1])) + 0.03], FOOT[f] * np.array([0.5, 0.5, 0.5, 1], np.float32))

    def image(self, im, dr, t, top, bottom, font):
        pass


def _cmap(u):
    """u in [0, 1] (0 near) -> RGB, near warm (red / yellow) to far cool (cyan / blue), a jet reversed."""
    u = np.clip(u, 0, 1)
    v = 1.0 - u
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return (np.stack((r, g, b), -1) * 255).astype(np.uint8)


class DepthOverlay:
    NEAR, FAR, RW, RH, FOVY = 0.3, 3.5, 424, 240, 58.0

    def __init__(self, extra_path, track):
        self.e = dict(np.load(extra_path, allow_pickle=True))
        self.track = track
        R = self.e["cam_R"]                                    # camera frame (x forward, y left, z up) in the mount body
        xa, ya = R @ np.array([0.0, -1.0, 0.0]), R @ np.array([0.0, 0.0, 1.0])      # MuJoCo: x right, y up, looks along -z
        self.cameras = [dict(name="realsense_sim", body=str(self.e["cam_body"]), pos=[float(v) for v in self.e["cam_pos"]],
                             xyaxes=[float(v) for v in np.r_[xa, ya]], fovy=self.FOVY)]
        self.ren = None
        self.legend_text = None

    def scene(self, scn, x, d, app, t, z, k):
        if self.ren is None:
            m = x.model
            self.m, self.cid = m, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "realsense_sim")
            self.ren = mujoco.Renderer(m, self.RH, self.RW, max_geom=20000)
            self.ren.enable_depth_rendering()
            self.opt = mujoco.MjvOption()
            self.opt.geomgroup[:] = 0
            self.opt.geomgroup[1] = self.opt.geomgroup[2] = 1
            if getattr(x, "dress", None) is not None:
                self.opt.geomgroup[0] = 1
            else:
                self.opt.geomgroup[4] = self.opt.geomgroup[5] = 1
        self.ren.update_scene(d, camera=self.cid, scene_option=self.opt)
        self.depth = self.ren.render().copy()

    def image(self, im, dr, t, top, bottom, font):
        """Right-hand column, bottom up: the RealSense-style image with its scale bar, above it the policy's input, small."""
        from PIL import Image
        dep = self.depth
        valid = (dep >= self.NEAR) & (dep <= 6.0)
        rgb = _cmap((dep - self.NEAR) / (self.FAR - self.NEAR))
        rgb[~valid] = 0
        W, H = im.size
        sw = int(round(W * 0.31))
        sh = int(round(sw * self.RH / self.RW))
        x0 = W - sw - 8
        y0 = bottom - 8 - 30 - sh                               # image top; the scale bar below it
        big = Image.fromarray(rgb).resize((sw, sh), Image.BILINEAR)
        i = int(np.clip(np.searchsorted(self.e["t"], t - 0.021), 0, len(self.e["t"]) - 1))
        pin = np.asarray(self.e["depth"][i], float)
        if getattr(self, "minimal", False):                    # RENDER_MINIMAL (render.py): the depth view, a tiny label, the bar
            y0 = bottom - 8 - 22 - sh
            dr.rectangle((x0 - 4, y0 - 18, W - 4, bottom - 4), fill=(15, 18, 24))
            dr.text((x0, y0 - 17), "depth", fill=(230, 230, 230), font=font)
            im.paste(big, (x0, y0))
            ramp = _cmap(np.linspace(0, 1, sw))[None].repeat(6, 0)
            im.paste(Image.fromarray(ramp), (x0, y0 + sh + 2))
            for mtr in (0.5, 1, 2, 3):
                xx = x0 + int((mtr - self.NEAR) / (self.FAR - self.NEAR) * (sw - 1))
                dr.text((xx - 8, y0 + sh + 8), "%g m" % mtr, fill=(230, 230, 230), font=font)
            return
        tw = int(round(sw * 0.30))
        th = int(round(tw * pin.shape[0] / pin.shape[1]))
        ty = y0 - 40 - th - 8
        dr.rectangle((x0 - 6, ty - 6, W - 2, bottom - 2), fill=(15, 18, 24))
        # the policy's input (exact array fed at this step), white near
        g = (255 * (1.0 - np.clip(pin, 0, 1))).astype(np.uint8)
        im.paste(Image.fromarray(g).resize((tw, th), Image.NEAREST).convert("RGB"), (x0, ty))
        dr.text((x0 + tw + 8, ty), "policy input", fill=(240, 240, 240), font=font)
        dr.text((x0 + tw + 8, ty + 18), "%d x %d px, as fed" % (pin.shape[1], pin.shape[0]), fill=(200, 210, 220), font=font)
        dr.text((x0 + tw + 8, ty + 36), "white near, black far", fill=(200, 210, 220), font=font)
        # the RealSense-style view
        dr.text((x0, y0 - 38), "simulated RealSense depth, chest, 87x58 deg", fill=(240, 240, 240), font=font)
        dr.text((x0, y0 - 20), "black: no return (< 0.3 m or > 6 m)", fill=(200, 210, 220), font=font)
        im.paste(big, (x0, y0))
        ramp = _cmap(np.linspace(0, 1, sw))[None].repeat(8, 0)
        im.paste(Image.fromarray(ramp), (x0, y0 + sh + 3))
        for mtr in (0.5, 1, 2, 3):
            u = (mtr - self.NEAR) / (self.FAR - self.NEAR)
            xx = x0 + int(u * (sw - 1))
            dr.line((xx, y0 + sh + 1, xx, y0 + sh + 12), fill=(255, 255, 255))
            dr.text((xx - 8, y0 + sh + 12), "%g m" % mtr, fill=(230, 230, 230), font=font)
