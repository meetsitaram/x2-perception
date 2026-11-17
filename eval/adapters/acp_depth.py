"""ACP with its height scans either recorded (teacher data for the depth -> height-scan student) or replaced by the student's
estimate (user, 2026-11-16: "depth camera to height map teacher student learning phase so we can use ACP similar to how we
have the hiking model working"; plan: docs/TODO_depth_to_heightmap.md).

ACP itself is unchanged (acp:ACPHandover, ACP_HANDOVER_DIR etc. as usual). Its two height inputs both come from
view.height_scan(points) - root height: the planner's 17 x 11 grid (187 points) and the tracker's 9 x 7 grid (63 points). This
adapter hands ACP a wrapped view whose height_scan

  ACPD_MODE=record   passes the simulator's true heights through and logs them (the labels), with the chest depth camera
                     ray-cast every other control step (25 Hz) and ACP's own proprioception;
  ACPD_MODE=student  returns the student's estimate (ACPD_STUDENT = a TorchScript file, run on CPU) instead;
  ACPD_MODE=student_plan / student_track: the student's estimate for one of the two inputs only, the truth for the other
                     (which input's errors cost success? user 2026-11-16: "focus mainly on contact steps planner");
  ACPD_MODE=track_err: true scans, the tracker's gets ACPD_NOISE (sigma per step) and ACPD_OFFSET (+- per episode), metres;
  ACPD_MODE=dagger   ACP acts on the student's estimate while the true scans and the depth are recorded (DAgger data:
                     the states the student's own maps lead to, labelled with the truth);
  ACPD_MODE=noback   true scans except the planner cells more than 0.2 m behind, which copy the row at x = -0.2 (a
                     sensitivity test: how much ACP uses the map behind it).

Camera: the Hiking chest pose (torso_link, (0.131, 0, 0.259) m, 61.6 deg down; adapter card of the Hiking export, the bracket
in docs/hardware/chest_camera_mount/) plus a per-episode offset drawn around ACPD_CAM_PITCH / ACPD_CAM_ROLL (deg, camera frame:
pitch + = further down) with spreads ACPD_CAM_SPREAD="pitch,roll" (deg), default centre (+4, -1.5) and spread (8, 4): covers
the first real mount (+8.25, -3.5 from the leg calibration, 2026-11-16) and the trained pose. Pinhole 87 x 58 deg at
ACPD_RES (default 80x45), z-depth in metres, the robot's own body included (self-occlusion), no return = 0.
Record output: ACPD_OUT/<pid>_<n>.npz per episode: depth (T, H, W) uint16 mm, prop (T, P) float32 (ACP proprio at the
camera steps), scan_plan (T, 187), scan_track (T, 63) float32 (height minus root height, as ACP reads them), root (T, 7)
pelvis position + quaternion, plan_hist (T, 495) + goal (T, 3) = the planner's other inputs, targets (T, 9, 3) its chosen
contact targets, cam (2,) the episode's camera offsets, course / seed info.
Fine-map student data (ACPD_FINE=1, default): cam_T (T, 12) the camera's rotation (3 x 3, camera -> heading frame) and
position (heading frame, relative to the root), which the robot gets from its IMU attitude + leg kinematics; fine (T, 52, 44)
float16 the true height minus root height on 2.5 cm cells (FINE_*), the label.
ACPD_HOLD_HEADING=1 (default): on a hold without a station heading, within 0.15 m, keep the heading the robot arrived with
(the eval adapter faces the bearing of the point, which swings wildly once the robot stands on it: it span in place).
ACPD_DEPTH_NOISE=1: the training's D435-like noise on the camera frames in the loop (otherwise clean ray-cast depth).
ACPD_ANCHOR=1 (student modes): shift the estimate so the map under the lower foot equals that sole's height (leg kinematics).
ACPD_EXTRA=<dir>: also write per episode the maps ACP was given (student or truth) and its planned landings, for the
renders' ACP overlay (eval/overlays.py ACPOverlay: plan_xy / plan_h, scan_xy / scan_h, tgt_w, occ, t).
"""
from __future__ import annotations

import atexit
import dataclasses
import json
import math
import os
import time

import mujoco
import numpy as np

from adapters.acp import ACPHandover, PLAN_SCAN, SCAN

HFOV, VFOV = 87.0, 58.0
# fine map around the robot (heading frame, height minus root height), the label of the fine-map student: 2.5 cm cells,
# x -0.40 .. 0.875 m (52 rows), y -0.5375 .. 0.5375 m (44 columns); covers the tracker grid and the planner's cells from
# x -0.4 m (its rear rows are not needed by ACP: 88 % with them blanked, 2026-11-16)
FINE_RES, FINE_X0, FINE_NX, FINE_NY = 0.025, -0.40, 52, 44
CAM_POS = np.array([0.131, 0.0, 0.259])          # torso_link frame (Hiking card)
CAM_DOWN = 61.6                                  # deg below the torso's x axis (Hiking card)


def _rot_y(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])


def _rot_x(deg):
    a = math.radians(deg)
    return np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])


class ChestCam:
    """Pinhole depth camera on torso_link; camera frame x forward, y left, z up (as hiking_deploy.core.camera_rays)."""

    def __init__(self, m, W, H):
        self.m, self.W, self.H = m, W, H
        fx, fy = W / 2 / math.tan(math.radians(HFOV / 2)), H / 2 / math.tan(math.radians(VFOV / 2))
        u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
        r = np.stack([np.ones_like(u), -(u - W / 2) / fx, -(v - H / 2) / fy], -1).reshape(-1, 3)
        self.rays = r / np.linalg.norm(r, axis=1, keepdims=True)
        self.xcos = self.rays[:, 0]                         # z-depth = distance * cos(angle to the optical axis)
        self.body = m.body("torso_link").id
        self.gg = np.zeros(6, np.uint8)
        self.gg[1] = self.gg[2] = self.gg[4] = self.gg[5] = 1   # robot visual geoms, terrain, walls (eval/x2.py groups)
        n = len(self.rays)
        self.gid, self.dist = np.zeros(n, np.int32), np.zeros(n)
        self.set_offset(0.0, 0.0)

    def set_offset(self, pitch, roll):
        self.R_off = _rot_y(CAM_DOWN + pitch) @ _rot_x(roll)    # torso -> camera

    def render(self, d):
        Rb = d.xmat[self.body].reshape(3, 3)
        p = d.xpos[self.body] + Rb @ CAM_POS
        dirs = np.ascontiguousarray(self.rays @ (Rb @ self.R_off).T)
        mujoco.mj_multiRay(self.m, d, p, dirs.ravel(), self.gg, 1, -1, self.gid, self.dist, None, len(dirs), 6.0)
        z = np.where(self.gid >= 0, self.dist * self.xcos, 0.0)
        return z.reshape(self.H, self.W)


class _View:
    """The harness's view with height_scan intercepted for ACP's two grids."""

    def __init__(self, view, owner):
        self._v, self._o = view, owner

    def __getattr__(self, k):
        return getattr(self._v, k)

    def height_scan(self, xy, *a, **k):
        n = len(xy)
        o = self._o
        use = {"student": (len(PLAN_SCAN), len(SCAN)), "student_plan": (len(PLAN_SCAN),), "student_track": (len(SCAN),)}
        if n in use.get(o.mode, ()) and o.est is not None:
            out = o.est[n] + o.root_z                      # the student's height minus root height, back to world heights
            o.given[n] = (np.asarray(xy, float)[:, :2], out)
            return out
        h = self._v.height_scan(xy, *a, **k)
        if n in (len(PLAN_SCAN), len(SCAN)):
            o.true[n] = np.asarray(h, float) - o.root_z
            if o.mode == "dagger" and o.est is not None:   # act on the student's scan, record the truth (the label)
                return o.est[n] + o.root_z
        if o.mode == "track_err" and n == len(SCAN):
            # true scans; the tracker's 9 x 7 gets controlled errors: per-step noise ACPD_NOISE (m, sigma) and a per-episode
            # offset ACPD_OFFSET (m, drawn +-): which error kind costs the tracker? (2026-11-16: student tracker scan -> 48 %)
            return np.asarray(h, float) + o.ep_offset + o.rng.normal(0.0, o.noise, n)
        if n in (len(PLAN_SCAN), len(SCAN)):
            o.given[n] = (np.asarray(xy, float)[:, :2], np.asarray(h, float))
        if o.mode == "noback" and n == len(PLAN_SCAN):
            # perfect scans except the planner rows more than 0.2 m behind (x -0.8 .. -0.3, the first 6 of 17), which copy the
            # x = -0.2 row: does ACP need the map behind it? (user, 2026-11-16: "technically we don't need the map behind")
            g = np.asarray(h, float).reshape(17, 11).copy()
            g[:6] = g[6]
            return g.ravel()
        return h


class ACPDepth(ACPHandover):
    def __init__(self):
        super().__init__()
        self.mode = os.environ.get("ACPD_MODE", "record")
        self.out = os.environ.get("ACPD_OUT", "acpd_out")
        W, H = (int(v) for v in os.environ.get("ACPD_RES", "80x45").split("x"))
        self.res = (W, H)
        self.cam_c = (float(os.environ.get("ACPD_CAM_PITCH", "4.0")), float(os.environ.get("ACPD_CAM_ROLL", "-1.5")))
        self.cam_s = tuple(float(v) for v in os.environ.get("ACPD_CAM_SPREAD", "8,4").split(","))
        self.cam, self.k, self.ep, self.buf = None, 0, 0, None
        self.true, self.est, self.root_z, self.given = {}, None, 0.0, {}
        self.anchor = os.environ.get("ACPD_ANCHOR", "0") == "1"   # foot-anchored map (_anchor)
        self.depth_noise = os.environ.get("ACPD_DEPTH_NOISE", "0") == "1"   # D435-like noise on the camera in the loop (_d435)
        self.z_prev, self.band, self.patches = None, 0, []
        self.soles, self.anchor_log = None, []
        self.hold_fix = os.environ.get("ACPD_HOLD_HEADING", "1") == "1"   # keep the arrival heading on a heading-less hold
        self.hold_yaw = None
        self.fine_rec = os.environ.get("ACPD_FINE", "1") == "1"   # record the fine map label + the camera pose (fine-map student)
        self.extra_dir = os.environ.get("ACPD_EXTRA")        # per-episode overlay data (ep<n>.npz, n = episode in this process)
        if self.extra_dir:
            atexit.register(self._flush_extra)
        self.student = None
        if self.mode in ("student", "student_plan", "student_track", "dagger"):
            import torch
            self.student = torch.jit.load(os.environ["ACPD_STUDENT"], map_location="cpu").eval()
            self.kind = os.environ.get("ACPD_STUDENT_KIND") or ("fine" if hasattr(self.student, "g_plan") else "grid")
            self.h = None
        if self.mode in ("record", "dagger"):
            os.makedirs(self.out, exist_ok=True)
            atexit.register(self._flush)
        self.observes = self.observes + "; height scans %s (acp_depth, chest depth %dx%d)" % (
            "recorded" if self.mode == "record" else "from the depth student", W, H) if hasattr(self, "observes") else ""

    def reset(self, view, rng):
        self._flush()
        self._flush_extra()
        if self.cam is None or self.cam.m is not view.m:
            self.cam = ChestCam(view.m, *self.res)
            m = view.m
            self.soles = [[g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name == f and m.geom_contype[g]
                           and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE] for f in ("left_ankle_roll_link", "right_ankle_roll_link")]
        self.rng = rng
        self.noise = float(os.environ.get("ACPD_NOISE", "0"))
        off = float(os.environ.get("ACPD_OFFSET", "0"))
        self.ep_offset = off * (1.0 if rng.random() < 0.5 else -1.0)
        self.z_prev = None
        H_, W_ = self.res[1], self.res[0]
        ri = rng.integers if hasattr(rng, "integers") else rng.randint     # Generator or RandomState
        self.band = int(ri(0, 4))                   # per episode: left strip width and a few fixed holes (shiny / dark spots)
        self.patches = [(int(ri(0, H_ - 12)), int(ri(0, W_ - 16)), int(ri(3, 12)), int(ri(3, 16))) for _ in range(int(ri(0, 3)))]
        self.cam_off = (self.cam_c[0] + rng.uniform(-1, 1) * self.cam_s[0], self.cam_c[1] + rng.uniform(-1, 1) * self.cam_s[1])
        self.cam.set_offset(*self.cam_off)
        self.k, self.ep = 0, self.ep + 1
        self.buf = dict(depth=[], prop=[], scan_plan=[], scan_track=[], root=[], plan_hist=[], goal=[], targets=[], cam_T=[], fine=[]) if self.mode in ("record", "dagger") else None
        self.est, self.h = None, None
        super().reset(view, rng)

    def act(self, view, cmd):
        p, _, R = view.body_pose("pelvis")
        self.root_z = float(p[2])
        cam_step = self.k % 2 == 0                         # 25 Hz camera
        self.prop_now = self._proprio(view)                # ang vel, gravity, q - q0, qd, last action (ACP's own terms)
        if cam_step and self.mode in ("record", "student", "student_plan", "student_track", "dagger"):     # noback: no camera
            z = z_clean = self.cam.render(view.d)
            if self.depth_noise:
                z = self._d435(z)                          # the student sees the noisy frame; the recorder keeps the clean one
            if self.mode != "record":
                self._estimate(z, view)
        self.true, self.given = {}, {}
        if self.hold_fix and cmd.hold and cmd.target_heading is None and cmd.dist < 0.15:
            # holding on a waypoint with no heading: the eval adapter turns to the bearing of the point, which swings wildly once
            # the robot stands on it (it span 200-490 deg at the end of the ladder courses, 2026-11-16): keep the arrival heading
            if self.hold_yaw is None:
                self.hold_yaw = math.atan2(R[1, 0], R[0, 0])
            cmd = dataclasses.replace(cmd, target_heading=self.hold_yaw)
        else:
            self.hold_yaw = None
        a = super().act(_View(view, self), cmd)
        if self.extra_dir and len(self.given) == 2:        # what ACP was given (student or truth), for eval/overlays.ACPOverlay
            last = getattr(self.contacts, "last", None) or {}
            tg = np.asarray(last.get("targets", np.zeros((9, 3))), float).reshape(-1, 3)
            yaw = math.atan2(R[1, 0], R[0, 0])
            c_, s_ = math.cos(yaw), math.sin(yaw)
            tw = np.c_[p[0] + c_ * tg[:, 0] - s_ * tg[:, 1], p[1] + s_ * tg[:, 0] + c_ * tg[:, 1], p[2] + tg[:, 2]]
            x = self.xbuf
            x["t"].append(float(view.d.time))
            x["plan_xy"].append(self.given[len(PLAN_SCAN)][0]); x["plan_h"].append(self.given[len(PLAN_SCAN)][1])
            x["scan_xy"].append(self.given[len(SCAN)][0]); x["scan_h"].append(self.given[len(SCAN)][1])
            x["tgt_w"].append(tw); x["occ"].append(np.asarray(last.get("occupancy", np.zeros(len(tg))), float))
        if self.mode in ("record", "dagger") and cam_step and len(self.true) == 2 and self.fine_rec:
            b = self.buf
            yaw = math.atan2(R[1, 0], R[0, 0])
            c_, s_ = math.cos(yaw), math.sin(yaw)
            b["cam_T"].append(self._cam_T(view))
            xs = FINE_X0 + FINE_RES * np.arange(FINE_NX)
            ys = FINE_RES * (np.arange(FINE_NY) - (FINE_NY - 1) / 2)
            gx, gy = np.meshgrid(xs, ys, indexing="ij")
            wxy = np.stack((p[0] + c_ * gx.ravel() - s_ * gy.ravel(), p[1] + s_ * gx.ravel() + c_ * gy.ravel()), 1)
            b["fine"].append((np.asarray(view.height_scan(wxy), float) - p[2]).reshape(FINE_NX, FINE_NY).astype(np.float16))
        if self.mode in ("record", "dagger") and cam_step and len(self.true) == 2:
            b = self.buf
            b["depth"].append(np.clip(z_clean * 1000.0, 0, 65535).astype(np.uint16))
            b["prop"].append(np.asarray(self.prop_now, np.float32))
            b["scan_plan"].append(self.true[len(PLAN_SCAN)].astype(np.float32))
            b["scan_track"].append(self.true[len(SCAN)].astype(np.float32))
            b["root"].append(np.concatenate((p, view.d.qpos[3:7])).astype(np.float32))
            # the planner's own inputs and its chosen targets (on the scan it was given), to replay the real planner offline on
            # true vs estimated scans: its foot target height is read straight off the scan at the chosen cell (planner.decode)
            b["plan_hist"].append(np.asarray(self.hist.flat(), np.float32))
            b["goal"].append(np.asarray(self.goal_now, np.float32))
            last = getattr(self.contacts, "last", None) or {}
            b["targets"].append(np.asarray(last.get("targets", np.zeros((9, 3))), np.float32).reshape(9, 3))
        self.k += 1
        return a

    def _estimate(self, z, view):
        import torch
        prop = np.asarray(self.prop_now, np.float32)
        with torch.no_grad():
            d_ = torch.from_numpy(z.astype(np.float32))[None, None]
            h_ = self.h if self.h is not None else torch.zeros(0)
            if self.kind == "fine":                         # fine-map student: also the camera pose (models/acp_depth/fine_student.py)
                plan, track, self.h, self.fine_now = self.student(d_, torch.from_numpy(prop)[None], torch.from_numpy(self._cam_T(view))[None], h_)
            else:
                plan, track, self.h = self.student(d_, torch.from_numpy(prop)[None], h_)
        self.est = {len(PLAN_SCAN): plan[0].numpy().astype(float), len(SCAN): track[0].numpy().astype(float)}
        if self.anchor:
            self._anchor(view)

    def _cam_T(self, view):
        """Camera rotation (camera -> heading frame) and position (heading frame, relative to the root): on the robot, from the
        IMU attitude + leg kinematics; here from the simulator's body poses (the same forward kinematics)."""
        p, _, R = view.body_pose("pelvis")
        yaw = math.atan2(R[1, 0], R[0, 0])
        c_, s_ = math.cos(yaw), math.sin(yaw)
        Rh = np.array([[c_, s_, 0], [-s_, c_, 0], [0, 0, 1]])
        Rb = view.d.xmat[self.cam.body].reshape(3, 3)
        pc = view.d.xpos[self.cam.body] + Rb @ CAM_POS
        return np.concatenate(((Rh @ (Rb @ self.cam.R_off)).ravel(), Rh @ (pc - p))).astype(np.float32)

    def _d435(self, z):
        """The training noise model (models/acp_depth/train.py d435) on one frame, in the loop: axial noise ~ z^2, holes at
        depth edges and below 0.3 m, speckle, a left invalid strip, random patches; plus a repeated (late) frame now and then."""
        r = self.rng
        z = z + r.normal(size=z.shape) * (0.002 + 0.004 * z * z)
        gx = np.abs(np.diff(z, axis=1, prepend=z[:, :1]))
        gy = np.abs(np.diff(z, axis=0, prepend=z[:1, :]))
        drop = ((np.maximum(gx, gy) > 0.06) & (r.random(z.shape) < 0.6)) | (r.random(z.shape) < 0.01) | (z < 0.3)
        drop[:, :self.band] = True
        for (y0, x0, h, w) in self.patches:
            drop[y0:y0 + h, x0:x0 + w] = True
        z = np.where(drop, 0.0, z)
        if self.z_prev is not None and r.random() < 0.05:
            return self.z_prev
        self.z_prev = z
        return z

    def _anchor(self, view):
        """Shift the estimate so the map under the lower (stance) foot equals that sole's height below the pelvis. The sole's
        height comes from the leg kinematics (joint angles + IMU attitude; here the simulator's body poses, which equal that
        forward kinematics: no terrain information). 2026-11-16: the student's map sat 2-3 cm above the ground on every course
        and 7 cm above it during stair descents (it lagged the step down)."""
        m, d = view.m, view.d
        p, _, R = view.body_pose("pelvis")
        yaw = math.atan2(R[1, 0], R[0, 0])
        c_, s_ = math.cos(yaw), math.sin(yaw)
        best = None
        for gs in self.soles:
            z = min(d.geom_xpos[g][2] - m.geom_size[g][0] for g in gs)
            xy = np.mean([d.geom_xpos[g][:2] for g in gs], 0) - p[:2]
            hf = np.array([c_ * xy[0] + s_ * xy[1], -s_ * xy[0] + c_ * xy[1]])      # heading frame
            if best is None or z < best[0]:
                best = (z, hf)
        z, hf = best
        sole_rel = z - p[2]
        g = self.est[len(SCAN)].reshape(9, 7)                                       # tracker grid: x -0.2 .. 0.6, y -0.3 .. 0.3
        fx, fy = (hf[0] + 0.2) / 0.1, (hf[1] + 0.3) / 0.1
        if 0 <= fx <= 8 and 0 <= fy <= 6:
            i, j = min(int(fx), 7), min(int(fy), 5)
            a_, b_ = fx - i, fy - j
            at = (g[i, j] * (1 - a_) * (1 - b_) + g[i + 1, j] * a_ * (1 - b_) + g[i, j + 1] * (1 - a_) * b_ + g[i + 1, j + 1] * a_ * b_)
            off = float(np.clip(sole_rel - at, -0.15, 0.15))
            for k in self.est:
                self.est[k] = self.est[k] + off
            self.anchor_log.append(off)

    def _flush_extra(self):
        x = getattr(self, "xbuf", None)
        if self.extra_dir and x and x["t"]:
            os.makedirs(self.extra_dir, exist_ok=True)
            np.savez_compressed(os.path.join(self.extra_dir, "ep%04d.npz" % self.ep), **{k: np.asarray(v) for k, v in x.items()})
        self.xbuf = dict(t=[], plan_xy=[], plan_h=[], scan_xy=[], scan_h=[], tgt_w=[], occ=[])

    def _flush(self):
        b = getattr(self, "buf", None)
        if not b or not b["depth"]:
            return
        # pid + a per-process unique stamp: one worker runs several courses with a fresh adapter each (2026-11-16: pid_episode
        # names collided across courses and 2/3 of data_v1 was overwritten)
        name = os.path.join(self.out, "%d_%d_%04d.npz" % (os.getpid(), int(time.time() * 1e3) % 10 ** 9, self.ep))
        np.savez_compressed(name, **{k: np.stack(v) for k, v in b.items()}, cam=np.array(self.cam_off, np.float32),
                            res=np.array(self.res), meta=json.dumps(dict(t=time.time(), steps=len(b["depth"]))))
        self.buf = dict(depth=[], prop=[], scan_plan=[], scan_track=[], root=[], plan_hist=[], goal=[], targets=[])
