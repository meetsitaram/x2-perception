"""Several X2s on the terrain park in one film: ACP on the fine depth student, each robot an eval episode replayed in one
textured scene (the physics ran in the eval, one robot per simulation; park_sims.py). Follows the ACP render rules
(render_showcase.sh): ACP's overlay drawn from the maps ACP was GIVEN (planner tiles, tracker dots, planned landing rings,
stance squares; never the simulator's map), the featured robot's chest depth inset bottom right (what its camera saw: the
other robots are not in it), one apparatus model (the CAD regenerated from the current revision, which matches the eval's
geometry: models/acp_depth/runs/park_app), a one-line caption. A robot flashes red during each eval side hit (user,
2026-11-17: "Any robot that stumbles - flash it red during that stumble"; the stumble-penalty events, docs/EVAL.md).

    python3 models/acp_depth/render_park.py --scene runs/park_scene --sims runs/park --app-run runs/vis_fine3/app \
        --app-extra runs/vis_fine3/extra_app/ep0001.npz --out runs/park_film/park.mp4

Shots: a wide opening over the whole park, then one chase shot per feature (4x4s, 4x4 field, blocks, ramps, wedges, stairs
straight / L / U, up and down), then the full apparatus route at 2x.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys

import mujoco
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "eval"))
sys.path.insert(0, str(HERE))
import overlays as OV  # noqa: E402
import render as RD  # noqa: E402
from render_vis import cam_R  # noqa: E402

ROBOT = pathlib.Path(os.environ.get("PA_ROBOT_DIR", pathlib.Path.home() / "Projects/gwbc-x2/gear_sonic/data/assets/robot_description")) / "mjcf" / "x2_ultra.xml"
NQ = 7 + 31
SOLE = 0.073                                       # ankle_roll_link origin above the sole's lowest point (eval/x2.py SOLE_Z)
TITLE = {"4x4s": "stepping over 4x4s", "4x4_field": "a field of 4x4s", "blocks": "concrete blocks", "ramps": "up and down ramps",
         "wedges": "the wedges", "stairs_straight_up": "straight stairs, up and down", "stairs_L_up": "L-shaped stairs, up",
         "stairs_L_down": "L-shaped stairs, down", "stairs_U_up": "U-shaped stairs, up", "stairs_U_down": "U-shaped stairs, down",
         "stairs_straight_down": "straight stairs, down", "homes": "a home: walls, doors, tables and chairs",
         "homes_L6": "another home layout", "blocks_big": "big concrete blocks (up to 25 cm)",
         "wedges_steep": "steep slanted wedges", "hallway": "a hallway with 4x4s", "hallway_L6": "another hallway with 4x4s",
         "hallway_L8": "a third hallway with 4x4s", "apparatus": "our apparatus, the full route"}


FLASH_MIN = 0.3                                    # s: a stumble stays red at least this long
# A stumble: a contact that jolts the robot (user, 2026-11-17: "The red flash sometimes hits even stable x2s"; every
# hit counts from 20 N, which a stable robot shrugs off). 0.3 m/s is the eval penalty's full-disturbance scale (docs/EVAL.md).
JOLT_DV, JOLT_TILT = 0.30, 5.0
RED = np.array([1.0, 0.08, 0.05, 1.0], np.float32)


class Robot:
    def __init__(self, name, slot, traj, extra, shift, results=None):
        z = np.load(traj)
        ev = json.load(open(results))[0].get("events", []) if results else []
        def jolt(e):
            if e.get("hit"):                          # a scored side hit (4x4 / wedge / stair)
                return e.get("dv", 0.0) >= JOLT_DV or e.get("dtilt", 0.0) >= JOLT_TILT
            # walls and furniture: logged by the eval but never flagged as hits (user, 2026-11-17: a robot bumping a home's
            # furniture did not flash); stricter, since normal gait moves the pelvis 0.2-0.3 m/s
            return e.get("peak_h", 0.0) >= 100.0 and (e.get("dv", 0.0) >= 0.4 or e.get("dtilt", 0.0) >= JOLT_TILT)
        self.hits = [(e["t"] - 0.02, e["t"] + max(e.get("dur", 0.0), FLASH_MIN)) for e in ev if jolt(e)]
        self.name, self.slot = name, slot
        self.t, self.qpos, self.load = z["t"], z["qpos"][:, :NQ].copy(), z["load"]
        self.qpos[:, :3] += shift
        self.e = dict(np.load(extra)) if extra and os.path.exists(extra) else None
        self.shift = np.asarray(shift, float)
        self.T = float(self.t[-1])
        # the end of the walk: the last time the pelvis moves faster than 0.05 m/s (user, 2026-11-17: "too much idle time
        # after the robot stops at the end of each walk")
        v = np.linalg.norm(np.diff(self.qpos[:, :2], axis=0), axis=1) / np.maximum(np.diff(self.t), 1e-6)
        moving = np.nonzero(v > 0.05)[0]
        self.T_move = float(self.t[moving[-1] + 1]) + 0.4 if len(moving) else self.T
        # and its start: 1 s after it first moves (user: "cut down first second of each walk where it is still getting started")
        self.T_go = min(float(self.t[moving[0]]) + 1.0 if len(moving) else 0.0, self.T_move)
        self.mid = min(self._busiest(), self.T_move)

    def _busiest(self):
        """When the feet meet the obstacle: the peak of the height range in ACP's own 9 x 7 tracker scan around the feet
        (smoothed over 1.5 s); without the scan, the most pelvis travel up and down over 6 s."""
        if self.e is not None and len(self.e["t"]) > 10:
            rng = np.ptp(np.asarray(self.e["scan_h"], float), axis=1)
            dt = float(np.median(np.diff(self.e["t"])))
            w = max(1, int(1.5 / dt))
            sm = np.convolve(rng, np.ones(w) / w, "same")
            return float(self.e["t"][int(np.argmax(sm))])
        q = self.qpos[:, 2]
        if len(q) < 10:
            return self.T / 2
        dz = np.abs(np.diff(q))
        dt = float(np.median(np.diff(self.t)))
        w = max(1, int(6.0 / dt))
        if len(dz) <= w:
            return self.T / 2
        s = np.convolve(dz, np.ones(w), "valid")
        return float(self.t[int(np.argmax(s)) + w // 2])

    def stumbling(self, tl):
        return any(a <= tl <= b for a, b in self.hits)

    def k(self, tl):
        return int(np.clip(np.searchsorted(self.t, tl), 0, len(self.t) - 1))

    def ei(self, tl):
        return int(np.clip(np.searchsorted(self.e["t"], tl - 0.021), 0, len(self.e["t"]) - 1))


def build(scene_xml, n, size):
    """The park scene with its own X2 (the sim builder puts one in) and the chest camera on it. Attaching a second robot
    MJCF corrupted the floor textures (blue patches through the terrain, 2026-11-17), so the scene's robot is the one drawn,
    once per episode each frame."""
    spec = mujoco.MjSpec.from_file(str(scene_xml))
    spec.visual.global_.offwidth, spec.visual.global_.offheight = size
    R = cam_R()
    xa, ya = R @ np.array([0.0, -1.0, 0.0]), R @ np.array([0.0, 0.0, 1.0])
    c = spec.body("torso_link").add_camera(name="realsense_sim", pos=[0.131, 0.0, 0.259], fovy=OV.DepthOverlay.FOVY)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.stack((xa, ya, np.cross(xa, ya)), 1).ravel())     # as eval/x2.py: cameras look along -z
    c.quat = q
    m = spec.compile()
    m.vis.map.znear = 0.06 / max(m.stat.extent, 1e-3)
    return m, [m.jnt_qposadr[m.joint("floating_base_joint").id]]


# Height-map colours by height above the stance foot (user, 2026-11-17: "anything at 4x4 height - make yellow, stairs level
# orange, concrete block level red"): level green, 4x4 (8.9 cm) yellow, a stair step (11-19 cm) orange, a block red, below blue.
def height_rgba(dh, a=0.8):
    if dh < -0.04:
        return np.array([0.15, 0.4, 1.0, a], np.float32)
    if dh < 0.04:
        return np.array([0.2, 0.85, 0.3, a], np.float32)
    if dh < 0.12:
        return np.array([1.0, 0.95, 0.1, a], np.float32)
    if dh < 0.20:
        return np.array([1.0, 0.55, 0.05, a], np.float32)
    return np.array([0.95, 0.1, 0.1, a], np.float32)


# planned landings: bold and saturated (user, 2026-11-17: "make the contact circles more contrasted. Right now the flat square
# is the main focus area"); left blue, right orange as in the eval's overlay
RING = (np.array([0.0, 0.45, 1.0, 1.0], np.float32), np.array([1.0, 0.35, 0.0, 1.0], np.float32))


def draw_overlay(scn, rb, tl, d, feet_ids):
    e = rb.e
    if e is None:
        return
    i = rb.ei(tl)
    k = rb.k(tl)
    load = rb.load[k]
    feet = [d.xpos[b] for b in feet_ids]
    st = int(np.argmax(load))
    stance = float(feet[st][2]) - SOLE
    floor = float(rb.shift[2])                         # the tile's floor (the apparatus: its deck)
    # colours by height above the floor, so a block stays red with a foot on it (user: "Not enough contrast for such high
    # block"); on stairs (and the apparatus stairs / landing) above the stance foot, so the next step is orange
    ref = stance if rb.slot.startswith("stairs") or stance - floor > 0.35 else floor
    sx, sy, sz = rb.shift
    for (x_, y_), h in zip(e["plan_xy"][i], e["plan_h"][i]):
        OV._add(scn, OV.BOX, [0.034, 0.034, 0.002], [x_ + sx, y_ + sy, h + sz + 0.005], height_rgba(h + sz - ref, 0.5))
    for (x_, y_), h in zip(e["scan_xy"][i], e["scan_h"][i]):
        OV._add(scn, OV.SPH, [0.011, 0, 0], [x_ + sx, y_ + sy, h + sz + 0.012], [0.08, 0.08, 0.1, 1])
    for f in range(2):
        if e["occ"][i][f] > 0.5:
            px, py, pz = e["tgt_w"][i][f]
            OV._add(scn, OV.CYL, [0.10, 0.006, 0], [px + sx, py + sy, pz + sz + 0.02], RING[f] * np.array([1, 1, 1, 0.55], np.float32))
            for a in np.linspace(0, 2 * np.pi, 20, endpoint=False):
                OV._add(scn, OV.SPH, [0.02, 0, 0], [px + sx + 0.10 * math.cos(a), py + sy + 0.10 * math.sin(a), pz + sz + 0.03],
                        RING[f])
        if load[f] > 20.0:
            q = feet[f]
            OV._add(scn, OV.BOX, [0.025, 0.025, 0.012], [q[0], q[1], q[2] - SOLE + 0.03], OV.FOOT[f] * np.array([0.5, 0.5, 0.5, 1], np.float32))


def draw_inset(im, dr, dep, top, font):
    """The chest depth, top right under the caption, 22 % of the width (user, 2026-11-17: "Depth cam blocking contact points
    most of the time" bottom right, where the low leg camera puts the feet)."""
    from PIL import Image
    D = OV.DepthOverlay
    rgb = OV._cmap((dep - D.NEAR) / (D.FAR - D.NEAR))
    rgb[~((dep >= D.NEAR) & (dep <= 6.0))] = 0
    W, _ = im.size
    sw = int(round(W * 0.22))
    sh = int(round(sw * D.RH / D.RW))
    x0, y0 = W - sw - 8, top + 24
    dr.rectangle((x0 - 4, top + 4, W - 4, y0 + sh + 26), fill=(15, 18, 24))
    dr.text((x0, top + 5), "depth (chest camera)", fill=(230, 230, 230), font=font)
    im.paste(Image.fromarray(rgb).resize((sw, sh), Image.BILINEAR), (x0, y0))
    im.paste(Image.fromarray(OV._cmap(np.linspace(0, 1, sw))[None].repeat(5, 0)), (x0, y0 + sh + 2))
    for mtr in (1, 2, 3):
        xx = x0 + int((mtr - D.NEAR) / (D.FAR - D.NEAR) * (sw - 1))
        dr.text((xx - 8, y0 + sh + 7), "%g m" % mtr, fill=(230, 230, 230), font=font)


def main():
    import faulthandler
    faulthandler.dump_traceback_later(int(os.environ.get("RP_HANG_S", "0")) or 10 ** 6, exit=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--sims", required=True)
    ap.add_argument("--app-run", required=True)
    ap.add_argument("--app-extra", required=True)
    ap.add_argument("--app-seed", type=int, default=2007)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--size", default="1280x720")
    ap.add_argument("--label", default="ACP on depth only (fine-map student)")
    ap.add_argument("--max-per-slot", type=int, default=5)
    ap.add_argument("--check", action="store_true", help="only write one still per shot")
    ap.add_argument("--cut", default="", help="slot:seconds,... the close-ups to show, in order (default: every slot); "
                    "the opening and the apparatus are always in")
    ap.add_argument("--wide-s", type=float, default=14.0, help="opening fly-over length (s)")
    a = ap.parse_args()
    W, H = (int(v) for v in a.size.split("x"))
    scene = pathlib.Path(a.scene)
    info = json.load(open(scene / "scene.json"))
    shift_of = {t["course"].strip(): np.array(t["shift"], float) for t in info["tiles"]}
    sims = pathlib.Path(a.sims)
    plan = json.load(open(sims / "plan.json"))
    tiles = {"single_beams": "tp:single_beams:4", "beam_fields": "tp:beam_fields:4", "blocks": "tp:blocks:4", "stairs": None, "wedges": None,
             "homes": "tp:homes:4"}
    robots, per = [], {}
    for j in plan["jobs"]:
        d = sims / "ladder" / "park" / j["name"]
        try:
            r = json.load(open(d / "results.json"))[0]
        except (OSError, ValueError):
            continue
        if not r["success"] or per.get(j["slot"], 0) >= a.max_per_slot:
            continue
        lev = int(j["tile_id"].split("_L")[1].split("_")[0])
        vk = ({"ramps": "long_ramp", "wedges": "pinwheel", "wedges_steep": "pinwheel"}.get(j["slot"]) if j["ttype"] == "wedges"
              else j["slot"].split("_")[1] if j["ttype"] == "stairs" else "")
        key = ("tp:%s:%d %s" % (j["ttype"], lev, vk)).strip()
        sh = shift_of.get(key)
        if sh is None:
            print("no tile for", j["name"], key)
            continue
        per[j["slot"]] = per.get(j["slot"], 0) + 1
        robots.append(Robot(j["name"], j["slot"], d / "traj" / "3000.npz", sims / ("extra_" + j["name"]) / "ep0001.npz", sh,
                            d / "results.json"))
    app_sh = shift_of["apparatus:eval"]
    robots.append(Robot("apparatus", "apparatus", pathlib.Path(a.app_run) / "traj" / ("%d.npz" % a.app_seed), a.app_extra, app_sh,
                        pathlib.Path(a.app_run) / "results.json"))
    print("%d robots: %s" % (len(robots), ", ".join("%s(%.0fs)" % (r.slot, r.T) for r in robots)), flush=True)

    print("building the scene ...", flush=True)
    m, adr = build(scene / "scene.xml", 1, (W, H))   # ONE robot, drawn once per episode each frame (47 copies took minutes to compile)
    A = adr[0]
    print("scene: %d geoms" % m.ngeom, flush=True)
    d = mujoco.MjData(m)
    foot_ids = [m.body("%s_ankle_roll_link" % s).id for s in ("left", "right")]
    cam_id = m.camera("realsense_sim").id
    pert = mujoco.MjvPerturb()

    class Feet:                                       # foot positions of one robot's pose, for draw_overlay
        def __init__(self, xp):
            self.xpos = xp

    def pose(i, tl):
        d.qpos[A:A + NQ] = robots[i].qpos[robots[i].k(tl)]
        mujoco.mj_kinematics(m, d)
        mujoco.mj_camlight(m, d)                      # camera poses (the chest camera) are not part of mj_kinematics

    ren = mujoco.Renderer(m, H, W, max_geom=60000)
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    for g in (0, 1, 2, 4, 5):
        opt.geomgroup[g] = 1
    dren = mujoco.Renderer(m, OV.DepthOverlay.RH, OV.DepthOverlay.RW, max_geom=60000)
    dren.enable_depth_rendering()
    dopt = mujoco.MjvOption()
    dopt.geomgroup[:] = 0
    for g in (1, 2, 4, 5):
        dopt.geomgroup[g] = 1
    inset = OV.DepthOverlay.__new__(OV.DepthOverlay)
    inset.e, inset.minimal = dict(t=np.zeros(1), depth=np.zeros((1, 45, 80))), True

    # shots: (title, featured robot index or None, duration s, speed, camera)
    shots = [("the terrain park", None, a.wide_s, 1.0, "wide")]
    order = ["4x4s", "4x4_field", "blocks", "ramps", "wedges", "stairs_straight_up", "stairs_straight_down", "stairs_L_up",
             "stairs_L_down", "stairs_U_up", "stairs_U_down", "homes", "homes_L6", "blocks_big", "wedges_steep", "hallway",
             "hallway_L6", "hallway_L8"]
    cut = [(c.split(":")[0], float(c.split(":")[1])) for c in a.cut.split(",") if c] or \
          [(s, 9.0 if s.startswith("stairs") else 7.0) for s in order]
    for s, length in cut:
        idx = [i for i, r in enumerate(robots) if r.slot == s]
        if idx:
            # the best take of the slot: the biggest obstacle under the feet (ACP's tracker-scan height range at its contact),
            # without a stumble in the shot (user, 2026-11-17: "pick the best parts")
            def take(i, L=length):
                r = robots[i]
                peak = float(np.ptp(np.asarray(r.e["scan_h"], float), axis=1).max()) if r.e is not None else 0.0
                a0 = float(np.clip(r.mid - L / 2, r.T_go, max(r.T_move - L, r.T_go)))
                clean = not any(a0 <= h0 <= a0 + L for h0, _ in r.hits)
                return (clean, peak)
            idx = sorted(idx, key=take, reverse=True)
            stairs = s.startswith("stairs")
            shots.append((TITLE[s], idx[0], min(length, robots[idx[0]].T_move), 1.0,
                          "stairs" if stairs else "hall" if s.startswith("hallway") else "chase"))
    ia = len(robots) - 1
    shots.append((TITLE["apparatus"], ia, robots[ia].T_move - robots[ia].T_go, 1.0, "app"))       # normal speed (user, 2026-11-17)

    import imageio
    from PIL import Image, ImageDraw
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    wr = None if a.check else imageio.get_writer(str(out), fps=a.fps, codec="libx264", macro_block_size=1,
                                                   ffmpeg_params=["-crf", "21", "-preset", "medium"])
    sc = H / 720.0
    font, small = RD._font(max(10, round(19 * sc))), RD._font(max(9, round(16 * sc)))
    bar = max(12, round(26 * sc)) + 6
    cx = np.mean([r.qpos[0, 0] for r in robots])
    cy = np.mean([r.qpos[0, 1] for r in robots])
    x_lo, x_hi = min(r.qpos[0, 0] for r in robots), max(r.qpos[0, 0] for r in robots)
    for si, (title, fi, dur, speed, kind) in enumerate(shots):
        n = int(round(dur * a.fps))
        # each robot's local time in this shot: the featured one from its window, the others in the middle of their action
        # Robots share tiles (separate simulations, overlapping routes): each one's start is chosen to keep it clear of the
        # robots placed before it (the featured one first), preferring the middle of its action (user, 2026-11-17: "Why so
        # many robots at one spot?").
        start = {}
        samp = np.arange(0.0, dur * speed + 1e-9, 0.5)
        placed = []                                   # (robot, start)

        def track(r, s0):
            return np.array([r.qpos[r.k(s0 + u)][:2] for u in samp])

        order_i = ([fi] if fi is not None else []) + [i for i in range(len(robots)) if i != fi]
        for i in order_i:
            r = robots[i]
            if i == fi:
                s0 = r.T_go if kind == "app" else float(np.clip(r.mid - dur / 2, r.T_go, max(r.T_move - dur, r.T_go)))
            else:
                near = [(q, sq) for q, sq in placed if np.linalg.norm(robots[q].qpos[0, :2] - r.qpos[0, :2]) < 12.0]
                best, s0 = -1e9, 0.0
                for c in np.arange(r.T_go, max(r.T_move - dur * speed, r.T_go) + 1e-9, 0.5):     # moving through the shot if it can
                    tr = track(r, c)
                    clear = min((float(np.min(np.linalg.norm(tr - track(robots[q], sq), axis=1))) for q, sq in near), default=9.0)
                    score = min(clear, 2.5) - 0.05 * abs(c - (r.mid - dur / 2))
                    if score > best:
                        best, s0 = score, float(c)
            start[i] = s0
            placed.append((i, s0))
        lk, yaw_s = None, None
        frames = [n // 2] if a.check else range(n)
        for f in frames:
            ts = f / a.fps * speed
            cam = mujoco.MjvCamera()
            if kind == "wide":
                u = f / max(n - 1, 1)                 # a slow fly-over along the park, close enough to see the robots
                span = min(0.7, 0.05 * dur)            # a short opening travels less, so it does not rush
                cam.lookat[:] = [x_lo + (x_hi - x_lo) * (0.5 - span / 2 + span * u), cy, 0.4]
                cam.distance, cam.elevation, cam.azimuth = 14.0, -30.0, 60.0 - 20.0 * u
            else:
                r = robots[fi]
                qf = r.qpos[r.k(start[fi] + ts)]
                p = qf[:3].copy()
                w_, x_, y_, z_ = qf[3:7]
                yaw = math.atan2(2 * (w_ * z_ + x_ * y_), 1 - 2 * (y_ * y_ + z_ * z_))
                yaw_s = yaw if yaw_s is None else yaw_s + 0.04 * math.atan2(math.sin(yaw - yaw_s), math.cos(yaw - yaw_s))
                lk = p.copy() if lk is None else lk + 0.15 * (p - lk)
                # the legs and the steps from the robot's 2 o'clock (front right, 60 deg off its heading), low (user, 2026-11-17)
                cam.lookat[:] = [lk[0], lk[1], lk[2] - 0.12]           # feet to head in the frame (user: "head is cut off")
                cam.azimuth = math.degrees(yaw_s) + 120.0
                # hallways: the corridor walls (2.44 m, part of the terrain height field, cannot be faded) are looked over
                cam.distance, cam.elevation = {"chase": (3.1, -12.0), "stairs": (3.4, -20.0), "app": (3.4, -20.0),
                                               "hall": (3.4, -58.0)}[kind]
            d.qpos[A + 2] = -50.0
            mujoco.mj_kinematics(m, d)
            ren.update_scene(d, cam, opt)              # the park
            look = np.array(cam.lookat)
            scn = ren.scene
            if kind != "wide":                        # walls, door headers, furniture between the camera and the robot: faded
                az, el = math.radians(cam.azimuth), math.radians(cam.elevation)
                eye = look - cam.distance * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
                for tgt_ in (look + [0, 0, 0.2], look - [0, 0, 0.35]):
                    ab = tgt_ - eye
                    for gi in range(scn.ngeom):
                        gm = scn.geoms[gi]
                        if gm.objtype != mujoco.mjtObj.mjOBJ_GEOM or m.geom_group[gm.objid] not in (0, 5):
                            continue
                        c = np.array(gm.pos)
                        if c[2] < look[2] - 0.55:         # floors, steps and beams under the robot stay
                            continue
                        tt = float(np.clip((c - eye) @ ab / (ab @ ab), 0.0, 1.0))
                        if tt < 0.95 and np.linalg.norm(eye + tt * ab - c) < float(m.geom_rbound[gm.objid]) + 0.3:
                            gm.rgba[3] = min(gm.rgba[3], 0.12)
            pf = robots[fi].qpos[robots[fi].k(start[fi] + ts)][:3] if fi is not None else None
            if pf is not None:
                az, el = math.radians(cam.azimuth), math.radians(cam.elevation)
                eye = look - cam.distance * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
            drawn = []
            for i in order_i:                         # every robot: its pose, its geoms; red while stumbling
                r = robots[i]
                tl = start[i] + ts
                pxy = r.qpos[r.k(tl)][:2]
                if any(np.linalg.norm(pxy - q) < 0.7 for q in drawn):     # still on top of another one: not drawn this frame
                    continue
                drawn.append(pxy)
                if pf is not None and i != fi:        # the robot in focus stays the subject: nobody on it or in front of it
                    pi = r.qpos[r.k(tl)][:3]
                    if np.linalg.norm(pi[:2] - pf[:2]) < 1.0:
                        continue
                    ab = pf - eye
                    tt = float(np.clip((pi - eye) @ ab / (ab @ ab), 0.0, 1.0))
                    if tt < 0.98 and np.linalg.norm(eye + tt * ab - pi) < 1.3:
                        continue
                pose(i, tl)
                n0 = scn.ngeom
                mujoco.mjv_addGeoms(m, d, opt, pert, mujoco.mjtCatBit.mjCAT_DYNAMIC, scn)
                keepg = range(n0, scn.ngeom)
                if r.stumbling(tl):
                    for g in keepg:
                        scn.geoms[g].rgba[:] = RED
                if kind == "wide" or np.linalg.norm(r.qpos[r.k(tl)][:2] - look[:2]) < 7.0:   # ACP's overlay on every robot in view
                    draw_overlay(scn, r, tl, Feet(d.xpos.copy()), foot_ids)                    # (user, 2026-11-17)
            im = Image.fromarray(ren.render())
            dr = ImageDraw.Draw(im)
            dr.rectangle((0, 0, W, bar), fill=(15, 18, 24))
            dr.text((10, 4), "%s | %s" % (a.label, title), fill=(240, 240, 240), font=font)
            if fi is not None:                        # the featured robot's chest depth: only that robot is in the model
                pose(fi, start[fi] + ts)
                dren.update_scene(d, camera=cam_id, scene_option=dopt)
                inset.depth = dren.render().copy()
                draw_inset(im, dr, inset.depth, bar, small)
            if a.check:
                im.save(out.parent / ("check_%02d.jpg" % si), quality=88)
            else:
                wr.append_data(np.asarray(im))
        print("shot %d/%d %-40s %.1f s" % (si + 1, len(shots), title, dur), flush=True)
    if wr is not None:
        wr.close()
    print("->", out)


if __name__ == "__main__":
    main()
