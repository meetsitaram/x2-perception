"""Videos of recorded trials, from MuJoCo's own camera on the recorded simulator states (qpos replayed frame by frame).

    python eval/render.py RUN_DIR [--seeds 1000 1003] [--views chase legs] [--track NAME] [--t0 S --t1 S] [--stills N]

Two views per trial:
  chase   behind and to the side of the robot, the whole body in frame. Apparatus walls are drawn see-through so they never
          hide the robot (they are solid in the physics).
  legs    side-on at shin height, half speed. Walls are not drawn (still in the physics); the camera takes the side of the
          robot where the ground does not rise between it and the legs.
The current waypoint is a thin magenta pole with a ball on top that floats from 0.9 m to 1.6 m above the ground: nothing
of it is at foot height, so it cannot be read as a foothold. Side hits (a foot, shin or knee against the side of a 4x4, a
wedge or a stair riser) are marked where they happened: a ball at the contact point with a thin post above it, red for a
4x4, yellow for a wedge, cyan for a stair, for at least 0.8 s, with a red banner giving the limb, force, impulse and
penalty; a scuff below the hit threshold is a small grey ball. Every frame is labelled with simulator, track, checkpoint,
apparatus and seed, and carries the simulator's own readings (foot loads, pelvis height) so the picture can be checked.

Scene look (2026-11-14, user: renders had "too much saturation" and were "single colored"): the apparatus is drawn from its own
MJCF and materials and an extended-map tile from its export primitives, coloured per class (eval/dress.py); softer light. The
physics height field and wall boxes stay in the model (hidden), so replays and depth rays are unchanged. RENDER_DRESS=0 gives
the old look. Overlays (eval/overlays.py): `overlay` adds scene geoms per frame, a legend line and an inset (ACP: the planner's
height map, the tracker scan, side obstacles, planned vs physics foot contacts; Hiking / SSR: the policy's depth image).
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

def _wrap(dr, text, font, width):
    """Split a label at ' | ' (then at spaces) into lines that fit `width` pixels."""
    out, cur = [], ""
    for part in text.split(" | "):
        cand = part if not cur else cur + " | " + part
        if dr.textlength(cand, font=font) <= width:
            cur = cand
            continue
        if cur:
            out.append(cur)
        cur = ""
        for w in part.split(" "):
            c2 = w if not cur else cur + " " + w
            if dr.textlength(c2, font=font) <= width or not cur:
                cur = c2
            else:
                out.append(cur)
                cur = w
    if cur:
        out.append(cur)
    return out


CREDIT = os.environ.get("RENDER_CREDIT")          # per-track defaults below (2026-11-13: Hiking renders carried a wrong Bones Studio credit)
CREDITS = {"acp": "Motion data used in training: Bones Studio and the user's Pico captures. Robot: AgiBot X2 model.",
           "ssr": "Style reference in training: the user's Pico captures and the Hiking in the Wild paper motions retargeted to X2 (CC BY-NC 4.0). Robot: AgiBot X2 model.",
           "hiking": "Style reference in training: Hiking in the Wild paper motions retargeted to X2 (InstinctLab release, CC BY-NC 4.0). Robot: AgiBot X2 model."}


def credit_for(track):
    return CREDIT or CREDITS.get(str(track).lower(), "Robot: AgiBot X2 model.")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import x2 as X  # noqa: E402
from apparatus import Apparatus  # noqa: E402

MAGENTA = np.array([0.95, 0.10, 0.75, 1.0], np.float32)
# side hits (harness: foot, shin or knee against the side of a 4x4, a wedge or a stair riser), by obstacle class
HIT_RGBA = {"4x4": np.array([1.0, 0.05, 0.05, 0.9], np.float32), "wedge": np.array([1.0, 0.85, 0.0, 0.9], np.float32),
            "stair": np.array([0.0, 0.9, 1.0, 0.9], np.float32)}
HIT_NAME = {"4x4": "red", "wedge": "yellow", "stair": "cyan"}
HIT_SHOW = 0.8          # s: a hit's marker stays at least this long


def _add(scn, gtype, size, pos, rgba, mat=None):
    if scn.ngeom >= scn.maxgeom:
        return
    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], gtype, np.asarray(size, float), np.asarray(pos, float),
                        (np.eye(3) if mat is None else mat).ravel(), rgba)
    scn.ngeom += 1


def _font(size):
    from PIL import ImageFont
    for f in ("arial.ttf", "DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def render(run: pathlib.Path, seed: int, view: str, track: str, t0=None, t1=None, stills=0, fps=25, size=(1280, 720), caption=None, out_path=None, crf=22,
           app=None, overlay=None, dress=None, speed=None):
    """caption: replaces the two top label lines (text split at ' | '); out_path: the MP4 to write; fonts scale with the height.
    app: the course (default: from summary.json); overlay: see eval/overlays.py; dress: scene dressing on (default RENDER_DRESS,
    on); speed: playback speed factor (default 1, legs 0.5)."""
    import imageio
    from PIL import Image, ImageDraw
    s = json.load(open(run / "summary.json"))
    res = {r["seed"]: r for r in json.load(open(run / "results.json"))}[seed]
    z = np.load(run / "traj" / ("%d.npz" % seed))
    from ladder import load_course
    if app is None:
        app = None if s["apparatus"] == "flat" else load_course(s["apparatus"])
    if dress is None:
        dress = os.environ.get("RENDER_DRESS", "1") != "0"
    import dress as DR
    dz = DR.for_course(app) if dress else None
    x = X.build(app, wall_alpha=0.30, render_size=size, dress=dz, cameras=getattr(overlay, "cameras", None))       # same model; gains do not matter for a replay
    dressed = getattr(x, "dress", None)
    m = x.model
    d = mujoco.MjData(m)
    W, H = size
    ren = mujoco.Renderer(m, H, W, max_geom=20000)
    cam, opt = mujoco.MjvCamera(), mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[1] = opt.geomgroup[2] = 1                    # the robot's visual meshes
    opt.geomgroup[X.GROUP_TERRAIN] = 1
    opt.geomgroup[X.GROUP_WALL] = 1 if view == "chase" else 0
    walls_all = set(x.wall_geoms)
    if dressed is not None:                                    # the dressed scene replaces the drawn height field and boxes
        opt.geomgroup[0] = 1
        opt.geomgroup[X.GROUP_TERRAIN] = opt.geomgroup[X.GROUP_WALL] = 0
        walls_all = set(dressed.walls)
    legs = view == "legs"
    t = z["t"]
    k0 = int(np.searchsorted(t, t0)) if t0 is not None else 0
    k1 = int(np.searchsorted(t, t1)) if t1 is not None else len(t)
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.02
    every = max(1, int(round(1.0 / (fps * dt))))
    slow = speed if speed is not None else (0.5 if legs else 1.0)
    name = "%s_seed%d_%s" % (view, seed, "t%.0f-%.0f" % (t[k0], t[min(k1, len(t)) - 1]) if (t0 is not None or t1 is not None) else "full")
    out = run / "renders" if out_path is None else pathlib.Path(out_path).parent
    out.mkdir(parents=True, exist_ok=True)
    path = out / (name + ".mp4") if out_path is None else pathlib.Path(out_path)
    wr = imageio.get_writer(str(path), fps=slow / (every * dt), codec="libx264", macro_block_size=1, ffmpeg_params=["-crf", str(crf), "-preset", "medium"])
    sc = H / 720.0
    font, small = _font(max(10, round(19 * sc))), _font(max(9, round(16 * sc)))
    LB, LSm = max(12, round(26 * sc)), max(11, round(22 * sc))
    label = "%s (MuJoCo, not the training simulator) | track: %s | %s: %s | seed %d | %s, %d trials" % (s["simulator"], track, "LADDER TEST (not the apparatus)" if s.get("ladder_step") else "apparatus",
                                                       s["apparatus"].replace("ladder:", ""), seed, s.get("protocol", ""), s.get("trials", 0))
    label2 = ("%s | " % s["label"] if s.get("label") else "") + "plant %s | checkpoint: %s" % (s.get("plant", "?"), s["checkpoint"])
    if caption is not None:
        label, label2 = caption, ""
    minimal = os.environ.get("RENDER_MINIMAL", "0") == "1"     # 2026-11-15, user: "one or two liner with least number of words"
    if minimal and overlay is not None:
        overlay.minimal = True
    frames = list(range(k0, k1, every))
    still_at = set(frames[int(i * (len(frames) - 1) / max(stills - 1, 1))] for i in range(stills)) if stills and frames else set()
    lk, yaw_s, side = None, None, 1.0
    wps = app.waypoints if app is not None else []
    marks = [e for e in res.get("events", []) if "penalty" in e]         # side hits and scuffs on the three obstacle classes
    for k in frames:
        d.qpos[:] = z["qpos"][k]
        mujoco.mj_forward(m, d)
        p = d.qpos[:3].copy()
        yaw = X.yaw_of(d.qpos[3:7])
        yaw_s = yaw if yaw_s is None else yaw_s + 0.04 * X.wrap(yaw - yaw_s)     # slow, so the camera does not swing with the hips
        lk = p.copy() if lk is None else lk + 0.15 * (p - lk)
        ground = float(app.height(p[0], p[1])) if app is not None else 0.0
        if legs:
            cam.distance, cam.elevation = 2.3, -6
            if app is not None:                                 # take the side where the ground does not hide the feet
                def blocked(sgn):
                    q = p[:2] + sgn * np.array([math.sin(yaw_s), -math.cos(yaw_s)]) * np.linspace(0.25, 2.0, 30)[:, None]
                    return float(app.height(q[:, 0], q[:, 1]).max()) - ground
                b_now, b_other = blocked(side), blocked(-side)
                if b_now > 0.10 and b_other < b_now - 0.05:
                    side = -side
            cam.lookat[:] = [lk[0], lk[1], lk[2] - 0.38]
            cam.azimuth = math.degrees(yaw_s) + (90.0 if side > 0 else -90.0)
        else:
            fixed = os.environ.get("RENDER_FIXED")                 # "x,y,z,azimuth,elevation,distance": a fixed camera (opt-in)
            if fixed:
                fx, fy, fz, faz, fel, fdist = (float(v) for v in fixed.split(","))
                cam.lookat[:] = [fx, fy, fz]
                cam.azimuth, cam.elevation, cam.distance = faz, fel, fdist
            else:
                cam.distance = float(os.environ.get("RENDER_CHASE_DIST", "3.2"))      # wider / higher: shows a staircase's shape
                cam.elevation = float(os.environ.get("RENDER_CHASE_ELEV", "-22"))
                cam.lookat[:] = [lk[0], lk[1], lk[2] - 0.20]
                cam.azimuth = math.degrees(yaw_s) + 35.0             # behind and to the right
        ren.update_scene(d, cam, opt)
        if dressed is not None:                                 # dressed walls: see-through (chase) or not drawn (legs), as the boxes
            for i in range(ren.scene.ngeom):
                gm = ren.scene.geoms[i]
                if gm.objtype == mujoco.mjtObj.mjOBJ_GEOM and gm.objid in walls_all:
                    gm.rgba[3] = 0.0 if legs else 0.30
        if not legs:                                            # walls between the camera and the robot are faded right out
            az, el = math.radians(cam.azimuth), math.radians(cam.elevation)
            eye = np.array(cam.lookat) - cam.distance * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
            body = np.array([p[0], p[1], p[2] - 0.15])
            ab = body - eye
            for i in range(ren.scene.ngeom):
                gm = ren.scene.geoms[i]
                if gm.objtype == mujoco.mjtObj.mjOBJ_GEOM and gm.objid in walls_all:
                    c = np.array(gm.pos)
                    tt = float(np.clip((c - eye) @ ab / (ab @ ab), 0.0, 1.0))
                    reach = float(m.geom_rbound[gm.objid]) if dressed is not None else float(np.linalg.norm(m.geom_size[gm.objid]))
                    if tt < 0.97 and np.linalg.norm(eye + tt * ab - c) < reach + 0.45:
                        gm.rgba[3] = 0.03
        wi = int(z["wp"][k])
        if wi < len(wps):                                       # the target: a floating pole, 0.9 to 1.6 m above the ground
            w = wps[wi]
            g = float(app.height(w["x"], w["y"]))
            _add(ren.scene, mujoco.mjtGeom.mjGEOM_CYLINDER, [0.012, 0.35, 0], [w["x"], w["y"], g + 1.25], MAGENTA)
            _add(ren.scene, mujoco.mjtGeom.mjGEOM_SPHERE, [0.05, 0, 0], [w["x"], w["y"], g + 1.65], MAGENTA)
        live = [e for e in marks if e["t"] - 0.04 <= t[k] <= e["t"] + max(e["dur"], HIT_SHOW)]
        for e in live:                                          # a ball at the contact point, with a thin post above it
            c = HIT_RGBA[e["cls"]] if e["hit"] else np.array([0.8, 0.8, 0.8, 0.7], np.float32)
            _add(ren.scene, mujoco.mjtGeom.mjGEOM_SPHERE, [0.03 if e["hit"] else 0.018, 0, 0], [e["xy"][0], e["xy"][1], e["z"]], c)
            if e["hit"]:
                _add(ren.scene, mujoco.mjtGeom.mjGEOM_CYLINDER, [0.006, 0.20, 0], [e["xy"][0], e["xy"][1], e["z"] + 0.23], c)
        n0 = ren.scene.ngeom
        if overlay is not None:
            overlay.scene(ren.scene, x, d, app, t[k], z, k)
        if legs and dressed is not None:                        # legs close-up: anything raised between the camera and the legs
            az, el = math.radians(cam.azimuth), math.radians(cam.elevation)       # (rails, posts, overlay obstacle blocks) is faded out
            eye = np.array(cam.lookat) - cam.distance * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
            for tgt_ in (np.array([p[0], p[1], ground + 0.15]), np.array([p[0], p[1], ground + 0.55])):
                ab = tgt_ - eye
                for i in range(ren.scene.ngeom):
                    gm = ren.scene.geoms[i]
                    mine = gm.objtype == mujoco.mjtObj.mjOBJ_GEOM and gm.objid in dressed.ids
                    ovl = i >= n0 and int(gm.type) == int(mujoco.mjtGeom.mjGEOM_BOX) and gm.size[2] > 0.05
                    if not (mine or ovl):
                        continue
                    c = np.array(gm.pos)
                    if ovl:                                     # overlay obstacle blocks: every one in front of the legs
                        tt = float((c - eye) @ ab / (ab @ ab))
                        if tt < 0.93 and np.linalg.norm(eye + tt * ab - c) < 0.8:
                            gm.rgba[3] = min(gm.rgba[3], 0.06)
                        continue
                    if c[2] < ground + 0.12 and mine:
                        continue
                    tt = float(np.clip((c - eye) @ ab / (ab @ ab), 0.0, 1.0))
                    reach = float(m.geom_rbound[gm.objid]) if mine else float(np.linalg.norm(gm.size))
                    if tt < 0.9 and np.linalg.norm(eye + tt * ab - c) < reach + 0.10:
                        gm.rgba[3] = min(gm.rgba[3], 0.06)
        im = Image.fromarray(ren.render())
        dr = ImageDraw.Draw(im)
        if minimal:                                             # one short line on a thin strip; no footer, banners or legend
            bar = LB + 6
            dr.rectangle((0, 0, W, bar), fill=(15, 18, 24))
            dr.text((10, 4), label, fill=(240, 240, 240), font=font)
            if overlay is not None:
                overlay.image(im, dr, t[k], bar, H, small)
            wr.append_data(np.asarray(im))
            if k in still_at:
                im.save(out / ("%s_t%06.2f.jpg" % (name if out_path is None else pathlib.Path(out_path).stem, t[k])), quality=88)
            continue
        top =[(l_, font, (240, 240, 240)) for l_ in _wrap(dr, label, font, W - 20)] + [(l_, small, (200, 210, 220)) for l_ in _wrap(dr, label2, small, W - 20)]
        top_h = 6 + sum(LB if f_ is font else LSm for _, f_, _ in top)
        dr.rectangle((0, 0, W, top_h), fill=(15, 18, 24))
        y_ = 4
        for l_, f_, col in top:
            dr.text((10, y_), l_, fill=col, font=f_)
            y_ += LB if f_ is font else LSm
        note = ("legs close-up, half speed; walls not drawn" if legs else "chase view; walls see-through, faded in front of the robot")
        tgt = ("waypoint %d / %d: %s%s" % (wi + 1, len(wps), wps[wi]["name"], " (holding)" if z["hold"][k] else "")) if wi < len(wps) else ("route done" if wps else "no route")
        c = z["cmd"][k]
        bottom = [(l_, (200, 210, 220)) for l_ in _wrap(dr, credit_for(track), small, W - 20)]
        if overlay is not None and getattr(overlay, "legend_text", None):
            bottom = [(l_, (230, 230, 200)) for l_ in _wrap(dr, overlay.legend_text, small, W - 20)] + bottom
        bottom += [(l_, (250, 250, 250)) for l_ in _wrap(dr, "t %.2f s | %s | cmd vx %.2f m/s, yaw %.2f rad/s | magenta pole = target, floating, not a foothold" % (t[k], tgt, c[0], c[2]), small, W - 20)]
        bottom += [(l_, (200, 210, 220)) for l_ in _wrap(dr, "simulator: foot load L %.0f N, R %.0f N | pelvis %.2f m above ground | outcome: %s | side hits %d, stumble penalty %.2f | %s" % (
            z["load"][k][0], z["load"][k][1], z["h"][k], res["reason"], res.get("side_hit_count", 0), res.get("stumble_penalty", 0.0), note), small, W - 20)]
        dr.rectangle((0, H - LSm * len(bottom) - 6, W, H), fill=(15, 18, 24))
        for i_, (l_, col) in enumerate(bottom):
            dr.text((10, H - LSm * (len(bottom) - i_) - 3), l_, fill=col, font=small)
        if overlay is not None:
            overlay.image(im, dr, t[k], top_h, H - LSm * len(bottom) - 6, small)
        hits_live = [e for e in live if e["hit"]]
        if hits_live:
            dr.rectangle((0, top_h + 2, W, top_h + 2 + 24 * len(hits_live[:4]) + 6), fill=(120, 10, 10))
            for i, e in enumerate(hits_live[:4]):
                dr.text((10, top_h + 5 + 24 * i),"SIDE HIT (%s ball): %s %s on a %s side at t %.2f s, peak horizontal %.0f N, impulse %.1f N s, pelvis dv %.2f m/s, tilt +%.1f deg, penalty %.2f" % (
                    HIT_NAME[e["cls"]], e["side"], {"foot_side": "foot"}.get(e["kind"], e["kind"]), e["cls"], e["t"], e["peak_h"], e["impulse"], e["dv"], e["dtilt"], e["penalty"]),
                    fill=(255, 255, 255), font=small)
        wr.append_data(np.asarray(im))
        if k in still_at:
            im.save(out / ("%s_t%06.2f.jpg" % (name if out_path is None else pathlib.Path(out_path).stem, t[k])), quality=88)
    wr.close()
    ren.close()
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--views", nargs="*", default=["chase", "legs"])
    ap.add_argument("--track", default=None)
    ap.add_argument("--t0", type=float, default=None)
    ap.add_argument("--t1", type=float, default=None)
    ap.add_argument("--stills", type=int, default=0)
    a = ap.parse_args()
    import memcap
    memcap.guard(cap_gb=6, need_free_gb=12)
    run = pathlib.Path(a.run)
    s = json.load(open(run / "summary.json"))
    seeds = a.seeds if a.seeds is not None else sorted(set(s.get("renders", {}).values())) or [0]
    for seed in seeds:
        for v in a.views:
            print(render(run, seed, v, a.track or s["adapter"], a.t0, a.t1, a.stills))
