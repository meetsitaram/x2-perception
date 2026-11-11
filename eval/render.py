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
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import mujoco
import numpy as np

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


def render(run: pathlib.Path, seed: int, view: str, track: str, t0=None, t1=None, stills=0, fps=25, size=(1280, 720)):
    import imageio
    from PIL import Image, ImageDraw
    s = json.load(open(run / "summary.json"))
    res = {r["seed"]: r for r in json.load(open(run / "results.json"))}[seed]
    z = np.load(run / "traj" / ("%d.npz" % seed))
    app = None if s["apparatus"] == "flat" else Apparatus(s["apparatus"])
    x = X.build(app, wall_alpha=0.30, render_size=size)       # same model; gains do not matter for a replay
    m = x.model
    d = mujoco.MjData(m)
    W, H = size
    ren = mujoco.Renderer(m, H, W, max_geom=20000)
    cam, opt = mujoco.MjvCamera(), mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[1] = opt.geomgroup[2] = 1                    # the robot's visual meshes
    opt.geomgroup[X.GROUP_TERRAIN] = 1
    opt.geomgroup[X.GROUP_WALL] = 1 if view == "chase" else 0
    legs = view == "legs"
    t = z["t"]
    k0 = int(np.searchsorted(t, t0)) if t0 is not None else 0
    k1 = int(np.searchsorted(t, t1)) if t1 is not None else len(t)
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.02
    every = max(1, int(round(1.0 / (fps * dt))))
    slow = 0.5 if legs else 1.0
    name = "%s_seed%d_%s" % (view, seed, "t%.0f-%.0f" % (t[k0], t[min(k1, len(t)) - 1]) if (t0 is not None or t1 is not None) else "full")
    out = run / "renders"
    out.mkdir(exist_ok=True)
    path = out / (name + ".mp4")
    wr = imageio.get_writer(str(path), fps=slow / (every * dt), codec="libx264", ffmpeg_params=["-crf", "22", "-preset", "medium"])
    font, small = _font(19), _font(16)
    label = "%s | track: %s | apparatus: %s | seed %d | %s" % (s["simulator"], track, s["apparatus"], seed, s.get("protocol", ""))
    label2 = ("%s | " % s["label"] if s.get("label") else "") + "checkpoint: %s" % s["checkpoint"]
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
            cam.distance, cam.elevation = 3.2, -22
            cam.lookat[:] = [lk[0], lk[1], lk[2] - 0.20]
            cam.azimuth = math.degrees(yaw_s) + 35.0             # behind and to the right
        ren.update_scene(d, cam, opt)
        if not legs:                                            # walls between the camera and the robot are faded right out
            az, el = math.radians(cam.azimuth), math.radians(cam.elevation)
            eye = np.array(cam.lookat) - cam.distance * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
            body = np.array([p[0], p[1], p[2] - 0.15])
            ab = body - eye
            for i in range(ren.scene.ngeom):
                gm = ren.scene.geoms[i]
                if gm.objtype == mujoco.mjtObj.mjOBJ_GEOM and gm.objid in x.wall_geoms:
                    c = np.array(gm.pos)
                    tt = float(np.clip((c - eye) @ ab / (ab @ ab), 0.0, 1.0))
                    reach = float(np.linalg.norm(m.geom_size[gm.objid]))
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
        im = Image.fromarray(ren.render())
        dr = ImageDraw.Draw(im)
        dr.rectangle((0, 0, W, 52), fill=(15, 18, 24))
        dr.text((10, 4), label, fill=(240, 240, 240), font=font)
        dr.text((10, 29), label2, fill=(200, 210, 220), font=small)
        dr.rectangle((0, H - 52, W, H), fill=(15, 18, 24))
        note = ("legs close-up, half speed; walls not drawn" if legs else "chase view; walls see-through, faded in front of the robot")
        tgt = ("waypoint %d / %d: %s%s" % (wi + 1, len(wps), wps[wi]["name"], " (holding)" if z["hold"][k] else "")) if wi < len(wps) else ("route done" if wps else "no route")
        c = z["cmd"][k]
        dr.text((10, H - 49), "t %.2f s | %s | cmd vx %.2f m/s, yaw %.2f rad/s | magenta pole = target, floating, not a foothold" % (t[k], tgt, c[0], c[2]),
                fill=(250, 250, 250), font=small)
        dr.text((10, H - 26), "simulator: foot load L %.0f N, R %.0f N | pelvis %.2f m above ground | outcome: %s | side hits %d, stumble penalty %.2f | %s" % (
            z["load"][k][0], z["load"][k][1], z["h"][k], res["reason"], res.get("side_hit_count", 0), res.get("stumble_penalty", 0.0), note), fill=(200, 210, 220), font=small)
        hits_live = [e for e in live if e["hit"]]
        if hits_live:
            dr.rectangle((0, 54, W, 54 + 24 * len(hits_live[:4]) + 6), fill=(120, 10, 10))
            for i, e in enumerate(hits_live[:4]):
                dr.text((10, 57 + 24 * i), "SIDE HIT (%s ball): %s %s on a %s side at t %.2f s, peak horizontal %.0f N, impulse %.1f N s, pelvis dv %.2f m/s, tilt +%.1f deg, penalty %.2f" % (
                    HIT_NAME[e["cls"]], e["side"], {"foot_side": "foot"}.get(e["kind"], e["kind"]), e["cls"], e["t"], e["peak_h"], e["impulse"], e["dv"], e["dtilt"], e["penalty"]),
                    fill=(255, 255, 255), font=small)
        wr.append_data(np.asarray(im))
        if k in still_at:
            im.save(out / ("%s_t%06.2f.jpg" % (name, t[k])), quality=88)
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
