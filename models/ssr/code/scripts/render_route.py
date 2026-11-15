"""Render one SSR route rollout, ACP-style overlays for what SSR actually uses:
  - the policy's depth image (top-right inset, 36 x 36 as the network sees it)
  - the height map the encoder decodes from depth around the robot (tiles coloured by height vs the base)
  - imagined landings: the foothold model's predicted next contact per foot, ring radius = its uncertainty
    and the 22.5 x 10 cm sole patch at the landing, green = supported, red = not (the support check of Eq. 2)
  - actual stance feet (square), the active waypoint (green ring) and the velocity command arrow

python scripts/render_route.py runs/v1/route/best.npz out.mp4 "title" [t0 t1]
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import imageio
import json
import math
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_dilation

from route_eval import load_raster
from ssr import x2 as X
from ssr.env import HB_NX, HB_NY, HF_NX, HF_NY
from ssr.route import IN, MOBILITY

W, H, STEP = 1280, 720, 2
FOOT = (np.array([1.0, 0.5, 0.1, 1.0], np.float32), np.array([0.2, 0.55, 1.0, 1.0], np.float32))        # R orange, L blue, as in the ACP videos
OBST_R, OBST_RES, OBST_CAP = 1.5, 0.06, 0.18


def height_rgba(h, a=0.55):
    if h < -0.03:
        return np.array([0.25, 0.45, 1.0, a], np.float32)
    u = min(max(h / 0.3, 0.0), 1.0)
    return np.array([0.2 + 0.8 * u, 0.85 - 0.35 * u, 0.25 - 0.15 * u, a], np.float32)


def add(scn, kind, size, pos, rgba, mat=None):
    if scn.ngeom >= scn.maxgeom:
        return
    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], kind, np.asarray(size, float), np.asarray(pos, float), (np.eye(3) if mat is None else mat).ravel(), rgba)
    scn.ngeom += 1


def yaw_mat(y):
    c, s = math.cos(y), math.sin(y)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def ground(r, x, y):
    i = np.clip(np.round((x - r["x0"]) / r["res"]).astype(int), 0, r["z"].shape[0] - 1)
    j = np.clip(np.round((y - r["y0"]) / r["res"]).astype(int), 0, r["z"].shape[1] - 1)
    return r["z"][i, j]


SCENE = pathlib.Path("C:/Users/PC/projects/loco-manip-apparatus/sim/scene_x2_env_mobility.xml")
CAND = [(o, e, r) for o in (0.0, 20.0, -20.0, 40.0, -40.0, 60.0, -60.0, 90.0, -90.0, 120.0, -120.0, 150.0, -150.0, 180.0) for e in (35.0, 48.0, 62.0) for r in (2.0, 1.5)]
PREF = np.array([abs(o) / 40 + (e != 35) * 0.4 * abs(e - 35) / 13 + (r != 2.0) * 0.3 for o, e, r in CAND])


def camera_path(m, d, qpos, frames):
    """ACP's camera search: score each candidate view by ray casts from the torso, both feet and the ground ahead to the eye (robot moved
    away for the rays), pick a smooth unblocked sequence with Viterbi, low-pass it. Returns (lookat, azimuth, elevation, distance) per frame."""
    pelvis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    feet = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in X.FOOT_BODIES]
    yaws, look, targets = [], [], []
    lk = None
    for k in frames:
        d.qpos[:] = m.qpos0
        d.qpos[:38] = qpos[k]
        mujoco.mj_kinematics(m, d)
        yaw = math.atan2(d.xmat[pelvis][3], d.xmat[pelvis][0])
        p = d.xpos[pelvis].copy()
        t_ = p + [0, 0, -0.05]
        lk = t_ if lk is None else lk + 0.15 * (t_ - lk)
        yaws.append(yaw)
        look.append(lk.copy())
        targets.append((p + [0, 0, 0.15], d.xpos[feet[0]] + [0, 0, 0.05], d.xpos[feet[1]] + [0, 0, 0.05], p + [0.6 * math.cos(yaw), 0.6 * math.sin(yaw), -0.6]))
    d.qpos[:] = m.qpos0
    d.qpos[2] = -20.0
    mujoco.mj_forward(m, d)
    cost = np.zeros((len(frames), len(CAND)))
    gid = np.zeros(1, np.int32)
    for i in range(len(frames)):
        if i % 3:
            cost[i] = cost[i - 1]
            continue
        for j, (o, e, r) in enumerate(CAND):
            az, el = yaws[i] + math.radians(180 + o), math.radians(-e)
            eye = look[i] + r * np.array([-math.cos(el) * math.cos(az), -math.cos(el) * math.sin(az), math.sin(el)])
            bad = 0.0
            for wgt, t_ in zip((2.0, 1.0, 1.0, 0.5), targets[i]):
                v = eye - t_
                L = np.linalg.norm(v)
                hit = mujoco.mj_ray(m, d, t_, v / L, None, 1, -1, gid)
                bad += wgt * (0 <= hit < L)
            cost[i, j] = 60.0 * bad / 4.5 + PREF[j]
    ca = np.array(CAND, float)
    dO = np.abs(((ca[:, None, 0] - ca[None, :, 0]) + 180) % 360 - 180)
    trans = dO / 25 + np.abs(ca[:, None, 1] - ca[None, :, 1]) / 15 + np.abs(ca[:, None, 2] - ca[None, :, 2]) * 2
    acc, back = cost[0].copy(), np.zeros(cost.shape, int)
    for i in range(1, len(frames)):
        tot = acc[:, None] + trans
        back[i] = tot.argmin(0)
        acc = tot.min(0) + cost[i]
    idx = [int(acc.argmin())]
    for i in range(len(frames) - 1, 0, -1):
        idx.append(back[i, idx[-1]])
    sel = ca[idx[::-1]]
    ker = np.hanning(25)
    ker /= ker.sum()
    sm = lambda a: np.convolve(np.pad(a, 12, mode="edge"), ker, "valid")
    o = sm(np.degrees(np.unwrap(np.radians(sel[:, 0]))))
    el, dist = sm(sel[:, 1]), sm(sel[:, 2])
    ys = sm(np.unwrap(yaws))
    lk_s = np.stack([sm(np.array(look)[:, c]) for c in range(3)], -1)
    return [(lk_s[i], (math.degrees(ys[i]) + o[i] + 180) % 360, -el[i], dist[i]) for i in range(len(frames))]


def main():
    src, out, title = sys.argv[1], sys.argv[2], sys.argv[3]
    z = np.load(src)
    T = int(z["T"])
    t0 = int(float(sys.argv[4]) * 50) if len(sys.argv) > 5 else 0
    t1 = int(float(sys.argv[5]) * 50) if len(sys.argv) > 5 else T
    t1 = min(t1, T)
    raster = load_raster(ROOT / "assets/route/cad_20mm_walk.npz")
    # the real apparatus scene with its colours and textures (the same one the ACP videos use); robot in qpos[:38]
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    m.vis.global_.offwidth, m.vis.global_.offheight = W, H
    d = mujoco.MjData(m)
    ren = mujoco.Renderer(m, H, W, max_geom=9000)
    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    frames = list(range(t0, t1, STEP))
    path = camera_path(m, d, z["qpos"], frames)
    vis = load_raster(ROOT / "assets/route/cad_20mm.npz")
    obst = (vis["z"] - raster["z"]) > 0.25
    near = binary_dilation(obst, iterations=6) & ~obst
    wps = [(x * IN, y * IN) for _, x, y, _, _, _ in MOBILITY]
    names = [n for n, *_ in MOBILITY]
    font = ImageFont.truetype("arial.ttf", 20)
    small = ImageFont.truetype("arial.ttf", 15)
    tiny = ImageFont.truetype("arial.ttf", 12)
    writer = imageio.get_writer(out, fps=25, codec="libx264", quality=None, ffmpeg_params=["-crf", "22", "-preset", "medium", "-pix_fmt", "yuv420p"])
    sx, sy = X.SOLE_HALF
    gx = np.linspace(-sx, sx, HF_NX)
    gy = np.linspace(-sy, sy, HF_NY)
    for n_frame, k in enumerate(frames):
        q = z["qpos"][k]
        d.qpos[:] = m.qpos0
        d.qpos[:38] = q
        mujoco.mj_forward(m, d)
        p, yaw = q[:3], math.atan2(d.xmat[1][3], d.xmat[1][0])
        cam.lookat[:], cam.azimuth, cam.elevation, cam.distance = path[n_frame]
        ren.update_scene(d, cam, opt)
        scn = ren.scene
        contact, sole = z["contact"][k], z["sole"][k]
        planted = [sole[f, 2] for f in (0, 1) if contact[f]]
        ref_z = float(min(planted)) if planted else float(ground(raster, np.array(p[0]), np.array(p[1])))
        cy, sy = math.cos(yaw), math.sin(yaw)
        Rz = yaw_mat(yaw)
        # side obstacles: low translucent blocks (red), their margin (magenta)
        g = np.arange(-OBST_R, OBST_R + 1e-9, OBST_RES)
        gxx, gyy = np.meshgrid(p[0] + g, p[1] + g, indexing="ij")
        keep = np.hypot(gxx - p[0], gyy - p[1]) <= OBST_R
        ii = np.clip(np.round((gxx[keep] - raster["x0"]) / raster["res"]).astype(int), 0, obst.shape[0] - 1)
        jj = np.clip(np.round((gyy[keep] - raster["y0"]) / raster["res"]).astype(int), 0, obst.shape[1] - 1)
        for x_, y_, ob, ne in zip(gxx[keep], gyy[keep], obst[ii, jj], near[ii, jj]):
            if ob:
                add(scn, mujoco.mjtGeom.mjGEOM_BOX, [OBST_RES * 0.45] * 2 + [OBST_CAP / 2], [x_, y_, ref_z + OBST_CAP / 2], np.array([0.95, 0.1, 0.1, 0.45], np.float32))
            elif ne:
                add(scn, mujoco.mjtGeom.mjGEOM_BOX, [OBST_RES * 0.45] * 2 + [0.003], [x_, y_, ref_z + 0.008], np.array([0.9, 0.2, 0.9, 0.35], np.float32))
        # the height map around the robot, 1.6 x 1.0 m at 5 cm, coloured by height against the support foot
        for u in np.arange(-0.4, 1.2 + 1e-9, 0.05):
            for v in np.arange(-0.5, 0.5 + 1e-9, 0.05):
                x_, y_ = p[0] + cy * u - sy * v, p[1] + sy * u + cy * v
                i_, j_ = int(round((x_ - raster["x0"]) / raster["res"])), int(round((y_ - raster["y0"]) / raster["res"]))
                if 0 <= i_ < obst.shape[0] and 0 <= j_ < obst.shape[1] and not obst[i_, j_] and not near[i_, j_]:
                    h_ = float(raster["z"][i_, j_])
                    add(scn, mujoco.mjtGeom.mjGEOM_BOX, [0.0225, 0.0225, 0.003], [x_, y_, h_ + 0.006], height_rgba(h_ - ref_z), Rz)
        # the map the encoder decodes from depth: orange squares at the decoded height of its 17 x 10 cells
        pred = z["hb_pred"][k].reshape(HB_NX, HB_NY)
        square = np.array([1.0, 0.55, 0.05, 1.0], np.float32)
        for i in range(HB_NX):
            for j in range(HB_NY):
                u, v = -0.6 + 0.1 * i, 0.1 * (j - (HB_NY - 1) / 2)
                add(scn, mujoco.mjtGeom.mjGEOM_BOX, [0.02, 0.02, 0.004], [p[0] + cy * u - sy * v, p[1] + sy * u + cy * v, float(p[2] - pred[i, j]) + 0.014], square, Rz)
        # imagined landing per swing foot (ring radius from the model's uncertainty), sole support there, stance squares
        fm, fs = z["fm"][k], z["fs"][k]
        for i in (0, 1):
            if contact[i]:
                add(scn, mujoco.mjtGeom.mjGEOM_BOX, [0.025, 0.025, 0.012], sole[i] + [0, 0, 0.03], FOOT[i] * np.array([0.5, 0.5, 0.5, 1.0], np.float32))
                continue
            lx, ly = p[0] + cy * fm[i, 0] - sy * fm[i, 1], p[1] + sy * fm[i, 0] + cy * fm[i, 1]
            lz = float(ground(raster, np.array(lx), np.array(ly)))
            rad = max(0.07, float(fs[i]))
            for a_ in np.linspace(0, 2 * np.pi, 14, endpoint=False):
                add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.012, 0, 0], [lx + rad * math.cos(a_), ly + rad * math.sin(a_), lz + 0.02], FOOT[i])
            hs = []
            for a_ in gx:
                for b_ in gy:
                    cx, cyw = lx + cy * (a_ + X.SOLE_OFFSET_X) - sy * b_, ly + sy * (a_ + X.SOLE_OFFSET_X) + cy * b_
                    hs.append((cx, cyw, float(ground(raster, np.array(cx), np.array(cyw)))))
            top = max(h[2] for h in hs)
            for h in hs:
                col = np.array([0.1, 1.0, 0.25, 0.95] if top - h[2] < 0.03 else [1.0, 0.1, 0.1, 0.95], np.float32)
                add(scn, mujoco.mjtGeom.mjGEOM_BOX, [0.0085, 0.0085, 0.003], [h[0], h[1], h[2] + 0.016], col, Rz)
        # active waypoint
        wi = int(min(z["wi"][k], len(wps) - 1))
        wx, wy = wps[wi]
        gw = float(ground(raster, np.array(wx), np.array(wy)))
        for a_ in np.linspace(0, 2 * np.pi, 28, endpoint=False):
            add(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.018, 0, 0], [wx + 0.3 * math.cos(a_), wy + 0.3 * math.sin(a_), gw + 0.02], np.array([0.1, 0.9, 0.3, 1.0], np.float32))
        im = Image.fromarray(ren.render().copy())
        # depth inset, as the network sees it
        dep = (z["depth"][k] * 255).astype(np.uint8)
        ins = Image.fromarray(dep).resize((180, 180), Image.NEAREST)
        im.paste(ins.convert("RGB"), (W - 200, 64))
        dr = ImageDraw.Draw(im)
        dr.text((W - 200, 247), "policy depth, 36 x 36", font=small, fill=(255, 255, 255))
        dr.rectangle([0, 0, W, 54], fill=(17, 20, 24))
        dr.text((12, 4), title, fill=(255, 255, 255), font=font)
        c = z["cmd"][k]
        status = "completed" if bool(z["finished"]) else "reached %d / %d" % (int(z["reached"]), len(wps))
        dr.text((12, 31), "%.1f / %.1f s   next: %s (%d / %d)   command vx %+.2f vy %+.2f wz %+.2f   physics: %s" % (k * 0.02, T * 0.02, names[wi], wi + 1, len(wps), c[0], c[1], c[2], status), fill=(170, 190, 200), font=small)
        y0 = H - 30
        dr.rectangle([0, y0, W, H], fill=(17, 20, 24))
        x = 10
        for col, txt in (((40, 200, 70), "height map"), ((255, 140, 13), "map decoded from depth"), ((240, 30, 30), "side obstacle"), ((225, 50, 225), "margin"), ((51, 140, 255), "left foot"), ((255, 128, 25), "right foot")):
            dr.rectangle([x, y0 + 9, x + 12, y0 + 21], fill=col, outline=(200, 200, 200))
            dr.text((x + 17, y0 + 7), txt, fill=(220, 220, 220), font=tiny)
            x += 26 + int(dr.textlength(txt, font=tiny)) + 10
        dr.text((x, y0 + 7), "ring = imagined landing   green / red = sole support   square = stance   green circle = waypoint", fill=(220, 220, 220), font=tiny)
        writer.append_data(np.asarray(im))
    writer.close()
    print("wrote", out)


if __name__ == "__main__":
    main()
