"""Read back what the simulator applies (per-joint kp, kd, armature, effort limit, action scale, default pose) from the live
MuJoCo Warp model, and the ground footprint and sample depth frames of each camera option.

X2_PLANT=mc_20261024 python scripts/report_model.py ../eval/model_mc_20261024.json [../renders/camera_check]
"""
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np
import torch

from ssr import terrain as T
from ssr import x2 as X
from ssr.env import SSREnv

out = {"plant_env": os.environ.get("X2_PLANT", "(default file)")}
rdir = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else None
if rdir:
    rdir.mkdir(parents=True, exist_ok=True)
SX, SY = 4.0, 8.0                                   # spawn; the soles' front edge is about 0.15 m ahead of it
TOE0 = 0.038 + X.SOLE_HALF[0]


def raster(scene):
    r = T.generate()
    z = r["z"] * 0.0
    res, x0 = r["res"], r["x0"]
    ix = lambda x: int(round((SX + TOE0 + x - x0) / res))
    if scene.startswith("beam"):                    # a 4x4 (9 cm) across the path, its near face this far from the toes
        d = float(scene[4:])
        z[ix(d):ix(d + 0.09), :] = 0.09
    if scene == "stairs":                           # apparatus-like flight: 18.6 cm risers, 22 cm treads, foot 0.3 m ahead
        for k in range(5):
            z[ix(0.3 + 0.22 * k):, :] = 0.186 * (k + 1)
    r["z"] = z
    r["vision"] = dict(z=z.copy(), x0=r["vision"]["x0"], y0=r["vision"]["y0"], res=r["vision"]["res"])
    return r


rows = None
DEV = os.environ.get("REPORT_DEVICE", "cuda:0")
for cam in (os.environ.get("REPORT_CAMS") or "head,chest,torso60,chest_rs").split(","):
    for scene in (("flat",) if os.environ.get("REPORT_QUICK") else ("flat", "beam1.5", "beam0.3", "stairs")):
        env = SSREnv(2, camera=cam, train=False, device=DEV, spawn=torch.tensor([[SX, SY, 0.0]], device=DEV).expand(2, -1), raster=raster(scene), tiles=False)
        m = env.x2.model
        if rows is None:
            rows = []
            kp, kd = env.gain[0, :, 0].cpu().numpy(), (-env.bias[0, :, 2]).cpu().numpy()
            for i, n in enumerate(env.x2.joint_names):
                j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
                rows.append(dict(joint=n, policy=bool(n in X.NAMES21), kp=round(float(kp[i]), 2), kd=round(float(kd[i]), 3), armature=round(float(m.dof_armature[m.jnt_dofadr[j]]), 6),
                                 effort=round(float(m.actuator_forcerange[i, 1]), 1), action_scale=round(float(env.x2.action_scale[i]), 4), default=round(float(env.x2.default_q[i]), 3),
                                 lo=round(float(m.jnt_range[j, 0]), 3), hi=round(float(m.jnt_range[j, 1]), 3)))
            out["joints"] = rows
            out["sim"] = dict(timestep=m.opt.timestep, decimation=X.DECIMATION, total_mass=round(float(m.body_mass.sum()), 2))
        # no physics steps: the pose is the reset pose (base upright, default joints), so the camera geometry is not mixed with how the robot settles
        toe = float(env.sole[0, :, 0].mean()) + X.SOLE_HALF[0]                  # front edge of the soles at standing

        def measure(tag):
            d, t, o = env._ray_d[0].cpu().numpy(), env._ray_t[0].cpu().numpy(), env._ray_o[0].cpu().numpy()
            if scene == "flat":
                ground = -o[2] / np.minimum(d[:, 2], -1e-6)
                gx = o[0] + ground * d[:, 0] - toe
                gy = o[1] + ground * d[:, 1] - float(env.qpos[0, 1])
                seen = (t >= ground - 0.02) & (d[:, 2] < 0)
                vis = seen & (ground * env.rays[:, 0].cpu().numpy() < env.depth_max) & (ground < env.depth_max / 0.4 - 1e-3)
                near = np.abs(gx - 0.5) < 0.06
                out["camera_" + cam + "_" + tag] = dict(
                    body=X.CAMERAS[cam]["body"], position_in_body_m=list(X.CAMERAS[cam]["pos"]), pitch_down_deg=round(float(np.degrees(X.CAMERAS[cam]["pitch"])), 1),
                    height_above_ground_m=round(float(o[2]), 3), camera_ahead_of_toes_m=round(float(o[0] - toe), 3), square_fov_deg=round(float(np.degrees(env.cam_fov0)), 1), pixels="36 x 36",
                    depth_range_m=[0.1, env.depth_max],
                    nearest_ground_ahead_of_toes_m=round(float(gx[vis].min()), 3), farthest_ground_ahead_of_toes_m=round(float(gx[vis].max()), 3),
                    width_at_0p5m_ahead_m=round(float(2 * np.abs(gy[vis & near]).max()) if bool((vis & near).any()) else 0.0, 3),
                    legs_cover_share=round(float((t < ground - 0.02).mean()), 3))
            if rdir:
                from PIL import Image
                im = (env.depth[0].cpu().numpy() * 255).astype(np.uint8)
                Image.fromarray(im).resize((360, 360), Image.NEAREST).save(rdir / f"{cam}_{scene}_{tag}.png")
                if scene == "flat" and tag == "standing":             # landmarks: the toe line and a line 1.0 m ahead of it, drawn on the frame
                    from PIL import ImageDraw
                    big = Image.fromarray(im).resize((360, 360), Image.NEAREST).convert("RGB")
                    dr = ImageDraw.Draw(big)
                    marks = {}
                    for dist, col in ((0.0, (255, 60, 60)), (0.55, (80, 140, 255)), (1.0, (60, 255, 60))):
                        gx_c = (o[0] + (-o[2] / np.minimum(d[:, 2], -1e-6)) * d[:, 0] - toe).reshape(36, 36)[:, 17:19].mean(1)   # ground distance of each image row, centre columns, from the real rays
                        rc = (np.arange(36) + 0.5) / 36
                        row = float(np.interp(dist, gx_c[::-1], rc[::-1])) if gx_c.min() <= dist else 1.0 + (gx_c.min() - dist)   # past the bottom edge: > 1
                        marks[dist] = round(float(row), 3)
                        if 0 <= row <= 1:
                            dr.line([(0, row * 360), (360, row * 360)], fill=col, width=1)
                    big.save(rdir / f"{cam}_flat_standing_marked_toe_red_0p55m_blue_1m_green.png")
                    dc = d.reshape(36, 36, 3)[17:19, 17:19].mean((0, 1))
                    out["camera_" + cam + "_axis_below_horizontal_deg"] = round(float(np.degrees(np.arcsin(-dc[2] / np.linalg.norm(dc)))), 2)
                    out["camera_" + cam + "_landmark_rows_from_top"] = dict(toe_line=marks[0.0], line_0p55m_ahead=marks[0.55], line_1m_ahead=marks[1.0])
                np.save(rdir / f"{cam}_{scene}_{tag}_36x36.npy", env.depth[0].cpu().numpy())

        measure("standing")
        if scene == "flat":
            names = env.x2.joint_names
            for n_, v_ in (("right_hip_pitch_joint", -0.75), ("right_knee_joint", 0.95), ("right_ankle_pitch_joint", -0.25), ("left_hip_pitch_joint", 0.1), ("left_knee_joint", 0.35)):
                env.qpos[:, 7 + names.index(n_)] = v_                           # a stride pose: right leg swung forward, set kinematically
            env.refresh()
            env._render()
            measure("midstride")
        del env
pathlib.Path(sys.argv[1]).parent.mkdir(parents=True, exist_ok=True)
pathlib.Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
for c in out:
    if c.startswith("camera_"):
        print(c, out[c])
print("joint kp kd armature effort scale default")
for j in out["joints"]:
    if j["policy"] or "waist" in j["joint"]:
        print(j["joint"], j["kp"], j["kd"], j["armature"], j["effort"], j["action_scale"], j["default"])
