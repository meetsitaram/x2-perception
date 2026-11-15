"""Offline test of the deploy path: CPU MuJoCo at 1 kHz, ONNX via deploy/ssr_core.py, the camera a real ray cast at full 1280 x 720 geometry
downsampled through the same preprocess as the robot, IMU on torso_link.

python deploy/sim2sim.py --onnx deploy/ssr_x2.onnx --scene stairs --seconds 12
"""
import argparse
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np

from ssr_core import CROP, Core, preprocess_depth
from ssr import terrain as T
from ssr import x2 as X


def scene(kind):
    n = int(16 / T.RES)
    z = np.zeros((n, int(4 / T.RES)), np.float32)
    i = lambda v: int(v / T.RES)
    if kind == "stairs":
        x, h = 2.5, 0.0
        for _ in range(4):
            h += 0.15
            z[i(x):] = h
            x += 0.30
        x += 1.2
        for _ in range(4):
            h -= 0.15
            z[i(x):] = h
            x += 0.30
    elif kind == "beam":
        z[i(3.0):i(3.09)] = 0.09
    elif kind == "platform":
        z[i(3.0):i(4.5)] = 0.25
    return dict(z=z, x0=-1.0, y0=-2.0, res=T.RES, base=0.1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--scene", default="flat", choices=("flat", "stairs", "beam", "platform"))
    ap.add_argument("--seconds", type=float, default=12)
    ap.add_argument("--vx", type=float, default=0.5)
    a = ap.parse_args()
    x2 = X.build(scene(a.scene))
    m = x2.model
    m.opt.timestep = 0.001
    d = mujoco.MjData(m)
    d.qpos[:3] = [0.0, 0.0, 0.665]
    d.qpos[3] = 1.0
    d.qpos[7:] = x2.default_q
    d.ctrl[:] = x2.default_q
    core = Core(a.onnx, x2)
    torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    head = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "head_pitch_link")
    mujoco.mj_forward(m, d)
    # coarse pixel grid for the 720 px crop: one ray per 10 px block is enough to feed a 20 px average
    half = (CROP / 2) / X.CAM_FX_720
    n = 72
    uv = (np.arange(n) + 0.5) / n * 2 - 1
    Ry = np.array([[math.cos(X.CAM_PITCH_DOWN), 0, math.sin(X.CAM_PITCH_DOWN)], [0, 1, 0], [-math.sin(X.CAM_PITCH_DOWN), 0, math.cos(X.CAM_PITCH_DOWN)]])
    gid = np.array([-1], np.int32)
    target = x2.default_q.copy()
    fell = None
    for k in range(int(a.seconds * 50)):
        Rh, oh = d.xmat[head].reshape(3, 3), d.xpos[head]
        o = oh + Rh @ np.array(X.CAM_POS)
        img = np.zeros((n, n), np.float32)
        for r, v in enumerate(uv):
            for c, u in enumerate(uv):
                ray = np.array([1.0, -u * half, -v * half])
                dcam = Rh @ Ry @ (ray / np.linalg.norm(ray))
                t = mujoco.mj_ray(m, d, o, dcam, np.array([1, 0, 0, 0, 0, 0], np.uint8), 1, -1, gid)
                img[r, c] = (t * (ray[0] / np.linalg.norm(ray))) if t > 0 else 0.0
        mm = np.repeat(np.repeat((img * 1000).astype(np.uint16), 10, 0), 10, 1)
        full = np.zeros((720, 1280), np.uint16)
        full[:, 280:1000] = mm
        dep = preprocess_depth(full, flip_180=False)
        qp, qv = d.qpos[7:].copy(), d.qvel[6:].copy()
        qt = np.zeros(4)
        mujoco.mju_mat2Quat(qt, d.xmat[torso])
        wb = d.xmat[torso].reshape(3, 3).T @ d.cvel[torso][:3]
        target = core.step(qp, qv, qt, wb, dep, np.array([a.vx, 0.0, 0.0], np.float32))
        for _ in range(20):
            d.ctrl[:] = target
            mujoco.mj_step(m, d)
        if d.qpos[2] < 0.3 or d.xmat[1][8] < 0.5:
            fell = k * 0.02
            break
    print(f"{a.scene}: x travelled {d.qpos[0]:.2f} m, {'fell at %.1f s' % fell if fell is not None else 'upright'}, depth mean {dep.mean():.2f}")


if __name__ == "__main__":
    main()
