"""Where each camera's rays land on flat ground with the robot standing, measured from the feet, plus how much of the image the legs hide,
and a picture of each view. Run: python scripts/camera_coverage.py"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import Image

from ssr import terrain as T
from ssr import x2 as X
from ssr.env import DEPTH_MAX, DEPTH_N, SSREnv

r = T.generate()
r["z"] = r["z"] * 0.0
T.generate = lambda s=0: r
tile = []
for cam in X.CAMERAS:
    env = SSREnv(4, camera=cam, train=False, spawn=torch.tensor([[4.0, 8.0, 0.0]], device="cuda:0").expand(4, -1), raster=r)
    a = torch.zeros(4, 21, device=env.dev)
    for _ in range(8):
        env.step(a)
    d, t, o = env._ray_d[0].cpu().numpy(), env._ray_t[0].cpu().numpy(), env._ray_o[0].cpu().numpy()
    ground = -o[2] / np.minimum(d[:, 2], -1e-6)
    gx = o[0] + ground * d[:, 0] - float(env.fp[0, :, 0].mean())
    gy = o[1] + ground * d[:, 1] - float(env.fp[0, :, 1].mean())
    in_range = (ground * np.abs(d[:, 2]) < 10) & (t < DEPTH_MAX / 0.4 - 1e-3)
    occ = float((t < ground - 0.02).mean())
    # ground visible, measured ahead of the mean foot position
    vis = (t >= ground - 0.02) & (d[:, 2] < 0) & in_range
    print(f"{cam:6s} height {o[2]:.2f} m, ground seen {gx[vis].min():+.2f} to {gx[vis].max():+.2f} m ahead of the feet (image width at nearest row +-{np.abs(gy[vis][gx[vis] < gx[vis].min() + 0.1]).max():.2f} m), limbs cover {100 * occ:.0f} % of the image", flush=True)
    im = Image.fromarray((env.depth[0].cpu().numpy() * 255).astype(np.uint8)).resize((288, 288), Image.NEAREST)
    tile.append(np.asarray(im))
Image.fromarray(np.concatenate(tile, 1)).save(ROOT / "runs/camera_views.png")
