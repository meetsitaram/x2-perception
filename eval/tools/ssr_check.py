"""Check the SSR adapter against the track's obs_check.npz (eval monitor, 2026-11-12).

    python eval/tools/ssr_check.py HANDOVER_DIR

For each of the 50 recorded steps: put the recorded state into the eval's MuJoCo model (flat plane), build the newest
history frame and the depth image with the adapter, and compare with the recording; run the ONNX on the recorded inputs
(hist, depth, gru_in) and compare the action and GRU output; map the action to the 31 joint targets and compare.
"""
import os
import pathlib
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import x2 as X  # noqa: E402
from adapters.base import View  # noqa: E402
from adapters.ssr import SSRHandover  # noqa: E402

d = sys.argv[1]
z = np.load(os.path.join(d, "obs_check.npz"))
ad = SSRHandover(d)
names = X.joint_names_of(X.robot_dir() / "mjcf" / "x2_ultra.xml")
x = X.build(None, ad.plant(names))
data = mujoco.MjData(x.model)
view = View(x, data, None)
ad.reset(view, None)
src = [str(n) for n in z["joint_names_31"]]
perm = np.array([src.index(n) for n in names])
XY0 = np.r_[z["qpos"][0, :2].astype(float)]
res = {k: [] for k in ("frame", "depth", "depth_noocc", "ray_t", "ray_o", "ray_d", "action", "gru", "target", "hist_rows")}
for k in range(len(z["t"])):
    qp, qv = z["qpos"][k].astype(float), z["qvel"][k].astype(float)
    data.qpos[:7], data.qvel[:6] = qp[:7], qv[:6]
    data.qpos[:2] -= XY0                     # the eval plane is 40 m across; the recording stands at x 33.5 m on flat ground (z 0)
    data.qpos[x.qadr], data.qvel[x.dadr] = qp[7:][perm], qv[6:][perm]
    mujoco.mj_forward(x.model, data)
    ad.a_prev = z["action_onnx"][k - 1].astype(float) if k > 0 else z["hist"][0, -1, 51:72].astype(float)   # step 0: the action before the recording (not recorded) taken from its own frame
    f = ad.frame(view, z["cmd"][k])
    res["frame"].append(np.abs(f - z["hist"][k, -1]).max())
    if k > 0:
        res["hist_rows"].append(np.abs(z["hist"][k, :-1] - z["hist"][k - 1, 1:]).max())
    o, dw = ad.camera(view)
    res["ray_o"].append(np.abs(o + np.r_[XY0, 0.0] - z["ray_origin_world"][k]).max())
    res["ray_d"].append(np.abs(dw - z["ray_dir_world"][k]).max())
    t = ad.ray_lengths(view)
    res["ray_t"].append(np.abs(t - z["ray_t"][k]).max())
    res["depth"].append(np.abs(ad.depth(view) - z["depth"][k]).max())
    ad.occlude = False
    res["depth_noocc"].append(np.abs(ad.depth(view) - z["depth"][k]).max())
    ad.occlude = True
    a, g = ad.step(z["hist"][k], z["depth"][k], z["gru_in"][k][None].astype(np.float32))
    res["action"].append(np.abs(a - z["action_onnx"][k]).max())
    res["gru"].append(np.abs(g[0] - z["gru_out"][k]).max())
    q21 = np.clip(ad.q0 + ad.scale * np.clip(a, -8, 8), ad.q_lo, ad.q_hi)
    q = ad.q0_31.copy()
    q[ad.idx21] = q21
    res["target"].append(np.abs(q - z["joint_target_31"][k][perm]).max())
    if k < len(z["t"]) - 1:
        res.setdefault("gru_chain", []).append(np.abs(z["gru_out"][k] - z["gru_in"][k + 1]).max())
for k, v in res.items():
    v = np.array(v)
    print("%-12s max %.3g  median %.3g" % (k, v.max(), np.median(v)))
fr = np.array(res["frame"]); dp = np.array(res["depth"])
lim = dict(frame=1e-4, depth=0.02, action=1e-4, target=1e-4)
ok = all(np.array(res[k]).max() < v for k, v in lim.items())
print("PASS" if ok else "FAIL", lim)
