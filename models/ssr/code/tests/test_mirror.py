"""Checks that fail on a wrong mirror: joint signs against X2 kinematics, involutions, encoder equivariance."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np
import torch

from ssr import mirror as M
from ssr import networks as N
from ssr import x2 as X
from ssr.env import DEPTH_N, H, PRIV_DIM, PROP_DIM, AMP_DIM


def test_joint_signs_kinematics(bad=False):
    """Mirroring q must reflect every body across the sagittal plane."""
    x2 = X.build(foot_boxes=False)
    m, d = x2.model, mujoco.MjData(x2.model)
    rng = np.random.default_rng(0)
    idx = x2.idx21
    pairs = [(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i), None) for i in range(1, m.nbody)]
    names = [p[0] for p in pairs]
    sign = X.MIRROR_SIGN.copy()
    if bad:
        sign[1] *= -1
    for _ in range(20):
        q = rng.uniform(-0.25, 0.25, 21)
        d.qpos[:] = 0
        d.qpos[3] = 1.0
        d.qpos[7:][idx] = q
        mujoco.mj_kinematics(m, d)
        p0 = d.xpos.copy()
        d.qpos[7:][idx] = q[X.MIRROR_PERM] * sign
        mujoco.mj_kinematics(m, d)
        for n in names:
            if "wrist" in n or "hand" in n or "head" in n:
                continue
            o = n.replace("left_", "@").replace("right_", "left_").replace("@", "right_")
            a, b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n), mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, o)
            e = np.abs(p0[a] * [1, -1, 1] - d.xpos[b]).max()
            assert e < 2e-3, (n, o, e)


def test_involutions():
    for f, x in ((M.prop, torch.randn(8, PROP_DIM)), (M.priv, torch.randn(8, PRIV_DIM)), (M.amp, torch.randn(8, AMP_DIM)),
                 (M.actor_obs, torch.randn(8, N.ACTOR_DIM)), (M.joints, torch.randn(8, 21)), (M.depth, torch.randn(8, 36, 36))):
        assert (f(f(x)) - x).abs().max() < 1e-5, f.__name__


def test_encoder_equivariance(broken=False):
    e = N.Encoder().eval()
    hist, depth = torch.randn(4, H, PROP_DIM), torch.rand(4, DEPTH_N, DEPTH_N)
    h = torch.randn(4, 256) * 0.2
    with torch.no_grad():
        a = e.step(hist, depth, h, sample=False)
        h_m = N.swap(h)
        b = e.step(M.prop(hist), (depth if broken else M.depth(depth)), h_m, sample=False)
    err = (M.latent(a["z"]) - b["z"]).abs().max().item()
    err_h = (N.swap(a["h"]) - b["h"]).abs().max().item()
    assert max(err, err_h) < 1e-4, (err, err_h)


def test_map_equivariance(broken=False):
    e = N.Encoder(map_memory=True).eval()
    hist, depth, mp = torch.randn(4, H, PROP_DIM), torch.rand(4, DEPTH_N, DEPTH_N), torch.rand(4, 2, 64, 64)
    h = torch.randn(4, 256) * 0.2
    with torch.no_grad():
        a = e.step(hist, depth, h, sample=False, map_img=mp)
        b = e.step(M.prop(hist), M.depth(depth), N.swap(h), sample=False, map_img=(mp if broken else mp.flip(-1)))
    err = (M.latent(a["z"]) - b["z"]).abs().max().item()
    assert err < 1e-4, err


if __name__ == "__main__":
    test_joint_signs_kinematics()
    test_involutions()
    test_encoder_equivariance()
    test_map_equivariance()
    try:
        test_map_equivariance(broken=True)
    except AssertionError:
        print("bad case caught: unflipped map")
    else:
        raise SystemExit("BAD CASE NOT CAUGHT: unflipped map")
    for f, kw, name in ((test_joint_signs_kinematics, dict(bad=True), "wrong joint sign"), (test_encoder_equivariance, dict(broken=True), "unflipped depth")):
        try:
            f(**kw)
        except AssertionError:
            print("bad case caught:", name)
        else:
            raise SystemExit("BAD CASE NOT CAUGHT: " + name)
    print("all mirror checks pass")
