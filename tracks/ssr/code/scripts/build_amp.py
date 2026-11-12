"""Build the per-terrain AMP motion priors (paper App. A.8): flat walking from BONES-SEED plus the Pico terrain captures.

The apparatus' own route clips are never used. Output data/amp.npz: for each discriminator a float32 array of 63-D frames
and the start index of every clip. Run: python scripts/build_amp.py
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import mujoco
import numpy as np

from ssr import x2 as X
from ssr.env import AMP_DIM

TRAIN = pathlib.Path("C:/Users/PC/projects/loco-manip-apparatus/build/train")
PARK = pathlib.Path("C:/Users/PC/projects/terrain-park/data/train_sets")
FLAT_KEEP = re.compile(r"walk|turn|jog|run|stride|back", re.I)
FLAT_DROP = re.compile(r"injured|crouch|jump|zombie|drunk|hop|crawl|tiptoe|limp|bump|sneak", re.I)
FPS = 50.0

# discriminator -> [(file, name filter, max seconds)]
FLAT = (TRAIN / "x2_loco_flat_train.pkl", lambda k: FLAT_KEEP.search(k) and not FLAT_DROP.search(k), 150)
BLOCKS = (TRAIN / "x2_loco_blocks_train.pkl", lambda k: FLAT_KEEP.search(k) and not FLAT_DROP.search(k), 150)
SETS = {
    "ramp": [FLAT, (PARK / "acp_tp_ramp_sh2_x30.pkl", None, 400)],
    "stairs": [FLAT, (PARK / "tp_stairs_x10_sh2.pkl", None, 300), (PARK / "tp_stairs_chain_x10.pkl", None, 150), (TRAIN / "gst_v1.pkl", None, 250),
               (TRAIN / "gs2_v1.pkl", None, 200), (TRAIN / "gs2dn_v1.pkl", None, 185)],
    "beam": [FLAT, (PARK / "tp_t4x4u_y20.pkl", None, 217), (PARK / "pico_2026-10-29_2x2_train.pkl", None, 49), BLOCKS],
    "platform": [FLAT, (PARK / "tp_stairs_x10.pkl", None, 45), (PARK / "tp_stairs_x10_sh2.pkl", None, 200), (TRAIN / "gs2_v1.pkl", None, 200),
                 (TRAIN / "x2_chains.pkl", lambda k: "box" in k, 300)],
    "flat": [(TRAIN / "x2_loco_flat_train.pkl", lambda k: FLAT_KEEP.search(k) and not FLAT_DROP.search(k), 400)],
}


def resample(v):
    n0 = len(v["dof"])
    t0 = np.arange(n0) / float(v["fps"])
    t1 = np.arange(0, t0[-1], 1 / FPS)
    i = np.clip(np.searchsorted(t0, t1, side="right") - 1, 0, n0 - 2)
    a = ((t1 - t0[i]) / (t0[i + 1] - t0[i]))[:, None]
    root = v["root_trans_offset"][i] * (1 - a) + v["root_trans_offset"][i + 1] * a
    dof = v["dof"][i] * (1 - a) + v["dof"][i + 1] * a
    q0, q1 = v["root_rot"][i], v["root_rot"][i + 1].copy()
    q1[(q0 * q1).sum(-1) < 0] *= -1
    q = q0 * (1 - a) + q1 * a
    return root, q[:, [3, 0, 1, 2]] / np.linalg.norm(q, axis=-1, keepdims=True), dof


def frames(x2, data, ids, root, quat, dof):
    m, T = x2.model, len(root)
    R = np.zeros((T, 3, 3))
    pts = np.zeros((T, 4, 3))
    for k in range(T):
        data.qpos[:3], data.qpos[3:7], data.qpos[7:] = root[k], quat[k], dof[k]
        mujoco.mj_kinematics(m, data)
        R[k] = data.xmat[1].reshape(3, 3)
        pts[k] = data.xpos[ids]
    v = np.gradient(root, 1 / FPS, axis=0)
    dR = np.einsum("tji,tjk->tik", R, np.gradient(R, 1 / FPS, axis=0))
    w = np.stack((dR[:, 2, 1], dR[:, 0, 2], dR[:, 1, 0]), -1)
    q = dof[:, x2.idx21]
    qd = np.gradient(q, 1 / FPS, axis=0)
    f = np.empty((T, AMP_DIM), np.float32)
    f[:, :3] = -R[:, 2, :]
    f[:, 3:6] = np.einsum("tji,tj->ti", R, v)
    f[:, 6:9] = w
    f[:, 9:30] = q - x2.default_q[x2.idx21]
    f[:, 30:51] = qd
    f[:, 51:63] = np.einsum("tji,tpj->tpi", R, pts - root[:, None]).reshape(T, 12)
    return f[2:-2]


def main():
    x2 = X.build(foot_boxes=False)
    m = x2.model
    data = mujoco.MjData(m)
    ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in X.HAND_BODIES + X.FOOT_BODIES]
    rng = np.random.default_rng(0)
    out, cache = {}, {}
    for name, srcs in SETS.items():
        all_f, starts, n, secs = [], [], 0, {}
        for path, keep, cap in srcs:
            if path not in cache:
                cache[path] = joblib.load(path)
            clips = [k for k in cache[path] if keep is None or keep(k)]
            rng.shuffle(clips)
            got = 0.0
            for k in clips:
                if got >= cap:
                    break
                v = cache[path][k]
                if len(v["dof"]) / float(v["fps"]) < 0.5:
                    continue
                f = frames(x2, data, ids, *resample(v))
                if not np.isfinite(f).all() or np.abs(f[:, 30:51]).max() > 25:
                    continue
                all_f.append(f)
                starts.append(n)
                n += len(f)
                got += len(f) / FPS
            secs[path.stem] = round(got)
        out[name] = np.concatenate(all_f)
        out[name + "_starts"] = np.array(starts)
        print(f"{name:9s} {n / FPS:6.0f} s {len(starts):4d} clips  {secs}", flush=True)
    np.savez(ROOT / "data/amp.npz", **out)


if __name__ == "__main__":
    main()
