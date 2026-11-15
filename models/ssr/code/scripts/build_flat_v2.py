"""Rebuild the flat motion prior from every selected flat clip (the first build stopped at 400 s of randomly chosen clips), on the node.
Writes data/amp_v2.npz = data/amp.npz with the "flat" set replaced. CPU only. Run from tracks/ssr/code."""
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import joblib, mujoco, numpy as np
import build_amp as B
from ssr import x2 as X
SRC = pathlib.Path("/mnt/filesystem-g4/data/motions/ssr_prior_sources/apparatus_build_train/x2_loco_flat_train.pkl")
x2 = X.build(foot_boxes=False)
m = x2.model
data = mujoco.MjData(m)
ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in X.HAND_BODIES + X.FOOT_BODIES]
d = joblib.load(SRC)
keys = sorted(k for k in d if B.FLAT_KEEP.search(k) and not B.FLAT_DROP.search(k))
fr, starts, n = [], [], 0
for k in keys:
    v = d[k]
    if len(v["dof"]) / float(v["fps"]) < 0.5:
        continue
    f = B.frames(x2, data, ids, *B.resample(v))
    if not np.isfinite(f).all() or np.abs(f[:, 30:51]).max() > 25:
        continue
    fr.append(f); starts.append(n); n += len(f)
z = dict(np.load(ROOT / "data/amp.npz"))
old = z["flat"]
z["flat"], z["flat_starts"] = np.concatenate(fr), np.array(starts)
np.savez(ROOT / "data/amp_v2.npz", **z)
sp = lambda a: np.linalg.norm(a[:, 3:5], axis=1)
print("flat before: %.0f s, %d clips; after: %.0f s, %d clips" % (len(old) / 50, len(np.load(ROOT / "data/amp.npz")["flat_starts"]), n / 50, len(starts)))
for tag, a in (("before", old), ("after", z["flat"])):
    s_ = sp(a)
    print(tag, "seconds by planar speed <0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0, 1.0-1.2, >1.2:", (np.histogram(s_, [0, .2, .4, .6, .8, 1.0, 1.2, 9])[0] / 50).round().tolist())
