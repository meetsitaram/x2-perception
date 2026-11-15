"""amp_v4 (SSR 2026-11-12 20:50): data/amp_v2.npz with the ramp, stairs and beam sets rebuilt with the user's new Pico takes.
- stairs: + tp_stairs_all_v1_1 (44 corrected stair takes incl. turning) and its mirror tp_stairs_all_v1_1_mir
- beam:   2x2 set -> pico_2026-10-29_2x2_train_v1_1 (drops the take with hip yaw stuck at its limit) + its mirror; + tp_t4x4u_y20_mir
- ramp:   + tp_ramp_base_x30_mir
Mirrors are used as motion only (the discriminator sees proprio frames; props / height maps are not used by SSR's prior).
flat and platform sets: unchanged from amp_v2. CPU only. Run from tracks/ssr/code."""
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import joblib, mujoco, numpy as np
import build_amp as B
from ssr import x2 as X
TR = pathlib.Path("/mnt/filesystem-g4/data/motions/ssr_prior_sources/apparatus_build_train")
PK = pathlib.Path("/mnt/filesystem-g4/data/motions/ssr_prior_sources/terrain_park_train_sets")
FLAT = (TR / "x2_loco_flat_train.pkl", lambda k: B.FLAT_KEEP.search(k) and not B.FLAT_DROP.search(k), 150)
BLOCKS = (TR / "x2_loco_blocks_train.pkl", lambda k: B.FLAT_KEEP.search(k) and not B.FLAT_DROP.search(k), 150)
SETS = {
    "ramp": [FLAT, (PK / "acp_tp_ramp_sh2_x30.pkl", None, 400), (PK / "tp_ramp_base_x30_mir.pkl", None, 400)],
    "stairs": [FLAT, (PK / "tp_stairs_x10_sh2.pkl", None, 300), (PK / "tp_stairs_chain_x10.pkl", None, 150), (TR / "gst_v1.pkl", None, 250),
               (TR / "gs2_v1.pkl", None, 200), (TR / "gs2dn_v1.pkl", None, 185),
               (PK / "tp_stairs_all_v1_1.pkl", None, 1e9), (PK / "tp_stairs_all_v1_1_mir.pkl", None, 1e9)],
    "beam": [FLAT, (PK / "tp_t4x4u_y20.pkl", None, 217), (PK / "tp_t4x4u_y20_mir.pkl", None, 217),
             (PK / "pico_2026-10-29_2x2_train_v1_1.pkl", None, 1e9), (PK / "pico_2026-10-29_2x2_train_v1_1_mir.pkl", None, 1e9), BLOCKS],
}
x2 = X.build(foot_boxes=False)
m = x2.model
data = mujoco.MjData(m)
ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in X.HAND_BODIES + X.FOOT_BODIES]
rng = np.random.default_rng(0)
z = dict(np.load(ROOT / "data/amp_v2.npz"))
cache = {}
for name, srcs in SETS.items():
    all_f, starts, n, secs, bad = [], [], 0, {}, 0
    for path, keep, cap in srcs:
        if path not in cache:
            cache[path] = joblib.load(path)
        clips = sorted(k for k in cache[path] if keep is None or keep(k))
        assert not any("211326_0_2x2" in k for k in clips), "bad 2x2 take present"
        rng.shuffle(clips)
        got = 0.0
        for k in clips:
            if got >= cap:
                break
            v = cache[path][k]
            if len(v["dof"]) / float(v["fps"]) < 0.5:
                continue
            f = B.frames(x2, data, ids, *B.resample(v))
            if not np.isfinite(f).all() or np.abs(f[:, 30:51]).max() > 25:
                bad += 1
                continue
            all_f.append(f); starts.append(n); n += len(f); got += len(f) / B.FPS
        secs[path.stem] = round(got)
    old = len(z[name]) / B.FPS
    z[name], z[name + "_starts"] = np.concatenate(all_f), np.array(starts)
    print(f"{name:7s} {old:6.0f} s -> {n / B.FPS:6.0f} s, {len(starts)} clips, rejected {bad}  {secs}", flush=True)
np.savez(ROOT / "data/amp_v4.npz", **z)
print("wrote data/amp_v4.npz", {k: v.shape for k, v in z.items() if not k.endswith("_starts")})
