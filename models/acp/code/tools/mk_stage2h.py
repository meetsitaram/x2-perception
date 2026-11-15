"""Data set 2h (ACP agent, 2026-11-13 04:40 UTC): 2g plus the corrected Pico sets and the mirrored sets.
(1) pico_2026-10-29_2x2_train_v1_1 replaces pico_2026-10-29_2x2_train_hold (old 2x2 set has the broken take
    pico_20261029_211326_0_2x2; old slot kept at weight 0 so set indices do not move); (2) tp_stairs_all_v1_1 (44 Pico stair
    takes) + its mirror, and a turn/approach subset (C, S, side approach, down then turn) as an upweighted extra set; (3) mirrors
    of 4x4, ramp (own mirrored height map placed 6 m beside the original in ONE raster, its clips shifted by the same offset) and
    2x2. Same end stand (1.5 s, _hold) and start stand (1.0 s, _hs) rules as 2g. Two takes dropped from the existing stairs set
    (114308, 114330: 2 cm contact check, state/acp.md 17b). New sets are appended (exclude_clips use set indices).
Writes data/clips/*, data/clips/tp_ramp_hmap_x30_2h_40mm.npz and cfg/stage2h_sets.json."""
import json, os, shutil, sys, joblib, numpy as np, torch
sys.path.insert(0, "/mnt/filesystem-g4/tracks/acp/code"); os.chdir("/mnt/filesystem-g4/tracks/acp/code")
from acp.data import Dataset
D = "/mnt/filesystem-g4/tracks/acp/data/clips"; C = "cfg/"
SRC = "/mnt/filesystem-g4/data/motions/ssr_prior_sources/terrain_park_train_sets"
cfg = json.load(open(C + "stage2g_sets.json"))
NEW = [  # (raw pkl, props or "", weight, hold, prepend, extra keys)
    ("pico_2026-10-29_2x2_train_v1_1.pkl", "pico_2026-10-29_2x2_props.json", 0.02, True, False, dict(exclude_leg_flip=True)),
    ("pico_2026-10-29_2x2_train_v1_1_mir.pkl", "pico_2026-10-29_2x2_train_v1_1_mir_props.json", 0.01, True, False, dict(exclude_leg_flip=True)),
    ("tp_stairs_all_v1_1.pkl", "tp_stairs_all_v1_1_props.json", 0.03, True, True, dict(ground_stance=True)),
    ("tp_stairs_all_v1_1_mir.pkl", "tp_stairs_all_v1_1_mir_props.json", 0.02, True, True, dict(ground_stance=True)),
    ("tp_t4x4u_y20_mir.pkl", "tp_t4x4u_y20_mir_props.json", 0.04, True, True, dict(ground_stance=True)),
    ("tp_ramp_base_x30_mir.pkl", "", 0.015, True, True, dict(ground_stance=True)),
]
TURN_CLS = {"C (up)", "S (up)", "C (down)", "S (down)", "side approach then up", "down then turn"}
for f in [n[0] for n in NEW] + [n[1] for n in NEW if n[1]] + ["tp_stairs_all_v1_1_meta.json", "tp_stairs_all_v1_1_mir_meta.json"]:
    if f == "pico_2026-10-29_2x2_props.json":
        continue
    shutil.copy2(os.path.join(SRC, f), os.path.join(D, f))
W = {"tp_stairs_x10_sh2_hs.pkl": 0.03, "tp_ramp_sh2_hs.pkl": 0.035, "tp_t4x4u_y20_hs.pkl": 0.08, "pico_2026-10-29_2x2_train_hold.pkl": 0.0}
for s in cfg["sets"]:
    if s["pkl"] in W:
        s["weight"] = W[s["pkl"]]
# ---- one raster for the ramp and its mirror
z0 = np.load(os.path.join(D, cfg["hmap"])); zm = np.load(os.path.join(SRC, "tp_ramp_hmap_x30_mir.npz"))
assert abs(float(z0["x0"]) - float(zm["x0"])) < 1e-6 and abs(float(z0["y0"]) - float(zm["y0"])) < 1e-6
g0 = z0["grid"]; r = float(z0["res"]); assert abs(float(zm["res"]) * 2 - r) < 1e-9
def pool(g):   # the trained 40 mm raster is the 20 mm raster sampled at every second vertex (checked: max diff 0)
    return np.nan_to_num(g)[::2, ::2][:g0.shape[0], :g0.shape[1]].astype(np.float32)
gm = pool(zm["grid"]); go = pool(np.load(os.path.join(D, "tp_ramp_hmap_x30.npz"))["grid"])
print("pool check (orig 20mm -> 40mm vs trained raster): max abs diff", float(np.abs(go - g0).max()))
DYC = 150; DY = DYC * r
G = np.zeros((g0.shape[0], DYC + g0.shape[1]), np.float32); G[:, :g0.shape[1]] = g0; G[:, DYC:DYC + gm.shape[1]] = gm
HM = "tp_ramp_hmap_x30_2h_40mm.npz"; np.savez(os.path.join(D, HM), x0=z0["x0"], y0=z0["y0"], res=z0["res"], grid=G)
x_lo, y_lo = float(z0["x0"]), float(z0["y0"]); x_hi, y_hi = x_lo + G.shape[0] * r, y_lo + G.shape[1] * r
print(f"raster {G.shape} x {x_lo:.2f}..{x_hi:.2f} y {y_lo:.2f}..{y_hi:.2f}; mirror offset dy {DY:.2f} m")
# ---- raw sets into a temporary cfg, for contact labels
tmp = dict(cfg); tmp["armfix"] = ""; tmp["hmap"] = HM; base = len(cfg["sets"]); tmp["sets"] = [dict(s) for s in cfg["sets"]]
for pk, pr, w, hold, pre, ex in NEW:
    s = dict(pkl=pk, props=pr, weight=w, **ex)
    if pk.startswith("tp_ramp_base_x30_mir"):
        s["shift"] = [0.0, DY]
    tmp["sets"].append(s)
json.dump(tmp, open(C + "tmp_stage2h_raw.json", "w"), indent=1)
d = Dataset(C + "tmp_stage2h_raw.json", torch.device("cpu"), cache="")
m = d.motion
bad = {}
for c, nm in enumerate(m.names):
    si = int(nm.split("/")[0][1:]); a, e = int(m.clip_start[c]), int(m.clip_end[c]); P = m.root_p[a:e + 1, :2]
    inside = bool(((P[:, 0] > x_lo) & (P[:, 0] < x_hi) & (P[:, 1] > y_lo) & (P[:, 1] < y_hi)).any())
    if inside != tmp["sets"][si]["pkl"].startswith("tp_ramp"):
        bad[tmp["sets"][si]["pkl"]] = bad.get(tmp["sets"][si]["pkl"], 0) + 1
print("clips on the wrong side of the ramp raster (set: count):", bad)
first, last = {}, {}
for c, nm in enumerate(m.names):
    si, rest = nm.split("/", 1); key, seg = (rest.split("#") + ["0"])[:2]; k = (int(si[1:]), key)
    if k not in first or int(seg) < first[k][0]:
        first[k] = (int(seg), c)
    if k not in last or int(seg) > last[k][0]:
        last[k] = (int(seg), c)
ds = lambda fr: bool(d.contact[fr, 0]) and bool(d.contact[fr, 1])
def arr_edit(clip, k_pre, k_post):
    T = len(clip["dof"]); out = {}
    for k2, v in clip.items():
        if hasattr(v, "__len__") and not isinstance(v, str) and len(v) == T:
            v = np.asarray(v); parts = ([np.repeat(v[:1], k_pre, 0)] if k_pre else []) + [v] + ([np.repeat(v[-1:], k_post, 0)] if k_post else [])
            v = np.concatenate(parts)
        out[k2] = v
    return out
meta = json.load(open(os.path.join(D, "tp_stairs_all_v1_1_meta.json")))
turn_set = None
for i, (pk, pr, w, hold, pre, ex) in enumerate(NEW):
    si = base + i; src = joblib.load(os.path.join(D, pk)); new = {}; nh = npre = 0; miss = 0
    for key, clip in src.items():
        fps = float(clip["fps"]); kp = kh = 0
        if (si, key) not in first:
            miss += 1
        if pre and (si, key) in first and ds(int(m.clip_start[first[(si, key)][1]])):
            kp = int(round(1.0 * fps)); npre += 1
        if hold and (si, key) in last and ds(int(m.clip_end[last[(si, key)][1]])):
            kh = int(round(1.5 * fps)); nh += 1
        new[key] = arr_edit(clip, kp, kh)
    out = pk.replace(".pkl", "_hs.pkl" if pre else "_hold.pkl")
    joblib.dump(new, os.path.join(D, out)); print(f"{pk}: {len(src)} clips ({miss} filtered by the loader), end stand {nh}, start stand {npre} -> {out}")
    tmp["sets"][si]["pkl"] = out
    if pk == "tp_t4x4u_y20_mir.pkl":   # mirrors of the two 4x4 clips excluded in 2f (soles 9-10 cm inside the log)
        tmp["sets"][si]["exclude_clips"] = [f"s{si}/tp_t4x4_105103_2_mir", f"s{si}/tp_t4x4_104900_0_mir"]
    if pk == "tp_stairs_all_v1_1.pkl":
        sub = {k: v for k, v in new.items() if meta.get(k, {}).get("cls") in TURN_CLS}
        joblib.dump(sub, os.path.join(D, "tp_stairs_turn_v1_1_hs.pkl")); print("turn/approach subset:", len(sub), "clips")
        turn_set = dict(pkl="tp_stairs_turn_v1_1_hs.pkl", props=pr, weight=0.01, ground_stance=True)
tmp["sets"].append(turn_set)
xs = [k for k in joblib.load(os.path.join(D, "tp_stairs_x10_sh2_hs.pkl")) if "114308" in k or "114330" in k]
assert tmp["sets"][3]["pkl"] == "tp_stairs_x10_sh2_hs.pkl"
tmp["sets"][3]["exclude_clips"] = tmp["sets"][3].get("exclude_clips", []) + ["s3/" + k for k in xs]; print("stairs x10 excluded:", xs)
tot = sum(s["weight"] for s in tmp["sets"])
for s in tmp["sets"]:
    if s["pkl"] == "x2_loco_flat_train_hold.pkl":
        s["weight"] = round(s["weight"] - (tot - 1.0), 4); print("flat weight ->", s["weight"])
print("weight sum", round(sum(s["weight"] for s in tmp["sets"]), 4))
tmp["armfix"] = "armfix_stage2h_sets.pt"
tmp["note"] = ("stage 2h (2026-11-13, ACP agent, tools/mk_stage2h.py): as 2g plus corrected Pico sets (2x2 v1_1 replaces the old 2x2 set, "
               "old slot at weight 0 for the indices; tp_stairs_all_v1_1, 44 takes), mirrored sets (2x2, stairs, 4x4, ramp with its mirrored map "
               "6 m beside the original in one raster " + HM + "), a stairs turn/approach subset (weight 0.01), stairs x10 minus 114308/114330. "
               "Weights: stairs 0.08 -> 0.03 x10 + 0.03 all + 0.02 mir + 0.01 turn, 4x4 0.12 -> 0.08 + 0.04 mir, ramp 0.05 -> 0.035 + 0.015 mir, "
               "2x2 0.03 -> 0.02 + 0.01 mir; flat takes the remainder. " + cfg["note"])
json.dump(tmp, open(C + "stage2h_sets.json", "w"), indent=1); os.remove(C + "tmp_stage2h_raw.json")
print("DONE cfg/stage2h_sets.json")
