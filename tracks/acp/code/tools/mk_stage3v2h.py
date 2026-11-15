"""Stage-3 data cfg on the 2h reference sets (ACP agent, 2026-11-13, state/acp.md 17k).
Stage 3 continues the reference training "through proportional sampling from the original dataset" (paper): the original
dataset of this run is stage 2h, so cfg/stage3v2h_sets.json = cfg/stage2h_sets.json (sets, weights, filters, armfix) + the goal
tiles of v2b (data/goals/goal_tiles_v2.json: our 6 families + the extended map v2b, no ladders or vaults, user rule).
One raster: hmap_ext_v2h_40mm.npz = hmap_ext_v2_40mm.npz (extended map + 2g ramp) with the 2h ramp raster (ramp + its mirror
6 m beside) pasted by max at the same origin. Checks: the same origin and resolution; the extended raster is empty where the
mirror goes; no goal tile spawn or waypoint inside the pasted mirror area.
    ~/acp_env/bin/python tools/mk_stage3v2h.py
"""
import json, os, numpy as np
T = "/mnt/filesystem-g4/tracks/acp"; D = f"{T}/data/clips"; C = f"{T}/code/cfg"
ext = np.load(f"{D}/hmap_ext_v2_40mm.npz"); r2h = np.load(f"{D}/tp_ramp_hmap_x30_2h_40mm.npz"); r2g = np.load(f"{D}/tp_ramp_hmap_x30_40mm.npz")
for k in ("x0", "y0", "res"):
    assert abs(float(ext[k]) - float(r2h[k])) < 1e-9, k
G = ext["grid"].copy(); a, b = r2h["grid"].shape; nb = r2g["grid"].shape[1]
old = G[:a, :b]
assert float(np.abs(old[:, :nb] - np.maximum(np.nan_to_num(r2g["grid"]), 0)).max()) < 1e-6, "2g ramp not where expected in the ext raster"
assert float(np.abs(old[:, nb:]).max()) == 0.0, "extended raster not empty where the mirror goes"
G[:a, :b] = np.maximum(old, np.maximum(np.nan_to_num(r2h["grid"]), 0))
x0, y0, res = float(ext["x0"]), float(ext["y0"]), float(ext["res"])
mx = (x0, x0 + (a - 1) * res); my = (y0 + nb * res, y0 + (b - 1) * res)
gt = json.load(open(f"{T}/data/goals/goal_tiles_v2.json"))
bad = [t["name"] for t in gt["tiles"] for w in t["route"] + [t["spawn"]] if mx[0] <= w[0] <= mx[1] and my[0] <= w[1] <= my[1]]
assert not bad, f"goal tiles inside the mirror area: {bad[:5]}"
np.savez_compressed(f"{D}/hmap_ext_v2h_40mm.npz", grid=G, x0=np.float64(x0), y0=np.float64(y0), res=np.float64(res))
c2h, c3 = json.load(open(f"{C}/stage2h_sets.json")), json.load(open(f"{C}/stage3v2b_sets.json"))
out = dict(c2h); out["hmap"] = "hmap_ext_v2h_40mm.npz"; out["goal_tiles"] = c3["goal_tiles"]
out["note"] = ("stage 3 v2h (2026-11-13, ACP agent, tools/mk_stage3v2h.py): stage 2h reference sets (unchanged) + goal tiles v2b "
               f"({c3['goal_tiles']}) + one raster hmap_ext_v2h_40mm.npz (extended map v2 + the 2h ramp raster with its mirror). | " + c2h["note"])
json.dump(out, open(f"{C}/stage3v2h_sets.json", "w"), indent=1)
print(f"raster {G.shape}, mirror area x {mx[0]:.2f}..{mx[1]:.2f} y {my[0]:.2f}..{my[1]:.2f}; goal tiles {len(gt['tiles'])}, families {gt['families']}")
print("wrote", f"{C}/stage3v2h_sets.json", "sets", len(out["sets"]))
