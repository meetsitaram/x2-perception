"""ACP adapter: the training model exactly as acp/data.py Dataset builds it (load_sets -> Terrain -> build_x2 with the prop
slots, the cfg's height raster and the terrain foot boxes). Environment as the training launch: ACP_PLANT, X2_EXCLUDE_PAIRS,
ACP_HF_CHUNK, ACP_HF_FIX (pass --hf-fix 0/1 to choose; default: the caller's environment).
Tiles: one per clip set (the set's first clip with its props as mocap boxes, robot in the clip's first frame), so every
terrain kind the cfg trains on is covered (the ramp raster through the ramp set)."""
from __future__ import annotations

import os
import sys

import numpy as np

CODE = "/mnt/filesystem-g4/tracks/acp/code"
FOOT = "left_ankle_roll_link"


def build(opts: dict) -> dict:
    os.environ.setdefault("ACP_PLANT", "mc_20261024")
    os.environ.setdefault("X2_EXCLUDE_PAIRS", "hip")
    if opts.get("hf_fix") is not None:
        os.environ["ACP_HF_FIX"] = str(opts["hf_fix"])
    sys.path.insert(0, CODE)
    import torch
    from acp.data import load_sets
    from gear_sonic.supermimic.assets import build_x2
    from gear_sonic.supermimic.terrain import Terrain
    cfg = opts.get("cfg") or "cfg/stage2g_sets.json"
    clips, props, set_of, c = load_sets(cfg if os.path.isabs(cfg) else os.path.join(CODE, cfg))
    names = list(clips)
    hmap = os.path.join(c["dir"], c["hmap"]) if c.get("hmap") else ""
    terr = Terrain(names, props, "cpu", hmap)
    x2 = build_x2(os.environ["ACP_PLANT"], prop_slots=terr.slot_half or None, raster=terr.raster, foot_terrain=True)
    m = x2.model
    tiles = []
    for si, s in enumerate(c["sets"]):
        mine = [i for i, n in enumerate(names) if set_of[n] == si]
        if not mine:
            continue
        i = next((j for j in mine if names[j] in props), mine[0])
        cl = clips[names[i]]
        q = np.zeros(m.nq)
        q[:3] = np.asarray(cl["root_trans_offset"])[0]
        x, y, z, w = np.asarray(cl["root_rot"])[0]          # xyzw
        q[3:7] = [w, x, y, z]
        q[7:] = np.asarray(cl["dof"])[0]
        q[2] += 0.01
        mocap = None
        if m.nmocap:
            p, qq = terr.mocap(torch.tensor([i]))
            mocap = (p[0].numpy().astype(np.float64), qq[0].numpy().astype(np.float64))
        tiles.append(dict(name=s["pkl"].replace(".pkl", ""), qpos=q, mocap=mocap))
    raster = None if terr.raster is None else dict(grid=terr.raster["grid"], x0=terr.raster["x0"], y0=terr.raster["y0"],
                                                   res=terr.raster["res"])
    return dict(track="acp", model=m, foot=FOOT, raster=raster, tiles=tiles, q_default=x2.default_q, diverge_qvel=200.0,
                nconmax=64, njmax=450, warp=True,
                config=dict(cfg=cfg, hmap=os.path.basename(hmap), ACP_HF_FIX=os.environ.get("ACP_HF_FIX", "0"),
                            ACP_HF_CHUNK=os.environ.get("ACP_HF_CHUNK", "400"), ACP_PLANT=os.environ["ACP_PLANT"],
                            X2_EXCLUDE_PAIRS=os.environ.get("X2_EXCLUDE_PAIRS", "")))
