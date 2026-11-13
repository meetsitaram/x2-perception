"""SSR adapter: the training model exactly as ssr/train.py builds it with the extended map (terrain.generate ->
terrain.add_extended -> SSREnv's x2.build(raster, walls) with ext_phys parked wall boxes). Environment as the launch:
X2_PLANT, X2_EXCL_PELVIS_HIPROLL, X2_HF_CHUNK (default: the caller's environment; the builder's default is 400).
opts["ssr"]: the train flags that shape the terrain (defaults = run r7 v14, 2026-11-12 23:16).
Tiles: every base type at levels 0, 10 and 19 (spawn 0.6 m into the tile, mid-lane) and one route start per extended tile
type."""
from __future__ import annotations

import os
import sys

CODE = "/mnt/filesystem-g4/tracks/ssr/code"
FOOT = "left_ankle_roll_link"
R7 = dict(seed=0, plant="mc_20261024", excl_pelvis_hiproll=True, ext_types="tight_spots,log_corridors,homes,street_front,stairs,"
          "wedges,single_beams,beam_fields,blocks", ext_variants=2, ext_min_clear=0.0, ext_stairs_inset=True, ext_prop_routes=True,
          ext_stations=1, ext_slope_turns=2, ext_phys=24)


def build(opts: dict) -> dict:
    a = dict(R7, **(opts.get("ssr") or {}))
    os.environ.setdefault("X2_PLANT", a["plant"])
    if a["excl_pelvis_hiproll"]:
        os.environ.setdefault("X2_EXCL_PELVIS_HIPROLL", "1")
    if opts.get("hf_chunk") is not None:
        os.environ["X2_HF_CHUNK"] = str(opts["hf_chunk"])
    sys.path.insert(0, CODE)
    os.chdir(CODE)
    from ssr import terrain as T
    from ssr import x2 as X
    r = T.add_extended(T.generate(a["seed"]), types=tuple(a["ext_types"].split(",")), variants=a["ext_variants"],
                       min_along=a["ext_min_clear"], stairs_inset=a["ext_stairs_inset"], prop_routes=a["ext_prop_routes"],
                       stations=a["ext_stations"], slope_turns=a["ext_slope_turns"])
    walls = [dict(center=(0.0, 0.0, -20.0 - 0.5 * i), half=(0.05, 0.05, 0.05), quat=(1.0, 0.0, 0.0, 0.0))
             for i in range(a["ext_phys"])]
    x2 = X.build(r, walls or None)
    m = x2.model
    tiles = []
    bn, tx, ty = T._i(T.BORDER), T._i(T.TILE_X), T._i(T.TILE_Y)
    for c, name in enumerate(T.TYPES):
        for lv in (0, 10, T.NLEVEL - 1):
            x = r["x0"] + (bn + c * tx) * r["res"] + 0.6
            y = r["y0"] + (bn + lv * ty) * r["res"] + T.TILE_Y / 2
            tiles.append(dict(name=f"{name}_L{lv}", xy=(x, y), yaw=0.0))
    ext = r.get("ext") or {}
    seen = set()
    for rt in ext.get("routes", []):
        tl = ext["tiles"][rt["tile"]] if isinstance(rt.get("tile"), int) and rt["tile"] < len(ext.get("tiles", [])) else {}
        kind = tl.get("type") or tl.get("kind") or str(rt.get("tile"))
        if kind in seen:
            continue
        seen.add(kind)
        sx, sy, syaw = (list(rt["start"]) + [0.0])[:3]
        tiles.append(dict(name=f"ext_{kind}", xy=(sx, sy), yaw=float(syaw)))
    raster = dict(grid=r["z"], x0=r["x0"], y0=r["y0"], res=r["res"])
    return dict(track="ssr", model=m, foot=FOOT, raster=raster, tiles=tiles, q_default=x2.default_q, diverge_qvel=60.0,
                nconmax=150, njmax=900, warp=True,
                config=dict(ssr=a, X2_HF_CHUNK=os.environ.get("X2_HF_CHUNK", "400 (builder default)"),
                            X2_PLANT=os.environ["X2_PLANT"], raster_shape=list(r["z"].shape)))
