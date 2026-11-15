# Spawn fix for faulting goal tiles (ACP, 2026-11-14). For each tile in FAULT list: candidate spawns on the same tile (shifts
# along the spawn -> first-waypoint line, back and forward, and sideways), kept only if the floor under the feet is flat at one
# height (raster range <= 2 cm in a 0.30 m disc), no raster cell rises more than 0.10 m above that floor within CLEAR m (body and
# arms under the +-0.4 rad reset yaw), and the candidate stays at least 0.5 m short of the first waypoint. Ranked by clearance,
# then by shift. Writes a tiles file in which each faulting tile gets its best candidate (to be verified dynamically) or none.
# usage: python spawn_fix.py TILES_IN dyn.json TILES_OUT [--clear 0.6] [--exclude names.json] [--rank k]
import json, sys, math, argparse, collections, numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("tiles_in"); ap.add_argument("dyn"); ap.add_argument("tiles_out")
ap.add_argument("--clear", type=float, default=0.6); ap.add_argument("--rank", type=int, default=0)
ap.add_argument("--only", default="", help="json list of tile names to fix (default: faulting in dyn)")
a = ap.parse_args()
T = '/mnt/filesystem-g4/tracks/acp'
cfg = json.load(open(f'{T}/code/cfg/stage3v2h_sets.json'))
z = np.load(f"{cfg['dir']}/{cfg['hmap']}")
G = np.nan_to_num(np.maximum(z['grid'], 0.0)); x0, y0, res = float(z['x0']), float(z['y0']), float(z['res'])
def disc(x, y, r):
    i0, i1 = int(np.floor((x - r - x0) / res)), int(np.ceil((x + r - x0) / res))
    j0, j1 = int(np.floor((y - r - y0) / res)), int(np.ceil((y + r - y0) / res))
    if i0 < 0 or j0 < 0 or i1 >= G.shape[0] or j1 >= G.shape[1]: return None, None
    ii, jj = np.meshgrid(np.arange(i0, i1 + 1), np.arange(j0, j1 + 1), indexing='ij')
    d = np.sqrt((x0 + ii * res - x) ** 2 + (y0 + jj * res - y) ** 2); m = d <= r
    return G[ii[m], jj[m]], d[m]
gt = json.load(open(a.tiles_in)); dyn = json.load(open(a.dyn))
fix = set(json.load(open(a.only))) if a.only else {n for n, r in dyn.items() if r["fault"] > 0}
log = {}
for t in gt["tiles"]:
    if t["name"] not in fix: continue
    x, y, sz, yaw = t["spawn"]; w = t["route"][0]
    dx, dy = w[0] - x, w[1] - y; L = math.hypot(dx, dy); ux, uy = (dx / L, dy / L) if L > 1e-6 else (math.cos(yaw), math.sin(yaw))
    cands = []
    for s in np.arange(-1.0, 1.01, 0.1):
        if s > 0 and L - s < 0.5: continue
        for o in (0.0, 0.15, -0.15, 0.3, -0.3, 0.45, -0.45):
            cx, cy = x + s * ux - o * uy, y + s * uy + o * ux
            feet, _ = disc(cx, cy, 0.30)
            if feet is None: continue
            fz = float(np.median(feet))
            if float(feet.max() - feet.min()) > 0.02: continue
            h, d = disc(cx, cy, 1.2)
            if h is None: continue
            tall = d[h > fz + 0.10]
            clr = float(tall.min()) if len(tall) else 1.2
            if clr < a.clear: continue
            cands.append((-round(clr, 2), abs(s) + abs(o), cx, cy, fz))
    cands.sort()
    if len(cands) > a.rank:
        _, sh, cx, cy, fz = cands[a.rank]
        log[t["name"]] = dict(old=t["spawn"], new=[round(cx, 3), round(cy, 3), round(fz, 3), yaw], shift=round(sh, 2), clearance=-cands[a.rank][0], n_cands=len(cands))
        t["spawn"] = [round(cx, 3), round(cy, 3), round(fz, 3), yaw]
        t.setdefault("info", {})["spawn_fix"] = "2026-11-14: spawn moved %.2f m (clearance %.2f m), old %s" % (sh, -cands[a.rank][0], log[t["name"]]["old"])
    else:
        log[t["name"]] = dict(old=t["spawn"], new=None, n_cands=len(cands))
json.dump(gt, open(a.tiles_out, "w"))
json.dump(log, open(a.tiles_out + ".fixlog.json", "w"), indent=0)
c = collections.Counter((n.rsplit("_L", 1)[0], v["new"] is not None) for n, v in log.items())
print("fix candidates", dict(c))
