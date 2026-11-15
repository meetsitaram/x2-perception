# Build goal_tiles_v2c.json (ACP, 2026-11-14): goal_tiles_v2 with the spawn fixes of the CPU spawn check
# (runs/s3_v2_diag: dyn_all.json -> spawn_fix.py passes -> spawn_search.sh = tiles_best.json); tiles that still fault get the
# spawn and route of a clean sibling of the same family and level (the tile count, order and clip indices stay as in v2, so
# checkpoints, curriculum levels and the replay's clip ids stay valid). usage: python mk_tiles_v2c.py BEST REMAINING DYN_ALL OUT
import json, sys, copy
best, rem, dyn, out = json.load(open(sys.argv[1])), set(json.load(open(sys.argv[2]))), json.load(open(sys.argv[3])), sys.argv[4]
orig = {t["name"]: t for t in json.load(open('/mnt/filesystem-g4/tracks/acp/data/goals/goal_tiles_v2.json'))["tiles"]}
fixed = [t["name"] for t in best["tiles"] if t["spawn"] != orig[t["name"]]["spawn"]]
clean = lambda t: t["name"] not in rem and (dyn[t["name"]]["fault"] == 0 or t["name"] in fixed)
sub = {}
for t in best["tiles"]:
    if t["name"] in rem:
        sib = [s for s in best["tiles"] if s["family"] == t["family"] and s["level"] == t["level"] and clean(s)]
        assert sib, t["name"]
        s = sib[0]
        nm = t["name"]
        for k in ("boxes", "spawn", "route", "v_max"):
            t[k] = copy.deepcopy(s[k])
        t["info"] = dict(copy.deepcopy(s["info"]), replaced="2026-11-14: spawn check found no clear spawn; spawn and route of %s" % s["name"])
        sub[nm] = s["name"]
best["note"] = best["note"] + " | v2c (2026-11-14, ACP agent): spawn check of all tiles (training resets, 10 steps, simcheck): %d spawns moved to a clear point on the same tile, %d tiles without a clear spawn take a clean sibling's spawn and route (state/acp.md 17m)" % (len([f for f in fixed if f not in rem]), len(sub))
json.dump(best, open(out, "w"))
print("moved", len([f for f in fixed if f not in rem]), "substituted", sub)
