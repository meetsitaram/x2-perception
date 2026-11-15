# Pass 2 of goal_tiles_v2c (ACP, 2026-11-14): tiles with a deep (> 0.3 m) fault in the full check of pass 1 (dyn_v2c_all.json)
# take the spawn and route of the cleanest sibling of the same family and level. usage: python mk_tiles_v2c_pass2.py PASS1 DYN OUT
import json, copy, sys
g = json.load(open(sys.argv[1])); d = json.load(open(sys.argv[2]))
deep = [k for k, r in d.items() if r["fault"] and r["depth_max"] > 0.3]
src = {t["name"]: copy.deepcopy(t) for t in g["tiles"]}
sub = {}
for t in g["tiles"]:
    if t["name"] in deep:
        sib = sorted([s for s in src.values() if s["family"] == t["family"] and s["level"] == t["level"] and s["name"] not in deep],
                     key=lambda s: (d[s["name"]]["fault"], d[s["name"]]["depth_max"]))
        s = sib[0]
        for k in ("boxes", "spawn", "route", "v_max"): t[k] = copy.deepcopy(s[k])
        t["info"] = dict(copy.deepcopy(s["info"]), replaced="2026-11-14 pass 2: deep fault in the v2c check; spawn and route of %s" % s["name"])
        sub[t["name"]] = s["name"]
g["note"] += " | pass 2: %d tiles with a deep (> 0.3 m) fault in the v2c check take the cleanest sibling" % len(sub)
json.dump(g, open(sys.argv[3], "w")); print(sub)
