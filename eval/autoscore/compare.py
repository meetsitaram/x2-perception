"""Compare an autoscore run with an earlier hand-run result, seed by seed.
  python eval/autoscore/compare.py NEW_DIR OLD_DIR      (folders holding results.json, or partial.json)
Prints per seed: strict success, outcome, waypoints and final position of both; and SAME / DIFFERENT."""
import json, pathlib, sys


def load(d):
    d = pathlib.Path(d)
    for f in ("results.json", "partial.json"):
        if (d / f).exists():
            return {r["seed"]: r for r in json.load(open(d / f))}
    return {}


def key(r):
    xy = (r.get("final") or {}).get("xy") or [0, 0]
    return (bool(r.get("strict_success")), r.get("reason"), r.get("waypoints"), round(xy[0], 4), round(xy[1], 4))


new, old = load(sys.argv[1]), load(sys.argv[2])
same = diff = 0
for s in sorted(new):
    if s not in old:
        continue
    a, b = key(new[s]), key(old[s])
    ok = a == b; same += ok; diff += not ok
    if not ok:
        print("  seed %d DIFFERENT: new %s / old %s" % (s, a, b))
print("%s vs %s: %d same, %d different (%d new seeds without an old one)" % (sys.argv[1], sys.argv[2], same, diff, len(set(new) - set(old))))
