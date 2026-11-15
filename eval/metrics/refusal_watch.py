# Refusal watch (user, 2026-11-14: "Stopped at the foot of the stairs - these have to be converted to - fell ... slowly").
# For the newest scored checkpoints of each track, split station-leg trials into success / attempt (fell or stalled past the
# obstacle foot) / refusal (stalled or timed out at "foot of the stairs" or before the ramps' first face). Run at every refresh.
import csv, glob, os, re, sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def classify(x):
    o = (x.get("outcome") or "").lower()
    if (x.get("strict_success") == "1") or o.startswith("success"):
        return "success"
    wp = (x.get("last_waypoint") or "").lower()
    if ("stall" in o or "time limit" in o) and "foot of" in wp:
        return "refusal"
    return "attempt/other"


def main(n=3):
    for track in ("acp", "ssr", "hiking"):
        dirs = sorted(glob.glob(os.path.join(ROOT, "eval", "out", track, "*", "auto", "legs", "leg5", "trials.csv")),
                      key=os.path.getmtime)[-n:]
        for f in dirs:
            ck = f.split(os.sep)[-5]
            c = Counter(classify(x) for x in csv.DictReader(open(f)))
            tot = sum(c.values())
            print("%-6s %-24s leg5: success %2d  attempt/other %2d  refusal %2d  (of %d)" % (
                track, ck, c["success"], c["attempt/other"], c["refusal"], tot))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
