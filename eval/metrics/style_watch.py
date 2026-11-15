# Style reward watch (user, 2026-11-13: "also monitor the style reward"). Prints, per track with a style term, the mean over the
# last 100 logged iterations against the 100 before the v4 switch and against the previous refresh, and appends one line to
# eval/metrics/out/style_watch.csv. Run after `python eval/metrics/plots.py` (the hourly refresh does both).
import csv, datetime, math, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval" / "metrics"))
import plots  # noqa: E402

# (track, label, key, v4 switch iteration)
KEYS = [("ssr", "style r_s (per step)", "r_s", 11350),
        ("ssr", "D(policy)", "d_f", 11350),
        ("hiking", "discriminator reward", "Step/discriminator_reward", 11125),
        ("hiking", "D(actor)", "Train/discriminator_actor", 11125)]
OUT = ROOT / "eval" / "metrics" / "out" / "style_watch.csv"


def mean(rows, its, key):
    v = [rows[i].get(key) for i in its]
    v = [x for x in v if isinstance(x, (int, float)) and math.isfinite(x)]
    return round(sum(v) / len(v), 4) if v else None


def main():
    prev = {}
    if OUT.exists():
        for r in csv.DictReader(open(OUT)):
            prev[(r["track"], r["key"])] = r
    cache = {}
    lines = []
    now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M")
    for track, label, key, sw in KEYS:
        if track not in cache:
            cache[track] = plots.load(track)[0]
        rows = cache[track]
        its = sorted(i for i in rows if key in rows[i])
        if not its:
            print(track, label, "no data"); continue
        last = its[-100:]
        before = [i for i in its if i < sw][-100:]
        m, b = mean(rows, last, key), mean(rows, before, key)
        p = prev.get((track, key), {}).get("last100")
        print("%-7s %-24s it %6d  last100 %s  before v4 %s  previous refresh %s" % (track, label, its[-1], m, b, p))
        lines.append(dict(utc=now, track=track, key=key, label=label, iteration=its[-1], last100=m, before_v4=b))
    new = not OUT.exists()
    with open(OUT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["utc", "track", "key", "label", "iteration", "last100", "before_v4"])
        if new:
            w.writeheader()
        w.writerows(lines)


if __name__ == "__main__":
    main()
