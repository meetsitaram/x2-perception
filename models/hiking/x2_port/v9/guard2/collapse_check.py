# Collapse watchdog (FINDINGS H7b, hiking track 2026-11-15). Reads the tail of metrics.jsonl (file order; a resume appends
# iterations again from the checkpoint, so a non-increasing iteration marks a restart).
# exit 0 = fine or warm-up; exit 2 = collapse, prints "COLLAPSE <onset_iteration> <detail>".
# Rule: >= 350 rows since the last restart, mean episode length of the last 10 rows < 0.4 x the median of rows -300..-50,
# and mean reward of the last 10 < 0.5 x the median reward of those rows. Onset = first of the last 100 rows with length
# < 0.7 x that median. v9's collapse at 22,116 (length 973 -> 625 -> 55) fires at about 22,121 with onset 22,116.
import json, sys, os, statistics
def check(rows):
    start = 0
    for i in range(1, len(rows)):
        if rows[i][0] <= rows[i - 1][0]:
            start = i
    rows = rows[start:]
    if len(rows) < 350:
        return (0, "OK warm-up rows_since_restart %d" % len(rows))
    base = rows[-300:-50]
    bl = statistics.median(r[1] for r in base); br = statistics.median(r[2] for r in base)
    rl = sum(r[1] for r in rows[-10:]) / 10; rr = sum(r[2] for r in rows[-10:]) / 10
    if rl < 0.4 * bl and (br <= 0 or rr < 0.5 * br):
        onset = next((r[0] for r in rows[-100:] if r[1] < 0.7 * bl), rows[-10][0])
        return (2, "COLLAPSE %d length %.0f -> %.0f reward %.1f -> %.1f (iteration %d)" % (onset, bl, rl, br, rr, rows[-1][0]))
    return (0, "OK length %.0f / %.0f reward %.1f / %.1f iteration %d" % (rl, bl, rr, br, rows[-1][0]))

if __name__ == "__main__":
    path = sys.argv[1]
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        f.seek(max(0, size - 30_000_000))
        raw = f.read().decode("utf-8", "replace").splitlines()[1:]
    rows = []
    for l in raw[-3000:]:
        try:
            d = json.loads(l)
            rows.append((int(d["iteration"]), float(d["length"]), float(d["reward"])))
        except Exception:
            pass
    code, msg = check(rows)
    print(msg); sys.exit(code)
