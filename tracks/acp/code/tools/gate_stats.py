"""Window statistics of an ACP run's metrics.jsonl (ACP agent, 2026-11-13): mean of throughput, finish and per-set finish keys
(probe = planner contacts, train = per-set training finish) over the last N rows, optionally only rows with it >= MIN_IT.
usage: gate_stats.py <metrics.jsonl> <N> [min_it] [out.json]"""
import json, sys
P, N = sys.argv[1], int(sys.argv[2]); MIN = int(sys.argv[3]) if len(sys.argv) > 3 else 0
rows = []
for l in open(P):
    try:
        r = json.loads(l)
    except Exception:
        continue
    if r.get("it", 0) >= MIN:
        rows.append(r)
rows = rows[-N:]
out = {"n": len(rows), "it_first": rows[0]["it"] if rows else None, "it_last": rows[-1]["it"] if rows else None}
keys = set()
for r in rows:
    keys |= {k for k, v in r.items() if isinstance(v, (int, float)) and (k in ("env_steps_per_s", "iter_s", "reward_per_step", "finish_frac", "term_frac")
             or k.startswith(("probe/finish/", "finish/", "set_finish/", "sim_fault")) or "finish" in k)}
for k in sorted(keys):
    v = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
    out[k] = round(sum(v) / len(v), 4) if v else None
s = json.dumps(out, indent=0)
if len(sys.argv) > 4:
    open(sys.argv[4], "w").write(s)
print(s)
