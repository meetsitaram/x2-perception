"""One-line run summary at an iteration from metrics.jsonl (last record with iteration <= N; mean over a window).
  python3 report_line.py <run folder> <iteration> [window=10]
Falls = episodes ended by base_contact, bad_orientation or root_height; `stalled` (run 3 option d) is a reset, listed apart."""
import json, sys

run, it = sys.argv[1], int(sys.argv[2])
win = int(sys.argv[3]) if len(sys.argv) > 3 else 10
recs = {}
for line in open(run + "/metrics.jsonl"):
    try:
        d = json.loads(line)
    except Exception:
        continue
    if "iteration" in d:
        recs[d["iteration"]] = d  # a redone iteration: keep the last
sel = [recs[i] for i in sorted(recs) if it - win < i <= it]
if not sel:
    sys.exit("no records near %d (newest %s)" % (it, max(recs) if recs else None))


def m(k):
    v = [d[k] for d in sel if k in d]
    return sum(v) / len(v) if v else float("nan")


W = "Curriculum/walk_monitor/"
falls = sum(m("Episode_Termination/" + k) for k in ("base_contact", "bad_orientation", "root_height"))
lv = {k[len(W + "level/"):]: d for k, d in sel[-1].items() if k.startswith(W + "level/")}
print("it %d-%d: speed ratio %.3f (%.2f/%.2f m/s), both feet down %.0f%%, no support %.1f%%, step rate %.1f Hz, air-time reward/ep %.4f, "
      "wait penalty/ep %.2f, alive/ep %.1f, ep len %.0f, falls %.0f%%, stalled resets %.0f%%, timeouts %.0f%%, level mean %.2f | %s"
      % (sel[0]["iteration"], sel[-1]["iteration"], m(W + "speed_ratio"), m(W + "actual_speed"), m(W + "cmd_speed"),
         100 * m(W + "double_support"), 100 * m(W + "no_support"), m(W + "step_rate_hz"),
         m("Episode_Reward/rewards_feet_air_time/sum"), m("Episode_Reward/rewards_dont_wait/sum"), m("Episode_Reward/rewards_is_alive/sum"),
         m("length"), 100 * falls, 100 * m("Episode_Termination/stalled"), 100 * m("Episode_Termination/time_out"),
         m("Curriculum/terrain_levels"), " ".join("%s %.2f" % (k, v) for k, v in sorted(lv.items()))))
