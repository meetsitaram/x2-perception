#!/usr/bin/env bash
# Live-evaluation test (coordinator, 2026-11-12): two paper-eval sizes beside training, one after the other (never two
# at once), right after checkpoint IT; training speed before / during (from eval_live.sh) and after (this script).
#   tmux new -d -s acp-evaltest "bash /mnt/filesystem-g4/tracks/acp/code/ops/live_test.sh 2750"
set -u
IT=$1; T=/mnt/filesystem-g4/tracks/acp; R=$T/runs/s2_v1; O=$T/eval/paper_eval; LOG=$O/live_test_$IT.log
say() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
after() {   # mean training speed over the iterations logged in the N seconds after an eval
  sleep "$2"; python3 - "$R/metrics.jsonl" "$1" "$2" <<'PY' >> "$LOG"
import json, sys, time, calendar
rows = [json.loads(l) for l in open(sys.argv[1])][-60:]
t1, n = int(sys.argv[2]), int(sys.argv[3])
ts = lambda r: calendar.timegm(time.strptime(r["time"], "%Y-%m-%dT%H:%M:%S"))
a = [r for r in rows if t1 + 8 <= ts(r) <= t1 + n]
print(json.dumps(dict(after_seconds=n, iterations=len(a), train_steps_per_s_after=round(sum(r["env_steps_per_s"] for r in a) / max(len(a), 1)),
                      gpu_mem_gb_after=a[-1]["gpu_mem_gb"] if a else None)))
PY
}
say "waiting for ckpt_$(printf %06d $IT).pt"
for PER in 100 334; do
  say "eval per-set-per-seed $PER: start"
  EVAL_TIMEOUT=1500 bash $T/code/tools/eval_live.sh "$R" "$IT" "$PER" mc_20261024 >> "$LOG" 2>&1
  say "eval $PER: rc $? ; $(cat $O/live_s2_v1_$(printf %06d $IT)_per${PER}.live.json 2>/dev/null | tr -d '\n ')"
  after "$(date +%s)" 90
done
say "done"
