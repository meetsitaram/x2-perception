#!/usr/bin/env bash
# Live paper evaluation, at most one per 1,000 iterations (coordinator, 2026-11-12: accepted; cost about 85 s of training
# each): right after each listed checkpoint, the full paper eval (334 per set per seed, 3 seeds) beside training through
# tools/eval_live.sh (lock, 80 GB start check, nice, cores 0-3, 25-minute limit, self-abort under 10 GB free), then the
# training speed for 90 s after. One line per eval in eval/paper_eval/live_costs.log.
#   tmux new -d -s acp-evallive "bash /mnt/filesystem-g4/tracks/acp/code/ops/live_every.sh 4000 5000 6000"
set -u
T=/mnt/filesystem-g4/tracks/acp; R=$T/runs/s2_v1; O=$T/eval/paper_eval; LOG=$O/live_costs.log
for IT in "$@"; do
  P=$(printf %06d "$IT")
  while [ ! -f "$R/ckpt_$P.pt" ]; do [ -f "$R/STOP" ] && sleep 60; sleep 20; done
  EVAL_TIMEOUT=1500 bash $T/code/tools/eval_live.sh "$R" "$IT" 334 mc_20261024 > "$O/live_s2_v1_${P}_per334.driver.txt" 2>&1
  rc=$?; t1=$(date +%s); sleep 90
  after=$(python3 - "$R/metrics.jsonl" "$t1" <<'PY'
import json, sys, time, calendar
rows = [json.loads(l) for l in open(sys.argv[1])][-40:]
t1 = int(sys.argv[2]); ts = lambda r: calendar.timegm(time.strptime(r["time"], "%Y-%m-%dT%H:%M:%S"))
a = [r["env_steps_per_s"] for r in rows if t1 + 8 <= ts(r) <= t1 + 90]
print(round(sum(a) / max(len(a), 1)))
PY
)
  echo "$(date -u +%FT%TZ) it $IT rc $rc after_steps_per_s $after $(tr -d '\n ' < $O/live_s2_v1_${P}_per334.live.json 2>/dev/null)" >> "$LOG"
done
