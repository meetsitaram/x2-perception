#!/usr/bin/env bash
# Tripwire of option (c), fixed before the start (ACP agent, 2026-11-12). Finish rate = end/finished_phase099, mean over
# the last 50 iterations. At 4,300: >= 60 % or (b). At 4,600: >= 78 % or (b). Target: >= 87 % by 5,000 (s1_v1 ended at 86.9 % on this measure, CSI starts).
# (b) = fresh stage 1 on the corrected body with data 2e, run s1_v3 from scratch (no init), started automatically.
set -u
T=/mnt/filesystem-g4/tracks/acp; R=$T/runs/s1_v2; L=$R/tripwire.log
say() { echo "$(date -u +%FT%TZ) $*" >> "$L"; }
fin() { python3 - "$R/metrics.jsonl" "$1" <<'PY'
import json, sys
it = int(sys.argv[2]); rows = {}
for l in open(sys.argv[1]):
    r = json.loads(l); rows[r["it"]] = r.get("end/finished_phase099")
a = [rows[i] for i in range(it - 49, it + 1) if rows.get(i) is not None]
print(f"{100 * sum(a) / max(len(a), 1):.2f}" if len(a) >= 40 else "nan")
PY
}
say "armed: 4300 >= 60, 4600 >= 78, target 5000 >= 87"
for pair in 4300:60 4600:78 5000:0; do
  it=${pair%:*}; th=${pair#*:}
  while [ "$(tail -1 $R/metrics.jsonl 2>/dev/null | python3 -c 'import json,sys;print(json.loads(sys.stdin.read())["it"])' 2>/dev/null || echo 0)" -lt $it ]; do sleep 60; done
  f=$(fin $it); say "iteration $it: finish rate $f % (threshold $th)"
  if [ "$f" = nan ] || python3 -c "import sys; sys.exit(0 if float('$f') >= $th else 1)"; then continue; fi
  say "tripwire at $it: switching to option (b), fresh stage 1 s1_v3"
  touch $R/STOP; for i in $(seq 90); do ! tmux has-session -t acp 2>/dev/null && break; sleep 2; done
  mkdir -p $T/runs/s1_v3
  tmux new -d -s acp "X2_EXCLUDE_PAIRS=hip ACP_PLANT=mc_20261024 DATA=cfg/stage2e_sets.json bash $T/code/supervise.sh $T/runs/s1_v3 36000 12000"
  say "s1_v3 started"; exit 0
done
say "option (c) kept"
