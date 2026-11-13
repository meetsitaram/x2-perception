#!/usr/bin/env bash
# Needed restart of stage 2 at checkpoint IT: install the staged code (planner in chunks), check that resume from the
# checkpoint works (small throw-away run), then resume the real run. Heartbeat states the reason.
#   tmux new -d -s acp-restart2e "DATA=cfg/stage2e_sets.json BUDGET=10000 bash /mnt/filesystem-g4/tracks/acp/code/ops/restart_s2e.sh 4000 reason"  (waits for the live eval at IT to finish first)
set -u
IT=$(printf %06d "$1"); WHY=$2
T=/mnt/filesystem-g4/tracks/acp; R=$T/runs/s2_v1; L=$R/restart.log; HB=/mnt/filesystem-g4/status/acp.json; PY=$HOME/acp_env/bin/python
DATA=${DATA:-cfg/stage2_sets.json}
EXTRA="--stage 2 --init $R/init_from_s1_v1_002250.pt --planner-arch attn --u-obs 200 --u-rew 1700 --n-ann 1500"
say() { echo "$(date -u +%FT%TZ) $*" >> "$L"; }
gpu_free() { ! nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; }
say "waiting for ckpt_$IT.pt and the live eval at it $((10#$IT))"; while ! grep -q " it $((10#$IT)) " /mnt/filesystem-g4/tracks/acp/eval/paper_eval/live_costs.log 2>/dev/null; do sleep 15; done; exec 8> /tmp/acp_eval.lock; flock -w 1800 8
while [ ! -f "$R/ckpt_$IT.pt" ]; do sleep 5; done
sleep 4
touch "$R/STOP"
for i in $(seq 60); do gpu_free && ! tmux has-session -t acp 2>/dev/null && break; sleep 2; done
say "trainer stopped; GPU free: $(gpu_free && echo yes || echo no)"
printf '{"track":"acp","state":"restart at checkpoint %s (under 10 minutes): %s","run":"%s","newest_checkpoint":"%s","unix":%d,"time":"%s"}\n' "$IT" "$WHY" "$R" "$R/ckpt_$IT.pt" "$(date +%s)" "$(date +%FT%T%z)" > "$HB"
TR=$T/_trash/$(date -u +%F); mkdir -p "$TR"
[ -d "$T/code_prev" ] && mv "$T/code_prev" "$TR/code_prev_$(date -u +%H%M%S)" && echo "code_prev_*: roll-back copy of code/acp and code/tools from before an install; superseded" >> "$TR/NOTES.txt"
mkdir -p "$T/code_prev"; cp -r "$T/code/acp" "$T/code/tools" "$T/code/supervise.sh" "$T/code/backup.sh" "$T/code_prev/"
cp "$T"/code_next/acp/*.py "$T/code/acp/"; cp "$T"/code_next/tools/*.py "$T"/code_next/tools/*.sh "$T/code/tools/"
cd "$T/code"
rm -rf /tmp/resume_test; mkdir -p /tmp/resume_test; cp "$R/ckpt_$IT.pt" /tmp/resume_test/latest.pt
N0=$((10#$IT))
if ACP_PLANT=mc_20261024 timeout 240 "$PY" -m acp.train --data "$DATA" --run /tmp/resume_test --envs 4096 --iters $((N0 + 3)) --njmax 450 --resume $EXTRA --heartbeat /tmp/hb.json > /tmp/resume_test.out 2>&1 \
   && grep -q "^\[resume\] /tmp/resume_test/latest.pt at iteration $N0" /tmp/resume_test.out && [ "$(wc -l < /tmp/resume_test/metrics.jsonl)" -ge 3 ]; then
  say "resume check passed: $(grep '^\[resume\]\|^\[stage 2\]' /tmp/resume_test.out | tr '\n' ' ' | cut -c1-300) | last row: $(tail -1 /tmp/resume_test/metrics.jsonl | cut -c1-160)"
else
  say "resume check or new code FAILED: restoring the previous code (see /tmp/resume_test.out, mirrored to backup/tmp_outputs)"
  cp "$T"/code_prev/acp/*.py "$T/code/acp/"; cp "$T"/code_prev/tools/* "$T/code/tools/" 2>/dev/null
fi
for i in $(seq 30); do gpu_free && break; sleep 2; done
mkdir -p "$TR/stop_flags"; mv "$R/STOP" "$TR/stop_flags/STOP_s2_v1_$IT" && echo "stop_flags/STOP_s2_v1_$IT: empty stop flag of the restart at checkpoint $IT" >> "$TR/NOTES.txt"
echo "--- $(date -u +%FT%T) restart at checkpoint $IT: $WHY" >> "$R/train.out"
tmux new -d -s acp "ACP_PLANT=mc_20261024 DATA=$DATA EXTRA='$EXTRA' bash $T/code/supervise.sh $R 36000 ${BUDGET:-10000}"
say "supervisor restarted (data $DATA)"
