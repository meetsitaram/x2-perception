#!/usr/bin/env bash
# Option (c) of the hip-roll decision (ACP agent, 2026-11-12): after the 4,000 live eval of s2_v1, stop stage 2 and
# fine-tune the tracker as the paper's stage 1 (oracle contacts) on the corrected body (X2_EXCLUDE_PAIRS=hip), data 2e
# (style filter + 2d arm fix), from s2_v1 ckpt_004000, as run s1_v2. Then the tripwire (tripwire_s1v2.sh) decides between
# keeping it and a fresh stage 1 (option b).
#   tmux new -d -s acp-restart "bash /mnt/filesystem-g4/tracks/acp/code/ops/restart_s1v2.sh"
set -u
T=/mnt/filesystem-g4/tracks/acp; R2=$T/runs/s2_v1; R=$T/runs/s1_v2; L=$R/restart.log; HB=/mnt/filesystem-g4/status/acp.json
PY=$HOME/acp_env/bin/python; IT=004000; DATA=cfg/stage2e_sets.json; BUDGET=${BUDGET:-12000}
mkdir -p "$R"
say() { echo "$(date -u +%FT%TZ) $*" >> "$L"; }
gpu_free() { ! nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; }
say "waiting for s2_v1 ckpt_$IT and its live eval"
while [ ! -f "$R2/ckpt_$IT.pt" ]; do sleep 15; done
while ! grep -q " it 4000 " $T/eval/paper_eval/live_costs.log 2>/dev/null; do sleep 15; done
exec 8> /tmp/acp_eval.lock; flock -w 1800 8
tmux kill-session -t acp-evallive 2>/dev/null; say "s2_v1 live-eval watcher ended (s2_v1 stops here)"
cp "$R2/ckpt_$IT.pt" "$R/init_from_s2_v1_$IT.pt"; say "init copied: $R/init_from_s2_v1_$IT.pt"
touch "$R2/STOP"
for i in $(seq 90); do gpu_free && ! tmux has-session -t acp 2>/dev/null && break; sleep 2; done
say "stage 2 stopped; GPU free: $(gpu_free && echo yes || echo no)"
printf '{"track":"acp","state":"restart: stage-1 fine-tune on the corrected body (pelvis/hip-roll excluded), data 2e, from s2_v1 4000 (under 10 minutes)","run":"%s","newest_checkpoint":"%s","unix":%d,"time":"%s"}\n' "$R" "$R2/ckpt_$IT.pt" "$(date +%s)" "$(date +%FT%T%z)" > "$HB"
TR=$T/_trash/$(date -u +%F); mkdir -p "$TR/stop_flags"
[ -d "$T/code_prev" ] && mv "$T/code_prev" "$TR/code_prev_$(date -u +%H%M%S)" && echo "code_prev_*: roll-back copy from before an install; superseded" >> "$TR/NOTES.txt"
mkdir -p "$T/code_prev"; cp -r "$T/code/acp" "$T/code/tools" "$T/code_prev/"
cp "$T"/code_next/acp/*.py "$T/code/acp/"
cd "$T/code"
rm -rf /tmp/s1v2_check; mkdir -p /tmp/s1v2_check
if X2_EXCLUDE_PAIRS=hip ACP_PLANT=mc_20261024 timeout 300 "$PY" -m acp.train --data $DATA --run /tmp/s1v2_check --envs 4096 --iters $((10#$IT + 3)) --njmax 450 \
     --init "$R/init_from_s2_v1_$IT.pt" --latest-minutes 0.0001 --heartbeat /tmp/hb_s1v2.json > /tmp/s1v2_check.out 2>&1 \
   && [ "$(wc -l < /tmp/s1v2_check/metrics.jsonl)" -ge 3 ] && grep -q "^\[save\] latest.pt" /tmp/s1v2_check.out; then
  say "check passed: $(grep '^\[resume\]\|^\[env\]\|^\[save\]' /tmp/s1v2_check.out | tr '\n' ' ' | cut -c1-300) | $(tail -1 /tmp/s1v2_check/metrics.jsonl | cut -c1-200)"
  for i in $(seq 30); do gpu_free && break; sleep 2; done
  tmux new -d -s acp "X2_EXCLUDE_PAIRS=hip ACP_PLANT=mc_20261024 DATA=$DATA EXTRA='--init $R/init_from_s2_v1_$IT.pt' bash $T/code/supervise.sh $R 36000 $BUDGET"
  say "s1_v2 started (stage 1, oracle contacts, X2_EXCLUDE_PAIRS=hip, data $DATA, budget $BUDGET)"
  tmux new -d -s acp-trip "bash $T/code/ops/tripwire_s1v2.sh"
else
  say "check FAILED (see /tmp/s1v2_check.out): restoring the previous code, resuming s2_v1 on 2c"
  cp "$T"/code_prev/acp/*.py "$T/code/acp/"
  for i in $(seq 30); do gpu_free && break; sleep 2; done
  tmux new -d -s acp "ACP_PLANT=mc_20261024 DATA=cfg/stage2c_sets.json EXTRA='--stage 2 --init $R2/init_from_s1_v1_002250.pt --planner-arch attn --u-obs 200 --u-rew 1700 --n-ann 1500' bash $T/code/supervise.sh $R2 36000 6500"
fi
mv "$R2/STOP" "$TR/stop_flags/STOP_s2_v1_$IT" 2>/dev/null && echo "stop_flags/STOP_s2_v1_$IT: stop flag of the s1_v2 restart" >> "$TR/NOTES.txt"
