#!/usr/bin/env bash
# Stage 2 on the corrected body (ACP agent, 2026-11-12): stop s1_v2, start s2_v2 from its latest.pt with data 2f
# (2e + always-failing defect clips out + 4x4 weight 0.12 + 2 cm arm fix + 1.5 s stands at clip ends), the planner from
# $PLANNER, a slower ramp (n_ann 3000), a 20 % reference-contact floor (eta_obs <= 0.8) and the flat/blocks ramp gate (2 points).
#   tmux new -d -s acp-restart "PLANNER=<planner .pt> bash /mnt/filesystem-g4/tracks/acp/code/ops/restart_s2v2.sh"
set -u
T=/mnt/filesystem-g4/tracks/acp; R1=$T/runs/s1_v2; R=$T/runs/s2_v2; L=$R/restart.log; HB=/mnt/filesystem-g4/status/acp.json
PY=$HOME/acp_env/bin/python; DATA=cfg/stage2f_sets.json; BUDGET=${BUDGET:-14000}; PLANNER=${PLANNER:?}
mkdir -p "$R"
say() { echo "$(date -u +%FT%TZ) $*" >> "$L"; }
gpu_free() { ! nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; }
exec 8> /tmp/acp_eval.lock; flock -w 1800 8
touch "$R1/STOP"
for i in $(seq 90); do gpu_free && ! tmux has-session -t acp 2>/dev/null && break; sleep 2; done
say "s1_v2 stopped; GPU free: $(gpu_free && echo yes || echo no)"
tmux kill-session -t acp-trip 2>/dev/null; tmux kill-session -t acp-evallive2 2>/dev/null
tmux kill-session -t acp-export 2>/dev/null; tmux new -d -s acp-export "bash $T/code/ops/export_every.sh"   # picks up runs/s2_v2
IT=$("$PY" -c "import torch,sys; print(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['it'])" "$R1/latest.pt" 2>/dev/null | tail -1)
cp "$R1/latest.pt" "$R/init_from_s1_v2_$IT.pt"; cp "$PLANNER" "$R/planner_init.pt"; say "init: s1_v2 latest.pt at iteration $IT; planner $PLANNER"
printf '{"track":"acp","state":"restart: stage 2 on the corrected body from s1_v2 %s, data 2f (under 10 minutes)","run":"%s","unix":%d,"time":"%s"}\n' "$IT" "$R" "$(date +%s)" "$(date +%FT%T%z)" > "$HB"
TR=$T/_trash/$(date -u +%F); mkdir -p "$TR/stop_flags"
mv "$R1/STOP" "$TR/stop_flags/STOP_s1_v2_$IT" && echo "stop_flags/STOP_s1_v2_$IT: stop flag of the s2_v2 restart" >> "$TR/NOTES.txt"
[ -d "$T/code_prev" ] && mv "$T/code_prev" "$TR/code_prev_$(date -u +%H%M%S)" && echo "code_prev_*: roll-back copy from before an install; superseded" >> "$TR/NOTES.txt"
mkdir -p "$T/code_prev"; cp -r "$T/code/acp" "$T/code/tools" "$T/code_prev/"
cp "$T"/code_next/acp/*.py "$T/code/acp/"
cd "$T/code"
EXTRA="--stage 2 --init $R/init_from_s1_v2_$IT.pt --planner-arch attn --planner-init $R/planner_init.pt --u-obs 200 --u-rew 1700 --n-ann 3000 --eta-obs-max 0.8 --ramp-gate 2 --ramp-gate-sets x2_loco_flat_train_hold,x2_loco_blocks_train_hold"
rm -rf /tmp/s2v2_check; mkdir -p /tmp/s2v2_check
if X2_EXCLUDE_PAIRS=hip ACP_PLANT=mc_20261024 timeout 400 "$PY" -m acp.train --data $DATA --run /tmp/s2v2_check --envs 4096 --iters $((IT + 3)) --njmax 450 \
     $EXTRA --heartbeat /tmp/hb_s2v2.json > /tmp/s2v2_check.out 2>&1 && grep -q "^\[stage 2\]" /tmp/s2v2_check.out && [ "$(wc -l < /tmp/s2v2_check/metrics.jsonl)" -ge 3 ]; then
  say "check passed: $(grep '^\[resume\]\|^\[stage 2\]\|^\[env\]' /tmp/s2v2_check.out | tr '\n' ' ' | cut -c1-400) | $(tail -1 /tmp/s2v2_check/metrics.jsonl | cut -c1-200)"
  for i in $(seq 30); do gpu_free && break; sleep 2; done
  tmux new -d -s acp "X2_EXCLUDE_PAIRS=hip ACP_PLANT=mc_20261024 DATA=$DATA EXTRA='$EXTRA' bash $T/code/supervise.sh $R 36000 $BUDGET"
  say "s2_v2 started (stage 2, corrected body, data $DATA, budget $BUDGET)"
  B=$(( (IT / 1000 + 1) * 1000 + 500 ))
  tmux new -d -s acp-evallive2 "RUNNAME=s2_v2 bash $T/code/ops/live_every_run.sh $B $((B+1000)) $((B+2000)) $((B+3000)) $((B+4000)) $((B+5000))"
  say "live eval watcher: from $B every 1,000"
else
  say "check FAILED (see /tmp/s2v2_check.out): previous code back, s1_v2 resumes"
  cp "$T"/code_prev/acp/*.py "$T/code/acp/"
  for i in $(seq 30); do gpu_free && break; sleep 2; done
  tmux new -d -s acp "X2_EXCLUDE_PAIRS=hip ACP_PLANT=mc_20261024 DATA=cfg/stage2e_sets.json EXTRA='--init $R1/init_from_s2_v1_004000.pt' bash $T/code/supervise.sh $R1 36000 12000"
fi
