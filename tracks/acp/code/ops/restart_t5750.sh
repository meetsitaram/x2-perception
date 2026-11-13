#!/usr/bin/env bash
# Fallback prepared 2026-11-12 (ACP agent): if the 7,000 frozen-planner pair still stalls at the turn-around, restart the TRACKER from
# 5,750 too: new run s2_v3 from pair_s2_v2_005750's checkpoint (tracker + planner), planner frozen at the same checkpoint, data 2g,
# probe gate, same ramp settings (s2_start 5,500 carried by the checkpoint).
#   tmux new -d -s acp-restart "bash /mnt/filesystem-g4/tracks/acp/code/ops/restart_t5750.sh"
set -u
T=/mnt/filesystem-g4/tracks/acp; R2=$T/runs/s2_v2; R=$T/runs/s2_v3; L=$R/restart.log; PY=$HOME/acp_env/bin/python; DATA=cfg/stage2g_sets.json
CK=$T/eval/pair_s2_v2_005750/tracker_ckpt_005750.pt; mkdir -p "$R"
say() { echo "$(date -u +%FT%TZ) $*" >> "$L"; }
gpu_free() { ! nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; }
exec 8> /tmp/acp_eval.lock; flock -w 1800 8
cp "$CK" "$R/init_from_s2_v2_005750.pt"
touch "$R2/STOP"; for i in $(seq 90); do gpu_free && ! tmux has-session -t acp 2>/dev/null && break; sleep 2; done
TR=$T/_trash/$(date -u +%F); mkdir -p "$TR/stop_flags"; mv "$R2/STOP" "$TR/stop_flags/STOP_s2_v2_to_s2_v3" && echo "stop_flags/STOP_s2_v2_to_s2_v3: s2_v2 stopped for s2_v3" >> "$TR/NOTES.txt"
say "s2_v2 stopped; GPU free: $(gpu_free && echo yes || echo no)"
EX="--stage 2 --init $R/init_from_s2_v2_005750.pt --planner-arch attn --u-obs 200 --u-rew 1700 --n-ann 3000 --eta-obs-max 0.8 --ramp-gate 2 --probe-envs 720 --gate-signal probe --ramp-gate-sets tp_t4x4u_y20_hs,tp_ramp_sh2_hs,x2_loco_flat_train_hold,x2_loco_blocks_train_hold --freeze-planner $R/init_from_s2_v2_005750.pt"
cd "$T/code"; rm -rf /tmp/v3_check; mkdir -p /tmp/v3_check
if X2_EXCLUDE_PAIRS=hip ACP_PLANT=mc_20261024 timeout 400 "$PY" -m acp.train --data $DATA --run /tmp/v3_check --envs 4096 --iters 5753 --njmax 450 $EX --heartbeat /tmp/hb_v3.json > /tmp/v3_check.out 2>&1 \
   && grep -q "planner frozen" /tmp/v3_check.out && [ "$(wc -l < /tmp/v3_check/metrics.jsonl)" -ge 3 ]; then
  say "check passed: $(grep '^\[resume\]\|^\[stage 2\]' /tmp/v3_check.out | tr '\n' ' ' | cut -c1-300)"
  for i in $(seq 30); do gpu_free && break; sleep 2; done
  tmux new -d -s acp "X2_EXCLUDE_PAIRS=hip ACP_PLANT=mc_20261024 DATA=$DATA EXTRA='$EX' bash $T/code/supervise.sh $R 36000 14000"
  say "s2_v3 started from 5,750 (tracker and planner), planner frozen"
  sed -i "s#\$T/runs/s2_v2; do#\$T/runs/s2_v2 \$T/runs/s2_v3; do#" $T/code/ops/export_every.sh; tmux kill-session -t acp-export; tmux new -d -s acp-export "bash $T/code/ops/export_every.sh"
else
  say "check FAILED (/tmp/v3_check.out): s2_v2 resumes"
  tmux new -d -s acp "X2_EXCLUDE_PAIRS=hip ACP_PLANT=mc_20261024 DATA=$DATA EXTRA='--stage 2 --init $R2/init_from_s1_v2_5500.pt --planner-arch attn --planner-init $R2/planner_init.pt --u-obs 200 --u-rew 1700 --n-ann 3000 --eta-obs-max 0.8 --ramp-gate 2 --probe-envs 720 --gate-signal probe --ramp-gate-sets tp_t4x4u_y20_hs,tp_ramp_sh2_hs,x2_loco_flat_train_hold,x2_loco_blocks_train_hold --freeze-planner $CK' bash $T/code/supervise.sh $R2 36000 14000"
fi
