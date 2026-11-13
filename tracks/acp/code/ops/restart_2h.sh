#!/usr/bin/env bash
# Data-2h restart (ACP agent, 2026-11-13, state/acp.md 17d): the ONE planned restart; afterwards none for 2,000 iterations.
# s2_v3 stops; run s2_v4 resumes the tracker from s2_v3's latest checkpoint, planner still frozen at 5,750, eta cap 0.8 kept.
# Carries: data 2h (cfg/stage2h_sets.json), ACP_HF_FIX=1 (chunked fields + foot box off the field + solref 0.01),
# simcheck patch (faults = truncations, sim_fault_share in the heartbeat), ACP_PAPER_DR=1 (Table S2: planner-target noise
# +-0.1 m, tracker scan U(+-0.1) per step, friction (robot U(0.3,1.2) + ground 1.0)/2, odometry drift). Gate rebased.
# Fire:  tmux new -d -s acp-restart "bash /mnt/filesystem-g4/tracks/acp/code/ops/restart_2h.sh"
set -u
T=/mnt/filesystem-g4/tracks/acp; R3=$T/runs/s2_v3; R=$T/runs/s2_v4; L=$R/restart.log; PY=$HOME/acp_env/bin/python
DATA=cfg/stage2h_sets.json; PF=$T/eval/pair_s2_v2_005750/tracker_ckpt_005750.pt; mkdir -p "$R"
say() { echo "$(date -u +%FT%TZ) $*" >> "$L"; }
gpu_free() { ! nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; }
[ -f "$T/code/cfg/armfix_stage2h_sets.pt" ] && [ -f "$T/code/cfg/labels_stage2h_sets.json.pt" ] || { say "2h data not complete (armfix / labels missing): restart NOT done"; exit 1; }
exec 8> /tmp/acp_eval.lock; flock -w 1800 8
touch "$R3/STOP"; for i in $(seq 120); do gpu_free && ! tmux has-session -t acp 2>/dev/null && break; sleep 2; done
TR=$T/_trash/$(date -u +%F); mkdir -p "$TR/stop_flags"
mv "$R3/STOP" "$TR/stop_flags/STOP_s2_v3_to_s2_v4" && echo "stop_flags/STOP_s2_v3_to_s2_v4: s2_v3 stopped for the data-2h restart" >> "$TR/NOTES.txt"
cp "$R3/latest.pt" "$R/init_from_s2_v3_latest.pt"
IT=$("$PY" -c "import torch,sys; print(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['it'])" "$R/init_from_s2_v3_latest.pt" 2>/dev/null | tail -1)
say "s2_v3 stopped at latest $IT; GPU free: $(gpu_free && echo yes || echo no)"
[ -d "$T/code_prev" ] && mv "$T/code_prev" "$TR/code_prev_$(date -u +%H%M%S)" && echo "code_prev_*: roll-back copy before 2h; superseded" >> "$TR/NOTES.txt"
mkdir -p "$T/code_prev"; cp -r "$T/code/acp" "$T/code_prev/"; cp "$T"/code_next/acp/*.py "$T/code/acp/"; cp "$T"/code_next/tools/{mk_stage2h,gate_stats,edit_2h}.py "$T/code/tools/"
EX="--stage 2 --init $R/init_from_s2_v3_latest.pt --planner-arch attn --u-obs 200 --u-rew 1700 --n-ann 3000 --eta-obs-max 0.8 --ramp-gate 2 --probe-envs 720 --gate-signal probe --ramp-gate-sets tp_t4x4u_y20_hs,tp_ramp_sh2_hs,x2_loco_flat_train_hold,x2_loco_blocks_train_hold --freeze-planner $PF --gate-backoff 1 --gate-rebase"
OLD="--stage 2 --init $R3/init_from_s2_v2_007000.pt --planner-arch attn --u-obs 200 --u-rew 1700 --n-ann 3000 --eta-obs-max 0.8 --ramp-gate 2 --probe-envs 720 --gate-signal probe --ramp-gate-sets tp_t4x4u_y20_hs,tp_ramp_sh2_hs,x2_loco_flat_train_hold,x2_loco_blocks_train_hold --freeze-planner $PF --gate-backoff 1 --gate-rebase"
ENVV="X2_EXCLUDE_PAIRS=hip ACP_PLANT=mc_20261024 ACP_HF_FIX=1 ACP_PAPER_DR=1"
cd "$T/code"; rm -rf /tmp/v4_check; mkdir -p /tmp/v4_check; rm -f /tmp/hb_v4.json
if env $ENVV timeout 600 "$PY" -m acp.train --data $DATA --run /tmp/v4_check --envs 4096 --iters $((IT + 3)) --njmax 450 --resume $EX --heartbeat /tmp/hb_v4.json > /tmp/v4_check.out 2>&1 \
   && grep -q "planner frozen" /tmp/v4_check.out && [ "$(wc -l < /tmp/v4_check/metrics.jsonl)" -ge 3 ]; then
  say "check passed: $(grep '^\[resume\]\|^\[stage 2\]\|^\[data\] sets' /tmp/v4_check.out | tr '\n' ' ' | cut -c1-400); hb $(python3 -c "import json;d=json.load(open('/tmp/hb_v4.json'));print('sim_fault_share', d.get('sim_fault_share'), 'finish', d.get('finish_frac'))" 2>&1)"
  for i in $(seq 30); do gpu_free && break; sleep 2; done
  tmux new -d -s acp "$ENVV DATA=$DATA EXTRA='$EX' bash $T/code/supervise.sh $R 36000 16000"
  say "s2_v4 started from s2_v3 $IT (data 2h, HF fix, simcheck, paper DR); fall-back watcher acp-watch2h"
  grep -q 'runs/s2_v4' $T/code/ops/export_every.sh || sed -i "s#\$T/runs/s2_v3; do#\$T/runs/s2_v3 \$T/runs/s2_v4; do#" $T/code/ops/export_every.sh
  tmux kill-session -t acp-export 2>/dev/null; tmux new -d -s acp-export "bash $T/code/ops/export_every.sh"
  tmux kill-session -t acp-watch2h 2>/dev/null; tmux new -d -s acp-watch2h "bash $T/code/ops/watch_2h.sh $IT"
else
  say "check FAILED (/tmp/v4_check.out, tail: $(tail -3 /tmp/v4_check.out | tr '\n' ' ' | cut -c1-300)): previous code back, s2_v3 resumes unchanged"
  cp "$T"/code_prev/acp/*.py "$T/code/acp/"
  for i in $(seq 30); do gpu_free && break; sleep 2; done
  tmux new -d -s acp "X2_EXCLUDE_PAIRS=hip ACP_PLANT=mc_20261024 DATA=cfg/stage2g_sets.json EXTRA='$OLD' bash $T/code/supervise.sh $R3 36000 14000"
fi
