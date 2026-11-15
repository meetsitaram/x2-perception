#!/usr/bin/env bash
# s2_v2 restart (ACP agent, 2026-11-12): add 720 planner-only probe envs and gate the ramp on their per-set finish
# (4x4, ramps, flat, blocks); everything else as before. Resumes s2_v2 from latest.pt.
set -u
T=/mnt/filesystem-g4/tracks/acp; R=$T/runs/s2_v2; L=$R/restart.log; PY=$HOME/acp_env/bin/python; DATA=cfg/stage2g_sets.json
say() { echo "$(date -u +%FT%TZ) $*" >> "$L"; }
gpu_free() { ! nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; }
exec 8> /tmp/acp_eval.lock; flock -w 1800 8
touch "$R/STOP"
for i in $(seq 90); do gpu_free && ! tmux has-session -t acp 2>/dev/null && break; sleep 2; done
TR=$T/_trash/$(date -u +%F); mkdir -p "$TR/stop_flags"
IT=$("$PY" -c "import torch,sys; print(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['it'])" "$R/latest.pt" 2>/dev/null | tail -1)
mv "$R/STOP" "$TR/stop_flags/STOP_s2_v2_freeze_$IT" && echo "stop_flags/STOP_s2_v2_probe_$IT: stop flag of the probe-gate restart" >> "$TR/NOTES.txt"
say "freeze restart: s2_v2 stopped at latest $IT; GPU free: $(gpu_free && echo yes || echo no)"
[ -d "$T/code_prev" ] && mv "$T/code_prev" "$TR/code_prev_$(date -u +%H%M%S)" && echo "code_prev_*: roll-back copy; superseded" >> "$TR/NOTES.txt"
mkdir -p "$T/code_prev"; cp -r "$T/code/acp" "$T/code_prev/"; cp "$T"/code_next/acp/*.py "$T/code/acp/"
cd "$T/code"
OLD="--stage 2 --init $R/init_from_s1_v2_5500.pt --planner-arch attn --planner-init $R/planner_init.pt --u-obs 200 --u-rew 1700 --n-ann 3000 --eta-obs-max 0.8 --ramp-gate 2 --probe-envs 720 --gate-signal probe --ramp-gate-sets tp_t4x4u_y20_hs,tp_ramp_sh2_hs,x2_loco_flat_train_hold,x2_loco_blocks_train_hold"
NEW="--stage 2 --init $R/init_from_s1_v2_5500.pt --planner-arch attn --planner-init $R/planner_init.pt --u-obs 200 --u-rew 1700 --n-ann 3000 --eta-obs-max 0.8 --ramp-gate 2 --probe-envs 720 --gate-signal probe --ramp-gate-sets tp_t4x4u_y20_hs,tp_ramp_sh2_hs,x2_loco_flat_train_hold,x2_loco_blocks_train_hold --freeze-planner /mnt/filesystem-g4/tracks/acp/eval/pair_s2_v2_005750/tracker_ckpt_005750.pt"
rm -rf /tmp/probe_check; mkdir -p /tmp/probe_check; cp "$R/latest.pt" /tmp/probe_check/latest.pt
if X2_EXCLUDE_PAIRS=hip ACP_PLANT=mc_20261024 timeout 400 "$PY" -m acp.train --data $DATA --run /tmp/probe_check --envs 4096 --iters $((IT + 3)) --njmax 450 --resume $NEW --heartbeat /tmp/hb_probe.json > /tmp/probe_check.out 2>&1 \
   && grep -q "planner frozen" /tmp/probe_check.out && [ "$(wc -l < /tmp/probe_check/metrics.jsonl)" -ge 3 ] && grep -q "probe/finish" /tmp/probe_check/metrics.jsonl; then
  say "probe check passed: $(grep '^\[resume\]\|^\[stage 2\]' /tmp/probe_check.out | tr '\n' ' ' | cut -c1-300)"; EX=$NEW
else
  say "probe check FAILED (/tmp/probe_check.out): previous code back, s2_v2 resumes unchanged"; cp "$T"/code_prev/acp/*.py "$T/code/acp/"; EX=$OLD
fi
for i in $(seq 30); do gpu_free && break; sleep 2; done
tmux new -d -s acp "X2_EXCLUDE_PAIRS=hip ACP_PLANT=mc_20261024 DATA=$DATA EXTRA='$EX' bash $T/code/supervise.sh $R 36000 14000"
say "s2_v2 resumed ($( [ "$EX" = "$NEW" ] && echo probe gate || echo unchanged ))"
