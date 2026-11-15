#!/usr/bin/env bash
# Re-arm s2_v4 once after the gate-B fall-back of 07:17 (ACP agent, 2026-11-13, state/acp.md 17f). Resumes s2_v4 from its own
# latest.pt (10,750), same content as restart_2h.sh (data 2h, ACP_HF_FIX, simcheck, ACP_PAPER_DR); watcher gate C on base_2h.json.
set -u
T=/mnt/filesystem-g4/tracks/acp; R3=$T/runs/s2_v3; R=$T/runs/s2_v4; L=$R/restart.log
DATA=cfg/stage2h_sets.json; PF=$T/eval/pair_s2_v2_005750/tracker_ckpt_005750.pt
say() { echo "$(date -u +%FT%TZ) $*" >> "$L"; }
gpu_free() { ! nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; }
exec 8> /tmp/acp_eval.lock; flock -w 1800 8
TR=$T/_trash/$(date -u +%F); mkdir -p "$TR/stop_flags"
[ -f "$R/FALLBACK" ] && mv "$R/FALLBACK" "$TR/stop_flags/FALLBACK_s2_v4_gateB_0717" && echo "stop_flags/FALLBACK_s2_v4_gateB_0717: gate-B fall-back marker, superseded by the re-arm (17f)" >> "$TR/NOTES.txt"
cp "$R/win_B.json" "$R/base_2h.json"
touch "$R3/STOP"; for i in $(seq 120); do gpu_free && ! tmux has-session -t acp 2>/dev/null && break; sleep 2; done
mv "$R3/STOP" "$TR/stop_flags/STOP_s2_v3_rearm_s2_v4" && echo "stop_flags/STOP_s2_v3_rearm_s2_v4: s2_v3 stopped for the s2_v4 re-arm" >> "$TR/NOTES.txt"
say "s2_v3 stopped at $(tail -1 $R3/metrics.jsonl | cut -c1-20); GPU free: $(gpu_free && echo yes || echo no)"
EX="--stage 2 --init $R/init_from_s2_v3_latest.pt --planner-arch attn --u-obs 200 --u-rew 1700 --n-ann 3000 --eta-obs-max 0.8 --ramp-gate 2 --probe-envs 720 --gate-signal probe --ramp-gate-sets tp_t4x4u_y20_hs,tp_ramp_sh2_hs,x2_loco_flat_train_hold,x2_loco_blocks_train_hold --freeze-planner $PF --gate-backoff 1 --gate-rebase"
ENVV="X2_EXCLUDE_PAIRS=hip ACP_PLANT=mc_20261024 ACP_HF_FIX=1 ACP_PAPER_DR=1"
tmux new -d -s acp "$ENVV DATA=$DATA EXTRA='$EX' bash $T/code/supervise.sh $R 36000 16000"
say "s2_v4 RE-ARMED from its own latest.pt (10750; data 2h, HF fix, simcheck, paper DR); watcher gate C at +400 on base_2h.json (17f)"
tmux kill-session -t acp-watch2h 2>/dev/null; tmux new -d -s acp-watch2h "REARM=1 BASE=$R/base_2h.json PERSET=0.04 bash $T/code/ops/watch_2h.sh 10750"
