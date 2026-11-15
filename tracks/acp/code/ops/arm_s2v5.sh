#!/usr/bin/env bash
# s2_v5: stage 2 by Eq. 3 on the USER'S FIXED SCHEDULE, then stage 3, one run to its fixed end (state/acp.md 17j, 17k;
# docs/USER_RULES.md "ACP fixed stage lengths (2026-11-13)"; "do not stop the training runs", 2026-11-13).
# ARMED: waits for runs/s2_v5/GO. Then:
#   1. refuses unless runs/s2_v5/PREFLIGHT_OK exists
#   2. stops s2_v4 (STOP file, waits for a free GPU), copies s2_v4/latest.pt to s2_v5/init_from_s2_v4_latest.pt
#   3. anchors the schedule on that checkpoint, no judgement: IT0 = its iteration; eta_obs0 / eta_rew0 = the values s2_v4's own
#      schedule gives at IT0. From there both rise at 1 / 6,000 per iteration (N_ann 6,000, user) to 1.
#      Stage 2 ends 2,300 iterations after eta_rew reaches 1 (S3IT); stage 3 runs exactly 15,000 iterations (RUN_END = S3IT + 15,000).
#      All four iterations are written to runs/s2_v5/schedule.json.
#   4. starts s2_v5 to S3IT (supervise.sh, tmux acp), the fault watcher (watch_s2v5.sh, tmux acp-watch5), the automatic stage-3
#      switch (auto_stage3.sh, tmux acp-s3auto) and the CPU pair export loop (acp-export, never touches the trainer).
# Nothing here pauses training for evaluation; only the fault fall-backs in watch_s2v5.sh / auto_stage3.sh act on their own.
#   tmux new -d -s acp-arm5 "bash /mnt/filesystem-g4/tracks/acp/code/ops/arm_s2v5.sh"
set -u
T=/mnt/filesystem-g4/tracks/acp; R=$T/runs/s2_v5; R4=$T/runs/s2_v4; L=$R/restart.log; PY=$HOME/acp_env/bin/python
NANN=6000; TAIL=2300; S3LEN=15000; DATA=cfg/stage2h_sets.json
mkdir -p "$R"
say() { echo "$(date -u +%FT%TZ) [arm] $*" >> "$L"; }
gpu_free() { ! nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; }
say "armed (user schedule: N_ann $NANN, stage-2 tail $TAIL, stage 3 $S3LEN, automatic switch); waiting for $R/GO"
while [ ! -f "$R/GO" ]; do sleep 30; done
say "GO found ($(cat "$R/GO" 2>/dev/null | head -1))"
[ -f "$R/PREFLIGHT_OK" ] || { say "REFUSED: no PREFLIGHT_OK"; exit 1; }
[ -f "$R/latest.pt" ] && { say "REFUSED: s2_v5 already has latest.pt (started before?)"; exit 1; }
exec 8> /tmp/acp_eval.lock; flock -w 1800 8
TR=$T/_trash/$(date -u +%F); mkdir -p "$TR/stop_flags"
touch "$R4/STOP"; for i in $(seq 180); do gpu_free && ! tmux has-session -t acp 2>/dev/null && break; sleep 2; done
gpu_free || { say "REFUSED: GPU still busy after stopping s2_v4"; exit 1; }
mv "$R4/STOP" "$TR/stop_flags/STOP_s2_v4_for_s2_v5" && echo "stop_flags/STOP_s2_v4_for_s2_v5: s2_v4 stopped for the s2_v5 restart (17j)" >> "$TR/NOTES.txt"
cp "$R4/latest.pt" "$R/init_from_s2_v4_latest.pt"
python3 "$T/code/tools/gate_stats.py" "$R4/metrics.jsonl" 150 0 "$R/base_s2v4.json" > /dev/null   # watcher base: s2_v4 last 150 rows
read IT0 O0 W0 H S2 < <(CUDA_VISIBLE_DEVICES= "$PY" - "$R/init_from_s2_v4_latest.pt" <<'EOF'
import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
it, s2, h = int(ck["it"]), int(ck["s2_start"]), int(ck.get("eta_hold", 0))
u = it - s2 - h
print(it, round(min(0.8, max(0.0, (u - 200) / 3000)), 4), round(min(1.0, max(0.0, (u - 1700) / 3000)), 4), h, s2)
EOF
)
[ -n "${W0:-}" ] || { say "REFUSED: could not read the anchor from the checkpoint"; exit 1; }
read OBS1 REW1 < <(python3 -c "import math;print($IT0+math.ceil(round((1-$O0)*$NANN,6)), $IT0+math.ceil(round((1-$W0)*$NANN,6)))")
S3IT=$((REW1 + TAIL)); RUN_END=$((S3IT + S3LEN))
printf '{"anchor_it": %d, "eta_obs0": %s, "eta_rew0": %s, "n_ann": %d, "eta_hold_in_ckpt": %d, "s2_start": %d, "eta_obs_1_at": %d, "eta_rew_1_at": %d, "stage2_tail": %d, "stage3_start": %d, "stage3_iters": %d, "run_end": %d, "go_time": "%s"}\n' \
  "$IT0" "$O0" "$W0" "$NANN" "$H" "$S2" "$OBS1" "$REW1" "$TAIL" "$S3IT" "$S3LEN" "$RUN_END" "$(date -u +%FT%TZ)" > "$R/schedule.json"
say "schedule $(cat "$R/schedule.json")"
EX="--stage 2 --init $R/init_from_s2_v4_latest.pt --planner-arch attn --u-obs 200 --u-rew 1700 --n-ann $NANN --eta-anchor-it $IT0 --eta-obs0 $O0 --eta-rew0 $W0 --replay-balance set_clip --popt-reset --probe-envs 720 --gate-signal probe --ramp-gate 0"
ENVV="X2_EXCLUDE_PAIRS=hip ACP_PLANT=mc_20261024 ACP_HF_FIX=1 ACP_PAPER_DR=1"
echo "$ENVV DATA=$DATA EXTRA='$EX'" > "$R/launch_flags.txt"
mkdir -p "$R/code_snapshot"; cp -r "$T/code/acp" "$T/code/ops" "$T/code/supervise.sh" "$R/code_snapshot/" 2>/dev/null
tmux new -d -s acp "$ENVV DATA=$DATA EXTRA='$EX' bash $T/code/supervise.sh $R 36000 $S3IT"
say "s2_v5 STARTED from s2_v4 $IT0 to stage-2 end $S3IT (flags in launch_flags.txt)"
tmux kill-session -t acp-watch5 2>/dev/null; tmux new -d -s acp-watch5 "bash $T/code/ops/watch_s2v5.sh $IT0 $S3IT"
tmux kill-session -t acp-s3auto 2>/dev/null; tmux new -d -s acp-s3auto "bash $T/code/ops/auto_stage3.sh"
tmux kill-session -t acp-export 2>/dev/null; tmux new -d -s acp-export "bash $T/code/ops/export_every.sh"
say "watcher acp-watch5, stage-3 switch acp-s3auto and export loop acp-export (re)started"
