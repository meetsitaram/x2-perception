#!/usr/bin/env bash
# One stop: end stage 1 (run s1_v1) and start stage 2 (run s2_v1) from the given stage-1 checkpoint.
#   bash to_stage2.sh 2250
set -u
IT=$(printf %06d "$1")
T=/mnt/filesystem-g4/tracks/acp; R1=$T/runs/s1_v1; R2=$T/runs/s2_v1; L=$R1/switch.log; HB=/mnt/filesystem-g4/status/acp.json
say() { echo "$(date -u +%FT%TZ) $*" >> "$L"; }
gpu_free() { ! nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; }
touch "$R1/STOP"
for i in $(seq 60); do gpu_free && ! tmux has-session -t acp 2>/dev/null && break; sleep 2; done
say "stage 1 stopped for the switch to stage 2 (from ckpt_$IT.pt); GPU free: $(gpu_free && echo yes || echo no)"
printf '{"track":"acp","state":"stage switch: stage 1 ended at checkpoint %s, stage 2 (planner in the loop) starting; under 10 minutes","run":"%s","newest_checkpoint":"%s","unix":%d,"time":"%s"}\n' "$IT" "$R2" "$R1/ckpt_$IT.pt" "$(date +%s)" "$(date +%FT%T%z)" > "$HB"
TR=$T/_trash/$(date -u +%F); mkdir -p "$TR"
[ -d "$T/code_prev" ] && mv "$T/code_prev" "$TR/code_prev_$(date -u +%H%M%S)" && echo "code_prev: roll-back copy of code/acp and code/tools from before the previous install; superseded" >> "$TR/NOTES.txt"
mkdir -p "$T/code_prev"; cp -r "$T/code/acp" "$T/code/tools" "$T/code/supervise.sh" "$T/code/backup.sh" "$T/code_prev/"
cp "$T"/code_next/acp/*.py "$T/code/acp/"; cp "$T"/code_next/tools/* "$T/code/tools/"; cp "$T/code_next/supervise.sh" "$T/code_next/backup.sh" "$T/code/"
mkdir -p "$R2"; cp "$R1/ckpt_$IT.pt" "$R2/init_from_s1_v1_$IT.pt"
tmux new -d -s acp "ACP_PLANT=mc_20261024 DATA=cfg/stage2_sets.json EXTRA='--stage 2 --init $R2/init_from_s1_v1_$IT.pt --planner-arch attn --u-obs 200 --u-rew 1700 --n-ann 1500' bash $T/code/supervise.sh $R2 36000 6500"
tmux kill-session -t acp-backup 2>/dev/null; tmux new -d -s acp-backup "bash $T/code/backup.sh"
say "stage 2 supervisor started: run $R2, plant mc_20261024, data cfg/stage2_sets.json, eta_obs from 200 over 1500, eta_rew from 1700 over 1500"
