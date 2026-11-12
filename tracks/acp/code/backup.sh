#!/usr/bin/env bash
# Backup loop for the ACP track (user rule 2026-11-11): everything a restart needs stays on the shared filesystem.
#   tmux new -d -s acp-backup "bash /mnt/filesystem-g4/tracks/acp/code/backup.sh"
# Every 20 minutes: copy what the node keeps on its local disk (benchmark and diagnostic outputs, the environment's
# pinned package list), snapshot code and configs next to each run, prune checkpoints (newest two + every 1,000
# iterations + latest.pt), and write a manifest. Runs, logs, clips, labels and code are written to the shared
# filesystem directly, so they need no copy.
set -u
T=/mnt/filesystem-g4/tracks/acp
LOCAL=$HOME/acp_local            # anything a job writes locally goes here and is mirrored to $T/backup/local
mkdir -p "$LOCAL" "$T/backup/local" "$T/backup/tmp_outputs" "$T/env"
while true; do
  now=$(date -u +%FT%TZ)
  timeout 600 rsync -a "$LOCAL/" "$T/backup/local/" 2>/dev/null
  timeout 120 rsync -a --include='*.out' --include='hb.json' --exclude='*' /tmp/ "$T/backup/tmp_outputs/" 2>/dev/null
  [ -x "$HOME/acp_env/bin/pip" ] && timeout 120 "$HOME/acp_env/bin/pip" freeze > "$T/env/requirements.lock.txt.new" 2>/dev/null \
    && mv "$T/env/requirements.lock.txt.new" "$T/env/requirements.lock.txt"
  for run in "$T"/runs/*/; do
    [ -d "$run" ] || continue
    timeout 300 rsync -a --exclude='gear_sonic/data' --exclude='__pycache__' --exclude='scratch' --exclude='*.pt' "$T/code/" "$run/code_snapshot/" 2>/dev/null
    # prune: keep the newest two, every multiple of 1,000, and latest.pt
    mapfile -t cks < <(ls "$run"ckpt_*.pt 2>/dev/null | sort)
    n=${#cks[@]}
    for ((i = 0; i < n - 2; i++)); do
      it=$(basename "${cks[$i]}" .pt); it=$((10#${it#ckpt_}))
      if [ $((it % 1000)) -ne 0 ]; then   # nothing on the shared filesystem is deleted (rule of 2026-11-12): pruned checkpoints are moved
        tr="$T/_trash/$(date -u +%F)/pruned_checkpoints/$(basename "$run")"; mkdir -p "$tr"; mv "${cks[$i]}" "$tr/"
      fi
    done
  done
  {
    echo "ACP track backup manifest, written $now by backup.sh"
    echo "local-only items mirrored: $LOCAL -> backup/local ; /tmp/*.out -> backup/tmp_outputs"
    for run in "$T"/runs/*/; do echo "run $run: $(ls "$run" | grep -c 'ckpt_') checkpoints: $(ls "$run" | grep 'ckpt_' | tr '\n' ' ')"; done
    du -sh "$T"/code "$T"/data "$T"/runs "$T"/backup 2>/dev/null
  } > "$T/backup/MANIFEST.txt"
  sleep 1200
done
