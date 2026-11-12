#!/usr/bin/env bash
# Supervisor for one ACP training run: start, watch for stalls, auto-resume from latest.pt, heartbeat.
#   tmux new -d -s acp "bash /mnt/filesystem-g4/tracks/acp/code/supervise.sh <run_dir> <envs> <iters>"
# Stop for good: touch <run_dir>/STOP (the supervisor kills the trainer and exits).
set -u
RUN=$1; ENVS=${2:-12000}; ITERS=${3:-30000}
CODE=/mnt/filesystem-g4/tracks/acp/code
PY=$HOME/acp_env/bin/python
HB=/mnt/filesystem-g4/status/acp.json
STALL_S=900          # no new log line for this long = stalled (the first iteration compiles: allow START_S)
START_S=1800
mkdir -p "$RUN"
tmux has-session -t acp-backup 2>/dev/null || tmux new -d -s acp-backup "bash $CODE/backup.sh"   # backup loop (backup.sh)
LOG=$RUN/train.out
SUP=$RUN/supervisor.log
say() { echo "$(date +%FT%T) $*" | tee -a "$SUP"; }
restarts=0
while true; do
  [ -f "$RUN/STOP" ] && { say "STOP file found, exiting"; exit 0; }
  if [ -f "$RUN/ckpt_$(printf %06d "$ITERS").pt" ]; then say "run complete"; exit 0; fi
  if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then
    say "GPU is busy with another process, waiting"; sleep 60; continue
  fi
  say "starting trainer (restart $restarts)"
  cd "$CODE" && OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 "$PY" -m acp.train --data "${DATA:-cfg/stage1_sets.json}" --run "$RUN" \
      --envs "$ENVS" --iters "$ITERS" --njmax "${NJMAX:-450}" --resume ${EXTRA:-} >> "$LOG" 2>&1 &
  pid=$!
  started=$(date +%s)
  while kill -0 $pid 2>/dev/null; do
    sleep 30
    [ -f "$RUN/STOP" ] && { say "STOP file found, killing trainer $pid"; kill $pid; sleep 5; kill -9 $pid 2>/dev/null; exit 0; }
    now=$(date +%s)
    last=$(stat -c %Y "$RUN/log.jsonl" 2>/dev/null || echo 0)
    [ "$last" -lt "$started" ] && { limit=$START_S; ref=$started; } || { limit=$STALL_S; ref=$last; }
    if [ $((now - ref)) -gt $limit ]; then
      say "stalled ($((now - ref)) s without progress), killing trainer $pid"
      kill $pid; sleep 10; kill -9 $pid 2>/dev/null
      break
    fi
  done
  wait $pid 2>/dev/null; rc=$?
  say "trainer exited rc=$rc"
  restarts=$((restarts + 1))
  printf '{"track":"acp","state":"restarting","run":"%s","restarts":%d,"last_exit":%d,"unix":%d,"time":"%s"}\n' "$RUN" $restarts $rc "$(date +%s)" "$(date +%FT%T%z)" > "$HB.tmp" && mv "$HB.tmp" "$HB"
  if [ $restarts -ge 20 ]; then say "20 restarts, giving up"; exit 1; fi
  sleep 20
done
