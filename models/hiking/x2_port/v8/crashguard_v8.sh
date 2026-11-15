#!/bin/bash
# crashguard_v8.sh = crashguard_v7.sh for v8 (restarts with start_x2_parkour_v8.sh). NEVER falls back to v5b.
# - Every trainer error exit ("run ended: exit 1") gets an ALERT line in gates.txt and ALERTS.txt with the last error / [GUARD]
#   lines of train.log. The supervisor itself resumes v7 from the newest checkpoint with the same v7 flags (and with the
#   instinct_rl finite guard, which the new process imports).
# - If the supervisor tmux (hiking_x2) is gone without a STOP from us, restart it with start_x2_parkour_v8.sh (same v7 flags).
# - Only after 5 error exits within 2 h: touch STOP (supervisor stops), ALERT "STOPPED", and exit. Resume the hiking agent.
T=/mnt/filesystem-g4/tracks/hiking; X=$T/code/x2_port; RUN=$T/runs/x2_parkour_v4; S=$RUN/_supervisor; G=$S/gates.txt; A=$S/ALERTS.txt
SL=$(wc -l < $S/supervisor.log); EXITS=()
echo "$(date -u '+%F %T') crashguard_v8 started (no fallback; alert per crash; stop after 5 crash exits in 2 h)" >> $G
while true; do
  NEW=$(tail -n +$((SL + 1)) $S/supervisor.log); SL=$(wc -l < $S/supervisor.log)
  while read -r line; do
    [ -z "$line" ] && continue
    NOW=$(date +%s); EXITS+=($NOW)
    ERR=$(tail -c 3000000 $S/train.log | grep -aE "^\[GUARD\]|RuntimeError|Error:" | tail -3 | cut -c1-300 | tr '\n' ' ')
    MSG="$(date -u '+%F %T') ALERT crash exit: $line | $ERR"
    echo "$MSG" >> $G; echo "$MSG" >> $A
  done < <(printf '%s\n' "$NEW" | grep "run ended: exit 1")
  KEEP=(); for t in "${EXITS[@]}"; do [ $(( $(date +%s) - t )) -lt 7200 ] && KEEP+=($t); done; EXITS=("${KEEP[@]}")
  if [ ${#EXITS[@]} -ge 5 ]; then
    MSG="$(date -u '+%F %T') ALERT STOPPED: ${#EXITS[@]} crash exits within 2 h; supervisor STOP written; resume the hiking agent"
    echo "$MSG" >> $G; echo "$MSG" >> $A; touch $S/STOP; exit 0
  fi
  if ! tmux has -t hiking_x2 2>/dev/null && [ ! -f $S/STOP ]; then
    sleep 60
    if ! tmux has -t hiking_x2 2>/dev/null && [ ! -f $S/STOP ]; then
      MSG="$(date -u '+%F %T') ALERT supervisor gone without STOP; restarting v7 with start_x2_parkour_v8.sh"
      echo "$MSG" >> $G; echo "$MSG" >> $A; bash $X/start_x2_parkour_v8.sh
    fi
  fi
  sleep 30
done
