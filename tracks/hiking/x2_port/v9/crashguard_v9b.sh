#!/bin/bash
# crashguard_v9b (hiking track 2026-11-15, FINDINGS H7b) = crashguard_v9 plus:
# - a copy of the level keeper's terrain_levels.pt for every new checkpoint (_supervisor/levels_hist/tl_<iteration>.pt);
# - a collapse watchdog (v9/guard2/collapse_check.py on metrics.jsonl): on a collapse it ALERTs, writes STOP, moves every
#   checkpoint newer than (onset - 25) to runs/_collapse_backups/ (outside the run folder; nothing deleted), restores the
#   levels of the newest remaining checkpoint and restarts v9 with start_x2_parkour_v9.sh (same flags). A second collapse
#   within 3 h: ALERT and STOP only (resume the hiking agent).
# Unchanged from v9: an ALERT per crash exit; restart if the supervisor is gone without STOP; STOP after 5 crash exits in 2 h.
T=/mnt/filesystem-g4/tracks/hiking; X=$T/code/x2_port; RUN=$T/runs/x2_parkour_v4; S=$RUN/_supervisor; G=$S/gates.txt; A=$S/ALERTS.txt
PY=$HOME/hiking-env/bin/python; CC=$X/v9/guard2/collapse_check.py; mkdir -p $S/levels_hist
SL=$(wc -l < $S/supervisor.log); EXITS=(); LASTC=""; LASTRB=0; N=0
newest() { ls $RUN/2*/model_*.pt 2>/dev/null | sed 's/.*model_\([0-9]*\)\.pt/\1/' | sort -n | tail -1; }
alert() { local m="$(date -u '+%F %T') ALERT $*"; echo "$m" >> $G; echo "$m" >> $A; }
echo "$(date -u '+%F %T') crashguard_v9b started (v9 + levels per checkpoint + collapse watchdog with rollback)" >> $G
while true; do
  NEW=$(tail -n +$((SL + 1)) $S/supervisor.log); SL=$(wc -l < $S/supervisor.log)
  while read -r line; do
    [ -z "$line" ] && continue
    NOW=$(date +%s); EXITS+=($NOW)
    ERR=$(tail -c 3000000 $S/train.log | grep -aE "^\[GUARD\]|RuntimeError|Error:" | tail -3 | cut -c1-300 | tr '\n' ' ')
    alert "crash exit: $line | $ERR"
  done < <(printf '%s\n' "$NEW" | grep "run ended: exit 1")
  KEEP=(); for t in "${EXITS[@]}"; do [ $(( $(date +%s) - t )) -lt 7200 ] && KEEP+=($t); done; EXITS=("${KEEP[@]}")
  if [ ${#EXITS[@]} -ge 5 ]; then
    alert "STOPPED: ${#EXITS[@]} crash exits within 2 h; supervisor STOP written; resume the hiking agent"; touch $S/STOP; exit 0
  fi
  C=$(newest)
  if [ -n "$C" ] && [ "$C" != "$LASTC" ]; then cp $S/terrain_levels.pt $S/levels_hist/tl_$C.pt 2>/dev/null; LASTC=$C; fi
  N=$((N + 1))
  if [ $((N % 2)) -eq 0 ] && tmux has -t hiking_x2 2>/dev/null && [ ! -f $S/STOP ]; then
    OUT=$($PY $CC $RUN/metrics.jsonl 2>&1); RC=$?
    if [ $RC -eq 2 ]; then
      if [ $LASTRB -gt 0 ] && [ $(( $(date +%s) - LASTRB )) -lt 10800 ]; then
        alert "collapse again within 3 h of an automatic rollback ($OUT); STOP written, no further rollback; resume the hiking agent"
        touch $S/STOP; exit 0
      fi
      ONSET=$(echo "$OUT" | awk '{print $2}')
      alert "collapse: $OUT; STOP, rollback to the newest checkpoint <= $((ONSET - 25))"
      touch $S/STOP
      for i in $(seq 200); do tmux has -t hiking_x2 2>/dev/null || break; sleep 3; done
      BK=$T/runs/_collapse_backups/x2_parkour_v4_$(date -u +%m%d_%H%M); mkdir -p $BK
      for f in $(ls $RUN/2*/model_*.pt); do
        n=$(echo $f | sed 's/.*model_\([0-9]*\)\.pt/\1/'); [ $n -gt $((ONSET - 25)) ] && mv $f $BK/
      done
      cp $S/terrain_levels.pt $BK/terrain_levels_at_rollback.pt
      K=$(newest)
      LV=$(ls $S/levels_hist/tl_*.pt 2>/dev/null | sed 's/.*tl_\([0-9]*\)\.pt/\1/' | sort -n | awk -v k=$K '$1<=k' | tail -1)
      if [ -n "$LV" ]; then cp $S/levels_hist/tl_$LV.pt $S/terrain_levels.pt; else alert "no levels snapshot <= $K; levels file left as is"; fi
      TR=$T/_trash/$(date +%F); mkdir -p $TR; DST=$TR/STOP_x2_parkour_v4_$(date +%H%M%S); mv $S/STOP $DST
      echo "$(date '+%F %T') $DST: stop flag of the automatic collapse rollback (crashguard_v9b)" >> $TR/NOTES.txt
      echo "$(date -u '+%F %T') v9 automatic collapse rollback: checkpoints > $((ONSET - 25)) moved to $BK; levels tl_$LV; restart from model_$K" >> $S/launch.txt
      bash $X/start_x2_parkour_v9.sh; LASTRB=$(date +%s); LASTC=$K; sleep 60; SL=$(wc -l < $S/supervisor.log)
    fi
  fi
  if ! tmux has -t hiking_x2 2>/dev/null && [ ! -f $S/STOP ]; then
    sleep 60
    if ! tmux has -t hiking_x2 2>/dev/null && [ ! -f $S/STOP ]; then
      alert "supervisor gone without STOP; restarting v9 with start_x2_parkour_v9.sh"; bash $X/start_x2_parkour_v9.sh
    fi
  fi
  sleep 30
done
