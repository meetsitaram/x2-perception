#!/bin/bash
# crashguard_v4.sh (2026-11-12, after the 12:27 / 12:42 KeyError loop): (1) if the trainer exits with an error twice from the
# same checkpoint after this guard started, stop the supervisor, put back the pre-stop levels and the v4d staged.py, and resume
# v4d (start_x2_parkour_v4d.sh) instead of restarting blindly; (2) 80 iterations after the guard starts (after the level pin
# releases at about 62), write level and episode length to gates.txt and flag a drop below 0.8 x / 0.5 x of $1 / $2.
T=/mnt/filesystem-g4/tracks/hiking; X=$T/code/x2_port; D=$X/files_instinctlab/tasks/parkour/config/x2; RUN=$T/runs/x2_parkour_v4; S=$RUN/_supervisor; G=$S/gates.txt
L0=${1:-5.4}; E0=${2:-850}
SL=$(wc -l < $S/supervisor.log); B=$(stat -c %s $S/train.log); CHECKED=0
while true; do
  N=$(tail -n +$((SL + 1)) $S/supervisor.log | grep "run ended: exit 1" | awk '{print $NF}' | sort | uniq -c | awk '$1>=2{print $2}' | tail -1)
  if [ -n "$N" ]; then
    echo "$(date -u '+%F %T') CRASHGUARD: two error exits from checkpoint $N; falling back to v4d with pre-stop levels" >> $G
    touch $S/STOP; while tmux has -t hiking_x2 2>/dev/null; do sleep 3; done
    TR=$T/_trash/$(date +%F); mkdir -p $TR; mv $S/STOP $TR/STOP_crashguard_$(date +%H%M%S)
    cp $D/staged.py $TR/staged.py.crashguard_$(date +%H%M%S); cp ${FB_STAGED:-$D/staged.py.next3} $D/staged.py; bash $X/apply.sh >/dev/null
    [ -f $S/terrain_levels_prestop.pt ] && cp $S/terrain_levels_prestop.pt $S/terrain_levels.pt
    bash ${FB_START:-$X/start_x2_parkour_v4d.sh}; exit 0
  fi
  if [ $CHECKED = 0 ]; then
    NEW=$(tail -c +$((B + 1)) $S/train.log)
    if [ $(printf '%s' "$NEW" | grep -ac "Learning iteration") -ge 80 ]; then
      L1=$(printf '%s' "$NEW" | grep -a "Curriculum/terrain_levels" | tail -5 | awk '{s+=$NF;n++} END{print s/n}')
      E1=$(printf '%s' "$NEW" | grep -a "Mean episode length" | tail -5 | awk '{s+=$NF;n++} END{print s/n}')
      P=$(printf '%s' "$NEW" | grep -a "levels_pinned" | tail -1 | awk '{print $NF}')
      F=$(python3 -c "print('OK' if $L1 >= 0.8 * $L0 and $E1 >= 0.5 * $E0 else 'DROP (resume the hiking agent)')")
      echo "$(date -u '+%F %T') AFTER PIN: level $L1 (before $L0), episode $E1 (before $E0), pinned $P -> $F" >> $G; CHECKED=1
    fi
  fi
  sleep 30
done
