#!/bin/bash
# continue_v4.sh: once run 4's model_1000.pt exists, stop run 4 through its own supervisor, install staged.py.next (option t) and
# monitor.py.next (measured air time per touchdown), old files to _trash/<date>/, and continue the same run from its newest checkpoint
# with options a,c,d,f,r,s,t (stage 500,1500 kept, so every ramp continues), t weight 140. Falls back to run 4's own options
# (start_x2_parkour_v4.sh, which behaves the same with the new staged.py) on a Traceback, a trainer restart, the supervisor exiting,
# or fewer than 3 new iterations within 10 minutes. Nothing is deleted.
T=/mnt/filesystem-g4/tracks/hiking; X=$T/code/x2_port; D=$X/files_instinctlab/tasks/parkour/config/x2
NAME=x2_parkour_v4; RUN=$T/runs/$NAME; S=$RUN/_supervisor
TR=$T/_trash/$(date +%F); mkdir -p $TR
note() { echo "$(date '+%F %T') $*" >> $TR/NOTES.txt; }
for i in $(seq 1 120); do ls $RUN/*/model_1000.pt >/dev/null 2>&1 && break; sleep 10; done
ls $RUN/*/model_1000.pt >/dev/null 2>&1 || { echo "$(date +%T) continue_v4: no model_1000.pt after 20 min, nothing done" >> $S/launch.txt; exit 1; }
T0=$(date +%s)
touch $S/STOP
while tmux has -t hiking_x2 2>/dev/null; do sleep 3; done
[ -f $S/STOP ] && { DST=$TR/STOP_${NAME}_$(date +%H%M%S); mv $S/STOP $DST; note "$DST: stop flag of $NAME, used by continue_v4.sh"; }
H=$(date +%H%M%S)
mv $D/staged.py $TR/staged.py.run4_$H && mv $D/staged.py.next $D/staged.py && note "$TR/staged.py.run4_$H: staged.py of run 4 iterations 0-1000, replaced by the version with option t"
mv $D/monitor.py $TR/monitor.py.run4_$H && mv $D/monitor.py.next $D/monitor.py && note "$TR/monitor.py.run4_$H: monitor without the measured air time per touchdown"
bash $X/apply.sh > /dev/null
ENVV="HIKING_X2_OPTS=a,c,d,f,r,s,t HIKING_X2_STAGE=500,1500 HIKING_X2_CAM_RAND=1 HIKING_X2_ALIVE_EARLY=0.3 HIKING_X2_WAIT_EARLY=2 HIKING_X2_STYLE_EARLY=0.3 HIKING_X2_FALL_PENALTY=8 HIKING_X2_REF_START=0.5 HIKING_X2_AIR_TD_WEIGHT=140"
printf '#!/bin/bash\n# Resume %s with the options of its continuation at 1,000 (continue_v4.sh, option t added).\ntmux new -d -s hiking_x2 "%s %s %s/supervise_x2.py --task Instinct-Parkour-Fixed-Amp-X2-v0 --name %s --total 30000 --num_envs 8192 2>&1 | tee -a %s/runs/%s_supervisor_tmux.log"\n' \
  $NAME "$ENVV" '$HOME/hiking-env/bin/python' $X $NAME $T $NAME > $X/start_x2_parkour_v4b.sh
LOG0=$(stat -c %s $S/train.log 2>/dev/null || echo 0)
SUP0=$(grep -c ' start pid [0-9]* from iteration' $S/supervisor.log 2>/dev/null); SUP0=${SUP0:-0}
echo "continue from newest checkpoint: $ENVV started $(date '+%Y-%m-%d %H:%M:%S')" >> $S/launch.txt
bash $X/start_x2_parkour_v4b.sh
OK=0; WHY="fewer than 3 new iterations within 10 minutes"
for i in $(seq 1 60); do
  sleep 10
  NEW=$(tail -c +$((LOG0 + 1)) $S/train.log 2>/dev/null)
  if printf '%s' "$NEW" | grep -aq "Traceback"; then WHY="Traceback in train.log"; break; fi
  if [ $(printf '%s' "$NEW" | grep -ac "Learning iteration") -ge 3 ]; then OK=1; break; fi
  N=$(grep -c ' start pid [0-9]* from iteration' $S/supervisor.log 2>/dev/null); if [ "${N:-0}" -ge $((SUP0 + 2)) ]; then WHY="trainer restarted by its supervisor"; break; fi
  if ! tmux has -t hiking_x2 2>/dev/null; then WHY="supervisor exited"; break; fi
done
if [ $OK = 1 ]; then
  echo "$(date +%T) continuation with t is training, GPU idle about $(( $(date +%s) - T0 )) s" >> $S/launch.txt
else
  touch $S/STOP
  while tmux has -t hiking_x2 2>/dev/null; do sleep 3; done
  [ -f $S/STOP ] && { DST=$TR/STOP_${NAME}_$(date +%H%M%S); mv $S/STOP $DST; note "$DST: stop flag of the failed continuation"; }
  bash $X/start_x2_parkour_v4.sh
  echo "$(date +%T) continuation with t FAILED ($WHY; see train.log after byte $LOG0); run 4 resumed with its own options" >> $S/launch.txt
fi
