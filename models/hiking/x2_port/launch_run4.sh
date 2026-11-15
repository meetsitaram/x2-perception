#!/bin/bash
# launch_run4.sh: stop run 3 through its own supervisor (after its model_500.pt exists, at most 15 minutes of waiting), start run 4
# (x2_parkour_v4, task Instinct-Parkour-Fixed-Amp-X2-v0, from scratch, options a,c,d,f,r,s), watched. Falls back to resuming run 3
# with its own options on a Traceback, a trainer restart by its supervisor, the supervisor exiting, or no "Learning iteration 2/"
# within 10 minutes. Used STOP flags are moved to _trash/<date>/. Nothing is deleted or killed except through a supervisor.
T=/mnt/filesystem-g4/tracks/hiking; X=$T/code/x2_port
OLD=x2_parkour_v3; NAME=x2_parkour_v4; TASK=Instinct-Parkour-Fixed-Amp-X2-v0
RUN=$T/runs/$NAME; S=$RUN/_supervisor; mkdir -p $S
TR=$T/_trash/$(date +%F); mkdir -p $TR
trash_stop() {
  if [ -f $1/STOP ]; then D=$TR/STOP_$2_$(date +%H%M%S); mv $1/STOP $D && echo "$(date '+%F %T') $D: stop flag of $2, used by launch_run4.sh" >> $TR/NOTES.txt; fi
}
for i in $(seq 1 90); do ls $T/runs/$OLD/*/model_500.pt >/dev/null 2>&1 && break; sleep 10; done
bash $X/apply.sh > /dev/null
T0=$(date +%s)
touch $T/runs/$OLD/_supervisor/STOP
while tmux has -t hiking_x2 2>/dev/null; do sleep 3; done
trash_stop $T/runs/$OLD/_supervisor $OLD
trash_stop $S $NAME
sleep 5
LOG0=$(stat -c %s $S/train.log 2>/dev/null || echo 0)
SUP0=$(grep -c ' start pid [0-9]* from iteration' $S/supervisor.log 2>/dev/null); SUP0=${SUP0:-0}
ENVV="HIKING_X2_OPTS=a,c,d,f,r,s HIKING_X2_STAGE=500,1500 HIKING_X2_CAM_RAND=1 HIKING_X2_ALIVE_EARLY=0.3 HIKING_X2_WAIT_EARLY=2 HIKING_X2_STYLE_EARLY=0.3 HIKING_X2_FALL_PENALTY=8 HIKING_X2_REF_START=0.5"
printf '#!/bin/bash\n# Start or resume %s (task %s) with the options it was launched with (launch_run4.sh).\ntmux new -d -s hiking_x2 "%s %s %s/supervise_x2.py --task %s --name %s --total 30000 --num_envs 8192 2>&1 | tee -a %s/runs/%s_supervisor_tmux.log"\n' \
  $NAME $TASK "$ENVV" '$HOME/hiking-env/bin/python' $X $TASK $NAME $T $NAME > $X/start_$NAME.sh
echo "task $TASK $ENVV started $(date '+%Y-%m-%d %H:%M:%S')" >> $S/launch.txt
bash $X/start_$NAME.sh
OK=0; WHY="no iteration 2 within 10 minutes"
for i in $(seq 1 60); do
  sleep 10
  NEW=$(tail -c +$((LOG0 + 1)) $S/train.log 2>/dev/null)
  if printf '%s' "$NEW" | grep -aq "Learning iteration 2/"; then OK=1; break; fi
  if printf '%s' "$NEW" | grep -aq "Traceback"; then WHY="Traceback in train.log"; break; fi
  N=$(grep -c ' start pid [0-9]* from iteration' $S/supervisor.log 2>/dev/null); if [ "${N:-0}" -ge $((SUP0 + 2)) ]; then WHY="trainer restarted by its supervisor"; break; fi
  if ! tmux has -t hiking_x2 2>/dev/null; then WHY="supervisor exited"; break; fi
done
if [ $OK = 1 ]; then
  echo "$(date +%T) $NAME is training, GPU idle about $(( $(date +%s) - T0 )) s" >> $S/launch.txt
else
  touch $S/STOP
  while tmux has -t hiking_x2 2>/dev/null; do sleep 3; done
  trash_stop $S $NAME
  bash $X/start_$OLD.sh
  echo "$(date +%T) $NAME FAILED to start ($WHY; see train.log after byte $LOG0); $OLD resumed from its newest checkpoint" >> $S/launch.txt
fi
