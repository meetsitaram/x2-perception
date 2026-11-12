#!/bin/bash
# Stop run 2 and start the next run from scratch on the staged task, watched; falls back to resuming run 2 only on a real
# failure of the new run: a Traceback, a trainer restart by its supervisor, or no "Learning iteration 2/" within 10 minutes.
# No process is killed except through each run's own supervisor (STOP file in that run's own _supervisor folder); a STOP flag
# that has been used is moved to tracks/hiking/_trash/<date>/ (nothing under /mnt/filesystem-g4 is deleted).
#   bash launch_next.sh <run name> <options, e.g. a,b,c> [stage "hold,end"] [camera randomisation 0|1] [early is_alive scale, option a] [early dont_wait scale, option d] [early style coefficient scale, option s]
# 2026-11-12: the supervisor and training logs are appended across attempts, so every check reads only what was written after
# this launch (byte offsets / line counts taken before the start). The first version counted 'start pid' over the whole log
# and stopped run 3 twice during Isaac start-up.
T=/mnt/filesystem-g4/tracks/hiking; PY=$HOME/hiking-env/bin/python
NAME=$1; OPTS=$2; STAGE=${3:-500,1500}; CAMRAND=${4:-0}; ALIVE=${5:-0.3}; WAIT=${6:-4.0}; STYLE=${7:-0.3}
TASK=Instinct-Parkour-Staged-Amp-X2-v0
RUN=$T/runs/$NAME; S=$RUN/_supervisor; mkdir -p $S
TR=$T/_trash/$(date +%F); mkdir -p $TR
trash_stop() {  # $1 = supervisor folder, $2 = label
  if [ -f $1/STOP ]; then
    D=$TR/STOP_$2_$(date +%H%M%S); mv $1/STOP $D && echo "$(date '+%F %T') $D: empty stop flag of $2, used by launch_next.sh" >> $TR/NOTES.txt
  fi
}
bash $T/code/x2_port/apply.sh > /dev/null
T0=$(date +%s)
touch $T/runs/x2_parkour_v2/_supervisor/STOP
while tmux has -t hiking_x2 2>/dev/null; do sleep 3; done
trash_stop $T/runs/x2_parkour_v2/_supervisor x2_parkour_v2
trash_stop $S $NAME
sleep 5
LOG0=$(stat -c %s $S/train.log 2>/dev/null || echo 0)
SUP0=$(grep -c ' start pid [0-9]* from iteration' $S/supervisor.log 2>/dev/null); SUP0=${SUP0:-0}
echo "task $TASK options $OPTS stage $STAGE camera_randomisation $CAMRAND alive_early $ALIVE wait_early $WAIT style_early $STYLE started $(date '+%Y-%m-%d %H:%M:%S')" >> $S/launch.txt
tmux new -d -s hiking_x2 "HIKING_X2_OPTS=$OPTS HIKING_X2_STAGE=$STAGE HIKING_X2_CAM_RAND=$CAMRAND HIKING_X2_ALIVE_EARLY=$ALIVE HIKING_X2_WAIT_EARLY=$WAIT HIKING_X2_STYLE_EARLY=$STYLE $PY $T/code/x2_port/supervise_x2.py --task $TASK --name $NAME --total 30000 --num_envs 8192 2>&1 | tee -a $T/runs/${NAME}_supervisor_tmux.log"
printf '#!/bin/bash\n# Start or resume %s with the options it was launched with.\ntmux new -d -s hiking_x2 "HIKING_X2_OPTS=%s HIKING_X2_STAGE=%s HIKING_X2_CAM_RAND=%s HIKING_X2_ALIVE_EARLY=%s HIKING_X2_WAIT_EARLY=%s HIKING_X2_STYLE_EARLY=%s %s %s/code/x2_port/supervise_x2.py --task %s --name %s --total 30000 --num_envs 8192 2>&1 | tee -a %s/runs/%s_supervisor_tmux.log"\n' $NAME $OPTS $STAGE $CAMRAND $ALIVE $WAIT $STYLE '$HOME/hiking-env/bin/python' $T $TASK $NAME $T $NAME > $T/code/x2_port/start_$NAME.sh
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
  bash $T/code/x2_port/start_x2_parkour_v2.sh
  echo "$(date +%T) $NAME FAILED to start ($WHY; see train.log after byte $LOG0); run 2 resumed from its newest checkpoint" >> $S/launch.txt
fi
