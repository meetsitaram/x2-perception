#!/bin/bash
# Stop run 2 and start the next run from scratch on the staged task, watched; falls back to resuming run 2 if the new task does
# not reach its third iteration. No process is killed except through each run's own supervisor (STOP file).
#   bash launch_next.sh <run name> <options, e.g. a,b,c> [stage "hold,end"] [camera randomisation 0|1]
T=/mnt/filesystem-g4/tracks/hiking; PY=/home/NODE_USER/hiking-env/bin/python
NAME=$1; OPTS=$2; STAGE=${3:-500,1500}; CAMRAND=${4:-0}
TASK=Instinct-Parkour-Staged-Amp-X2-v0
RUN=$T/runs/$NAME; mkdir -p $RUN/_supervisor
bash $T/code/x2_port/apply.sh > /dev/null
T0=$(date +%s)
touch $T/runs/x2_parkour_v2/_supervisor/STOP
while tmux has -t hiking_x2 2>/dev/null; do sleep 3; done
sleep 5
echo "task $TASK options $OPTS stage $STAGE camera_randomisation $CAMRAND started $(date '+%Y-%m-%d %H:%M:%S')" >> $RUN/_supervisor/launch.txt
tmux new -d -s hiking_x2 "HIKING_X2_OPTS=$OPTS HIKING_X2_STAGE=$STAGE HIKING_X2_CAM_RAND=$CAMRAND $PY $T/code/x2_port/supervise_x2.py --task $TASK --name $NAME --total 30000 --num_envs 8192 2>&1 | tee -a $T/runs/${NAME}_supervisor_tmux.log"
printf '#!/bin/bash\n# Start or resume %s with the options it was launched with.\ntmux new -d -s hiking_x2 "HIKING_X2_OPTS=%s HIKING_X2_STAGE=%s HIKING_X2_CAM_RAND=%s %s %s/code/x2_port/supervise_x2.py --task %s --name %s --total 30000 --num_envs 8192 2>&1 | tee -a %s/runs/%s_supervisor_tmux.log"\n' $NAME $OPTS $STAGE $CAMRAND $PY $T $TASK $NAME $T $NAME > $T/code/x2_port/start_$NAME.sh
OK=0
for i in $(seq 1 60); do
  sleep 10
  if grep -aq "Learning iteration 2/" $RUN/_supervisor/train.log 2>/dev/null; then OK=1; break; fi
  if [ "$(grep -c 'start pid' $RUN/_supervisor/supervisor.log 2>/dev/null)" -ge 2 ]; then break; fi
done
if [ $OK = 1 ]; then
  echo "$(date +%T) $NAME is training, GPU idle about $(( $(date +%s) - T0 )) s" >> $RUN/_supervisor/launch.txt
else
  touch $RUN/_supervisor/STOP
  while tmux has -t hiking_x2 2>/dev/null; do sleep 3; done
  bash $T/code/x2_port/start_x2_parkour_v2.sh
  echo "$(date +%T) $NAME FAILED to start (see train.log); run 2 resumed from its newest checkpoint" >> $RUN/_supervisor/launch.txt
fi
