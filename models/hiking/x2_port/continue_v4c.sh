#!/bin/bash
# continue_v4c.sh: once run 4's model_1750.pt exists, stop run 4 through its supervisor and continue the same run from it with
# options k (terrain levels kept across restarts; this time restored from the per-type means logged at the newest iteration,
# afterwards from the saved file) and y (turn in place, 15% of command periods, dont_turn_wait -0.5), and the walking-command
# floor back to the paper's 0.45 m/s (HIKING_X2_VX_MIN; run 4 used 0.6). Everything else as the t continuation (t weight 140,
# save interval 125). Falls back to the t continuation (start_x2_parkour_v4b.sh with the old staged.py) on a Traceback, a trainer
# restart, the supervisor exiting, or fewer than 3 new iterations in 10 min. Nothing is deleted (old files to _trash/<date>/).
WAIT_FOR=${1:-1750}
T=/mnt/filesystem-g4/tracks/hiking; X=$T/code/x2_port; D=$X/files_instinctlab/tasks/parkour/config/x2
NAME=x2_parkour_v4; RUN=$T/runs/$NAME; S=$RUN/_supervisor
TR=$T/_trash/$(date +%F); mkdir -p $TR
note() { echo "$(date '+%F %T') $*" >> $TR/NOTES.txt; }
until ls $RUN/*/model_${WAIT_FOR}.pt >/dev/null 2>&1; do sleep 20; done
sleep 30
T0=$(date +%s)
touch $S/STOP
while tmux has -t hiking_x2 2>/dev/null; do sleep 3; done
[ -f $S/STOP ] && { DST=$TR/STOP_${NAME}_$(date +%H%M%S); mv $S/STOP $DST; note "$DST: stop flag of $NAME, used by continue_v4c.sh"; }
# per-type mean levels at the newest logged iteration (10-iteration mean)
MEANS=$(python3 - $RUN/metrics.jsonl <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
last = rows[-1]["iteration"]; rows = [r for r in rows if r["iteration"] > last - 10]
keys = sorted({k for r in rows for k in r if k.startswith("Curriculum/walk_monitor/level/")})
print(",".join("%s:%.3f" % (k.split("/")[-1], sum(r[k] for r in rows if k in r) / max(1, sum(k in r for r in rows))) for k in keys))
PY
)
H=$(date +%H%M%S)
cp $D/staged.py $TR/staged.py.run4t_$H && note "$TR/staged.py.run4t_$H: staged.py of the t continuation (iterations 1000-$WAIT_FOR), replaced by staged.py.next2 (options k, y, HIKING_X2_VX_MIN)"
cp $D/staged.py.next2 $D/staged.py
bash $X/apply.sh > /dev/null
ENVV="HIKING_X2_OPTS=a,c,d,f,r,s,t,k,y HIKING_X2_STAGE=500,1500 HIKING_X2_CAM_RAND=1 HIKING_X2_ALIVE_EARLY=0.3 HIKING_X2_WAIT_EARLY=2 HIKING_X2_STYLE_EARLY=0.3 HIKING_X2_FALL_PENALTY=8 HIKING_X2_REF_START=0.5 HIKING_X2_AIR_TD_WEIGHT=140 HIKING_X2_VX_MIN=0.45 HIKING_X2_TURN_SHARE=0.15 HIKING_X2_LEVELS_OUT=$S/terrain_levels.pt HIKING_X2_LEVELS_IN=$S/terrain_levels.pt HIKING_X2_LEVELS_MEANS=$MEANS"
printf '#!/bin/bash\n# Resume %s with the options of its continuation from %s (continue_v4c.sh: k, y, vx floor 0.45 added).\ntmux new -d -s hiking_x2 "%s %s %s/supervise_x2.py --task Instinct-Parkour-Fixed-Amp-X2-v0 --name %s --total 30000 --num_envs 8192 --save_interval 125 2>&1 | tee -a %s/runs/%s_supervisor_tmux.log"\n' \
  $NAME $WAIT_FOR "$ENVV" '$HOME/hiking-env/bin/python' $X $NAME $T $NAME > $X/start_x2_parkour_v4c.sh
LOG0=$(stat -c %s $S/train.log 2>/dev/null || echo 0)
SUP0=$(grep -c ' start pid [0-9]* from iteration' $S/supervisor.log 2>/dev/null); SUP0=${SUP0:-0}
echo "continue from newest checkpoint (v4c): $ENVV started $(date '+%Y-%m-%d %H:%M:%S')" >> $S/launch.txt
bash $X/start_x2_parkour_v4c.sh
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
  R=$(tail -c +$((LOG0 + 1)) $S/train.log | grep -a "level_keeper\]" | head -2 | tr '\n' ' ')
  echo "$(date +%T) continuation v4c (k, y, vx 0.45) is training, GPU idle about $(( $(date +%s) - T0 )) s; $R" >> $S/launch.txt
else
  touch $S/STOP
  while tmux has -t hiking_x2 2>/dev/null; do sleep 3; done
  [ -f $S/STOP ] && { DST=$TR/STOP_${NAME}_$(date +%H%M%S); mv $S/STOP $DST; note "$DST: stop flag of the failed v4c continuation"; }
  cp $TR/staged.py.run4t_$H $D/staged.py && bash $X/apply.sh > /dev/null
  bash $X/start_x2_parkour_v4b.sh
  echo "$(date +%T) continuation v4c FAILED ($WHY; see train.log after byte $LOG0); t continuation resumed (start_x2_parkour_v4b.sh)" >> $S/launch.txt
fi
