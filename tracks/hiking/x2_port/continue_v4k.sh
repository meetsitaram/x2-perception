#!/bin/bash
# continue_v4k.sh (2026-11-12 22:50, from continue_v4k.sh). v4k = v4k plus, in ONE restart (coordinator: the last planned restart
# for at least 2,000 iterations):
#  v  slow.py: forward command capped at U(0.1, 0.45) m/s for 25% of robots, |yaw| capped at U(0.1, 0.4) rad/s for 40% (eval 4750:
#     legs 3-4 stalled at stand-turn commands 0.1-0.5 rad/s; turn_in_place failures ended 0.23-0.39 m off the station after drifting);
#     `turn_effort` (any turn command > 0.1 rad/s, turning slower than half) replaces `dont_turn_wait` (> 0.4 only); `turn_drift` -2.0
#  backward: HIKING_X2_VX_BACK=0 (random-velocity columns no longer command backward; user: never step back)
#  terrain: tracks/hiking/data/extended_v4k = extended_v4j + extended_v1_1_prop_routes (1,116 routes on beams, fields, corridors,
#     wedges, blocks; spawn-into-props filter re-applied)
#  motion: tracks/hiking/data/motion_v4 = v2 + sideways (t4x4u_y20 + mir, 18) + turning stairs v1_1 + mir (24) + 2x2 v1_1 + mir (18)
#     + driveway v1_1 + mir (12): 827 s, 0 joint-limit frames
# CPU pre-flight (precheck_v4k.py) before v4j is stopped; construction check (routes >= 100, wall_contact_graded and turn_effort in
# the reward table); trial 31 iterations (level >= 0.8 x, episode >= 0.4 x, turn rate >= 0.5 x: slow turns lower the mean rate);
# fall-back v4k (files restored, pre-stop levels); then a crash guard.
WAIT_FOR=${1:-6750}
T=/mnt/filesystem-g4/tracks/hiking; X=$T/code/x2_port; D=$X/files_instinctlab/tasks/parkour/config/x2
NAME=x2_parkour_v4; RUN=$T/runs/$NAME; S=$RUN/_supervisor
TR=$T/_trash/$(date +%F); mkdir -p $TR
note() { echo "$(date '+%F %T') $*" >> $TR/NOTES.txt; }
until ls $RUN/*/model_${WAIT_FOR}.pt >/dev/null 2>&1; do sleep 20; done
sleep 30
# 2026-11-12 20:25: CPU pre-flight before v4j is stopped (the 19:22 check died on `unknown HIKING_X2_OPTS: [g]`, which no CPU test covered)
if ! $HOME/hiking-env/bin/python $X/v4j/precheck_v4k.py $X/continue_v4k.sh $D > $S/precheck_v4k.log 2>&1; then
  echo "$(date +%T) v4k NOT started: CPU pre-flight failed ($(grep -a FAIL $S/precheck_v4k.log | head -2 | tr '\n' ' ')); v4j keeps training, not stopped" >> $S/launch.txt
  exit 1
fi
echo "$(date +%T) v4k CPU pre-flight passed (motion v4, extended_v4k, opts v allowed, slow.py tests)" >> $S/launch.txt
L0=$(grep -a "Curriculum/terrain_levels" $S/train.log | tail -20 | awk '{s+=$NF;n++} END{print s/n}')
E0=$(grep -a "Mean episode length" $S/train.log | tail -20 | awk '{s+=$NF;n++} END{print s/n}')
W0=$(grep -a "turn/rate_along_cmd" $S/train.log | tail -20 | awk '{s+=$NF;n++} END{print s/n}')
T0=$(date +%s)
touch $S/STOP
while tmux has -t hiking_x2 2>/dev/null; do sleep 3; done
[ -f $S/STOP ] && { DST=$TR/STOP_${NAME}_$(date +%H%M%S); mv $S/STOP $DST; note "$DST: stop flag of $NAME, used by continue_v4k.sh"; }
pkill -f crashguard_v4.sh && echo "$(date -u '+%F %T') crashguard of v4j stopped by continue_v4k.sh" >> $S/gates.txt
cp $S/terrain_levels.pt $S/terrain_levels_prestop.pt
# v4j's files kept for the fall-back
H=$(date +%H%M%S); BK=$TR/v4j_files_$H; mkdir -p $BK
for f in extended.py routes.py keep.py; do cp $D/$f $BK/$f; done
note "$BK: extended.py, routes.py, keep.py of v4j (iterations 5500-$WAIT_FOR), replaced by the .v4k files"
cat > $X/fallback_v4j.sh <<EOF
#!/bin/bash
# restore v4j's files and resume v4j (written by continue_v4k.sh at $H)
for f in extended.py routes.py keep.py; do cp $BK/\$f $D/\$f; done
rm -f $D/slow.py
bash $X/apply.sh > /dev/null
rm -f $T/code/hiking-in-the-wild/InstinctLab/source/instinctlab/instinctlab/tasks/parkour/config/x2/slow.py
cp $S/terrain_levels_prestop.pt $S/terrain_levels.pt
bash $X/start_x2_parkour_v4j.sh
EOF
for f in extended.py routes.py keep.py contact.py slow.py; do cp $D/$f.v4k $D/$f; done
bash $X/apply.sh > /dev/null
VENV="HIKING_X2_OPTS=a,c,d,f,r,s,t,k,y,o,z,w,g,v HIKING_X2_VX_BACK=0 HIKING_X2_SLOW_FWD_SHARE=0.25 HIKING_X2_SLOW_TURN_SHARE=0.4 HIKING_X2_SLOW_VX=0.1 HIKING_X2_TURN_DRIFT_WEIGHT=-2.0 HIKING_X2_SIDESTEP_SHARE=0.3 HIKING_X2_SIDESTEP_V=0.3 HIKING_X2_ROUTE_VY=0.2 HIKING_EXTENDED_TERRAINS=$T/data/extended_v4k HIKING_X2_MOTION_DIR=$T/data/motion_v4 HIKING_X2_MOTION_YAML=x2_parkour_motion_without_run_v4.yaml"
CK=$RUN/_check_v4k_$(date +%H%M%S); mkdir -p $CK
( cd $T/code/hiking-in-the-wild/InstinctLab && env $VENV HIKING_X2_ROUTE_SHARE=0.7 HIKING_X2_STAGE=500,1500 HIKING_X2_VX_MIN=0.45 \
  HIKING_X2_TURN_SHARE=0.15 HIKING_X2_STAND_SHARE=0.10 HIKING_X2_STOP_ANY_YAW=1 OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 \
  timeout 1800 $HOME/hiking-env/bin/python $X/check_x2_env.py --headless --task Instinct-Parkour-ExtendedFixed-Amp-X2-v0 \
  --num_envs 256 --steps 150 --out $CK ) > $CK/check.log 2>&1
CKRC=$?
echo "$(date +%T) v4k construction check exit $CKRC (log $CK/check.log)" >> $S/launch.txt
NR=$(grep -ao "\[X2 routes\] [0-9]* routes" $CK/check.log | head -1 | awk '{print $3}')
echo "$(date +%T) route table in the v4k check: ${NR:-none} routes" >> $S/launch.txt
if [ $CKRC != 0 ] || grep -aq "Traceback" $CK/check.log || [ "${NR:-0}" -lt 100 ] || ! grep -aq "wall_contact_graded" $CK/check.log || ! grep -aq "turn_effort" $CK/check.log; then
  bash $X/fallback_v4j.sh
  echo "$(date +%T) v4k NOT started: construction check failed (exit $CKRC, routes ${NR:-none}, or wall_contact_graded not in the reward table); v4j resumed; last error: $(grep -aE '^[A-Za-z]*(Error|Exception)' $CK/check.log | tail -1 | cut -c1-200)" >> $S/launch.txt
  tmux new -d -s guard4j "FB_STAGED=$D/staged.py FB_START=$X/start_x2_parkour_v4j.sh bash $X/crashguard_v4.sh $L0 $E0"
  exit 1
fi
ENVV="HIKING_X2_OPTS=a,c,d,f,r,s,t,k,y,o,z,w,g,v HIKING_X2_VX_BACK=0 HIKING_X2_SLOW_FWD_SHARE=0.25 HIKING_X2_SLOW_TURN_SHARE=0.4 HIKING_X2_SLOW_VX=0.1 HIKING_X2_TURN_DRIFT_WEIGHT=-2.0 HIKING_X2_STAGE=500,1500 HIKING_X2_CAM_RAND=1 HIKING_X2_ALIVE_EARLY=0.3 HIKING_X2_WAIT_EARLY=2 HIKING_X2_STYLE_EARLY=0.3 HIKING_X2_FALL_PENALTY=8 HIKING_X2_REF_START=0.5 HIKING_X2_AIR_TD_WEIGHT=140 HIKING_X2_VX_MIN=0.45 HIKING_X2_TURN_SHARE=0.15 HIKING_X2_LEVELS_OUT=$S/terrain_levels.pt HIKING_X2_LEVELS_IN=$S/terrain_levels.pt HIKING_X2_STAND_SHARE=0.10 HIKING_X2_OVERSPEED_WEIGHT=-3.0 HIKING_X2_STOP_WEIGHT=-2.0 HIKING_X2_STOP_ANY_YAW=1 HIKING_X2_ROUTE_SHARE=0.7 HIKING_X2_SIDESTEP_SHARE=0.3 HIKING_X2_SIDESTEP_V=0.3 HIKING_X2_ROUTE_VY=0.2 HIKING_EXTENDED_TERRAINS=$T/data/extended_v4k HIKING_X2_MOTION_DIR=$T/data/motion_v4 HIKING_X2_MOTION_YAML=x2_parkour_motion_without_run_v4.yaml"
printf '#!/bin/bash\n# Resume %s with the options of its continuation from %s (continue_v4k.sh: v4j + graded contact, sidestep, stairs inset, motion v3).\ntmux new -d -s hiking_x2 "%s %s %s/supervise_x2.py --task Instinct-Parkour-ExtendedFixed-Amp-X2-v0 --name %s --total 30000 --num_envs 8192 --save_interval 125 2>&1 | tee -a %s/runs/%s_supervisor_tmux.log"\n' \
  $NAME $WAIT_FOR "$ENVV" '$HOME/hiking-env/bin/python' $X $NAME $T $NAME > $X/start_x2_parkour_v4k.sh
LOG0=$(stat -c %s $S/train.log 2>/dev/null || echo 0)
SUP0=$(grep -c ' start pid [0-9]* from iteration' $S/supervisor.log 2>/dev/null); SUP0=${SUP0:-0}
echo "continue from newest checkpoint (v4k): $ENVV started $(date '+%Y-%m-%d %H:%M:%S')" >> $S/launch.txt
bash $X/start_x2_parkour_v4k.sh
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
  for i in $(seq 1 90); do
    NEW=$(tail -c +$((LOG0 + 1)) $S/train.log 2>/dev/null)
    if printf '%s' "$NEW" | grep -aq "Traceback"; then OK=0; WHY="Traceback in train.log during the trial"; break; fi
    [ $(printf '%s' "$NEW" | grep -ac "Learning iteration") -ge 31 ] && break
    sleep 20
  done
  if [ $OK = 1 ]; then
    NEW=$(tail -c +$((LOG0 + 1)) $S/train.log)
    L1=$(printf '%s' "$NEW" | grep -a "Curriculum/terrain_levels" | tail -5 | awk '{s+=$NF;n++} END{print (n?s/n:0)}')
    E1=$(printf '%s' "$NEW" | grep -a "Mean episode length" | tail -5 | awk '{s+=$NF;n++} END{print (n?s/n:0)}')
    N1=$(printf '%s' "$NEW" | grep -ac "Learning iteration")
    W1=$(printf '%s' "$NEW" | grep -a "turn/rate_along_cmd" | tail -5 | awk '{s+=$NF;n++} END{print (n?s/n:0)}')
    echo "$(date +%T) v4k trial after $N1 iterations: level $L1 (before $L0), episode $E1 (before $E0), turn rate $W1 (before $W0)" >> $S/launch.txt
    if ! python3 -c "import sys; sys.exit(0 if $N1 >= 31 and $L1 >= 0.8 * $L0 and $E1 >= 0.4 * $E0 and $W1 >= 0.5 * $W0 else 1)"; then
      OK=0; WHY="trial failed: level $L1 vs $L0, episode $E1 vs $E0, turn $W1 vs $W0 after $N1 iterations"
    fi
  fi
fi
if [ $OK = 1 ]; then
  R=$(tail -c +$((LOG0 + 1)) $S/train.log | grep -a "level_keeper\]\|v4k\]" | head -3 | tr '\n' ' ')
  echo "$(date +%T) continuation v4k is training, GPU idle about $(( $(date +%s) - T0 )) s; $R" >> $S/launch.txt
  tmux new -d -s guard4k "FB_STAGED=$D/staged.py FB_START=$X/fallback_v4j.sh bash $X/crashguard_v4.sh $L0 $E0"
else
  touch $S/STOP
  while tmux has -t hiking_x2 2>/dev/null; do sleep 3; done
  [ -f $S/STOP ] && { DST=$TR/STOP_${NAME}_$(date +%H%M%S); mv $S/STOP $DST; note "$DST: stop flag of the failed v4k continuation"; }
  bash $X/fallback_v4j.sh
  echo "$(date +%T) continuation v4k FAILED ($WHY; see train.log after byte $LOG0); v4j resumed (fallback_v4j.sh)" >> $S/launch.txt
  tmux new -d -s guard4j "FB_STAGED=$D/staged.py FB_START=$X/start_x2_parkour_v4j.sh bash $X/crashguard_v4.sh $L0 $E0"
fi
