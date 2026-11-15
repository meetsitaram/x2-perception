#!/bin/bash
# continue_v4j.sh (2026-11-12, from continue_v4i.sh). v4j = v4i plus, in one restart (user rule "Wall and prop contact in
# training", coordinator 19:00):
#  g  contact.py: `wall_contact_graded` replaces `undesired_contacts` (trunk / legs strong 1-4 per body; elbows, wrists, hands free
#     under 30 N and 0.5 s, then 0.25 x); `base_contact` ends the episode only for a torso resting on something (> 50 N, vertical
#     part dominant); v4j logs (episode ends and route arrivals per tile group, forces per body group, tunnelling, speed near props)
#  sidestep: HIKING_X2_SIDESTEP_SHARE=0.3 of route robots on tight-spot / home tiles walk the route sideways; HIKING_X2_ROUTE_VY=0.2
#     lateral command for a waypoint off to the side within 1 m (routes.py.v4j)
#  terrain: HIKING_EXTENDED_TERRAINS=tracks/hiking/data/extended_v4j (stairs inset tiles v0-v5 at the extended_v1 rows with 700
#     routes of 10 kinds; route starts moved off props, spawns near props dropped)
#  motion: HIKING_X2_MOTION_DIR=tracks/hiking/data/motion_v3 (v2 + 9 sideways Pico takes + 12 turning stair takes, 121 s; tp_stairs_113941 dropped 20:14, retarget artefact)
# Construction check with 256 robots in the pause; trial at 31 iterations (level >= 0.8 x, episode >= 0.4 x, turn rate >= 0.7 x);
# fall-back v4i (files restored, pre-stop levels); then a crash guard with fall-back v4i.
WAIT_FOR=${1:-5500}
T=/mnt/filesystem-g4/tracks/hiking; X=$T/code/x2_port; D=$X/files_instinctlab/tasks/parkour/config/x2
NAME=x2_parkour_v4; RUN=$T/runs/$NAME; S=$RUN/_supervisor
TR=$T/_trash/$(date +%F); mkdir -p $TR
note() { echo "$(date '+%F %T') $*" >> $TR/NOTES.txt; }
until ls $RUN/*/model_${WAIT_FOR}.pt >/dev/null 2>&1; do sleep 20; done
sleep 30
# 2026-11-12 20:25: CPU pre-flight before v4i is stopped (the 19:22 check died on `unknown HIKING_X2_OPTS: [g]`, which no CPU test covered)
if ! $HOME/hiking-env/bin/python $X/v4j/precheck_v4j.py $X/continue_v4j.sh $D > $S/precheck_v4j.log 2>&1; then
  echo "$(date +%T) v4j NOT started: CPU pre-flight failed ($(grep -a FAIL $S/precheck_v4j.log | head -2 | tr '\n' ' ')); v4i keeps training, not stopped" >> $S/launch.txt
  exit 1
fi
echo "$(date +%T) v4j CPU pre-flight passed (motion v3 without tp_stairs_113941; opts g allowed)" >> $S/launch.txt
L0=$(grep -a "Curriculum/terrain_levels" $S/train.log | tail -20 | awk '{s+=$NF;n++} END{print s/n}')
E0=$(grep -a "Mean episode length" $S/train.log | tail -20 | awk '{s+=$NF;n++} END{print s/n}')
W0=$(grep -a "turn/rate_along_cmd" $S/train.log | tail -20 | awk '{s+=$NF;n++} END{print s/n}')
T0=$(date +%s)
touch $S/STOP
while tmux has -t hiking_x2 2>/dev/null; do sleep 3; done
[ -f $S/STOP ] && { DST=$TR/STOP_${NAME}_$(date +%H%M%S); mv $S/STOP $DST; note "$DST: stop flag of $NAME, used by continue_v4j.sh"; }
pkill -f crashguard_v4.sh && echo "$(date -u '+%F %T') crashguard of v4i stopped by continue_v4j.sh" >> $S/gates.txt
cp $S/terrain_levels.pt $S/terrain_levels_prestop.pt
# v4i's files kept for the fall-back
H=$(date +%H%M%S); BK=$TR/v4i_files_$H; mkdir -p $BK
for f in extended.py routes.py keep.py; do cp $D/$f $BK/$f; done
note "$BK: extended.py, routes.py, keep.py of v4i (iterations 5000-$WAIT_FOR), replaced by the .v4j files"
cat > $X/fallback_v4i.sh <<EOF
#!/bin/bash
# restore v4i's files and resume v4i (written by continue_v4j.sh at $H)
for f in extended.py routes.py keep.py; do cp $BK/\$f $D/\$f; done
rm -f $D/contact.py
bash $X/apply.sh > /dev/null
rm -f $T/code/hiking-in-the-wild/InstinctLab/source/instinctlab/instinctlab/tasks/parkour/config/x2/contact.py
cp $S/terrain_levels_prestop.pt $S/terrain_levels.pt
bash $X/start_x2_parkour_v4i.sh
EOF
for f in extended.py routes.py keep.py contact.py; do cp $D/$f.v4j $D/$f; done
bash $X/apply.sh > /dev/null
VENV="HIKING_X2_OPTS=a,c,d,f,r,s,t,k,y,o,z,w,g HIKING_X2_SIDESTEP_SHARE=0.3 HIKING_X2_SIDESTEP_V=0.3 HIKING_X2_ROUTE_VY=0.2 HIKING_EXTENDED_TERRAINS=$T/data/extended_v4j HIKING_X2_MOTION_DIR=$T/data/motion_v3 HIKING_X2_MOTION_YAML=x2_parkour_motion_without_run_v3.yaml"
CK=$RUN/_check_v4j_$(date +%H%M%S); mkdir -p $CK
( cd $T/code/hiking-in-the-wild/InstinctLab && env $VENV HIKING_X2_ROUTE_SHARE=0.7 HIKING_X2_STAGE=500,1500 HIKING_X2_VX_MIN=0.45 \
  HIKING_X2_TURN_SHARE=0.15 HIKING_X2_STAND_SHARE=0.10 HIKING_X2_STOP_ANY_YAW=1 OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 \
  timeout 1800 $HOME/hiking-env/bin/python $X/check_x2_env.py --headless --task Instinct-Parkour-ExtendedFixed-Amp-X2-v0 \
  --num_envs 256 --steps 150 --out $CK ) > $CK/check.log 2>&1
CKRC=$?
echo "$(date +%T) v4j construction check exit $CKRC (log $CK/check.log)" >> $S/launch.txt
NR=$(grep -ao "\[X2 routes\] [0-9]* routes" $CK/check.log | head -1 | awk '{print $3}')
echo "$(date +%T) route table in the v4j check: ${NR:-none} routes" >> $S/launch.txt
if [ $CKRC != 0 ] || grep -aq "Traceback" $CK/check.log || [ "${NR:-0}" -lt 100 ] || ! grep -aq "wall_contact_graded" $CK/check.log; then
  bash $X/fallback_v4i.sh
  echo "$(date +%T) v4j NOT started: construction check failed (exit $CKRC, routes ${NR:-none}, or wall_contact_graded not in the reward table); v4i resumed" >> $S/launch.txt
  tmux new -d -s guard4i "FB_STAGED=$D/staged.py FB_START=$X/start_x2_parkour_v4i.sh bash $X/crashguard_v4.sh $L0 $E0"
  exit 1
fi
ENVV="HIKING_X2_OPTS=a,c,d,f,r,s,t,k,y,o,z,w,g HIKING_X2_STAGE=500,1500 HIKING_X2_CAM_RAND=1 HIKING_X2_ALIVE_EARLY=0.3 HIKING_X2_WAIT_EARLY=2 HIKING_X2_STYLE_EARLY=0.3 HIKING_X2_FALL_PENALTY=8 HIKING_X2_REF_START=0.5 HIKING_X2_AIR_TD_WEIGHT=140 HIKING_X2_VX_MIN=0.45 HIKING_X2_TURN_SHARE=0.15 HIKING_X2_LEVELS_OUT=$S/terrain_levels.pt HIKING_X2_LEVELS_IN=$S/terrain_levels.pt HIKING_X2_STAND_SHARE=0.10 HIKING_X2_OVERSPEED_WEIGHT=-3.0 HIKING_X2_STOP_WEIGHT=-2.0 HIKING_X2_STOP_ANY_YAW=1 HIKING_X2_ROUTE_SHARE=0.7 HIKING_X2_SIDESTEP_SHARE=0.3 HIKING_X2_SIDESTEP_V=0.3 HIKING_X2_ROUTE_VY=0.2 HIKING_EXTENDED_TERRAINS=$T/data/extended_v4j HIKING_X2_MOTION_DIR=$T/data/motion_v3 HIKING_X2_MOTION_YAML=x2_parkour_motion_without_run_v3.yaml"
printf '#!/bin/bash\n# Resume %s with the options of its continuation from %s (continue_v4j.sh: v4i + graded contact, sidestep, stairs inset, motion v3).\ntmux new -d -s hiking_x2 "%s %s %s/supervise_x2.py --task Instinct-Parkour-ExtendedFixed-Amp-X2-v0 --name %s --total 30000 --num_envs 8192 --save_interval 125 2>&1 | tee -a %s/runs/%s_supervisor_tmux.log"\n' \
  $NAME $WAIT_FOR "$ENVV" '$HOME/hiking-env/bin/python' $X $NAME $T $NAME > $X/start_x2_parkour_v4j.sh
LOG0=$(stat -c %s $S/train.log 2>/dev/null || echo 0)
SUP0=$(grep -c ' start pid [0-9]* from iteration' $S/supervisor.log 2>/dev/null); SUP0=${SUP0:-0}
echo "continue from newest checkpoint (v4j): $ENVV started $(date '+%Y-%m-%d %H:%M:%S')" >> $S/launch.txt
bash $X/start_x2_parkour_v4j.sh
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
    echo "$(date +%T) v4j trial after $N1 iterations: level $L1 (before $L0), episode $E1 (before $E0), turn rate $W1 (before $W0)" >> $S/launch.txt
    if ! python3 -c "import sys; sys.exit(0 if $N1 >= 31 and $L1 >= 0.8 * $L0 and $E1 >= 0.4 * $E0 and $W1 >= 0.7 * $W0 else 1)"; then
      OK=0; WHY="trial failed: level $L1 vs $L0, episode $E1 vs $E0, turn $W1 vs $W0 after $N1 iterations"
    fi
  fi
fi
if [ $OK = 1 ]; then
  R=$(tail -c +$((LOG0 + 1)) $S/train.log | grep -a "level_keeper\]\|v4j\]" | head -3 | tr '\n' ' ')
  echo "$(date +%T) continuation v4j is training, GPU idle about $(( $(date +%s) - T0 )) s; $R" >> $S/launch.txt
  tmux new -d -s guard4j "FB_STAGED=$D/staged.py FB_START=$X/fallback_v4i.sh bash $X/crashguard_v4.sh $L0 $E0"
else
  touch $S/STOP
  while tmux has -t hiking_x2 2>/dev/null; do sleep 3; done
  [ -f $S/STOP ] && { DST=$TR/STOP_${NAME}_$(date +%H%M%S); mv $S/STOP $DST; note "$DST: stop flag of the failed v4j continuation"; }
  bash $X/fallback_v4i.sh
  echo "$(date +%T) continuation v4j FAILED ($WHY; see train.log after byte $LOG0); v4i resumed (fallback_v4i.sh)" >> $S/launch.txt
  tmux new -d -s guard4i "FB_STAGED=$D/staged.py FB_START=$X/start_x2_parkour_v4i.sh bash $X/crashguard_v4.sh $L0 $E0"
fi
