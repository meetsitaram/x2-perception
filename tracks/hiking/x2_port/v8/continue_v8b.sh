#!/bin/bash
# continue_v8b.sh (2026-11-14 09:20: v8 + stand 0.10->0.05, turn 0.15->0.10, slow fwd 0.25->0.12; armed by continue_v8.sh when v8 does not pass; failure -> v7). Based on continue_v8.sh (2026-11-14, Hiking priority: the paper's terrains first). ONE planned restart of x2_parkour_v4 (v7) from its
# newest checkpoint, armed on $S/GO_v8b (the coordinator writes it on the user's yes). v8 = v7 + HIKING_X2_FAIR=1 (paper
# curriculum signal: stand / turn / slow-forward episodes neither promote nor demote) + HIKING_X2_AIR_TD_PROGRESS=1 (touchdown
# pay x forward progress under walking commands); code v8/fair.py + v8/extended.py.v8 (hook). Policy, critic, discriminator
# and optimisers carry over (same observations, actions, discriminator inputs). Pre-flight before v7 stops; construction
# check; 76-iteration trial; any failure -> fallback_v7.sh (v7 files and flags, guard7).
T=/mnt/filesystem-g4/tracks/hiking; S=$T/runs/x2_parkour_v4/_supervisor   # defined BEFORE ENVV (bug of 11-14 08:50)
ENVV="HIKING_X2_FAIR=1 HIKING_X2_AIR_TD_PROGRESS=1 HIKING_X2_OPTS=a,c,d,f,r,s,t,k,y,o,z,w,g,v,j HIKING_X2_VX_BACK=0 HIKING_X2_SLOW_FWD_SHARE=0.12 HIKING_X2_SLOW_TURN_SHARE=0.4 HIKING_X2_SLOW_VX=0.1 HIKING_X2_TURN_DRIFT_WEIGHT=-2.0 HIKING_X2_STAGE=500,1500 HIKING_X2_CAM_RAND=1 HIKING_X2_ALIVE_EARLY=0.3 HIKING_X2_WAIT_EARLY=2 HIKING_X2_STYLE_EARLY=0.3 HIKING_X2_FALL_PENALTY=8 HIKING_X2_REF_START=0.5 HIKING_X2_AIR_TD_WEIGHT=140 HIKING_X2_AIR_TD_CAP=ref HIKING_X2_VX_MIN=0.45 HIKING_X2_TURN_SHARE=0.10 HIKING_X2_LEVELS_OUT=$S/terrain_levels.pt HIKING_X2_LEVELS_IN=$S/terrain_levels.pt HIKING_X2_STAND_SHARE=0.05 HIKING_X2_OVERSPEED_WEIGHT=-3.0 HIKING_X2_STOP_WEIGHT=-2.0 HIKING_X2_STOP_ANY_YAW=1 HIKING_X2_ROUTE_SHARE=0.7 HIKING_X2_SIDESTEP_SHARE=0.3 HIKING_X2_SIDESTEP_V=0.3 HIKING_X2_ROUTE_VY=0.2 HIKING_EXTENDED_TERRAINS=$T/data/extended_v4k HIKING_X2_MOTION_DIR=$T/data/motion_v7 HIKING_X2_MOTION_YAML=parkour_motion_without_run_x2_v4.yaml"
T=/mnt/filesystem-g4/tracks/hiking; X=$T/code/x2_port; D=$X/files_instinctlab/tasks/parkour/config/x2
I=$T/code/hiking-in-the-wild/InstinctLab/source/instinctlab/instinctlab
NAME=x2_parkour_v4; RUN=$T/runs/$NAME; S=$RUN/_supervisor
TR=$T/_trash/$(date +%F); mkdir -p $TR
# ENVV paths must exist (11-14 08:50: ENVV used $T/$S before they were set -> /data/extended_v4k, check failed)
for kv in $ENVV; do case "$kv" in HIKING_EXTENDED_TERRAINS=*|HIKING_X2_MOTION_DIR=*) [ -d "${kv#*=}" ] || { echo "$(date -u '+%F %T') v8b NOT started: ENVV path ${kv#*=} missing" >> $S/launch.txt; exit 1; };; esac; done
note() { echo "$(date '+%F %T') $*" >> $TR/NOTES.txt; }
say() { echo "$(date -u '+%F %T') v8b $*" >> $S/launch.txt; }
tree_ok() {  # one code source: every files_instinctlab file equals its installed copy, no extra files in the installed x2 dir
  local f; for f in $(cd $X/files_instinctlab && find . -type f ! -path "*__pycache__*"); do cmp -s $X/files_instinctlab/$f $I/$f || return 1; done
  for f in $(ls $I/tasks/parkour/config/x2 | grep -v __pycache__); do [ -e $D/$f ] || return 1; done; return 0; }
newest() { ls $RUN/*/model_*.pt 2>/dev/null | sed 's/.*model_\([0-9]*\)\.pt/\1/' | sort -n | tail -1; }
say "armed: waiting for $S/GO_v8b (\"now\" = newest checkpoint, else the next one)"
until [ -f $S/GO_v8b ]; do sleep 30; done
if grep -q now $S/GO_v8b; then WAIT_FOR=$(newest); else WAIT_FOR=$(( ($(newest) / 125 + 1) * 125 )); fi
say "GO seen; restart from model_$WAIT_FOR"
until ls $RUN/*/model_${WAIT_FOR}.pt >/dev/null 2>&1; do sleep 20; done
sleep 30
# CPU pre-flight again, BEFORE v7 stops (v7 keeps training on any failure)
if ! bash $X/v8/precheck_v8b.sh $X/v8/continue_v8b.sh > $S/precheck_v8b.log 2>&1; then
  say "NOT started: precheck failed ($(grep -a FAIL $S/precheck_v8b.log | head -3 | tr '\n' ' ' | cut -c1-300)); v7 keeps training"; exit 1; fi
if ! bash $X/v8/cfgscan_v8b.sh $S/cfgscan_v8b.log; then
  say "NOT started: cfg scan failed ($(grep -a '^FAIL\|Error' $S/cfgscan_v8.log | head -3 | tr '\n' ' ' | cut -c1-300)); v7 keeps training"; exit 1; fi
if ! $HOME/hiking-env/bin/python $X/v8/test_fair.py > $S/test_fair_v8b.log 2>&1; then
  say "NOT started: fair.py unit test failed; v7 keeps training"; exit 1; fi
say "CPU pre-flight passed (precheck, cfgscan, unit test)"
L0=$(grep -a "Curriculum/terrain_levels" $S/train.log | tail -20 | awk '{s+=$NF;n++} END{print s/n}')
E0=$(grep -a "Mean episode length" $S/train.log | tail -20 | awk '{s+=$NF;n++} END{print s/n}')
W0=$(grep -a "turn/rate_along_cmd" $S/train.log | tail -20 | awk '{s+=$NF;n++} END{print s/n}')
BF=$S/baseline_v7_for_v8.txt; if [ -n "$(find $BF -mmin -180 2>/dev/null)" ]; then read L0 E0 W0 < $BF; say "trial baseline = v7 numbers before v8 ($L0 $E0 $W0)"; fi
T0=$(date +%s)
# planned stop: the v7 guard goes first so it does not restart v7
tmux kill-session -t guard7 2>/dev/null && echo "$(date -u '+%F %T') crashguard_v7 stopped by continue_v8.sh (planned v8 restart)" >> $S/gates.txt
touch $S/STOP
while tmux has -t hiking_x2 2>/dev/null; do sleep 3; done
[ -f $S/STOP ] && { DST=$TR/STOP_${NAME}_$(date +%H%M%S); mv $S/STOP $DST; note "$DST: stop flag of $NAME, used by continue_v8b.sh"; }
cp $S/terrain_levels.pt $S/terrain_levels_prestop_v8b.pt
H=$(date +%H%M%S); BK=$TR/v7_files_v8b_$H; mkdir -p $BK; cp $D/extended.py $BK/extended.py
note "$BK: extended.py of v7 (to $WAIT_FOR), replaced by v8/extended.py.v8 (+ fair.py)"
cat > $X/fallback_v7.sh <<EOF
#!/bin/bash
# restore v7's files and resume v7 (written by continue_v8.sh at $H)
cp $BK/extended.py $D/extended.py
[ -f $D/fair.py ] && mv $D/fair.py $TR/fair.py.src_$H
[ -f $I/tasks/parkour/config/x2/fair.py ] && mv $I/tasks/parkour/config/x2/fair.py $TR/fair.py.installed_$H
bash $X/apply.sh > /dev/null
cp $S/terrain_levels_prestop_v8b.pt $S/terrain_levels.pt
bash $X/start_x2_parkour_v7.sh
tmux has -t guard7 2>/dev/null || tmux new -d -s guard7 "bash $X/crashguard_v7.sh"
EOF
cp $X/v8/fair.py $D/fair.py; cp $X/v8/extended.py.v8 $D/extended.py
bash $X/apply.sh > /dev/null
if ! tree_ok; then
  bash $X/fallback_v7.sh; say "NOT started: installed InstinctLab copy differs from files_instinctlab after apply.sh; v7 resumed"; exit 1; fi
CK=$RUN/_check_v8b_$(date +%H%M%S); mkdir -p $CK
( cd $T/code/hiking-in-the-wild/InstinctLab && env $ENVV OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 \
  timeout 1800 $HOME/hiking-env/bin/python $X/check_x2_env.py --headless --task Instinct-Parkour-ExtendedFixed-Amp-X2-v0 \
  --num_envs 256 --steps 150 --out $CK ) > $CK/check.log 2>&1
CKRC=$?
NR=$(grep -ao "\[X2 routes\] [0-9]* routes" $CK/check.log | head -1 | awk '{print $3}')
say "construction check exit $CKRC, routes ${NR:-none} (log $CK/check.log)"
if [ $CKRC != 0 ] || grep -aq "Traceback" $CK/check.log || [ "${NR:-0}" -lt 100 ] || ! grep -aq "\[X2 fair\]" $CK/check.log; then
  bash $X/fallback_v7.sh; say "NOT started: construction check failed (exit $CKRC, routes ${NR:-none}, or no [X2 fair] line); v7 resumed"; exit 1; fi
sed "s#^tmux new -d -s hiking_x2 \"#tmux new -d -s hiking_x2 \"HIKING_X2_FAIR=1 HIKING_X2_AIR_TD_PROGRESS=1 #; s#HIKING_X2_SLOW_FWD_SHARE=0.25#HIKING_X2_SLOW_FWD_SHARE=0.12#; s# HIKING_X2_TURN_SHARE=0.15# HIKING_X2_TURN_SHARE=0.10#; s#HIKING_X2_STAND_SHARE=0.10#HIKING_X2_STAND_SHARE=0.05#; s#continue_v7.sh: v5b + reference-cadence touchdown cap + paper motion v4, seed 42#continue_v8b.sh: v8 + fewer non-walking commands (stand 0.05, turn 0.10, slow fwd 0.12)#" \
  $X/start_x2_parkour_v7.sh > $X/start_x2_parkour_v8b.sh
LOG0=$(stat -c %s $S/train.log 2>/dev/null || echo 0)
SUP0=$(grep -c ' start pid [0-9]* from iteration' $S/supervisor.log 2>/dev/null); SUP0=${SUP0:-0}
echo "continue from newest checkpoint (v8b): $ENVV started $(date '+%Y-%m-%d %H:%M:%S')" >> $S/launch.txt
bash $X/start_x2_parkour_v8b.sh
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
  for i in $(seq 1 200); do
    NEW=$(tail -c +$((LOG0 + 1)) $S/train.log 2>/dev/null)
    if printf '%s' "$NEW" | grep -aq "Traceback"; then OK=0; WHY="Traceback in train.log during the trial"; break; fi
    [ $(printf '%s' "$NEW" | grep -ac "Learning iteration") -ge 76 ] && break
    sleep 20
  done
  if [ $OK = 1 ]; then
    NEW=$(tail -c +$((LOG0 + 1)) $S/train.log)
    L1=$(printf '%s' "$NEW" | grep -a "Curriculum/terrain_levels" | tail -30 | awk '{s+=$NF;n++} END{print (n?s/n:0)}')
    E1=$(printf '%s' "$NEW" | grep -a "Mean episode length" | tail -30 | awk '{s+=$NF;n++} END{print (n?s/n:0)}')
    N1=$(printf '%s' "$NEW" | grep -ac "Learning iteration")
    W1=$(printf '%s' "$NEW" | grep -a "turn/rate_along_cmd" | tail -30 | awk '{s+=$NF;n++} END{print (n?s/n:0)}')
    P1=$(printf '%s' "$NEW" | grep -a "paper_cmd_episode" | tail -30 | awk '{s+=$NF;n++} END{print (n?s/n:-1)}')
    say "trial after $N1 iterations (last 30): level $L1 (before $L0), episode $E1 (before $E0), turn rate $W1 (before $W0), paper-command episode share $P1"
    # level gate 0.7 x (v7: 0.8 x): v8 is meant to move refusing robots DOWN, so a level drop is the expected effect
    if ! python3 -c "import sys; sys.exit(0 if $N1 >= 75 and $L1 >= 0.7 * $L0 and $E1 >= 0.4 * $E0 and $W1 >= 0.5 * $W0 else 1)"; then
      OK=0; WHY="trial failed: level $L1 vs $L0, episode $E1 vs $E0, turn $W1 vs $W0 over the last 30 of $N1 iterations"
    fi
  fi
fi
if [ $OK = 1 ]; then
  say "is training, GPU idle about $(( $(date +%s) - T0 )) s"
  tmux new -d -s guard8b "bash $X/v8/crashguard_v8b.sh"
else
  touch $S/STOP
  while tmux has -t hiking_x2 2>/dev/null; do sleep 3; done
  [ -f $S/STOP ] && { DST=$TR/STOP_${NAME}_$(date +%H%M%S); mv $S/STOP $DST; note "$DST: stop flag of the failed v8b continuation"; }
  bash $X/fallback_v7.sh
  say "FAILED ($WHY; see train.log after byte $LOG0); v7 resumed (fallback_v7.sh), guard7 restarted"
fi
