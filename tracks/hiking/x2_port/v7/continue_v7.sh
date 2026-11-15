#!/bin/bash
# continue_v7.sh (2026-11-13) = continue_v6.sh with the motion reference replaced by the paper reference v4 (user approved v4
# 2026-11-13: "G1 vs x2 walks look good with the forward leans seem correct"; v3 had read the torsoBase form wrongly) and a
# fixed seed 42 passed to train.py (--seed=42). ONE planned restart of x2_parkour_v4 from v5b's newest checkpoint, then it runs
# to --total 30000 with no further planned restarts. Content: v5b + touchdown cap speed-dependent (AIR_TD_CAP=ref, weight 140,
# variants/staged.py.v6) + motion data/motion_v7 (copy of hiking_release_x2/parkour_motion_without_run_x2_v4.npz). The
# discriminator carries over (same 550 inputs; the loader keeps it when shapes match). Gate: $S/GO_v7 ("now" = newest
# checkpoint). CPU pre-flight (precheck_v7.py, cfgscan_v7.sh) BEFORE v5b stops; construction check; 75-iteration trial (marks
# over the last 30); any failure -> v5b resumed with its own files (fallback_v5b.sh).
T=/mnt/filesystem-g4/tracks/hiking; X=$T/code/x2_port; D=$X/files_instinctlab/tasks/parkour/config/x2
I=$T/code/hiking-in-the-wild/InstinctLab/source/instinctlab/instinctlab
NAME=x2_parkour_v4; RUN=$T/runs/$NAME; S=$RUN/_supervisor
SRC=$T/data/motion_v7/parkour_motion_without_run_x2_v4_retargetted.npz
TR=$T/_trash/$(date +%F); mkdir -p $TR
note() { echo "$(date '+%F %T') $*" >> $TR/NOTES.txt; }
ENVV="HIKING_X2_OPTS=a,c,d,f,r,s,t,k,y,o,z,w,g,v,j HIKING_X2_VX_BACK=0 HIKING_X2_SLOW_FWD_SHARE=0.25 HIKING_X2_SLOW_TURN_SHARE=0.4 HIKING_X2_SLOW_VX=0.1 HIKING_X2_TURN_DRIFT_WEIGHT=-2.0 HIKING_X2_STAGE=500,1500 HIKING_X2_CAM_RAND=1 HIKING_X2_ALIVE_EARLY=0.3 HIKING_X2_WAIT_EARLY=2 HIKING_X2_STYLE_EARLY=0.3 HIKING_X2_FALL_PENALTY=8 HIKING_X2_REF_START=0.5 HIKING_X2_AIR_TD_WEIGHT=140 HIKING_X2_AIR_TD_CAP=ref HIKING_X2_VX_MIN=0.45 HIKING_X2_TURN_SHARE=0.15 HIKING_X2_LEVELS_OUT=$S/terrain_levels.pt HIKING_X2_LEVELS_IN=$S/terrain_levels.pt HIKING_X2_STAND_SHARE=0.10 HIKING_X2_OVERSPEED_WEIGHT=-3.0 HIKING_X2_STOP_WEIGHT=-2.0 HIKING_X2_STOP_ANY_YAW=1 HIKING_X2_ROUTE_SHARE=0.7 HIKING_X2_SIDESTEP_SHARE=0.3 HIKING_X2_SIDESTEP_V=0.3 HIKING_X2_ROUTE_VY=0.2 HIKING_EXTENDED_TERRAINS=$T/data/extended_v4k HIKING_X2_MOTION_DIR=$T/data/motion_v7 HIKING_X2_MOTION_YAML=parkour_motion_without_run_x2_v4.yaml"
tree_ok() {  # one code source: every files_instinctlab file equals its installed copy, no extra files in the installed x2 dir
  local f; for f in $(cd $X/files_instinctlab && find . -type f ! -path "*__pycache__*"); do cmp -s $X/files_instinctlab/$f $I/$f || return 1; done
  for f in $(ls $I/tasks/parkour/config/x2 | grep -v __pycache__); do [ -e $D/$f ] || return 1; done; return 0; }
echo "$(date +%T) v7 armed: waiting for $S/GO_v7 and $SRC" >> $S/launch.txt
until [ -f $S/GO_v7 ] && [ -f $SRC ]; do sleep 30; done
newest() { ls $RUN/*/model_*.pt 2>/dev/null | sed 's/.*model_\([0-9]*\)\.pt/\1/' | sort -n | tail -1; }
if grep -q now $S/GO_v7; then WAIT_FOR=$(newest); else WAIT_FOR=$(( ($(newest) / 125 + 1) * 125 )); fi
echo "$(date +%T) v7 GO seen; restart from model_$WAIT_FOR" >> $S/launch.txt
until ls $RUN/*/model_${WAIT_FOR}.pt >/dev/null 2>&1; do sleep 20; done
sleep 30
if ! $HOME/hiking-env/bin/python $X/v7/precheck_v7.py $X/v7/continue_v7.sh > $S/precheck_v7.log 2>&1; then
  echo "$(date +%T) v7 NOT started: CPU pre-flight failed ($(grep -a FAIL $S/precheck_v7.log | head -3 | tr '\n' ' ' | cut -c1-300)); v5b keeps training, not stopped" >> $S/launch.txt
  exit 1
fi
if ! bash $X/v7/cfgscan_v7.sh $S/cfgscan_v7.log; then
  echo "$(date +%T) v7 NOT started: CPU cfg scan failed ($(grep -a "^FAIL\|Error" $S/cfgscan_v7.log | head -3 | tr '\n' ' ' | cut -c1-300)); v5b keeps training, not stopped" >> $S/launch.txt
  exit 1
fi
echo "$(date +%T) v7 CPU pre-flight passed ($(grep -a '^motion\|^refcap\|^code tree' $S/precheck_v7.log | tr '\n' ' ' | cut -c1-400))" >> $S/launch.txt
L0=$(grep -a "Curriculum/terrain_levels" $S/train.log | tail -20 | awk '{s+=$NF;n++} END{print s/n}')
E0=$(grep -a "Mean episode length" $S/train.log | tail -20 | awk '{s+=$NF;n++} END{print s/n}')
W0=$(grep -a "turn/rate_along_cmd" $S/train.log | tail -20 | awk '{s+=$NF;n++} END{print s/n}')
T0=$(date +%s)
touch $S/STOP
while tmux has -t hiking_x2 2>/dev/null; do sleep 3; done
[ -f $S/STOP ] && { DST=$TR/STOP_${NAME}_$(date +%H%M%S); mv $S/STOP $DST; note "$DST: stop flag of $NAME, used by continue_v7.sh"; }
pkill -f crashguard_v4.sh && echo "$(date -u '+%F %T') crashguard of v5b stopped by continue_v7.sh" >> $S/gates.txt
cp $S/terrain_levels.pt $S/terrain_levels_prestop_v7.pt
H=$(date +%H%M%S); BK=$TR/v5b_files_v7_$H; mkdir -p $BK; cp $D/staged.py $BK/staged.py
note "$BK: staged.py of v5b (to $WAIT_FOR), replaced by variants/staged.py.v6 (v7)"
cat > $X/fallback_v5b.sh <<EOF
#!/bin/bash
# restore v5b's staged.py and resume v5b (written by continue_v7.sh at $H)
cp $BK/staged.py $D/staged.py
bash $X/apply.sh > /dev/null
cp $S/terrain_levels_prestop_v7.pt $S/terrain_levels.pt
bash $X/start_x2_parkour_v5b.sh
EOF
cp $X/variants/staged.py.v6 $D/staged.py
bash $X/apply.sh > /dev/null
if ! tree_ok; then
  bash $X/fallback_v5b.sh
  echo "$(date +%T) v7 NOT started: installed InstinctLab copy differs from files_instinctlab after apply.sh; v5b resumed" >> $S/launch.txt
  exit 1
fi
CK=$RUN/_check_v7_$(date +%H%M%S); mkdir -p $CK
( cd $T/code/hiking-in-the-wild/InstinctLab && env $ENVV OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 \
  timeout 1800 $HOME/hiking-env/bin/python $X/check_x2_env.py --headless --task Instinct-Parkour-ExtendedFixed-Amp-X2-v0 \
  --num_envs 256 --steps 150 --out $CK ) > $CK/check.log 2>&1
CKRC=$?
echo "$(date +%T) v7 construction check exit $CKRC (log $CK/check.log)" >> $S/launch.txt
NR=$(grep -ao "\[X2 routes\] [0-9]* routes" $CK/check.log | head -1 | awk '{print $3}')
if [ $CKRC != 0 ] || grep -aq "Traceback" $CK/check.log || [ "${NR:-0}" -lt 100 ] || ! grep -aq "wall_contact_graded" $CK/check.log || ! grep -aq "turn_effort" $CK/check.log || ! grep -aq "sim_fault" $CK/check.log || ! grep -aq "feet_air_touchdown" $CK/check.log; then
  bash $X/fallback_v5b.sh
  echo "$(date +%T) v7 NOT started: construction check failed (exit $CKRC, routes ${NR:-none}, or a term missing); v5b resumed; last error: $(grep -aE '^[A-Za-z]*(Error|Exception)' $CK/check.log | tail -1 | cut -c1-200)" >> $S/launch.txt
  tmux new -d -s guard5b "FB_STAGED=$BK/staged.py FB_START=$X/start_x2_parkour_v5b.sh bash $X/crashguard_v4.sh $L0 $E0"
  exit 1
fi
printf '#!/bin/bash\n# Resume %s with the options of its continuation from %s (continue_v7.sh: v5b + reference-cadence touchdown cap + paper motion v4, seed 42).\ntmux new -d -s hiking_x2 "%s %s %s/v5/supervise_x2_v5.py --task Instinct-Parkour-ExtendedFixed-Amp-X2-v0 --name %s --total 30000 --num_envs 8192 --save_interval 125 --extra=--seed=42 2>&1 | tee -a %s/runs/%s_supervisor_tmux.log"\n' \
  $NAME $WAIT_FOR "$ENVV" '$HOME/hiking-env/bin/python' $X $NAME $T $NAME > $X/start_x2_parkour_v7.sh
LOG0=$(stat -c %s $S/train.log 2>/dev/null || echo 0)
SUP0=$(grep -c ' start pid [0-9]* from iteration' $S/supervisor.log 2>/dev/null); SUP0=${SUP0:-0}
echo "continue from newest checkpoint (v7): $ENVV started $(date '+%Y-%m-%d %H:%M:%S')" >> $S/launch.txt
bash $X/start_x2_parkour_v7.sh
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
    echo "$(date +%T) v7 trial after $N1 iterations (marks over the last 30): level $L1 (before $L0), episode $E1 (before $E0), turn rate $W1 (before $W0)" >> $S/launch.txt
    if ! python3 -c "import sys; sys.exit(0 if $N1 >= 75 and $L1 >= 0.8 * $L0 and $E1 >= 0.4 * $E0 and $W1 >= 0.5 * $W0 else 1)"; then
      OK=0; WHY="trial failed: level $L1 vs $L0, episode $E1 vs $E0, turn $W1 vs $W0 over the last 30 of $N1 iterations"
    fi
  fi
fi
if [ $OK = 1 ]; then
  echo "$(date +%T) continuation v7 is training, GPU idle about $(( $(date +%s) - T0 )) s" >> $S/launch.txt
  tmux new -d -s guard7 "FB_STAGED=$BK/staged.py FB_START=$X/fallback_v5b.sh bash $X/crashguard_v4.sh $L0 $E0"
else
  touch $S/STOP
  while tmux has -t hiking_x2 2>/dev/null; do sleep 3; done
  [ -f $S/STOP ] && { DST=$TR/STOP_${NAME}_$(date +%H%M%S); mv $S/STOP $DST; note "$DST: stop flag of the failed v7 continuation"; }
  bash $X/fallback_v5b.sh
  echo "$(date +%T) continuation v7 FAILED ($WHY; see train.log after byte $LOG0); v5b resumed (fallback_v5b.sh)" >> $S/launch.txt
  tmux new -d -s guard5b "FB_STAGED=$BK/staged.py FB_START=$X/start_x2_parkour_v5b.sh bash $X/crashguard_v4.sh $L0 $E0"
fi
