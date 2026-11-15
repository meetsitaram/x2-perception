#!/bin/bash
# continue_v9.sh (2026-11-14, user decisions "Hiking v9 decisions"): ONE planned restart of x2_parkour_v4 from v8's newest
# checkpoint, armed on $S/GO_v9 (written by the hiking agent after the CPU pre-flight and the 19:00 v8 check data). The run then
# continues to 30,000. v9 = v8 (FAIR curriculum kept) + the paper's command mix (stand 0.05, no turn-in-place / slow-forward /
# slow-yaw shares, paper Rough (Stand) column ranges) + no touchdown reward (option t removed) + per-type curriculum log + contact
# weight scales at 1.0 (unchanged; for a later user decision). Code: v9/v9.py + v9/extended.py.v9 (one hook after apply_fair).
# Policy, critic, discriminator and optimisers carry over (same observations, actions, discriminator inputs).
# Pre-flight before v8 stops; construction check; 76-iteration trial; any failure -> fallback_v8.sh (v8 files, start_x2_parkour_v8.sh, guard8).
T=/mnt/filesystem-g4/tracks/hiking; S=$T/runs/x2_parkour_v4/_supervisor   # defined BEFORE ENVV
ENVV="HIKING_X2_V9=1 HIKING_X2_PAPER_STAND=1 HIKING_X2_STUMBLE_SCALE=1.0 HIKING_X2_CONTACT_SCALE=1.0 HIKING_X2_SCALE_SCOPE=paper HIKING_X2_FAIR=1 HIKING_X2_AIR_TD_PROGRESS=0 HIKING_X2_OPTS=a,c,d,f,r,s,k,y,o,z,w,g,v,j HIKING_X2_VX_BACK=0 HIKING_X2_SLOW_FWD_SHARE=0 HIKING_X2_SLOW_TURN_SHARE=0 HIKING_X2_SLOW_VX=0.1 HIKING_X2_TURN_DRIFT_WEIGHT=-2.0 HIKING_X2_STAGE=500,1500 HIKING_X2_CAM_RAND=1 HIKING_X2_ALIVE_EARLY=0.3 HIKING_X2_WAIT_EARLY=2 HIKING_X2_STYLE_EARLY=0.3 HIKING_X2_FALL_PENALTY=8 HIKING_X2_REF_START=0.5 HIKING_X2_AIR_TD_WEIGHT=0 HIKING_X2_AIR_TD_CAP=ref HIKING_X2_VX_MIN=0.45 HIKING_X2_TURN_SHARE=0 HIKING_X2_LEVELS_OUT=$S/terrain_levels.pt HIKING_X2_LEVELS_IN=$S/terrain_levels.pt HIKING_X2_STAND_SHARE=0.05 HIKING_X2_OVERSPEED_WEIGHT=-3.0 HIKING_X2_STOP_WEIGHT=-2.0 HIKING_X2_STOP_ANY_YAW=1 HIKING_X2_ROUTE_SHARE=0.7 HIKING_X2_SIDESTEP_SHARE=0.3 HIKING_X2_SIDESTEP_V=0.3 HIKING_X2_ROUTE_VY=0.2 HIKING_EXTENDED_TERRAINS=$T/data/extended_v4k HIKING_X2_MOTION_DIR=$T/data/motion_v7 HIKING_X2_MOTION_YAML=parkour_motion_without_run_x2_v4.yaml"
X=$T/code/x2_port; D=$X/files_instinctlab/tasks/parkour/config/x2
I=$T/code/hiking-in-the-wild/InstinctLab/source/instinctlab/instinctlab
NAME=x2_parkour_v4; RUN=$T/runs/$NAME
TR=$T/_trash/$(date +%F); mkdir -p $TR
note() { echo "$(date '+%F %T') $*" >> $TR/NOTES.txt; }
say() { echo "$(date -u '+%F %T') v9 $*" >> $S/launch.txt; }
for kv in $ENVV; do case "$kv" in HIKING_EXTENDED_TERRAINS=*|HIKING_X2_MOTION_DIR=*) [ -d "${kv#*=}" ] || { say "NOT started: ENVV path ${kv#*=} missing"; exit 1; };; esac; done
tree_ok() {  # one code source: every files_instinctlab file equals its installed copy, no extra files in the installed x2 dir
  local f; for f in $(cd $X/files_instinctlab && find . -type f ! -path "*__pycache__*"); do cmp -s $X/files_instinctlab/$f $I/$f || return 1; done
  for f in $(ls $I/tasks/parkour/config/x2 | grep -v __pycache__); do [ -e $D/$f ] || return 1; done; return 0; }
newest() { ls $RUN/*/model_*.pt 2>/dev/null | sed 's/.*model_\([0-9]*\)\.pt/\1/' | sort -n | tail -1; }
say "armed: waiting for $S/GO_v9 (\"now\" = newest checkpoint, else the next one)"
until [ -f $S/GO_v9 ]; do sleep 30; done
if grep -q now $S/GO_v9; then WAIT_FOR=$(newest); else WAIT_FOR=$(( ($(newest) / 125 + 1) * 125 )); fi
say "GO seen; restart from model_$WAIT_FOR"
until ls $RUN/*/model_${WAIT_FOR}.pt >/dev/null 2>&1; do sleep 20; done
sleep 30
# CPU pre-flight again, BEFORE v8 stops (v8 keeps training on any failure)
if ! bash $X/v9/precheck_v9.sh $X/v9/continue_v9.sh > $S/precheck_v9.log 2>&1; then
  say "NOT started: precheck failed ($(grep -a FAIL $S/precheck_v9.log | head -3 | tr '\n' ' ' | cut -c1-300)); v8 keeps training"; exit 1; fi
if ! bash $X/v9/cfgscan_v9.sh $S/cfgscan_v9.log; then
  say "NOT started: cfg scan failed ($(grep -a '^FAIL\|Error' $S/cfgscan_v9.log | head -3 | tr '\n' ' ' | cut -c1-300)); v8 keeps training"; exit 1; fi
if ! $HOME/hiking-env/bin/python $X/v8/test_fair.py > $S/test_fair_v9.log 2>&1 || ! $HOME/hiking-env/bin/python $X/v9/test_v9.py > $S/test_v9.log 2>&1; then
  say "NOT started: unit test failed (test_fair_v9.log / test_v9.log); v8 keeps training"; exit 1; fi
say "CPU pre-flight passed (precheck, cfgscan, unit tests)"
L0=$(grep -a "Curriculum/terrain_levels" $S/train.log | tail -20 | awk '{s+=$NF;n++} END{print s/n}')
E0=$(grep -a "Mean episode length" $S/train.log | tail -20 | awk '{s+=$NF;n++} END{print s/n}')
echo "$L0 $E0" > $S/baseline_v8_for_v9.txt
T0=$(date +%s)
# planned stop: the v8 guard goes first so it does not restart v8
tmux kill-session -t guard8 2>/dev/null && echo "$(date -u '+%F %T') crashguard_v8 stopped by continue_v9.sh (planned v9 restart)" >> $S/gates.txt
tmux kill-session -t cont8b 2>/dev/null && say "cont8b (v8b, armed since 08:58) disarmed: superseded by v9"
touch $S/STOP
while tmux has -t hiking_x2 2>/dev/null; do sleep 3; done
[ -f $S/STOP ] && { DST=$TR/STOP_${NAME}_$(date +%H%M%S); mv $S/STOP $DST; note "$DST: stop flag of $NAME, used by continue_v9.sh"; }
cp $S/terrain_levels.pt $S/terrain_levels_prestop_v9.pt
H=$(date +%H%M%S); BK=$TR/v8_files_v9_$H; mkdir -p $BK; cp $D/extended.py $BK/extended.py
note "$BK: extended.py of v8 (to $WAIT_FOR), replaced by v9/extended.py.v9 (+ v9.py)"
cat > $X/fallback_v8.sh <<EOF
#!/bin/bash
# restore v8's files and resume v8 (written by continue_v9.sh at $H)
cp $BK/extended.py $D/extended.py
[ -f $D/v9.py ] && mv $D/v9.py $TR/v9.py.src_$H
[ -f $I/tasks/parkour/config/x2/v9.py ] && mv $I/tasks/parkour/config/x2/v9.py $TR/v9.py.installed_$H
bash $X/apply.sh > /dev/null
cp $S/terrain_levels_prestop_v9.pt $S/terrain_levels.pt
bash $X/start_x2_parkour_v8.sh
tmux has -t guard8 2>/dev/null || tmux new -d -s guard8 "bash $X/v8/crashguard_v8.sh"
EOF
cp $X/v9/v9.py $D/v9.py; cp $X/v9/extended.py.v9 $D/extended.py
bash $X/apply.sh > /dev/null
if ! tree_ok; then
  bash $X/fallback_v8.sh; say "NOT started: installed InstinctLab copy differs from files_instinctlab after apply.sh; v8 resumed"; exit 1; fi
CK=$RUN/_check_v9_$(date +%H%M%S); mkdir -p $CK
( cd $T/code/hiking-in-the-wild/InstinctLab && env $ENVV OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 \
  timeout 1800 $HOME/hiking-env/bin/python $X/check_x2_env.py --headless --task Instinct-Parkour-ExtendedFixed-Amp-X2-v0 \
  --num_envs 256 --steps 150 --out $CK ) > $CK/check.log 2>&1
CKRC=$?
NR=$(grep -ao "\[X2 routes\] [0-9]* routes" $CK/check.log | head -1 | awk '{print $3}')
say "construction check exit $CKRC, routes ${NR:-none} (log $CK/check.log)"
if [ $CKRC != 0 ] || grep -aq "Traceback" $CK/check.log || [ "${NR:-0}" -lt 100 ] || ! grep -aq "\[X2 fair\]" $CK/check.log || ! grep -aq "\[X2 v9\] paper stand column" $CK/check.log; then
  bash $X/fallback_v8.sh; say "NOT started: construction check failed (exit $CKRC, routes ${NR:-none}, or no [X2 fair] / [X2 v9] line); v8 resumed"; exit 1; fi
# start line: v8's supervisor command with v9's ENVV (paths expanded here)
CMD=$(grep -o '\$HOME/hiking-env/bin/python [^"]*' $X/start_x2_parkour_v8.sh | head -1)
[ -n "$CMD" ] || { bash $X/fallback_v8.sh; say "NOT started: could not read the supervisor command from start_x2_parkour_v8.sh; v8 resumed"; exit 1; }
{ echo '#!/bin/bash'
  echo "# Resume x2_parkour_v4 as v9 (continue_v9.sh: v8 + paper command mix + no touchdown reward), generated $(date -u '+%F %T')."
  echo "tmux new -d -s hiking_x2 \"$ENVV $CMD\""; } > $X/start_x2_parkour_v9.sh
LOG0=$(stat -c %s $S/train.log 2>/dev/null || echo 0)
SUP0=$(grep -c ' start pid [0-9]* from iteration' $S/supervisor.log 2>/dev/null); SUP0=${SUP0:-0}
echo "continue from newest checkpoint (v9): $ENVV started $(date '+%Y-%m-%d %H:%M:%S')" >> $S/launch.txt
bash $X/start_x2_parkour_v9.sh
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
    P1=$(printf '%s' "$NEW" | grep -a "paper_cmd_episode" | tail -30 | awk '{s+=$NF;n++} END{print (n?s/n:-1)}')
    B1=$(printf '%s' "$NEW" | grep -ac "Curriculum/by_type/")
    say "trial after $N1 iterations (last 30): level $L1 (before $L0), episode $E1 (before $E0), paper-command episode share $P1 (v8 0.41), by_type lines $B1"
    # no turn-rate gate: v9 has no turn-in-place robots off the stand column (paper), so turn/rate_along_cmd is not comparable.
    # paper-command share >= 0.6 proves the paper mix is live (v8 0.41; expected about 0.9 x the non-route share).
    if ! python3 -c "import sys; sys.exit(0 if $N1 >= 75 and $L1 >= 0.7 * $L0 and $E1 >= 0.4 * $E0 and $P1 >= 0.6 and $B1 > 0 else 1)"; then
      OK=0; WHY="trial failed: level $L1 vs $L0, episode $E1 vs $E0, paper share $P1, by_type lines $B1 over the last 30 of $N1 iterations"
    fi
  fi
fi
if [ $OK = 1 ]; then
  say "is training, GPU idle about $(( $(date +%s) - T0 )) s"
  tmux new -d -s guard9 "bash $X/v9/crashguard_v9.sh"
else
  touch $S/STOP
  while tmux has -t hiking_x2 2>/dev/null; do sleep 3; done
  [ -f $S/STOP ] && { DST=$TR/STOP_${NAME}_$(date +%H%M%S); mv $S/STOP $DST; note "$DST: stop flag of the failed v9 continuation"; }
  bash $X/fallback_v8.sh
  say "FAILED ($WHY; see train.log after byte $LOG0); v8 resumed (fallback_v8.sh), guard8 restarted"
fi
