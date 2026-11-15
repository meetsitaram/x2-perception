#!/bin/bash
# One planned restart to load the rollout guard (FINDINGS H7b): wait for model_22125 of the run resumed at 00:36, then STOP,
# install guard2 ppo.py / wasabi.py (originals to _trash), restart v9 (same flags) and replace crashguard_v9 by crashguard_v9b.
T=/mnt/filesystem-g4/tracks/hiking; X=$T/code/x2_port; RUN=$T/runs/x2_parkour_v4; S=$RUN/_supervisor; G2=$X/v9/guard2
A=$T/code/hiking-in-the-wild/instinct_rl/instinct_rl/algorithms; PY=$HOME/hiking-env/bin/python
say() { echo "$(date -u '+%F %T') v9 guard2 $*" >> $S/launch.txt; }
WAIT=${1:-22125}
say "armed: restart at model_$WAIT to load the rollout guard"
until ls $RUN/2*/model_${WAIT}.pt >/dev/null 2>&1; do sleep 15; done
sleep 20
cmp -s $A/ppo.py $G2/orig/ppo.py && cmp -s $A/wasabi.py $G2/orig/wasabi.py || { say "NOT installed: installed files differ from guard2/orig"; exit 1; }
$PY -m py_compile $G2/new/ppo.py $G2/new/wasabi.py || { say "NOT installed: py_compile failed"; exit 1; }
tmux kill-session -t guard9 2>/dev/null && echo "$(date -u '+%F %T') crashguard_v9 stopped by install_guard2.sh (planned restart for the rollout guard)" >> $S/gates.txt
touch $S/STOP
for i in $(seq 200); do tmux has -t hiking_x2 2>/dev/null || break; sleep 3; done
TR=$T/_trash/$(date +%F); mkdir -p $TR; H=$(date +%H%M%S)
cp $A/ppo.py $TR/ppo.py.before_guard2_$H; cp $A/wasabi.py $TR/wasabi.py.before_guard2_$H
echo "$(date '+%F %T') $TR/{ppo,wasabi}.py.before_guard2_$H: instinct_rl originals replaced by v9/guard2/new (rollout guard)" >> $TR/NOTES.txt
cp $G2/new/ppo.py $A/ppo.py; cp $G2/new/wasabi.py $A/wasabi.py
if ! $PY -c "import instinct_rl.algorithms.wasabi, instinct_rl.algorithms.ppo as p; assert hasattr(p.PPO, '_guard_rollout_bad')" > $S/guard2_import.log 2>&1; then
  cp $TR/ppo.py.before_guard2_$H $A/ppo.py; cp $TR/wasabi.py.before_guard2_$H $A/wasabi.py; say "import failed, originals restored"
fi
DST=$TR/STOP_x2_parkour_v4_$H; mv $S/STOP $DST; echo "$(date '+%F %T') $DST: stop flag of install_guard2.sh" >> $TR/NOTES.txt
K=$(ls $RUN/2*/model_*.pt | sed 's/.*model_\([0-9]*\)\.pt/\1/' | sort -n | tail -1)
mkdir -p $S/levels_hist; cp $S/terrain_levels.pt $S/levels_hist/tl_$K.pt
bash $X/start_x2_parkour_v9.sh
tmux new -d -s guard9 "bash $X/v9/crashguard_v9b.sh"
say "installed ($(cmp -s $A/ppo.py $G2/new/ppo.py && echo new || echo ORIGINAL) ppo.py); v9 restarted from model_$K; crashguard_v9b in tmux guard9"
touch $G2/INSTALL.DONE
