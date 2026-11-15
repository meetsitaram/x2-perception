#!/bin/bash
# One planned restart to load the action clip (FINDINGS H7c): at the next checkpoint, STOP, install aclip/new/vecenv_wrapper.py
# (original to _trash), restart v9 with the same flags plus HIKING_X2_ACTION_CLIP (and optional per-joint bounds), keep
# crashguard_v9b. Usage: install_aclip.sh <clip> [joint bounds "5:50"] [checkpoint to wait for, default next multiple of 125]
T=/mnt/filesystem-g4/tracks/hiking; X=$T/code/x2_port; RUN=$T/runs/x2_parkour_v4; S=$RUN/_supervisor; AC=$X/v9/aclip
F=$T/code/hiking-in-the-wild/InstinctLab/source/instinctlab/instinctlab/utils/wrappers/instinct_rl/vecenv_wrapper.py
PY=$HOME/hiking-env/bin/python
say() { echo "$(date -u '+%F %T') v9 aclip $*" >> $S/launch.txt; }
CLIP=${1:?clip value}; JOINT=${2:-}
K0=$(ls $RUN/2*/model_*.pt | sed 's/.*model_\([0-9]*\)\.pt/\1/' | sort -n | tail -1)
WAIT=${3:-$((K0 + 125))}
say "armed: restart at model_$WAIT with HIKING_X2_ACTION_CLIP=$CLIP joint bounds '${JOINT}'"
until ls $RUN/2*/model_${WAIT}.pt >/dev/null 2>&1; do sleep 15; done
sleep 20
cmp -s $F $AC/orig/vecenv_wrapper.py || { say "NOT installed: installed wrapper differs from aclip/orig"; exit 1; }
$PY -m py_compile $AC/new/vecenv_wrapper.py || { say "NOT installed: py_compile failed"; exit 1; }
# start script with the clip variables; it replaces start_x2_parkour_v9.sh (original to _trash) so crashguard_v9b's
# rollbacks and supervisor restarts keep the clip
sed "s#tmux new -d -s hiking_x2 \"#tmux new -d -s hiking_x2 \"HIKING_X2_ACTION_CLIP=$CLIP HIKING_X2_ACTION_CLIP_JOINT=$JOINT #" $X/start_x2_parkour_v9.sh > $AC/start_x2_parkour_v9_aclip.sh
grep -q "HIKING_X2_ACTION_CLIP=$CLIP" $AC/start_x2_parkour_v9_aclip.sh || { say "NOT installed: start script edit failed"; exit 1; }
tmux kill-session -t guard9 2>/dev/null && echo "$(date -u '+%F %T') crashguard_v9b stopped by install_aclip.sh (planned restart for the action clip)" >> $S/gates.txt
touch $S/STOP
for i in $(seq 200); do tmux has -t hiking_x2 2>/dev/null || break; sleep 3; done
TR=$T/_trash/$(date +%F); mkdir -p $TR; H=$(date +%H%M%S)
cp $F $TR/vecenv_wrapper.py.before_aclip_$H
echo "$(date '+%F %T') $TR/vecenv_wrapper.py.before_aclip_$H: InstinctLab original replaced by v9/aclip/new (action clip)" >> $TR/NOTES.txt
cp $AC/new/vecenv_wrapper.py $F
cp $X/start_x2_parkour_v9.sh $TR/start_x2_parkour_v9.sh.before_aclip_$H; cp $AC/start_x2_parkour_v9_aclip.sh $X/start_x2_parkour_v9.sh
echo "$(date '+%F %T') $TR/start_x2_parkour_v9.sh.before_aclip_$H: v9 start script before the clip variables" >> $TR/NOTES.txt
DST=$TR/STOP_x2_parkour_v4_$H; mv $S/STOP $DST; echo "$(date '+%F %T') $DST: stop flag of install_aclip.sh" >> $TR/NOTES.txt
K=$(ls $RUN/2*/model_*.pt | sed 's/.*model_\([0-9]*\)\.pt/\1/' | sort -n | tail -1)
mkdir -p $S/levels_hist; cp $S/terrain_levels.pt $S/levels_hist/tl_$K.pt
bash $X/start_x2_parkour_v9.sh
tmux new -d -s guard9 "bash $X/v9/crashguard_v9b.sh"
say "installed ($(cmp -s $F $AC/new/vecenv_wrapper.py && echo new || echo ORIGINAL) wrapper); v9 restarted from model_$K with clip $CLIP '${JOINT}'; crashguard_v9b in tmux guard9; check [ACLIP] lines in train.log"
touch $AC/INSTALL.DONE
