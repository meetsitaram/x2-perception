#!/usr/bin/env bash
# Stage 1 of the depth -> height-scan student, unattended: wait for the teacher data (collect.py), train on the local GPU, score
# ACP with the student on our apparatus (same 50 seeds as the perfect-map baseline, 46/50 = 92 % on this PC, 2026-11-16).
#   models/acp_depth/run_v1.sh            (logs: models/acp_depth/runs/run_v1.log; markers: runs/student_v1/{TRAINED,SCORED})
set -uo pipefail
cd "$(dirname "$0")/../.."
R=models/acp_depth/runs
D=$R/data_v1; S=$R/student_v1
ACP=$(cd ../../../release/ckpt/acp 2>/dev/null && pwd || echo "$HOME/projects/x2-perception/release/ckpt/acp")
until grep -q "^DONE" $R/data_v1.log 2>/dev/null; do sleep 30; done
echo "$(date +%T) data: $(grep '^DONE' $R/data_v1.log)"
systemd-run --user --scope -q -p MemoryMax=24G "$HOME/miniconda3/envs/g1_train/bin/python" models/acp_depth/train.py \
  --data $D/data --out $S --epochs 40 > $S.train.log 2>&1 || { echo "$(date +%T) TRAIN FAILED"; tail -5 $S.train.log; exit 1; }
touch $S/TRAINED; echo "$(date +%T) trained: $(tail -1 $S.train.log)"
( ulimit -v 16000000; EVAL_REGRESSION=0 ACPD_MODE=student ACPD_STUDENT=$S/student.pt ACP_HANDOVER_DIR=$ACP ACP_GOAL_R=0.2 \
  ACP_LOOKAHEAD=0.3 X2_EXCLUDE_PAIRS=hip PA_ROBOT_DIR=$HOME/Projects/gwbc-x2/gear_sonic/data/assets/robot_description \
  timeout 10800 "$HOME/Projects/gwbc-x2/.venv/bin/python" eval/harness.py --adapter acp_depth:ACPDepth --apparatus ours \
  --trials 50 --seed0 2000 --out $S/apparatus50 > $S.apparatus50.log 2>&1 )
touch $S/SCORED; echo "$(date +%T) scored: $(grep 'acp on ours' $S.apparatus50.log | cut -c1-160)"
echo DONE
