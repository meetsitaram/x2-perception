#!/bin/bash
# Pause training at a checkpoint, run the paper evaluation (X2) on that checkpoint for a bounded time, restart training.
#   bash pause_and_eval.sh <run name> <checkpoint iteration> <budget seconds> <terrains comma list> [num_envs] [extra eval args]
# Never more than one Isaac instance: training is stopped first (it resumes from its newest checkpoint afterwards).
T=/mnt/filesystem-g4/tracks/hiking
NAME=$1; IT=$2; BUDGET=$3; TERRAINS=$4; N=${5:-1000}; shift 5
RUN=$T/runs/$NAME; PY=$HOME/hiking-env/bin/python
OUT=$T/eval/paper_eval/${NAME}_$IT; mkdir -p $OUT
until ls $RUN/*/model_$IT.pt >/dev/null 2>&1; do sleep 10; done
CK=$(ls $RUN/*/model_$IT.pt | head -1)
sleep 3; touch $RUN/_supervisor/STOP
while tmux has -t hiking_x2 2>/dev/null; do sleep 3; done
sleep 3  # (the kill-every-GPU-process line was removed 2026-11-12: the supervisor stops its own training process)
T0=$(date +%s)
echo "{\"track\": \"hiking\", \"t\": $T0, \"when\": \"$(date '+%Y-%m-%d %H:%M:%S')\", \"state\": \"paper eval at checkpoint $IT, training paused for at most $BUDGET s\", \"checkpoint\": \"$CK\", \"checkpoint_iteration\": $IT}" > /mnt/filesystem-g4/status/hiking.json
cd $T/code/hiking-in-the-wild
HIKING_OBS_CHECK=$OUT/obs_check_sim.npz OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 timeout $BUDGET $PY $T/code/x2_port/eval_paper_x2.py \
  --checkpoint $CK --num_envs $N --terrains $TERRAINS --out $OUT/results_$(date +%H%M%S).json "$@" > $OUT/eval_$(date +%H%M%S).log 2>&1
echo "$(date +%T) eval exit $? after $(( $(date +%s) - T0 )) s" >> $OUT/pause.log
sleep 3  # (the kill-every-GPU-process line was removed 2026-11-12: the supervisor stops its own training process)
bash $T/code/x2_port/start_x2_parkour_v1.sh
echo "$(date +%T) training restarted, paused $(( $(date +%s) - T0 )) s" >> $OUT/pause.log
