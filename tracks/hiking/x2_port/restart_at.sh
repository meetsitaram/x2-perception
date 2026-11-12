#!/bin/bash
# Restart training right after a checkpoint so a config change takes effect with nothing lost:  bash restart_at.sh <run> <iteration>
T=/mnt/filesystem-g4/tracks/hiking; RUN=$T/runs/$1
until ls $RUN/*/model_$2.pt >/dev/null 2>&1; do sleep 10; done
sleep 3; bash $T/code/x2_port/apply.sh > /dev/null
touch $RUN/_supervisor/STOP
while tmux has -t hiking_x2 2>/dev/null; do sleep 3; done
sleep 3  # (the kill-every-GPU-process line was removed 2026-11-12: the supervisor stops its own training process)
bash $T/code/x2_port/start_x2_parkour_v1.sh
echo "$(date +%T) restarted from model_$2" >> $RUN/_supervisor/restarts.log
