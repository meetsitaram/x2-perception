#!/bin/bash
# Replaces the r5 line of queue2.sh (queue2 is stopped after d4 by runs/QUEUE_STOP). Waits for d4 to finish, then runs r5 under the supervisor.
# Stop before it starts: touch runs/QUEUE3_STOP. Stop r5: touch runs/STOP.
cd /mnt/filesystem-g4/tracks/ssr/code
R=/mnt/filesystem-g4/tracks/ssr/runs
PY=../env/venv/bin/python
sleep 30
while pgrep -f "[s]upervise_node.py" > /dev/null || pgrep -f "[s]sr.train" > /dev/null || pgrep -f "[q]ueue2.sh" > /dev/null; do sleep 20; done
[ -f $R/QUEUE3_STOP ] && exit 0
$PY scripts/supervise_node.py --name r5 --iters 20000 --num_envs 8192 -- --adv_norm separate --prior_min_speed 0.2 --rsi_min_speed 0.3 --w_style 1.0 --plant mc_20261024 --camera chest_rs --amp_file data/amp_v2.npz --prior_balance
