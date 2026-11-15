#!/bin/bash
# One GPU job after another. Stop the queue: touch runs/QUEUE_STOP (and runs/STOP for the running job).
cd /mnt/filesystem-g4/tracks/ssr/code
R=/mnt/filesystem-g4/tracks/ssr/runs
rm -f $R/QUEUE_STOP
PY=../env/venv/bin/python
[ -f $R/QUEUE_STOP ] && exit 0
$PY scripts/supervise_node.py --name d3 --iters 300 --num_envs 8192 -- --flat_only --only_terms lin_vel,ang_vel --w_style 0 --prior_min_speed 0.2 --rsi_min_speed 0.3 --plant mc_20261024 --camera head
[ -f $R/QUEUE_STOP ] && exit 0
$PY scripts/supervise_node.py --name d4 --iters 300 --num_envs 8192 -- --flat_only --w_style 0 --prior_min_speed 0.2 --rsi_min_speed 0.3 --plant mc_20261024 --camera head
[ -f $R/QUEUE_STOP ] && exit 0
$PY scripts/supervise_node.py --name r5 --iters 20000 --num_envs 8192 -- --adv_norm separate --prior_min_speed 0.2 --rsi_min_speed 0.3 --w_style 1.0 --plant mc_20261024 --camera torso60
