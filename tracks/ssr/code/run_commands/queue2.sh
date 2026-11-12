#!/bin/bash
# Runs after the job that is training now: camera check for chest_rs, then d4, then r5. Stop: touch runs/QUEUE_STOP (and runs/STOP for the running job).
cd /mnt/filesystem-g4/tracks/ssr/code
R=/mnt/filesystem-g4/tracks/ssr/runs
PY=../env/venv/bin/python
while pgrep -f "[s]upervise_node.py" > /dev/null; do sleep 20; done
rm -f $R/QUEUE_STOP
REPORT_CAMS=chest_rs X2_PLANT=mc_20261024 timeout 600 $PY scripts/report_model.py ../eval/model_mc_20261024_chest_rs.json ../renders/camera_check > $R/report_model_chest_rs.log 2>&1
[ -f $R/QUEUE_STOP ] && exit 0
$PY scripts/supervise_node.py --name d4 --iters 300 --num_envs 8192 -- --flat_only --w_style 0 --prior_min_speed 0.2 --rsi_min_speed 0.3 --plant mc_20261024 --camera head
[ -f $R/QUEUE_STOP ] && exit 0
$PY scripts/supervise_node.py --name r5 --iters 20000 --num_envs 8192 -- --adv_norm separate --prior_min_speed 0.2 --rsi_min_speed 0.3 --w_style 1.0 --plant mc_20261024 --camera chest_rs
