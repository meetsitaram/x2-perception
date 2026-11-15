#!/usr/bin/env bash
cd /mnt/filesystem-g4/tracks/acp/code; export CUDA_VISIBLE_DEVICES=; export ACP_PLANT=mc_20261024; PY=$HOME/acp_env/bin/python
$PY tools/weak_sets.py /mnt/filesystem-g4/tracks/acp/eval/paper_eval/s1_v1_002250_plant_mc_20261024_rollouts.pt cfg/stage1_sets.json 2>&1 | grep -v "^Warp\|^Module\|^   [A-Z\"/]\|^$" > /mnt/filesystem-g4/tracks/acp/eval/paper_eval/s1_v1_002250_plant_mc_20261024_weak_sets.txt
$PY -m acp.planner label /mnt/filesystem-g4/tracks/acp/eval/paper_eval/s1_v1_002250_plant_mc_20261024_rollouts.pt /mnt/filesystem-g4/tracks/acp/runs/planner_s1_002250_mc_20261024/data.pt
$PY -m acp.planner train /mnt/filesystem-g4/tracks/acp/runs/planner_s1_002250_mc_20261024/data.pt /mnt/filesystem-g4/tracks/acp/runs/planner_s1_002250_mc_20261024/planner_mlp.pt --arch mlp --epochs 30
$PY -m acp.planner export /mnt/filesystem-g4/tracks/acp/runs/planner_s1_002250_mc_20261024/planner_mlp.pt /mnt/filesystem-g4/tracks/acp/runs/planner_s1_002250_mc_20261024/planner_mlp.onnx
cp /mnt/filesystem-g4/tracks/acp/runs/s1_v1/ckpt_002250.pt /mnt/filesystem-g4/tracks/acp/eval/handover_pair_s1_002250_mc_20261024_planner/tracker_ckpt_002250.pt
$PY tools/handover.py /mnt/filesystem-g4/tracks/acp/eval/handover_pair_s1_002250_mc_20261024_planner/tracker_ckpt_002250.pt /mnt/filesystem-g4/tracks/acp/eval/handover_pair_s1_002250_mc_20261024_planner
$PY /mnt/filesystem-g4/eval/code/tools/export_actor.py /mnt/filesystem-g4/tracks/acp/eval/handover_pair_s1_002250_mc_20261024_planner/tracker_ckpt_002250.pt /mnt/filesystem-g4/tracks/acp/eval/handover_pair_s1_002250_mc_20261024_planner/actor.npz --meta /mnt/filesystem-g4/tracks/acp/eval/handover_pair_s1_002250_mc_20261024_planner/meta.json --goal-r 0.3
cp /mnt/filesystem-g4/tracks/acp/runs/planner_s1_002250_mc_20261024/planner_mlp.onnx /mnt/filesystem-g4/tracks/acp/runs/planner_s1_002250_mc_20261024/planner_mlp.pt /mnt/filesystem-g4/tracks/acp/eval/handover_pair_s1_002250_mc_20261024_planner/
$PY tools/card_planner.py /mnt/filesystem-g4/tracks/acp/eval/handover_pair_s1_002250_mc_20261024_planner /mnt/filesystem-g4/tracks/acp/eval/handover_pair_s1_002250_mc_20261024_planner/planner_mlp.pt /mnt/filesystem-g4/tracks/acp/eval/handover_pair_s1_002250_mc_20261024_planner/planner_mlp.onnx
timeout 2400 $PY tools/pair_eval.py /mnt/filesystem-g4/tracks/acp/eval/handover_pair_s1_002250_mc_20261024_planner/tracker_ckpt_002250.pt /mnt/filesystem-g4/tracks/acp/runs/planner_s1_002250_mc_20261024/planner_mlp.pt /mnt/filesystem-g4/tracks/acp/eval/handover_pair_s1_002250_mc_20261024_planner/mlp 12
echo ALL_DONE
