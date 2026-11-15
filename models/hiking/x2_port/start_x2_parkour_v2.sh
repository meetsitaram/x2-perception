#!/bin/bash
# Start or resume run 2 (torso camera, mc_20261024 gains) under the supervisor.
tmux new -d -s hiking_x2 "HIKING_X2_CAM_RAND=${HIKING_X2_CAM_RAND:-0} $HOME/hiking-env/bin/python /mnt/filesystem-g4/tracks/hiking/code/x2_port/supervise_x2.py --task Instinct-Parkour-TorsoCam-Amp-X2-v0 --name x2_parkour_v2 --total 30000 --num_envs 8192 2>&1 | tee -a /mnt/filesystem-g4/tracks/hiking/runs/x2_parkour_v2_supervisor_tmux.log"
