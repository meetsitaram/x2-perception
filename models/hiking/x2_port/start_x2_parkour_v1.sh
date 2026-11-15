#!/bin/bash
# Start (or restart) the X2 paper-terrain run under the supervisor, in tmux session hiking_x2. Resumes from the newest checkpoint.
tmux new -d -s hiking_x2 "$HOME/hiking-env/bin/python /mnt/filesystem-g4/tracks/hiking/code/x2_port/supervise_x2.py --task Instinct-Parkour-Target-Amp-X2-v0 --name x2_parkour_v1 --total 30000 --num_envs 8192 2>&1 | tee -a /mnt/filesystem-g4/tracks/hiking/runs/x2_parkour_v1_supervisor_tmux.log"
