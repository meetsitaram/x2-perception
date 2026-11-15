#!/bin/bash
# restore v5b's staged.py and resume v5b (written by continue_v7.sh at 133333)
cp /mnt/filesystem-g4/tracks/hiking/_trash/2026-11-13/v5b_files_v7_133333/staged.py /mnt/filesystem-g4/tracks/hiking/code/x2_port/files_instinctlab/tasks/parkour/config/x2/staged.py
bash /mnt/filesystem-g4/tracks/hiking/code/x2_port/apply.sh > /dev/null
cp /mnt/filesystem-g4/tracks/hiking/runs/x2_parkour_v4/_supervisor/terrain_levels_prestop_v7.pt /mnt/filesystem-g4/tracks/hiking/runs/x2_parkour_v4/_supervisor/terrain_levels.pt
bash /mnt/filesystem-g4/tracks/hiking/code/x2_port/start_x2_parkour_v5b.sh
