#!/bin/bash
# restore v7's files and resume v7 (written by continue_v8.sh at 090025)
cp /mnt/filesystem-g4/tracks/hiking/_trash/2026-11-14/v7_files_v8_090025/extended.py /mnt/filesystem-g4/tracks/hiking/code/x2_port/files_instinctlab/tasks/parkour/config/x2/extended.py
[ -f /mnt/filesystem-g4/tracks/hiking/code/x2_port/files_instinctlab/tasks/parkour/config/x2/fair.py ] && mv /mnt/filesystem-g4/tracks/hiking/code/x2_port/files_instinctlab/tasks/parkour/config/x2/fair.py /mnt/filesystem-g4/tracks/hiking/_trash/2026-11-14/fair.py.src_090025
[ -f /mnt/filesystem-g4/tracks/hiking/code/hiking-in-the-wild/InstinctLab/source/instinctlab/instinctlab/tasks/parkour/config/x2/fair.py ] && mv /mnt/filesystem-g4/tracks/hiking/code/hiking-in-the-wild/InstinctLab/source/instinctlab/instinctlab/tasks/parkour/config/x2/fair.py /mnt/filesystem-g4/tracks/hiking/_trash/2026-11-14/fair.py.installed_090025
bash /mnt/filesystem-g4/tracks/hiking/code/x2_port/apply.sh > /dev/null
cp /mnt/filesystem-g4/tracks/hiking/runs/x2_parkour_v4/_supervisor/terrain_levels_prestop_v8.pt /mnt/filesystem-g4/tracks/hiking/runs/x2_parkour_v4/_supervisor/terrain_levels.pt
bash /mnt/filesystem-g4/tracks/hiking/code/x2_port/start_x2_parkour_v7.sh
tmux has -t guard7 2>/dev/null || tmux new -d -s guard7 "bash /mnt/filesystem-g4/tracks/hiking/code/x2_port/crashguard_v7.sh"
