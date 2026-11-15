#!/bin/bash
# restore v8's files and resume v8 (written by continue_v9.sh at 194614)
cp /mnt/filesystem-g4/tracks/hiking/_trash/2026-11-14/v8_files_v9_194614/extended.py /mnt/filesystem-g4/tracks/hiking/code/x2_port/files_instinctlab/tasks/parkour/config/x2/extended.py
[ -f /mnt/filesystem-g4/tracks/hiking/code/x2_port/files_instinctlab/tasks/parkour/config/x2/v9.py ] && mv /mnt/filesystem-g4/tracks/hiking/code/x2_port/files_instinctlab/tasks/parkour/config/x2/v9.py /mnt/filesystem-g4/tracks/hiking/_trash/2026-11-14/v9.py.src_194614
[ -f /mnt/filesystem-g4/tracks/hiking/code/hiking-in-the-wild/InstinctLab/source/instinctlab/instinctlab/tasks/parkour/config/x2/v9.py ] && mv /mnt/filesystem-g4/tracks/hiking/code/hiking-in-the-wild/InstinctLab/source/instinctlab/instinctlab/tasks/parkour/config/x2/v9.py /mnt/filesystem-g4/tracks/hiking/_trash/2026-11-14/v9.py.installed_194614
bash /mnt/filesystem-g4/tracks/hiking/code/x2_port/apply.sh > /dev/null
cp /mnt/filesystem-g4/tracks/hiking/runs/x2_parkour_v4/_supervisor/terrain_levels_prestop_v9.pt /mnt/filesystem-g4/tracks/hiking/runs/x2_parkour_v4/_supervisor/terrain_levels.pt
bash /mnt/filesystem-g4/tracks/hiking/code/x2_port/start_x2_parkour_v8.sh
tmux has -t guard8 2>/dev/null || tmux new -d -s guard8 "bash /mnt/filesystem-g4/tracks/hiking/code/x2_port/v8/crashguard_v8.sh"
