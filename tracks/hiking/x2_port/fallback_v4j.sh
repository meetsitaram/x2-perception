#!/bin/bash
# restore v4j's files and resume v4j (written by continue_v5.sh at 012847)
for f in extended.py routes.py keep.py contact.py staged.py fixed.py; do cp /mnt/filesystem-g4/tracks/hiking/_trash/2026-11-13/v4j_files_v5_012847/$f /mnt/filesystem-g4/tracks/hiking/code/x2_port/files_instinctlab/tasks/parkour/config/x2/$f; done
rm -f /mnt/filesystem-g4/tracks/hiking/code/x2_port/files_instinctlab/tasks/parkour/config/x2/slow.py
bash /mnt/filesystem-g4/tracks/hiking/code/x2_port/apply.sh > /dev/null
rm -f /mnt/filesystem-g4/tracks/hiking/code/hiking-in-the-wild/InstinctLab/source/instinctlab/instinctlab/tasks/parkour/config/x2/slow.py
cp /mnt/filesystem-g4/tracks/hiking/runs/x2_parkour_v4/_supervisor/terrain_levels_prestop.pt /mnt/filesystem-g4/tracks/hiking/runs/x2_parkour_v4/_supervisor/terrain_levels.pt
bash /mnt/filesystem-g4/tracks/hiking/code/x2_port/start_x2_parkour_v4j.sh
