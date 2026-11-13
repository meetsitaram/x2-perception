#!/bin/bash
# restore v4i's files and resume v4i (written by continue_v4j.sh at 205258)
for f in extended.py routes.py keep.py; do cp /mnt/filesystem-g4/tracks/hiking/_trash/2026-11-12/v4i_files_205258/$f /mnt/filesystem-g4/tracks/hiking/code/x2_port/files_instinctlab/tasks/parkour/config/x2/$f; done
rm -f /mnt/filesystem-g4/tracks/hiking/code/x2_port/files_instinctlab/tasks/parkour/config/x2/contact.py
bash /mnt/filesystem-g4/tracks/hiking/code/x2_port/apply.sh > /dev/null
rm -f /mnt/filesystem-g4/tracks/hiking/code/hiking-in-the-wild/InstinctLab/source/instinctlab/instinctlab/tasks/parkour/config/x2/contact.py
cp /mnt/filesystem-g4/tracks/hiking/runs/x2_parkour_v4/_supervisor/terrain_levels_prestop.pt /mnt/filesystem-g4/tracks/hiking/runs/x2_parkour_v4/_supervisor/terrain_levels.pt
bash /mnt/filesystem-g4/tracks/hiking/code/x2_port/start_x2_parkour_v4i.sh
