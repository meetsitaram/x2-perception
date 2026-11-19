#!/bin/bash
# PC2: the chest depth on the robot's face screen (PC3) while our stack runs, as gear-sonic/interact/face_camera_feed.sh
# does for the head camera, but one long-running process pushing the newest frame as a 1 s still (no 3 s clips, no per-update
# ROS start): the screen lags by about one copy + play round trip. Ctrl-C / TERM restores the vendor idle face.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PC3="${X2_PC3_HOST:-$(cat ~/.x2_pc3_host)}"   # PC3 address: kept on PC2, out of the repo
DEST_DIR=/opt/x2_interact/media
CLIP_S="${CLIP_S:-0}"         # 0 = the newest frame as a 1 s still, refreshed as fast as copy + play allow
WORK=/tmp/x2p_face
PY=/home/run/gear-sonic/venv/bin/python
FACE=/home/run/gear-sonic/interact/x2_face.sh
export LD_LIBRARY_PATH=/opt/ros/humble/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/opt/ros/humble/local/lib/python3.10/dist-packages:/opt/ros/humble/lib/python3.10/site-packages
mkdir -p "$WORK"
trap '"$FACE" off; exit 0' INT TERM
"$PY" "$HERE/face_depth_capture.py" --loop --out "$WORK" --pc3 "$PC3"
"$FACE" off
