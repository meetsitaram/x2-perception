#!/usr/bin/env bash
# The standard ACP depth-student showcase renders (user, 2026-11-16: "use the same approach ... save it so rendered scripts in
# future follow the same"). Every render:
#   - ACP's own overlay drawn from the maps ACP was GIVEN (the student's estimate in a student run, never the simulator's):
#     planner 17 x 11 tiles (green level / orange high / blue low), tracker 9 x 7 black dots, planned landing rings, stance squares
#   - the chest depth camera inset bottom right (ACP's chest camera pose, distance scale)
#   - our apparatus with its real textures (terrain-park's CAD export, absolute asset paths), chase camera raised (-48 deg) so
#     its walls do not hide the robot; the held-out ladder courses with the standard chase camera
#   - five clips: our apparatus route (seed 2007), its door + both 4x4s with a fixed camera on the far side facing the robot
#     (until it reaches button 2 behind the back 4x4), stairs up, stairs down, a 4x4 log (eval ladder, held out of training)
#
#   models/acp_depth/render_showcase.sh STUDENT.pt OUT_DIR "label" [--noise] [--anchor]
#     --noise   D435-like noise on the camera in the loop (ACPD_DEPTH_NOISE=1); the inset shows the clean render
#     --anchor  foot-anchored map (ACPD_ANCHOR=1)
# Output: OUT_DIR/{apparatus,stairs_up,stairs_down,log_4x4}.mp4 + check_*.jpg (one frame each, to look at before sharing).
set -u
cd "$(dirname "$0")/../.."
STUDENT=$(readlink -f "$1"); V=$2; LABEL=$3; shift 3
EXTRA_ENV=""
for a in "$@"; do case "$a" in --noise) EXTRA_ENV="$EXTRA_ENV ACPD_DEPTH_NOISE=1";; --anchor) EXTRA_ENV="$EXTRA_ENV ACPD_ANCHOR=1";; esac; done
mkdir -p "$V"
PY=$HOME/Projects/gwbc-x2/.venv/bin/python
export PA_ROBOT_DIR=$HOME/Projects/gwbc-x2/gear_sonic/data/assets/robot_description
ACP=$(cd ../../../release/ckpt/acp 2>/dev/null && pwd || echo "$HOME/projects/x2-perception/release/ckpt/acp")
TP=$HOME/Projects/terrain-park/apparatus/ours
ENV="EVAL_REGRESSION=0 ACPD_MODE=student ACPD_STUDENT=$STUDENT ACP_HANDOVER_DIR=$ACP ACP_GOAL_R=0.2 ACP_LOOKAHEAD=0.3 X2_EXCLUDE_PAIRS=hip
     PA_ROBOT_DIR=$HOME/Projects/gwbc-x2/gear_sonic/data/assets/robot_description $EXTRA_ENV"
# 1. the four episodes, recording the maps ACP was given (ACPD_EXTRA)
( ulimit -v 10000000; env $ENV ACPD_EXTRA=$V/extra_app $PY eval/harness.py --adapter acp_depth:ACPDepth --apparatus ours --trials 1 \
  --seed0 2007 --keep-traj all --out $V/app > $V/app.log 2>&1 ) &
for t in acp_stairs_up_L2 acp_stairs_down_L2 acp_log_L2; do
  ( ulimit -v 10000000; env $ENV ACPD_EXTRA=$V/extra_$t $PY eval/ladder.py --adapter acp_depth:ACPDepth --step 20 --tests $t \
    --trials 1 --out $V/ladder > $V/$t.log 2>&1 ) &
done
wait
# 2. the apparatus dressing with absolute asset paths (terrain-park's apparatus.xml uses paths relative to its own folder)
sed -e "s|file=\"\.\./cad/|file=\"$TP/cad/|g" -e "s|file=\"meshes/|file=\"$TP/sim/meshes/|g" $TP/sim/apparatus.xml > $V/apparatus_abs.xml
#    ... and lowered onto the eval's physics deck (the CAD's z = 0 is the venue floor, ~0.108 m below the deck top)
( ulimit -v 6000000; $PY models/acp_depth/align_apparatus.py $V/apparatus_abs.xml $V/apparatus_aligned.xml ) >> $V/render.log 2>&1
# 3. renders (imageio from ~/.cache/x2-render-pydeps, without its numpy: the eval venv's numpy stays)
R() { ( ulimit -v 12000000; PYTHONPATH=$HOME/.cache/x2-render-pydeps RENDER_MINIMAL=1 MUJOCO_GL=glfw APPARATUS_SCENE=$PWD/$V/apparatus_aligned.xml \
        "$@" ) >> $V/render.log 2>&1; }
R env RENDER_CHASE_ELEV=-48 RENDER_CHASE_DIST=3.6 $PY models/acp_depth/render_vis.py $V/app 2007 $V/apparatus.mp4 \
  "ACP on depth only ($LABEL): height maps + footholds from the depth student | our apparatus" --extra $V/extra_app/ep0001.npz &
T1=$($PY -c "import numpy as np; z=np.load('$V/app/traj/2007.npz'); w=z['wp']; i=int(np.argmax(w>=10)) if (w>=10).any() else len(w)-1; print(min(float(z['t'][i])+1.5, float(z['t'][-1])))")
R env RENDER_FIXED="3.0,1.2,0.25,180,-32,3.4" $PY models/acp_depth/render_vis.py $V/app 2007 $V/apparatus_4x4s.mp4 \
  "ACP on depth only ($LABEL) | door and both 4x4s, camera facing the robot" --extra $V/extra_app/ep0001.npz --t0 0 --t1 $T1 &
R $PY models/acp_depth/render_vis.py $V/ladder/train/acp_stairs_up_L2 1000 $V/stairs_up.mp4 "ACP on depth only ($LABEL) | stairs up, held-out course" \
  --extra $V/extra_acp_stairs_up_L2/ep0001.npz &
R $PY models/acp_depth/render_vis.py $V/ladder/train/acp_stairs_down_L2 1000 $V/stairs_down.mp4 "ACP on depth only ($LABEL) | stairs down, held-out course" \
  --extra $V/extra_acp_stairs_down_L2/ep0001.npz &
R $PY models/acp_depth/render_vis.py $V/ladder/train/acp_log_L2 1000 $V/log_4x4.mp4 "ACP on depth only ($LABEL) | 4x4 log, held-out course" \
  --extra $V/extra_acp_log_L2/ep0001.npz &
wait
for f in apparatus apparatus_4x4s stairs_up stairs_down log_4x4; do
  n=$(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 $V/$f.mp4)
  ffmpeg -loglevel error -y -i $V/$f.mp4 -vf "select='eq(n\,$((n/5)))+eq(n\,$((n/2)))+eq(n\,$((n*4/5)))',scale=420:-1,tile=3x1" -frames:v 1 $V/check_$f.jpg
done
for f in apparatus apparatus_4x4s stairs_up stairs_down log_4x4; do      # a phone / web copy under the 30 MB upload limit
  ffmpeg -loglevel error -y -i $V/$f.mp4 -vf scale=1024:-2 -c:v libx264 -crf 28 -preset medium $V/${f}_small.mp4
done
grep -h "acp on ours\|^acp_" $V/*.log | cut -c1-100
ls -la $V/*.mp4
echo DONE
