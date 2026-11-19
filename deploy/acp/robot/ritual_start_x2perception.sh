#!/bin/bash
# PC2: x2-perception (ACP on the chest depth camera) launched by the gamepad start ritual, in place of SONIC's
# ritual_start_sonic.sh: x2-pad-autopair.service runs pc2_pad_daemon.py --start-cmd <this script>, so L1+R1+L2+R2 held 3 s,
# then Y, runs it while the operator holds the robot.
#
# Chain (each in its own tmux session, as SONIC's script does; log in $LOG):
#   x2p_depth   chest D435 -> /x2_acp/chest_depth (80 x 45 z-depth)
#   x2p_pad     SONIC's pad daemon (ZMQ :5569) -> our layout and e-stop -> the node (UDP :5690); vibrations back on :5570
#   gate        joint + IMU + depth topics fresh, else STOP (nothing commanded, MC untouched)
#   MC stop     aima em stop-app mc (the operator holds the robot from here)
#   x2p_deploy  ACP node: limp, straight into the leaky assist (the ritual was the start sequence), then the policy
#   x2p_face    the chest depth on the face screen (PC3), as interact/face_camera_feed.sh
# Stop: the pad e-stop (A+X, then pump L2 and R2) makes the robot limp; x2perception_stop.sh ends the sessions.
# Nothing here installs anything: SONIC's venv (onnxruntime, numpy, pyzmq, cv2) plus the system ROS, paths spelled out.
set -u
X2P=/home/run/x2-perception
LOG=$X2P/log/ritual_fired.log
PY=/home/run/gear-sonic/venv/bin/python
AIMDK=/agibot/software/housekeeper/bin/aimdk_msgs
ROSENV="source /opt/ros/humble/setup.bash && export AMENT_PREFIX_PATH=$AIMDK:\$AMENT_PREFIX_PATH \
  && export LD_LIBRARY_PATH=$AIMDK/lib:\$LD_LIBRARY_PATH \
  && export PYTHONPATH=$X2P/deploy/hiking:$X2P/deploy/acp:$AIMDK/local/lib/python3.10/dist-packages:\${PYTHONPATH:-} \
  && export PYTHONUNBUFFERED=1"
ACP=$X2P/models/acp
STUDENT=$X2P/models/student.onnx
CAM_PITCH="${X2P_CAM_PITCH:-0}"; CAM_ROLL="${X2P_CAM_ROLL:-0}"     # the mount calibration (deploy/hiking/tools/mount_calib.py)
mkdir -p $X2P/log
echo "$(date +%F_%T) X2-PERCEPTION RITUAL START" >> $LOG
start_tmux() {  # name, command
  tmux has-session -t $1 2>/dev/null && { echo "$(date +%F_%T) $1: running -- skipped" >> $LOG; return 0; }
  tmux new-session -d -s $1 "$2 ; rc=\$?; echo \"\$(date +%F_%T) $1: EXITED rc=\$rc\" >> $LOG"
  echo "$(date +%F_%T) $1: STARTED" >> $LOG
}
start_tmux x2p_depth "$ROSENV && $PY $X2P/deploy/acp/robot/d435_pub.py --dev ${X2P_D435_DEV:-/dev/video11} 2>&1 | tee -a $X2P/log/depth.log"
start_tmux x2p_pad   "$PY $X2P/deploy/hiking/hiking_deploy/pad.py --source zmq 2>&1 | tee -a $X2P/log/pad.log"

# gate: fresh joints, IMU and depth before anything is commanded (MC still has the robot)
bash -c "$ROSENV && $PY $X2P/deploy/acp/robot/gate.py" >> $LOG 2>&1
if [ $? -ne 0 ]; then
  echo "$(date +%F_%T) GATE FAILED -- MC NOT STOPPED, DEPLOY NOT STARTED" >> $LOG; exit 1
fi
echo "$(date +%F_%T) GATE PASSED" >> $LOG

# MC off (the operator holds the robot), then our node starts limp and assists to the stand pose
if ! aima em stop-app mc >> $LOG 2>&1; then
  echo "$(date +%F_%T) MC STOP FAILED -- DEPLOY NOT STARTED" >> $LOG; exit 1
fi
echo "$(date +%F_%T) MC stopped" >> $LOG
start_tmux x2p_deploy "$ROSENV && cd $X2P/deploy/acp && $PY -m acp_deploy.node --acp $ACP --student $STUDENT --assist-on-start \
  --cam-pitch $CAM_PITCH --cam-roll $CAM_ROLL 2>&1 | tee -a $X2P/log/deploy.log"
start_tmux x2p_face "bash $X2P/deploy/acp/robot/face_depth_feed.sh >> $X2P/log/face.log 2>&1"
