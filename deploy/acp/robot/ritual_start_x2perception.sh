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
  && export PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1"   # tiny matrices: one math thread (2026-11-19)
ACP=$X2P/models/acp
STUDENT=$X2P/models/student.onnx
CAM_PITCH="${X2P_CAM_PITCH:-8.25}"; CAM_ROLL="${X2P_CAM_ROLL:--3.5}"   # the mount calibration (mount_calib.py; standing check 2026-11-19)
# Dry run while $X2P/DRY_RUN exists (or X2P_DRY_RUN=1): the whole chain, but MC is never stopped and the node never commands
# (--no-publish): the ritual, haptics, gate, node and screen can be tested with MC holding the robot. rm $X2P/DRY_RUN = live.
ACCEL="${X2P_ACCEL:-cuda}"   # planner + student: cuda (falls back to the CPU); trt only once its engines are cached (first build is slow)
DRY=0; { [ -f $X2P/DRY_RUN ] || [ "${X2P_DRY_RUN:-0}" = 1 ]; } && DRY=1
mkdir -p $X2P/log
echo "$(date +%F_%T) X2-PERCEPTION RITUAL START$([ $DRY = 1 ] && echo ' (DRY RUN: MC stays, nothing commanded)')" >> $LOG
# the outcome on the gamepad (SONIC's pad daemon: rumble PULL on :5570): two long buzzes = live, our node has the robot (limp,
# then the assist); one medium = dry run up; three short = refused (gate, MC stop or a stale session; see this log). The pad already gave 3 pulses + a long buzz for the ritual itself.
rumble() {  # strength ms count gap_ms
  $PY -c "import zmq,json,time; s=zmq.Context().socket(zmq.PUSH); s.setsockopt(zmq.LINGER,500); s.connect('tcp://127.0.0.1:5570'); \
s.send_string(json.dumps(dict(strength=$1,ms=$2,count=$3,gap_ms=$4))); time.sleep(0.2)" >/dev/null 2>&1 &
}
refuse() {  # why: log it, buzz three times, and end the sessions this ritual started so the next one starts clean
  echo "$(date +%F_%T) $1" >> $LOG
  rumble 1.0 120 3 150
  for s in $STARTED; do tmux kill-session -t $s 2>/dev/null && echo "$(date +%F_%T) $s: ended (start refused)" >> $LOG; done
  exit 1
}
STARTED=""
start_tmux() {  # name, command
  tmux has-session -t $1 2>/dev/null && { echo "$(date +%F_%T) $1: running -- skipped" >> $LOG; return 0; }
  tmux new-session -d -s $1 "$2 ; rc=\$?; echo \"\$(date +%F_%T) $1: EXITED rc=\$rc\" >> $LOG"
  echo "$(date +%F_%T) $1: STARTED" >> $LOG
  STARTED="$STARTED $1"
}
# depth publisher restarted whenever it exits (2026-11-19: the capture stalled mid-walk, the publisher exited as designed, and
# nothing restarted it: no depth for the rest of the run, so the node refused to walk). The node stands while depth is stale.
start_tmux x2p_depth "$ROSENV && while true; do $PY $X2P/deploy/acp/robot/d435_pub.py --dev ${X2P_D435_DEV:-/dev/video11}; \
  echo \"\$(date +%F_%T) d435_pub exited rc=\$?, restarting in 2 s\" >> $LOG; sleep 2; done 2>&1 | tee -a $X2P/log/depth.log"
# camera recordings for reviewing walks (head camera from its ROS topic, chest D435 colour small): log/video/run_<time>/
start_tmux x2p_rec "$ROSENV && export FASTRTPS_DEFAULT_PROFILES_FILE=/agibot/software/entry/cfg/ros_dds_configuration.xml \
  && $PY $X2P/deploy/acp/robot/cam_record.py --out $X2P/log/video >> $X2P/log/cam_record.log 2>&1"   # no pipe: a broken pipe once kept the videos from being finalised
start_tmux x2p_pad   "$PY $X2P/deploy/hiking/hiking_deploy/pad.py --source zmq 2>&1 | tee -a $X2P/log/pad.log"
# the C++ command relay: writes the bus at a steady 500 Hz from what the node hands it (FINDINGS X9, X10); silent until fed
RELAY_BIN=$X2P/ws_robot/install/x2p_cmd_relay/lib/x2p_cmd_relay/x2p_cmd_relay
start_tmux x2p_relay "$ROSENV && $RELAY_BIN 2>&1 | tee -a $X2P/log/relay.log"

# gate: fresh joints, IMU and depth before anything is commanded (MC still has the robot)
bash -c "$ROSENV && $PY $X2P/deploy/acp/robot/gate.py" >> $LOG 2>&1
if [ $? -ne 0 ]; then
  refuse "GATE FAILED -- MC NOT STOPPED, DEPLOY NOT STARTED"
fi
echo "$(date +%F_%T) GATE PASSED" >> $LOG

if [ $DRY = 1 ]; then
  start_tmux x2p_deploy "$ROSENV && cd $X2P/deploy/acp && $PY -m acp_deploy.node --acp $ACP --student $STUDENT --no-publish \
    --cam-pitch $CAM_PITCH --cam-roll $CAM_ROLL --accel $ACCEL --blackbox $X2P/log/blackbox --relay-prefix /x2p/relay --imu-on-pelvis 2>&1 | tee -a $X2P/log/deploy.log"
  start_tmux x2p_face "bash $X2P/deploy/acp/robot/face_depth_feed.sh >> $X2P/log/face.log 2>&1"
  echo "$(date +%F_%T) DRY RUN: MC left running, node observe-only" >> $LOG; rumble 0.5 300 1 0; exit 0
fi
# our node already running (after an e-stop): hand it the start instead (re-warm; user, 2026-11-19). In DAMP it assists to the
# stand pose and the policy follows; in any other state (e-stop still set) it drops the request and logs why. MC stays as it is.
# A dry-run node never commands, so it cannot be restarted live this way: refuse.
if tmux has-session -t x2p_deploy 2>/dev/null; then
  if pgrep -f "acp_deploy.node.*--no-publish" >/dev/null; then
    refuse "x2p_deploy is a DRY-RUN node -- not restarted live; stop our sessions first"
  fi
  touch /dev/shm/x2p_start
  echo "$(date +%F_%T) RE-START: start handed to the running node (assist if limp; ignored if the e-stop is still set)" >> $LOG
  rumble 1.0 400 2 250; exit 0
fi
# no relay = the node's commands would reach nothing once MC is off: refuse
sleep 0.5
if ! pgrep -x x2p_cmd_relay >/dev/null; then   # the binary itself (a -f match would also hit its tmux shell)
  refuse "x2p_relay NOT RUNNING ($RELAY_BIN) -- MC NOT STOPPED, DEPLOY NOT STARTED"
fi
# MC off (the operator holds the robot), then our node starts limp and assists to the stand pose
# MC already stopped (nothing commands the legs for 1 s) is fine (user, 2026-11-19); else stop it, with a timeout (the CLI has
# hung before), and confirm on the bus that nothing commands any more
MCCHK="$ROSENV && $PY $X2P/deploy/acp/robot/gate.py --mc-silent"
if bash -c "$MCCHK 1.0" >> $LOG 2>&1; then
  echo "$(date +%F_%T) MC already stopped (no leg commands on the bus)" >> $LOG
else
  # MC runs on PC1: its environment manager's HTTP API first (as SONIC's deploy_x2.sh mc_em_post), else the aima CLI, which
  # only knows PC2's own apps on this firmware (2026-11-19: "app mc does not exist"). PC1's address: ~/.x2_pc1_host (not in
  # the repo), else PC1_HOST from SONIC's robot_env.env.
  PC1="$(cat ~/.x2_pc1_host 2>/dev/null || (. /home/run/gear-sonic/robot_env.env >/dev/null 2>&1; echo "${PC1_HOST:-}"))"
  rc=1
  if [ -n "$PC1" ]; then
    curl -fsS -X POST -H 'Content-Type: application/json' --connect-timeout 3 --max-time 8 -d '{"app_name":"mc"}' \
      "http://$PC1:${PC1_EM_PORT:-50080}/json/stop_app" >> $LOG 2>&1; rc=$?
    echo "$(date +%F_%T) MC stop via PC1 EM HTTP: rc $rc" >> $LOG
  fi
  if [ $rc -ne 0 ]; then
    timeout 20 aima em stop-app mc < /dev/null >> $LOG 2>&1; rc=$?
    echo "$(date +%F_%T) MC stop via aima CLI: rc $rc" >> $LOG
  fi
  sleep 1
  if ! bash -c "$MCCHK 1.5" >> $LOG 2>&1; then
    refuse "MC STOP FAILED (rc $rc, still commanding) -- DEPLOY NOT STARTED"
  fi
  echo "$(date +%F_%T) MC stopped (rc $rc, bus silent)" >> $LOG
fi
rm -f /dev/shm/x2p_start                      # no stale start request for the new node
start_tmux x2p_deploy "$ROSENV && cd $X2P/deploy/acp && $PY -m acp_deploy.node --acp $ACP --student $STUDENT --assist-on-start \
  --cam-pitch $CAM_PITCH --cam-roll $CAM_ROLL --accel $ACCEL --blackbox $X2P/log/blackbox --relay-prefix /x2p/relay --imu-on-pelvis --start-file /dev/shm/x2p_start 2>&1 | tee -a $X2P/log/deploy.log"
start_tmux x2p_face "bash $X2P/deploy/acp/robot/face_depth_feed.sh >> $X2P/log/face.log 2>&1"
rumble 1.0 400 2 250
echo "$(date +%F_%T) LIVE: node started (limp, then the assist)" >> $LOG
