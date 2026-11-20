#!/usr/bin/env bash
# Hiking sim stack in the gr00t-x2sim docker image: gwbc-x2's MuJoCo bridge (the robot), our depth stand-in (the chest
# camera) and our deploy node, one container, host network, ROS_DOMAIN_ID 73 on loopback (cannot reach a real X2).
# The pad runs on the host and sends UDP to the deploy node.
#
#   deploy/hiking/sim/run_sim.sh [--policy hiking|acp] [--gantry] [--scene park|flat|<course>] [--depth sim|flat] [--no-viewer] [--pad gamepad|keyboard|none|script:FILE   (keyboard = keys in the launching terminal)]
#   deploy/hiking/sim/run_sim.sh --stop
#
# Logs: deploy/hiking/logs/<time>/{bridge,depth,deploy,pad}.log. Paths: GWBC (~/Projects/gwbc-x2), TERRAIN_PARK
# (~/Projects/terrain-park), MODEL (deploy/hiking/models/x2-hiking-in-the-wild), PYDEPS (~/.cache/x2-hiking-pydeps:
# onnxruntime for the image's python 3.10). Pad: SONIC's layout (hiking_deploy/pad.py).
# --policy acp: ACP + the fine depth student instead (deploy/acp/PLAN.md): ACP_DIR (the hand-over folder), ACP_STUDENT
# (student.pt), its 80 x 45 sim camera (deploy/acp/sim/acp_depth_pub.py); CAM_PITCH / CAM_ROLL: the simulated camera's offset;
# ACP_NODE_ARGS: extra acp_deploy.node arguments. BRIDGE_SPAWN="x,y,yaw_deg": start there (bridge_run.py).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEP="$(cd "$HERE/.." && pwd)"
REPO="$(cd "$DEP/../.." && pwd)"
GWBC="${GWBC:-$HOME/Projects/gwbc-x2}"
TP="${TERRAIN_PARK:-$HOME/Projects/terrain-park}"
MODEL="$(readlink -f "${MODEL:-$DEP/models/x2-hiking-in-the-wild}")"
PYDEPS="${PYDEPS:-$HOME/.cache/x2-hiking-pydeps}"
IMAGE="${IMAGE:-gr00t-x2sim:latest}"
NAME=x2hiking
ACP_DIR="${ACP_DIR:-$HOME/projects/x2-perception/release/ckpt/acp}"
ACP_STUDENT="$(readlink -f "${ACP_STUDENT:-$REPO/models/acp_depth/runs/fine_v3/student.pt}")"
SCENE=park; VIEWER=1; PAD=gamepad; DEPTH=sim; POLICY=hiking

if [[ "${1:-}" == "--stop" ]]; then
  docker stop -t 3 "${NAME}_view" >/dev/null 2>&1 || true
  docker stop -t 3 "$NAME" >/dev/null 2>&1 && echo "stack stopped" || echo "stack not running"
  PIDF="$(dirname "${BASH_SOURCE[0]}")/../logs/pose_log.pid"
  [[ -f "$PIDF" ]] && { kill "$(cat "$PIDF")" 2>/dev/null || true; rm -f "$PIDF"; }
  PADF="$(dirname "${BASH_SOURCE[0]}")/../logs/pad.pid"       # a pad left running would fight the next one (SIGINT: pygame)
  if [[ -f "$PADF" ]]; then P=$(cat "$PADF"); kill -INT "$P" 2>/dev/null && echo "pad stopped"; sleep 1; kill -9 "$P" 2>/dev/null; rm -f "$PADF"; fi
  exit 0
fi
while [[ $# -gt 0 ]]; do
  case "$1" in
    --scene) SCENE="$2"; shift 2 ;;
    --policy) POLICY="$2"; shift 2 ;;
    --gantry) # the robot hangs from a soft gantry band, limp and crouched, feet on the floor; the band drops 3 s after the
              # deploy node's policy starts (it asks bridge_run), as one lowers a gantry once the controller holds the robot
              # (the bridge's pull point is z = 1.0 + band-length: -0.32 puts it just above standing height)
              export BRIDGE_START="--init-pose gantry-hang --band-length -0.32 --band-kp-mult 0.3 --band-release-after-s -1"; shift ;;
    --no-viewer) VIEWER=0; shift ;;
    --pad) PAD="$2"; shift 2 ;;
    --depth) DEPTH="$2"; shift 2 ;;     # sim = ray-cast the scene (the camera); flat = hiking_deploy/flat_depth.py (robot-side,
                                        # IMU + joints only: what a camera-less real robot would feed the policy)
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac
done
NEED=("$MODEL/adapter_card.json"); [[ "$POLICY" == acp ]] && NEED=("$ACP_DIR/actor.npz" "$ACP_DIR/planner_attn.onnx" "$ACP_STUDENT")
[[ "$POLICY" == acp && "$DEPTH" != sim ]] && { echo "--policy acp needs --depth sim" >&2; exit 2; }
for p in "$GWBC/gear_sonic_deploy/scripts/x2_mujoco_ros_bridge.py" "${NEED[@]}" "$TP/apparatus/ours/eval/route.json" "$PYDEPS/onnxruntime"; do
  [[ -e "$p" ]] || { echo "missing: $p (see deploy/hiking/STATUS.md)" >&2; exit 1; }
done
docker ps -q --filter "name=^${NAME}(_view)?$" | grep -q . && { echo "stack already running ($0 --stop first)" >&2; exit 1; }

RD="$GWBC/gear_sonic/data/assets/robot_description"
S="$DEP/logs/$(date +%Y%m%d_%H%M%S)"; mkdir -p "$S"
SC="$HERE/scenes/$SCENE"
DOCK=(docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp -v "$HOME:$HOME" -w "$DEP"
      -e PA_ROBOT_DIR="$RD" -e PA_APPARATUS_DIR="$TP/apparatus/ours" -e PA_TERRAIN_DIR="$TP/terrains/extended_v1" -e APPARATUS_MODEL)
if [[ ! -f "$SC/scene.xml" ]]; then
  echo "[1/3] building scene '$SCENE' ..."
  "${DOCK[@]}" "$IMAGE" python3 "$HERE/make_scene.py" --scene "$SCENE" --out "$SC" > "$S/scene.log" 2>&1 \
    || { tail -20 "$S/scene.log"; exit 1; }
  tail -1 "$S/scene.log"
else
  echo "[1/3] scene '$SCENE' (cached: $SC; delete it to rebuild)"
fi

FLAT="$HERE/scenes/flat"
if [[ "$DEPTH" == flat && ! -f "$FLAT/scene.xml" ]]; then
  echo "      building the flat scene for --depth flat ..."
  "${DOCK[@]}" "$IMAGE" python3 "$HERE/make_scene.py" --scene flat --out "$FLAT" > "$S/scene_flat.log" 2>&1 || { tail -20 "$S/scene_flat.log"; exit 1; }
fi
VIEW=0; X11=()
if (( VIEWER )) && [[ -n "${DISPLAY:-}" ]]; then
  VIEW=1      # our mirror viewer (own process); the bridge's in-loop viewer halves the sim speed on the park
  X11=(-e DISPLAY -e QT_X11_NO_MITSHM=1 -v /tmp/.X11-unix:/tmp/.X11-unix:rw)
  [[ -n "${XAUTHORITY:-}" ]] && X11+=(-e XAUTHORITY -v "$XAUTHORITY:$XAUTHORITY:ro")
fi
echo "[2/3] container: bridge + depth + deploy (logs $S)"
ROSENV=(-e ROS_DOMAIN_ID=73 -e ROS_LOCALHOST_ONLY=1 -e ROS_LOG_DIR="$S/ros" -e PYTHONUNBUFFERED=1 -e PYTHONPATH="$DEP:$REPO/deploy/acp:$PYDEPS")
# each process logs its exit code to exits.log (on 2026-11-15 a viewer crash took the whole stack down twice, unexplained)
"${DOCK[@]}" -d --name "$NAME" --network host --ipc shareable "${ROSENV[@]}" -e HIKING_DEBUG -e BRIDGE_IMU -e BRIDGE_START -e BRIDGE_SPAWN -e ACP_NODE_ARGS -e ACP_TRUTH -e ACP_RELAY -e DEPTH="$DEPTH" \
  -e POLICY="$POLICY" -e ACP_DIR="$ACP_DIR" -e ACP_STUDENT="$ACP_STUDENT" -e CAM_PITCH="${CAM_PITCH:-0}" -e CAM_ROLL="${CAM_ROLL:-0}" "$IMAGE" bash -c '
    source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash
    S="$1"; shift; SC="$1"; shift; MODEL="$1"; shift; GWBC="$1"; shift; RD="$1"; shift
    run() { local n="$1"; shift; "$@" > "$S/$n.log" 2>&1; echo "$(date +%T) $n exited $?" >> "$S/exits.log"; }
    run bridge python3 "'"$HERE"'/bridge_run.py" "$GWBC/gear_sonic_deploy/scripts/x2_mujoco_ros_bridge.py" --mjcf "$SC/scene.xml" \
      --imu-from ${BRIDGE_IMU:-torso} ${BRIDGE_START:---init-pose stand_settled --no-elastic-band} --no-scene-zmq --no-hand-zmq &
    sleep 4
    if [[ "$POLICY" == acp ]]; then
      run depth python3 "'"$REPO"'/deploy/acp/sim/acp_depth_pub.py" --scene "$SC/scene.xml" --cam-pitch "$CAM_PITCH" --cam-roll "$CAM_ROLL" ${ACP_TRUTH:+--truth-scans} &
      if [[ -n "$ACP_RELAY" ]]; then   # the C++ command relay (deploy/acp/robot/cmd_relay, built in ws_sim) writes the bus
        run relay "'"$REPO"'/deploy/acp/robot/cmd_relay/ws_sim/install/x2p_cmd_relay/lib/x2p_cmd_relay/x2p_cmd_relay" &
        RELAY_ARGS="--relay-prefix /x2p/relay"
      fi
      run deploy python3 -m acp_deploy.node --acp "$ACP_DIR" --student "$ACP_STUDENT" --sim-reset-port 5691 ${HIKING_DEBUG:+--debug} $RELAY_ARGS $ACP_NODE_ARGS &
    else
      if [[ "$DEPTH" == flat ]]; then run depth python3 -m hiking_deploy.flat_depth --mjcf "'"$FLAT"'/scene.xml" --model "$MODEL" &
      else run depth python3 "'"$HERE"'/depth_pub.py" --scene "$SC/scene.xml" --model "$MODEL" & fi
      run deploy python3 -m hiking_deploy.node --model "$MODEL" --sim-reset-port 5691 ${HIKING_DEBUG:+--debug} --limits-mjcf "$RD/mjcf/x2_ultra.xml" &
    fi
    trap "echo \"\$(date +%T) container got TERM/INT\" >> \"$S/exits.log\"; kill 0" TERM INT; wait' _ "$S" "$SC" "$MODEL" "$GWBC" "$RD" > "$S/container.id"
if (( VIEW )); then   # the viewer in a container of its own: its crash can never stop the robot; it reopens after a crash
  "${DOCK[@]}" -d --name "${NAME}_view" --network host --ipc "container:$NAME" --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all \
    "${ROSENV[@]}" "${X11[@]}" -w "$HERE" "$IMAGE" bash -c '
      source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash
      until python3 mirror_view.py --scene "$2/scene.xml" >> "$1/viewer.log" 2>&1; do
        echo "$(date +%T) viewer crashed ($?), reopening" >> "$1/exits.log"; sleep 2; done' _ "$S" "$SC" > /dev/null
fi
sleep 8
for f in bridge depth deploy; do
  [[ -f "$S/$f.log" ]] && printf '  %-7s %s\n' "$f" "$(grep -v '^\s*$' "$S/$f.log" | tail -1 | cut -c1-150)"
done
docker ps -q --filter "name=^${NAME}$" | grep -q . || { echo "container exited; see $S/*.log" >&2; exit 1; }
ln -sfn "$S" "$DEP/logs/latest"

PY="${PAD_PYTHON:-$GWBC/.venv/bin/python}"
"$PY" "$HERE/pose_log.py" --out "$S/pose.csv" > "$S/pose_log.log" 2>&1 &   # ground-truth pelvis pose (sim only); --stop ends it
echo $! > "$DEP/logs/pose_log.pid"
case "$PAD" in
  none) echo "[3/3] no pad. Run: $PY $DEP/hiking_deploy/pad.py --source gamepad|keyboard" ;;
  script:*) echo "[3/3] scripted pad: ${PAD#script:}"; [[ -f "$DEP/logs/pad.pid" ]] && kill -0 "$(cat "$DEP/logs/pad.pid")" 2>/dev/null \
       && { echo "another pad is running (pid $(cat "$DEP/logs/pad.pid")); $0 --stop first" >&2; exit 1; }
     "$PY" "$DEP/hiking_deploy/pad.py" --source script --script "${PAD#script:}" > "$S/pad.log" 2>&1
     sleep 1; "$PY" "$HERE/pose_log.py" --summary "$S/pose.csv" "$S/deploy.log" | tee "$S/summary.txt" ;;
  keyboard|terminal) echo "[3/3] keys in this terminal (q quits the pad; $0 --stop ends the sim)."
     "$PY" "$DEP/hiking_deploy/pad.py" --source terminal ;;
  *) echo "[3/3] gamepad (SONIC layout): hold L1+L2+R1+R2 3 s, then Y = stand; hold L2 + left stick = walk (up) / turn (sideways); right stick = camera;"
     echo "      L1/R1 speed -/+; e-stop A+X + pump both triggers; View/Back 1 s = reset. Ctrl-C quits the pad; $0 --stop ends the sim."
     ( "$PY" "$DEP/hiking_deploy/pad.py" --source "$PAD" & echo $! > "$DEP/logs/pad.pid"; wait ) 2>&1 | tee "$S/pad.log" ;;
esac
