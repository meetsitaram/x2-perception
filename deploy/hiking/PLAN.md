# Hiking on the real X2: sim-first deployment plan (2026-11-15)

**Start here, Linux session.** The user rebooted this PC from Windows into Ubuntu to work on this. Read this file, then
`agentic/docs/TOKEN_EFFICIENCY.md`, then work milestone by milestone. Ask the user before anything that touches the
real robot. Prompt to start a new Claude session on Linux:

> Read deploy/hiking/PLAN.md in x2-perception and start at milestone M0. Report after each milestone.

## Decisions so far (user, 2026-11-15; also in agentic/docs/USER_RULES.md)
- First model on the robot: **Hiking, checkpoint 24,500** (best apparatus checkpoint; the same export is on Hugging Face,
  `tinkerbuggy/x2-hiking-in-the-wild`).
- **Sim first:** "get the sim stack ready first with the deployed binary in docker sim that i can control and command using
  gamepad before we test on the real robot".
- Deploy code lives in **this repo** (`deploy/hiking/`), not in gwbc-x2.
- Camera computer: try **PC2 (Jetson Orin NX)** first; if the RealSense cannot be connected there, an **external Jetson Thor
  strapped to the robot's back**.
- Real-robot tests, later: flat walk, minor slopes, a single 4x4 (0.089 m) in front of the robot in a gantry.
- Chest camera: RealSense D435 on the printed bracket in `docs/hardware/chest_camera_mount/` (Hiking variant H, trained pose
  (131, 0, 259) mm on torso_link, 61.6 deg down). `docs/camera/README.md` still says 53 deg; that was an earlier fit to a
  reference photo and is superseded by the bracket, which reproduces the trained pose exactly. Measure it once mounted.

## Reference: how SONIC runs on the X2 (repo gwbc-x2, branch main; it is a sparse checkout on Windows, full on Linux)
- Robot: PC2 = Jetson Orin NX runs the deploy; PC1 runs the vendor motion controller (MC) and Environment Manager (EM).
- ROS 2 Humble + `aimdk_msgs`. Joint state/command `/aima/hal/joint/{leg,waist,arm,head}/{state,command}` (position,
  velocity, stiffness, damping), IMU `/aima/hal/imu/torso/state` (sensor_msgs/Imu, xyzw), SensorDataQoS.
- Deploy node `gear_sonic_deploy/src/x2/agi_x2_deploy_onnx_ref/` (C++): 50 Hz inference, separate 500 Hz command writer
  (the writer once starved to 3.4 Hz when it shared a thread), ONNX Runtime CPU. States STANDBY -> INIT -> WAIT_FOR_CONTROL
  -> CONTROL -> SAFE_HOLD (tilt watchdog) -> RECOVER; RAMP_OUT -> HOLD_FOR_MC -> MC STAND_DEFAULT on stop.
- MC handover: `stop_app mc` via EM HTTP API (fallback `aima em stop-app mc`), `gear_sonic_deploy/deploy_x2.sh:20-50`.
  MC must never share the bus with the deploy.
- Sim: `gear_sonic/scripts/sim_onnx_planner.sh` -> `simstack_local.sh` runs the same deploy binary plus
  `gear_sonic_deploy/scripts/x2_mujoco_ros_bridge.py` (MuJoCo 1 kHz, joint state 200 Hz) in the `gr00t-x2sim` container
  (Ubuntu 22.04, `gear_sonic_deploy/docker_x2/docker-compose.yml`: host network, ROS_DOMAIN_ID=73, NVIDIA runtime, X11).
  Viewer on the host (`sim_mirror_viewer.py`). Docs: `docs/x2/F09_mujoco_sim.md`, `F10_real_robot_colcon.md`,
  `ARCHITECTURE.md`, `F01_gamepad.md`, `gamepad_cheatsheet.md`.
- Gamepad: Xbox-class or DualSense via pygame on the host (sim) / xpadneo Bluetooth on PC2 (robot). Deadman L2; e-stop
  A+X then both triggers (latched; clear with the same gesture >= 5 s later); start chord L1+L2+R1+R2 3 s then Y.
- Licence: gwbc-x2 carries NVIDIA's dual licence notice. **Do not copy its code into this public repo.** Use it at run
  time (its docker image, bridge and message packages) and write our own node here.

## What the Hiking policy needs (source of truth: the export's `adapter_card.json`)
Export: Hugging Face `tinkerbuggy/x2-hiking-in-the-wild` (`policy.onnx`, or `0-depth_encoder.onnx` + `actor.onnx`,
`adapter_card.json`, `obs_check.npz`, `run_params/env.yaml`, `agent.yaml`, `checkpoint/model_24500.pt`). Local Windows
copy: `release/ckpt/hiking/x2_parkour_v4_24500/` (git-ignored).
- 50 Hz control. Actions: 29 joints (Isaac order, see `action_joint_order`); head yaw/pitch held at default (kp 3.4 / 10).
  Target = default + scale x action. Gains, scales and default pose: Hiking's own (`kp`, `kd`, `action_scale`,
  `default_pose` in the card; e.g. hip 100/4, knee 150/5, waist 40), NOT SONIC's armature rule. Load from the card at run
  time; do not hard-code.
- **Action clip:** 24,500 was trained WITHOUT a clip (the +-20 clip went in at 26,250 or later, `models/hiking/LEDGER.md`
  row 35c). In MuJoCo the policy holds waist_pitch at |a| 18-28 normally; every other joint stays <= 18.1. Deploy with the
  user's clip: **+-20 on 28 joints, +-50 on waist_pitch** (binds never on normal steps), plus a joint-limit clamp on
  targets and a soft-start ramp.
- Proprio obs, 8-step history, oldest first (800 values): base_ang_vel x 0.25 and projected gravity **in the pelvis
  frame**; velocity command (vx, vy, wz) fed directly; joint_pos (relative to default), joint_vel x 0.05, last actions.
  Reference implementation: `eval/adapters/hiking.py` (`proprio`, `observe`), checked against `obs_check.npz`.
- **Pelvis frame from the torso IMU:** the robot publishes only the torso IMU. Rotate it into the pelvis frame with the
  waist joints (yaw, pitch, roll) by FK: R_pelvis = R_torso * R_waist(q)^T; omega_pelvis = R_waist(q) * omega_torso -
  (waist joint rates term). Validate in sim (MuJoCo gives both) before the robot.
- Depth: 8 frames of 36 x 32 (rows x cols), 0..1. Raw 64 x 36 pinhole 87 x 58 deg, z-depth; crop columns 16..48; Gaussian
  blur 3 x 3 sigma 1 (reflect); clip 0..3 m; / 3. Frames every 5th control step from a 37-frame history at 50 Hz, oldest
  first, the newest one control step old. Camera on torso_link (0.131, 0, 0.259) m, 61.6 deg down. Implementation:
  `eval/adapters/hiking.py` (`depth_frame`, ring buffer in `observe`).
- Commands seen in training: vx 0..0.8 m/s (forward only; 0.45..1.0 on rough, 0.45..0.8 on stairs/gaps), vy +-0.3,
  wz +-1.0; standing (all zero) is in the mix. After a zero command the policy runs on about 0.34 m before stopping.

## Architecture (ours, in deploy/hiking/)
```
gamepad (pygame, host) --ROS /x2_hiking/cmd_vel + /x2_hiking/mode--> hiking_deploy node (rclpy)
MuJoCo bridge (gwbc-x2, container)  --/aima/hal/joint/*/state, /aima/hal/imu/torso/state-->  hiking_deploy
sim depth publisher (ours) or RealSense node --/x2_hiking/chest_depth (32FC1, metres, 64 x 36 after resampling)--> hiking_deploy
hiking_deploy --/aima/hal/joint/*/command (500 Hz writer, own thread/callback group)--> bridge or robot
```
- `hiking_deploy` (Python, rclpy, MultiThreadedExecutor): 50 Hz policy timer; 500 Hz writer in its own callback group
  (if jitter > 1 ms p99, move only the writer to C++). State machine modelled on SONIC's (own code): DAMP -> RAMP_TO_DEFAULT
  (2 s) -> STAND (zero command) -> WALK; SAFE_HOLD on tilt (gravity_z), stale state (> 0.1 s), stale depth (> 0.2 s, then
  stand), deadman release; MC handover reuses SONIC's EM calls (robot only).
- `sim_depth_pub` (Python, runs next to the bridge): renders the chest depth from the bridge's MuJoCo state with the same
  ray-cast as `eval/adapters/hiking.py` and publishes `/x2_hiking/chest_depth` at 50 Hz (option: 30/60 Hz + latency
  to mimic the D435). On the robot the RealSense node (or a small pyrealsense2 publisher) publishes the same topic, so
  the deploy node is identical in sim and on the robot.
- `pad_cmd` (Python, pygame): left stick vx (forward only, 0..0.8) and vy (+-0.3), right stick wz (+-1.0), deadzone 0.15,
  rate limit 0.5 m/s^2; L2 deadman; buttons: Y stand/walk, B damp, A+X e-stop (latched). Speed presets 0.3 / 0.5 / 0.8.
- `configs/`: the card path, topics, clip, limits; `scenes/`: MuJoCo scenes for flat, 10 deg slope, single 4x4.

## Milestones
- **M0 Linux baseline (no new code).** Boot Ubuntu; `git pull` x2-perception; full checkout of gwbc-x2; confirm the
  `gr00t-x2sim` image builds/starts and SONIC's sim runs with the gamepad (`sim_onnx_planner.sh`, F09). Download the
  Hiking export from Hugging Face into `deploy/hiking/models/` (git-ignored). Done = SONIC walks in the docker sim under the pad.
- **M1 Offline parity.** `hiking_deploy`'s observation builder and action mapping as a pure-Python module; unit test
  against `obs_check.npz` (exact inputs/outputs) and against `eval/adapters/hiking.py` on the same MuJoCo states,
  including the torso-IMU -> pelvis conversion. Done = max error < 1e-5 on obs, < 1e-4 on actions.
- **M2 Depth in the sim.** `sim_depth_pub` beside the bridge; check the image against `docs/hardware/chest_camera_mount/
  expected_depth_hiking.png` on flat ground (rows/values) and that the robot's own toes/arms appear as in training.
- **M3 Closed loop in the docker sim.** `hiking_deploy` + bridge + depth + pad; stand, ramp, walk. Done = stands 60 s,
  walks 0.3 / 0.5 / 0.8 m/s straight 20 m, turns in place both ways, stops on release, no SAFE_HOLD trips.
- **M4 Terrain in sim.** Scenes: 10 deg slope up/down, single 4x4 at 0.089 m approached at 0.5 m/s, also from 30 deg
  oblique; 10 trials each. Then robustness: 20-40 ms action latency, depth at 30 Hz with 33 ms latency, depth noise and
  holes (DPL-style), IMU noise. Done = success rates recorded; any failure explained in `docs/FINDINGS.md`.
- **M5 Robot prep (ask the user first).** RealSense on PC2 (librealsense/pyrealsense2 on aarch64, USB3 port, power);
  measure policy + preprocessing time on the Orin CPU (budget 20 ms per step incl. the depth encoder; else onnxruntime
  GPU/TensorRT, else the Thor); measure the mounted camera pose; dry run with the robot hung in the gantry, feet off the
  ground, MC handover and back. Real tests only after the user says go.

## Known risks
- Pelvis-frame IMU derived from the torso IMU (waist FK) - a sign or order error looks like a lean and makes it fall.
- Depth latency and frame rate differ from training (50 Hz frames, 1-step lag); the D435 runs at 30/60/90 fps.
- Real stereo depth has holes at step edges and on dark/shiny surfaces; training used clean ray-cast depth + blur.
- Hiking's default pose differs from MC's stand pose: ramp slowly (2 s or more) from wherever MC leaves the joints.
- X2 waist pitch is weak (+-18 deg, 24 N*m, often saturated: FINDINGS X1/X2).
- Gains in training are Isaac PD at 200 Hz sim / 50 Hz policy; the robot runs PD in firmware at 500 Hz writer rate.
