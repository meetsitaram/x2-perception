# ACP on the real X2: deployment plan (2026-11-17)

User, 2026-11-17: "work on getting acp ready to be deployed to real robot". ACP = contact planner + tracker
(`models/acp/`, export `tinkerbuggy/x2-acp-contact-planner-tracker`); its two height scans come from the chest RealSense via
the fine-map depth student (`models/acp_depth/`, fine_v3: 46/50 on our apparatus with D435-like noise in the loop, = perfect
scans). Same rules and stack as the Hiking deploy (`deploy/hiking/PLAN.md`): sim first under the gamepad, then the robot;
nothing touches PC1/PC2 without the user's per-action go; no gwbc-x2 code copied (used at run time).

## What runs every control step (50 Hz)
1. Proprioception: pelvis angular velocity + projected gravity (from the torso IMU by the waist FK, as Hiking), joint
   positions minus ACP's default pose, joint velocities, last action; a 5-step history (495).
2. Every other step (25 Hz): the chest depth (80 x 45 z-depth, 87 x 58 deg) + the camera pose (torso IMU + waist FK + the
   bracket pose) -> the fine student -> fine map -> ACP's planner scan (17 x 11) and tracker scan (9 x 7).
3. Planner (planner_attn.onnx): history, goal, planner scan -> nine contact targets + occupancy.
4. Tracker (actor.npz MLP): history, tracker scan, contact targets (x valid), goal -> 31 actions, clip +-20,
   target = default + scale x action, ACP's own kp / kd.
Goal (ACP is goal-conditioned, not velocity-commanded): the pad gives a direction and a turn: left stick up / sideways =
walk / turn as for Hiking; goal = unit direction in the heading frame (or shrunk inside goal_r when stopping) and the heading
still to turn (rad).

## Milestones
- **A1 parity:** `acp_deploy/core.py` from robot-available inputs only; test against the eval's acp_depth adapter on the same
  sim states (actions, scans, targets). Done = action error < 1e-4.
  **Done 2026-11-17:** `tests/test_a1_parity.py 12` (our apparatus, seed 2007, 601 steps, 7 waypoints): action error 4e-14,
  scans 0, camera pose 3e-17. (The test refreshes the simulator's kinematics before each step: after mj_step the body poses
  lag qpos by one step, which alone gave 3.6e-4 rad pelvis error and 0.7 action error.)
- **A2 sim stack:** the ACP node in the docker sim (gwbc bridge, sim chest camera 80 x 45, pad); stand, walk, turn, 4x4,
  stairs in the park; the same safety states as Hiking (policy takes over from the MC stand, SAFE on tilt / stale data).
  **Status 2026-11-17 (built, not passed):** `acp_deploy/node.py` (Hiking's node with ACP behind its policy hooks; pad ->
  goal: forward stick = unit goal, ACP sets its own pace (a goal under 1 means "arrived": it stood still at vx 0.3); turn stick
  = bearing, released = heading hold on the torso IMU yaw (without it 0.2 rad drift over 5 m)), `sim/acp_depth_pub.py` (80 x 45
  sim camera, 25 Hz), `deploy/hiking/sim/run_sim.sh --policy acp`, `BRIDGE_SPAWN`, `sim/route_pad.py` (drives an eval course
  through the pad). Flat: walk 5.3 m straight, turns, stop. Park lane y=0: 25 m over single beams, beam fields, blocks, no fall.
  Apparatus (park copy = terrain-park CAD physics, full 22-waypoint route, reach 0.25 m):
  | | full routes | where it failed |
  | --- | --- | --- |
  | student, bearing capped 1 rad | 1 / 3 | ramps (stuck), ramps |
  | student, full bearing (as the eval) | 1 / 3 | stairs (stuck), ramps (fell) |
  | student, full bearing, implicitfast | 1 / 3 | ramps (stuck), back 4x4 (stuck) |
  | true scans (sim diagnostic `--truth-scans`) | 2 / 3 | stairs (fell) |
  Those runs were on a park with BOTH apparatus models stacked (the eval raster under a stale CAD export, docs/FINDINGS.md
  B-app1) and a driver unlike the eval (0.25 m arrival, stick goals): void. One-map park (eval geometry), driver = the eval's
  own goal (`route_pad.py --eval-goal`, node `--goal-from-pad`), arrival 0.20 m:
  | | full routes | where it failed |
  | --- | --- | --- |
  | student | 1 / 3 | stairs (fell), stairs (stuck) |
  | true scans | 1 / 3 | ramps diagonal (stuck), near button 2 (fell) |
  The student matches true scans here; both are far below the eval (student 46/50, true scans 46/50): the gap is the stack
  (bridge physics: Euler 1 ms vs the eval's implicitfast 2.5 ms; wall-clock control over DDS; the park's copy of the
  geometry), not the depth. Open: the ramps (student; true scans pass them every
  time), the stairs (both). Turning in place on a ramp slid the robot off (pad: walk while turning there). Next: render the
  ramp failures with the student's map; score the student on the CAD apparatus in the eval harness (geometry vs stack).
- **A3 real camera:** PC2 depth publisher (D435 Z16 640 x 360 -> 80 x 45 z-depth, holes 0) on the same topic; check the fine map
  on the robot's captures (4x4 one foot ahead of the toes: place and height).
- **A4 robot prep (ask first):** timing on the Orin CPU (budget 20 ms per step incl. the student at 25 Hz; else onnxruntime GPU),
  student export for PC2 (TorchScript or ONNX), MC handover, gantry dry run; real tests only on the user's go.

## Known risks
- Real stereo depth vs the noise model (holes at edges, sunlight, the bracket's 8 deg extra pitch; training covered -4..+12 deg).
- Camera pose from the torso IMU: the IMU-to-torso_link frame is unverified (Hiking M5 check), and the bracket pose is
  calibrated to +-1 deg only with nominal intrinsics.
- Which apparatus geometry is real (`docs/FINDINGS.md` B-app1).
