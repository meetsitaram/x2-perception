# Hiking deploy — status / how to resume (2026-11-15, afternoon)

Work lives in the worktree `.claude/worktrees/funny-edison-4f12a2` (user: "lets work here"), uncommitted. The main checkout
still holds the morning's copies of hiking_deploy/ and tests/ (superseded; the model in its deploy/hiking/models is symlinked here).

## Run it
```
deploy/hiking/sim/run_sim.sh --scene park --pad gamepad      # or --pad keyboard (pygame window), --pad script:FILE, --no-viewer
deploy/hiking/sim/run_sim.sh --stop
```
Pad (SONIC's layout, user 2026-11-15): hold L1+L2+R1+R2 ~3 s then Y = stand (2.5 s ramp, then the policy holds it); hold L2
+ left stick = walk, right stick X = turn, release L2 = stop; L1/R1 speed -/+ 0.1 (starts 0.5); e-stop A+X + pump both
triggers. Xbox Series X pad works. `--pad keyboard` = keys in the launching terminal (Enter stand, arrows, Space stop).
The viewer shows the node state + next step top left; it takes no keys (its key callbacks arrive late and in bursts). Logs: deploy/hiking/logs/latest/{bridge,depth,deploy,viewer,pad}.log, pose.csv (ground truth),
summary.txt (script runs: per segment distance, speed, yaw rate).

## Pipeline (lightweight, user 2026-11-15: pad -> model -> robot msg queues; no teleop/replay)
One gr00t-x2sim container (host net, ROS_DOMAIN_ID 73 loopback):
- gwbc-x2 bridge, run unchanged via `sim/bridge_run.py` (run-time use; fixes its torso gyro to the body frame, mjOBJ_XBODY);
  `--imu-from torso` (as the robot), band at standing height (`--band-length -0.313`, pelvis 0.687 m), released 4.5 s after
  the first command.
- `sim/depth_pub.py`: mirrors the bridge state (ZMQ robot_pose + joint states) and ray-casts the trained chest depth,
  /x2_hiking/chest_depth 32FC1 64 x 36 raw metres at 50 Hz (5-8 ms per frame).
- `hiking_deploy/node.py`: WAIT -Y-> RAMP -> STAND (policy, zero cmd) -Y-> WALK; DAMP; SAFE (tilt > 60 deg, stale joint/IMU
  > 0.1 s, e-stop). Policy step 0.8 ms p50 / 1.5 ms p99. Writer timer 500 Hz in its own callback group.
- `sim/mirror_view.py`: viewer in its own process (the bridge's --viewer halved the sim speed on the park).
Host: `hiking_deploy/pad.py` (pygame; gwbc .venv python) -> UDP JSON :5690; `sim/pose_log.py` (ground truth).
onnxruntime for the image's python 3.10: `~/.cache/x2-hiking-pydeps` (pip --target, 1.20.1).

## Scenes (`sim/make_scene.py`, built in the container on first use, cached in sim/scenes/, git-ignored)
Robot = gwbc's x2_ultra.xml (motor actuators kept for the bridge; soles as the eval). Terrain from this repo's eval course
builders + `~/Projects/terrain-park` (cloned 2026-11-15, public). `park`: three 40 m lanes from a flat spawn tile, 8 m tiles
edge to edge; y=0: Hiking types row 2 slope_up, stairs_up, platform_up, gap; y=+8: boxes, small_boxes, rough, stairs_down;
y=-8: ext single 4x4, wedges, stairs up, our apparatus. Height field in 2 m chunks (one big field cost 80-230 ms per depth
frame). Also `flat` and any course name (hik_<type>_L<row>, ext_*, apparatus:eval).

## Milestones
- M0 (SONIC under the pad): still no gamepad on this PC. Superseded in practice by our own stack under the pad; needs a pad.
- M1 DONE: `tests/test_m1_parity.py` passes: obs and action error 0, pelvis gyro 4e-15, gravity 2e-15, depth 0 over 550
  closed-loop steps (waist up to 0.38 rad). The earlier failure was the test's (xmat one substep stale after mj_step).
- M2 partly: depth runs in the sim. Expected-depth check: ours vs expected_depth_hiking.npy differs (median 0.10 m, top rows
  1.75 vs 2.2 m) BUT my check put the pelvis upright; the expected image has soles flat (torso 7.4 deg back). Redo from the
  live stand (depth_pub output) before calling it.
- M3 partly (scripted pad): flat: stands, walks 0.48 m/s at 0.5, turns +1.08/-0.93 rad/s at +-0.8, stops, no SAFE trips.
  Park lane y=0: 18.4 m at 0.47 m/s through slope_up and down/up the stairs_up pit, pelvis 0.60-0.72 m above terrain, no
  falls. Still to do: 60 s stand, 0.3/0.8 m/s 20 m, scripted M3 suite, user drive with a pad.

## Findings
- Hiking's default pose does not stand under PD alone (fell forward 1 s after the band released). Nothing but the policy may
  hold the robot after the ramp, on the robot too (after MC hands over).
- The bridge's in-loop viewer runs the park at 0.56x real time; headless 1.00x. The deploy runs on wall clock, so a slow sim
  means a policy too fast for its physics. Keep the viewer out of the bridge.
- MuJoCo viewer hides geom groups 3-5 by default; the terrain (4) and walls (5) must be switched on.
- gwbc bridge torso gyro used mjOBJ_BODY (inertial frame) — patched at run time in bridge_run.py.
- Real IMU mounting vs torso_link frame still unverified (M5).

## Next
1. User drives the park (keyboard now; gamepad when plugged in). 2. M2 depth check from the live stand. 3. M3 scripted suite
(pad_scripts/), then M4 scenes (10 deg slope, single 4x4, latency/noise). 4. Show the diff for commit approval.

## Update 13:45 — floor start like the robot (user: "launch it right on the floor like we would launch on the robot")
- Bridge starts in gwbc's `stand_settled` init pose (MC-handover joint angles measured on the real X2, feet on the floor, no
  band) and holds it frozen until the first command (stands in for MC). `BRIDGE_START` env overrides.
- A 2.5 s PD ramp to Hiking's default pose fell over without the band (0.8 m forward, down). So the node now starts the
  policy straight from the measured pose on Y, targets blended in over 1 s (`--ramp-s 0` default, `--blend-s 1.0`). Same
  sequence planned for the robot after MC hands over (check in the gantry first, M5).
- Flat test: stands, walks 0.48 m/s at 0.5, stops, no SAFE. Open: 0.84 m of drift in the first 7 s of standing (settling
  steps after takeover) — measure and compare with the training's stand behaviour.
- `--stop` also stops the pad (PID file); a leftover pad from an earlier launch fought the next one.

## Update 14:30 — terrain park, reset, HUD, textures
- `--scene park` is now terrain-park's extended map (`extended_v4k`, what the policies trained on; user: "where is our extended
  home terrain ... are you even using the terrains from our terrain park?"): all nine types (single_beams, beam_fields, blocks,
  wedges, tight_spots, log_corridors, homes, street_front, stairs) along +x at levels 2 / 5 / 8 in lanes y = 0 / 8 / 16 (stairs
  straight / L / U), plus lane y = -8: Hiking paper types + our apparatus. Physics = floor raster + boxes/meshes not in the
  raster (README rule); raster props drawn visual-only by label (wood beams/treads/rails/furniture, plaster walls, concrete
  blocks, lawn, paving, red cars, rocks). The old layout is `--scene hiking_park`.
- Surfaces: grass / concrete / dirt / wood / plaster / asphalt / indoor floor generated (tileable noise), plywood from
  terrain-park. Floors are never wood (user: "everything is brown").
- Reset (sim): View/Back held 1 s -> node WAIT, then bridge_run puts the robot in the stand pose where it fell, on the nearest
  flat 0.5 m patch within 2.5 m, same heading, frozen until the next stand (user: "reset only where it falls").
- HUD: pad inputs (sticks, triggers, held buttons, vmax, ignition) bottom-left in the viewer; state + next step top-left.
- Camera follows the robot (mj_comPos in the mirror; the tracking camera uses subtree_com).

## Update 15:10 — apparatus physics, open door, viewer isolation
- Apparatus: the robot walked through drawn rails. The eval's wall boxes (`apparatus/ours/eval/cad_walls.json`, source
  loco-manip-apparatus `scene_x2_env_mobility.xml`) do not match terrain-park's CAD export `apparatus/ours/sim/apparatus.xml`:
  377 of 437 tall CAD parts had no wall box within 8 cm. In the sim the CAD model is now the apparatus physics (boxes,
  cylinders, spheres, group 5 so the depth sees them; meshes drawn only) and the eval walls are dropped on that tile.
  OPEN (for the eval owners, docs/FINDINGS.md): which of the two is the apparatus as built? If the CAD is, the apparatus
  eval's walls are stale.
- Door and both entrance gates drawn open at 90 deg (eval door open_deg 90; gates are not in the eval walls).
- Viewer in its own container (`x2hiking_view`, shares the stack's IPC for Fast DDS shared memory; reopens after a crash);
  a viewer crash took the whole stack down twice. `exits.log` records every process exit.
- Pad HUD drawn (pad_hud.py): rounded pad, sticks with lit direction triangles, triggers, L1/R1, ABXY, View, vmax, ignition.

## Update 2026-11-16 08:30 — RealSense on PC2 (M5 started, read-only so far)
- PC2 = `run@<PC2_IP>` on the robogym Wi-Fi (hostname `agi`, Orin NX, L4T R36.4.3 / JetPack 6.2; <PC2 internal IP> is its
  internal address). The back USB-C port is NOT on PC2 (3-minute USB watch: no change). The ports the user found (2 USB-A,
  2 USB-C) are: D435i (8086:0b3a) enumerates on PC2 behind its VIA USB-C hub, **but at USB 2 (480M)**: cable or seating.
- First frames grabbed via V4L2 (librealsense bypassed): depth Z16 /dev/video11 (640x480 @30 max on USB 2), color YUYV
  /dev/video15; garage roof at 2.5 m, 88% valid. Renders: sim/renders/pc2_d435_{color,depth}.png.
- librealsense on PC2: ros-humble-librealsense2 2.56.4 installed, but `rs-*` fail "bad optional access" (V4L backend vs the
  JetPack 6 kernel without the UVC patches). Fix = librealsense with the RSUSB backend (install: ask first). pyrealsense2 absent.
- PC1 (two internal addresses from PC2) not accessed: the session's permission guard blocked it; the user decides.

## HOW TO RESUME (2026-11-16)
- Sim: `deploy/hiking/sim/run_sim.sh --scene park --pad gamepad` (Xbox pad; `--depth flat` = fake flat-floor depth from IMU +
  joints, works in sim, paused by the user). Stop: `--stop`.
- Robot (user turned it off 08:35): the D435i port is the one the user found on PC2 (not the back USB-C).
  Next: 1) get it to USB 3 (seat / flip / other USB-C port / Intel cable; check `lsusb -t` shows 5000M);
  2) with approval: librealsense RSUSB build on PC2 (or V4L2 direct + nominal intrinsics);
  3) PC2 publisher: 848x480 (or 640x480 on USB 2) depth -> trained 64x36 z-depth on /x2_hiking/chest_depth;
  4) mount on the bracket, measure the pose; 5) node robot mode (EM/MC handover, 500 Hz writer timing on the Orin);
  6) IMU frame check standing; 7) gantry dry run. Sim M2-M4 suite still to run headless.
- Uncommitted: everything in deploy/hiking/ of this worktree + .gitignore (models symlink, sim/scenes). No commit without the
  user's approval of the diff.

## Update 2026-11-16 — live chest depth (camera on the chest, PC2 USB-C)
- `deploy/hiking/tools/d435_live.py`: live D435 depth on this PC via ssh + v4l2-ctl (no install on PC2; frames via a FIFO:
  v4l2-ctl prints progress on stdout). 15 fps at 640x480 (still USB 2). Floor ahead reads 1.08 m (bracket design: camera
  1.09 m above the floor), the toes in view as in training, 90% valid.
- Finding: whole policy cells (64x36 min-pooled) that are all stereo holes (toes inside the ~0.3 m min range, occlusion
  shadows beside the feet) became "far" (3 m, the training rule for no return) right at the toes. `fill_holes` (min of finite
  8-neighbours, up to 3 cells) removes them; large no-return areas stay far. To carry into the PC2 publisher; test the
  policy's sensitivity with / without it in M4 (holes + noise).
- Open: 640x480 is narrower than the trained 87 deg (use 640x360 on USB 2 or 848x480 on USB 3); nominal intrinsics only.

## Update 2026-11-16 — chest camera mount calibration (capture logs/d435/cal_1116_0737, robot in MC stand, 4x4 at ~0.305 m)
- Leg calibration (`tools/calib_legs.py`, user: "use the leg positions to calibrate first"): the robot's own legs/feet in the
  real depth vs the sim robot in the measured pose (joint states + torso IMU). Best fit: the camera **8.25 deg further down**
  than the trained mount (69.8 vs 61.6 deg from the torso) and **rolled -3.5 deg**; IoU 0.67 (trained mount 0.07), clear peak.
- Consistent in direction with the floor-plane fit (+12.6 deg, sensitive to the nominal FOV) and the 4x4 rows (~+7 deg).
  Camera height 1.05 m (trained pose 1.085 m). Intrinsics still nominal (87 x 58): librealsense calibration will refine.
- Tools: `tools/compare_mount.py` (plane + rows; fixed an 80 GB full SVD that froze the desktop: run capped, ulimit -v 6 GB),
  `tools/calib_legs.py`.
- Next: check how the bracket sits; shim or reprint ~8 deg shallower and fix the roll; re-run calib_legs (1 min, robot still).
