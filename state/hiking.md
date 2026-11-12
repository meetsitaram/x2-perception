# hiking state

Updated 2026-11-11 23:10 UTC (Hiking track agent). Node `hiking`; address and login are in `config/nodes.local.json`.

## What is running: run 2 (`x2_parkour_v2`), from scratch since 22:58 UTC

- Task `Instinct-Parkour-TorsoCam-Amp-X2-v0`, 8,192 robots, to 30,000 iterations, tmux `hiking_x2`, supervisor
  `code/x2_port/supervise_x2.py` (stall detection, auto-resume, checkpoint every 250, metrics, GPU log, backup manifest).
  Start or resume: `bash /mnt/filesystem-g4/tracks/hiking/code/x2_port/start_x2_parkour_v2.sh`.
  Stop: `touch /mnt/filesystem-g4/tracks/hiking/runs/x2_parkour_v2/_supervisor/STOP`.
- Run folder `/mnt/filesystem-g4/tracks/hiking/runs/x2_parkour_v2/` (`<start time>/model_<it>.pt`, `metrics.jsonl`, `gpu.csv`,
  `_supervisor/`). Heartbeat `/mnt/filesystem-g4/status/hiking.json`.
- First throughput (iteration 11): 19,600 samples/s, 10.0 s per iteration (collection 8.0 s, learning 2.0 s), 51 GB of 96 GB GPU
  memory. Slower than run 1 (23,000-25,000/s, 35 GB) because the depth input is twice as tall (36 rows instead of 18).
- No checkpoint yet (first at iteration 250, about 23:40 UTC).
- Run 1 (`x2_parkour_v1`) was stopped at 22:54 UTC at iteration about 650; its checkpoints `model_250.pt` and `model_500.pt` are
  kept. GPU idle between the runs: 237 s.

### Standing rules for this run (user, via the coordinator, 2026-11-11)

- No training stop whose only purpose is an evaluation, a recording or a render. Isaac evaluation only inside a restart that is
  needed anyway. `code/x2_port/pause_and_eval.sh` stays unused unless the coordinator says otherwise. Everything else comes from
  the training logs, the MuJoCo port on the desktop CPU, or the eval monitor.
- The run-2 versions of `export_for_eval.py` and the MuJoCo port exist only on the desktop
  (`C:/Users/PC/projects/perception-agents-work/hiking_sim2sim_x2/`): the upload to the shared filesystem was refused by the
  permission system and nobody copies them until the user allows it. The copies on the node (`code/x2_port/export_for_eval.py`,
  `code/x2_port/sim2sim/`) are the run-1 versions; do not export a run-2 checkpoint with them. Run-2 renders are made on the
  desktop and shown from there.
- Nothing under `/mnt/filesystem-g4` is deleted: unwanted files are moved to `tracks/hiking/_trash/<date>/` with a one-line
  note (`/tmp` on the node is free to use). Helpers get the same rule. `code/x2_port/bench_envs.sh` still removes its scratch
  run folders with `rm -rf` (it did so on 2026-11-11 before this rule): change that to a move before it is used again.
- Run 2 runs to iteration 1,000 before any judgement; one line at 500 and 1,000 (signals table below).
- MuJoCo, run 2 `model_250` (desktop, 14 s flat): stands and shuffles in place drifting backwards, 0.54 m net, mean forward
  speed -0.04 m/s against 0.74 commanded, both feet down 87%, no fall; actions did not depend on the depth image.

### Plan: an Isaac evaluation beside the running training (proposed, NOT started; the coordinator decides)

What is known from this node (measured unless marked "estimate"):

- GPU memory: training run 2 holds 56.9 GB of 95.6 GB at iteration 314 (51 GB at start, so it grows), leaving about 39 GB.
  Training-style memory against robot count (run 1 task): 21.0 GB at 4,096, 34.9 GB at 8,192, 49.1 GB at 12,288, i.e. about
  7 GB fixed plus 3.4 GB per 1,000 robots including the PPO buffers. An evaluation has no PPO buffers; estimate 8 GB at 200
  robots and 10-11 GB at 1,000 (never measured: the 64-robot checks did not log memory).
- GPU compute: training already uses the card at 50-98% and is compute-bound (profile above). A second instance takes its share
  for as long as it runs: expect training throughput to drop by roughly the evaluation's share, not for free.
- Start-up: an Isaac start takes 3-5 minutes (terrain generation, URDF conversion, ray-cast mesh build) and briefly used 7.8
  CPU cores; the node has 24 cores, training uses about 1.8, RAM 197 GB free. CPU and RAM are not the limit.
- Two instances on one card: never tried on this node. Each is its own process with its own PhysX GPU context; the shader
  cache (11 GB inside the venv's `isaacsim/kit/cache`) is already warm, so the second instance should only read it, but two
  instances writing it at once is untested. Converted robot USD files go to separate time-stamped folders in `/tmp/IsaacLab`.
- What a failure of the second instance could do to the first: an out-of-memory error normally hits the process that asks for
  memory, so the evaluation would die, but training's own memory use still grows (51 to 57 GB so far) and an allocation of its own would fail if the card is full
  at that moment (the supervisor then resumes from the newest checkpoint, losing up to 250 iterations, about 40 minutes). A
  driver-level fault (Xid) would take both. The supervisor's own scripts are safe: it only refuses to START while another GPU
  job runs, so if training crashes during an evaluation it waits for the evaluation to end and then resumes.
- Things in my own scripts that must not be used with two instances: `pause_and_eval.sh`, `launch_v2.sh` and `restart_at.sh`
  kill every GPU process on the card.

Proposed test (one shot, about 12 minutes, started by hand right after a checkpoint is written so a training crash loses nothing):

1. Wait for `model_<N>.pt` to appear. Record 3 minutes of training throughput from the log (samples/s, iteration time) and GPU
   memory: "before".
2. Start one paper-eval worker on that checkpoint: one terrain, 200 robots, Table 2 only (751 steps), `nice -n 10`, wall-clock
   limit 600 s (`timeout`), its own log, no kill-all anywhere. Refuse to start if free GPU memory is under 25 GB.
3. A watchdog loop every 2 s: if free GPU memory falls under 8 GB, or the training log stops advancing for 120 s, kill the
   evaluation process (by its pid only).
4. Record training throughput and GPU memory "during" (start-up phase and stepping phase separately) and for 5 minutes "after";
   check the training log for PhysX errors and the supervisor log for a restart.
5. Pass if: training never stopped or restarted, no PhysX error, throughput after equals before within 3%, the evaluation
   finished inside the limit. Report the throughput loss during, peak memory, and the evaluation's own start-up and run time.
6. Only if that passes: 1,000 robots, then several terrains in sequence (each is a separate start, so most of the cost is
   start-up; nine terrains would take about 40 minutes of shared GPU).

Until the coordinator decides, Hiking evaluation stays on the MuJoCo port (desktop CPU) and the training logs.

### What run 2 changes against run 1 (everything else identical; the paper's terrains, commands, rewards, curriculum, PPO)

| Item | Run 2 | Why |
|---|---|---|
| PD gains | plant `mc_20261024` (table below); armature, effort, default pose and action scale unchanged | user: "Use the ones used for supermimic training last iteration" |
| Depth camera | chest RealSense: on `torso_link` at (0.131, 0.0, 0.259), 61.6 deg below horizontal, 87 x 58 deg, all 36 rows kept, the paper's left/right crop (36 x 32 per frame), range 0-3 m | the user's camera; a departure from the paper's camera at the user's direction |
| Depth encoder input | 8 frames of 36 x 32 (run 1: 18 x 32); conv output 4 x 36 x 32 = 4,608 into the same head | follows from the crop |
| Motion reference | v2: standing cut to 3% (run 1 until iteration 500: 21%) | measured defect, see "Standing" |
| Torso tilt term | on `torso_link` from the start (run 1: from iteration 250) | |
| Logging | commanded vs actual speed, support shares, step rate, terrain level per type, from iteration 0 | |

### Camera (run 2): calibration and acceptance

- Reference: the user's infrared frame, `docs/camera/` in this repo. Acceptance: at the standing pose on flat ground the toe tips
  are just inside the bottom edge, a floor line 1.0 m ahead of the toes is at 27% (+-3%) of the image height from the top, and a
  0.55 m line is near 52%.
- Solved for the simulated standing pose (`code/x2_port/chest_solve.py`), lens 3 cm proud of the chest shell (shell front from the
  torso mesh: x = 0.101 m at that height): **lens 1.055 m above the ground, 61.6 deg below horizontal, 1.1 cm ahead of the toe
  tips**. Result: toe line at 99.9%, 0.55 m at 52.1%, 1.0 m at 26.2% from the top; top edge sees 1.66 m ahead of the toes.
  Picture with the lines marked: `state/hiking_v2_camera_check.png` (MuJoCo preview, not a training-simulator frame); more in
  `/mnt/filesystem-g4/tracks/hiking/renders/camera_check/` (colour view, depth 0-3 m, the 36 x 32 policy view, `camera_check.txt`).
- The coordinator's reading (0.90 m, 53 deg, lens 12 cm behind the toes) fits the two main landmarks too but puts the 0.55 m line
  at about 50%; the mesh says the chest shell is only 1.3 cm behind the toe tips in the default pose. The user's measured height
  and angle would settle it; a change is two numbers in `x2_parkour_target_amp_cfg.py` (`X2_TORSO_CAMERA_POS`, `..._PITCH_DOWN`).
- The view reaches 1.66 m, not the 2 m the user mentioned earlier: with a 58 deg vertical field of view, seeing the toes and 2 m
  at once needs the camera about 1.25 m high.
- Earlier candidates, for the record (flat ground, standing, ahead of the toe tips): run 1 head camera, 40 deg, paper crop:
  0.40 to 1.27 m. Paper's G1 camera placed on the X2 at the same height (1.24 m), 48 deg, paper crop: 0.23 to 1.01 m (that point is
  inside the X2's head shell at the G1's 0.049 m forward offset; the head's face is at 0.076 m). Same height, 0.10 m forward,
  60 deg, all rows: 0.01 to 2.00 m. G1 itself (not computed by FK; about 1.24-1.27 m high): about 0.3 to 1.1 m from the camera.
- The SSR track's estimate for the same camera: `torso_link` (-0.026, 0, 0.125), 53.5 deg (toe line 96%, 0.55 m at 49.5%, 1.0 m
  at 27.1%, -0.02 to 1.85 m). At that height (0.92 m) the mesh puts the chest shell front at x = +0.096 m, so that point is
  12 cm inside the torso; the two poses differ by 16 cm forward, 13 cm in height and 8 deg. One of them must be adopted by both.
- Mount randomisation (user: the real lens height is "somewhere in between"): per robot, redrawn at each episode reset, height
  -10.5 to +5 cm (0.95 to 1.105 m), forward +-2 cm, pitch +-4 deg, roll and yaw +-2 deg (`X2_CAMERA_MOUNT_RANGES`, InstinctLab's
  `randomize_camera_offsets`). **Written but OFF in the running process**: it has never run in Isaac, and an unattended
  auto-resume must not pick up untested code. Turn it on at the next watched restart:
  `HIKING_X2_CAM_RAND=1 bash code/x2_port/start_x2_parkour_v2.sh` and watch the first iterations. Until then run 2 trains with
  the fixed pose. Not yet in the adapter card.
- Not done: an Isaac-rendered frame at the chest pose (the frames
  dumped by the pre-launch check are from the superseded brow-height pose and are marked SUPERSEDED in `runs/_smoke_v2/`).
- Row order (checked on run 1's Isaac recording): row 0 is the top of the image (far), the last row is nearest the feet.

### Gains applied in run 2, read back from the Isaac articulation (`runs/_smoke_v2/applied_*.json`)

| Joints | kp / kd applied (mc_20261024) | kp / kd run 1 (armature rule) | armature | effort N.m | velocity rad/s | action scale rad |
|---|---|---|---|---|---|---|
| hip pitch | 100 / 4 | 114.8 / 7.31 | 0.02908 | 120 | 11.94 | 0.261 |
| hip roll, hip yaw | 100 / 3 | 114.8 / 7.31 | 0.02908 | 120 | 11.94 | 0.261 |
| knee | 150 / 5 | 114.8 / 7.31 | 0.02908 | 120 | 11.94 | 0.261 |
| ankle pitch | 40 / 3 | 34.9 / 2.22 | 0.00884 | 36 | 13.09 | 0.258 |
| ankle roll | 30 / 2 | 12.4 / 0.79 | 0.003133 | 24 | 15.08 | 0.485 |
| waist yaw | 40 / 8 | 114.8 / 7.31 | 0.02908 | 120 | 11.94 | 0.261 |
| waist pitch, roll | 40 / 5 | 12.4 / 0.79 | 0.003133 | 24 | 13.09 | 0.485 |
| shoulder pitch | 30 / 1 | 34.9 / 2.22 | 0.00884 | 36 | 13.09 | 0.258 |
| shoulder roll | 20 / 1 | 34.9 / 2.22 | 0.00884 | 36 | 13.09 | 0.258 |
| shoulder yaw | 20 / 1 | 12.4 / 0.79 | 0.003133 | 24 | 15.08 | 0.485 |
| elbow, wrist yaw | 50 / 1 | 12.4 / 0.79 | 0.003133 | 24 | 15.08 | 0.485 |
| wrist pitch, roll | 20 / 1 | 16.8 / 1.07 | 0.00425 | 6 | 20.94 | 0.089 |
| head yaw (held at 0) | 3.4 / 0.114 | 16.8 / 1.07 | 0.00425 | 2.6 | 6.02 | not commanded |
| head pitch (held at 0) | 10 / 0.8 | 16.8 / 1.07 | 0.00425 | 0.6 | 6.28 | not commanded |

- Every applied kp and kd equals the plant file's value (first-match substring rule reproduced); no joint differs from the intent.
- Action scale = 0.25 x effort / (armature x (2 pi 10)^2), from the armature, not from the new stiffness. Actuator type:
  delayed PD, 0-2 physics steps of delay, torque clipped to effort. Default pose: hip pitch -0.312, knee 0.669, ankle pitch
  -0.363, elbow -0.6, shoulder pitch 0.2, shoulder roll +-0.2, the rest 0.
- Run 1 used the armature-rule column; its checkpoints must be played with those gains.

### Model audit (Isaac's conversion of the URDF against the MJCF)

- Total mass: Isaac 41.967 kg = URDF link sum 41.967 kg; **MJCF 43.475 kg**. The whole difference is the pelvis: 3.523 kg in the
  URDF and Isaac, 5.032 kg in the MJCF (+1.509 kg). Every other body agrees within 5 g. Which is right is not known to me.
- Joint limits agree except waist pitch (URDF +-0.314, MJCF +-0.349) and waist roll (URDF +-0.488, MJCF +-0.524).
- Armature per joint: identical in Isaac and the MJCF.
- Fixed links (IMUs, cameras, lidar) are merged into their parents by the converter with their mass; the body masses above
  already include them. Inertia tensors were not compared.
- Feet: Isaac uses the convex hull of the foot mesh, sole 0.073 m below the ankle; the MJCF has 12 spheres of 5 mm per foot.
- Self-collision: on, minus the 5 always-overlapping pairs.
- The URDF used in training is the shared one minus the massless `base_link` and the XML declaration.

## Standing: what was measured (run 1) and what run 2 does about it

- Run 1 at checkpoint 500 stands and does not walk: Isaac Stair Up test 0 of 200 reached, 198 still standing; whole-population
  log at iterations 520-615: commanded 0.63 m/s, actual 0.00 m/s, both feet down 95%.
- MuJoCo run of `model_500` (CPU): stands, leans back 9 deg, no steps, 0.15 m in 14 s. Files and validation of the port:
  `tracks/hiking/renders/ckpt500_mujoco/`, `tracks/hiking/code/x2_port/sim2sim/`.
- At checkpoint 500 the depth encoder's output is exactly zero for all 50 recorded depth stacks: the policy ignored depth.
- Style data: reference v1 was 21% standing (30% under 0.1 m/s; median speed 0.32 m/s); the paper's G1 file is 5% (7.5%; median
  0.41). Reference v2 (`data/motions/hiking_x2/x2_parkour_motion_without_run_v2_retargetted.npz`, 400 s): 3.3% standing kept on
  purpose, median 0.53 m/s, 67% of frames inside the 0.45-1.0 m/s command range. Script `build_x2_motion.py`, numbers by
  `motion_stats.py` in the same folder.
- The G1 log (iterations 0-5,098) has no phase like run 1's: full-length episodes never above 69%, wait penalty never below -7.0,
  air-time reward never below 0.015 after iteration 150. Run 1 at 499: 89%, -9.6, 0.008.
- Reward balance per step (weighted): standing X2 at 499 earns 0.048, the walking G1 at 5,000 earns 0.068; the gaps are tracking
  (-0.010), alive (-0.009) and the wait penalty (-0.006). Standing does not out-earn walking on the task terms; the style reward
  (discriminator reward about 0.5-0.65 per step before its 0.25 coefficient) is paid in full for standing when a fifth of the
  reference stands.
- Ankle action scale and default pose equal the X2's working values in gwbc-v6 (same formula and numbers).
- Run 1 was restarted from `model_500` with reference v2 at 22:28; in the 150 iterations it then ran the speed ratio stayed at 0.
  That is too short to judge the fix, and run 2 starts with v2 from scratch.
- Run 2, iteration 11: speed ratio 0.24, both feet down 51% (untrained policy flailing; not a result).

### Which terms pay the standing X2 (answer to the SSR track's question)

Weighted reward per control step (episode sum / episode length), from the per-term logs:

| Term | X2 run 1 at 499 (stands) | G1 at 500 | G1 at 5,000 |
|---|---|---|---|
| is_alive | +0.0483 | +0.0577 | +0.0568 |
| track_ang_vel_z_exp | +0.0160 | +0.0132 | +0.0180 |
| track_lin_vel_xy_exp | +0.0116 | +0.0201 | +0.0218 |
| stand_still | +0.0000 | +0.0010 | +0.0022 |
| volume_points_penetration (edges) | -0.0000 | -0.0041 | -0.0045 |
| feet_slide | -0.0010 | -0.0038 | -0.0032 |
| dof_acc_l2 | -0.0010 | -0.0023 | -0.0018 |
| feet_flat_ori | -0.0012 | -0.0021 | -0.0023 |
| dof_pos_limits | -0.0021 | -0.0001 | -0.0004 |
| action_rate_l2 | -0.0030 | -0.0050 | -0.0068 |
| pelvis_orientation_l2 (torso tilt on the X2) | -0.0031 | -0.0010 | -0.0009 |
| heading_error | -0.0040 | -0.0069 | -0.0033 |
| dont_wait | -0.0096 | -0.0058 | -0.0039 |
| all task terms | +0.0478 | +0.0569 | +0.0679 |
| style (discriminator) reward per step, before its 0.25 coefficient | 0.646 | 0.436 | 0.511 |

- What pays the stander: staying alive (the largest term by far); yaw tracking, which it earns in full by turning on the spot
  toward the target; and about half of a walker's linear-tracking reward, because the exponential kernel (std 0.5) still pays
  exp(-0.63^2 / 0.25) = 0.2 at zero speed. It also avoids about 0.011 per step of penalties a walker pays (edge penetration, foot
  slide, action rate, joint acceleration, foot tilt). The wait penalty (-0.0096) and lost tracking do not outweigh these by much:
  the stander is at 0.048 against 0.057 for the G1 at the same iteration.
- The style reward is where the X2 stander does better than the G1 ever did (0.65 against 0.44-0.51): consistent with a reference
  that was one fifth standing. How the 0.25-weighted style reward compares in size with the task terms inside the trainer was not
  traced.
- Did the G1 pass through a shuffle? Yes, by its air-time reward: per 100 steps 0.0054 at 500, 0.0071 at 1,000, 0.0088 at 5,000,
  against roughly 0.05-0.06 at 20,000 (0.56 per episode). So for its first 5,000 iterations the G1 moved with very short swing
  phases while falling in 34-71% of episodes. It never stood: the X2 at 499 has 0.0008 per 100 steps, seven times lower again.

### Rest pose with zero actions (answer on the lean-back question)

- Isaac, pre-launch check of the run-2 task (mc_20261024 gains, 64 robots on the training terrain with reset noise, 2.2 s):
  1.05 episodes ended per robot, 63 of 71 by tilt over 1 rad. Run 1's gains under the same check: 1.86 per robot in 3 s; the G1
  task: 2.45. So with zero actions the default pose is not held in Isaac under either plant; it falls rather than settling into
  a lean. Pelvis pitch at rest was not recorded (direction of the fall not logged).
- MuJoCo, stock MJCF with sphere soles, flat ground, no noise (`rest_pose.py`): falls forward at 1.4 s under both plants. In
  the default pose the centre of mass is 0.016 m ahead of the pelvis origin and the sole contact points span -0.083 to +0.121 m
  (centre 0.019 m), so the pose is balanced only if the joints hold it exactly; knees sag 0.044 rad under mc gains.
- This does not reproduce the SSR track's "settles 7.6 deg leaning back" (its model holds the waist stiffly and uses its own
  feet). The 9 deg backward lean of run 1's checkpoint 500 is the policy's choice of stance, not the rest pose.
- Nothing was changed in the default pose.

### Run 2 signals (trainer's printed means over all 8,192 robots; not a score)

| Iteration | Reward | Episode length | Terrain level | Speed ratio (commanded / actual m/s) | Both feet down | Air-time reward | Wait penalty | Episodes ending in a fall | Style reward |
|---|---|---|---|---|---|---|---|---|---|
| 50 | 1.6 | 71 | 0.00 | 0.45 (0.62 / 0.28) | 66% | 0.004 | -0.41 | 100% | 0.36 |
| 100 | 3.1 | 104 | 0.00 | 0.28 (0.61 / 0.17) | 72% | 0.005 | -0.82 | 100% | 0.41 |
| 150 | 10.9 | 282 | 0.09 | 0.17 (0.62 / 0.11) | 79% | 0.009 | -2.25 | 87% | 0.44 |
| 200 | 22.6 | 573 | 0.25 | 0.02 (0.63 / 0.01) | 81% | 0.014 | -6.16 | 58% | 0.47 |
| 249 | 35.1 | 736 | 0.40 | 0.02 (0.62 / 0.01) | 84% | 0.016 | -8.31 | 35% | 0.50 |
| G1 at 500 (equal samples to X2 250) | 43.9 | 731 | 1.48 | not logged | not logged | 0.040 | -4.27 | 60% | 0.44 |

**Run 2 is going the same way as run 1: as the robots learn to stay up, forward speed goes to zero** (speed ratio 0.45 at
iteration 50 while everything falls, 0.02 from iteration 200 on). So the standing is not explained by the old gains, the head
camera or the standing share of the motion reference alone: all three were changed and the pattern is the same at 250. The early
"speed" at iterations 50-100 is robots falling forward, not walking. Checkpoint: `runs/x2_parkour_v2/20261111_225809/model_250.pt`.

### Does the style reward tell standing from walking? (offline, CPU, `code/x2_port/score_discriminator.py`)

Each checkpoint's discriminator scored on 10-frame windows built from the motion file as the task builds them; reward =
clamp(1 - 0.25 (D - 1)^2, 0); the policy gets 0.25 x this.

| Discriminator | Reference walking (> 0.3 m/s) | Reference standing stretches | Synthetic perfect stand (default pose, zero velocity) | Stand as share of walking |
|---|---|---|---|---|
| X2 run 2, iteration 250 (reference v2) | 0.917 (D +0.44) | 0.811 | 0.720 (D -0.06) | 78% |
| same discriminator on reference v1 | 0.909 | 0.823 | 0.720 | 79% |
| G1, iteration 500 | 0.948 (D +0.55) | 0.888 | 0.875 (D +0.29) | 92% |
| G1, iteration 5,000 | 0.923 | 0.840 | 0.835 | 90% |
| G1, iteration 20,000 (walks stairs) | 0.874 | 0.771 | 0.750 (D 0.00) | 86% |

- The style reward separates standing from walking only weakly on both robots: a motionless robot in the default pose collects
  78% (X2) or 86-92% (G1) of what real walking collects. The discriminator's outputs stay between about -0.1 and +0.6 (targets
  are +1 for the reference and -1 for the policy), and with this reward form D = 0 already pays 0.75.
- **The G1 walked with a discriminator that separates even less than the X2's.** So the style prior is not what makes the G1
  walk and the X2 stand; the difference has to be in the task terms or the robot (staying alive against the wait penalty and
  tracking, and how costly a first step is for each robot).
- Which inputs carry what little separation there is (X2): joint positions (walking windows with stand joint positions drop
  0.917 -> 0.811) and base linear velocity (0.917 -> 0.873); joint velocities, gravity and angular velocity carry almost none
  (0.914-0.924). Same ranking on the G1.
- Logged discriminator terms are alike on both: weight decay 92 (X2 run 1 at 260) against 109-117 (G1), gradient penalty 0.043
  against 0.044, loss 0.36 against 0.21-0.38. The weight-decay figure is not an anomaly.
- The policy's own windows were not scored (no rollout of run 2 available on the CPU yet); the trainer logs 0.50 for them at 249,
  below the synthetic stand's 0.72, so the discriminator does mark the policy's actual standing as less reference-like than a
  perfect stand.
- Approximations: reference velocities are finite differences of the file without the task's interpolation; clip seams skipped.

### Terms that correspond to a "get off the ground" recipe (for carrying one over from SSR as a listed deviation)

| Role | Term here | Value now |
|---|---|---|
| Survival reward | `is_alive` | +3.0 per step (x 0.02 s): the largest term, about 0.06 per step |
| Penalty for not moving when told to | `dont_wait` | -0.5 (about -0.011 per step for a stander) |
| Penalty for moving when told to stand | `stand_still` | -0.3, offset 4.0 |
| Linear tracking | `track_lin_vel_xy_exp` | +2.0, kernel exp(-err^2 / 0.5^2): pays 0.2 at zero speed for a 0.63 m/s command |
| Yaw tracking | `track_ang_vel_z_exp` | +2.0, same kernel; a stander earns it in full by turning on the spot |
| Heading | `heading_error` | -1.0 |
| Swing reward | `feet_air_time` | +0.5, only when commanded above 0.15 m/s |
| Gait penalties a first step pays | `action_rate_l2` -0.005, `feet_slide` -0.4, `dof_acc_l2` -1.25e-7, `feet_flat_ori` -0.4, `volume_points_penetration` -4.0 | |
| Style | discriminator reward | x 0.25 |
| Curriculum | `tracking_exp_vel`, thresholds 0.3 / 0.6 | robots that stand stay on the easy rows |
| Command | forward 0.45-1.0 m/s to a target, 5% told to stand, resampled every 8-12 s | |

**Watch in run 2** (`metrics.jsonl`): `Curriculum/walk_monitor/speed_ratio`, `double_support`, `Episode_Reward/rewards_feet_air_time/sum`,
`rewards_dont_wait/sum`, at 250, 500, 1,000, 2,000. G1 reference at 500 (4,096 robots): air time 0.040, wait penalty -4.3,
full-length episodes 40%, terrain level 1.48.

## Run 1 (`x2_parkour_v1`, stopped 22:54 UTC): record

- **X2 paper-terrain run `x2_parkour_v1`**: task `Instinct-Parkour-Target-Amp-X2-v0`, **8,192 robots** since iteration 250
  (4,096 before), to 30,000 iterations, first started 2026-11-11 20:22 UTC. tmux session `hiking_x2` on the node, under
  `code/x2_port/supervise_x2.py` (stall detection 900 s without log output, kill and resume from the newest checkpoint, one GPU
  job, checkpoint every 250 iterations, metrics and backup housekeeping).
- Heartbeat: `/mnt/filesystem-g4/status/hiking.json` (every 10 s: state, iteration, reward, episode length, terrain level,
  newest checkpoint, GPU, robot count). During a pause for the paper eval the state reads "paper eval ..., training paused".
- Run folder: `/mnt/filesystem-g4/tracks/hiking/runs/x2_parkour_v1/` (one timestamped folder per start or resume with
  `model_<it>.pt`, `params/`; `metrics.jsonl`, `gpu.csv`; `_supervisor/train.log`, `_supervisor/supervisor.log`).
- Newest checkpoint: `checkpoint` in the heartbeat. `model_250.pt` is the only one I have exported so far.
- History of stops: 20:55-21:22 robot-count benchmark (27 min, too long; limit is now 10 min); 21:26-21:36 restart to apply the
  torso-orientation term and the new supervisor (resumed from `model_250`, about 20 iterations redone).
- 22:13-22:18 (291 s): paper-eval plumbing test at checkpoint 500, training restarted by itself from `model_500`.
- Nothing else is queued on the node.

## Run 1 early signals (trainer's own printed means; not a score)

| Iteration | Robots | Mean reward | Episode length (of 1,000 steps) | Terrain level | Velocity-tracking term (0-1) | Episodes ending in a fall |
|---|---|---|---|---|---|---|
| 25 | 4,096 | 2.0 | 66 | 0.00 | 0.03 | 100% |
| 100 | 4,096 | 5.2 | 122 | 0.00 | 0.05 | 100% |
| 150 | 4,096 | 12.5 | 420 | 0.03 | 0.16 | 93% |
| 200 | 4,096 | 34.0 | 728 | 0.16 | 0.30 | 57% |
| 249 | 4,096 | 41.6 | 832 | 0.30 | 0.29 | 39% |
| 364 (after two resumes) | 8,192 | 44.5 | 850 | 0.75 | n/a | n/a |
| 499 | 8,192 | 58.4 | 1,000 | 1.00 | 0.29 | 11% |

- The G1 reproduction's log at the same iterations (desktop run, 4,096 robots): 250: reward 31.9, length 645, level 0.55,
  tracking 0.35; 500: 43.9, 731, 1.48; 1,000: 44.5, 687, 2.06; 3,000: 46.3, 727, 1.93. The X2 curve has the same shape so far.
- After each resume the printed means dip (reward 4.5, length 129) and recover within about 110 iterations (statistics restart,
  robots re-spread over terrain rows). Terrain level right after a resume is not comparable with before it.
- **Warning sign at iteration 500: the robots stay up but do not walk.** Paper-eval test on Stair Up (200 robots, 15 s): 0 of 200
  reached the target 3.7 m away, 198 were still standing at the time limit, 2 fell. Robot 0 was commanded 0.74 m/s and moved
  0.07 m in its first second. In the training log at 499: 89% of episodes run the full 20 s, the `dont_wait` penalty sum is -9.6
  (G1 run at 500: -4.3), `feet_air_time` sum 0.008 (G1: 0.040), tracking reward per step 0.012 (G1: 0.020), terrain level 1.00
  (G1: 1.48). So the X2 is surviving more and stepping less than the G1 did at the same iteration: the standing-still pattern the
  SSR track reported. Too early to call (500 of 30,000; the G1 also took thousands of iterations to climb), but this is the thing
  to watch: if `feet_air_time` and tracking per step are not rising by about iteration 1,500, look at the X2-specific suspects
  first (style data with long standing stretches, action scale on the ankles, the 40 deg camera seeing less ground, 8,192 robots).
- Not measured yet: double-support share, any video; actual speed only for one robot in one second. "Walks" is not established;
  only that robots stay up for most of an episode and the tracking term matches the G1 run's at this stage.
- Numbers at a few thousand iterations: not reached yet (about 8.5 s per iteration; iteration 2,000 around 02:00 UTC 2026-11-12).

## GPU use (user: "make sure the GPUs are well used")

Benchmark at checkpoint 250, same task, 12 iterations each, one job at a time (`runs/_bench/bench.log`, `code/x2_port/bench_envs.sh`):

| Robots | Samples per second | Iteration time | GPU memory | Note |
|---|---|---|---|---|
| 4,096 | 15,400 | 6.4 s | 21.0 GB | paper's count |
| 8,192 | 22,100 (+43%) | 9.0 s | 34.9 GB | chosen; no PhysX errors |
| 12,288 | 24,500 (+59%) | 12.1 s | 49.1 GB | PhysX "collisionStackSize buffer overflow, contacts have been dropped" 6 times: needs larger PhysX buffers, untested |
| 16,384 | not measured | | | stopped during startup to get training going again |

- Running now: 22,800-23,400 samples/s, 8.4-8.6 s per iteration, 35 GB of 96 GB, GPU utilisation 50-95% (`gpu.csv`).
- Kept fixed: 24 steps per robot per iteration, 4 minibatches, 5 epochs, adaptive learning rate (KL 0.01), 30,000 iterations,
  all rewards. Effect on the comparison with the paper's 4,096: each iteration uses twice the samples (minibatches twice as
  large), so iteration counts are not one-to-one with the G1 run; 30,000 iterations take about 71 h instead of 53 h.
- What limits it (py-spy on the live trainer, 25 s, `runs/_bench/pyspy_8192_iter370.raw`): the main Python thread spends about 36%
  in sensor updates (30% in the depth-camera ray-cast, mostly inside `warp synchronize`, i.e. waiting for the GPU), 14% in the
  policy step, 10% rewards, 8% the PhysX step call, 7% actuators, 6% motion reference. The single thread mostly launches GPU work
  and waits for it, so the limit is GPU compute done one kernel after another (ray-casting and physics), then per-step Python
  overhead; not GPU memory, and not something more CPU cores would fix.
- Still open: 12,288 with `sim.physx.gpu_collision_stack_size` and `gpu_max_rigid_patch_count` enlarged (+11% samples over 8,192
  at +34% iteration time); only worth testing inside a 10-minute pause.

## Backup (user: everything a restart needs on the shared filesystem)

Nothing a restart needs is only on the node's local disk: the run writes straight to the shared filesystem.

| Kind | Where (under `/mnt/filesystem-g4/`) | Restore |
|---|---|---|
| Checkpoints (all kept, 59 MB each, every 250 iterations: more than "newest two plus every 1,000") | `tracks/hiking/runs/x2_parkour_v1/<start time>/model_<it>.pt` | `start_x2_parkour_v1.sh` resumes from the newest |
| Training log, supervisor log | `tracks/hiking/runs/x2_parkour_v1/_supervisor/` | read only |
| Metrics, GPU log | `tracks/hiking/runs/x2_parkour_v1/metrics.jsonl`, `gpu.csv` | rebuilt from `train.log` by `log_metrics.py` (delete `metrics.jsonl*` first) |
| Run config as trained | `tracks/hiking/runs/x2_parkour_v1/<start time>/params/{env,agent}.yaml` | |
| X2 port code, supervisor, eval and export scripts, URDF variant | `tracks/hiking/code/x2_port/` | `apply.sh` copies the task files into InstinctLab |
| Upstream code at pinned commits + patches | `tracks/hiking/code/hiking-in-the-wild/` (from `tracks/hiking/hiking-in-the-wild.bundle`) | `env/setup_node.sh` |
| X2 motion reference and its build script | `data/motions/hiking_x2/` | `build_x2_motion.py <mjcf> <flat pkl> <stairs pkl> <out npz>` on the desktop (sources listed below) |
| X2 model | `data/robots/x2_ultra/` | |
| Extended terrain set and its export scripts | `data/terrains/extended_v1/`, `tracks/hiking/code/x2_port/terrain_export/` | `export_extended_terrains.py` (20 s on CPU) |
| Benchmarks, profile | `tracks/hiking/runs/_bench/` | |
| Eval hand-over, paper eval results | `tracks/hiking/eval/` | `export_for_eval.py`, `pause_and_eval.sh` |
| Environment: exact versions | `tracks/hiking/env/freeze.txt` (rewritten every 30 min), `constraints.txt` | `bash env/setup_node.sh` rebuilds `~/hiking-env` (about 15 min) |

- Automatic part: the supervisor rewrites `metrics.jsonl` every 5 min, `gpu.csv` every minute, and every 30 min `env/freeze.txt`
  and `tracks/hiking/backup_manifest.txt` (checkpoint list, file counts).
- Only on the node's local disk, all rebuildable: the venv `~/hiking-env` (29 GB), `~/Datasets` (the released G1 motion,
  unzipped from `data/motions/hiking_release_g1/data_model.zip`), Isaac's converted USD in `/tmp/IsaacLab`.
- Environment pins that differ from the desktop: numpy 1.26.0; `setuptools<81` then `flatdict==4.0.1 --no-build-isolation`;
  instinct_rl installed with `--no-deps --ignore-requires-python`; `mujoco==3.3.7 websockets==12.0`; torch 2.7.0+cu128;
  `isaacsim[all,extscache]==5.1.0` from pypi.nvidia.com; Python 3.11 from uv.

## Commands

```bash
S="ssh -i ~/.ssh/nebius_corpus -o IdentitiesOnly=yes -o BatchMode=yes <user>@<hiking-node>"
$S 'cat /mnt/filesystem-g4/status/hiking.json'                                             # state
$S 'tail -5 /mnt/filesystem-g4/tracks/hiking/runs/x2_parkour_v1/_supervisor/supervisor.log'
$S 'touch /mnt/filesystem-g4/tracks/hiking/runs/x2_parkour_v1/_supervisor/STOP'            # stop (supervisor kills training, exits)
$S 'bash /mnt/filesystem-g4/tracks/hiking/code/x2_port/start_x2_parkour_v1.sh'             # start or resume (newest checkpoint)
$S 'bash /mnt/filesystem-g4/tracks/hiking/env/setup_node.sh'                               # rebuild code tree + env (idempotent)
$S 'tmux new -d -s evalwait "bash /mnt/filesystem-g4/tracks/hiking/code/x2_port/pause_and_eval.sh x2_parkour_v1 <iteration> 540 stair_up,gap 1000"'  # paper eval in a pause
$S 'CUDA_VISIBLE_DEVICES= ~/hiking-env/bin/python /mnt/filesystem-g4/tracks/hiking/code/x2_port/export_for_eval.py <model_N.pt>'  # eval hand-over, CPU
$S 'bash /mnt/filesystem-g4/tracks/hiking/code/x2_port/apply.sh'                           # copy the X2 port into InstinctLab again
# quick env check (not a result); needs the GPU free, so stop training first:
$S 'cd /mnt/filesystem-g4/tracks/hiking/code/hiking-in-the-wild/InstinctLab && OMNI_KIT_ACCEPT_EULA=YES /home/<user>/hiking-env/bin/python ../../x2_port/check_x2_env.py --headless --num_envs 64 --steps 150'
```

## Layout on the node

| Path | What |
|---|---|
| `/mnt/filesystem-g4/tracks/hiking/code/hiking-in-the-wild/` | clone of the bundle (branch `g1-20k-and-extended-terrain`) with `InstinctLab` de6b92f + patch, `instinct_rl` 574c8d5 + patch, `IsaacLab` 37ddf62 (v2.3.2), `instinct_onboard` b6965a4 cloned beside it |
| `/mnt/filesystem-g4/tracks/hiking/code/x2_port/` | the X2 port: `files_instinctlab/` (new files only, copied into InstinctLab by `apply.sh`), `robot/x2_ultra_pelvis_root.urdf`, `supervise_x2.py`, `check_x2_env.py`, `start_x2_parkour_v1.sh` |
| `/mnt/filesystem-g4/tracks/hiking/env/` | `setup_node.sh`, `constraints.txt`, `freeze.txt`, setup logs |
| `/home/<user>/hiking-env` | the Python 3.11 venv (local disk: the shared filesystem takes about 17 ms per small file). Isaac Sim 5.1.0, Isaac Lab 0.54.2 (2.3.2), torch 2.7.0+cu128 |
| `/home/<user>/Datasets` | the released G1 motion file (the G1 task reads `~/Datasets`) |

Do not run Python with `code/x2_port` as the working directory if a folder named `instinctlab` is in it (it shadowed the package
once; the folder is now `files_instinctlab`).

## Copies made to the shared data folder (rsync --bwlimit=3000, 2026-11-11)

| To | From (desktop) | Size |
|---|---|---|
| `data/robots/x2_ultra/urdf/` (x2_ultra.urdf, two sphere-feet variants, meshes) | `gwbc-v6/gear_sonic/data/assets/robot_description/urdf/x2_ultra/` | 112 MB |
| `data/robots/x2_ultra/mjcf/x2_ultra.xml` | `gwbc-v6/gear_sonic/data/assets/robot_description/mjcf/x2_ultra.xml` (identical to `ssr-x2/assets/robot/mjcf/x2_ultra.xml`) | 60 KB |
| `data/motions/hiking_x2/` (`x2_parkour_motion_without_run_retargetted.npz`, `.yaml`, `build_x2_motion.py`) | built on the desktop from `loco-manip-apparatus/build/train/x2_loco_flat_train.pkl` (BONES-SEED X2 retargets) and `terrain-park/data/train_sets/tp_stairs_x10_sh2.pkl` (Pico stair takes) | 3 MB |
| `data/motions/hiking_release_g1/data_model.zip` | `hiking-in-the-wild/data_zip/data_model.zip` (the paper's release) | 12 MB |

Motion credit: BONES-SEED, Motion Data by Bones Studio (bones.studio). No clip names in anything shown.

## The X2 port (task `Instinct-Parkour-Target-Amp-X2-v0`, play variant `...-X2-Play-v0`)

Inherited unchanged from the paper's config (`ParkourEnvCfg`): terrains, commands, rewards and weights, observations, curriculum,
randomisation, terminations (except one threshold), network, PPO and discriminator settings.

Changed, robot only:

| Item | X2 value | Reason |
|---|---|---|
| Model | shared `x2_ultra.urdf` (mesh feet) with the massless `base_link` and the XML declaration removed (`x2_port/robot/`), root = pelvis | Isaac named the root `base_link`; pytorch_kinematics cannot parse the declaration. The G1 task uses a torso-rooted URDF; I kept the X2's own root so the shared model stays as is |
| Actuators | delayed PD (0-2 steps) as on the G1; armature, effort, velocity from the vendor plant (gwbc-v6 `config/robot_plant/x2_ultra.yaml`); kp = armature x (2 pi 10)^2, kd = 4 x armature x 2 pi 10 | same BeyondMimic rule as the G1 task |
| Actions | 29 joints, scale 0.25 x effort / kp; the two head joints are not commanded (held at zero by PD) | head motors are 2.6 and 0.6 N.m and carry the camera |
| Default pose | gwbc-v6 X2 pose (same leg angles as the G1 pose, elbows -0.6), spawn height 0.71 m | pelvis stands at 0.641 m (MuJoCo FK) |
| Self-collision | on, minus 5 link pairs two joints apart whose hulls always overlap (knee/ankle_roll, wrist_yaw/wrist_roll per side, torso/head_pitch) | measured 6-20 kN between them at spawn; it tripped the torso-contact termination every step |
| Depth camera | the X2's head camera: on `head_pitch_link` at (0.0576, -0.0112, -0.0484), 40 deg below horizontal; optics, crop, noise, history as in the paper | deployable mount; the G1's is 48 deg down on the torso |
| Foot numbers | sole 0.073 m below the ankle (feet_at_plane offset); volume points x -0.045..0.12, y +-0.05, z -0.078..-0.038 | X2 sole from the model's contact points |
| Torso tilt term | `pelvis_orientation_l2` points at `torso_link` (from iteration 250) | the G1's root is the torso and the term adds the pelvis; the X2's root is the pelvis, so the term adds the torso |
| Root-height termination | 0.39 m (G1: 0.5 m for its torso root) | same fraction of standing root height |
| Motion reference | one concatenated 405.7 s file at 50 Hz: 322 s flat walking and turning (26 BONES-SEED clips), 83 s stairs (16 Pico takes); head and wrist joints zeroed; clips chained in xy and yaw | the release is one 379.6 s file of walk and climb; a single long file keeps `dataset_exhausted` from ending episodes early |
| Mirror maps | derived from joint names in Isaac order (roll and yaw negate) | checked with MuJoCo FK: 2.5 mm max error; a wrong sign gives 0.28 m |

Apparatus parts copied into training: none (paper terrains only).

## Verified

- G1 released task starts and trains headless on this GPU (256 robots, 3 iterations, 4.0 s per iteration): `runs/_smoke/g1_smoke.log`.
- X2 env: joint order equals `X2_ISAAC_JOINTS` (assert), 29 actions, no contact above 1 N at spawn after the filter, depth image
  shows ground with no self-occlusion blobs. Zero actions for 3 s: 1.86 episodes ended per robot (G1 under the same check: 2.45),
  so standing without a policy is no worse than the G1.

## Not verified

- Nothing is scored: no common eval, no MuJoCo run of an X2 checkpoint yet.
- Feet in MuJoCo are 12 small spheres per foot (the shared MJCF); training uses the mesh foot (convex hull). The sphere-feet URDF
  was not used because 5 mm spheres let terrain edges pass between them.
- Depth camera: pose from the URDF, intrinsics left at the paper's (D435i-like); not compared with the real Gemini 335 stream.
- Head held by PD at zero: on the robot the head pitch must be held the same way.
- Whether the stair clips (0.3 m total rise, two steps) are enough of a climbing prior for the high platforms.

## Decisions (date, decision, reason)

- 2026-11-11: venv on the node's local disk, rebuild script on the shared filesystem. Reason: small-file speed.
- 2026-11-11: Linux pins differ from the desktop (see Backup). Reason: Isaac Sim's own pins.
- 2026-11-11: pelvis root, 29 commanded joints, head camera, mesh feet, filtered pairs: see the port table.
- 2026-11-11: wrote `data/robots`, `data/motions` and `data/terrains/extended_v1` myself, as the coordinator's messages said.
- 2026-11-11: 8,192 robots, not 12,288. Reason: PhysX dropped contacts at 12,288 with the task's buffer sizes; +11% samples
  did not justify an untested buffer change while training was stopped.
- 2026-11-11: `pelvis_orientation_l2` now penalises `torso_link` on the X2 (from iteration 250). Reason: on the G1 the root is the
  torso and that term adds the pelvis; with the X2's pelvis root the unchanged term penalised the pelvis twice and the torso not at
  all (seen in the log: both terms printed identical values). Iterations 0-250 were trained with the duplicate.
- 2026-11-11: all checkpoints are kept (7 GB for the whole run) rather than pruned to "newest two plus every 1,000".
- 2026-11-11: paper eval is run in pauses of at most 10 minutes, a few terrains per pause (each terrain is its own Isaac start).

## Logs: what is recorded per iteration (`runs/x2_parkour_v1/metrics.jsonl`, one JSON object per iteration)

Written by `code/x2_port/log_metrics.py` from the trainer's printed block; names are the trainer's. An iteration that was redone
after a resume appears twice (take the last). dt = 0.02 s.

| Name | What it is | Paper | Weight |
|---|---|---|---|
| `iteration`, `steps_per_s`, `collection_s`, `learning_s`, `Iteration_time`, `Total_timesteps`, `Total_time`, `ETA` | throughput | | |
| `reward` | mean episode return (all terms, weighted, summed over the episode) | | |
| `length` | mean episode length in control steps (max 1,000 = 20 s) | | |
| `action_noise_std` | policy action noise | Table 6 (init 1.0) | |
| `surrogate_loss`, `value_loss`, `entropy`, `total_loss` | PPO losses | Table 5 | |
| `discriminator_loss`, `discriminator_gradient_penalty`, `discriminator_weight_decay`, `logit_weight_decay`, `amp_total_loss` | AMP discriminator losses | A.9 | gradient penalty 5.0 |
| `discriminator_reward` | mean style reward per step, before the 0.25 coefficient | A.9 | 0.25 |
| `Episode_Reward/rewards_<term>/sum` | weighted episode sum of the term | Table 4 | below |
| `Episode_Reward/rewards_<term>/timestep` | weighted mean per step (weight x dt x raw) | | |
| `Episode_Reward/rewards_<term>/max_episode_len_s` | weighted sum divided by 20 s | | |
| `raw/<term>` | unweighted mean per step = timestep / (weight x 0.02) (4-decimal print, so tiny terms are coarse) | | |
| `Curriculum/terrain_levels` | mean terrain row (0-9) over all robots | A.3 | |
| `Metrics/base_velocity/error_vel_xy`, `error_vel_yaw` | command tracking error summed over the command period | | |
| `Metrics/base_velocity/tracking_exp_vel_xy`, `tracking_exp_vel_yaw` | exponential tracking score used by the curriculum | A.3 | |
| `Episode_Termination/<cause>` | share of ended episodes per cause: `time_out`, `terrain_out_bound`, `base_contact` (torso), `bad_orientation` (tilt > 1 rad), `root_height` (pelvis < 0.39 m), `dataset_exhausted` | A.2 | |

Reward terms and weights (the paper's, unchanged): `track_lin_vel_xy_exp` +2.0, `track_ang_vel_z_exp` +2.0, `heading_error` -1.0,
`dont_wait` -0.5, `is_alive` +3.0, `stand_still` -0.3, `volume_points_penetration` -4.0 (foot points inside terrain edges),
`feet_air_time` +0.5, `feet_slide` -0.4, `joint_deviation_hip` -0.5, `ang_vel_xy_l2` -0.05, `dof_torques_l2` -1.5e-7,
`dof_acc_l2` -1.25e-7, `dof_vel_l2` -1e-4, `action_rate_l2` -0.005, `flat_orientation_l2` -3.0 (root tilt),
`pelvis_orientation_l2` -3.0 (on the X2: torso tilt), `feet_flat_ori` -0.4, `feet_at_plane` -0.1, `feet_close_xy` +0.4,
`energy` -5e-5, `freeze_upper_body` -0.004, `dof_pos_limits` -1.0, `dof_vel_limits` -1.0, `torque_limits` -0.01,
`undesired_contacts` -1.0 (any body except the feet over 1 N).

`gpu.csv`: unix time, iteration, GPU memory used and total (GB), GPU utilisation (%), trainer CPU (% of one core), every minute.

Not logged (gaps): terrain level per terrain type (only the mean over all types); commanded speed distribution and the robot's
actual speed; gait statistics (air time per foot, double-support share, step length); the paper's Table 1 and 2 metrics during
training (success rate, reaching time, landing area: these come from the paper eval below); per-expert MoE gate weights.
Plan: a small monitor term for per-type terrain level, actual vs commanded speed and double-support share, added at a restart.

## Paper's own evaluation (Tables 1 and 2) for the X2

- Script: `code/x2_port/eval_paper_x2.py`, generated by `make_eval_paper_x2.py` from the G1 reproduction's `scripts/eval_paper.py`
  (protocol and its 21 assumptions: `docs/EVAL_NOTES.md` in the code tree). Paper figures ("Ours", G1) are read from the paper
  text and printed beside ours: Table 2 success 99.09-100% on all nine terrains, reaching time 5.15-8.03 s; Table 1 landing-area
  fraction 0.77 (stair up), 0.67 (stair down), 0.84 (gap), 0.81 (small box).
- X2 differences: the X2 task and model; landing-area rays cover the X2 sole (0.18 x 0.10 m centred 0.037 m ahead of the ankle,
  sole 0.073 m below it; G1 shoe: 0.18 x 0.06 m); the interaction-range grid hangs under the pelvis; reaching distance is from the
  pelvis; terminations are the X2 task's. Terrain sizes are the paper's Table 8 (20 cm stairs, 40 cm platforms and boxes, gaps),
  unchanged, on a robot whose hip is at 0.64 m instead of the G1's roughly 0.76 m.
- How to run: `bash code/x2_port/pause_and_eval.sh x2_parkour_v1 <checkpoint iteration> <budget s> <terrains> [robots]` in tmux. It
  waits for the checkpoint, stops training, runs one Isaac worker per terrain, restarts training, logs the pause length.
  Each terrain costs about 3 to 5 minutes of start-up, so the nine terrains need four or five 10-minute pauses, or one 40-minute
  pause: to be agreed with the coordinator before a full table.
- Status: plumbing proven at checkpoint 500 on one terrain with 200 robots and a shortened Table 1 run (400 steps): 288 s for
  the terrain including Isaac start-up, tables printed with the paper's row beside ours, `results_*.json` written
  (`tracks/hiking/eval/paper_eval/x2_parkour_v1_500/`). Result at 500: Stair Up 0 of 200 (see Early signals). No full table
  (nine terrains, 1,000 robots, 10,000-step Table 1) has been run. This is a track report, not the team's apparatus score.

## Eval hand-over (`/mnt/filesystem-g4/tracks/hiking/eval/x2_parkour_v1_500/`, and `..._250/` with synthetic observations)

- `policy.onnx` (flat observation 5,408 -> 29 actions), also `0-depth_encoder.onnx` + `actor.onnx` (the paper's deploy layout),
  `adapter_card.json` (50 Hz, joint order, default pose, action scale, PD gains, effort and velocity limits, each observation term
  with offset and size, the command and how to make it from waypoints, depth sensor pose, optics and preprocessing),
  `obs_check.npz` (50 observations with torch and ONNX actions, max difference 7e-7).
- Made on the CPU by `code/x2_port/export_for_eval.py <checkpoint>` while training runs.
- `x2_parkour_v1_500/obs_check.npz` holds 50 observations recorded in the training simulator (robot 0, Stair Up, first second);
  `obs_check_sim_state.npz` holds the matching simulator state per step (root pose and velocities, gravity vector, joint positions
  and velocities, command, default pose, joint names) so an adapter can rebuild each observation and compare. ONNX on those
  observations reproduces the simulator policy's actions to 1e-6.
- Checked on the recording: history entries are oldest first (newest last) for joint positions (error 0.009 = the noise level;
  0.036 the other way round).
- Limits: `model_500` stands and does not walk (plumbing only). The order of the 8 depth frames is not verified. The card's
  depth preprocessing was read from the config, not rebuilt outside Isaac.

## Extended terrain set (`/mnt/filesystem-g4/data/terrains/extended_v1/`, 9.4 MB, CPU export, 2026-11-11)

- 330 tiles of 8 x 8 m, ten difficulty levels per type: single beams 30, beam fields with V pairs and slanted 4x4s 30, blocks 30,
  wedges (pinwheel cells, long ramps, narrow strips) 40, tight spots and dead ends 20, log corridors with wall, rail and post
  sides 40, homes 30, street front 30, stairs 80 (straight, L, U, amphitheatre).
- Format: per type one `.npz` (2 cm height rasters 400 x 400, obstacle mask, boxes with centre, half size, quaternion and role,
  meshes for non-box shapes) and one `.json` (level, seed, parameters, spawn poses, routes); `index.json`, `README.md`,
  `load_example.py`, `preview_top.png`, `preview_closeups.png` (previews, not simulator renders).
- Routes (after the street-front fix of 2026-11-11 23:20): 710 defined, 60 dropped by `check_routes.py`, **650 kept**; 51 kept
  routes had end points pulled back inside the tile. Street front: 120 of 120 kept, four per tile (street to porch over the curb;
  street to garage up the driveway; porch to street over the rock bed and curb; garage to street), least clearance 0.40 m, from a
  generator variant `terrain_export/street_routed.py` (obstacles placed clear of the routes, car moved, garage floor slab added,
  door headers raised with the floor); its tiles differ from the original `street.py` for the same seed.
- Still weak: wedges 21 routes dropped (pinwheel routes into rails or brace stands, or off steep gables), stairs 19 (all
  amphitheatre), homes 19 (kitchen clutter on fixed route lines); causes are the sub-agent's reading, not individually diagnosed.
  The 0.35 m clearance rule is enforced for street only.
- `targets` in the JSON are now only the checked ones; the 44 outside the tile are under `raw_targets_unchecked`.
- Not in it: the paper's own nine terrain types (their generator needs Isaac Sim; `review/terrain/terrain.ply` could be
  rasterised later); roll wedges; rails on amphitheatre aisles.
- Held-out check: the generators read no apparatus file; no tile reproduces an apparatus layout or route. Apparatus-like single
  parts present: 4x4 beams 5-15 cm (flat or one end raised up to 0.20 m), V pairs of 2.6 m beams, 2 x 2 pinwheel ramp cells
  (1.27-1.94 m, 8.8-25 deg), rails on 8.9 cm posts, brace stands, stair flights (risers 8.4-21.6 cm, 3-15 steps), doorways.
  No pinwheel matches our apparatus cell (about 1.2 m, 15 deg) in both size and all slopes.
- Not verified: nothing loaded in a simulator yet; spawns not tested against the X2 footprint.
- Export scripts: `tracks/hiking/code/x2_port/terrain_export/`.

## Next

1. Run 2 signals at 250, 500, 1,000 and 2,000 iterations (see "Watch in run 2").
2. Make `export_for_eval.py` and the MuJoCo port take the camera mount and gains from the task (they still describe run 1's head camera and armature-rule gains): needed before any run-2 checkpoint is exported or rendered.
3. Paper table only when a checkpoint walks, in pauses under 10 minutes.
4. Extended task as a second task name (`Instinct-Parkour-Extended-Amp-X2-v0`): **first version written, never run in Isaac**
   (`code/x2_port/files_instinctlab/tasks/parkour/config/x2/extended.py`). It has: the paper's 20 terrain columns plus 12 new ones
   (one per exported tile type, two for stairs, two velocity-command columns), tile meshes built from the set's boxes and meshes
   (builder tested on the CPU for all nine types: 96 to 9,260 faces a tile, inside 8 x 8 m); velocity-mode commands forward
   -0.4..0.8, sideways +-0.3 m/s, turn +-1 rad/s; `feet_stumble` (weight -2.0) and `foot_clearance` (-0.5) beside the paper's edge
   term. Missing: the set's checked routes as spawn/target pairs (targets are sampled flat patches; spawn yaw not applied),
   backward and sideways clips in the motion reference, graded knee and shin costs. Before any training:
   `check_x2_env.py --task Instinct-Parkour-Extended-Amp-X2-v0` in a pause. What was planned originally: Needed: an Isaac terrain
   importer for `extended_v1` tiles beside the paper's nine types, route spawns and targets, backward and sideways commands with
   matching motion clips, the stumble, foothold and clearance terms (the user: "INCLUDE THEM"). Start only after the paper-terrain run has a checkpoint that walks the paper's terrains.

## Blockers

None. Open: the user's measured camera height and angle; which pelvis mass is right (URDF 3.52 kg or MJCF 5.03 kg).
