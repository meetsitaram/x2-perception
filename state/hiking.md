# hiking state

Updated 2026-11-11 22:00 UTC (Hiking track agent). Node `hiking`; address and login are in `config/nodes.local.json`.

## What is running

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
- Queued on the node (tmux `evalwait`): at checkpoint 500 a plumbing test of the paper eval (one terrain, 200 robots, at most
  480 s), then training restarts by itself. Result folder `tracks/hiking/eval/paper_eval/x2_parkour_v1_500/`.

## Early signals (trainer's own printed means; not a score)

| Iteration | Robots | Mean reward | Episode length (of 1,000 steps) | Terrain level | Velocity-tracking term (0-1) | Episodes ending in a fall |
|---|---|---|---|---|---|---|
| 25 | 4,096 | 2.0 | 66 | 0.00 | 0.03 | 100% |
| 100 | 4,096 | 5.2 | 122 | 0.00 | 0.05 | 100% |
| 150 | 4,096 | 12.5 | 420 | 0.03 | 0.16 | 93% |
| 200 | 4,096 | 34.0 | 728 | 0.16 | 0.30 | 57% |
| 249 | 4,096 | 41.6 | 832 | 0.30 | 0.29 | 39% |
| 364 (after two resumes) | 8,192 | 44.5 | 850 | 0.75 | n/a | n/a |

- The G1 reproduction's log at the same iterations (desktop run, 4,096 robots): 250: reward 31.9, length 645, level 0.55,
  tracking 0.35; 500: 43.9, 731, 1.48; 1,000: 44.5, 687, 2.06; 3,000: 46.3, 727, 1.93. The X2 curve has the same shape so far.
- After each resume the printed means dip (reward 4.5, length 129) and recover within about 110 iterations (statistics restart,
  robots re-spread over terrain rows). Terrain level right after a resume is not comparable with before it.
- Not measured yet: actual forward speed against the command, double-support share, any video. "Walks" is not established;
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
- Status: never run to completion on any robot (the G1 script was unit-tested only). First plumbing test queued at checkpoint 500.
  No X2 paper-eval number exists yet. This is a track report, not the team's apparatus score.

## Eval hand-over (`/mnt/filesystem-g4/tracks/hiking/eval/x2_parkour_v1_250/`)

- `policy.onnx` (flat observation 5,408 -> 29 actions), also `0-depth_encoder.onnx` + `actor.onnx` (the paper's deploy layout),
  `adapter_card.json` (50 Hz, joint order, default pose, action scale, PD gains, effort and velocity limits, each observation term
  with offset and size, the command and how to make it from waypoints, depth sensor pose, optics and preprocessing),
  `obs_check.npz` (50 observations with torch and ONNX actions, max difference 7e-7).
- Made on the CPU by `code/x2_port/export_for_eval.py <checkpoint>` while training runs.
- Limits: `model_250` does not walk yet (plumbing only). The 50 observations are synthetic, so they prove ONNX equals torch, not
  that an adapter builds the observation as the simulator does; a recording from the simulator (`obs_check_sim.npz`: observation,
  action, root state, joint state, command for 50 steps) is produced by the paper-eval pause. The order of the 8 depth frames and
  of history entries (oldest first assumed) is not verified.

## Extended terrain set (`/mnt/filesystem-g4/data/terrains/extended_v1/`, 9.4 MB, CPU export, 2026-11-11)

- 330 tiles of 8 x 8 m, ten difficulty levels per type: single beams 30, beam fields with V pairs and slanted 4x4s 30, blocks 30,
  wedges (pinwheel cells, long ramps, narrow strips) 40, tight spots and dead ends 20, log corridors with wall, rail and post
  sides 40, homes 30, street front 30, stairs 80 (straight, L, U, amphitheatre).
- Format: per type one `.npz` (2 cm height rasters 400 x 400, obstacle mask, boxes with centre, half size, quaternion and role,
  meshes for non-box shapes) and one `.json` (level, seed, parameters, spawn poses, routes); `index.json`, `README.md`,
  `load_example.py`, `preview_top.png`, `preview_closeups.png` (previews, not simulator renders).
- Routes: 680 defined by the generators, 131 dropped by `check_routes.py` (119 obstacle at body height, 12 drops over 0.26 m),
  549 kept; 51 kept routes had end points pulled back inside the tile. Street front keeps only 19 of 90 routes (the generator
  puts the hydrant, pole, bushes and car across its fixed routes): needs a generator fix.
- Not in it: the paper's own nine terrain types (their generator needs Isaac Sim; `review/terrain/terrain.ply` could be
  rasterised later); roll wedges; rails on amphitheatre aisles.
- Held-out check: the generators read no apparatus file; no tile reproduces an apparatus layout or route. Apparatus-like single
  parts present: 4x4 beams 5-15 cm (flat or one end raised up to 0.20 m), V pairs of 2.6 m beams, 2 x 2 pinwheel ramp cells
  (1.27-1.94 m, 8.8-25 deg), rails on 8.9 cm posts, brace stands, stair flights (risers 8.4-21.6 cm, 3-15 steps), doorways.
  No pinwheel matches our apparatus cell (about 1.2 m, 15 deg) in both size and all slopes.
- Not verified: nothing loaded in a simulator yet; spawns not tested against the X2 footprint.
- Export scripts: `tracks/hiking/code/x2_port/terrain_export/`.

## Next

1. Read the checkpoint-500 paper-eval plumbing test (`eval/paper_eval/x2_parkour_v1_500/`), fix what it shows, confirm training
   restarted by itself.
2. Early numbers at about 1,000 and 2,000 iterations from `metrics.jsonl`; add actual-speed and double-support logging.
3. MuJoCo sim2sim of an X2 checkpoint (port `scripts/sim2sim_mujoco.py`: X2 joint order, head camera, mesh vs sphere feet).
4. Extended task as a second task name (`Instinct-Parkour-Extended-Amp-X2-v0`): **not started**. Needed: an Isaac terrain
   importer for `extended_v1` tiles beside the paper's nine types, route spawns and targets, backward and sideways commands with
   matching motion clips, the stumble, foothold and clearance terms (whether these training penalties stay is an open question in
   `docs/USER_RULES.md`). Start only after the paper-terrain run has a checkpoint that walks the paper's terrains.

## Blockers

None. Open questions for the coordinator: one 40-minute pause or several 10-minute pauses for a full paper table; whether the
extended task's training penalties stay.
