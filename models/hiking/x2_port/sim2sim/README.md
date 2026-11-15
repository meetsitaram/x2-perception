# X2 Ultra, Hiking track: MuJoCo sim2sim player (CPU only)

Port of `hiking-in-the-wild/scripts/sim2sim_mujoco.py` (G1) to the AgiBot X2 Ultra, checked against observations
recorded in the training simulator (Isaac Sim) for checkpoint 500 of `x2_parkour_v1`.

Result in one line: in MuJoCo the checkpoint **stands and does not walk** (0.15 m net in 14 s at a 0.74 m/s command,
no fall), the same kind of behaviour Isaac showed.

## Files

| file | what |
|---|---|
| `sim2sim_mujoco_x2.py` | the player: `EnvConfig`, `Terrain`, `build_model`, `DepthCamera`, `X2Sim` (state, observation, step), `target_command`, `run_closed_loop`, CSV, video |
| `validate_x2.py` | validation against the Isaac recording, writes `validation.txt` (+ `validation_depth.npz`: recorded vs MuJoCo flat-ground depth frames) |
| `validation.txt` | all validation numbers |
| `x2_ckpt500_mujoco_{flat,stairs}_{chase,legs}.mp4` | the four clips, 14 s, 1280x720, 25 fps real time, H.264 |
| `x2_ckpt500_mujoco_{flat,stairs}.csv` | one row per control step (50 Hz): time, command, base velocity in the root frame, base position, roll/pitch, foot contact flags, 31 joint positions (Isaac order) |
| `x2_ckpt500_mujoco_{flat,stairs}.json`, `..._rollout.npz` | summary numbers; raw rollout (qpos, actions, foot forces, depth frames) |
| `summary.txt` | one line per clip |
| `checks/frames/` | frames extracted from the clips for the video-vs-CSV check; `checks/variants/` summaries of the extra runs below |
| `assets/` | copies fetched from the training node: `eval/` (ONNX, adapter card, recordings), `run/env.yaml`, `src/` (task source, reference only) |

## Run

```
PY=C:/Users/PC/.venvs/hiking/Scripts/python.exe      # mujoco 3.16, onnxruntime, imageio-ffmpeg, PIL, yaml; nothing installed
$PY sim2sim_mujoco_x2.py --name x2_ckpt500_mujoco_flat   --terrain flat   --seconds 14 --vmax 0.74
$PY sim2sim_mujoco_x2.py --name x2_ckpt500_mujoco_stairs --terrain stairs --seconds 14 --vmax 0.74
$PY validate_x2.py
```

Options: `--terrain flat|stairs|stairs_pyramid`, `--foot hull|spheres` (default hull), `--vmax`, `--target-dist` (3.7),
`--stand`, `--delay-steps 0..2`, `--depth-lag`, `--no-self-collision`, `--no-video`. One process, about 2 s of CPU for a
14 s run plus about 20 s per video. CUDA is hidden; ONNX runs on the CPU provider; the mp4 uses MuJoCo's OpenGL renderer
for drawing only. `ffmpeg` is not on PATH: the encoder is the binary bundled with `imageio-ffmpeg` in the venv.

From another script:

```python
import sim2sim_mujoco_x2 as s
ec = s.EnvConfig(); policy = s.OnnxPolicy(); sim = s.X2Sim(ec, s.Terrain("stairs"))
sim.reset((0, 0, ec.init_z))
for k in range(700):
    pos, quat, R, v_b, w_b = sim.root_state()
    cmd = s.target_command(ec, pos, quat, target=(3.7, 0, 1.0), vmax=0.74)
    sim.step(policy(sim.observe(cmd)))          # observe -> (5408,), step = 4 physics steps of explicit PD
```

`sim.set_state(root_pos, root_quat_wxyz, joint_pos, joint_vel, root_lin_vel_b, root_ang_vel_b)` takes Isaac-style
quantities (joint arrays in the 31-joint Isaac order).

## Model

* URDF `gwbc-v6/.../urdf/x2_ultra/x2_ultra.urdf` with `base_link` removed, free root = `pelvis`, fixed links fused:
  32 bodies, 31 hinge joints, 41.97 kg. Pelvis 0.641 m above the soles in the default pose (same number as the task source).
* Collision = convex hull of every link's collision mesh (what MuJoCo does with mesh geoms), including the feet
  (**foot model used: hull**, the training one). Self-collision on, with the three pair types filtered in training
  excluded. Checked: in the default pose exactly those pairs overlap (knee/ankle_roll 42 mm, wrist_yaw/wrist_roll 30 mm,
  torso/head_pitch 21 mm) and nothing else touches after excluding them.
* `--foot spheres`: the 12 sole spheres (5 mm) of the gwbc MJCF instead of the foot hull. Supported, not the training
  geometry.
* Actuators: torque motors; the explicit PD `tau = kp (q* - q) - kd qd`, clipped to the effort limit, is computed in
  python every 5 ms physics step. kp, kd, armature, effort, action scale, default pose are read from `env.yaml` and
  asserted equal to `adapter_card.json`. 29 joints commanded, `head_yaw`/`head_pitch` held at 0 by their PD.
* Terrain `stairs`: a **straight flight** made of boxes: flat for 1.25 m, 5 steps of 20 cm riser x 30 cm tread across
  an 8 m width, landing at +1.0 m from x = 2.75 m; the target 3.7 m ahead is on the landing. `stairs_pyramid` is the same
  profile on four sides (pit, like the Isaac `pyramid_stairs_inv` tile). No perlin roughness, no walls. The straight
  flight was used for the clips so the side camera of the legs view is not inside a side staircase.

## Observation build (5408 values, float32)

Per control step (50 Hz), in this order. Every proprio term is an 8-step history, **oldest first, newest last**,
flattened term after term (`[t-7 .. t]` of term 1, then of term 2, ...). At reset each history is filled with the first
value.

| start | size | term | newest-step value |
|---|---|---|---|
| 0 | 24 | `base_ang_vel` | pelvis angular velocity in the pelvis frame, rad/s, x 0.25 |
| 24 | 24 | `projected_gravity` | `R_pelvis^T [0,0,-1]` |
| 48 | 24 | `velocity_commands` | `[vx, vy, yaw_rate]` (root frame) |
| 72 | 248 | `joint_pos` | `q - default`, 31 joints, Isaac `joint_order` |
| 320 | 248 | `joint_vel` | `qd` x 0.05, 31 joints |
| 568 | 232 | `actions` | previous raw policy output, 29 values, `action_joint_order` |
| 800 | 4608 | `depth_image` | 8 frames x 18 rows x 32 columns, frames oldest first, newest last |

* Action: `q_target[action joints] = default + action_scale * action`, held for 4 physics steps.
* Command from a target point (training rule, `PoseVelocityCommand`): `vx = clip(2 * x_b, 0, vmax)`, `vy = 0`,
  `yaw_rate = clip(2 * wrap(bearing - yaw), -1, 1)`, all zero within 0.4 m of the target. `vmax` is the sampled speed
  (0.45..0.8 stairs, 0.45..1.0 rough ground). In the recording the command of the reset step is `[0,0,0]`.
* Depth: pinhole 64x36 (fx = fy = 32.28 px), on `head_pitch_link` at (0.05761, -0.011183, -0.04837), frame x forward /
  y left / z up rotated by quaternion wxyz (cos 20, 0, sin 20, 0) = 40 deg down. z-depth (distance to the image plane)
  by ray-casting terrain + the robot's visual meshes, hits closer than 0.1 m ignored. Then crop rows 18..35 and columns
  16..47, gaussian blur 3x3 sigma 1 (reflect padding), clip to 0..2.5 m, divide by 2.5. Row 0 is the top of the image
  (far), row 17 the bottom (near); column 0 is the left. A 50 Hz history; the stack is frames `t-1-35, t-1-30, ..., t-1`:
  **the newest frame is one control step old** in the Isaac recording (`--depth-lag 1`, default).
* Mind when writing an adapter on MuJoCo: body-frame velocity must be read with `mjOBJ_XBODY` (link frame). `mjOBJ_BODY`
  is the inertial frame, which is rotated for the X2 pelvis; using it gave a wrong gyro (error up to 2.8 rad/s) and made
  the robot fall at 1.7 s in a first run of this port. The G1 reference script uses `mjOBJ_BODY` for its torso.

## Validation numbers (`validation.txt`)

1. Proprio rebuilt from the recorded simulator state, newest step, 50 steps, |rebuilt - recorded observation| in
   observation units (the recording has training noise):

   | term | max | mean | noise bound (scaled) |
   |---|---|---|---|
   | base_ang_vel | 0.0498 | 0.0229 | 0.05 |
   | projected_gravity | 0.0496 | 0.0256 | 0.05 |
   | velocity_commands | 0 | 0 | 0 |
   | joint_pos (31) | 0.0100 | 0.0049 | 0.01 |
   | joint_vel (31) | 0.0250 | 0.0124 | 0.025 |
   | actions (29) | 0 | 0 | 0 |

   All within the noise bound, means at half the bound as uniform noise gives. Noise-free: projected gravity from the
   recorded quaternion vs the recorded `projected_gravity_b` 4e-8.
2. History order: for all six proprio terms slot i of step t equals the newest slot of step t-(7-i) exactly (mismatch
   0.0); the newest-first hypothesis mismatches by 0.2..5.4. Confirmed oldest first, filled at reset.
3. Depth frame order: frame i of step t equals frame i+1 of step t-5 exactly; frames first change at steps
   37, 32, 27, 22, 17, 12, 7, 2 for i = 0..7. **Oldest first, newest = index 7**, 5 control steps apart. The newest frame
   is one step old: frame 7 of step 1 is bit-identical to step 0, and against a MuJoCo ray-cast from the recorded poses
   the near-ground rows of steps 2..8 fit with mean abs error 0.0016 at lag 1 vs 0.0044 at lag 0 and 0.0050 at lag 2.
4. Depth vs MuJoCo flat ground from the recorded poses (ground assumed flat at the lowest sole height, lag 1):
   value range of the recording 0.415..0.668 (1.04..1.67 m); rows 5..17 (near) agree to within 0.001 on average (2.5 mm),
   rows 0..4 (far) are 0.003..0.021 shorter in Isaac, as expected with a stair riser ahead (the MuJoCo `stairs` tile shows
   the same kind of bump in the top rows of its first frame). Orientation: correlation +0.986 with the MuJoCo frame,
   -0.972 with it flipped top-bottom (near ground at the bottom rows: confirmed). Left/right: the per-step left-right
   depth difference correlates +0.86 with MuJoCo and has the same range (-0.022..0). The robot's own body is not in
   the image in either simulator in these poses (0 of 576 cropped pixels, 0 of 2304 uncropped).
5. ONNX plumbing: recorded observations -> `policy.onnx` on CPU: max |action - actions_onnx| = 1.2e-7.
6. Dynamics one control step ahead from each recorded state with Isaac's action (extra): in the air (steps 1..3, no
   contact) joint positions land within rms 0.004 rad (max 0.015) of Isaac; on the ground rms 0.011 (legs 0.018, max
   0.085) against a recorded motion of rms 0.021 rad per step. The reset step (0) fits worse (rms 0.013, max 0.040).
7. Closed loop for 1 s from the recorded initial state on flat ground with the recorded commands: Isaac moved 0.067 m,
   pelvis 0.731 -> 0.677 m, tilt at the end 8.4 deg (max 10.0). MuJoCo hull feet: moved 0.17..0.22 m (delay 0, 1, 2),
   0.731 -> 0.66..0.67 m, tilt at the end 6.7..7.7 deg (max 11..12), no fall. Sphere feet: 0.09..0.12 m, tilt 8.9..9.9.
   Same kind of behaviour (lands, stands leaning back, small drift); the trajectories are not the same (joint positions
   differ by rms 0.09..0.10 rad over the second).
8. Finding: the depth encoder output is exactly zero for all 50 recorded depth stacks (0 of 6400 latent values non-zero)
   and the actions do not change when the recorded depth is scaled by 0.8 or set to 1.0. At this checkpoint the policy
   does not use the depth image for realistic inputs.

## The clips (commanded 0.74 m/s, target 3.7 m ahead, hull feet, delay 0)

Both clips are numerically identical in robot motion (the policy ignores the depth image and the robot never reaches
the first step 1.25 m ahead):

* moved 0.15 m net in 14 s (+0.08 m forward, +0.13 m to the left), 0.03 m in the first second; pelvis path length
  1.45 m (sway); mean forward speed 0.005 m/s against 0.74 commanded
* 6 touchdown events = the landing of both feet from the 7 cm spawn drop plus 4 brief unloadings (longest 0.04 s):
  no steps; both feet on the ground 98.6% of the time
* pelvis settles at 0.64 m, leaning back 9 deg (pitch -0.17 rad), roll +2 deg; did not fall; no body other than the
  feet touched the ground; torque limit reached in 11 of 2800 physics steps, joint velocity limit exceeded in 1
* video vs CSV (frames in `checks/frames/`): the chase view shows the robot standing in place with a backward lean, arms
  down, the target flag ahead (on the landing in the stairs clip); the legs view shows both feet planted with small
  shuffles (right foot unloaded at 0.8 s in both the frame and the CSV). The render does not show contact forces, joint
  torques, or the 7 cm drop clearly (first 0.1 s); the lean is easier to see in the legs view than in the chase view.

Extra runs, no video (`checks/variants/`): delay 1 and 2 steps, vmax 0.45, depth lag 0, pyramid tile, sphere feet: all
stand, net 0.02..0.15 m in 14 s, no fall (delay 1/2: more shuffling, 15..22 lift-offs, each at most 0.06 s). Zero
command: stands, drifts 0.34 m in 14 s.

## Agreement with Isaac

Isaac: stands, 0 of 200 reach the target, robot 0 moved 0.07 m in its first second. MuJoCo: stands, 0.03 m in the first
second from the default spawn, 0.17..0.22 m in the first second from Isaac's recorded initial state (which has reset
velocities), then 0.15 m in 14 s. Agreement in kind; MuJoCo drifts somewhat more in the first second.

## Deviations from Isaac

* MuJoCo soft contacts (default solref/solimp, pyramidal cone, Newton solver, implicitfast) instead of PhysX; friction
  1.0 (training randomises 0.3..1.6), restitution 0.
* Exact convex hulls (PhysX hulls are limited to 64 vertices).
* Fixed action delay (`--delay-steps`, default 0); training randomises 0..2 physics steps per robot.
* No observation noise, no domain randomisation, no pushes, no reset noise.
* Joint velocity is not clamped (PhysX clamps to `velocity_limit_sim`); exceedances are counted.
* Terrain: clean boxes, no perlin roughness, no walls, straight flight instead of the pyramid for the clips.
* The first control step after a reset is not reproduced exactly (see validation 6).

## Not verified

* Depth on stairs: the Isaac terrain mesh was not available, so only flat-ground rows, orientation, range and timing
  were compared; the far rows could not be compared numerically.
* The robot's own body in the depth image: not visible in the recorded poses, so the self-hit path (visual meshes,
  0.1 m minimum distance) is untested against Isaac.
* Which action delay robot 0 had in the recording (the one-step test cannot separate 0 from 1: rms 0.0037 vs 0.0032).
* Contact behaviour while walking or climbing: the policy never walks, so foot-on-step contacts of the hull feet were
  not exercised by the policy. The stair tiles were only run with the robot standing in front of them.
* Ground height in the recording is an estimate (lowest sole point, steady to 0.1 mm over steps 25..49).
* Why the depth encoder output is zero (only observed on the ONNX files, not traced in the checkpoint).

---

# Follow-up: settings come from the run (run 1 and run 2)

## What changed

* `sim2sim_mujoco_x2.py`: `EnvConfig(run, card_json=None)` reads everything from the run's `env.yaml` (`--run`: the file,
  a folder with it, or a run folder with `params/env.yaml`): per-joint kp / kd / armature / effort / velocity limit,
  default pose, action joints and scale, observation layout, command rule, and the camera (body, offset, quaternion,
  apertures, crop, depth range, frame shape, history). The joint order is computed from the URDF (Isaac's breadth-first
  rule) and no longer taken from a card. `--export` names the folder with `policy.onnx` and, optionally,
  `adapter_card.json`; when the card is there every number in it is asserted equal to `env.yaml`. The defaults of
  `--run` / `--export` still point at run 1 (`assets/run`, `assets/eval`). Robot facts that are not in `env.yaml` stay in
  the file: URDF path, the filtered collision pairs of the training asset, the sole spheres.
* Captions are built from the run: line 1 "MuJoCo, not the training simulator; X2, Hiking track, run <run>, checkpoint
  <n>, commanded <v> m/s"; line 2 gains (label + kp/kd of hip pitch, knee, ankle pitch) and the camera mount (body,
  tilt, field of view, frame shape, range). Run and checkpoint come from the card (`--run-name`, `--checkpoint`,
  `--plant-label` override).
* `depth_check_run2.py` (new) writes `depth_check_run2.txt`.
* `assets/node_src/export_for_eval.py` (new version of the node's export script; `export_for_eval.orig.py` is the one
  fetched from the node). **It was NOT put back on the node**: overwriting the shared file was refused by the
  permission system of this session, and nothing was written to the node at all. The exports below were made on the
  desktop CPU with the same script (the hiking venv has `instinct_rl`), from checkpoints fetched read-only.

## Run 1 regression (nothing changed)

* `validation.txt` regenerated with the generalised code is byte-identical to the one before
  (`checks/run1_before/validation_before_generalising.txt`).
* The 14 s flat run gives a byte-identical CSV, with the old card and with the card written by the new export script.

## New export script against the existing run-1 card (model_500)

Regenerated on the desktop from `model_500.pt` + run 1's `params/`, with the recorded observations:
* `policy.onnx`: actions on the 50 recorded observations equal the existing export's exactly (max diff 0.0).
* Card: all 245 numeric values shared with the existing card are equal (kp, kd, armature, effort and velocity limits,
  default pose, action scale, offsets and sizes of every observation term, camera position, raw size, field of view,
  rates) except `onnx_vs_torch_max_abs_error` (7.7e-7 on the node, 1.1e-6 on the desktop: machine dependent).
* Text that differs: `obs_check` (path of the recording), `robot.model` (now the URDF path of env.yaml with the spawn
  settings), `control.head` (joint names, "hold at the default pose"), the `what` text of the terms (noise is "before
  scaling"; depth "oldest first (the last one is the newest)"), `observation.history` (adds "oldest first"),
  `command.training_ranges` / `how_training_makes_it` (generated per terrain; states vx = clip(2 * forward distance, 0,
  speed) and the 0.4 m stop radius), `depth.orientation` (no longer says "matches the URDF joint rgbd_head_front"),
  `depth.preprocessing` ("beyond range = 1" instead of "1.0"), `depth.stack` (the configured 0-1 frame delay is cast
  to an integer and is 0 in practice), `depth.noise`, `not_verified` (rewritten).
* Added: `run`, `source`, `held_joints`, `robot.joint_order_source`, `control.{actuator, gains_source, gains_label,
  gains_follow_armature_rule, action_scale_rule, action_delay_physics_steps}`, per term `shape / scale /
  noise_half_width`, `observation.noise_enabled_in_training`, command numbers, `depth.{quat_wxyz, convention,
  pitch_down_deg, apertures, crop_region, frame_shape, depth_range_m, output_range, blur, history_frames, stack_frames,
  stack_skip, pipeline}`, and `verified` (the facts checked against the Isaac recording, each marked "verified on run 1
  only"). Nothing was removed.
* Run 2 card (`assets/eval_v2_250/adapter_card.json`): kp, kd, effort, velocity limit, armature, default pose (31 joints)
  and the 29 action scales equal Isaac's read-back table `applied_Instinct-Parkour-TorsoCam-Amp-X2-v0.json` (0
  mismatches). Its `checkpoint` and `robot.model` fields hold desktop paths because it was exported here.

How to use it on the node (not done here):
`CUDA_VISIBLE_DEVICES= nice -n 10 ~/hiking-env/bin/python export_for_eval.py <run>/model_250.pt --plant mc_20261024`
(the task name is read from the run's `_supervisor/supervisor.log`; `--obs rec.npz` for a recording; `--card-only`
writes just the card). Not run on the node, so its node-specific branches (supervisor log, the cross-check against
`X2_ISAAC_JOINTS`) are untested.

## Run 2 depth check (`depth_check_run2.txt`), standing pose on flat ground

Camera from env.yaml: `torso_link`, offset (0.131, 0, 0.259), 61.6 deg down, 87.0 x 58.0 deg, crop (0, 0, 16, 16) ->
36 x 32, range 0-3.0 m, observation 10016. Lens 1.055 m above the ground, 1 mm ahead of the toe tips (foot hull).

| landmark | wanted | port |
|---|---|---|
| floor line 1.0 m ahead of the toes | 27% +-3% from the top | 25.8% (26.0% measured from the MJCF sole-sphere toe) |
| floor line 0.55 m ahead | near 52% | 51.5% (51.7%) |
| toe tips | just inside the bottom edge | project at 98.9% of the height, 34% / 66% from the left |

The same numbers come out of the ray table (centre ray of each row). The toe tips are inside the frame by 0.4 of a row:
no pixel centre lands on a foot in the standing pose (0 foot pixels). The top row sees the floor 1.61 m ahead of the toes.
Own body in the 36 x 32 frame: standing 1.4% (16 pixels, bottom two rows: thighs / knees); synthetic mid-stride pose
(one leg swung forward) 7.8% (thigh, knee, foot of the leading leg, rows 25-35); over the poses of the checkpoint-250
rollout 0.6% to 9.8%, mean 7.2%.

Run 2's observation build is validated against Isaac only through run 1's recording (same code path, run 1's camera
and crop) until an Isaac recording of run 2 exists. The run-2 check above is geometry in MuJoCo only.

## Run 2, checkpoint 250, 14 s on flat ground (no video): `x2v2_ckpt250_mujoco_flat.csv / .json`

Commanded 0.74 m/s, target 3.7 m ahead, hull feet, delay 0: **stands and shuffles in place, drifting backwards**; no
fall. Net 0.54 m in 14 s, all of it backwards (x -0.53 m); mean forward speed -0.039 m/s against +0.74 commanded;
0.03 m in the first second. 46 touchdowns (2 are the landing), 56 lift-offs, the longest 0.12 s: shuffling, not
steps that carry the robot forward. Both feet down 87% of the time, pelvis 0.65 m, leaning back 6.7 deg, torque limit
reached in 242 of 2800 physics steps, nothing but the feet touched the ground.
Other runs (`checks/variants_v2/`): delay 1 step 0.09 m net, delay 2 steps 0.15 m, sphere feet 0.02 m, zero command
0.07 m; none falls. The stairs tile gives the bit-identical run although the depth frames differ by up to 0.11: the
actions did not depend on the image there (on other frames of the rollout the encoder output is non-zero, and scaling
the depth by 0.8 moves the actions by 0.013 at most).

## Not verified (follow-up)

* The new export script on the node (not uploaded, not run there); the node folder `eval/x2_parkour_v2_250/` was not created.
* Run 2 against Isaac in any form (no recording): depth image, gains in closed loop.
* Mid-stride body coverage uses a synthetic pose and a rollout of a policy that does not walk yet.
* Holding the default pose with zero actions falls within 5 s in MuJoCo with either run's gains, so standing depends
  on the policy; whether Isaac does the same was not checked.
