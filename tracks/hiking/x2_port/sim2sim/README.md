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
