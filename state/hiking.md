# hiking state

Updated 2026-11-11 (Hiking track agent). Node `hiking`, <hiking-node>, user the login in `config/nodes.local.json`.

## What is running

- **X2 paper-terrain run `x2_parkour_v1`**: task `Instinct-Parkour-Target-Amp-X2-v0`, 4096 robots, to 30,000 iterations, started
  2026-11-11 20:22 (node time, UTC). tmux session `hiking_x2` on the node, under `code/x2_port/supervise_x2.py` (stall detection
  900 s without log output, kill and resume from the newest checkpoint, one GPU job, checkpoint every 250 iterations).
- Heartbeat: `/mnt/filesystem-g4/status/hiking.json` (every 10 s: state, iteration, reward, episode length, terrain level,
  newest checkpoint, GPU).
- Run folder: `/mnt/filesystem-g4/tracks/hiking/runs/x2_parkour_v1/` (one timestamped folder per start or resume, `model_<it>.pt`
  inside; `_supervisor/train.log`, `_supervisor/supervisor.log`).
- Newest checkpoint: read `checkpoint` in the heartbeat (see "Early signals" below for the last one I looked at).

## Commands

```bash
S="ssh -i ~/.ssh/nebius_corpus -o IdentitiesOnly=yes -o BatchMode=yes <user>@<hiking-node>"
$S 'cat /mnt/filesystem-g4/status/hiking.json'                                             # state
$S 'tail -5 /mnt/filesystem-g4/tracks/hiking/runs/x2_parkour_v1/_supervisor/supervisor.log'
$S 'touch /mnt/filesystem-g4/tracks/hiking/runs/x2_parkour_v1/_supervisor/STOP'            # stop (supervisor kills training, exits)
$S 'bash /mnt/filesystem-g4/tracks/hiking/code/x2_port/start_x2_parkour_v1.sh'             # start or resume (newest checkpoint)
$S 'bash /mnt/filesystem-g4/tracks/hiking/env/setup_node.sh'                               # rebuild code tree + env (idempotent)
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
- 2026-11-11: Linux pins differ from the desktop: numpy 1.26.0 (Isaac Sim's pin), mujoco 3.3.7 (newer needs websockets >= 13,
  Isaac Sim needs 12.0), flatdict built without isolation, instinct_rl installed without its torch 2.11 pin.
- 2026-11-11: pelvis root, 29 commanded joints, head camera, mesh feet, filtered pairs: see the table.
- 2026-11-11: wrote `data/robots` and `data/motions` myself, as the coordinator's task message said (TRACK_RULES says to ask).

## Early signals

(filled in below as the run progresses)

## Next

1. Watch `x2_parkour_v1` at a few hundred and a few thousand iterations (walks? terrain level rising?).
2. MuJoCo sim2sim of an X2 checkpoint (port `scripts/sim2sim_mujoco.py` to the X2 joint order and camera) before trusting it.
3. Then the extended task as a second task name (`docs/TODO_extended_training.md`, generators in `review/new_terrains/`); the two
   challenge apparatuses stay out of training as whole layouts.

## Blockers

None.
