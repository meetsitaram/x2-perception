# ssr state

Updated 2026-11-11 by the SSR track agent. A new agent can continue from this file alone.

## Where things are

- Node `ssr`: `ssh -i ~/.ssh/nebius_corpus -o IdentitiesOnly=yes -o BatchMode=yes <user>@<ssr-node>` (helper on the desktop:
  `~/bin/ssr_ssh.sh <timeout_s> '<command>'`). Passwordless sudo works (verified).
- Code: `/mnt/filesystem-g4/tracks/ssr/code` (copy of the desktop `ssr-x2` at commit `7ab61ff`, without `runs/`, `.git`,
  `data/bc.npz` and the old logs; the desktop repo is untouched). Changed files: `ssr/env.py`, `ssr/train.py`, new
  `scripts/supervise_node.py`. `scripts/bc.py` (the SuperMimic fit) is deleted from the copy.
- Environment: `/mnt/filesystem-g4/tracks/ssr/env/venv` (Python 3.12, torch 2.11.0+cu128, mujoco 3.17.0, mujoco-warp 3.17.0,
  warp-lang 1.18.0).
- Runs and checkpoints: `/mnt/filesystem-g4/tracks/ssr/runs/<name>/` (`last.pt` every 50 iterations, `ckpt_NNNNNN.pt` every 500,
  `log.jsonl`); process output in `runs/<name>.log`; supervisor log `runs/supervisor.log`; heartbeat
  `/mnt/filesystem-g4/status/ssr.json`.
- Inputs copied to the shared filesystem: only inside `models/ssr/code` (`data/amp.npz`, 47 MB, the motion priors built on the
  desktop by `scripts/build_amp.py`; robot model in `assets/robot`; `assets/route` CAD raster, used by the route test only, never
  by training). Nothing was written to `/mnt/filesystem-g4/data`.

## The paper's training pipeline (arXiv 2605.30770), stage by stage

There is one stage. The paper calls it a "unified single-stage end-to-end reinforcement learning framework": no teacher, no
distillation, no tracking of reference motions. Everything below trains together from the first iteration.

1. **Simulation.** Isaac Gym, 4,096 X2 robots, depth by Warp ray casting against terrain and robot meshes (self-occlusion), 50 Hz
   policy, 21 joints (legs, waist yaw, 4 per arm), action scale 0.25 legs / 0.2 waist and arms. About 20k iterations on one RTX 4090.
2. **Terrain and curriculum.** Five types (slope 0-20 deg, stairs 25-40 cm tread and 5-20 cm rise, discrete obstacles 5-20 cm,
   gaps 10-80 cm, platforms 5-40 cm), 20 levels each, 8 x 4 m tiles, promoted and demoted as in Rudin et al. 2021 (legged_gym).
   Commands: walk along world x at U(0, 1.2) m/s (slopes U(-1, 1)) holding a heading (+-80 deg slopes, +-30 stairs/discrete, +-15
   gap/platform); yaw rate from the heading error. After the top level, an omnidirectional stage: vx U(-1, 1), vy U(-0.4, 0.4),
   yaw rate U(-1.2, 1.2) in the base frame. No flat terrain, no command curriculum, no command resampling rule is given.
3. **Observations.** Actor: 72-D proprioception (base angular velocity, gravity, command, joint positions, joint velocities, last
   action), a short history of it (length not given), a 36 x 36 depth image; the actor itself sees [proprioception, estimated base
   velocity (3), latent (48 = three 16-D heads)]. Critics (privileged): true base velocity, foot velocities, contacts, hand and foot
   positions, height maps around base and feet.
4. **Networks.** Mirror-equivariant encoder (depth CNN + proprioception MLP -> GRU 256 -> fusion -> 3 heads), velocity estimator,
   mixture-of-experts actor (5 experts), three critics, foothold imagination MLP, five discriminators.
5. **Rewards (RL).** Three groups, each with its own critic, weights 1.0 / 0.25 / 0.2:
   - locomotion (Table 6): velocity and yaw-rate tracking, orientation, roll/pitch rate, base height 0.68 m, action rate and
     smoothness, joint velocity / torque / deviation / limits, stand-still, single support (+0.2 when exactly one foot was down in
     the last 0.2 s, or the command is zero), impact, slip, air time over 0.4 s (penalty), stumble, lateral foot distance;
   - foothold: exp(-(sum of support deficiency)^2 / 0.0625); stance feet use the real sole patch, swing feet the expected deficiency
     under the imagined landing; switched on for good once the global mean terrain level exceeds 2; zero on slopes;
   - style: max(0, 1 - 0.25 (D_i - 1)^2) from the discriminator of the terrain type the robot is on.
6. **Motion prior.** Adversarial motion prior only (no tracking): 71 clips, 521 s (own mocap + AMASS subset), retargeted, split
   into five per-terrain sets with reuse; discriminator input is 5 frames x 63-D (gravity, base linear and angular velocity, joint
   positions and velocities, hand and foot positions); gradient penalty 20, weight decay 1e-4.
7. **PPO.** 24 steps, 5 epochs, 4 mini-batches, gamma 0.99, lambda 0.95, entropy 0.005, value coefficient 1.0, desired KL 0.01,
   minimum std 0.2, four Adam optimisers at 5e-4 (PPO; encoder + estimator; foothold model; discriminators). Each update also uses
   the mirrored copy of every sample (actor input, privileged state, action), the encoder being run on the original only.
8. **Domain randomisation (Table 10).** Base mass -2..12.5 kg, base CoM +-10 cm, link mass x0.9-1.1, link CoM +-1 cm, motor
   strength x0.8-1.2, PD gains x0.9-1.1, friction 0.2-1.25, action delay 0-2 steps, push 0.4 m/s every 12 s, camera pose / FoV
   offsets, noise on self-occluded pixels.
9. **Terminations (A.7).** Time limit, leaving the tile, contact force on a termination link, severe tilt, a foot far below ground
   on gaps. Thresholds not given.

**What is RL:** the actor and the three critics (PPO), driven by the three reward groups.
**What is not RL (but trains in the same loop):** the velocity estimator and the encoder's reconstruction / next-proprioception /
VAE losses (supervised), the foothold imagination model (Gaussian likelihood on future contacts), the discriminators (adversarial
classification of data vs policy). Motion data reaches the policy only through the style reward. No controller is imitated.

**Not stated in the paper:** the reset state (default pose or reference frames), how the critics' advantages are combined
(normalised jointly or each on its own), whether rewards are multiplied by dt, initial action std, history length, the support
tolerance, promotion/demotion thresholds, episode length. The paper's ablation without the style reward still traverses every
terrain (Table 1, "NoStyle"), so in the paper walking does not depend on the motion prior.

## SuperMimic: removed

- `scripts/bc.py` (collects SuperMimic tracker rollouts and fits the actor to them) deleted from the track copy; `data/bc.npz`
  never copied; no checkpoint from `chestbc` / `headbc` or any earlier run copied (all earlier runs stay on the desktop, unused).
- `ssr/train.py` has no code path that loads a SuperMimic checkpoint (it never did: the fit wrote a `last.pt` that `--resume`
  picked up). Every run here starts from random weights.
- `data/amp.npz` is retargeted human motion (BONES-SEED walking, Pico captures), not tracker output: it stays; it is the paper's
  motion prior.

## Decisions on the other additions in PAPER_NOTES

| Addition | Decision | Why |
| --- | --- | --- |
| Reference-state starts (half the walking episodes begin from captured walking frames) | **keep** (`--rsi_frac 0.5`) | The paper does not say how episodes start; this is the usual start for AMP training and adds no reward and no teacher. First lever to vary if needed (0 = default pose only). |
| Early schedule (velocity reward x3, joint-deviation x0.25, stepping bonus, friction and mass eased in, to iteration 3000) | **drop** (`--boot_iters 0`) | Changes the paper's rewards and randomisation. Kept behind a flag as a later fallback only. |
| Walking-clip tracking reward (`--imit_w`) | **drop** (0) | A tracking reward is not SSR. |
| Standing-still termination | **drop** (off; `--stuck_term` brings it back) | Not in the paper; the Rudin curriculum already demotes robots that do not travel. |
| Gate prior (cross-entropy tying experts to terrain types) | **drop** (`--gate_prior 0`) | Not in the paper; the mixture learns its own gating. |
| Side obstacles on tiles (walls, rails, door frames) | **keep** | The apparatus has a door, walls and rails; these are separate, randomised pieces, not the layout. |
| Rolling terrain map (`--map_memory`) | **drop for now** | The paper relies on the recurrent encoder; the map added instability earlier (NaN at 780). Revisit only if foot placement under the body proves to be the limit. |
| Chest camera | **drop for now**: head camera (`--camera head`) | The head Gemini 335 is what the X2 has; the chest camera would be added hardware. Revisit with eval numbers. |
| Route terrain types (ramp, stairs, beam, platform, flat lane) instead of the paper's five | **keep** | The eval is the apparatus; gaps and discrete blocks are not on it. |

The apparatus as a whole is not in training: `ssr.train` builds its terrain from `ssr/terrain.py` generators only (`SSREnv` with
`raster=None`); `assets/route` is read by `scripts/route_eval.py` / `render_route.py` only. Parts that resemble apparatus pieces
in the training tiles: beams across the lane (4x4-like), ramps, stair flights, platforms, lane walls, rails on posts, door frames;
all generated with random sizes and positions.

## Changes made on the node copy (2026-11-11), with reasons

1. **All rewards multiplied by the control step (0.02)** (`--rew_scale`). legged_gym does this and the paper follows legged_gym's
   curriculum and rsl_rl's settings. It leaves the objective unchanged (one common factor; advantages are normalised) but brings
   returns from about 300 down to about 6, which the critics can fit. In the earlier runs the value loss sat at 20-95 and the
   KL-adaptive step collapsed to 3e-5 to 6e-5: my leading suspect for the robot never leaving the standing optimum.
2. **Initial action std 1.0** (rsl_rl default) instead of 0.5: more exploration at the start. Paper gives only the minimum (0.2).
3. **Foothold guidance gate as in the paper**: on for good once the mean terrain level of the walkers exceeds 2 (was: per robot,
   above level 5).
4. **Noise on self-occluded depth pixels**, N(0, 0.1 m), training only (Table 10; was missing).
5. Early schedule, stuck termination, gate prior: off by default (see table). Flags keep them available.
6. Options added for diagnosis: `--adv_norm separate` (each critic's advantage standardised before weighting, which raises the
   motion prior's share of the policy gradient to 0.2/1.45), `--w_style`, `--rsi_frac`.
7. Checkpointing: `last.pt` written atomically at every save; `ckpt_*.pt` every 500; checkpoint stores `guide_on` and the args.
8. Log: added `gait` (for walkers commanded 0.3 m/s or more: `vx_ratio` = world-x speed / commanded speed, share of steps in
   single / double support / flight, touchdowns per second), and `lvl_mean`.

## Remaining deviations from the paper

| Paper | Here | Reason |
| --- | --- | --- |
| Isaac Gym | MuJoCo Warp, 200 Hz physics, 50 Hz policy | The user's stack; no Isaac Gym build for this GPU generation |
| Waist D435i looking forward | Head Gemini 335, 40 deg down | The X2's own camera |
| Own mocap + AMASS, 71 clips | BONES-SEED walking + Pico terrain captures (`data/amp.npz`) | The paper's data is not released |
| Terrain: slope, stairs, discrete, gap, platform | ramp, stairs, beam, platform, flat lane; side obstacles on the tiles | Apparatus content; see table above |
| Omnidirectional stage after the top level | 10 % of robots always on the open flat tile with omnidirectional commands | Same exposure, simpler |
| Contact-force termination on links | Trunk, thighs, head end by terrain clearance; elbows by force; touching a side obstacle ends the episode | mujoco_warp heightfield contact limits; obstacles are vision-only geometry |
| Depth against robot meshes | Spheres on the limbs for self-occlusion | Cost |
| Reset state not given | Default pose with joint noise; half of walking episodes from captured walking frames | See decisions |
| Actor mean unbounded | 8 tanh(x/8); PPO log-ratio clamp; skipped non-finite updates | Guards from the issues log; inactive in normal operation |
| Waist pitch and roll: not stated | Held at kp 120 / kd 6 | They sagged at the plant gains |

Paper-silent values kept from PAPER_NOTES: history 5, support tolerance 3 cm, 17 x 10 base height map at 10 cm, 3 x 3 imagined
landing stencil, PPO clip 0.2, action clip 8, squared tilt in the orientation reward.

## Run history (the current status is at the end of this file)

### Run `r1` (started 2026-11-11 20:05 UTC): first from-scratch run, no SuperMimic

- Command (inside tmux session `ssr` on the node):
  `cd /mnt/filesystem-g4/tracks/ssr/code && ../env/venv/bin/python scripts/supervise_node.py --name r1 --iters 20000`
  which runs `python -m ssr.train --num_envs 4096 --iters 20000 --name r1 --out_root /mnt/filesystem-g4/tracks/ssr/runs --save_every 50`
  (defaults: head camera, no map memory, `--rsi_frac 0.5 --rew_scale 0.02 --init_std 1.0 --adv_norm joint --w_style 0.2
  --boot_iters 0 --gate_prior 0`, no stuck termination).
- Speed: about 3.9 s per iteration (4,096 robots x 24 steps), so 20k iterations is about 22 h. GPU memory 18 GB of 96.
- Watch: `~/bin/ssr_ssh.sh 30 'python3 /mnt/filesystem-g4/tracks/ssr/code/scripts/summ.py r1 100'` (one line per 100
  iterations); heartbeat `cat /mnt/filesystem-g4/status/ssr.json`.
- Stop: `touch /mnt/filesystem-g4/tracks/ssr/runs/STOP` (the supervisor kills the run and exits). Resume: run the same tmux
  command again; it continues from `runs/r1/last.pt`.
- Smoke test before it (verified): environment bench 63.6k env-steps/s at 4,096 robots, 12 training iterations, save and the
  mirror tests (`tests/test_mirror.py`: all pass).

**r1 result: the robot stands still again.** Stopped by me at iteration 440 (20:35 UTC) to free the GPU. Numbers
(walkers commanded 0.3 m/s or more; `vx` = speed along the tile / commanded speed):

| iteration | episode length | r_l per step | style reward | mean level | vx | single support | 
| --- | --- | --- | --- | --- | --- | --- |
| 100 | about 4 s | 1.1 | 0.66 | 0.9 | 0.00 | 0.15 |
| 200 | 12.2 s | 1.49 | 0.53 | 0.12 | 0.03 | 0.09 |
| 300 | 13.5 s | 1.58 | 0.32 | 0.01 | 0.03 | 0.11 |
| 440 | 15.5 s | 1.68 | 0.26 | 0.00 | 0.03 | 0.11 |

Same course as the desktop runs: the policy learns to brake from the walking starts and stand, the curriculum falls to level 0,
the discriminator separates it from the data (style reward 0.26 and falling), and nothing pulls it out. So multiplying rewards
by dt and the wider initial std were not the cause. `runs/r1/last.pt` (iteration 400) is kept as the standing reference.

Why standing wins, from the reward table: a perfect walker gains about +0.6 per step in velocity tracking and +0.1 in single
support over standing, and loses about 0.5 in posture, joint-deviation and smoothness terms; with the style reward at weight 0.2
inside one jointly normalised advantage the motion prior adds at most +0.15. The margin is small and any fall costs seconds of
reward, so the search never leaves standing.

### Run `r2` (started 2026-11-11 20:35 UTC): motion prior at its stated weight, per-critic advantage normalisation

- Same as r1 plus `--adv_norm separate`: each of the three critics' advantages is standardised before the 1.0 / 0.25 / 0.2
  weighting (the paper has three critics and does not say how they are combined). The style group then holds 0.2/1.45 = 14 % of
  the policy gradient whatever the size of its reward. The paper's weights are unchanged.
- Command: `cd /mnt/filesystem-g4/tracks/ssr/code && ../env/venv/bin/python scripts/supervise_node.py --name r2 --iters 20000 -- --adv_norm separate`
  (tmux session `ssr`). Log line now also carries `adv_sd`, the weighted spread of each critic's advantage.
- r2 was paused at iteration 50 (20:40 UTC) for the throughput benchmark below and resumed at 20:46 from `runs/r2/last.pt`
  with **8,192 robots**: `... supervise_node.py --name r2 --iters 20000 --num_envs 8192 -- --adv_norm separate`.
- Measured at iteration 60 (`adv_sd`): weighted advantage spread locomotion 0.142, foothold 0.023, style 0.011. Under r1's joint
  normalisation the motion prior was about 7 % of the locomotion signal; under r2 it is 0.2 against 1.0.

## GPU use (user, 2026-11-11: "make sure the GPUs are well used"); benchmark 20:40-20:45 UTC, nothing else on the GPU

`runs/bench.sh` on the node: environment only (`scripts/bench_env.py`, zero actions) and 8 full training iterations per count.

| Robots | Env only, env-steps/s | Depth render per step | Training iteration | Training samples/s | vs 4,096 | GPU memory (nvidia-smi) |
| --- | --- | --- | --- | --- | --- | --- |
| 4,096 (paper) | 63.7k | 27 ms | 3.95 s | 24.9k | | 18 GB |
| 8,192 | 69.2k | 54 ms | 5.9 s | 33.3k | +34 % | 34 GB |
| 16,384 | 63.6k | 130 ms | 10.45 s | 37.6k | +51 % | 65 GB |

- The environment step is the limit and it is GPU compute, not memory: its time grows in proportion to the robot count
  (64 ms, 118 ms, 257 ms per step), so more robots do not raise simulation throughput. The gain at larger counts comes only from
  the update phase (network passes, PPO epochs, encoder roll-out), which grows less than linearly.
- About 42 % of the environment step is the depth render (a 30-pass ray march plus a 20-sphere occlusion loop in PyTorch); the
  rest is mujoco_warp physics (4 substeps), contact read-out and observations. CPU: one Python thread drives it; the other 23
  cores have nothing to do in this design.
- Decision: **8,192 robots**. +34 % samples per second for 1.5x the time per iteration. 16,384 gives +51 % but 2.65x per
  iteration, and the open question (does a gait start) is answered in iterations as much as in samples. Kept fixed: 24 steps,
  5 epochs, 4 mini-batches (so each mini-batch is twice the paper's), learning rate, all rewards and randomisation. The paper
  uses 4,096.
- Not done yet, and the real route to using the card: move the depth ray march into a Warp kernel (expected to cut the
  environment step by a third or more) and then re-benchmark.

**r2 result: stands still too.** Stopped at iteration 360 (21:17 UTC). vx 0.38 (it 20), 0.07 (120), 0.04 (210), 0.04 (290),
0.03 (360); episode length 17.7 s, mean level 0.01, double support 0.88, style reward 0.35. Raising the motion prior's share
alone does not start a gait. `runs/r2/last.pt` = iteration 350.

**Finding (21:15 UTC): the motion prior itself contains standing.** Measured on `data/amp.npz` (desktop, CPU):

| Set | Seconds | Frames with planar speed under 0.1 m/s | Standing (also joints still) | Median forward speed |
| --- | --- | --- | --- | --- |
| ramp | 564 | 21.7 % | 15.4 % | 0.21 m/s |
| stairs | 1241 | 24.8 % | 20.4 % | 0.24 m/s |
| beam | 605 | 21.5 % | 15.1 % | 0.14 m/s |
| platform | 922 | 30.9 % | 24.3 % | 0.19 m/s |
| flat | 406 | 25.9 % | 17.7 % | 0.09 m/s |

Two consequences, both pushing towards standing: (1) the discriminators are shown standing as reference motion, so a standing
robot keeps a style reward of about 0.35 instead of about 0; (2) the walking starts draw from the flat set and set the commanded
speed to the frame's forward speed, so half of all walking episodes were commanded at a median 0.09 m/s, below the 0.15 m/s
zero-command threshold under which the paper's rewards pay for standing. The paper's prior is traversal clips (71 clips, 521 s)
used with commands U(0, 1.2) m/s.

What the paper specifies for the levers the coordinator listed (checked in the paper text, 2026-11-11):

| Lever | Paper | Here before r3 |
| --- | --- | --- |
| Motion prior weight | style group 0.2 (Table 6); how the three advantages are combined is not stated | 0.2; joint (r1) and per-critic (r2) normalisation both stand |
| Reference windows | 5 frames x 63-D from traversal clips; nothing on standing frames | 15-24 % standing frames |
| Start state | not stated | default pose, half from "walking" frames (which were often standing) |
| Commands | forward U(0, 1.2) m/s (slopes U(-1, 1)), heading range per terrain; no minimum speed, no command curriculum, no resampling rule | same, except walking starts overrode the speed with the frame's |
| Early penalty schedule | none | none (dropped) |
| Termination for not moving | none (time limit, leaving the tile, link contact force, severe tilt, foot below ground on gaps) | none (dropped) |

### Run `r3` (started 2026-11-11 21:18 UTC): moving frames only in the prior and in the walking starts

- r2 plus `--prior_min_speed 0.2` (discriminator reference windows need a mean planar speed of 0.2 m/s or more: 55-64 % of
  windows kept per set) and `--rsi_min_speed 0.3` (walking starts only from frames moving forward at 0.3 m/s or more: 41 % of
  the flat frames). No new reward, no new termination. Fresh start, 8,192 robots.
- Command: `cd /mnt/filesystem-g4/tracks/ssr/code && ../env/venv/bin/python scripts/supervise_node.py --name r3 --iters 20000 --num_envs 8192 -- --adv_norm separate --prior_min_speed 0.2 --rsi_min_speed 0.3`
- Motion sets checked by clip name (desktop): single-part captures only (stairs up/down, ramp, 4x4 and 2x2 crossings, box
  step-ups, flat walking); no clip recorded on a full apparatus route.

## Backup and restore (user rule, 2026-11-11)

Everything a restart needs is written directly on the shared filesystem; training never writes to the node's local disk, so
there is no copy step that can be forgotten.

| Kind | Where (under `/mnt/filesystem-g4/`) | Written by |
| --- | --- | --- |
| Code and configs (with my changes) | `models/ssr/code/` | me, by scp; the change log is in this file |
| Checkpoints | `tracks/ssr/runs/<run>/last.pt` (every 50 iterations, about every 5 min), `prev.pt` (the one before), `ckpt_NNNNNN.pt` every 1,000 | `ssr.train` |
| Training logs | `tracks/ssr/runs/<run>/log.jsonl` (every 10 iterations), `runs/<run>.log` (process output), `runs/supervisor.log`, `runs/<run>/args.json` | `ssr.train`, supervisor |
| Benchmarks | `tracks/ssr/runs/bench.sh`, `runs/bench.log` | me |
| Motion prior | `models/ssr/code/data/amp.npz` (63-D frames per terrain set) | built on the desktop by `scripts/build_amp.py` |
| Motion prior sources (retargeted clips, 85 MB) | `data/motions/ssr_prior_sources/apparatus_build_train/` (6 files) and `.../terrain_park_train_sets/` (6 files), copied from the desktop 2026-11-11 21:40 UTC | me; `build_amp.py` reads them (its two path constants point at the desktop folders and must be changed to these to rebuild on the node) |
| Terrain | generated at start from `ssr/terrain.py` with the run's seed; nothing to store | |
| Python environment | `tracks/ssr/env/venv` (also on the shared filesystem) | see below |
| Only on the node's local disk | the Warp kernel cache (`~/.cache/warp`, rebuilt automatically in about 2 min) and the tmux session | |

Retention: `last.pt` + `prev.pt` (newest two) + one numbered checkpoint per 1,000 iterations. Runs r1-r3 used the earlier
setting (numbered every 500, no `prev.pt`).

Restore on a fresh node: mount the shared filesystem, then `sudo apt-get install -y python3.12-venv tmux` and, only if
`tracks/ssr/env/venv` is missing:
`python3 -m venv env/venv && env/venv/bin/pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128 && env/venv/bin/pip install mujoco==3.17.0 mujoco-warp==3.17.0 warp-lang==1.18.0 numpy==2.5.4 scipy==1.18.1 joblib==1.6.0 onnx==1.23.2 onnxruntime==1.31.1 pyyaml`
(Python 3.12.3, NVIDIA driver 580.173). Then start the supervisor command of the current run (see Status); it resumes from
`last.pt`. If `last.pt` does not load, copy `prev.pt` over it.

## The paper's own evaluation, and ours for the X2

What the paper evaluates (Sec. 4, Tables 1-2, Figs. 5-7):

| Item | Paper | Paper's figure for SSR |
| --- | --- | --- |
| Success rate (sim) | 1,000 randomised trials per terrain at a 1.0 m/s forward command; success = a 20 s traversal without termination | "near-100 %" at all training difficulties (Fig. 5; a plot, no table) |
| Safe foothold rate (SFR) | share of footholds with support ratio above 75 % (support ratio = share of the 22.5 x 10 cm sole patch within tolerance of the sole height) | highest of all methods (Fig. 6; no numbers in the text) |
| Mean support ratio (MSR) | mean support ratio over footholds; first checkpoint reaching 75 % | earliest (Fig. 6) |
| Average power (W), peak foot contact force (N) | Table 1 | stairs up 381.3 W / 507.7 N; stairs down 342.4 / 519.2; gap 214.4 / 463.4; high platform 253.0 / 480.4 |
| Training curves | episodic return, share of robots reaching the top terrain level, peak GPU memory (23.7 GB) | Fig. 7c |
| Symmetry | either-leg-led traversal, shown qualitatively | Fig. 7 |
| Real robot | 20 trials per terrain | stairs up and down 15/30 cm 20/20; gap 80 cm 19/20; platform 40 cm 20/20; gap 90 cm 17/20; platform 45 cm 19/20; one 1.3 km outdoor walk |
| Ablations and baselines | NoFoothold, NoImgn, NoSym, InpSym, NoStyle, SglDisc; HPL, PIM | Table 1 and Figs. 5-7 |

Ours: `scripts/paper_eval.py` in the code copy (from the desktop repo) computes, per terrain type at levels 4, 9, 14, 19: the
paper's survive-20-s rate at 1.0 m/s over 1,000 randomised trials, plus "crossed" (reached the far end, because a standing
robot also survives 20 s), SFR and MSR at touchdowns, average power, peak foot force, velocity tracking in eight directions
and two yaw rates, which foot leads, and the gate weights. Command (a GPU job: only when training is stopped):
`../env/venv/bin/python scripts/paper_eval.py --ckpt ../runs/<run>/last.pt --out ../eval/<run>_<iter>`.
**Not run yet on this node**: there is no walking checkpoint, and a standing one scores 100 % on the paper's success
definition, which says nothing. Training itself logs MSR, SFR, power and the top-level share every 10 iterations (legend below).

Cannot be reproduced: gap and discrete-obstacle terrains (not in our terrain set; beams and the flat lane replace them), the
real-robot trials, the outdoor walk, the DR02 results, the HPL and PIM baselines, and the Fig. 5/6 numbers (plots only).
Ablations are possible later as extra runs, one GPU job each.

## Training log legend (`runs/<run>/log.jsonl`, one JSON object every 10 iterations; full set from run r4 on)

Per-step means over all robots and the 24 steps of the iteration unless noted. W = the paper's weight (Table 6).

| Name | Meaning | Paper | W |
| --- | --- | --- | --- |
| `it`, `dt`, `elapsed_h`, `samples_per_s`, `num_envs`, `gpu_util`, `gpu_mem_mb`, `gpu_gb` | iteration, seconds per iteration, hours, robot-steps per second, robots, GPU utilisation % and memory at log time, peak torch memory | | |
| `r_l`, `r_f`, `r_s` | the three reward groups per step, unweighted | r^l, r^f, r^s | 1.0, 0.25, 0.2 |
| `group_weighted` | the same times the group weight | | |
| `terms` / `terms_raw` | each locomotion term: weighted contribution / value before the weight | Table 6 | |
| `lin_vel` | exp(-abs(v_xy - cmd)^2 / 0.25) | linear velocity tracking | 1.0 |
| `ang_vel` | exp(-(yaw rate - cmd)^2 / 0.25) | yaw rate tracking | 0.8 |
| `orientation` | exp(-abs(g_xy)^2 / 0.01) | orientation | 0.5 |
| `ang_vel_xy` | exp(-abs(w_xy)^2 / 0.25) | roll/pitch rate | 0.25 |
| `base_height` | exp(-(h - 0.68)^2 / 0.01) | base height | 0.4 |
| `action_rate`, `smoothness` | mean squared first / second action difference | action rate, smoothness | -0.12, -0.06 |
| `dof_vel`, `dof_torque` | mean squared joint velocity / torque over its limit | DoF velocity, torque | -0.96, -0.6 |
| `dof_dev` | mean absolute joint offset from default | DoF deviation | -1.8 |
| `dof_pos_lim`, `dof_vel_lim`, `dof_tau_lim` | share of joints beyond position / velocity / torque limit | DoF limits | -1.5, -6, -6 |
| `stand_still` | joint offset when the command is under 0.15 | stand still | -0.12 |
| `single_support` | 1 if a single-support step occurred in the last 0.2 s, or the command is under 0.15 | single support | 0.2 |
| `impact`, `slip` | squared vertical / horizontal foot speed while in contact | impact velocity, contact slippage | -1.3, -0.2 |
| `air_time` | air time beyond 0.4 s at touchdown | feet air time | -2.0 |
| `stumble` | horizontal foot force over 3x vertical | feet stumble | -2.0 |
| `lateral` | exp(min(d_y - 0.22, 0) / 0.03) | feet lateral distance | 0.08 |
| `air_bonus`, `imitate` | early stepping bonus, clip tracking (both off: always 0) | not in the paper | 0 |
| `disc.<type>` | per discriminator: `loss`, mean output on data `d_data` and on the policy `d_policy`, gradient penalty, `style_reward` of the robots on that type, their `share` | AMP, r^s | |
| `d_r`, `d_f` | the same outputs averaged over the five | | |
| `foothold` | `touchdowns` counted, `msr` mean support ratio and `sfr` share above 75 % at touchdown (not on ramps) | Sec. 4.2 | |
| `foot_cm`, `foot_nll` | error and loss of the imagined landing | Eq. 1 | |
| `guide_on` | imagined foothold guidance active (mean level passed 2) | Sec. 3.2 | |
| `done_share`, `dones_per_iter` | share of episode ends by cause: `tilt`, `low`, `link_force`, `clearance`, `diverged`, `side_obstacle`, `left_tile_side`, `left_tile_back`, `stuck` (off), `reached_end`, `omni_left_tile`, `time_limit` | App. A.7 | |
| `levels_by_type`, `lvl_mean`, `level_max_share` | curriculum level per terrain type, mean, share of robots at the top level | Fig. 7c | |
| `gait` (robots commanded 0.3 m/s or more) | `vx_ratio` achieved / commanded speed along the tile, `cmd_speed`, `vx_mps`, `lin_err` (m/s), `yaw_rate_err`, `yaw_rate_cmd` (rad/s), `single` / `double` / `flight` share of steps, `steps_per_s` touchdowns per second, `power_w` sum of abs(torque x joint velocity), `foot_force_max_n` | Table 1 (power, force); commands Table 4 | |
| `ep_ret`, `ep_len_s`, `n_done` | weighted, dt-scaled return and length of finished episodes | Fig. 7c | |
| `kl`, `lr`, `pi`, `v`, `std`, `entropy_std`, `adv_sd`, `skipped` | PPO: KL, step size, losses, action std (mean and per joint), weighted advantage spread per critic, skipped updates | Table 9 | |
| `hb`, `hf`, `np`, `v_est` | encoder losses: base and foot height maps, next proprioception, velocity estimate | Table 9 | 2, 1, 5, 2 |
| `nan`, `touch`, `stuck` | cumulative diverged worlds, side-obstacle touches, stuck terminations | | |

Gaps: per-episode peak foot force as in Table 1 (the log has the per-step mean of the larger foot force; `paper_eval.py` has
the peak), success rate per terrain (needs `paper_eval.py`), the VAE KL term (computed, not logged).

## Plan for the extended training map (`/mnt/filesystem-g4/data/terrains/extended_v1/`, exported by the Hiking track)

Not used yet; a gait on the paper-style tiles comes first. How SSR takes it without leaving the paper's method (terrain types,
each with a discriminator and a gate expert, 20 levels, tiles walked along x under a heading command, an omnidirectional stage):

| Map content | SSR terrain type and motion prior | Notes |
| --- | --- | --- |
| Wedges, ramps | `ramp` (the paper's slope), ramp captures | cross-slope wedges need lateral commands: some go on the omnidirectional tile |
| Stairs, stairs with turns | `stairs`, stair captures | straight flights as tiles; a turn is a landing between two flights walked under a changing heading target, which the paper's heading command already expresses |
| Straight and slanted 4x4s, log corridors | `beam` (in place of the paper's discrete obstacles), 4x4 / 2x2 crossing captures | corridor walls and rails enter as side obstacles (depth and critic map; touching ends the episode), as now |
| Platforms, kerbs, home and street steps | `platform`, step-up captures | |
| Flat corridors, doors, home and street floors | `flat` lane, flat walking | |
| Dead ends (turn in place), backward and sideways moves | no new type: the omnidirectional stage (base-frame vx, vy, yaw-rate commands, Table 5) on tiles with walls | needs turning and backward/sideways clips in the flat prior; it has 10 % backward frames and few pure turns: to be extended from BONES-SEED |

A new category would be needed only for content with no prior and no analogue among the five (none in the description so far);
the network has five experts and five discriminators, so a sixth type means widening both.

Needed from the export: (1) one height raster per tile at 2.5 cm or finer with x0, y0 and resolution in metres, walkable surface
only; (2) a second "vision" raster or boxes for walls, rails and door frames (centre, half-size, yaw), so depth sees them while
physics stays a height field; (3) per tile: category label, the difficulty parameters to sort tiles into 20 levels, entry and
exit points with headings, lane width, and tile size if not 8 x 4 m; (4) for routes: waypoint lists with headings, used only in
the omnidirectional stage; (5) which tiles copy apparatus parts, so they can be listed here; no whole apparatus layout or route.

## Standing-still diagnosis log (one line per run; `vx` = achieved / commanded speed of robots told to walk 0.3 m/s or more)

| Run | Change from the run before | Iterations | vx curve | Other | Verdict |
| --- | --- | --- | --- | --- | --- |
| r1 | paper settings, joint advantage normalisation, 4,096 robots | 440 | 0.38 (20), 0.00 (100), 0.03 (200), 0.03 (300), 0.03 (440) | double support 0.88, level 0.00, style 0.26 | stands |
| r2 | + per-critic advantage normalisation (style 14 % of the gradient); 8,192 robots from iteration 50 | 360 | 0.38 (20), 0.07 (120), 0.04 (210), 0.04 (290), 0.03 (360) | double support 0.88, style 0.35 | stands |
| r3 | + standing windows removed from the prior, walking starts only from frames at 0.3 m/s or more | 310 | 0.37 (30), 0.05 (90), 0.04 (150), 0.06 (200), 0.07 (270), 0.07 (300) | double support 0.82, single 0.17, style 0.33 | stands (slightly more foot activity) |
| r4 | + style weight 1.0 | 20 | not long enough to read | | stopped at 20 for the coordinator's bisect; to be repeated in step 2 |
| d1 | bisect step 1: flat tile only, only `lin_vel` and `ang_vel` rewards, style weight 0, joint normalisation; domain randomisation, walking starts and terminations unchanged | running | see Status | | |

What the standing policy's reward breakdown shows (r3, iteration 180, weighted per step): lin_vel 0.39 of 1.0, ang_vel 0.32 of
0.8, orientation 0.44 of 0.5, ang_vel_xy 0.09 of 0.25, base_height 0.36 of 0.4, single_support 0.14 of 0.2, action_rate -0.13,
smoothness -0.17, dof_dev -0.17, everything else under 0.03. Two things stand out: (1) the robot gives up 0.6 in velocity
tracking by not walking; (2) the action noise (std 0.73, not shrinking) shakes the robot enough to cost about 0.5 in yaw-rate
tracking, 0.16 in roll/pitch rate and 0.30 in the two smoothness terms: about 1.0 per step is lost to exploration noise, and the
feet chatter (5 touchdowns per second per robot while "standing"). The PPO step size keeps collapsing to 1e-5 with KL at
0.017-0.025.

Plumbing checked by reading the code (2026-11-11): the command is the world-x speed expressed in the base frame
(`cos(yaw) v, -sin(yaw) v`), base velocity is rotated into the base frame before comparison, the tracking kernels are
exp(-err^2 / 0.25) as in Table 7, MuJoCo free-joint angular velocity is already body-frame. PD gains, armature, effort limits,
default pose and action scale (0.25 x effort / kp) are the same vendor plant the Hiking track's X2 task uses (10 Hz natural
frequency, damping ratio 2: hip and knee kp 115, ankle pitch 35, ankle roll 12.4).

## Root-cause investigation of standing still (2026-11-11, 21:50-22:55 UTC)

Bisect runs (all 8,192 robots, vendor plant, head camera, walking starts from moving frames, domain randomisation and
terminations as in r3; "flat" = every robot on the open flat tile, no curriculum):

| Run | Rewards | Iterations | vx curve | Verdict |
| --- | --- | --- | --- | --- |
| d1 | flat; only `lin_vel` + `ang_vel`; style weight 0 | 280 | 0.64 (20), 0.47 (60), 0.20 (120), 0.22 (150), 0.50 (180), 0.69 (210), 0.84 (240), 0.91 (270) | **moves**: 59 % of episodes reach the tile end at 270 |
| d2 | d1 + posture and joint-deviation terms (`orientation`, `ang_vel_xy`, `base_height`, `lateral`, `dof_dev`, `stand_still`) | 240 | 0.44 (30), 0.16 (60), 0.14 (100), 0.18 (140), 0.26 (180), 0.41 (210), 0.55 (240) | **moves**, about 40 iterations later than d1 |
| d3 | d1 with plant `mc_20261024` | queued | | |
| d4 | flat, all Table 6 terms, style weight 0, plant `mc_20261024` | queued | | expected to stand; this is the run that names the penalty group |

What is proven by a run:
- The code can produce locomotion: commands, frames, velocity kernels, PD and action scale work (d1, d2).
- The posture and joint-deviation terms do not cause standing by themselves (d2).
- Not yet proven: that standing returns on flat ground once the remaining penalty terms are added (d4 tests it).

What d1's motion is: **a fast shuffle, not a walk.** Probe on flat ground, 256 robots, 12 s, command 0.6 m/s, no noise
(`eval/d1_probe/gait.json`, clip `eval/d1_probe/d1_it250_flat.mp4`: MuJoCo playback of the recorded MuJoCo Warp states, chase
view and legs close-up): speed 0.68 m/s, 6.8 touchdowns per foot per second, median air time 0.06 s (no air phase reaches
0.1 s), step length 0.10 m, swing clearance 2.2 cm, base height 0.666 m, all 256 survive. I looked at the frames: legs nearly
straight, feet pattering under the body. The yaw-rate error in the training log (1.0 rad/s against a 0.07 command) is the
action noise shaking the body (std 1.35): without noise the same policy has 0.28 rad/s.

**Measured: under the paper's reward table this shuffle earns less than standing.** Same probe, each policy scored with the
full Table 6 (weighted, per step, command 0.6 m/s, no noise; `eval/terms_probe/`):

| Term (weight) | Standing policy (r3, it 300) | Shuffling policy (d1, it 250) | Difference |
| --- | --- | --- | --- |
| lin_vel (1.0) | 0.240 | 0.929 | +0.69 |
| single_support (0.2) | 0.003 | 0.200 | +0.20 |
| base_height (0.4) | 0.351 | 0.392 | +0.04 |
| ang_vel (0.8) | 0.796 | 0.568 | -0.23 |
| orientation (0.5) | 0.498 | 0.363 | -0.13 |
| ang_vel_xy (0.25) | 0.244 | 0.130 | -0.11 |
| action_rate (-0.12) | -0.001 | -0.153 | -0.15 |
| smoothness (-0.06) | 0.000 | -0.152 | -0.15 |
| dof_dev (-1.8) | -0.189 | -0.324 | -0.13 |
| impact (-1.3) | -0.001 | -0.167 | -0.17 |
| slip (-0.2) | 0.000 | -0.134 | -0.13 |
| dof_vel, dof_torque, limits | -0.145 | -0.075 | +0.07 |
| lateral (0.08) | 0.080 | 0.080 | 0 |
| **r_l total** | **1.876** | **1.657** | **-0.22** |

So the cause is in the reward balance, not in a broken code path: with Table 6 as written, a robot that stands under a 0.6 m/s
command already collects 1.88 of about 3 per step, and the first gaits that exploration can reach (a shuffle) collect less,
because what they gain in velocity tracking (+0.69) and single support (+0.20) is taken back by yaw-rate and roll/pitch-rate
tracking, impact, slip, action smoothness and joint deviation (-1.11). Only a clean gait (quiet torso, soft landings, smooth
actions) beats standing, and nothing leads there from a stand. In training the action noise makes this worse: the standing
policy scores 1.40 with its own noise (std 0.72), and the noise is not reduced because the entropy bonus holds it.
Hypothesis, not yet proven: the paper's Isaac Gym setup gets across this gap (its ablation without style reward still walks),
through details the paper does not state (initial state, reward scheduling, noise level, contact model for impact and slip).

Plain bugs checked:
- Command frame and sign, tracking kernel width and units: correct (read; and d1 tracks 0.6 m/s at 0.68 m/s).
- Observation normalisation: the actor and critics use running normalisation; encoder inputs are clamped; d1/d2 learn with it.
- Action scale, PD gains, armature, default pose: read back from the live model (table below); the same vendor plant as the
  Hiking track's X2 task. The one hand-set value was waist pitch/roll at 120/6.
- Learning-rate collapse (1e-5 to 1e-3, KL 0.01-0.025): present in every run including d1 and d2, which learn; not the cause.
- Discriminator saturation: no; a standing robot's style reward is 0.26-0.35, not 0, and the discriminator output is -0.7.
- Mirror maps: `tests/test_mirror.py` passes on the node (kinematic check, with three cases that must fail).
- Standing-command share: the paper's U(0, 1.2) m/s gives 12.5 % of walkers a command under the 0.15 threshold. In r1/r2 the
  walking starts overrode the speed with the frame's (median 0.09 m/s), so about a third of walkers were commanded to stand:
  a real defect, fixed in r3 (`--rsi_min_speed 0.3`), not sufficient alone.
- Standing frames in the prior: 15-24 % (fixed in r3 with `--prior_min_speed 0.2`), not sufficient alone.
- Odd and unexplained: the standing policy without noise holds about 2 of 21 joints beyond their position limit
  (`dof_pos_lim` -0.139). To look at.

On the coordinator's question of what the two standing X2 tracks share: d1 ruled out the robot model, gains and action scale
as blockers of motion (the X2 moves at commanded speed in this simulator with them), and d1 kept the same motion data starts.
It points at reward balance against a too-good standing score; the shared motion data's standing share is a second, smaller
contributor (measured here). Whether the softer vendor gains make a clean step harder is what d3 tests.

Fix and plan (within the paper's method):
1. d3 (plant only) and d4 (all Table 6 terms on flat, no style) run now, 300 iterations each. d4 pass/fail: if vx is under 0.2
   at 300 the penalty group is confirmed as what holds the robot at a stand.
2. r5 (queued after them, runs unattended): the full method with the motion prior's weight raised to 1.0 under per-critic
   normalisation (style 44 % of the gradient), plant `mc_20261024`, camera `chest_rs`. The motion prior is the paper's own
   mechanism for shaping a clean gait, and a gait close to the data is smooth and soft-landing, so it also scores on Table 6.
   Pass: vx at least 0.3 and single support at least 0.3 with air times over 0.1 s by iteration 500, mean level rising by
   1,000. Stop at 500 otherwise.
3. If r5 fails: start from a moving policy instead of from a stand, still by RL and still without SuperMimic: train d2-style
   (tracking + posture) for 250 iterations, then switch on the remaining Table 6 terms and the style reward over the next
   500 iterations (`--only_terms` with a schedule; needs a small code change). This is a reward schedule the paper does not
   describe, so it would be listed as a deviation; the final rewards are the paper's.
4. Question for the Hiking track (via the coordinator): which reward terms pay its standing X2 the most per step, and does its
   G1 run go through a shuffle phase before walking?

## What the simulator applies (read back from the live MuJoCo Warp model; `eval/model_*.json`)

Physics 200 Hz, policy 50 Hz, total mass in the json. Armature, effort limit, action scale (0.25 x effort / armature kp) and
default pose are the same in both plants.

| Joint (both sides) | armature | effort N.m | action scale | default rad | kp / kd vendor_20260823 (r1-r4, d1, d2) | kp / kd mc_20261024 (d3 on) |
| --- | --- | --- | --- | --- | --- | --- |
| hip pitch | 0.02908 | 120 | 0.2613 | -0.312 | 114.8 / 7.31 | 100 / 4 |
| hip roll, hip yaw | 0.02908 | 120 | 0.2613 | 0 | 114.8 / 7.31 | 100 / 3 |
| knee | 0.02908 | 120 | 0.2613 | 0.669 | 114.8 / 7.31 | 150 / 5 |
| ankle pitch | 0.00884 | 36 | 0.2579 | -0.363 | 34.9 / 2.22 | 40 / 3 |
| ankle roll | 0.003133 | 24 | 0.4851 | 0 | 12.37 / 0.79 | 30 / 2 |
| waist yaw | 0.02908 | 120 | 0.2613 | 0 | 114.8 / 7.31 | 40 / 8 |
| waist pitch, roll (not policy joints, held) | 0.003133 | 24 | 0.4851 | 0 | 120 / 6 (hand-set) | 40 / 5 |
| shoulder pitch | 0.00884 | 36 | 0.2579 | 0.2 | 34.9 / 2.22 | 30 / 1 |
| shoulder roll | 0.00884 | 36 | 0.2579 | +-0.2 | 34.9 / 2.22 | 20 / 1 |
| shoulder yaw | 0.003133 | 24 | 0.4851 | 0 | 12.37 / 0.79 | 20 / 1 |
| elbow | 0.003133 | 24 | 0.4851 | -0.6 | 12.37 / 0.79 | 50 / 1 |

The MJCF and the vendor plant file in the code copy are byte-identical to `/mnt/filesystem-g4/data/robots/x2/`. The plant is
chosen with `--plant` (default `mc_20261024`, user decision 2026-11-11); the hand-set waist gains apply only to plants without
a stiffness table. Training randomises kp and kd by x0.9-1.1 and motor strength by x0.8-1.2 around these (Table 10).

## Cameras

User decision 2026-11-11 (final): a RealSense D435-class depth camera on the chest, centred, pitched down, as in the reference
frame `docs/camera/chest_realsense_depth_reference.jpg`; lens height "somewhere in between" 0.92 and 1.05 m. Option
`--camera chest_rs`, used from run r5 on. `head`, `chest` (the old 75 deg option) and `torso60` remain selectable.

**Pose and optics of `chest_rs` as r5 uses it** (`ssr/x2.py`, `CAMERAS`): parent body `torso_link`, position
(0.126, 0, 0.16) m in that body (x forward, y left, z up): 3 cm in front of the chest shell (shell front at x = 0.096 m,
Hiking track's mesh measurement), lens 0.98 m above the floor at the upright reset pose; rotated about the body's y axis so
the optical axis is 59.5 deg below horizontal when the torso is upright; field of view 87 deg horizontal x 58 deg vertical
(the D435 depth field), sampled on a regular 36 x 36 tangent-plane grid (pixels wider than tall), no crop; depth = distance
along the optical axis, clamped to 0.1-3.0 m and scaled to 0-1; limbs rendered as spheres; N(0, 0.1 m) noise on self-occluded
pixels. **Mount randomisation, fixed per episode** (user, 2026-11-11): forward +-2 cm, lateral +-1 cm, height +-5 cm,
pitch +-4 deg, roll and yaw +-2 deg; field of view +-1 deg. (Other cameras keep the paper's +-1 cm, +-1 deg.)

Acceptance check (base upright at the reset pose, flat ground, rows computed from the real rays; frame with the lines drawn:
`agentic/state/ssr_chest_rs_camera_check.png`, also `renders/camera_check/` on the shared filesystem): 0.55 m line at 51.5 % of the
image height from the top (reference about 52 %), 1.0 m line at 25.4 % (reference 27 % +-3). **The toe line falls 3 % below
the bottom edge** (ground is seen from 0.03 m ahead of the toe tips): with the lens on the shell surface it sits directly above
the toes, and a tilt that brought the toe line inside (61 deg) would push the 1.0 m line to 23 %. The toes enter the image
when a foot swings forward (legs cover 1 % of the image standing, 5 % mid-stride).

All options, same pose, plant `mc_20261024`, distances from the front edge of the soles (`eval/model_mc_20261024*.json`;
frames for flat ground, a 4x4 at 1.5 m and at 0.3 m, a stair foot at 0.3 m, and a kinematic mid-stride pose, as upscaled PNG
and raw 36 x 36 arrays, in `renders/camera_check/`; computed on the CPU so training is not interrupted):

| Option | Mount (in parent body) | Axis below horizontal | Field of view | Range | Lens height | Ground seen | Width at 0.5 m | Legs in image (standing / mid-stride) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `chest_rs` (r5 on) | `torso_link` (0.126, 0, 0.16) | 59.5 deg | 87 x 58 deg | 0.1-3.0 m | 0.98 m | 0.03 to 1.61 m | 2.08 m | 1.1 % / 5.1 % |
| `head` (r1-r4, d1-d4) | `head_pitch_link` (0.0576, 0, -0.0484) | 40 deg | 54.9 deg square | 0.1-2.0 m | 1.17 m | 0.44 to 1.55 m | 1.23 m | 3.2 % |
| `chest` | `torso_link` (0.155, 0, 0.20) | 75 deg | 54.9 deg square | 0.1-2.0 m | 1.02 m | -0.19 to 0.94 m | 1.13 m | 14.7 % / 13.0 % |
| `torso60` | `torso_link` (0.0488, 0, 0.4378) | 60 deg | 64 deg square | 0.1-3.0 m | 1.26 m | -0.11 to 2.21 m | 1.71 m | 6.3 % / 5.4 % |

Notes: (1) `chest_rs` reaches 1.61 m ahead, less than the 2 m the user first asked for; the reference frame's landmarks fix
that. (2) An earlier solve of mine, (-0.026, 0, 0.125) at 53.5 deg, met all three landmarks but lay 12 cm inside the chest
shell; it was never used in a run. (3) At 36 x 36 a 4x4 at 1.5 m is one to two pixel rows; four rows would need about
72 rows of vertical resolution (4x the pixels: the CNN's first layer and the ray march cost about 4x; to be measured).
(4) The paper's waist camera pose and field of view are not in the paper, so its footprint cannot be computed; a head-to-head
run (same seed, 1,000 iterations each) is planned once the robot walks.

## Checks made before r5 (2026-11-11, 23:07-23:20 UTC, all on the CPU)

**The prior's speed against the commands.** Commands are U(0, 1.2) m/s forward (the paper's Table 4); among walkers above
the 0.15 threshold the median is 0.68 m/s, 38 % are above 0.8 and 19 % above 1.0. Window mean planar speed of the reference
motion as fed to each discriminator (5-frame windows):

| Set | r3 / r4: `amp.npz`, moving windows only | r5: `amp_v2.npz`, moving windows, speed-balanced sampling |
| --- | --- | --- |
| flat | 228 s; p10 0.26, p50 0.56, p90 0.84; 13.5 % above 0.8 | 656 s; p10 0.28, p50 0.57, p90 0.88; 18.7 % above 0.8 |
| ramp | 357 s; p50 0.40, p90 0.75; 7.6 % above 0.8 | p50 0.48, p90 0.85; 12.7 % |
| stairs | 771 s; p50 0.41, p90 0.63; 2.4 % | p50 0.45, p90 0.72; 4.7 % |
| beam | 387 s; p50 0.46, p90 0.71; 5.2 % | p50 0.50, p90 0.78; 8.4 % |
| platform | 510 s; p50 0.44, p90 0.68; 3.4 % | p50 0.48, p90 0.75; 5.9 % |

- Before any filter the flat set's median forward speed was 0.09 m/s (43 % of its frames under 0.2 m/s).
- `amp_v2.npz` (`scripts/build_flat_v2.py`): the flat set rebuilt from all 107 selected flat clips (999 s; the first build
  stopped at 406 s of randomly chosen clips), other sets unchanged. `--prior_balance` samples reference windows with equal
  mass per 0.2 m/s speed bin, no window weighted more than 5x uniform.
- What cannot be fixed by selection: the source has little fast walking. Of 999 s of flat clips, 68 s are at 0.8-1.0 m/s and
  25 s above 1.0. The Hiking track's cleaned file (same sources) has median 0.53 m/s, so swapping to it would not change
  this. The prior therefore covers about 0.25-0.9 m/s and the commands reach 1.2: for the top third of commands the style
  reward and the velocity reward pull apart. Kept the paper's command range in r5 (one change at a time); if r5 walks but
  tracks fast commands poorly (`lin_err` by speed), cap the command at 1.0 m/s and list it as a deviation.

**The "lean".** Corrected: it is not a lean. CPU MuJoCo, flat floor, the training model build, both plants
(`scripts/lean_check.py`, `eval/lean_check_*.json`):
- At the default pose the centre of mass is 3.4 cm ahead of the ankles and 0.4 cm behind the centre of the sole patch
  (+-11.25 cm): statically well placed, pelvis and torso upright, no joint within 0.02 rad of a limit.
- With zero actions (PD targets = default pose) the robot rocks back 0.2 deg (mc) to 1.8 deg (vendor) in the first 0.5 s and
  then **falls forward within about 1.5 s**, under both plants; the Hiking track sees the same in MuJoCo and Isaac. My
  "settles 7.6 deg back" was one frame of that motion read as a resting pose.
- Why: the ankle pitch stiffness (2 x 40 N.m/rad with mc, 2 x 35 with vendor) is below the body's toppling stiffness
  (m g h, about 35 kg x 9.81 x 0.65 m = 223 N.m/rad), so no fixed target holds the robot; the ankles carried 10 N.m each and
  were 21 deg off target when it went over. Adding the gravity-sag offset to the targets does not help (it still falls,
  iterated three times). Standing on the X2 with these gains is an active balancing task for the policy.
- Consequence for the standing problem: it does not make the first step more costly as such, but it means "stand safely" is
  itself the first skill PPO has to find, and every early gait attempt is made by a robot that falls in about a second
  without feedback. It fits what the runs show (episodes of 1.2 s for the first 60-100 iterations, then standing).

**The two joints past their limits** (CPU probe of the r3 standing policy): both shoulder-roll joints sit on their inner
limit (+-0.061 rad, 0.26 rad from the default +-0.2), past it 94 % of the time: the policy holds the arms pressed in against
the body. No bug in the limit check; it costs the stander 0.14 per step in `dof_pos_lim` and some `dof_dev`. Not understood
why it prefers that; to watch in r5.

**d3 (plant only changed from d1):** vx 0.56 (0), 0.45 (60), 0.18 (100), 0.21 (150), 0.43 (180); d1 at the same iterations:
0.55, 0.47, about 0.2, 0.22, 0.50. The measured gains do not change how motion starts; the vendor-rule gains were not part of
the standing problem. (Final numbers at iteration 300 are in `runs/d3/log.jsonl`.)

**Fallback order if r5 is under vx 0.2 at iteration 500** (coordinator, 2026-11-11): no more runs on weights alone.
1. Staged reward schedule, already in the code (`--stage_a 250 --stage_b 750`): until iteration 250 only the tracking and
   posture terms (`lin_vel, ang_vel, orientation, ang_vel_xy, base_height, lateral, dof_dev, stand_still`, the set d2 moved
   with) and no style reward; from 250 to 750 the remaining Table 6 terms and the style weight ramp linearly to full. A
   deviation from the paper (which describes no schedule); the final rewards are the paper's. Logged as `stage_ramp`.
2. With it, a lower entropy coefficient (`--ent 0.002`; paper 0.005): the action noise alone costs the stander about 0.5 per
   step and the shuffler more.

## Results 23:25-00:20 UTC (d3, d4, discriminator scores, evaluation beside training)

| Run | Setting | Iterations | vx curve | Verdict |
| --- | --- | --- | --- | --- |
| d3 | d1 with plant `mc_20261024` | 290 | 0.45 (60), 0.18 (100), 0.21 (150), 0.43 (180), 0.59 (220), 0.71 (280), 0.73 (290) | moves like d1 (d1: 0.91 at 270): the gains are not the cause |
| d4 | flat, **all Table 6 terms**, no style, `mc_20261024` | 290 | 0.37 (50), 0.09 (80), 0.03 (110), 0.15 (240), 0.18 (270), 0.21 (290) | **the penalties bring standing back** on flat ground with no terrain, obstacles or style reward (double support 0.86 at 110); from 200 on it creeps: 0.12 m/s in 3 cm steps |
| r5 | full method, style weight 1.0, per-critic normalisation, rebuilt flat prior, `mc_20261024`, `chest_rs` | running | 0.52 (10), 0.35 (40), 0.20 (60), 0.05 (80), 0.03 (110-140) | standing by iteration 80 (double support 0.85) |

**What d4 ends as** (probe, 256 robots, 30 s, command 0.6 m/s, no noise, plant mc): 0.115 m/s, 3.8 touchdowns per foot per
second, median air time 0.04 s, step length 3 cm, clearance 0.5 cm: a stand with foot taps. Table 6 score **2.36 per step**:
lin_vel 0.39, ang_vel 0.76, orientation 0.48, ang_vel_xy 0.25, base_height 0.36, single_support 0.198 of 0.2, dof_dev -0.10,
everything else under 0.02. The taps collect the whole single-support reward (it only asks that one foot be unloaded at some
moment in the last 0.2 s) at no cost. The d3 shuffler (0.53 m/s, 10.9 touchdowns per foot per second) scores **1.00** on the
same table (action_rate -0.40, smoothness -0.44, impact -0.22, slip -0.11, dof_dev -0.29). So under the paper's table as
implemented here the ranking is: tap-in-place 2.36 > plain stand 1.88 > shuffle 1.0-1.66; only a clean walk (about 2.7-2.9
if everything is near its maximum) is above the tapping stand, and the gap to it is small.

**Does the style reward separate standing from walking?** Barely, and it prefers standing to the shuffle
(`scripts/disc_score.py`, CPU; `eval/disc_score_r3.json`, `_r2.json`; flat discriminator, style reward 0-1):

| Windows | r3's discriminator (prior without standing, it 300) | r2's (prior with standing, it 350) |
| --- | --- | --- |
| reference, 0.4 m/s or more | 0.93 | 0.95 |
| reference, 0.2-0.4 m/s | 0.88 | 0.92 |
| reference, standing (under 0.05 m/s) | 0.81 | 0.90 |
| standing policy, no action noise | 0.62 | 0.75 |
| standing policy, with its training noise | 0.34 | 0.36 |
| shuffle policy (d1), no noise | 0.59 | 0.47 |
| shuffle policy (d1), with noise | 0.18 | 0.01 |

A standing reference window earns 87 % of a walking one even after standing was removed from the prior (95 % before), and a
standing policy earns more than a shuffling one; what the discriminator punishes most is the action-noise jitter. So raising
the style weight cannot pull the robot off the stand; r5 confirms it. Same picture as the Hiking track's.

**Evaluation beside training** (user, 2026-11-11; measured on r5, 8,192 robots, 33.8 GB, baseline 30.3k-31.7k samples/s):

| Eval job | Wall time | Training samples/s while it ran | Loss | Peak GPU memory | Trainer warnings |
| --- | --- | --- | --- | --- | --- |
| gait probe, 256 robots, 30 s of simulation | 165 s (about 60 s of it start-up) | 24.4k-25.4k | 18 % | 35.6 GB (+1.8) | none |
| gait probe, 2,048 robots, 30 s | 210 s | 21.0k-22.0k | 31 % | 40.1 GB (+6.2) | none |

Training returned to 30-31.7k within one log interval after each. Rule for this track: evals beside training use at most
256 robots (about 512 would be the limit for 20 %), through `scripts/eval_guard.sh <tag> <seconds> <command>`: one at a time
(lock file), `nice`, wall-clock limit, refuses to start with under 10 GB free or over 70 GB used, kills the eval if free
memory drops under 10 GB; it logs start, end, duration and peak memory to `runs/eval_guard.log`. A 1,000-trial paper
evaluation would have to run as four 256-robot batches or wait for a restart.

Throughput note: `chest_rs` (87 x 58 deg, 3 m, 38 march steps) costs 9 % against the head camera (30.3k vs 33.3k samples/s).

**Next: recipe tests** (`runs/queue4.sh`, to start when r5 is stopped at its gate; one at a time; all on the flat tile with
style weight 0, plant mc, camera chest_rs; each followed by a gait probe written to `eval/<name>/gait.json`). Pass for each:
air time over 0.2 s and step length over 0.25 m at 0.6 m/s with the full Table 6 on at the end.

| Test | Flags | Iterations | Result |
| --- | --- | --- | --- |
| t_a staged schedule | `--stage_a 150 --stage_b 400` (tracking + posture only until 150, the rest ramped in by 400) | 550 (00:53-01:54 UTC) | **moves under the full table** (first SSR run to do so). Share of commanded speed: 5 % (80, the d2-like dip), 10 % (150), 20 % (200), 44 % (300), 60 % (400, ramp 1.0), 79 % (540); touchdowns/s (both feet) 11 -> 8.5, single support 0.38 -> 0.60, tilt ends 1 %, r_l 1.97. Probe at 549, 0.6 m/s, no noise: 0.50 m/s, single 0.70, 3.7 touchdowns/foot/s, air time median 0.10 s, step length 0.14 m, clearance 1.9 cm, Table 6 score 2.77/step (stander 1.88, tapper 2.36); with noise 0.49 m/s, air 0.08 s, step 0.12 m. Not yet the 0.2 s / 0.25 m swing; improving. Led to r6 |
| t_c, t_d, t_e, t_f | | | **not run**: coordinator (01:55) chose r6 once t_a moved; the code for t_e/t_f stays (`--prog_gate`, `--style_low`) |
| t_b command curriculum | `--cmd_cur_iters 200 --cmd_min0 0.5 --sig_v0 0.06` (no command under 0.5 m/s and a narrow tracking kernel, so zero speed pays nothing, until 200; released by 400) | 500 (ended by a STOP file at 190, 00:52 UTC) | **fail, stands**: vx 0.007 of 0.76 m/s (1-3% from 100 on), double support 0.89, tracking term 0.03/step; probe at 150 (0.6 m/s): 0.005 m/s, air 0.02 s, step 0.002 m. The stander still gets about 1.4/step from orientation 0.44, yaw kernel 0.41, base height 0.37, roll/pitch rate 0.12, lateral 0.08, single support 0.12 (taps); tilt 42% of episode ends |
| t_e progress gate (coordinator, 01:00 UTC; deviation) | `--prog_gate 1.0 --gate_air 0.2 --gate_ramp 300`: orientation, yaw kernel, base height, roll/pitch rate, lateral x (1 - g(1 - progress)), progress = clip(0.5 s mean velocity along the command / commanded speed, 0, 1), only under a moving command (> 0.15 m/s); gate ramps off over 300 iterations once mean touchdown air time > 0.2 s for 5 iterations (`runs/queue5.sh`, tmux `ssrq`, after t_a) | 650 | pending |
| t_c no-progress termination | `--stuck_term --stuck_until 300` | 450 | pending |
| t_d less noise | `--ent 0.001 --init_std 0.5` | 300 | pending |

Housekeeping rule (2026-11-11): nothing under `/mnt/filesystem-g4` is deleted any more; unwanted files go to
`tracks/ssr/_trash/<date>/` with a note. (Before the rule I deleted `scripts/bc.py`, the two Windows supervisors, the
`runs/smoke` test folder and superseded camera-check frames.)

## Status (2026-11-12 00:35 UTC): what runs, what is queued, how to continue

**No SSR policy walks yet.** Five full-method runs (r1-r5) stand still; the cause is measured (see "Root-cause
investigation" and "Results 23:25-00:20"). Diagnostic runs with reduced rewards move, as a shuffle.

- **r5 stopped by me at iteration 270 (00:30 UTC), before its iteration-500 gate.** Reason: vx 0.03-0.09 from iteration 80 on
  (0.086 at 270, double support 0.73, level 0.01, the same slow tap-and-creep d4 showed), the offline discriminator scores
  show the style reward prefers a stand to a shuffle, and the Hiking track relaunches at 01:50 UTC with whichever recipe test
  shows real swing, so the GPU went to the tests. r5 is not lost: `runs/r5/last.pt` (iteration 250) resumes with its command
  (below) if anyone wants the gate.
- **Running now: `runs/queue4.sh` in tmux session `ssr`**, the recipe tests, one at a time, each under the supervisor
  (heartbeat names the test) and each followed by a gait probe written to `eval/<name>/gait.json` and `gait_noise.json`:
  `t_b` (started 00:31 UTC, 500 iterations, about 55 min), then `t_a`, `t_c`, `t_d` (table in "Results"). After `t_d` the
  GPU is idle: the next step is a full run with the recipe that passed.
- Stop: `touch /mnt/filesystem-g4/tracks/ssr/runs/QUEUE4_STOP /mnt/filesystem-g4/tracks/ssr/runs/STOP`.
- r5's command, to resume: `cd /mnt/filesystem-g4/tracks/ssr/code && ../env/venv/bin/python scripts/supervise_node.py --name r5 --iters 20000 --num_envs 8192 -- --adv_norm separate --prior_min_speed 0.2 --rsi_min_speed 0.3 --w_style 1.0 --plant mc_20261024 --camera chest_rs --amp_file data/amp_v2.npz --prior_balance`
- Known artefact in d4 and t_b: one enormous slip penalty at the very first step of iteration 0 (the flat-tile reset did not
  refresh the stored foot positions); the value loss recovers by iteration 10. Fixed in the code for t_a onwards.
- Zero-command share and what a stander is paid (checked for the coordinator): 12.5 % of non-walking-start walkers draw a
  command under the 0.15 m/s threshold (about 6 % of all walkers, since half start from walking frames at 0.3 m/s or more),
  and about 1 % of the omnidirectional robots; no resampling within an episode. Under a moving command the velocity kernel
  exp(-v^2 / 0.25) still pays a stander 0.70 at 0.3 m/s, 0.24 at 0.6, 0.02 at 1.0 (mean 0.37 over the command range), and
  the yaw kernel pays its full 0.8 because the heading starts aligned. t_b removes both early on.

Watch: `python3 /mnt/filesystem-g4/tracks/ssr/code/scripts/d.py <run> 5 5` (iteration, vx, single / double support,
touchdowns per second, rewards, episode length, level, std, step size, shares of episode ends).
Gait probe of a checkpoint (GPU, only between jobs): `../env/venv/bin/python scripts/probe_gait.py --ckpt ../runs/<run>/last.pt --out ../eval/<name> --speed 0.6`
(set `X2_PLANT` to the run's plant). Camera and gains read-back (CPU, any time):
`CUDA_VISIBLE_DEVICES= REPORT_DEVICE=cpu X2_PLANT=mc_20261024 ../env/venv/bin/python scripts/report_model.py ../eval/model_mc_20261024.json ../renders/camera_check`.

Newest checkpoints: `runs/d3/last.pt` (diagnostic), `runs/d1/last.pt` (iteration 250, the shuffle), `runs/r3/last.pt`
(iteration 300, standing). None is a candidate for the common eval.

Code changes since the copy (all in `models/ssr/code`): `ssr/train.py` (flags `--rew_scale --init_std --adv_norm --adv_floor
--w_style --rsi_frac --rsi_min_speed --prior_min_speed --boot_iters --stuck_term --flat_only --only_terms --fixed_lr --plant
--camera chest_rs|torso60 --out_root --ckpt_every`; full log; `last`/`prev` checkpoints; robot-count change on resume),
`ssr/env.py` (guidance gate, occlusion noise, gait and termination counters, term mask, flat-only, per-camera optics),
`ssr/x2.py` (cameras `chest_rs`, `torso60`; waist override only for armature-rule plants), `ssr/terrain.py` (separate
horizontal and vertical field of view), `config/robot_plant/x2_ultra_mc_20261024.yaml` (copied from `data/robots/x2`),
new scripts `supervise_node.py`, `probe_gait.py`, `report_model.py`, `d.py`, `summ.py`; `scripts/bc.py` and the Windows
supervisors removed. `scripts/_patch2.py` and `scripts/patch3.py` are the applied patch scripts, kept as a record.

Not verified / open:
- d3 and d4 results; whether r5's larger motion-prior share starts a gait.
- `paper_eval.py`, `route_eval.py`, the ONNX export and the eval adapter card have not been run on the node (no walking policy).
  `tracks/ssr/eval/` holds only my probes; the adapter card and `obs_check.npz` the eval monitor asks for are not written.
- The depth ray march at 3 m range and 87 x 58 deg has only been checked on frames, not in a training run (r5 is the first);
  its extra march steps (38 instead of 26) will slow the environment step somewhat: re-measure samples per second in r5's log.
- The Warp-kernel ray march (the route to better GPU use) is not started.
- `chest_rs` mount point is an estimate until the Hiking track's value arrives.
- The standing policy's two joints at their position limits; the 7.6 deg backward settle under zero actions with `mc_20261024`.
- A short GPU overlap happened at 22:56 UTC: a 2-robot camera read-back (about 1 minute) ran while d3 trained, because the
  replacement queue did not see a supervisor to wait for. No effect on d3's log; later read-backs run on the CPU.

## Resumed by a new SSR agent (2026-11-12 00:53 UTC): items from the coordinator's list

- **t_b** ended (STOP file at 190, not mine): fails, stands (table above). Reported 00:56.
- **t_e added** (coordinator, ~01:00): code `scripts/patch_te.py` (applied; backups of the pre-patch files in
  `tracks/ssr/_bak_20261112/`), flags `--prog_gate --gate_air --gate_ramp`, default off so all earlier commands behave as
  before. New log fields: `air_td` (mean air time of touchdowns under a moving command), `air_long_share` (share over 0.2 s),
  `prog_mean` (mean progress under a moving command), `prog_gate`, `gate_start`. CPU smoke test (8 robots, 2 iterations)
  passed. Queue: `runs/queue5.sh` in tmux `ssrq` waits for `queue4.sh` to exit (QUEUE4_STOP set, so queue4 ends after t_a and
  its probe), then t_e (650 its), t_c, t_d. Stop: `touch runs/QUEUE5_STOP runs/STOP`.
- **Zero-command share and stander pay** (item 2, measured earlier, still valid): 12.5 % of non-walking-start walkers draw a
  command under 0.15 m/s (about 6 % of walkers, about 1 % of omnidirectional robots); no resampling within an episode. Under a
  moving command the linear kernel (std^2 0.25) pays a stander 0.70 at 0.3 m/s, 0.24 at 0.6, 0.02 at 1.0 (mean 0.37); the
  yaw kernel pays its full 0.8 (heading starts aligned; Hiking's "yaw earned by turning" has no SSR counterpart); posture terms
  (orientation, base height, roll/pitch rate, lateral) add about 1.0 more. Under Hiking's measure (25 % of robots paid in full
  for standing) SSR is about 6 %, so zero commands are not the SSR cause; the posture/alive terms are.
- **Flat prior from Hiking's cleaned file** (item 3; `scripts/build_flat_v3.py`, `data/amp_v3.npz`, numbers in
  `eval/flat_prior_v3.json`): 228 seams found (joint jump > 0.25 rad/frame, pelvis height jump > 2 cm, pelvis acceleration
  > 40 m/s^2), 83 pieces kept (261 s), 130 pieces under 1 s and 16 stair/height pieces left out. 5-frame window speed:

  | Set | seconds | all windows p10/p50/p90 | under 0.2 m/s | moving windows p10/p50/p90 | moving over 0.8 / 1.0 |
  | --- | --- | --- | --- | --- | --- |
  | v2 flat (r5, before the trainer's 0.2 m/s filter) | 999 | 0.02 / 0.41 / 0.79 | 33.7 % | 0.29 / 0.57 / 0.84 | 13.8 % / 3.7 % |
  | v3 flat (Hiking file) | 261 | 0.31 / 0.54 / 0.78 | 6.1 % | 0.35 / 0.56 / 0.79 | 9.2 % / 2.1 % |

  As fed to the discriminator (moving windows only, speed-balanced) the current prior is already at median 0.57 m/s; the
  "median 0.09" was the raw set before filtering. The Hiking file is not faster (it leaves out jog and run clips) and is a
  quarter of the size, so **amp_v2 stays**; amp_v3 is kept for a head-to-head if wanted. Commands: walkers' median 0.68 m/s.
- **Camera** (item 4): already as asked since r5 (`chest_rs`: torso_link (0.126, 0, 0.16), 3 cm in front of the shell, lens
  0.98 m, 59.5 deg, 1.0 m line at 25.4 % from the top, toe line 3 % below the bottom edge; per-episode mount randomisation
  forward +-2 cm, height +-5 cm, pitch +-4 deg, roll and yaw +-2 deg). All recipe tests use it.
- **t_f added** (coordinator, ~01:10, after Hiking found its style reward acts as an alive bonus): `scripts/patch_tf.py`
  (applied after patch_te; flag `--style_low`, default off; log field `w_style_now`). t_f = t_e plus style weight held at 0
  until the air-time latch, then ramped to the paper's 0.2 over 300 iterations; runs only if t_e reached the latch.
  Per-step pay with style (sent to the coordinator 01:05): plain stander 2.20 (paper w_s 0.2) / 2.48 (r5's 1.0), tapping
  stander 2.68 / 2.96, shuffle 1.25-1.95 / 1.26-2.09, clean walker (estimate) 3.1-3.3 / 3.9-4.1; style is 3 % (12 %) of a
  stander's pay, so in SSR the always-positive Table 6 posture terms (about 1.4) and foothold (0.25) are the alive bonus.

## Status (2026-11-12 01:05 UTC, new SSR agent)

- Running: t_a in tmux `ssr` (`runs/queue4.sh`; QUEUE4_STOP is set, so queue4 ends after t_a's probe, about 01:56 UTC).
- Queued: `runs/queue5.sh` in tmux `ssrq`: t_e (650), t_f (650, only if t_e latched), t_c (450), t_d (300), each followed by a
  gait probe in `eval/<name>/`. Exact commands: `models/ssr/code/run_commands/queue4.sh`, `queue5.sh`.
- Stop everything: `touch /mnt/filesystem-g4/tracks/ssr/runs/QUEUE5_STOP /mnt/filesystem-g4/tracks/ssr/runs/STOP`.
- Watch: `python3 /mnt/filesystem-g4/tracks/ssr/code/scripts/w.py <run> 10` (speed ratio, support shares, touchdown air
  time, progress, gate, stage ramp, reward, episode length, tilt share, action std).
- **Motion files** (coordinator, user rule "Motion data and the repo", 01:15): `models/ssr/code/MOTIONS.md` lists every
  source and built prior (path, bytes, sha256, clips and seconds per set, origin, build command, filters, licence line);
  full numbers in `eval/motion_inventory.json` (`scripts/motion_inventory.py`). The prior files moved out of `code/` to
  `/mnt/filesystem-g4/tracks/ssr/data/priors/` (01:12 UTC); `code/data/amp*.npz` are symlinks to them, so every command with
  `--amp_file data/amp_v2.npz` still works. To restore: copy the files back over the links.

## Status (2026-11-12 02:00 UTC): r6 running

- **r6 training since 01:59:46 UTC** (tmux `ssr6`, `runs/queue6.sh`, supervisor, heartbeat run "r6"). Command (from
  `models/ssr/code`): `../env/venv/bin/python scripts/supervise_node.py --name r6 --iters 20000 --num_envs 8192 -- --stage_a 150
  --stage_b 400 --w_style 0.2 --style_low 0 --gate_air 0.15 --gate_ramp 500 --prior_min_speed 0.2 --rsi_min_speed 0.3 --plant
  mc_20261024 --camera chest_rs --amp_file data/amp_v2.npz --prior_balance`. Full method with the terrain curriculum, from
  scratch. Listed departures from the paper: (1) staged schedule (tracking + posture only until 150, other Table 6 terms ramped
  linearly to full by 400); (2) style weight 0 until mean touchdown air time > 0.15 s for 5 iterations, then ramped to the
  paper's 0.2 over 500 iterations (`gate_start`, `w_style_now` in the log); plus the earlier ones (walking starts, prior speed
  filter and balancing, rew_scale, per-episode camera mount randomisation). Advantage normalisation: joint (as t_a), not r5's
  separate.
- Gates (coordinator): 40 % of commanded speed by 1,000; air time over 0.2 s by 2,000. If cadence stays at about 5
  touchdowns per foot per second at 1,500, add a feet-air-time reward as a listed departure.
- To resume after a crash the supervisor does it itself; by hand: the same command (it resumes from `runs/r6/last.pt`).
- Stop: `touch /mnt/filesystem-g4/tracks/ssr/runs/STOP`. Leftover flags `QUEUE4_STOP`, `QUEUE5_STOP` are harmless.
- Lesson: a queue that waits with `pgrep -f runs/queueN.sh` never ends if the tmux server itself was started with that
  command line (the first `tmux new-session` names the server process). Wait on a lock file or a PID instead.
- Newest checkpoints: `runs/t_a/last.pt` (iteration 549, moving gait on flat, style 0; not for the common eval yet).

## 02:46-02:55 UTC: r6 stands; cause and switch

- r6 at 420: 5 % of commanded speed, double support 0.86 (t_a: 44 % at 300). Args diff with t_a: only `flat_only` acts
  (style weight stays 0 until the latch, which never fired; `prior_balance` only affects the discriminator). Equal-iteration
  logs: both 10 % at 150 (the posture-phase dip); t_a 35 % / r6 8 % at 250; 60 % / 2.5 % at 400. r6's levels start drawn 0-3,
  obstacle-tile failures (side_obstacle 12 % of ends at 150, tilt 25 % at 250) demote everything to level 0 by 250 and the dip
  never recovers. **Cause: terrain from iteration 0, not a wrong setting.**
- New flag `--flat_until N` (`scripts/patch_flat_until.py`, plus a fix noted at its end; backup `_bak_20261112/train.py.pre_flat_until`):
  flat tile until N, then types redrawn uniformly, all levels 0, every robot reset, curriculum on; a resume after the switch
  keeps its levels. CPU smoke test from `t_a/last.pt` passed (switch at 550, levels all 0).
- `runs/queue7.sh` (tmux `ssr7`): at r6 iteration 600, if the mean share of the last three logs is under 15 %, STOP r6 and start
  **r6b** = copy of `t_a/last.pt` (iteration 549) continued with r6's flags plus `--flat_until 550` (supervisor name r6b,
  20,000 iterations). Otherwise r6 continues. Gates stay in iteration numbers (40 % by 1,000, air over 0.2 s by 2,000).
- **Coordinator 02:53: do not wait for 600.** Watcher `ssr7` cancelled; r6 stopped through its supervisor at iteration 490
  (02:53:52; files kept in `runs/r6/`). New: obstacle-tile share ramp (`scripts/patch_obst.py`, flags `--obst0 --obst_ramp`,
  log fields `flat_phase`, `obst_share`, `obst_target`; backups `_bak_20261112/*.pre_obst`): at the switch 25 % of robots get
  obstacle tiles at level 0, the rest stay on the flat type; the target share rises to 0.8 (the paper's uniform mix) over
  500 iterations, robots moving from flat to an obstacle type (level 0) only when their episode ends. Part of the flat-first
  departure. CPU smoke test passed.

## Status (2026-11-12 02:57 UTC): r6b running

- **r6b training since 02:55:14 UTC** (tmux `ssr6b`, `runs/queue8.sh`, supervisor, heartbeat "r6b"), resumed from a copy
  of `t_a/last.pt` (iteration 549) at 550; types at the switch: 477 / 519 / 477 / 500 obstacle (ramp, stairs, beam,
  platform), 6,219 flat. First iteration: 61 % of commanded speed. Command (from `models/ssr/code`):
  `../env/venv/bin/python scripts/supervise_node.py --name r6b --iters 20000 --num_envs 8192 -- --flat_until 550 --obst0 0.25
  --obst_ramp 500 --stage_a 150 --stage_b 400 --w_style 0.2 --style_low 0 --gate_air 0.15 --gate_ramp 500 --prior_min_speed 0.2
  --rsi_min_speed 0.3 --plant mc_20261024 --camera chest_rs --amp_file data/amp_v2.npz --prior_balance`
- Departures from the paper in r6b: staged schedule (t_a's first 400 iterations); flat phase until 550 with the obstacle
  share ramped 25 % -> 80 % over 550-1,050; style weight 0 until touchdown air > 0.15 s for 5 iterations, then ramped to 0.2
  over 500; plus the earlier ones (walking starts, prior speed filter and balancing, reward x dt, camera mount randomisation).
- Gates: 40 % of commanded speed by 1,000; air time over 0.2 s by 2,000. Report at 700 and 1,000.
- Stop: `touch /mnt/filesystem-g4/tracks/ssr/runs/STOP`. Old runs kept: `runs/r6/` (stood, stopped at 490), `runs/t_a/`.

## r6b at 700 and the 1,000 restart (coordinator, 03:20 UTC)

- r6b at 700: 58 % of commanded speed (61 % at the switch, 80 % at 560, 68 % at 600); obstacle share 42 % (on its ramp);
  levels flat 1.8, beam 0.77, ramp 0.34, platform 0.28, stairs 0.05; ends: reached_end 24 %, time limit 39 %, tilt 15 %,
  side_obstacle 14 %. Touchdown air 0.06 s, flat; style latch (0.15 s) not fired, style still 0.
- Restart at 1,000 (`runs/queue9.sh`, tmux `ssr9`, automatic): r6b stopped after its log passes 1,005 and resumed from its
  1,000 checkpoint under the same name. If mean touchdown air time < 0.10 s: `--style_from 1000` (style weight by schedule,
  0 -> the paper's 0.2 over 500 iterations) and `--feet_air_w 20 --feet_air_min 0.1`; otherwise `--gate_air 0.10` (latch
  lowered). Code: `scripts/patch_air.py` (backups `_bak_20261112/*.pre_air`), CPU smoke test passed.
- **Feet-air-time reward (listed departure)**: at each touchdown under a moving command, 20 x clamp(air time - 0.10 s, 0,
  0.30 s); logged as `feet_air`. Positive only, on purpose: Isaac Lab's form (air - threshold) x first contact charges every
  short step, which a stander never pays, so it would push back toward standing. Hiking equivalent: `feet_air_time` +0.5
  (Isaac Lab form, only above 0.15 m/s commanded).

  Per-step value under the new weights (style at its full 0.2; Table 6 as implemented; foothold 0.25 x about 1.0):

  | Gait | Table 6 | foothold | style (0.2 x r_s) | feet_air | total |
  | --- | --- | --- | --- | --- | --- |
  | stander | 1.88 | 0.25 | 0.07 (r_s 0.35) | 0 | 2.20 |
  | shuffler (t_a probe: 3.7 touchdowns/foot/s, air 0.06-0.10 s, step 0.14 m) | 2.77 (2.16 with noise) | 0.25 | 0.00-0.04 (r_s 0.01-0.18, d1 shuffle) | 0 to 0.003 | 3.02-3.06 |
  | walker, 0.2 s air, 0.6 m/s, 0.25-0.3 m steps (about 2.2 touchdowns/s) | about 2.7-2.9 (estimate) | 0.25 | 0.19 (r_s about 0.93, reference walking) | 2.2 / 50 x 20 x 0.10 = 0.088 | 3.23-3.43 |
  | walker, 0.3 s air (about 2.0 touchdowns/s) | about 2.7-2.9 | 0.25 | 0.19 | 2.0 / 50 x 20 x 0.20 = 0.16 | 3.30-3.50 |

  Gap walker - shuffler: about +0.2 to +0.45 per step (feet_air 0.09-0.16, style 0.15-0.19), against about 0 without them.

## Port-audit points checked for SSR (2026-11-12 03:20, CPU; `scripts/port_audit.py`, `eval/port_audit.json`)

Reference = moving windows (>= 0.2 m/s) of the flat prior `amp_v2`; self-contacts on 1,570 full-body frames of the BONES-SEED
flat clips in the SSR training model (MuJoCo; mesh geoms collide with each other, parent-child pairs excluded).

1. **Discriminator inputs**: 21 joints (12 leg, waist yaw, 8 arm: shoulder pitch/roll/yaw, elbow) with velocities, base
   gravity and velocities, wrist and foot positions in the base frame. Not seen: head, waist pitch/roll, wrists. **Applies.**
   Reference arms: shoulder roll past the X2 hard limit (+-0.061) in 66 % / 62 % of frames (R / L), shoulder yaw +0.58 / -0.59
   (default 0), elbow -0.35 (default -0.6); wrist links inside the hip/pelvis links in 46 % of frames (up to 59 mm vs the hip
   roll link, 101 mm vs the pelvis). Explains r3's standing policy pressing both shoulder rolls on the limit 94 % of the time.
   Reference waist pitch mean 0.214 rad (12 deg forward, p95 0.314 = limit); SSR holds waist pitch at 0 and does not show it to
   the discriminator, but the wrist positions carry the lean. Pelvis tilt in the reference 4.5 deg.
   **Fix (in from the 1,000 restart): `--disc_mask_arms`** (arm q, qd and wrist positions zeroed at the discriminator input, both
   sides; `scripts/patch_mask.py`; network size unchanged so the checkpoint resumes). Listed departure.
2. **Default pose vs prior mean**: knee 0.669 vs 0.32, hip pitch -0.312 vs -0.08, ankle pitch -0.363 vs -0.06, hip yaw 0 vs
   -0.21 / +0.21 (toe-out). **Applies** (same as Hiking). `dof_dev` (-1.8 x mean |q - default|) charges the reference posture
   about 0.40 per step (legs 0.19, arms 0.21) against 0.11-0.20 for the current walker, so style and dof_dev pull apart.
   Not fixed now (changing the default pose moves every action offset: a from-scratch change); options for the next fresh run:
   default pose = reference mean leg pose, or dof_dev measured from it.
3. **Action scale**: 0.261 rad per unit at hips, knee and waist yaw, 0.258 ankle pitch / shoulder pitch and roll, 0.485 ankle
   roll / shoulder yaw / elbow. Reference p5-p95 ranges in action units: knee 4.1, hip pitch 2.7, hip yaw 2.7, ankle pitch 1.6.
   **Applies weakly to the cost**: following the reference exactly costs 0.001-0.003 per step in action_rate (flat, ramp,
   stairs, beam, platform), against 0.10-0.16 measured, which is the exploration noise (0.12 x 2 sigma^2 at sigma 0.6-0.7). It
   matters for exploration: with sigma 0.6-1.0 a knee swing needs 4 units. Not changed (gains/scale are the user's plant rule).
4. **Self-collision**: none at the default pose. In reference poses 89 % of frames have a self-contact, 80 % with arms at the
   default: pelvis vs hip roll link (both sides, up to 11 mm) in most frames, the X2 hull overlap the audit found. **Applies to
   the physics** (contacts push against hip roll/pitch while stepping); no reward term charges them, and only elbow links end
   an episode on force (> 80 N). Not changed yet: whether the walking policy actually meets them is to be measured (probe with
   contact counts); excluding pelvis-hip_roll pairs would be a model change, proposed for the next fresh run.
5. **Limit bands**: SSR's `dof_pos_lim` uses the hard limits (no soft factor). Reference frames outside them: shoulder roll
   62-66 % (removed from style by fix 1), elbow 2-6 %, knee 0.15 %; within 5 % of the range: hip roll 45 % (p5 -0.23 vs limit
   -0.235), knee 64 % (knee p5 0.095 vs limit 0). **Applies mildly** (cost 0.07 per joint-step outside; the r3 stander paid
   0.14 from the two shoulder rolls).

## Status (2026-11-12 03:53 UTC): r6b past its 1,000 gate, restarted with style + feet-air

- **Gate at 1,000 passed**: 59 % of commanded speed (gate 40 %); 1,000 range 57-64 % over 800-1,000. Air time 0.066 s (flat
  since 550), so the restart took the scheduled-style branch: r6b stopped at 1,010 (03:46:49) and resumed from its 1,000
  checkpoint at 03:47:13 with `--style_from 1000 --feet_air_w 20 --feet_air_min 0.1 --disc_mask_arms` (all confirmed in
  `runs/r6b/args.json`). Kept: `runs/r6b/ckpt_001000.pt`.
- First 40 iterations after the restart (1,040): 70 % of commanded speed, touchdown air 0.075 s (rising from 0.066),
  share of touchdowns over 0.2 s 1.4 % (0.9 % before), single support 0.62, style weight 0.016 (ramping to 0.2 by 1,500),
  feet_air 0.04/step, obstacle share 79 %; levels flat 5.1, beam 2.5, platform 1.7, stairs 1.5, ramp 1.0; 30k samples/s.
- Next gates: air time over 0.2 s by 2,000; speed must stay above 30 % (report if not).
- Full current command (from `models/ssr/code`): `../env/venv/bin/python scripts/supervise_node.py --name r6b --iters 20000
  --num_envs 8192 -- --flat_until 550 --obst0 0.25 --obst_ramp 500 --stage_a 150 --stage_b 400 --w_style 0.2 --style_low 0
  --gate_ramp 500 --prior_min_speed 0.2 --rsi_min_speed 0.3 --plant mc_20261024 --camera chest_rs --amp_file data/amp_v2.npz
  --prior_balance --disc_mask_arms --style_from 1000 --feet_air_w 20 --feet_air_min 0.1 --gate_air 0.15` (tmux `ssr9`, which
  ran `runs/queue9.sh`; heartbeat "r6b").
- Departures in r6b, all listed: staged schedule (t_a, 0-400); flat phase to 550 and obstacle share 25 % -> 80 % over
  550-1,050; style 0 until 1,000, then ramped to the paper's 0.2 over 500; feet-air reward (20 x clamp(air - 0.10, 0, 0.30)
  per touchdown, moving commands, from 1,000); discriminator arms masked (from 1,000); plus the earlier ones.

## Status (2026-11-12 05:00 UTC, resumed SSR agent)

- **r6b healthy, not restarted**: tmux `ssr9`, supervisor pid 110000, 0 restarts since 03:47:13; iteration 1,660 at 05:00
  (6.5 s per iteration, 30.5k samples/s; GPU 50-94 % by sample, 34 GB). Heartbeat current (04:53:49, log age 30 s).
- Numbers at 1,600 (mean of 1,560-1,610 in brackets): 63 % of commanded speed (58-66 %), touchdown air 0.110 s (0.105-0.111),
  share of touchdowns over 0.2 s 13 %, step rate 6.2 touchdowns/s, single support 0.68, terrain level mean 4.3 (flat 8.1,
  beam 6.2, platform 3.6, stairs 2.7, ramp 1.3), style weight at the paper's 0.2 since 1,500, action std 0.84 (rising).
- Air-time trend since the 1,000 restart: 0.067 (980), 0.077 (1,070), 0.082 (1,170), 0.091 (1,270), 0.100 (1,370),
  0.108 (1,470), 0.111 (1,570), 0.110 (1,600): flattening. Projected about 0.12 s at 2,000, so the 0.2 s gate will very
  likely fail. Step rate 8.4 -> 6.2, also flattening.
- Checkpoints: `runs/r6b/last.pt` every 50 (written on the shared filesystem, newest 04:53, iteration 1,600 at copy time),
  `prev.pt`, `ckpt_001000.pt`; numbered every 1,000 (so no 1,500). No separate backup step: runs write to
  `/mnt/filesystem-g4/tracks/ssr/runs/` directly.
- **Eval package (item 3), done on the node CPU while training ran**: `scripts/export_eval.py` (new) writes
  `/mnt/filesystem-g4/tracks/ssr/eval/export/r6b_001600/{policy.onnx, adapter_card.json, obs_check.npz}` from a frozen copy
  `eval/export/ckpts/r6b_001600.pt` (+ `args.json`). ONNX batch 1, fixed sizes, inputs hist 1x5x72, depth 1x36x36, gru 1x256,
  outputs action 1x21, gru_out 1x256; torch vs ONNX max difference 1.9e-6 over the 50 recorded steps. obs_check: 50 steps after
  100 warm-up steps of one robot in the training simulator (train=False, flat, 0.6 m/s, nominal mount/gains/masses, friction
  1.0), mean vx 0.51 m/s; holds qpos/qvel, command, hist, depth, camera ray origins/directions/hits, camera rays in the camera
  frame, GRU in/out, velocity estimate, latent, actor input, torch and ONNX actions, the 31 joint targets. Card built from the
  live training objects (plant mc_20261024 with kp/kd/effort/armature, default_q, action scale, clip, target clamp, held
  joints, observation terms and noise, depth camera, command behaviour: no waypoint shaping in training).
  Not done: the eval adapter `eval/adapters/ssr.py` (eval monitor's tree). For it: SSR's depth includes self-occlusion by limb
  spheres (`x2.OCCLUDERS`), not robot meshes; the eval's View ray cast sees only terrain and walls.
- Earliest checkpoint the eval monitor can score on ladder step 1 flat tests: `r6b_001600` now (only blocker: the adapter).
- **Fix**: `scripts/probe_gait.py` built its env with the default `head` camera, not the training `chest_rs`; earlier gait
  probes (t_a, t_b, d1) ran with the wrong camera (flat floor, so small effect, but not the trained input). Now defaults to
  `chest_rs` (`PROBE_CAMERA` to override); backup `_bak_20261112/probe_gait.py.pre_camera`.
- **Extended terrain map (item 4)**: plan above unchanged; date: the categories go in at the next fresh run (item 2 run), or
  at the r6b 3,000 checkpoint if r6b keeps going, target 2026-11-12 08:00 UTC; the tile loader from `extended_v1`
  (`index.json`, `load_example.py` present) is written on CPU before then.
- If the 2,000 gate fails (expected): candidate next experiment, from the port audit: the default pose (knee 0.669 vs the
  reference 0.32, ankle -0.36 vs -0.06) and dof_dev measured from it pull against style, and pelvis/hip-roll self-contacts
  are present in 89 % of reference stepping poses. Step 1 (resumable from r6b 2,000, no action-offset change): exclude the
  pelvis/hip-roll pairs and measure dof_dev from the reference mean leg pose, 500 iterations, watch air time. Step 2 (fresh):
  default pose = reference mean.

## Step 1 at the 2,000 gate (coordinator decision 2026-11-12 05:00; set up 05:05 UTC)

- Watcher `runs/queue10.sh` (tmux `ssr10`, output `runs/queue10.log`, gate numbers `runs/gate2000.json`): when r6b's log
  passes 2,005 and `ckpt_002000.pt` exists, takes the mean of logs 1,980-2,000. Air time per touchdown >= 0.2 s: r6b left
  alone. Under 0.2 s: STOP r6b through its supervisor (one GPU job at a time; waits for the trainer and supervisor to exit),
  copy `r6b/ckpt_002000.pt` to `runs/r6c/last.pt`, start supervisor `r6c` to iteration 2,500 with r6b's full command plus
  `--excl_pelvis_hiproll --dof_dev_ref`. r6b's folder is untouched, so plain r6b resumes from `ckpt_002000.pt` at any time.
- **Two new deviations from the paper (step 1)**, code `scripts/patch_step1.py` (backups `_bak_20261112/*.pre_step1`):
  1. `--excl_pelvis_hiproll`: MuJoCo body exclusions pelvis / left_hip_roll_link and pelvis / right_hip_roll_link (their hulls
     overlap in 89 % of reference stepping frames, up to 11 mm). Checked on the CPU: at hip roll +-0.25 and hip pitch -0.6,
     MuJoCo Warp finds 4 pelvis/hip-roll contacts without it and 0 with it (nexclude 1 -> 3).
  2. `--dof_dev_ref`: the dof_dev penalty is measured from the reference mean leg pose (amp_v2 flat, moving windows; R/L hip
     pitch -0.082/-0.080, hip roll 0.032/-0.033, hip yaw -0.208/0.211, knee 0.321/0.318, ankle pitch -0.060/-0.060, ankle roll
     -0.053/0.052) instead of the default pose; waist and arms still from the default. Action offsets (default pose) unchanged.
  CPU smoke test (8 robots, resume from r6b 1,600, one iteration with both flags) passed.
- **Success at +500 (r6c 2,500), against r6b's 2,000 numbers:** air time per touchdown >= 0.15 s or at least +0.04 s over
  r6b's 2,000 value and still rising; step rate down by at least 0.8 touchdowns/s (toward 5/s or less); speed share no more
  than 5 points under r6b's 2,000 value. Report at +250 and +500.
- **Fall back to plain r6b from `ckpt_002000.pt` (stop r6c early)** if at any point after +50: speed share under 40 % for
  30 iterations, or tilt plus diverged ends over twice r6b's 2,000 share, or terrain level mean down by more than 1.0.
- **No clear gain at +500** (air time gain under 0.02 s and step rate within 0.3/s): start step 2 (fresh run, default pose =
  reference mean), being prepared on the CPU now.

## Step 2 prepared on the CPU (2026-11-12 05:10 UTC, not started)

- `--default_from_ref` (`scripts/patch_step2.py`, backups `_bak_20261112/*.pre_step2`; default off, so r6b/r6c behave as
  before): the 12 leg default positions come from `code/config/default_ref_v1.json` (reference mean, left/right symmetrised:
  hip pitch -0.081, hip roll +-0.0325, hip yaw -/+0.2095 (toe-out), knee 0.3195, ankle pitch -0.060, ankle roll -/+0.0525).
  Action offsets, dof_dev, stand_still and the reset pose follow. The motion prior and walking starts (stored as q minus the
  plant default) are shifted by (old - new default) before mirroring, so they are the same absolute poses (shift printed:
  hip pitch -0.231, knee +0.350, ankle pitch -0.303, hip yaw +-0.209). `export_eval.py` rebuilds the model from `args.json`.
- Standing pelvis height above the soles: 0.641 m at the plant default, 0.663 m at the reference-mean default; the
  base_height target is 0.68 m (unchanged, closer to the new pose).
- Launch file `code/run_commands/step2_r7.sh` (run r7, from scratch, r6b's final flags from iteration 0 plus
  `--default_from_ref --excl_pelvis_hiproll`; refuses to start while a trainer runs). CPU smoke test (8 robots, 1 iteration
  from scratch) passed; step-1 resume smoke test re-run after the patch, passed.
- Deviation list for r7: as r6b, plus default leg pose from the reference mean and pelvis/hip-roll exclusion; feet-air and
  arm-masked discriminators from iteration 0 instead of 1,000.

## Slow commands (coordinator question 05:20; eval monitor: r6b_001600 does not start below 0.6 m/s in MuJoCo)

- **Same in the training simulator** (`eval/speed_bins.sh`, `probe_gait.py` on the CPU, 16 robots, 6 s, deterministic, flat,
  `chest_rs`, start from the default stand; results in `eval/speed_bins/`), r6b_001600:

  | command m/s | 0.0 | 0.2 | 0.35 | 0.5 | 0.6 | 0.8 | 1.0 |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | achieved m/s | 0.006 | 0.008 | 0.010 | 0.379 | 0.494 | 0.710 | 0.886 |
  | single support | 0.00 | 0.00 | 0.01 | 0.80 | 0.84 | 0.88 | 0.91 |
  | steps per foot per s | 0 | 0 | 0 | 2.08 | 2.00 | 1.99 | 2.08 |
  | air time median s | - | - | - | 0.18 | 0.20 | 0.22 | 0.22 |

  A zero command stands (good); 0.2-0.35 m/s stands too (bad); from 0.5 m/s it walks, at about 2 steps per foot per second
  with 0.18-0.22 s swings. **The training log's 0.11 s air time and 6.2 touchdowns/s are the noisy training rollouts
  (action std 0.84, obstacle tiles), not the deterministic gait**: the deterministic flat gait already has about 0.2 s swings.
  The log has no per-command-bin speed; the per-bin table above stands in for it.
- **Why (recipe)**: the velocity kernel exp(-err^2 / 0.25) pays a stander 0.70 of the walking reward at 0.3 m/s and 0.37 at
  0.5 m/s, less than the effort, smoothness and posture costs of stepping. Walking starts only above 0.3 m/s
  (`--rsi_min_speed 0.3`), so a slow-command episode almost always starts from a stand. The prior has only moving windows
  (`--prior_min_speed 0.2`, p10 0.29 m/s), so slow walking is rare in it.
- **Folded into step 1 (r6c), cheap existing flags**: `--sig_v0 0.1 --cmd_cur_iters 20000` (kernel width 0.1 instead of the
  paper's 0.25 for the whole run; a stander now gets 0.41 at 0.3 m/s and 0.08 at 0.5). Listed deviation. This mixes two
  effects in r6c; the per-bin probe at +250 and +500 separates the slow-command result from the air-time result.
- **Next, if slow commands still stand at +500**: walking starts from 0.15 m/s (`--rsi_min_speed 0.15`), slow clips (0.1-0.3 m/s)
  added to the prior and a per-bin speed reward (relative error |v - c| / max(c, 0.2)); and the eval's stop near a waypoint
  needs station holding, which training has none of (stands only at zero commands).

## Gate at 2,000 (2026-11-12 05:38 UTC): failed; r6c (step 1) running

- r6b logs 1,980-2,000 (`runs/gate2000.json`): air per touchdown **0.124 s** (gate 0.2), share over 0.2 s 21 %, speed 59 %
  of command, 5.5 touchdowns/s, single support 0.67, level mean 5.45 (flat 9.5, beam 8.8, platform 4.6, stairs 3.3, ramp 1.4);
  ends: time limit 33 %, reached end 26 %, side obstacle 21 %, tilt 6.4 %, link force 5.8 %, diverged 1.8 %.
- Check before acting (brief rule, my own data): the training-log air time is measured on noisy rollouts (std 0.84, 80 %
  obstacle tiles). The deterministic flat gait at 1,600 already swings 0.18-0.22 s at 2 steps per foot per second from 0.5 m/s
  up; the real defect is **slow commands (0.2-0.35 m/s) stand** (table in "Slow commands"), which the apparatus needs.
  So step 1 goes ahead with the slow-command kernel folded in (decision mine, 05:24).
- r6b stopped by its supervisor at 2,010 (05:38:53); **r6c** started 05:39:02 from a copy of `r6b/ckpt_002000.pt`
  (supervisor r6c, to 2,500; flags = r6b + `--excl_pelvis_hiproll --dof_dev_ref --sig_v0 0.1 --cmd_cur_iters 20000`, confirmed
  in the log). r6b's folder is unchanged apart from its log to 2,010.
- **Watcher `runs/queue11.sh`** (tmux `ssr11`, log `runs/queue11.log`), acting by itself:
  - harm (from 2,050, last 3 logs): best speed share under 40 %, or tilt + diverged ends over 16.6 % (twice r6b's 8.3 %), or
    level mean under 4.45 -> stop r6c, copy `r6b/ckpt_002000.pt` to `r6b/last.pt`, resume plain r6b to 20,000;
  - at 2,250: per-bin probe on the CPU (`eval/speed_bins/r6c_002250.json`);
  - at 2,500 (r6c ends): per-bin probe, then **continue r6c to 20,000** if (air >= 0.15 s or +0.04 over 0.124, with speed share
    >= 54 %) or (speed at a 0.35 m/s command >= 0.20 m/s, with speed share >= 54 %); **otherwise start step 2 (r7)**.
  - The step-rate criterion written at 05:05 is dropped from the automatic decision (the deterministic gait already steps at
    about 2 per foot per second; the training number is mostly noise), it is still reported.
- r7 (`code/run_commands/step2_r7.sh`) now also has `--sig_v0 0.1 --cmd_cur_iters 20000 --rsi_min_speed 0.15` (slow-command
  fixes; walking starts from 0.15 m/s).
- Baseline per-bin probe of r6b 2,000 running on the CPU (`eval/speed_bins/r6b_002000.json`).

## r6c stopped at 2,050 by its tripwire; cause found; r6d set up (2026-11-12 05:58 UTC)

- r6c (from r6b 2,000 with three changes at once) at its very first log (2,010): speed 49 % of command, tilt 56 % of ends,
  orientation term 0.10 (r6b 0.41), impact/slip/stumble 3-4x. Plain r6b resumed from the same checkpoint at 05:45 had at its
  first log: 76 %, tilt 2.9 %, orientation 0.43. A collapse within 10 iterations is physics, not learning.
- **A/B on the CPU, the same r6b_002000 weights, deterministic, flat, no learning**: with `X2_EXCL_PELVIS_HIPROLL=1` the
  robot falls (survival 0 % at 0.6 m/s, 6 % at 0.8; single support 0.48); without it, 100 % and 0.57 / 0.74 m/s. **The
  policy depends on the pelvis/hip-roll hull contacts** (an artefact of the convex hulls; it probably leans on them). That
  change was the harm; the +50 tripwire was right for it. The other two (dof_dev reference, kernel 0.1) are untested.
  Whether the eval's X2 model has these contacts too decides whether the exclusion belongs in training at all: question
  for the eval monitor (via the coordinator). Until then, exclusion is out of r7 too.
- **Decision (mine): r6d = r6b + one change, the velocity kernel narrowed for slow commands, ramped 0.25 -> 0.1 over
  2,100-2,400** (`--sig_v_to 0.1 --sig_v_from 2100 --sig_v_ramp 300`, backup `_bak_20261112/train.py.pre_sigramp`; deviation
  from the paper). Reason: the nearer apparatus blocker is that 0.2-0.35 m/s commands stand (eval monitor and my per-bin
  table; r6b_002000: 0.0/0.2/0.35 -> 0.01 m/s, 0.5 -> 0.44, 0.6 -> 0.57, 0.8 -> 0.74, 1.0 -> 0.90).
  Watcher `runs/queue12.sh` (tmux `ssr12`, log `runs/queue12.log`): stops r6b after 2,105, starts r6d from r6b's last.pt
  (2,100) to 2,600. Tripwires: catastrophic at +50 to +150 (tilt + diverged over 50 % of ends); from +150 a trend on the
  last 15 logs (speed share under 45 % or tilt + diverged over 16.6 %) -> stop and start r7. Probes at +250 and +500.
  At 2,600: continue r6d to 20,000 if speed at a 0.35 m/s command >= 0.20 m/s and share >= 54 %, else start r7.
  The dof_dev-reference change, if tried on r6d's line, gets its own ramp (from default to reference mean over 300) later.
- r7 (fresh, `run_commands/step2_r7.sh`): `--default_from_ref`, kernel 0.1, walking starts from 0.15 m/s; no exclusion.
- **Gates that match the apparatus** (replacing the 0.2 s training-log air time, which is mostly exploration noise):
  (1) per-bin deterministic speed: commands 0.2 / 0.35 / 0.5 m/s achieve >= 70 % of command, zero command stands
  (< 0.05 m/s, 1 s and longer); (2) flat-probe swing clearance median >= 0.10 m (9 cm log plus margin); (3) beam (4x4
  analogue) and stairs levels; (4) the eval's flat tests. The probe already reports clearance and step length.
- **Laggards ramp 1.9, stairs 3.9** (flat 10, beam 9.3, platform 5.2): ramp tiles draw speed U(-1, 1), half the slope
  episodes are backward walks, and the prior is about 10 % backward; stairs need swing clearance over the riser, which the
  short noisy swings lack. Not yet measured per type (the log has no per-type ends or speed): add per-type speed share and
  ends to the log at the next restart, then decide (backward commands on ramps held to the forward-only share until the
  forward gait is clean, stair clips weighted up).
- Eval export of `r6b_002000` started on the CPU by queue12 (`eval/export/r6b_002000/`).

## Port audit read; r6d stopped; audit test (a)1 running (2026-11-12 06:27 UTC)

- Read `agentic/state/ssr_port_audit.md`. Accepted: r6b and r6d hold hip roll on a hull stop (hip-roll action at the +-8 bound, joint
  at 0.057 rad); the eval uses the same MJCF; the coordinator's shared-model decision is pelvis/hip-roll pairs excluded for
  all runs from now on. Every checkpoint so far (r6b, r6d, t_a) was trained on that wrong body.
- **Decision (mine): r6d stopped at 2,350** (06:26; 2,320-2,340: 58-61 % of command, air 0.12 s, kernel at 0.13 on its ramp;
  `runs/r6d/last.pt` kept). Its 2,600 answer would have been about the wrong body.
  Stop note: killing `queue12.sh` also killed the r6d supervisor it had started with `&`, but not the trainer; the trainer
  (pid 135580) kept running alone until I sent it SIGTERM right after its 2,350 save. **Lesson: stop runs only through
  STOP + the supervisor; never kill a queue script that owns a supervisor.**
- **Audit test (a)1 now** (`runs/queue13.sh`, tmux `ssr13`, log `runs/queue13.log`), from scratch, 300 iterations each:
  - **e1** (started 06:27:12) = d4's exact arguments (flat, all Table 6 terms, no staged schedule, no air-time reward,
    style 0, head camera, amp.npz) plus `--excl_pelvis_hiproll`: a clean A/B against d4, which stood (2 % of command at 90,
    10 % at 190, 21 % at 290; double support 0.69 at 290).
  - **e2** (after e1) = e1 with the paper's style weight 0.2, the chest_rs camera, the amp_v2 prior: the paper's plain table
    with style, exclusion on.
  - After each: eval export with obs_check (CPU) and a hip-roll check (action mean, share at |a| > 7.5, joint mean), then
    the per-bin speed probe (exclusion on, the run's camera). Results go into queue13.log.
- What counts: e1/e2 walk if speed share passes d4's at equal iterations by a clear margin (>= 40 % by 290) with single
  support above 0.5, and hip-roll actions are off the bound (|a| > 7.5 in under 10 % of steps). Then r7 becomes: exclusion +
  the paper's table + slow-experience fixes (promotion at half the tile, walking starts from 0.1 m/s, command resampling
  within an episode, prior windows from 0.1 m/s) + default pose from the reference, with as few other deviations as possible.
  If they stand like d4: the hull stop was not the main cause of standing; r7 then keeps the staged recipe that made t_a walk,
  with the exclusion.
- r7 launch file to be rewritten after e1/e2 (the slow-experience fixes need code: promotion distance, command
  resampling). The GPU is idle after e2 (about 07:35 UTC) until I decide r7.

## e1 result and the plan to r7 (2026-11-12 07:10 UTC)

- **e1 (d4 + pelvis/hip-roll exclusion, flat, from scratch) against d4**, share of commanded speed:
  90: 3.2 % vs 2 %; 190: 5.4 % vs 10 %; 290: 14.5 % vs 21 %. At 290: single support 0.19 vs 0.31, double 0.81 vs 0.69,
  air per touchdown 0.034 s. Deterministic per-command speed (exclusion on, head camera): 0.00 / 0.01 / 0.03 / 0.05 / 0.05 /
  0.04 / 0.02 m/s at 0 / 0.2 / 0.35 / 0.5 / 0.6 / 0.8 / 1.0. **It stands like d4.**
- **Hip roll in e1 is off the bound** (obs_check, 50 deterministic steps; rerun with the venv python after the system-python
  failure in queue13): action mean -0.79 / +0.64 (R / L), 0 % of steps at |a| > 7.5, joint -0.009..0.013 / -0.022..0.007 rad.
  So the exclusion removes the lock, but **the hull stop was not the main cause of standing**.
- **Gait cost, per step, from the training logs** (all negative terms; e1 and d4 are standers, t_a a walker):
  e1 290: 0.436 = action_rate 0.087 + smoothness 0.109 + dof_dev 0.139 + dof_torque 0.025 + dof_vel 0.027 + impact 0.024 +
  slip 0.019 + limits 0.006. t_a 540 (walker): 0.544 (dof_dev 0.148, action_rate 0.104, smoothness 0.117, impact 0.058,
  slip 0.051). r6b 2,000: 0.728 (noise std 0.84).
  - action_rate + smoothness (0.196 in e1) are almost exactly the exploration noise: 0.12 x 2 sigma^2 + 0.06 x 6 sigma^2 at
    sigma 0.57 = 0.195. Every robot pays them, stander or walker; they do not depend on the gait. In action units they do not
    depend on the action scale either, but the same noise moves the 0.485-scale joints (ankle roll, shoulder yaw, elbow)
    about twice as far in radians, which feeds impact, slip and falls.
  - dof_dev (0.14) and dof_torque depend on the pose: measured from the G1 default (knee 0.669), and a crouch needs knee torque.
  - Walking minus standing, measured: costs +0.11 (0.544 vs 0.436), pays +0.58 (t_a 2.51 vs e1 1.93). **Walking already
    wins per step under our table**; the standers are a local optimum of learning from a stand, not a stand that the reward
    prefers. t_a's staged start (tracking and posture first) is what got out of it.
- **Queued, automatic (`runs/queue14.sh`, tmux `ssr14`, log `runs/queue14.log`)**: after queue13 (e2 and its checks), the
  e1 / e2 hip-roll checks with the venv; then **e3 = e1 + `--default_from_ref`** (one variable, 300 iterations); then
  **r7 starts by itself** (no idle GPU): base = the paper's plain table if e2 walked at 270-290 (share >= 0.40 and single
  >= 0.5), otherwise t_a's staged recipe (r6b's flags without the feet-air reward); plus the exclusion, walking starts from
  0.15 m/s, kernel 0.25 -> 0.1 over 600-900; plus `--default_from_ref` if e3 beat e1 by >= 0.10 share. The chosen flags go to
  `runs/r7_flags.txt` and the log.
- **Paper action scales ready, not run** (needs the user's agreement under the plant rule): `--action_scale_paper`
  (0.25 legs and waist, 0.2 arms; backups `_bak_20261112/*.pre_ascale`; export_eval follows the flag).
- Still to write (next restart of r7): promotion at half the tile, command resampling within an episode, prior windows from
  0.1 m/s, per-type speed and ends in the log.

## e2, e3 and r7 restarted without the reference pose (2026-11-12 08:15 UTC)

- **e2** (paper table with style 0.2, chest_rs, amp_v2, exclusion, flat, from scratch): 20 % of command at 290, single
  support 0.27, air 0.040 s; hip roll off the bound (actions -0.58 / +0.44, joint 0.017 / -0.028 rad). Deterministic:
  0.00 / 0.07 / 0.10 / 0.11 / 0.11 / 0.11 / 0.09 m/s at 0 / 0.2 / 0.35 / 0.5 / 0.6 / 0.8 / 1.0. Stands (slight creep).
- **e3** (e1 + `--default_from_ref`): training share 0.311, single 0.469 at 270-290, **but deterministically it falls**:
  survival 0 % at 0.6 m/s, tilt 0.41, -0.52 to -0.57 m/s at every command including zero (it topples backward along x);
  hip-roll joints -0.554 (R) / -0.232 (L) rad, actions -3.29 / -1.53: the whole body rolls to one side. Its first 160+
  iterations had 0.5 s episodes. **The training share is signed world-x pelvis velocity over command (env.py line 617), so a
  robot toppling or lunging along x scores as "speed"**; it fooled the queue14 rule.
- Checks (CPU, coordinator's list):
  1. Passive 5 s stand under the plant PD (mc_20261024, exclusion on, plain MuJoCo, plane floor): **neither pose stands
     passively**: plant default (COM 4 mm behind the sole centres) pitches +15 deg at 1 s and is down at 2 s, falling 0.8 m
     forward; reference-mean pose (COM 21 mm ahead) +20 deg at 1 s, down at 2 s. So passive stability does not separate them;
     the reference pose starts with the COM further forward and the straighter knees (0.32) leave less room to catch.
  2. `q mean` in the hip-roll check is the absolute joint angle (qpos). `config/default_ref_v1.json` is symmetric
     (hip roll +0.0325 R / -0.0325 L, hip yaw -0.2095 / +0.2095, ankle roll -0.0525 / +0.0525) and inside the limits
     (right hip roll [-2.906, 0.235], left [-0.235, 2.906]). e3's -0.554 (R) is legal for the right joint (outward), the left at
     -0.232 is at its limit: a body roll, not a pose defect.
  3. e3's deterministic speed is negative (backward along world x) with tilt 0.41 and no survivors: a fall, not a gait.
- **Decision (mine): r7 stopped through its supervisor at iteration 60 (08:13:30) and restarted from scratch at 08:13:55
  without `--default_from_ref`** (`runs/queue15.sh`, tmux `ssr15`). The 60-iteration first attempt is kept as `runs/r7_dfr`.
  r7 flags (`runs/r7_flags_v2.txt`): t_a's staged recipe (tracking and posture first, other terms ramped 150-400), flat until
  550 then obstacle share 25 % -> 80 % over 500, style 0.2 by schedule from 1,000, prior balanced and from 0.2 m/s,
  discriminator arms masked, plant mc_20261024, chest_rs, amp_v2, **pelvis/hip-roll exclusion**, walking starts from
  **0.15 m/s**, kernel 0.25 -> 0.1 over 600-900. No feet-air reward, no reference pose. Confirmed in `runs/r7/args.json`.
  queue15 runs the deterministic per-command probe on the CPU every 250 iterations from 250 to 2,000
  (`eval/speed_bins/r7_NNNNNN.json`, lines in `runs/queue15.log`).
- **Watcher rule fixed (for every later decision)**: "walking" is decided only from the deterministic probe: survival
  >= 90 %, signed speed along the command >= 50 % of command at 0.5 m/s, and single support >= 0.5; never from the training
  share.
- **Gates for r7** (deterministic probe on the CPU, exclusion on; the eval's flat tests when exported):
  - by 550 (end of the flat phase): walks at 0.5 / 0.6 m/s (>= 70 % of command, survival >= 90 %), zero command stands;
  - by 1,500: 0.2 / 0.35 / 0.5 m/s commands reach >= 70 % of command; zero command holds a stand (< 0.05 m/s) for 1 s and more;
  - by 2,000: flat-probe swing clearance median >= 0.10 m; stairs and ramp levels >= 4 (r6b at 2,000: 3.3 / 1.4), beam >= 6;
  - every 1,000: eval export for the eval monitor.
  A gate missed by more than half means a diagnosis before anything else, not a new flag.

## r7 550 gate (2026-11-12 09:20 UTC): passed on speed; the gait is a shuffle

- Deterministic probe (CPU, 16 robots, 6 s, flat, chest_rs, exclusion on), command -> speed m/s (survival, single support,
  median air s):
  - 250: 0 -> 0.00, 0.2 -> 0.01, 0.35 -> 0.01, 0.5 -> 0.22 (1.00, 0.24, 0.02), 0.6 -> 0.31, 0.8 -> 0.34, 1.0 -> 0.34;
    survival 100 % everywhere.
  - 500: 0 -> 0.00 (1.00, 0.00), 0.2 -> 0.09 (1.00, 0.27, 0.04), 0.35 -> 0.29 (1.00, 0.46, 0.06), 0.5 -> 0.48 (1.00, 0.63, 0.08),
    0.6 -> 0.57 (1.00, 0.65, 0.08), 0.8 -> 0.70 (1.00, 0.66, 0.08), 1.0 -> 0.53 (1.00, 0.69, 0.08).
- Gate "by 550: >= 70 % at 0.5 / 0.6 with >= 90 % survival, zero command stands": **passed** (96 % / 95 %, 100 %, 0.00 m/s).
  0.35 m/s already at 83 %, 0.2 m/s at 45 %, before the kernel narrows (600-900).
- **Defect to watch**: median swing (air) 0.08 s and the probe's step counter (steps longer than its length threshold)
  near 0: it moves by quick short shuffles. A 9 cm log needs real swings; the 2,000 clearance gate (>= 0.10 m) is the test.
  Decision: no change now (the obstacle phase from 550 and style from 1,000 are what lengthened r6b's swings); if the
  1,000 probe still shows air under 0.12 s, the next restart adds the feet-air reward (r6b's form) from 1,000.
- **Export for the eval**: `eval/export/r7_000500/` (policy.onnx, adapter_card.json, obs_check.npz; ONNX vs torch 1.4e-6,
  mean vx 0.61 m/s at 0.6 in the record). The card now has `trained_model`: MJCF path, excluded body pairs (pelvis/left and
  right hip roll, head_pitch/torso), flags for the exclusion, reference pose and paper action scale.
- Training at 600: 77 % of command, single 0.55, obstacle phase on, episode 13.6 s, std 0.57.
- Gravity-compensated default targets (coordinator idea): not run; queued as a 300-iteration flat test for a free slot,
  with a 5 s passive-stand check first. The paper's setup (Isaac Gym PD, paper's own default pose and gains) has no
  gravity compensation in the text (paper silent on a passive stand).

## r7 at 750 / 1,000; turning and slow starts; command mix added (2026-11-12 10:20 UTC)

- Deterministic probe (`probe_gait.py`): start = default stand on the open flat tile (heading 0), command fixed for 6 s,
  numbers averaged over **all 300 steps of the 6 s** (start transient included); 16 robots. The eval starts from a stand and
  averages the last 4 s of 8 s, so my speed under-reads a slow starter and over-reads nothing. New probe `scripts/probe_turn.py`
  matches the eval: default stand, 8 s, mean of the last 4 s, 8 robots, omnidirectional command held fixed.
- Per command (speed m/s, survival, single support, median air s, swing clearance median of real steps m, step length m):
  - 750: 0 -> 0.00; 0.2 -> 0.00 (1.0, 0.00); 0.35 -> 0.34 (1.0, 0.63, 0.08, -, 0.09); 0.5 -> 0.54 (1.0, 0.73, 0.10, 0.019, 0.15);
    0.6 -> 0.64 (0.74, 0.10, 0.021, 0.18); 0.8 -> 0.81 (0.80, 0.10, 0.029, 0.23); 1.0 -> 0.95 (0.81, 0.10, 0.038, 0.27).
  - 1,000: 0 -> 0.00; 0.2 -> 0.01 (1.0, 0.00); 0.35 -> 0.37 (1.0, 0.63, 0.08, 0.024, 0.10); 0.5 -> 0.53 (1.0, 0.71, 0.10, 0.024, 0.16);
    0.6 -> 0.61 (0.72, 0.10, 0.029, 0.19); 0.8 -> 0.76 (0.77, 0.12, 0.036, 0.24); 1.0 -> 0.89 (0.79, 0.12, 0.043, 0.28).
  - Survival 100 % at every command. Slow commands through the kernel ramp (600-900): 0.35 m/s improved (83 % at 500 ->
    106 % at 1,000); 0.2 m/s went back to a stand (0.09 at 500 -> 0.00-0.01). Dead zone between 0.2 and 0.35.
  - **Swing clearance 0.02-0.04 m**: far below the 9 cm log. This is the apparatus blocker on the gait side.
- Terrain levels at 1,100: flat 7.8, beam 4.5, platform 3.0, stairs 1.95, ramp 1.33. Training share 82 %, style weight 0.04
  (ramping to 0.2 by 1,500).
- Turn / start-from-stand probe (`eval/turn/`), yaw rate rad/s at commanded wz with vx = 0 (and vx from a stand):
  - r7_001000: wz +-0.4 -> 0.008 / -0.007; +-0.8 -> 0.44 / -0.34; forward 0.2 -> 0.00; 0.35 -> 0.39. All survive.
  - r6b_002000 (old body): +-0.4 -> 0.02; +-0.8 -> 0.85 / -0.83; forward 0.2 -> 0.00; 0.35 -> 0.00.
  So both have a dead zone for small yaw commands; r6b turned at 0.8 (why it passed the eval's turn test), r7 half. Cause
  (recipe): only 10 % of robots are omnidirectional and their vx is U(-1, 1), so a turn with no forward speed is rare; tile
  robots' yaw is a heading correction that starts aligned; one command per episode, so no robot ever starts from a stand into
  a small command; the kernels pay a stander 0.42 of the yaw reward at 0.4 rad/s and 0.67 of the linear reward at 0.2 m/s.
- **Decision (mine): r7 resumed with a command mix (10:17:53, from 1,150, same body; commands only, no physics change)**:
  `--omni_frac 0.2 --omni_turn 0.3 --omni_slow 0.3 --resample_s 8` (20 % omnidirectional robots, of them 30 % turn in place
  with |wz| 0.3-1.2 and no forward speed, 30 % slow forward 0.1-0.4 m/s; every robot gets a new command every 8 s, tile robots
  a new speed with the same heading, so stand -> slow start happens). Deviations from the paper (paper: commands per episode,
  omnidirectional stage separate). Code `env.py`/`train.py` (backups `*.pre_cmdmix`); CPU smoke test passed. Flags in
  `runs/r7_flags_v3.txt`, confirmed in `r7/args.json`. First logs: 91-93 % of command, tilt 2-3 %.
  Watcher `runs/queue16.sh` (tmux `ssr16`): speed and turn probes every 250 iterations up to 4,000.
- **Short swings, decision**: style reaches 0.2 at 1,500; if the 1,500 probe still shows clearance under 0.06 m, add a
  swing-height term (foot apex height over the ground during swing, positive only, moving commands) rather than r6b's air-time
  term (r6b's longer airs did not show clearance; clearance is what the log needs). Prepared on the CPU before 1,500.
- Gates (unchanged, plus): by 1,500 from a stand: wz +-0.4 -> >= 0.3 rad/s, +-0.8 -> >= 0.6; forward 0.2 -> >= 0.14 m/s,
  0.35 -> >= 0.25; zero stands.

## r7 1,250 / 1,500 gates; swing-height reward first (2026-11-12 11:30 UTC)

- From a stand (probe_turn: 8 s, mean of the last 4 s), all 100 % survival:
  - 1,250: yaw +-0.4 -> 0.03 / -0.025; +-0.8 -> 0.69 / -0.69; forward 0.2 -> 0.13; 0.35 -> 0.36.
  - 1,500: yaw +-0.4 -> 0.07 / -0.17; +-0.8 -> 0.49 / -0.66; forward 0.2 -> 0.00; 0.35 -> 0.36.
  - probe_gait (6 s incl. start), 1,500: 0 / 0.2 / 0.35 / 0.5 / 0.6 / 0.8 / 1.0 -> 0.00 / 0.01 / 0.34 / 0.52 / 0.62 / 0.78 / 0.91.
  - Clearance (median, real steps) / step length: 1,250: 0.020 / 0.09 m at 0.35, 0.021 / 0.15 at 0.5, 0.040 / 0.26 at 0.8;
    1,500: 0.00 / 0.10, 0.022 / 0.16, 0.038 / 0.27. Air 0.08-0.14 s.
  - Gates by 1,500: forward 0.2 FAIL (0.00; 0.13 at 1,250), forward 0.35 pass, yaw 0.4 FAIL, yaw 0.8 pass one side (0.66) and
    0.49 on the other, zero stands pass. Clearance 0.02-0.04 m (mark 0.06): FAIL.
- Terrain levels: 1,500 flat 11.3, beam 9.0, platform 6.3, stairs 3.75, ramp 2.0; 1,770 flat 12.8, beam 10.7, platform 7.5,
  stairs 4.0, ramp 2.2. The 1,150 resume kept the levels (1,150: 4.21 mean, 1,160: 4.25), and so did 11:25 (1,770 -> 1,760 after
  resume: ramp 2.24, stairs 4.03, beam 11.0): checkpoints carry the levels; no loss like Hiking's.
- **Order (mine): clearance first, slow commands second.** Reason: no adapter or command shaping can lift a foot over a 9 cm
  log, and stairs (4.0) and ramp (2.2) lag; the slow dead zone has a partial work-around (eval adapter command floor at the
  policy's trained minimum, SHARED_LESSONS "Slow commands" item 5) and 0.8 rad/s turns work. The slow-command change waits
  for the swing verdict at +250 (about 12:00 UTC).
- **Change 1, swing-height reward** (deviation; `env._swing_height`, flags `--swing_h_w 0.5 --swing_h_target 0.10`, backups
  `*.pre_swingh`): per step, sum over feet not in contact of min(foot sole height over the ground / 0.10, 1) x min(|foot
  velocity xy|, 1 m/s), only under moving commands; a stander earns 0. Tested first on the CPU with r7_001500 (w = 1): 0.159
  per step at 0.6 m/s with today's 2-4 cm swings, 0 at a zero command; at 10 cm it would pay about 0.6, so at w 0.5 the gain
  for a clean swing is about +0.2 per step (like full style).
  Resumed 11:25:55 from r7 at 1,750 (last.pt), flags `runs/r7_flags_v4.txt`; snapshot before the change
  `eval/export/ckpts/r7_pre_swing.pt`. First log after the resume: 88 % of command, swing_h 0.061, tilt 0, levels kept.
  Watcher `runs/queue17.sh` (tmux `ssr17`): tripwire from +50 to +250 (last 5 logs: share < 0.60 or tilt + diverged > 0.15)
  and a +250 probe (fail if clearance at 0.5 m/s < 0.035 m, speed at 0.5 < 0.45, or any survival < 0.9) -> fall back:
  resume from the snapshot with `r7_flags_v3.txt`. Probes every 250 iterations continue.
- **Change 2 (next, after the swing verdict), slow commands**: the reward is the cause. A stander keeps 0.67 of the linear
  reward at 0.2 m/s (kernel 0.1) and 0.42 of the yaw reward at 0.4 rad/s, and 350 iterations of more slow / turn commands did
  not move it (0.2 m/s went 0.13 -> 0.00). Smaller change than more command share: a relative-error kernel for small
  commands, exp(-|v - c|^2 / (sigma^2 max(|c|, c0)^2 / c_ref^2)) (same at 0.6 m/s, about 3x narrower at 0.2), for linear and
  yaw; alternative a wait penalty scaled to small commands (Hiking's `dont_wait`). Code and a CPU test of what a stander
  and a slow walker are paid, before any GPU time.

## Swing +250 (r7 2,000); turning lost; turn-heavy command mix (2026-11-12 12:10 UTC)

- probe_gait, speed / clearance (median real steps) / step length / survival:
  - 1,500: 0.35 -> 0.34 / 0.000 / 0.10; 0.5 -> 0.52 / 0.022 / 0.16; 0.6 -> 0.62 / 0.028 / 0.20; 0.8 -> 0.78 / 0.038 / 0.27;
    1.0 -> 0.91 / 0.046 / 0.33; all 100 %.
  - 2,000: 0.35 -> 0.38 / 0.025 / 0.14; 0.5 -> 0.52 / 0.039 / 0.20; 0.6 -> 0.60 / 0.047 / 0.24; 0.8 -> 0.74 / 0.060 / 0.30;
    1.0 -> 0.88 / 0.071 / 0.35; all 100 %. 0 and 0.2 m/s: stand.
  Swing reward works: clearance +70-80 % and steps +25-40 % longer in 250 iterations.
- Levels at 2,000: flat 13.8, beam 12.8, platform 9.0, stairs 5.05, ramp 2.82 (2,120: stairs 5.17, ramp 2.89).
- 2,000 gates: clearance >= 0.10 m FAIL (0.039 at 0.5, 0.071 at 1.0); stairs >= 4 PASS (5.05); ramp >= 4 FAIL (2.8); beam >= 6 PASS.
- Turning from a stand at 2,000: +-0.4 -> 0.02 / -0.018; +-0.8 -> 0.022 / -0.021 rad/s (1,500: 0.49 / -0.66; 1,250: 0.69).
  The fall-back test had no turning check, so it did not fire. Mine to fix: turning is now in every fall-back test.
- **Why turning vanished (CPU, deterministic, 8 robots, mean of the last 4 s of 8 s, swing weight 0.5, the 2,000 discriminators)**:
  - Task reward per step under (0, 0, 0.8): r7_001250 (turns at 0.69) earns **2.90**, r7_002000 (stands) earns **2.05**;
    the difference is the yaw term (0.73 vs 0.07). The swing term pays the turner about 0 (feet hardly move horizontally),
    the stander 0. So the reward strongly prefers turning at 0.8; the swing term is not what removed it.
  - Under (0, 0, 0.4): stander 2.48 vs 2.47 for the 1,250 policy that does not turn either: the yaw kernel pays a stander
    0.45 of 0.8 at 0.4 rad/s, no gain from turning: that is the 0.4 dead zone (kernel, change 2).
  - Style (flat discriminator, arms masked, x 0.2): stander 0.172, turner 0.146, walker at 0.6 m/s 0.122, slow 0.2 0.157.
    The discriminator scores a stand above walking and turning (r_s 0.86 vs 0.61 / 0.73), a small stand bias (0.03-0.05 per
    step); the flat prior has no turn-in-place windows (prior windows only at >= 0.2 m/s planar speed).
  - So the 0.8 loss is forgetting of a rare command (6 % of robots, 0.2 x 0.3) while the swing term reshaped the gait of the
    other 94 %, helped by the style's stand bias.
- **Decision (mine): one change, the command mix: `--omni_frac 0.3 --omni_turn 0.5`** (pure turns 15 % of robots, slow
  forward 9 %), resumed 12:08:28 from 2,130 (last.pt), flags `runs/r7_flags_v5.txt`, snapshot `eval/export/ckpts/r7_pre_turn.pt`.
  Not the swing-term fix (it does not pay or charge turners, so changing it would not bring turning back) and not change 2
  first (at 0.8 the kernel already pays a turner 0.73 vs 0.07; the kernel only explains 0.4).
  Watcher `runs/queue18.sh` (tmux `ssr18`): tripwire +50..+250 (share < 0.60 or tilt + diverged > 0.15); +250 probe (about
  12:40 UTC) fails if yaw at +-0.8 from a stand < 0.4 rad/s on either side, clearance at 0.5 < 0.035, speed at 0.5 < 0.45, or
  survival < 0.9 -> resume from the snapshot with `r7_flags_v4.txt`.
- Next: change 2 (relative-error kernel for small linear and yaw commands; CPU pay test for stander, slow walker, turner first),
  after the turn verdict. Prior turn / slow windows (prior_min_speed covers planar speed only) are a candidate for the style bias.
- Eval export: `eval/export/r7_002000/` (ONNX vs torch 2.1e-6, mean vx 0.63 at 0.6).

## Turn +250 (r7 2,250) passed; change 2 (relative-error kernels) running (2026-11-12 12:55 UTC)

- r7_002250 from a stand (100 % survival): yaw +-0.4 -> 0.14 / -0.25; +-0.8 -> 0.70 / -0.67 rad/s; forward 0.2 -> 0.18;
  0.35 -> 0.40 m/s. probe_gait speed / clearance / step / survival: 0 -> 0.00; 0.2 -> 0.13 / - / 0.04; 0.35 -> 0.38 / 0.046 /
  0.17; 0.5 -> 0.52 / 0.062 / 0.23; 0.6 -> 0.59 / 0.067 / 0.26; 0.8 -> 0.73 / 0.076 / 0.32; 1.0 -> 0.87 / 0.084 / 0.37; all 1.00.
  queue18 kept the turn mix (12:46:40, clearance 0.061). Turning at 0.8 is back; clearance kept rising.
- Levels at 2,410: flat 13.6, beam 14.4, platform 8.8, stairs 5.3, ramp 3.4 (2,460: ramp 3.7).
- Training share 0.59-0.77: the log's share already excludes omnidirectional robots (walkers told >= 0.3 m/s); the drop
  since 10:17 comes with the 8 s command resampling (speed steps mid-episode, acceleration lag), not with the gait (the
  deterministic probe is unchanged). Read the probe, not the share.
- Export: `eval/export/r7_002250/` (ONNX vs torch 2.4e-6).
- **Change 2, relative-error kernels** (`--rel_kernel`, `env._rel_scale`, backups `*.pre_relk`): kernel width x
  (max(|c|, floor) / c_ref)^2, capped at 1; linear c_ref 0.6 m/s floor 0.3, yaw c_ref 0.8 rad/s floor 0.4. CPU pay test first
  (r7_002250 rollouts, deterministic, lin + 0.8 yaw kernel terms, policy vs a perfect stander):

  | command | old: policy / stander / gain | relative: policy / stander / gain |
  | --- | --- | --- |
  | 0 | 1.80 / 1.80 / 0 | 1.80 / 1.80 / 0 |
  | 0.2 m/s | 1.74 / 1.47 / +0.27 | 1.59 / 1.00 / +0.58 |
  | 0.35 m/s | 1.72 / 1.09 / +0.62 | 1.55 / 0.83 / +0.72 |
  | 0.6 m/s | 1.73 / 0.83 / +0.90 | 1.67 / 0.83 / +0.84 |
  | yaw 0.4 | 1.59 / 1.42 / +0.17 | 1.32 / 1.06 / +0.26 (a full 0.4 turner: about +0.74 in the yaw term, old +0.38) |
  | yaw 0.8 | 1.71 / 1.06 / +0.65 | 1.63 / 1.06 / +0.57 |

  A first try with floors 0.15 / 0.2 made the zero components (yaw while walking, drift while turning) so tight that the
  walker's gain fell to +0.71 and the turner's to +0.31; floors 0.3 / 0.4 keep those within 10 % and double the slow gain.
- Resumed 12:49:00 from 2,460 (`runs/r7_flags_v6.txt`; snapshot `eval/export/ckpts/r7_pre_relk.pt`); levels kept.
  Watcher `runs/queue19.sh` (tmux `ssr19`): tripwire +50..+250; +250 probe (about 13:20 UTC) falls back to the snapshot
  with `r7_flags_v5.txt` if yaw at +-0.8 < 0.4 on either side, clearance at 0.5 < 0.05, speed at 0.5 < 0.45, or survival < 0.9.
  Success (reported): forward 0.2 from a stand >= 0.14 (now 0.18 start test / 0.13 probe_gait), yaw 0.4 >= 0.3 both sides.
- Next, one at a time after this: (a) motion prior: let the discriminator see slow and turning windows (amp_v2 flat: 8.6 % of
  frames are turns under 0.2 m/s with |wz| > 0.3, 18 % at 0.05-0.2 m/s; all cut by `--prior_min_speed 0.2`): keep windows with
  speed >= 0.05 or |wz| >= 0.3; (b) stands next to a drop (below).
- **Stands next to a drop**: not trained on purpose. Episodes start 0.35-0.9 m into a tile; zero / slow commands come only
  from the per-episode draw and the 8 s resample (about 1 in 8 under 0.15 m/s), so a robot stops near a platform or stair
  edge only by chance. Plan (ACP's fix): on platform and stair tiles, when a robot is within 0.2-0.5 m of a downward edge
  ahead, set its command to zero for 2 s in a share of episodes (and log edge-stand survival). Needs the edge positions from
  `ssr/terrain.py`; to be written after (a).

## Relative kernel: misfired fall-back, 1.0 m/s drop, second try (2026-11-12 13:30 UTC)

- 13:07:01 queue19 fell back at 2,610 on "share 0.595 tilt+div 0.033": the training speed share 0.005 under its 0.60 mark,
  the number I had already said not to read with 8 s resampling. **A misfire, my mistake**; about 35 GPU minutes lost.
  The deterministic probe of that run at 2,500 (40 iterations in) had hit most marks: from a stand forward 0.2 -> 0.19,
  0.35 -> 0.40; yaw +-0.4 -> 0.30 / -0.26, +-0.8 -> 0.79 / -0.80; clearance at 0.5 0.069 m, step 0.248 m; 100 % survival;
  but at 1.0 m/s 0.36 (before 0.87), single support 0.41, |yaw| 0.07, tilt 0.025.
- **Rule from now on: no fall-back reads the training speed share.** Fall-backs use the deterministic probes and the
  tilt + diverged share only.
- **1.0 m/s drop, cause**: the kernel cap made the linear kernel unchanged at >= 0.6 m/s, but the yaw kernel at a zero yaw
  command sat at its floor (width x 0.25) whatever the walking speed, so the yaw wobble of a fast gait was charged 4x harder;
  the policy slowed and stepped less (single support 0.41) at 1.0. The same held for the linear kernel while turning fast.
  **Fix**: each kernel's floor rises with the other command component: yaw floor = max(0.4, 0.8 |v| / 0.6), linear floor =
  max(0.3, 0.6 |wz| / 0.8). CPU check of the scales: (v, wz) = (0, 0) -> 0.25 / 0.25; (0.2, 0) -> 0.25 / 0.25; (0.6, 0) and
  (1.0, 0) -> 1.0 / 1.0 (exactly the old kernels at walking speed); (0, 0.4) -> 0.25 / 0.25; (0, 0.8) -> 1.0 / 1.0. So the
  pay test at 0.2 m/s and 0.4 rad/s is unchanged (+0.58, +0.26), and walking >= 0.6 m/s and turning >= 0.8 rad/s are paid as
  before the change (+0.90 at 0.6, +0.65 at 0.8). Backup `_bak_20261112/env.py.pre_relk2`.
- **Re-applied 13:24:55 from 2,600** (the newest good checkpoint, flags v5 + `--rel_kernel`; snapshot
  `eval/export/ckpts/r7_pre_relk2.pt`). Levels at 2,610: flat 13.6, beam 14.7, platform 8.2, stairs 5.4, **ramp 4.1** (ramp gate
  passes). Watcher `runs/queue20.sh` (tmux `ssr20`): tripwire tilt + diverged > 0.15 only; +100 probe (about 13:45 UTC):
  fall back on regressions (speed at 0.5 < 0.45, at 1.0 < 0.75, yaw at +-0.8 < 0.4, survival < 0.9); +250 probe (about
  14:05 UTC): all marks (plus forward 0.2 from a stand >= 0.14, yaw at +-0.4 >= 0.3 both sides, clearance at 0.5 >= 0.05).
  The verdict code was dry-run on r7_002500 (it would have failed there on 1.0 m/s 0.36 and yaw 0.4 at 0.26).
- Eval (agentic/state/eval.md): r7_002000 corrected, strict + retry: flat walk 50/50, stop-stand 14/50 standard / 50/50 keep-0.6,
  speed 0.2/0.35/0.5/0.6/0.8 -> 0.00/0.31/0.51/0.61/0.82, no turn in place, stop 0.11 m in 0.4-0.6 s. r7_002250 tables:
  speed 0.03/0.36/0.54/0.62/0.77, turn +0.4/+0.8/-0.4/-0.8 -> 0.03/0.72/-0.24/-0.69 rad/s (agrees with my probe), stop 0.16 m
  in 0.7 s; flat standard and keep runs, L1 / street / apparatus queued if walk and turn-and-return pass.

## Relative kernel dropped; edge stops running (2026-11-12 14:20 UTC)

- Second relative-kernel try fell back correctly at 13:44:11 (+100 probe at 2,770): 1.0 m/s -> 0.00 (the robot stood:
  single support 0.003, double 0.994), yaw +-0.4 -> 0.013, while 0.2 -> 0.18-0.19, 0.35 -> 0.35, 0.5 -> 0.51, 0.8 -> 0.78,
  yaw +-0.8 -> 0.69, clearance at 0.5 0.088 m, all survive.
- Why: with the rising floors the kernels at 1.0 m/s are exactly the old ones (scale 1.0, checked on the CPU), so a walker
  at 1.0 is paid as before (about 1.7 per step in lin + yaw) and a stander about 0.8 (lin exp(-1/0.1) = 0): the reward did
  not ask for standing at 1.0. The policy still lost 1.0 m/s and slow turning within 170 iterations of a reward change
  elsewhere in the command range. Twice now a kernel change has taken away commands the policy already handled; the
  command mix alone moved 0.2 m/s from 0.00 to 0.13-0.19 and yaw 0.4 from 0.02 to 0.14-0.30 without losing anything.
  **Decision: the relative kernel is dropped.** Slow commands are pursued with command share (and later the prior), not
  kernel shape.
- **Decision: next single change = stops at down edges** (`--edge_stop 0.3`, `env._edge_stop`, `scripts/patch_edge.py`,
  backups `*.pre_edge`): in 30 % of tile episodes, the first time the ground 0.45 m ahead along the robot's heading is
  >= 0.15 m below the ground under it, the command goes to zero for 2 s, then the episode's speed returns; counts in the log
  (`edge_stop: start / done`). Reason: the apparatus' last station is a stand 0.3 m from a 1.1 m drop, ACP held 0 of 30 there
  before training edge stands and 114 of 117 after; slow commands have a work-around in the adapter and partial skill, a
  fall at the last station fails the whole trial. Our tiles' drops are at most 0.40 m (platform 0.05-0.40 m, stairs
  0.05-0.20 m per step), not 1.1 m: a deeper drop would need taller platforms (terrain change, later if the apparatus result
  shows falls there).
  CPU test (32 robots, platform level 7, r7 at 2,850, edge_stop 1.0, 14 s): 12 edge stops started, 11 completed, 0
  terminations during a stop, mean |vx| while stopped 0.15 m/s (the untrained policy decelerates within the 2 s).
- Restart incidents (mine, 14:14-14:17): the first queue21 had a quoting error; bash stopped at it after it had already
  resumed r7. When I replaced the session with `tmux kill-session`, the SIGHUP killed the supervisor and left the trainer
  running alone for about a minute (no second GPU job); I stopped it with SIGTERM before it saved. **Lessons: run `bash -n`
  and stop on failure before any `tmux new-session` (no `;` after a failed check); never kill a tmux session that holds a
  supervisor.**
- r7 resumed 14:16:59 from 2,850 (last.pt) with `runs/r7_flags_v7.txt` (= v5 + `--edge_stop 0.3`; no relative kernel);
  snapshot `eval/export/ckpts/r7_pre_edge.pt`. First log: edge stops 2 started, 1 done; levels kept (ramp 4.7, stairs 5.2,
  beam 14.2, platform 6.9, flat 13.5). Watcher `runs/queue21.sh` (tmux `ssr21`): tripwire tilt + diverged > 0.15; +100
  (about 14:40 UTC) and +250 (about 15:00) probes fall back to the snapshot with v5 if speed at 0.5 < 0.45, at 1.0 < 0.75,
  yaw at +-0.8 < 0.4, forward 0.2 from a stand < 0.10, clearance at 0.5 < 0.05, or survival < 0.9. Probes every 250 after.
- Export of the newest good checkpoint (v5 run, 2,850): `eval/export/r7_snap2/`.
- Clearance trend: 0.022 (1,500) -> 0.039 (2,000) -> 0.062 (2,250) at 0.5 m/s; 0.069 at 2,500 and 0.088 at 2,770 were
  measured in the kernel runs (thrown away). The next v5-line number is the +100 edge probe. Gate 0.10 m.
- Levels at 2,800: flat 13.6, beam 14.1, platform 7.0, stairs 4.9, ramp 4.2 (ramp gate passed since 2,610).

## Eval corridor 0/50 (r7_002250, level 1): the station approach command (2026-11-12 14:35 UTC)

- Eval monitor: r7_002250 level 1 strict: log, wedge, stairs up, stairs down 50/50 each; corridor 0/50, every trial stopping
  0.27 m short of the final station.
- Checked `eval/adapters/ssr.py`: the default `SSR_CMD=standard` passes the harness command unchanged: forward speed
  min(distance, vmax) x cos(error) with a 0.15 m/s floor. Inside 0.35 m of a waypoint it asks for under 0.35 m/s; r7_002250
  walks from a stand only from about 0.35 m/s (0.2 m/s -> 0.13-0.18 in my probes, 0.03 in the eval's table) and a moving
  robot settles to a stand at 0.27 m/s. So the robot stops at about 0.27 m, outside the 0.20 m disc. My earlier "adapter
  work-around" exists only as the `keep` variant (constant vmax 0.6 until the disc), which the corridor run did not use.
  My statement was wrong for the standard runs.
- **What the adapter should send for SSR near a waypoint (declared in every result file, not hidden)**: forward speed
  max(min(distance, vmax), 0.40 m/s) x cos(error) while the pelvis is outside the arrival disc and the error is under 60 deg;
  zero speed (and the harness's turn to the station heading) once inside the disc, as Hiking's card does. Why 0.40: the
  policy's trained minimum where it reliably starts and keeps walking (0.35 m/s from a stand -> 0.36-0.40 in probe and eval).
  Its stop from walking is 0.11-0.16 m in 0.4-0.7 s at 0.6 m/s (eval), shorter from 0.4, so it ends inside the 0.20 m disc.
  For turns in place keep the harness yaw command, but at least 0.6 rad/s in magnitude while the error is over 60 deg
  (0.4 rad/s is still a dead zone: 0.03-0.25 rad/s).
- Passed to the eval monitor via the coordinator (eval code is theirs). This ranks with the edge stops for the apparatus
  score; the training side continues on slow commands (command share, then slow / turn windows in the prior).
- Taller platforms: decided at the first apparatus result for r7_002250; if it falls or steps off at the last station
  (0.3 m from a 1.1 m drop), platform heights up to about 1.1 m come forward as the next terrain change after edge stops.

## Edge stops +100 fell back on 1.0 m/s; the base line fails that test too (2026-11-12 14:45 UTC)

- queue21 +100 probe of edge stops (3,030): 0.2 -> 0.10 (probe_gait) / 0.14 (from a stand, 8 s); 0.35 -> 0.33; 0.5 -> 0.50;
  0.6 -> 0.60; 0.8 -> 0.70; **1.0 -> 0.00**; yaw +-0.4 -> 0.16 / -0.17; +-0.8 -> 0.76 / -0.74; clearance at 0.5 0.074 m;
  100 % survival; edge stops 8 started / 10 done per log interval, tilt 4 %. Fell back at 14:38:19 on the 1.0 m/s test.
- **Check of the base line at 1.0 m/s from a stand (probe_gait, CPU, 16 robots)**: r7_pre_relk (2,450, v5 flags):
  0.00 m/s, single support 0.002; r7_pre_relk2 (2,600, v5): 0.00; r7_pre_edge (2,850, v5): 0.67. So **the base line itself
  stands at a 1.0 m/s command from a stand at 2,450 and 2,600**: it is a flaky property of the line, not caused by the change.
  The relk2 fall-back (base 2,600: 0.00) and the edge fall-back were both judged on it. relk1 at 2,500 reached 0.36 at 1.0,
  better than its base. **My fall-back test was wrong**: 1.0 m/s from a stand is above the eval's vmax 0.6 and flaky on the base.
  Corrected test: speed at 0.8 >= 0.6 (and 0.5 >= 0.45); 1.0 is reported, not a fall-back.
- Consequence for the relative kernel: its two "losses" at 1.0 m/s are not evidence against it (base also 0.00); its real
  costs were yaw 0.4 at 0.013 at 2,770 (base about 0.14-0.17) and its gains fwd 0.2 0.19, clearance 0.069-0.088. It stays
  dropped for now (one change at a time; edge stops first), but it is not ruled out; a third try would compare against a
  base snapshot probed at the same time.
- **Edge stops re-applied 14:42:21** (queue22, tmux `ssr22`, snapshot `eval/export/ckpts/r7_pre_edge2.pt`, flags
  `runs/r7_flags_v7.txt`), fall-backs: speed at 0.5 >= 0.45, at 0.8 >= 0.6, yaw at +-0.8 >= 0.4, forward 0.2 from a stand >= 0.10,
  clearance at 0.5 >= 0.05, survival >= 0.9, tilt + diverged <= 0.15. +100 about 15:05, +250 about 15:25 UTC.

## Starts from a stand, relative marks, prior windows brought forward (2026-11-12 15:10 UTC)

- **Start probe** (`scripts/probe_start.py`, CPU, deterministic, flat, 64 robots, start yaw U(-15, 15) deg, 1 s at zero
  command then speed v along heading 0; started = mean forward speed over the last 4 s of 8 s >= 0.5 v; seeded, so every
  checkpoint sees the same starts; results `eval/start/*.json`):

  | checkpoint (v5 line) | 0.35 | 0.5 | 0.6 | 0.8 | 1.0 |
  | --- | --- | --- | --- | --- | --- |
  | 2,450 | 100 % (0.37) | 100 % (0.53) | 100 % (0.64) | 98 % (0.81) | 0 % (0.00) |
  | 2,600 | 100 % (0.37) | 100 % (0.50) | 100 % (0.60) | 19 % (0.15) | 0 % (0.00) |

  (share started, mean speed m/s). **0.35 / 0.5 / 0.6 never fail; 0.8 is intermittent (98 % -> 19 %), 1.0 fails at both.** The
  eval's vmax is 0.6, so station exits at 0.6 are safe on this line; 0.8 / 1.0 starts are a defect to fix later (the 8 s
  resample makes 0 -> 1.0 steps rare; a speed ramp after a stand in training is the candidate). 2,850 / 2,870 still running.
- **Probe size**: a deterministic policy from identical starts gives the same answer on every robot, so 16 robots did not
  sample anything: the 0.00 vs 0.67 at 1.0 m/s was the policy, not chance. The start probe varies the start (yaw +-15 deg)
  and uses 64 robots: a share's 95 % interval is then at most +-0.12, so 0 of 64 against 43 of 64 cannot be chance.
- **Edge stops, second try**: +100 (15:04:42) yaw 0.4 0.30, forward 0.2 from a stand 0.21, clearance at 0.5 0.082 m, 0.5 m/s
  0.48, 0.8 m/s 0.53, all survive; fell back on 0.8 < 0.6, a mark the base (2,600) also fails (19 % start). Third mis-judged
  run. Both edge +100 probes showed no regression on a property the base holds, so **edge stops are kept** (decision mine).
- **New fall-back rule (coordinator, adopted)**: every mark is relative to the base checkpoint probed in the same job with
  the same seeds and robot counts, and used only on properties the base passes (at its last two probes where available):
  start share at 0.35 / 0.5 / 0.6 (64 robots) >= base - 0.15; yaw from a stand at +-0.8 >= 0.4 if the base has it; clearance at
  0.5 >= base - 0.015; survival >= base - 0.1; tilt + diverged <= 0.15. Not used: training speed share, 0.8 / 1.0 starts.
- **Slow commands brought forward (before the edge +250, as asked)**: r7 resumed 15:07:02 from the edge run's checkpoint
  (`eval/export/ckpts/r7_edge2_p100.pt`, about 2,950; also the base for the marks) with `runs/r7_flags_v8.txt` = v7 (edge stops)
  + `--prior_min_speed 0.05 --prior_turn_min 0.3` (`scripts/patch_prior_turn.py`, backup `train.py.pre_priorturn`): the
  discriminators now also see slow walking (0.05-0.2 m/s) and turning windows (mean |yaw rate| >= 0.3), stands still excluded.
  Flat prior: 656 s -> 853 s, speed p10 0.28 -> 0.11, p25 0.39 -> 0.22. Watcher `runs/queue23.sh` (tmux `ssr23`): base probes
  now, +100 (about 15:25) and +250 (about 15:45) with the relative marks; fall-back = the base with v7.
- **Eval r7_002250 standard strict**: home 44/50 (6 stalls at one waypoint), street 0/50 (45 stalls short of two waypoints,
  5 falls), corridor 0/50. The stall is the slow approach command; `floor40` (my spec) is queued by the eval monitor. This is
  the largest loss today; the policy fix is the prior / command step now running, and the 0.2 m/s / 0.4 rad/s probes are its
  measure.

## First apparatus result and the slow turn (2026-11-12 15:30 UTC)

- Eval r7_002250, our apparatus, standard command: strict 0/50 (49 stalls, 1 fall), disc 0/50 (47 stalls, 3 falls). Button 1
  counted in 42/50 (over the front 4x4, station held). 40 stalls heading to "door side" after "back over the front 4x4",
  8 at button 1. Eval's hypothesis: the turn on the spot stalls under 0.4 rad/s.
- **Slow-yaw probe** (`scripts/probe_turn2.py`, from a stand, 8 robots, 8 s, mean of the last 4 s; drift = pelvis travel over
  those 4 s), base r7_edge2_p100 (about 2,950, edge stops):

  | command | yaw rate reached | drift |
  | --- | --- | --- |
  | wz +0.2 / -0.2 | 0.033 / -0.015 | 0.04 / 0.03 m |
  | wz +0.3 / -0.3 | 0.164 / -0.001 | 0.05 / 0.01 m |
  | wz +0.4 / -0.4 | 0.299 / -0.380 | 0.05 / 0.06 m |
  | wz +1.0 / -1.0 | 0.970 / -0.958 | 0.04 / 0.06 m |
  | (0.3 m/s, +-1.0) | 1.04 / -1.02 at 0.31 / 0.32 m/s | |
  | (0.4 m/s, 0.5) | 0.53 at 0.41 m/s | |

- **What rate the harness needs for a 180 deg turn** (`eval/harness.py` command(), moving waypoints): wz = clip(1.5 x err,
  +-wmax) with wmax 1.0 for SSR, and zero forward speed while |err| > 60 deg. So from 180 deg down to 38 deg (1.5 x err >= 1.0)
  the command is the cap 1.0 rad/s, and the forward command starts at 60 deg, where wz is still 1.0. **The in-place part of a
  180 deg (or 90 deg) turn only ever asks for 1.0 rad/s**, which the policy tracks (0.96-0.97). The slow rates appear only
  below 15 deg of error (1.5 x err < 0.4), by then the robot is walking (forward speed min(dist, 0.6) x cos(err) >= 0.3) and the
  combined commands are tracked (0.3 m/s + 1.0 rad/s: 1.04 rad/s at 0.31 m/s). At a station (holding) the heading command is
  1.5 x err with a 0.15 rad dead band, but the hold needs only the disc, not the heading.
  So the "turn stalls under 0.4 rad/s" mechanism does not fit "back over the front 4x4" -> "door side" (a 90 deg left turn,
  0.91 m away: in-place turn at 1.0 rad/s, then walking from 60 deg). Something else stalls there: candidates are the place
  (after re-crossing the 4x4 the robot stands near the door wall / the 4x4, and the path to door side runs beside them;
  training ends episodes on side-obstacle contact, so the policy may refuse to walk towards a wall it sees in depth), or the
  forward command at 60 deg (0.6 x cos 60 = 0.30 m/s with 1.0 rad/s, at the edge of what it starts from). Asked the eval
  monitor (via the coordinator) for the stall traces: commanded (vx, wz), heading error, distance, pose and a depth frame /
  render at the stall. The 8 stalls at button 1 are in the disc's heading turn or the station exit (to be seen in the traces).
- The slow-yaw dead zone (0.2 and 0.3 rad/s, one side) is still real and is what the prior step aims at; the +100 / +250
  probes of the prior run add probe_turn2 (0.2 / 0.3 / 0.4 / 1.0 and combined) against this base.
- **Prior +100 (3,050, 15:28): kept, no fall-back.** Against base 2,950 (edge stops): starts at 0.35 / 0.5 / 0.6 100 % / 100 % /
  100 % (same), mean 0.36 / 0.52 / 0.61; yaw from a stand +-0.8 0.76 / -0.76 (base 0.78 / -0.77); clearance at 0.5 **0.095 m**
  (base 0.082); survival 100 %. Slow yaw (probe_turn2): +-0.2 0.004 / 0.001 (base 0.033 / -0.015), +-0.3 0.13 / -0.17 (base
  0.16 / -0.00: now both sides), +-0.4 0.28 / -0.30 (base 0.30 / -0.38), +-1.0 0.97 / -0.96, combined 0.3 m/s + 1.0 rad/s tracked.
  Mixed at 100 iterations; the +250 probe (about 15:50 UTC) decides. Levels at 3,160: flat 13.2, beam 13.7, platform 6.2,
  stairs 4.7, ramp 5.1; edge stops 10 started / 14 done per interval.

## Refusal near walls and posts (2026-11-12 15:45 UTC)

- Eval traces (coordinator): five stalls, none a failed turn. Door side (seeds 1000-1002): turned, 0.25-0.37 m from the
  waypoint, commanded 0.25-0.37 m/s, speed 0.00-0.03, door-frame posts 0.40 m ahead-left; under floor40 seed 1000 still
  stands. Button 1 seed 1007: 1 cm outside the disc at 0.21 m/s (slow floor). Seed 1005: beside posts at the ramp edge,
  commanded 0.57 m/s, does not move. Depth at the stall: near surfaces fill the left of the view at 0.1-0.6 m.
- **CPU measurement** (`scripts/probe_post.py`; flat floor, obstacles only in the vision raster, so seen in depth but never
  touched: a stall is the policy's choice; 8 robots, start 1 s standing, then (0.37, 0, 0) for 8 s; robots spread +-0.14 m
  across the path; r7_002250):

  | scene | distance in 8 s | speed last 4 s | stalled |
  | --- | --- | --- | --- |
  | open floor | 3.20 m | 0.41 | 0 % |
  | post 0.12 m, 0.4 m ahead, 0.35 m left | 3.18 | 0.41 | 0 % |
  | door 0.9 m wide, 0.6 m ahead | 3.17 | 0.41 | 0 % |
  | posts both sides 0.45 m, every 1.2 m | 3.19 | 0.41 | 0 % |
  | wall along the left at 0.35 m | 2.28 | 0.31 | 25 % |
  | post 0.25 m ahead, 0.20-0.32 m left | 2.37 | 0.31 | 25 % |
  | wall stub 0.3 m ahead from 0.25 m left | 2.37 | 0.31 | 25 % |
  | wall along the left at 0.25 m | 1.23 | 0.18 | 50 % |

  (at 0.6 m/s: open / post 0.4 / door / posts both: no stalls; wall 0.35: 25 %). So the policy walks past posts and through
  a door on its line, but **stops when a surface is within about 0.25 m of its path** (the robots nearest the wall stall),
  from vision alone. The apparatus stalls are this: after turning, the path to "door side" passes the door-frame posts
  close on the left.
- **Why (training)**: (1) side-obstacle contact ends the episode, and "contact" is any of 12 limb centres within 0.06 m of an
  obstacle cell (`_touches_obstacle(radius=0.06)`); side_obstacle was 17-21 % of all episode ends in r6b, so near a wall the
  learned value is "about to die"; standing still under a slow command costs little. (2) The terrain set has walls, posts
  and door frames, but always symmetric around a lane the robot walks down the middle of (lane half width 0.48-1.6 m,
  start spread lane - 0.32 m), on its heading; passing a surface at 0.2-0.3 m, at an angle, after a turn, never happens.
  (3) The omnidirectional / slow / turn commands are given only on the open flat tile (level 0 has no side obstacles), so
  slow commands near walls are never trained. Not in the paper as such (its tiles have no side walls as far as I read).
- **Change (decision mine), next after the prior step's +250**: `--side_soft 2.0` (`scripts/patch_side_soft.py`, backups
  `*.pre_sidesoft`): only a limb centre inside an obstacle cell ends the episode; the 0.06 m band costs 2.0 per step
  (`side_graze` term). CPU check (32 robots, flat level 15, lane half width 0.56, r7_prior_p100, 8 s): side_obstacle ends 3 ->
  2, graze penalty paid on 17 robot-steps. Walls exist only in the vision raster (no physics), so the overlap termination
  stays: without it a robot could walk through a wall. queue24 (tmux `ssr24`) starts it when queue23 gives its +250 verdict,
  from r7's last.pt, with the relative marks plus the obstacle probe on base and candidate (stall share near walls is the
  success measure).
- **Then (larger, CPU work now)**: the extended terrain map (`/mnt/filesystem-g4/data/terrains/extended_v1/`: tight_spots,
  homes, street_front, log_corridors, with routes) and slow / turn commands on obstacle tiles. SSR uses none of the user's
  extended map yet (loader not written) and none of the apparatus parts directly (its tiles are the paper-style generator;
  beams 3-12 cm stand in for the 4x4s). Plan: route-following commands (the harness law) on those tiles for a share of
  robots; loader and a CPU test before any GPU time.
- **Termination -> penalty near narrow passages**: yes for the margin (the change above); full removal no, because the walls
  are vision-only in training. The physical fix would be walls as MuJoCo boxes (as the eval has), then contact can be a
  penalty everywhere; that comes with the extended map (its boxes json).
- Order: prior step (+250 now) -> soft side margin -> extended map / slow commands near walls. The door is waypoint 8 of 22,
  so this outranks edge stops (kept) and the 0.8 / 1.0 m/s starts.

## Command step-downs; prior kept; side margin running (2026-11-12 15:52 UTC)

- **Prior step kept**: queue23 +250 (15:45:39) passed the relative marks: starts at 0.35 / 0.5 / 0.6 100 %, yaw +-0.8 0.69,
  clearance at 0.5 0.089 m, survival 100 %. Flags v8 are now the line.
- **Side margin running**: queue24 resumed r7 at 3,250 (15:47:47) with v8 + `--side_soft 2.0`; base snapshot
  `eval/export/ckpts/r7_pre_side.pt`; +100 about 16:10, +250 about 16:30 UTC (relative marks + obstacle probe).
- **Step-down probe** (`scripts/probe_step.py`, CPU, open flat floor, 16 robots, 1 s stand, 0.6 m/s for 3 s, then v2 for 8 s;
  walking = mean vx over the last 4 s >= 0.5 v2):

  | step 0.6 -> | r7_002250: walking / vx last 4 s / distance after step | r7 3,250 (prior kept) |
  | --- | --- | --- |
  | 0.45 | 100 % / 0.49 / 3.93 m | 100 % / 0.47 / 3.76 m |
  | 0.40 | 100 % / 0.44 / 3.57 m | 100 % / 0.42 / 3.36 m |
  | 0.35 | 100 % / 0.40 / 3.20 m | 100 % / 0.37 / 3.00 m |
  | 0.30 | 100 % / 0.35 / 2.82 m | 100 % / 0.33 / 2.65 m |

  On open floor a falling command does not make it stop: it keeps walking at about the new command after every step down
  to 0.30. So the floor40 corridor stop (0.40 m/s, standing 0.215 m from the final station for 45 s) is not a step-down
  defect. A corridor has walls on both sides (and the final station sits near its end): it matches the measured refusal
  near surfaces (a wall at 0.25-0.35 m stops 25-50 % of robots), not the command.
- **Adapter rule**: floor40 is right for open floor (keeps walking at 0.40, its stop from 0.40 is short); `keep` (0.6 until the
  disc) would only overshoot more (flat stop-stand 41/50). No adapter rule fixes a refusal next to walls; that is the
  training change now running (side margin) and next (extended map with slow commands near walls). Suggest the eval logs
  wall distance at every stall so refusal and command stops can be told apart.
- **Resample share**: tile speeds are U(0, 1.2) (slopes U(-1, 1)) every 8 s, so about 58 % of resamples start from >= 0.5 m/s
  and 12.5 % land in 0.3-0.45: about 7 % of tile resamples are a fast -> 0.3-0.45 step (plus the omni robots' slow commands
  0.1-0.4), all on tiles, but slow commands next to walls stay rare (omni robots are on the open tile).
- Eval note: the front-left object at the door-side stalls is the apparatus's leaning board, modelled as a solid wedge up
  to 0.48 m (199 of 1,296 depth rays); a TEST with it as a tilted box is queued by the eval monitor.

## Extended terrain map in training (user decision 2026-11-12; CPU work 15:55-16:10 UTC)

- **Written** (`scripts/patch_ext.py`, backups `*.pre_ext`; default off):
  - `ssr/terrain.py add_extended()`: extended_v1 tiles (8 x 8 m, 2 cm) resampled to 2.5 cm and placed to the right of the
    paper-style grid (column = type x variant, row = level; 10 rows = the existing 80 m). Physics floor = the export's floor
    raster plus low lifted pieces (top within 0.30 m of the floor and not tall, e.g. a slanted 4x4 becomes solid); vision =
    floor + 1.3 m on `obstacle_mask` (walls, rails, posts; door headers are not in the mask, so doors stay open in depth).
  - `ssr/env.py` route robots: a share `--ext_frac` of the non-omni tile robots (after the obstacle ramp) get a checked route
    of a tile of their type at their level; start pose = route start (+-5 cm, +-10 deg); command = the eval harness's law
    (yaw 1.5 x error capped 1.0, no forward speed over 60 deg, forward clip(distance, 0.15, vmax) x cos(error), vmax U(0.4, 1.0)),
    next waypoint inside 0.25 m; at the target zero command, success after 1 s inside 0.20 m (so slow approaches, stops,
    turns in place and starts next to walls are all trained). Ends: own failures, touching a tall obstacle (overlap; with
    `--side_soft` the 0.06 m band is a penalty), leaving the tile, time limit. Curriculum per robot over the 10 levels (up on
    success, down under half the route). Discriminator: stairs tiles -> stairs, others -> flat. Log field `ext` (robots and
    level per type, successes / failures per type).
  - `ssr/train.py`: `--ext_frac F --ext_types ... --ext_variants K`.
- **Walls as MuJoCo boxes: not in this step.** 100 tiles hold about 3,000 obstacle boxes not in the floor raster; as static
  MuJoCo geoms they would be checked against every robot's collision geoms in every one of 8,192 worlds (MuJoCo Warp has one
  model for all worlds). Tall things are handled as on the existing tiles (vision raster, overlap ends the episode, margin
  penalty). The physical alternative (per-world box slots set at reset, about 60 per robot's tile) is the follow-up if
  contact matters in the results.
- **Start set**: tight_spots, log_corridors, homes, street_front, stairs; 2 variants per level, all 10 levels: 100 tiles,
  267 routes (only tiles with at least one checked route). Not yet: single_beams, beam_fields, blocks, wedges (the paper-style
  beam / ramp / platform tiles cover those categories for now). No apparatus layout or route is in the set (held-out rule).
- **Share**: `--ext_frac 0.3` of non-omni tile robots: with 30 % omni robots that is 21 % of all robots on extended routes,
  49 % on the current paper-style tiles, 30 % omni on the open tile.
- **CPU test** (`/tmp/ext_test.py`, 20 robots, level 0, r7 3,250 policy deterministic, 10 s, side_soft 2.0): raster
  5,041 x 3,361 (was 1,761 x 3,361), build 5.5 s, env 8 s; 18 robots on routes; commands as expected (turn in place 1.0 at the
  turn-around starts, 0.98 m/s forward on a straight). Mixed types: successes log_corridors 4, homes 1, stairs 2; failures
  tight_spots 5, homes 6, street_front 6, stairs 13; ends: side_obstacle 15, diverged 9, link force 4, time limit 6.
  Stairs only (18 robots): 13 successes, 20 failures, diverged 6. The divergences (simulator blow-ups, reset safely) are
  more than on the paper-style tiles (1.8 % of ends in r6b); watched in the run, stairs drop to levels 0-4 if they persist.
- **GPU start**: queue25 (tmux `ssr25`) starts it automatically when queue24 gives the side margin's +250 verdict (about
  16:30 UTC), from r7's last.pt, flags = the line + `--ext_frac 0.3` (`runs/r7_flags_v10.txt`); relative marks, obstacle probe,
  ext stats at +100 / +250, and a fall-back if samples/s drop under 60 % of the base run's.
- Side-margin base (r7 3,250) obstacle probe: wall at 0.25 m left 100 % stalled (r7_002250: 50 %), stub close 12 %, post close
  25 %: the refusal grew with the prior / edge steps; the side margin and the extended map are the answers.
- Eval note (coordinator): the corridor stop was not walls: the ladder course's edge 1.1 m ahead reads as a void in depth
  (119 of 1,296 pixels 'far'); a test with ground beyond the edge is queued.
- Non-stairs extended types only (tight_spots, log_corridors, homes, street_front; 20 robots, 10 s): ends side_obstacle 10,
  diverged 2, tilt 1, time limit 6, reached the end 5. So the divergences are mostly on the extended stairs (6 of 9 in the mixed
  test); the side-obstacle ends (robots touching walls on the narrow routes) are the main failure, as expected for a policy
  that has never walked them.

## 16:25-17:25 UTC (node clock): side margin kept, extended map run 1 discarded (placement bug), run 2 started

- **Side margin kept** (queue24 +250 at 16:25:38: starts 100 %, yaw +-0.8 0.78, clearance 0.096 m, survival 100 %). Flags v9 =
  v8 + `--side_soft 2.0`.
- **Extended map run 1**: queue25 resumed r7 at 3,550 (16:28:25) with v10 = v9 + `--ext_frac 0.3`; +100 (16:49) and +250 (17:06)
  passed the relative marks (+250: starts 100 %, yaw 0.80, clearance 0.133 m, samples/s about 31k, same as before). ext stats
  at +250: log_corridors 10 successes / 0 failures (level 1.96), stairs 9 / 5 (0.65), tight_spots 0 / 13, homes 0 / 12,
  street_front 0 / 20 (levels 0.0-0.2). Levels at 3,910: flat 10.5, beam 12.4, platform 6.1, stairs 7.5, **ramp 5.5**.
- **What failed (found 17:15-17:20 by replaying route ends on the CPU)**: tight-spot and street-front robots ended at t = 0 on
  side_obstacle and diverged. Cause: a bug of mine in `terrain.add_extended`: the tile centre in world coordinates was set to
  cx - BORDER, but the raster index _i(BORDER + cx) is world x = cx, so **every route was 2 m off its tile** (in -x and -y).
  Robots were spawned and driven into walls and kerbs; the open log corridors (and partly stairs) still worked, which is why
  only those had successes. Fixed (`gcx, gcy = cx, cy`; backup `terrain.py.pre_extcentre`). Also added (`patch_extfilter.py`):
  route starts need 0.35 m obstacle clearance (the X2's arms are 0.28 m from its centre line) and level ground (floor range
  <= 3 cm within 0.27 m): 170 of 267 routes kept, all 5 types x 10 levels still present. Geometric check after the fix: 1 of 170
  starts has an obstacle cell within 0.30 m (mask grown by one cell), median clearance > 0.6 m.
- **Decision (mine): run 1 discarded, restart from 3,550** (r7_pre_ext, before any extended training), because 460 iterations
  had taught "routes run into walls" and the wall refusal got worse (below). queue26 (tmux `ssr26`): r7 stopped at 4,010
  (17:21:00), resumed at 3,550 (17:21:37) with **the same flags as v10** (`runs/r7_flags_v10.txt`; the fix is in the code, not
  in the flags); relative marks against r7_pre_ext. +100 about 17:45, +250 about 18:05 UTC.
- **Wall probe, 64 robots** (`scripts/probe_wall.py`: one wall, vision only, along the left at distance d from the path; start
  on the line +-3 cm, yaw +-10 deg; 0.37 m/s for 8 s; stalled = mean speed of the last 4 s < 0.1 m/s), share stalled:

  | d (m) | 0.20 | 0.25 | 0.30 | 0.35 | 0.45 | no wall |
  | --- | --- | --- | --- | --- | --- | --- |
  | 3,550 (side margin, before ext) | 78 % | 66 % | 55 % | 55 % | 39 % | 0 % |
  | 3,800 (ext run 1, misplaced routes) | 97 % | 86 % | 75 % | 44 % | 33 % | - |

  The refusal reaches out to 0.45 m and grew in ext run 1 under 0.35 m (consistent with the bug). This probe is now the
  measure for ext run 2.
- **Route robots and surfaces under 0.3 m (question 2)**: in run 1 they did not pass any: they were placed 2 m off their tiles.
  Replay of run 2's geometry with the 3,800 policy is running on the CPU (tight_spots, street_front, homes ends).
- **Exports**: `eval/export/r7_003550/` (side margin, before any extended-map training) and `eval/export/r7_003800/` (ext run 1:
  passed its marks, but trained 250 iterations on misplaced routes; wall refusal worse). **Score r7_003550** with
  `floor40` (in its card, field `eval_command`; standard as the headline). 3,800 should not be scored. The next export is
  ext run 2's +250 (about 18:05) if it passes.
- **Leg rule (user)**: noted: full apparatus routes only after five station legs pass 45/50 each.
- **Sole spheres in height-field cells (eval's artefact)**: SSR's sole spheres are 2 cm radius (`SSR_SOLE_R`, default 0.02) on a
  2.5 cm height field, the same class as the eval's 2 cm spheres in 2 cm cells. The diverged ends on the extended stairs and
  the flat-floor force spikes (foot_force_max 600-760 N in the logs) may be that. CPU check planned: a standing and a walking
  robot on flat and stair rasters, per-sphere contact count, penetration depth and force, with 2 cm and 3 cm sole spheres.
- **Route-end replay with the fixed placement** (`/tmp/ext_ends.py`; r7 3,800 policy, deterministic, train-mode env, 24 robots
  of one type, level 0, 15 s, side_soft 2.0):
  - tight_spots: 28 ends, none at t = 0 (was 16 with the bug); 24 side_obstacle, 1 reached the target; mean route
    progress at the end 0.52.
  - street_front: 22 ends, none at t = 0; 13 side_obstacle, 4 diverged, 1 link force, 4 reached the target; progress 0.48.
  - homes: 17 ends, 3 at t = 0; 15 side_obstacle, 2 diverged; progress 0.25.

  So with the routes on their tiles the robots get about half-way, then touch a wall (they do pass surfaces under 0.3 m and
  touch them), which is what run 2 has to learn. The 3 homes ends at t = 0 are to be looked at (a start next to a low
  obstacle that is not in the mask, or a door leaf).

## 18:22-18:45 UTC (node clock): wall-refusal diagnosis, ext run 2 stopped, run 3 with X2-passable routes; drop test (fresh agent)

**What went wrong.** (1) The queue's wall probe (`wall_left_0.25`) measures an impassable wall: X2 limb centres (`side_ids`)
reach 0.343 m left of the base standing and 0.29-0.39 m walking, so a wall face 0.25 m from the centre line is 9 cm *inside*
the arms (and the probe's 8 robots start up to 0.14 m nearer); overlap share at 0.25 m was 0.67-1.0. Refusing it is correct
under the training rule (overlap ends the episode). Not a measure of the door. (2) The real defect: refusal at *passable*
distances grew during ext run 2. (3) Cause: the extended routes of tight_spots and homes cannot be passed by the X2: least
clearance along the route (`min_clearance_along_route_m`, README of extended_v1) is under 0.34 m on 80 % of their routes and
under 0.40 m on all (homes level 0: 0.20-0.38 m; tight_spots 0.36 at L0 down to 0.22 at L9). The 0.35 m start filter only
checked starts. Every such route ends in a limb overlap (tile stats at 3,800: tight 3/7, homes 2/11), so the run learned
"moving near walls ends the episode". Street_front (0.40-0.46) and log_corridors (0.44-0.68) fit; stairs 97 % >= 0.40.

**Wall probe v2** (`code/scripts/probe_wall2.py` = `eval/diag_1812/wall_diag.py`; 32 robots, start +-3 cm / +-10 deg, 0.37 m/s,
stalled = mean vx last 4 s < 0.1; results `eval/diag_1812/wall_*.json`). Stalled share by wall distance (centre line to face):

| d (m) | 0.25 | 0.30 | 0.35 | 0.40 | 0.45 | 0.60 | none |
| --- | --- | --- | --- | --- | --- | --- | --- |
| clearance to limb centre (standing 0.343) | -9 cm | -4 cm | +1 cm | +6 cm | +11 cm | +26 cm | |
| 3,550 r7_pre_ext | 0.63 | 0.56 | 0.31 | 0.38 | 0.25 | 0.06 | 0 |
| 3,800 ext2 +250 | 0.94 | 0.91 | 0.66 | 0.56 | 0.28 | 0.19 | 0 |
| 4,100 ext2 | 1.00 | 1.00 | 0.84 | 0.84 | 0.56 | 0.00 | 0 |

At 0.45 m neither the graze band nor overlap is touched (band share 0.00-0.03), yet 4,100 stalls 56 %: learned refusal, not
the penalty. The apparatus doorway is 0.87 m clear (0.435 m half width; EVAL.md), so 0.40-0.45 is the door regime.
Other candidates checked: side_soft band: standing robots at 0.40 m sit in the band too (band share 0.73 for stalled robots),
so the band does not pay for standing; walls vision-only: still true, not the cause measured here; share/level of map tiles:
moot while 2 of 5 types are infeasible.

**Decision (mine, 18:36): ext run 2 stopped at 4,220 (snapshot `eval/export/ckpts/r7_ext2_last.pt`), restart from r7_pre_ext
(3,550)** with `runs/r7_flags_v11.txt` = v10 + `--ext_types log_corridors,street_front,stairs --ext_min_clear 0.39`
(`scripts/patch_extclear.py`, backups `*.pre_extclear`; 60 tiles, 118 routes). CPU smoke (3,550 policy, 18 robots, level 0,
10 s): log_corridors 9 succ / 3 fail, street_front 5 / 3, ends: reached 14, side_obstacle 4. queue27 (tmux `ssr27`, log
`runs/queue27.log`) restarted r7 at 18:38:56; same relative marks and fall-backs as queue26, plus wall probe v2 at 0.40 / 0.45 at
+100 and +250 (tags r7_ext3_p100/p250, `eval/post/<tag>_wall.json`); **at +250 it falls back to v9 (no ext) if the mean stalled
share at 0.40/0.45 exceeds base 0.31 + 0.15**. tight_spots / homes stay out until walls are physical or the X2 can be shown to
pass them (they were designed narrower than the X2).

**Drop test, SSR's simulator** (MuJoCo Warp CPU, `eval/diag_1812/drop_test.py`, `drop_*.json`): 10 kg foot with the X2's 18
terrain spheres (12 sole r 0.02, 6 guards r 0.022), 2.5 cm height field, training options (dt 0.005, implicitfast, default
solref), dropped at vz 1-4.5 m/s with 1 m/s sideways, 150 N sideways pull 0.10-0.30 s, two cell offsets:

| vz (m/s) | 1 | 2 | 3 | 4 | 4.5 |
| --- | --- | --- | --- | --- | --- |
| deepest (mm), default contact | 6.3 | 14.0 | 21.8 | 29.3-29.6 | 33.2 |
| time deeper than 1 cm (s) | 0 | 0.03 | 0.05-0.055 | 0.06 | 0.065-0.07 |
| horizontal-normal contacts (max) | 0-2 | 11 | 11 | 11 | 11 |
| moved by the pull in 0.2 s (m) | 0.10 | 0.09-0.10 | 0.09-0.11 | 0.09-0.10 | 0.11 |

No trap: release within 0.07 s at every speed and the pull moves the foot as on free ground (eval's caged foot: 0.00 m, never
released). Sink is deep at high speed (soft default solref 0.02); terrain solref 0.01 with priority (0.005 clamps to 2 dt =
0.01) halves it (20.7 mm at 4.5 m/s, 0.015 s over 1 cm). No change to training physics.

**Export to score**: keep `r7_003550` (lowest refusal at 0.40/0.45 of all measured; 3,800 and later ext2 must not replace it).
Next candidate: r7_ext3_p250 if its wall probe beats 3,550.

## 18:50-19:20 UTC: physical walls graded by body part (user rule), my own regression, run 3 resumed

**User rule (coordinator 18:50)**: shoulders / hips / feet on walls bad, forearm brushes OK to some extent; sidestep in tight spots.
**Built (`code/scripts/patch_extphys.py` + `patch_extphys_fix2.py`, backups `*.pre_extphys`, flag `--ext_phys K`)**:
- K = 24 static wall boxes per world (batched geom_pos/quat/size/aabb/rbound; world-body geoms are not re-posed by kinematics, so
  the pose is also written into `dd.geom_xpos/xmat` at reset). Per route: the 24 tall boxes (role 2, headers > 1.6 m dropped) and
  tall meshes (as their AABB) nearest the route polyline; 803 far boxes dropped over 170 routes. Tiles have 19.5 boxes on average, 50 at most.
- Contact by body: forearm / hand (elbow, wrist bodies): free up to 40 N and 0.5 s continuous, beyond that -0.5 per step, never an
  episode end. Any other body (shoulders, torso, pelvis, hips, legs, feet): -2 x (1 + F/100 N, capped at 5) per step; episode end
  only above 400 N (cause `wall_hit`) or by the usual falls. Route robots with walls: no limb-overlap end, no graze band; base inside
  an obstacle cell still ends it. Wall contacts are excluded from foot load and the 80 N link-force end.
- Sidestep law: on route segments with clearance < 0.39 m (148 segments), turn 90 deg to the path (within 30 deg), then vx/vy toward
  the waypoint capped 0.3 m/s. Route law had no lateral before; omni robots already train vy +-0.4.
- Prior: the user's sideways Pico takes are in `tp_t4x4u_y20.pkl` (10 clips with lateral share > 0.5, about 35 s) and so in the
  *beam* discriminator only; the flat set has none. tight_spots / homes route robots now use the beam discriminator.
- CPU smoke (3,550 policy, 20 robots, 10 s, all 5 types, no clearance filter): tight_spots L0 9 succ / 0 fail (vision-only: 0/13),
  homes L0 3/8, L4 tight 4/4, homes 2/8; log_corridors 6/2, street_front 2/3, stairs 0/8; wall contacts graded (L0: body 41
  robot-steps, arm 73 of which 58 free; max body 387 N), wall_hit 4 at L4. CPU sim time about +25 %; GPU cost unknown (fall-back
  on samples/s < 60 %).

**What went wrong (mine)**: patch_extphys (18:56) cut `SSREnv`'s `nconmax` default from 150 to 15 (anchor text truncated in my
read). Probes then overflowed the contact pool (robots fell), queue27's +100 verdict failed at 19:00 and it fell back to v9 at 3,550;
that v9 run (19:01-19:11) also had nconmax 15: discarded. Fixed 19:11; run 3 resumed from r7_ext3_p100 (3,650, trained with the
correct code) by queue27b at 19:12. The first two physical-wall smoke tests were invalid for the same reason.

**Door-like probes** (stalled share, 0.37 m/s): wall at 0.40 / 0.45 m (32 robots): 3,550 0.38 / 0.25; run 3 +100 (3,650) **0.13 /
0.00**. 0.9 m door (probe_post door_0.9, 8 robots): 3,550 0/8, 3,650 0/8.

## 2026-11-12 20:30-21:00 UTC: station legs of r7_003550 diagnosed, v13 restart prepared (fresh SSR agent)

**Legs (eval, ours, solid floor, strict, 50 per leg): 28 / 5 / 23 / 34 / 0.** Read from
`eval/out/ssr/r7_3550/legs_ours_solid/leg*/` (trajectories `traj/<seed>.npz`, events.csv); script
`<scratchpad>/vel.py` bins body-frame speed by command.
- **Leg 5 "left the course" (34/50) is not the stairs.** All 34 end in the first section ("ramps, front side"), 3.5-16 s in. The leg
  starts at button 4 (4.11, 0.36), facing the wall (-90 deg), on the side-sloped wedge (cross slope about 16 deg: floor 0.02 m at
  y 0.1 to 0.16 m at y 0.6). Command: turn in place (yaw 1.0) toward "ramps, front side" (5.44, 0.61). In all 34 the arm or trunk
  hits the button wall at 0.6-2.1 s while turning (trunk 885 N in seed 1000); the turn stalls 25-45 deg short (yaw rate with
  |cmd| >= 0.9 on the ramps legs: 0.13-0.19 rad/s; leg 1 on flat: 1.07); then, commanded forward 0.4 m/s with yaw 1.0, the robot
  walks **backward** at 0.6-0.9 m/s (body frame) for 1 m and leaves the 1.0 m route band behind its start. Never seen on flat
  (leg 1: no backward samples); leg 3 (also ramps) has 19 % of saturated-turn samples backward.
- **Leg 1 "time limit" (19/50): wall approach at button 1, not a slow walk.** All 19 cross the front 4x4 by about 26 s (as the 28
  successes, median 28.6 s) and then stand 0.200-0.202 m from button 1 for 35 s: pelvis 0.375-0.385 m from the button-wall box,
  harness command flipping between stand (inside 0.20) and 0.2 m/s forward, the robot not moving. The button is 0.185 m from the
  wall face, so the reach radius needs the pelvis within about 0.385 m of the wall; the policy stops at 0.375-0.385. Same refusal
  the wall probe measures (3,550: stalled 0.375 at 0.40 m).
- **Leg 2** (5/50; 25 stalls, 19 falls) is the turn-around at the front 4x4: saturated-turn yaw rate 0.82-0.87 but forward speed
  0.02 m/s with vx >= 0.3 and yaw < 0.4, i.e. it refuses to step back over the 4x4 after turning.
- **Training gap behind all three**: no training situation has a stop target near a wall (ext routes keep >= 0.35-0.40 m clear, the
  stop is 0.20 m short of the target), no turn in place next to a wall, no turning on side slopes (ext map had no wedges), and the
  only 4x4 crossings on the map were in tight_spots/homes.
- **Ramp height map question**: SSR uses neither `tp_ramp_hmap_x30.npz` nor `props/ramp.json`. SSR's prior is motion only (the
  discriminator sees 63-D proprio frames); its ramps are procedural slope tiles plus (from v13) the extended map's wedge tiles.
- **Newer export to score now?** No. r7_phys_p250 (4,050) wall probe: 0.40 m stalled 0.438 (3,550: 0.375), 0.45 m 0.094 (0.25); nothing
  in v12 targets the leg-5 or leg-1 cause. Next candidate for legs 1, 2, 5: r7_v13_p250. Useful baseline for v13 if eval CPU is free:
  r7_v12_last (saved by queue29 at the restart) on legs 1 and 5.

**v13 (one restart; my decision 2026-11-12 20:50, reasons above).**
1. Prior `data/amp_v4.npz` (`scripts/build_amp_v4.py`, CPU, done): amp_v2 with ramp, stairs, beam rebuilt. stairs 1,241 -> 1,834 s
   (+ tp_stairs_all_v1_1 302 s, + its mirror 302 s); beam 605 -> 858 s (2x2 now v1_1, 44 s, without `pico_20261029_211326_0_2x2`
   (asserted), + 2x2 v1_1 mirror 44 s, + tp_t4x4u_y20_mir 212 s); ramp 564 -> 703 s (+ tp_ramp_base_x30_mir 140 s). flat and
   platform unchanged. Mirrored props (`_mir_props.json`) do not apply: SSR does not replay clips on geometry.
2. Extended map (`scripts/patch_v13.py`, backups `ssr/*.pre_v13`; flags `--ext_stairs_inset --ext_prop_routes --ext_stations 1`,
   `--ext_types` + wedges, single_beams, beam_fields, blocks; `--ext_frac 0.4` from 0.3 because 9 types share it): stair tiles from
   `extended_v1_2_stairs_inset` (404 routes kept after the start checks), `extended_v1_1_prop_routes` merged (wedges 184, single
   beams 164, blocks 135, beam fields 128, log corridors 59 routes). Wedges use the ramp discriminator, single beams and beam fields
   the beam one, blocks the platform one. 180 tiles, 1,276 routes (v12: 5 types, about 250 routes).
3. **Wall stations** (new, deviation from the paper: a route-shape choice, no new reward): per tile one route whose target is
   0.35-1.5 m from a tall wall box gets (a) an approach copy with a last segment straight at the wall to a point 0.15-0.22 m from
   the face (hold radius 0.20 -> pelvis stops 0.35-0.42 m from the wall, as at buttons 1-5) and (b) a departure route starting
   there facing the wall, back along the route (turn in place next to a wall, leave along it: leg 5). Other boxes >= 0.30 m from
   the new segment, level floor. Those segments are exempt from the sidestep law. 75 pairs (tight_spots 20, homes 19, log
   corridors 11, stairs 4).
4. Smoke (CPU, `scripts/smoke_v13.py`, r7 last.pt at 4,3xx, 36 robots, 9 types, level 1, station routes forced on 8 robots):
   see `runs/smoke_v13.log`; queue29 restarts only on DONE with station lines.
- **Watcher** `runs/queue29.sh` (tmux `ssr29`, log `runs/queue29.log`, prefix QUEUE29): after the smoke, stops r7, saves
  `eval/export/ckpts/r7_v12_last.pt`, resumes r7 with `runs/r7_flags_v13.txt`; +100 / +250 probes (start, turn, gait, wall)
  tagged r7_v13_p*; falls back to v12 from r7_v12_last on failed marks, tilt+div > 0.15, samples/s < 18,000, +250 wall stall > base
  + 0.15; says "v13 kept".

**What went wrong**: leg 5 was read as a stairs failure in the brief; it is the departure from button 4 (wall + side slope). My
predecessor's v11/v12 wall work measured walls along the path only; the apparatus asks for stops facing a wall 0.38 m away, which
no probe measured and no training route asked for.

## 2026-11-12 21:30-21:52 UTC (node clock): v13 fall-back diagnosed, stop-and-stand cause, v14 restart (fresh SSR agent)

**What went wrong 1: v13 fell back at 4,390 ("tilt+div 0.156").** The gate is the mean of tilt + diverged done shares over the last 5
log lines, threshold 0.15. Evidence (`runs/r7/log.jsonl`, lines 651-669): every resume starts all episodes together, so the first
time-limit wave (+40 iterations, 4,350) spikes falls: v13 4,350 = 0.266 (diverged 0.238), and the v12 resume of the same base also
spiked (0.146 at 4,350, 0.177 at 4,390). v12's own window before the switch: 0.10 (4,250-4,300). v13 without the 4,350 line: 0.129.
CPU replay (`code/scripts/replay_div.py`, 90 robots, 10 s, base 4,300, `runs/rdiv/*.log`): diverged (|qvel| > 60) happens on the four
new prop types (wedges 2, single_beams 1-2, beam_fields 1-2, blocks 1-4 per ~90 robot-s) and on stairs (v12 too); zero on tight_spots,
homes, street_front, log_corridors. Switching off stairs inset, prop routes or stations one at a time changes nothing (diverged 5-8
vs 7). So no v13 change is broken: the fall-back = the resume-wave spike + the expected falls of an untrained policy on four new
terrain types, measured against an absolute 0.15 that v12 itself crosses. Fix: gate from +80 iterations, mean of 10 lines, relative
to the v12 base (mean of its last 20 lines) + 0.08.

**What went wrong 2: stop-and-stand 50/50 (2,250) -> 2/50 (3,550 eval, standard command, strict).** Not the stand-command share
(unchanged), not the style prior (stand windows were never in it), not the reward (stand terms unchanged). Evidence: (a) eval trials
(`eval/out/ssr/r7_3550/ladder_solid/step1/flat_stop_stand/trials.csv`): the 39 stalled trials end at x 1.80 = exactly on the 0.20 m
disc edge, command 0; (b) my eval-law replay in the training sim (`scripts/probe_approach.py`): final distance to the waypoint 2,250
0.128 m, 3,550 0.147, 4,300 0.127 (all hold there); stop distance after a 0.5 -> 0 step (`scripts/probe_stop.py`) 0.13-0.17 m for every
checkpoint 2,250-4,300 (so stopping itself did not get worse); (c) `probe_step` at 0.15 / 0.2 m/s: 2,250 0.00 / 0.18, 3,550 0.04 /
0.16, 4,300 **0.00 / 0.00** m/s. Cause: the policy treats commands under ~0.2 m/s (4,300: under 0.25) as "stand"; the harness law
slows to 0.15-0.2 m/s near the disc, so it stops at the disc edge, and each sway out of the disc gives a 0.15 m/s command it does
not follow (stalled 45 s, "1979 corrections"). 2,250 arrived ~2 cm deeper and passed; the slow-walk floor has been rising since.
Training fix in v14: `--creep_w 0.5` (pays the share of the command made good for 0.10-0.35 m/s commands; standing earns 0.8 of
lin_vel there, which is why it stands) and `--ext_hold_r 0.10` (route robots must get within 0.10 m before the zero command).

**v14 (my decision, 21:49 node clock)**: flags `runs/r7_flags_v14.txt` = v13 + `--ext_slope_turns 2 --back_scale 0.5 --drift_w 0.5
--creep_w 0.5 --ext_hold_r 0.10`. Code (backups `ssr/{terrain,env,train}.py.pre_v14`; all new flags default off):
- `terrain._slope_turns`: per tile, up to 2 routes out along an existing route to the first point >= 1 m along on a 0.5 m plane sloped
  8-22 deg, then back to the start; the route law turns the robot in place there. Built: wedges 37, street_front 22 (of 1,339 routes).
  Beside walls: v13's wall stations (approach to 0.15-0.22 m of a face, departure = turn in place at the wall) carried unchanged.
- `env`: `back_scale` x lin_vel while base vx < -0.05 m/s and no backward commands on slope tiles (user rule 2026-11-12);
  `drift_w` x planar speed under a command with |v_xy| <= 0.15 (stand and turn in place); `creep_w`; `ext_hold_r`.
- Smoke (`runs/smoke_v14.log`, 4,300 policy): slope-turn robots 4/4 turned, 0 ended before turning; drift term -0.016, creep 0.075.
- Started 21:49:59 from **4,800** (v12 last, saved `eval/export/ckpts/r7_v12b_last.pt`), 31,200 samples/s, creep and drift logged.
- Also: exported `eval/export/r7_v12_last/` (4,300; ONNX vs torch 2.1e-6) which the eval's solidq2.sh was waiting for.

### 2026-11-12 22:45 UTC: v14 +250 checks and export (agent replaced at 191k; fresh agent)
- queue30 verdict: "v14 kept" (22:28:39). +250 (5,050): start ok at 0.35/0.5/0.6, survive 1.0; wall probe stalled 0.125 (base 0.078,
  gate base+0.15 passed); approach held 1.0, **final distance 0.154 m** (base 0.139, +100 0.111, 4,300 0.127): misses the < 0.12 wish.
- `probe_step` (eval/post/r7_v14_p250_step.json): 0.15 -> walking 0.88, vx 0.103; 0.2 -> 1.0, 0.178; 0.25 -> 1.0, 0.221 m/s.
  4,300 was 0.00 / 0.00: the slow-walk floor that stalled the harness near the disc is gone (wanted >= 0.1 / 0.15: passed).
- Speed made good fell +100 -> +250 (0.327 -> 0.282, 0.474 -> 0.410, 0.579 -> 0.502 m/s; 82-84 % of command). Not the creep / drift
  terms: in runs/r7.log creep (~0.10), drift (~-0.019) and lin_vel tracking (~0.73) are flat through v14. It coincides with the
  obstacle ramp (ext levels roughly doubled: tight_spots 0.77 -> 1.69, stairs 0.52 -> 1.02): a more cautious gait on flat ground.
  Problem only if it keeps falling: watch +500; if flat-start speed < 0.8 x command, note it for the post-6,800 idea list.
- Decision (agent, 22:45): export despite the approach distance, because the cause of the 4,300 approach failure (no slow walk) is
  fixed and the eval decides. Exported `eval/export/r7_v14_p250/` (export_eval.py, chest_rs; ONNX vs torch max err 3.5e-6, mean_vx 0.49).

### 2026-11-12 23:10 UTC: diverged episodes are the MuJoCo Warp large-height-field bug (ACP finding), confirmed on CPU
- v14 builds ONE height field of 7,601 x 3,361 vertices (RES 0.025; ssr/x2.py), 3.4x ACP's failing 2,260 x 1,819 field.
- `scripts/diag_hf.py` (r7_v14_p250, 72 robots, 9 ext types, level 12, 1,500 steps, CPU): single field 29 diverged / 150 dones;
  X2_HF_CHUNK=400 (fields <= 400 x 400, shared edges) **1 / 144**. Depth logging is weak (contacts read after the reset): 8 of 29 had a
  contact > 5 cm in the step before; speed before blow-up 0.2-1.0 m/s (no fall). Out: eval/post/diag_hf_{0,400}.json.
- Training impact: in runs/r7.log diverged = 10.8 % of dones at 5,450 (20,981 so far); `self.bad` is in `term`, so the policy is
  terminated for sim blow-ups on blocks / beams / wedges, likely teaching caution on edges (and maybe the flat-speed drop).
- Code: opt-in split added to ssr/x2.py (env X2_HF_CHUNK, default 0 = old single field, so the running job and any crash restart
  are unchanged; backup eval/post/x2.py.bak_20261112). Sim cost +9 % on CPU (513 vs 472 s). Restart is the coordinator's decision.

### 2026-11-12 23:30 UTC: resumed v14 with the split height field (coordinator approved 23:12; freeze counts from 21:49)
- queue31 (`runs/queue31.sh`, tmux `ssr31`, prefix QUEUE31). Hiccup: killing queue30 closed tmux ssr30 and took its supervisor with
  it, so STOP did not stop the orphaned trainer and the first queue31 run exited ("trainer alive"). I SIGTERMed the trainer
  (as the supervisor does) and re-ran queue31. Resumed from 5,550 (saved eval/export/ckpts/r7_v14_prehf.pt) at 23:16:24.
- +100 (23:28): samples/s 30,870 (base 29,704); diverged **0.12 %** of dones (base 10.6 %); tilt 0.039 (0.039); link_force 0.050
  (0.067); levels [13.18, 11.75, 13.28, 11.66, 13.24] vs [12.91, 8.98, 13.49, 8.79, 13.90]: two type levels jumped about 2.8.
- Default: ssr/x2.py X2_HF_CHUNK now defaults to 400 (0 = old single field); export_eval.py writes hfield_chunk into adapter_card.

### 2026-11-13 01:55 UTC: v15 prepared and armed (run r8; fresh agent; user rule "SSR back to the paper")
- **Resume, not scratch** (user 2026-11-13 "continue from where we left off"; also my choice): every v15 change keeps the actor,
  encoder, critic and foothold input / output shapes (the 18 x 10 map, the only shape change, is deferred by the user), so
  `--init_from` loads enc / aux / actor / critics / foot from v14's last save; discriminators (new groups, new inputs) and all
  optimisers start fresh; robots of type 4 (was flat, now discrete) start at level 0. **Critic warm-up**: 50 iterations of value fit
  and discriminator training with the policy frozen (`--critic_warmup 50`: loss = value loss only, the actor gets no gradient,
  step size held at v14's 1.7e-4), so the style critic and the fresh discriminators settle before the policy moves.
- **Code** `models/ssr/code_v15/` (copy of code/ + simcheck patch + v15 edits; v14 keeps `code/`): terrain.py (types ramp, stairs,
  beam, platform, discrete; FREE column 5 = open area for graduates; `_discrete` blocks 0.05-0.2 m), env.py (omni_top, promote_half
  with spawn_xrel, vy_clamp off, slope backward allowed, heading table 80/30/15/15/30, q_noise, foot_base_frame in
  support_imagined, home_type for route robots), train.py (init_from, critic_warmup, disc_part_w after normalisation, base-frame
  foothold targets, per-part discriminator gradient share `disc.<g>.grad_share`, `style_by_group`, `omni_share` in the log),
  networks.py (part_w), x2.py (X2_TERRAIN_SOLREF / X2_TERRAIN_MIX; waist 0.2 in --action_scale_paper), supervise_node.py
  (heartbeat r_l, r_f, r_s, style_by_group, omni_share, critic_warmup, sim_fault_share). New: scripts/build_amp_v5.py,
  scripts/smoke_v15_env.py, tests/test_equivariance.py. Flags with a comment each: `runs/r7_flags_v15.txt`.
- **Prior amp_v5** (Table 11; flat forward 35 % + turn 15 % + traversal 50 %, slope 30/5/15/50; no BONES-SEED). Code build from
  the v1 reference (`data/amp_v5_smoke.npz`, NOT for training; the real one waits for paper_ref_x2_v2.pkl and must match these
  per-source sizes within 1.5 s): ramp 261.3 = fwd 78.4 + back 0 + turn 52.3 + ramp (Pico style 27.3, acp_tp_ramp 51.7,
  tp_ramp_base_mir 51.7); stairs 295.7 = fwd 103.4 + turn 44.4 + paper stairs up 23.2 + down 15.9 + Pico style stairs 60.7 +
  tp_stairs_all_v1_1 48.1; discrete 136.0 = fwd 47.6 + turn 20.4 + Pico 2x2 44.3 + Pico style logs 23.7; beam 174.5 = fwd 61.1 +
  turn 26.2 + Pico style logs 34.7 + tp_t4x4u 52.6; platform 181.6 = fwd 63.6 + turn 27.2 + paper step onto 7.9 + step off 7.7 +
  tp_stairs_x10 43.8 + tp_stairs_all_v1_1 31.3. Flat forward / turn pools: paper 104.2 / 89.2 s + Pico 21.6 / 12.3 s (reused
  across groups as the paper's clips are). **Gaps**: no backward walking in any source (slope's 5 % went to turning); no side
  walk (Table 11 composition; Pico side_walk 43.9 s unused); open-floor turning in Pico thin (12.3 s); paper block clips lose 4
  frames each to the velocity stencil (7.7 / 7.9 of 9.4 / 9.3 s).
- **Checks** (`runs/v15_checks/`): equivariance on r7 6,850: encoder z 3.8e-7, h 9.7e-7 relative (exact); actor mirror error
  0.085 of the mean action (tolerance 0.10, mine); involutions 0; mirror tests 4/4; env smoke 17/17 (`env_smoke.log`); 2-iteration
  CPU train with init_from + warm-up ran. **Preflight** (reports `ssr_v15_*.json`): v14 drop 4 fails / 396 mm, rollouts 8 tile fails /
  180 mm; v15 without contact fix 8 / 132 / 10 / 180; v15 solmix (terrain solref 0.005 or 0.008, both clamped by MuJoCo to 2 x
  timestep = 0.01, identical) 4 / 127 / 3 / 120: not a pass, no worse than v14 on all four (interim rule). Remaining: one ext-map
  edge site (103.45, 16.0) holds the foot 0.8 s (v14's worst 0.36 s at other sites, not comparable: the ext map moved 8 m), elbow /
  hand guards 66-120 mm at chunk borders: owners FINDINGS B11 (boxes for edges, smaller chunks). Priority 1 not used: MuJoCo would
  take the terrain's friction (0.2) for every contact and break the per-world friction randomisation; solmix 1e4 gives the
  terrain's solref with max-combined friction.
- **Armed**: queue32 (tmux `ssr32`, `runs/queue32.log`): gate 1 passed 01:53; gate 2 waits for paper_ref_x2_v2.pkl, builds amp_v5,
  checks sizes; gate 3 = next r7 save at or after 6,800 (r7 was at 6,970 at 01:54, saves every 50). Then stops r7 (queue31 left
  alone: it exits when r7 stops), saves `eval/export/ckpts/r7_v14_final.pt`, starts r8 from code_v15 with X2_HF_CHUNK=400
  X2_TERRAIN_SOLREF=0.005. Fallback to v14 (resume r7, code/, v14 flags): crash, +60 samples/s < 0.8 x v14 or tilt > v14 + 0.05,
  +150 ep_len < 0.7 x or vx_ratio < 0.75 x v14. Trial exports `eval/export/r8_v15_p100` (50 policy iterations after the warm-up,
  more than one 20 s episode) and `_p300`.

### What went wrong this turn (2026-11-13)
- First prior build: discrete got no logs windows (beam was built first and took them all) and stairs / platform paper clips were
  classed from `paper_ref_terrain_clips.json` (1 stairs-down clip) instead of the categories table (9): fixed (explicit group order,
  categories by time range); flat-forward rule was too strict (75 s pool): now the reference labeler's walking rule minus turns.
- The simcheck adapter hard-codes `models/ssr/code`: preflight runs from a private copy `tracks/ssr/simcheck_v15` with SSR_CODE
  (shared tool untouched; owner of the adapter could add SSR_CODE upstream).
- No `tests/test_equivariance.py` existed; written (checkpoint-based encoder / actor / involution checks).

### 2026-11-13 02:20 UTC: v15 export bug fixed, v15 fell back at +150 (fresh agent, replaced the 282k one)
- **What went wrong 1 (export)**: `r8_v15_p100` export failed `IndexError: index 5 ... size 5` in `env._command_reset`: v15 made
  `T.FLAT = FREE = 5` (open-floor column) but the per-type heading table `hmax` kept 5 entries; export_eval.py (and paper_eval.py,
  disc_score.py) put the robot on T.FLAT. Training never indexes 5 (types 0-4), so the trainer was not affected.
  Fix: `code_v15/ssr/env.py` line 404 table gets a 6th entry 30 deg (= v14 flat); backup `ssr/env.py.bak_0215`. Running trainer untouched.
  Re-exported **`eval/export/r8_v15_p100/`** (iter 7,050; ONNX vs torch max err 3.7e-6; mean_vx 0.60) with DONE marker.
- **Pre-flight**: new `runs/export_dryrun.sh <code> <smoke_run_dir> <out>` exports the CPU train-smoke checkpoint with the new code
  and checks policy.onnx / adapter_card.json / obs_check.npz and max_err < 1e-4. Verified: old env.py -> FAIL (the IndexError),
  fixed -> PASS. Added to `runs/v15_cpu_checks.sh` (log `$O/export_dryrun.log`); every future queue gate must also require
  `grep -q "EXPORT_DRYRUN PASS" $O/export_dryrun.log`.
- **What went wrong 2 (v15 rejected by its own gate)**: +150 at 7,150: ep_len 13.64 (ok vs 0.7 x 13.61), **vx_ratio 0.556 <
  0.584 (0.75 x 0.778)** -> FALL BACK at 02:14:47; queue32 resumed r7 v14 from r7_v14_final (7,000) at 02:15:32 (one trainer, pid
  in tmux ssr32). Evidence (r8 log every 10 it): after the 50-it warm-up vx_ratio fell monotonically 0.80 (7,050) -> 0.45 (7,150),
  double support 0.21 -> 0.50, single 0.75 -> 0.48, steps/s 5.2 -> 5.8, cmd_speed steady 0.74: a shuffle in place, the
  standing/creep failure again. Style per group fell during the policy updates: 7,010 ramp .72 stairs .74 beam .72 platform .74
  discrete .75 -> 7,160 ramp .63 stairs .66 beam .64 platform .66 discrete .65; r_s 0.73 -> 0.65; r_l 1.66 -> 1.56; r_f 0.81 -> 0.88.
  Levels held (discrete 0 -> 2.0), so it is not a terrain collapse. Suspects (not yet separated): amp_v5 is about 5x smaller than
  amp_v4 (ramp 13k vs 35k frames, stairs 15k vs 92k) and has **no flat group** (v4 had flat 50k); fresh discriminators on it.
- **Decision (mine, 02:20)**: v15 not kept (the gate decided; the evidence agrees). v14 continues. Next experiment (v16) only after
  the cheap checks: (a) reference speed and double-support share per group, amp_v5 vs amp_v4 (find the 63-feature layout in the
  amp builder); (b) disc_score.py of the v14 policy against fresh amp_v5 discriminators. Candidate v16 = v15 + longer D-only
  warm-up (200 it) + the flat group kept from v4, if (a) shows v5 references are slower or more double-support.
- r8_v15_p100 (7,050, before the slowdown) is a valid export to score; low priority behind r7_v14_p250.

### 2026-11-13 03:30-04:00 UTC: v16 checks, v16 = v15 + resume transition, armed (fresh agent)
All outputs in `runs/v16_checks/` (script `code_v15/scripts/v16_checks.py`; checks.json, tiles.json, replay/summary.txt).
- **(a) amp_v5 vs amp_v4 per group** (kinematic contact: foot world speed < 0.3 m/s; moving = |v| > 0.2): v5 is NOT slower or more
  double-support. Mean moving speed v5 0.52-0.55 m/s (v4 0.43-0.58), standing share v5 0.11-0.21 (v4 0.19-0.31), double support
  v5 0.12-0.19 (v4 0.08-0.21), steps/s v5 1.9-2.8 (v4 1.5-2.3), vx p90 v5 0.88-1.07 (v4 0.60-0.82). Hypothesis rejected.
- **(b) fresh amp_v5 discriminators** (r8 at 7,050 = 50 warm-up iterations, and r8 last 7,160), flat open floor, cmd 0.5/0.74/1.0:
  every group's D already rates v14's walk below v15's shuffle at 7,050 (style 0.51-0.65 vs 0.58-0.67, gap 0.02-0.07), gap wider at
  7,160 (0.48-0.55 vs 0.60-0.67 at 0.74 m/s). References all ~0.81-0.85 incl. standing windows (D does not separate standing). So a
  push toward the shuffle exists. Its pull grew because v15 also switched **adv_norm joint -> separate**: logged adv_sd (weighted
  spreads) v14 [0.171, 0.019, 0.0072] = style 3.7 % of the policy gradient; separate with WEIGHTS (1, 0.25, 0.2) = style 14 %, foothold
  17 % (was 10 %). And sigma_v 0.1 (v14) -> 0.25 (Table 7) softened the speed penalty (0.45 x command earns 0.51 vs 0.19).
- **(c) where the shuffle is** (extra check): deterministic CPU rollouts at heading 0 do NOT show it (v15@7,160 as fast as v14 on
  flat and on ramp 13 / stairs 10 / beam 13 / platform 10 / discrete 2; only cadence +1-2 steps/s and beam double 0.38 vs 0.24).
  Both policies frozen in the v15 TRAINING env on CPU (64 robots, sampled actions, real command mix, 14 iterations): **v14 vx_ratio
  0.95 / double 0.18, v15 0.70 / 0.33**. So it is a property of the v15 policy under training conditions, not the curriculum.
- **Decision (mine, 03:58)**: v16 = v15 unchanged (paper end state) + a resume transition: `--critic_warmup 200` (policy frozen;
  critics + fresh D train against v14), then `--to_paper_ramp 300 --sig_v_start 0.1`: advantage blend joint -> separate and sigma_v
  0.1 -> 0.25 over 300 iterations; paper values from +500. Not changed: flat group (the paper has none; v5 is not slower),
  foothold base frame (paper; ramps with the separate norm). Code `code_v16/` = code_v15 + train.py patch (`runs/v16_checks/patch_v16.py`);
  flags `runs/r7_flags_v16.txt`. CPU checks `runs/v16_cpu_checks.sh`: equivariance (actor 0.080), mirror 4/4, env smoke 17/17, train
  smoke 3 lines with to_paper 0 -> 0.5, sig_v 0.1 -> 0.175, EXPORT_DRYRUN PASS (err 1.4e-6). Terrain code unchanged -> v15 pick solref 0.005.
- **Armed** 03:59:34: queue33 (tmux `ssr33`, `runs/queue33.log`, prefix QUEUE33), gate 1 passed; at the next r7 save it stops r7, saves
  `eval/export/ckpts/r7_v14_final2.pt`, starts **r9** (code_v16, 8,192 envs; `NUM_ENVS=16384` in the queue's environment is the one
  change for doubling, after the coordinator confirms). Fallback to v14 (resume r7): crash; from +60 sps < 0.8x or tilt > +0.05;
  at +350 and +500 ep_len < 0.7x or vx_ratio < 0.75x v14. Exports `r9_v16_p300`, `r9_v16_p500`.

### What went wrong this turn
- train smoke caught my log bug (`sig_v` logged twice -> TypeError); fixed before arming. queue33 first start lacked its path line
  (built from queue32 with an off-by-one); killed after 15 s, nothing ran (log kept as `v16_checks/queue33_badstart.log`).
- The node has 24 cores (not 28): one replay failed taskset on 24-27 and was rerun on 16-19.

### 2026-11-13 05:28-05:55 UTC: one code copy, cause separation of the v15 / v16 shuffle (fresh agent)
- **One code copy (user rule, via coordinator)**: code_v16 (= code_v15 + v16 transition) folded into `code/` behind flags whose defaults
  reproduce v14. Unflagged v15 items now flags: `--paper_terrain` (terrain.py PAPER layout: discrete in slot 4, beam = gap slot heading 15,
  FREE column; env X2_TERRAIN_PAPER=1, set by train before the imports; also disc_index and the heading table), `--simcheck` (simcheck
  faults = truncations, reward 0, no level move; default: v14 NaN / |qvel|>60 terminate). Files merged: ssr/{env,train,terrain,networks,x2}.py,
  scripts/supervise_node.py, new scripts build_amp_v5.py / smoke_v15_env.py / v16_checks.py, tests/test_equivariance.py, data/amp_v5*.
  Fixes added: home_type follows the resumed types on --resume (paper mode); init_from's flat->discrete level reset only in paper layout.
  Diagnostic flags (default off): `--diag_out --diag_disc_from --diag_disc_frozen`.
- **Equivalence (CPU, all RNGs seeded, 32 robots, 2 iterations, runs/merge/equiv/compare*.txt)**: old code/ twice = identical (so the check
  is deterministic); old code/ vs merged with v14 flags: 0 differing log values, max weight diff 0 (only new log keys added); old code_v16
  vs merged with v16 flags + `--paper_terrain --simcheck`: 0 / 0. Re-run after the diagnostic patch: same 0 / 0.
- Installed 05:43: old code/ kept as `_trash/2026-11-13/code_v14_before_merge`; code_v15, code_v16 and the staging tree moved to
  `_trash/2026-11-13/`. Flags r7_flags_v15/v16 carry `--paper_terrain --simcheck`; queue32/33, v15/v16 check scripts, replay scripts
  point at code/. grep code_v15|code_v16 over runs/*.sh|txt|py, runs/v16_checks/*.sh, code/: only runs/merge/equiv*.sh / install.sh
  (the check itself). Running r7 (started 04:40) keeps its imports; every restart now runs code/. Not merged: code_pre/, simcheck_v15/
  (older copies, left for the coordinator; not used by any script I found).
- **r9 timeline (runs/r9/log.jsonl)**: warm-up end 8150 vx_ratio 0.724; 8160 0.697 (to_paper 0.03, sig_v 0.105), 8170 0.659, 8180 0.586
  (0.10, 0.115), 8200 0.513 (0.16, 0.125), double 0.27 -> 0.48. The resumed v14 (r7, same code, same iterations) stays 0.72-0.76. So the
  shuffle starts within 30 policy iterations while the blend is <= 10 % separate and sigma_v <= 0.115: not "half way".
- **CPU gradient check (runs/merge/diag/summary.txt; v14 policy frozen, 96 robots, 20 iterations after 10 of critic fit)**: corr(A, speed
  over the next 10 steps), same rollouts: joint vs separate differ by <= 0.015 in every run (R0 v14 env +0.030 / +0.017; R1 paper env,
  fresh D, sigma 0.1 -0.021 / -0.020; R2 sigma 0.25 +0.035 / +0.033; R3 v14's D -0.054 / -0.053), standard error 0.04-0.05. The style
  advantage pulls toward slower in every setting incl. the stable v14 env (-0.08 / -0.07 / -0.03 / -0.01). Verdict: normalisation does not
  turn the step direction; sigma and D effects are below the CPU noise. Not separable on CPU -> GPU tests.
- **Decision (mine, 05:52)**: GPU tests by queue34 (tmux ssr34, runs/queue34.log, prefix QUEUE34), one at a time, v14 = fall-back:
  T1 = r9 repeated from r9/ckpt_008000 with joint norm and sigma 0.1 HELD after the warm-up (the only difference from r9 after 8150).
  If T1 falls (mean vx_ratio of 3 logs < 0.65): T2 = v14 env + flags, only fresh amp_v5 discriminators (init_from r7_v14_final3,
  warm-up 200). If T1 holds 150 policy iterations: TA = T1 with separate normalisation at once. Then r7 resumes (v14 flags, code/).
  Deviation from the coordinator's A/B/C order: T1 tests A and B together against r9's own run first (r9 data already makes them unlikely).
### What went wrong this turn
- Seed-less equivalence (first try) differed between two runs of the SAME code: train.py never seeds torch. Fixed for the check with
  runs/merge/seeded.py (all RNGs seeded, deterministic kernels); training itself still unseeded (noted, not changed).
- A pkill pattern matched my own ssh command twice (exit 255); no run was affected.

### 2026-11-13 08:29-08:33 UTC: GPU idle 2h15m (06:13-08:30), T2 crash loop; r7 resumed (coordinator-directed fix)
- **What went wrong**: T1 fell at 8180 (vx_ratio 0.63, 06:12). queue34 then started T2 (v14 env and flags, fresh amp_v5 D, init_from
  r7_v14_final3 at 8600) at 06:12:53. Every T2 start died before iteration 1: `KeyError: 'flat is not a file in the archive'`
  (runs/t2.log): the v14 flags build a 'flat' discriminator group, amp_v5.npz has no 'flat' group. The supervisor restarted it 92 times
  (`crash(1) at iter None`, runs/queue34_t2.sup; its own limit is 200) and queue34 only checked for the supervisor's exit, so the GPU
  trained nothing for about 2h15m. No CPU smoke test of the T2 flag set had been run before arming it (the cause of the cause).
- **Fix**: 08:29:45 killed the T2 supervisor (pid checked: `supervise_node.py --name t2`) and its trainer only. queue34 took its own
  CRASH branch and exec'd the r7 resume at 08:30:10 (v14 flags, code/, X2_HF_CHUNK=400, normal supervisor) from r7/last.pt =
  r7_v14_final3 (cmp identical, it 8600). First log line it 8610 at ~08:31, vx_ratio 0.964; GPU 42-60 %, 24 GB.
- **Guard added**: `runs/crash_guard.sh` (`crash_loop <sup file> <start line>`: more than 3 `crash(..) at iter None` lines; `kill_run
  <name>` kills only that run's supervisor and trainer). Sourced and wired into queue34 `test_run` (stop the test, verdict CRASH,
  queue resumes r7) and queue33's r9 loop (fall back to r7). Backups runs/_q34.bak, runs/_q33.bak. Tested: fires on
  queue34_t2.sup, quiet on queue34_t1.sup. **Rule: every new queue sources crash_guard.sh and checks crash_loop in its wait loop.**
- **T2 is not to run again** until (a) amp_v5 gets a 'flat' group or the v14 flags are adapted to amp_v5's groups, and (b) the exact
  T2 command passes a 2-iteration CPU smoke test (`CUDA_VISIBLE_DEVICES= ... --iters 2 --num_envs 64`). The T1 FALL verdict stands;
  the T2 question (does fresh amp_v5 D alone cause the v15/v16 slide?) is still open.


### 2026-11-13 13:28-14:25 UTC: amp_v5 from paper reference v4, the one planned restart (v17), run to 20,000 (fresh agent)
- **amp_v5 rebuilt** (`code/scripts/build_amp_v5.py`, log `runs/amp_v5_build.log`, meta `code/data/amp_v5_meta.json`): paper ref **v4**
  (`paper_ref_x2_v4.pkl`; categories from the v4 time ranges, checked: stairs up 8 clips 24.0 s, down 9 / 16.8, step onto block 14 / 9.3,
  step off 17 / 9.4, all equal to the table) + the user's Pico takes; no BONES-SEED. v14 layout (r7 has no `--paper_terrain`): groups
  ramp, stairs, beam, platform, flat. Seconds vs Table 11: ramp 261.3/261.3 (fwd 78.4, backward 0 - no backward window >= 1 s in v4 -
  so turn 52.3, ramp traversal 130.7), stairs 295.8/295.8, beam (gap slot) 174.3/174.5 (Pico logs 59.9 + 2x2 13.7 + 4x4 13.7), platform
  181.6/181.6, flat 136.0 (the discrete slot's size: fwd 68.0 / turn 47.6 / side 20.4; paper has no flat group, v14 needs one for
  the open-floor robots and the walking starts). Old v2-reference amp_v5 + builder moved to `_trash/2026-11-13/amp_v5_v2ref/`.
- **Discriminators warm, not fresh (my decision)**: full resume of r7/last.pt incl. D and its optimiser. Evidence: every run with fresh
  amp_v5 D slid to a shuffle (v15 +150, v16 +350, T1 at +30 policy its); check (b) showed fresh D rate the shuffle above v14's walk; T2
  (fresh D alone) never ran. Warm D keeps the reward the policy is in equilibrium with and adapts to the new data. Plus 50 iterations
  with the policy frozen (`--warm_until_abs 11401`, critics + D train) so D moves to v4 + Pico and the 0.3 arm weight first.
- **Code changes** (backups in `_trash/2026-11-13/`): train.py `--torch_seed` (seed **20261113** recorded in args.json, log "torch seed"),
  `--warm_until_abs` (absolute, so a crash-resume does not add another warm-up), checkpoints now save / restore `ext_level` and
  `ext_type` (coordinator 2026-11-13: ext levels reset to ~0 at every restart); env.py log field `gait.slow_made_good` (share of command
  made good, all robots with a 0.10-0.35 m/s command) and `gait.slow_share`. The ext-level save is in the file but the running trainer
  imported the old file: it takes effect at the next crash-resume of a newer save only; this restart itself reset ext levels (the v14
  checkpoint never stored them, nothing to restore).
- **Flags** `runs/r7_flags_v17.txt` = v14 flags with `--amp_file data/amp_v5.npz`, `--disc_part_w 1.0,0.3,0.0` instead of
  `--disc_mask_arms`, `--torch_seed 20261113`. 8,192 robots (see open conflict below).
- **Pre-flight** (`runs/v17_preflight.sh`, `runs/v17_checks/summary.txt`) PASS 13:36: amp_v5 loaded through `ssr.train.AMPData` for
  every T.TYPES key + `_starts` (and sampled), mirror tests, CPU train smoke of the exact command resuming a copy of r7/last.pt (11,300:
  2 frozen + 2 policy iterations, seed and warm-up lines present, slow_made_good logged), export dry-run PASS.
- **Restart**: queue35 (tmux `ssr35`, `runs/queue35.log`) stopped r7 v14 at the 11,350 save (13:39:29, `eval/export/ckpts/r7_v14_final4.pt`)
  and started v17 at once (supervisor output `runs/queue35_r7.sup`). Fixed end 20,000 (about 16 h at 6.4-6.8 s/it, ~06:30 UTC 11-14).
- **First 100 iterations** (mean of log lines; base = 10 lines before): diverged 0.1 % -> 0.2 %; tilt 3.1 % -> 2.9 %; share of command
  0.795 -> 0.77-0.78; double support 0.22 -> 0.24; steps/s 4.56 -> 4.40; r_s 0.33 -> 0.27-0.29; samples/s 29.0k -> 29.2k; slow commands
  made good (new field) 0.98. CPU start probe (64 robots, deterministic) base 11,350 vs 11,450: started 100 % / 100 % at 0.15, 0.2, 0.35;
  mean vx 0.198 / 0.188, 0.269 / 0.253, 0.364 / 0.376 (`eval/start/v17_base_11350.json`, `v17_p100.json`). No slide toward a shuffle.
  Levels: ramp 15.0 -> 13.4, stairs 8.1 -> 11.2, platform 8.5 -> 11.4, beam 13.6 -> 13.1, flat 13.0; ext levels reset 5-6 -> 0.2-0.6.
- **No pause in the setup (user rule 2026-11-13)**: queue35 only (1) fault fall-back on a crash loop (crash_guard.sh) -> resume r7 with
  v14 flags, (2) copies last.pt and runs CPU-only work (nice 15, cores 16-23): start probe at +100, `export_eval.py` every 500 iterations
  into `eval/export/r7_v17_<it>/` with a DONE marker (checkpoint copy `eval/export/ckpts/r7_v17_<it>.pt`). Nothing touches STOP, nothing
  runs on the GPU. queue31 / queue34 are finished (only their tmux shells remain).
- **Stairs decline (coordinator question)**: in the v14 layout route robots keep the route's type (no `--ext_home`) with their tile level
  from whatever type they came from, and ext robots never update `level`; so `levels_by_type.stairs/platform` average real stairs robots
  with route robots carrying stale levels. Evidence: right after this restart (all robots back on tiles) stairs 8.1 -> 11.2 and platform
  8.5 -> 11.4 within 100 iterations while ramp fell 15.0 -> 13.4: the per-type means re-mix, they are not a clean capability measure.
  The decline started with v13/v14 at 4,800 (9 ext types, ext_frac 0.4, inset stairs, stairs-type routes). Whether real stairs skill
  fell is not established: needs a CPU probe per stairs level on the final checkpoints (next step). amp_v5 is not expected to fix it:
  style is about 4 % of the policy gradient under joint normalisation; it is a curriculum / bookkeeping effect.

### What went wrong this turn
- My first build ran nothing: copying the builder by line range dropped the `main()` call (exit 0, no output); caught by the empty log,
  fixed by appending the original tail.
- queue35 prints a harmless `No such file` for the first `wc -l < queue35_r7.sup` (file not yet created); L0 = 1, the crash guard counts
  from the start of the file, correct.
- The +100 probe watcher on my side timed out (probe stdout carries Warp banner lines before the JSON); results read from the files.
- Ext-map levels were reset by this restart; the v14 checkpoints never stored them. Fixed for later saves (see above).

### 15:45 UTC 2026-11-13: v16/v17 exports failed the desktop hand-over check - fixed (FINDINGS E4)
- What went wrong: r9_v16_p300 and r7_v17_011500/012000 failed on depth (0.206 vs 0.02). Two defects: (1) the record's flat
  level-0 tile (v14 layout) sees the 1.3 m side-obstacle walls of flat level 1 within 3 m; the check replays on an open plane.
  (2) training's march (38 x 0.10 m) reports no hit beyond 3.88 m; the adapter's exact cast saw floor at 3.9-4.4 m (all earlier
  SSR scores had a few far corner pixels at 0.93-0.97 instead of 1.0). Residual: hits in the march's last 0.1 m flip on float32
  rounding (replay exact 64,767/64,767; a 2e-6 origin shift flips 84/450), not matchable pixel by pixel.
- Fix: export_eval.py zeroes the vision raster for the record and writes ray_march_iters/step_cap/tmin/ray_hit_range in the card
  (backups in tracks/ssr/_trash/export_eval.py.before_*); eval/adapters/ssr.py: hits beyond ray_hit_range = no hit;
  eval/tools/ssr_check.py: band pixels reported as depth_band, excluded from the depth max (FAIL if > 2% of image; check change
  for the coordinator). 11,500 and 12,000 re-exported (old folders in tracks/ssr/_trash/export_pre_*_20261113): PASS (depth
  3.2e-4 outside band, band share max 1.3%). v15 p100 still PASS. Training untouched. Scorer re-queued both at 15:40.

### 2026-11-13 18:20-19:10 UTC: style collapse after the v4 switch diagnosed (fresh agent, CPU only, run untouched) - FINDINGS E7
- Question (coordinator): r_s 0.33 -> 0.10, D(policy) -0.63 -> -0.88 since 11,350; user sees straight arms and waist tilt at 12,000.
- Cause: the arm action noise. Arms were masked from the discriminators until 11,350 (`--disc_mask_arms`), so the 8 arm stds drifted
  under the entropy bonus (13,500: 0.68 / 0.97 / 0.48 / 0.75 x action scale 0.258 / 0.258 / 0.485 / 0.485 = 0.17-0.36 rad white
  noise per step; elbow std rose 0.52 -> 0.75 after the switch). v17's `--disc_part_w 1.0,0.3,0.0` let D see arms: arm grad share
  0 -> 0.37-0.60 in all groups. Sampled-action arm qd spread 2.48 vs 0.36 in the reference.
- Evidence (`runs/style_diag/style_diag*.py`, `out.json`, `out2.json`, `out3.json`; flat, 64 robots, 0.6 m/s; v17 13,500 policy + D):
  sampled actions D -0.95, r 0.06, 47 % windows D < -1 (zero reward, zero gradient) = training; mean actions D +0.05, r 0.75;
  legs-only noise -0.44 / 0.48; arm std 0.3 -0.71 / 0.26; arm std 0.3 + ankle roll 0.6 -0.67 / 0.30; reference arms swapped
  into the sampled windows -0.95 -> +0.33. Style advantage spread 0.0073 -> 0.0037 (4.3 % -> 2.1 % of locomotion).
- Posture: mean-action elbows near 0 rad (reference -0.69; default -0.6), arm swing 0.1-0.2x reference; reference arms into the
  mean-action windows lift D 0.05 -> 0.56. amp_v5: no defect (qd = 50 Hz difference of q in all 352 clips, mirror exact, order and
  default pose match); reference shoulder roll nearly constant (sd 0.013 rad), noted. Disc normaliser not the cause.
- Decision: no restart (user rule; locomotion unaffected). At the next planned restart after 20,000: new train.py flag to reset
  the arm log-stds to log(0.3) and cap them at 0.4 (ankle roll cap 0.6 optional), D kept warm. Not yet coded.

### What went wrong this turn
- My first hypothesis (frozen discriminator normaliser fit to masked arms) was wrong: the normaliser's arm spread is 2x the
  reference's. The first probe used mean actions and showed D near 0, unlike training; the sampled-action probe resolved it.

### 2026-11-13 19:20-19:40 UTC: E7 flag written, pre-flight PASS, v18 restart armed on GO (fresh agent, CPU only, r7 untouched)
- train.py (backup `_trash/2026-11-13/train.py.before_armstd_*`): `--arm_std_reset V` sets actor log_std[13:21] = log V once per
  run (marker `arm_std_reset_done` saved in checkpoints, so crash-resumes do not reset again; Adam exp_avg of those entries
  zeroed), `--arm_std_cap C` clamps them after every policy step, `--ankle_roll_std_cap` (indices 5, 11). All default 0 = off.
- Ankle-roll cap: decided OFF. Probe gain is small (r 0.26 -> 0.30, D -0.71 -> -0.67), the cause is the arms, ankle roll is the
  balance joint on beams / stairs where locomotion is fine (share 0.79), paper has no cap. Revisit if r_s stays < 0.25 after v18.
- Warm-up: none (D warm and on the same data; v17's `--warm_until_abs` was for the data switch).
- Pre-flight PASS 19:35 (`runs/v18_checks/summary.txt`): loader OK all 5 groups; mirror OK; smoke resume of r7/last.pt copy
  (14,500): arm std before [0.67, 0.91, 0.42, 1.03] x2 (elbow now 1.03, was 0.75 at 13,500 - still drifting up), after reset 0.3,
  after 4 CPU policy updates 0.3, leg stds unchanged (max diff 0.0009), D warm (moved 0.011 in 6 updates); crash-style resume of its
  own save: no second reset, cap line present; export dry-run PASS (onnx err 3.3e-6).
- Armed: `runs/queue36.sh` in tmux `ssr36`, waiting for `runs/GO_v18` (see HOW TO RESUME).

### 2026-11-13 23:25-23:55 UTC: v18 style plateau diagnosed (fresh agent, CPU only, run untouched) - FINDINGS E7b
- Question (coordinator): v18 (started 20:11 from 14,850) r_s 0.18-0.20 for 150 its, then 0.15-0.16 (to 16,600), not the 0.26
  the E7 probe predicted.
- Stds (log `entropy_std`, and checkpoint 16,500): all 8 arm stds at the cap 0.4 (shoulder yaw 0.377-0.378) from ~15,000; the reset
  0.3 lasted ~150 iterations (entropy bonus pushes them up; the cap binds). Ankle roll back to 1.90, ankle pitch 0.78, waist 0.49.
- Log per group (16,600): arm grad share still 0.25-0.48 (stairs lowest), legs 0.26-0.47, pelvis 0.14-0.27; d_data fell 0.88 ->
  0.75 at the reset and re-sharpened to 0.83 by 16,000.
- Probe `runs/style_diag/style_diag4.py` -> `out4.json` (16,500 policy + D, flat, 64 robots, 0.6 m/s): sampled D -0.94 / r 0.07
  (45 % windows D<-1); same windows scored by the 13,500 D: -0.76 / 0.22 (D re-sharpened); det 0.01 / 0.72; arm noise 0: 0.37;
  arm std 0.2: 0.22; ankle roll 0.6: 0.09 (no gain); leg noise 0: 0.35. Swaps on sampled windows: arms -> +0.09, arm q only
  -> -0.25, arm qd only -> -0.38, legs -> -0.33, pelvis -0.93 (no effect). Grad share on sampled windows: arm q 0.82, arm qd 0.04,
  legs 0.10, pelvis 0.04.
- Cause: arm POSE, not only noise. `style_diag5.py` -> `log5.txt`: elbow mean action 6.8 (x 0.485 = target 3.3 rad past the stop;
  q_hi elbow = 0.0 rad), was 1.31 at 11,350; elbow q - default 0.58 in every mode (pinned at the stop, reference -0.09).
  Shoulder roll q - default L -0.24 / R +0.21 vs reference +0.26 / -0.26 (0.5 rad the other way). Pressing the elbow into the
  stop is how the policy removed elbow noise; now no sampled action leaves the stop, so PPO sees no reward difference for bending
  it and the D's main gradient (arm q) cannot act. Plus the std at 0.4 (not 0.3) and the re-sharpened D.
- Decision: no restart (user rule; locomotion unaffected). Next run (after 20,000): (1) action-bound loss on the arm action means
  (penalise PD targets outside [q_lo, q_hi], so the elbow mean is pulled back to the stop edge where noise explores the bent side),
  (2) arm std reset and cap 0.2 (paper Table 9 min std 0.2; probe 0.07 -> 0.22). Ankle-roll cap stays off (probe +0.02).
  To be coded behind default-off flags and pre-flighted on CPU before 06:30 UTC; not yet coded.

### 2026-11-14 23:40-03:05 UTC: E7b flags coded and tested on CPU; v19 armed to start at 20,000 (fresh agent, run untouched)
- Paper (arXiv HTML read 2026-11-14): no action clipping, no PD-target clipping, no bound loss; only the DoF position-limit reward
  (Table 6, -1.5, on measured q) and min policy std 0.2 (Table 9); action scales Table 14. Ours: action clamp +-8, mean 8 tanh(x/8),
  PD target clamped to [q_lo, q_hi] in env.step. That clamp IS the "clip the action to the joint range": it is why a mean past a stop
  is free (no reward change, no gradient), so a clip is not a fix; the fix form is a bound loss on the means (AMP/ASE practice; ours).
- train.py (backups `_trash/2026-11-14/train.py.before_e7b*`), all default-off: `--arm_bound_coef/--arm_bound_margin/--arm_bound_ramp/
  --arm_bound_from` (squared excess of the 8 arm action means outside the action interval mapping into [q_lo+m, q_hi-m]; log field
  `arm_bound`), `--arm_bias_shift/--arm_shift_to/--arm_shift_lr` (one-time shift of arm means past a stop, at the end of an iteration,
  with exact compensation of every consumer of the previous action and of q: actor/critic/foothold norm means, estimator/foothold/
  encoder first-layer biases; checks: other joints 1e-6, est/foot/critic 0, encoder residual 0), and the std reset marker is now per
  value (`arm_std_reset_val`; v18 saves count as 0.3, so `--arm_std_reset 0.2` resets once more, crash-resumes do not).
- CPU trials (16 robots, 8-12 its from r7 saves 16,700-17,500; logs `runs/v19_checks_wip/coef_c*.log`, probes `runs/style_diag/
  style_diag6.py` -> `log6_<tag>.txt`; flat 0.6 m/s, 64 robots). Control c0 (v18 flags): KL 0.02-0.04, probe vx 0.75 alive 1.00.
  c1 bound coef 1.0 at once: KL 73 then ~1, elbow mean in bounds by 8 its, probe vx -0.08 alive 0.33 (walking broken).
  c3 coef 0.05: PPO wins, bound 45 -> 58. c4 ramp 10 its: KL 7.6 then 0.6-0.9 per it. c5-c7 bias shift (value-only iteration, lr
  1e-5): KL 50-1000, elbows flung to -2.3 or back past the stop, alive 0.39. c8/c10 exact action-input compensation, c11 + q
  compensation: shift itself exact, but in the very next rollout the elbows return past the stop (bound 0 -> 50-56), then KL 2-16,
  probe vx 1.04 alive 0.58. Reading: the policy has a learned closed-loop attractor to the stop (it is how it removed elbow noise);
  any move off it destabilises the whole policy within a few iterations. Not fixable warm in the time left.
- Style probe r7_v18_016500 (`out6_016500b.json`): sampled r 0.074 (elbows on the stop 63-66 % of frames), arm std 0.2 r 0.223,
  means clamped to the bound edge r 0.189 (no local gain: corr(D, bend) -0.06), elbows held at the reference bend r 0.596 (alive 0.98,
  vx 0.53). So the style ceiling is there, but reaching it from the pinned policy needs a run that never lets the mean escape.
- DECISION (2026-11-14 03:00, mine): v19 = v18 flags with `--arm_std_reset 0.2 --arm_std_cap 0.2` only (`runs/r7_flags_v19.txt`;
  the bound/shift version kept as `runs/r7_flags_v19_bound.txt`, the std-only copy `r7_flags_v19_stdonly.txt`). Reason: it is the
  paper's min std, the probe gain is 0.07 -> 0.22, and it does not break walking; the bound loss / shift do (c1, c11). The bound loss
  goes into the recipe for the next from-scratch run, from iteration 0 (untested from scratch).
- User decision relayed by the coordinator 2026-11-14 ("SSR continues after 20,000", USER_RULES): continue from the 20,000 save to
  30,000, start automatically only if the pre-flight passed; the SSR agent may create GO. Done: `runs/pf_then_go.sh` (tmux `ssr37pf`,
  log `runs/pf_then_go.log`) runs `runs/v19_preflight.sh` -> `runs/v19_checks/summary.txt` (loader, mirror, smoke resume with the
  v19 flags, reset 0.2 / cap / marker, KL < 0.5, crash-resume no second reset, flags-off old path, export dry-run, walking probe
  alive >= 0.9 and vx >= 0.4) and writes `runs/GO_v19` (content 30000) only on PREFLIGHT_PASS.
- `runs/queue37.sh` (tmux `ssr37`, log `runs/queue37.log`; header comment still names the bound flags - it reads the flags file, which
  is std-only): phase 1 waits for the r7 v18 supervisor/trainer to end, keeps `eval/export/ckpts/r7_v18_final_<it>.pt` and exports
  `eval/export/r7_v18_<it>/` (CPU, DONE marker); phase 2 waits for GO, refuses without PREFLIGHT_PASS, with r7 training, or with any
  process on the GPU; starts `supervise_node.py --name r7 --iters 30000 --num_envs 8192 -- $(cat runs/r7_flags_v19.txt)` (output
  `runs/queue37_r7.sup`); writes `runs/v19_STARTED` (start time, iteration) once 20 iterations advance, or `runs/v19_START_SLOW`
  after 45 min without; +100: start probes `eval/start/v19_*.json` and style probe `runs/style_diag/log6_v19_p100.txt`; exports
  `eval/export/r7_v19_<it>/` every 500; crash loop -> resume r7 with the v18 flags to 30,000 (`runs/queue37_r7_v18fb.sup`).

### What went wrong this turn
- Two shell-quoting mistakes (a nested heredoc and a CRLF-written script) cost two failed trial launches; one `pkill -f` matched my
  own ssh shell. No effect on the run. Files written from Windows are now LF-converted on upload.
- First version of the auto-GO chain would have started the bound-loss recipe before its walking damage was known; I stopped it
  (01:06, no GO had been written) and added a walking probe as a pre-flight gate.

### 2026-11-14 15:37-16:30 UTC: stair refusal diagnosed (E8), v20 armed and GO written (fresh agent)
- v19 ran 06:00 UTC from 19,999 (queue37); at 25,120 levels ramp 14.7 / stairs 6.25 / beam 13.2 / platform 8.1 / flat 12.6.
- Probe `runs/stairs_diag/stairs_diag3.py <ckpt> <out.json> [steps per levels modes]` (env PROBE_TYPE=platform, STAIRS_V20=1/2 sets the
  v20 env flags): own tiles, per robot stall / attempt / fall / cause, curriculum outcome ours vs Rudin, reward per step at the foot vs on
  the flight (ours and Table 6 with sigma_v 0.25). Results in FINDINGS E8; jsons d25000 (fixed 0.6 + training commands), p25000 (cause
  split + no-wall), v20probe, plat25000. Each level row has ONE tile with a fixed sign: up-first 4, 16, 19; down-first 8, 12, 14, 17.
- User asks handled: "Finds ways to encourage to walk" and "investigate why our run refuses ... custom reward changes". Per-term answer
  (E8): no added term pays the stander; standing at the foot pays as much as climbing under our AND the paper's terms; what blocks is the
  curriculum (DEV-P) and the side-obstacle termination (ADDED). Ranked: 1 curriculum promote-at-tile-end (paper: half tile), 2 lane-wall
  termination on stairs/platform tiles (ours), 3 posture terms (paper) out-earn a slow climb - no paper change possible, 4 sigma_v 0.1
  (DEV-P) neutral at the stair foot, 5 swing_h (ours) pays climbing more, not standing; creep/drift 0 at 0.6 m/s; edge stop only at down
  edges; resample 8 s keeps <= 8% of refusals at slow commands (both rules).
- Code (default-off; backups `_trash/2026-11-14/{env,train}.py.before_v20`; patch `runs/patch_v20.py`): `--cur_rudin`, `--cur_fair`
  (not used), `--swing_prog` (not used), `--climb_types`, `--climb_side_noterm`, `--climb_creep`. Flags `runs/r7_flags_v20.txt` = v19 +
  `--cur_rudin --climb_types stairs,platform --climb_side_noterm --climb_creep`.
- Pre-flight `runs/v20_preflight.sh` PASS 16:25:39 (`runs/v20_checks/summary.txt`): loader, mirror, curriculum unit check (old [9,9,9,10],
  Rudin [9,11,9,11] for stall / 4.5 m / 2 m / 7 m), train smoke resume 25,400 KL 0.55 first then 0.02-0.03, flags-off path, export
  dry-run err 4.3e-6, walking probe vx 0.71 alive 1.00 (style r 0.12), stairs probe no side-obstacle endings. GO_v20 written by
  `runs/pf_then_go20.sh` 16:25:39 (user approved the fix restart; USER_RULES "SSR stair refusal fix approved").

### What went wrong this turn
- First probe run classified 25-62% of endings as "other": my cause list lacked the side-obstacle / left-tile checks; re-run with them.
- `--climb_creep` turned out weak in the probe (0.05 per step at the foot and on the flight alike); kept as the only progress pay, said so.

### 2026-11-14 16:45-17:15 UTC: every paper deviation checked against the live run; v21 armed (fresh agent; FINDINGS E8b)
- User: "Please fix. It's already two days of cloud training without good results so far." Table of all 26 DEV-P + 37 ADDED rows vs the
  live trainer command line: `docs/findings/ssr_paper_vs_code_status_20261114.md`. 8 applied before, 5 user overrides, 7 fixed now (v21),
  6 deferred (D1 = paper terrain set, owner me, ready by 2026-11-15 06:00 UTC for the from-scratch run; warm evidence v15/v16/T1).
- Code: `--beam_hmax` (new, default 30; `runs/patch_v21.py`, backups `_trash/2026-11-14/{env,train}.py.before_v21`). Everything else in
  v21 existed behind default-off flags since the v15/v16 merge. Flags `runs/r7_flags_v21.txt` (comment file `.md`), fall-back set
  `runs/r7_flags_v21b.txt` (= v21 without omni_top / omni_slow 0 / no_vy_clamp / foot_base_frame).
- Pre-flight `runs/v21_preflight.sh` (cores 4-15) -> `runs/v21_checks/summary.txt`; GO by `runs/pf_then_go21.sh` (tmux `ssr39pf`) on PASS.
  Smoke A (exact flags, resumed 25,800): KL 0.039/0.022/0.018/0.018, to_paper 0.003 -> 0.013, sig_v 0.1005 -> 0.102. Unit checks: beam
  heading max 29.98 -> 14.99 deg; omni_top: over-the-top robots (and only those) become omni on the flat tile at x = 4 m, back next episode.
- Live v20 at 25,820: stairs level 8.98 (6.25 at 25,120), level_max_share 0.063, omni share 0.35, vx_ratio 0.76. v20 +100 stairs probe
  (fixed 0.6 m/s, 24 robots): level 16 passed-riser 0.58 (base 25,000: 0.125), 17 0.29 (0.08), 19 0.21 (0.0), stall at 19 0.75 (1.0).
- Pre-flight PASS 17:19:34 (GO_v21 written by pf_then_go21): smoke B (paper end state at once) KL 0.042/0.022/0.018/0.017, sig_v 0.25;
  flags-off path clean; v21b smoke ok; export dry-run err 3.3e-6; walking probe vx base 0.628 / A 0.517 / B 0.584, alive 1.0 all, style r
  0.13 / 0.14 / 0.17. Its short stairs probe (8 robots, 8 s) showed level 12 stall 8/8 on B (base 3/8 passed); re-checked with 24 robots,
  20 s (`v21_checks/stairsL_{base,A,B}.json`): level 12 passed 1.00 / 1.00 / 1.00, level 19 0.375 / 0.167 / 0.333 -> the 8 s horizon was
  too short for the down-first tile; no regression shown (A at 19 is within 24-robot noise; B equals base).
- v20 +250 probe (25,800): level 16 passed 0.54, 17 0.375, 19 0.375 (stall 0.625). v20 stopped at 26,000 (17:20:45, kept
  `eval/export/ckpts/r7_v20_final_26000.pt`, exported for the scorer); **v21 started 17:20:56 from 26,000**, iterations advance 17:23:56.
  First lines: omni share 0.35 -> 0.054, vx_ratio 0.79-0.92, ep_len 11.4-14.9 (resume transient; gate floor 10.43), KL 0.010-0.014.
  **Gate +150 PASS** 17:39:10 (vx_ratio 0.752 vs min 0.577, ep_len 14.07 vs min 10.43). At 26,170: to_paper 0.57, sig_v 0.185, omni 0.038,
  stairs level 12.6 (per-type means re-mix at a restart, see 11-13 13:39). 7.2 s/it -> 30,000 at about 01:10 UTC 11-15.
- Risk I accept: omni_top cuts open-floor omni robots from ~35 % to the graduates (a few %): less open-floor turn-in-place practice; route
  stations still train it. Check: the scorer's flat turn tests on the r7_v21 500-exports; if turn-in-place drops, the next lever is
  omni_turn for route-free robots (user rule) - mine.

### 2026-11-14 21:37-23:10 UTC: the 30,000 decision (USER_RULES "SSR extension to 40,000 only if improving"; then "Keep SSR training")
- Evidence, v21 27,500 (desktop scorer) vs v20 final 26,000: legs 38/40/43/44/0 vs 49/50/49/44/5; leg-5 refusals 50/50 vs 42 (a, now
  REQUIRED, fails); successes 0 vs 5 (b fails); training courses 15 of 21 scored (ext_stairs_up pending) vs 18/22, stairs L10 11/20 vs 19,
  stairs L15 0/20 vs 20/20 (all stalled before the first riser, final_x -2.85), stairs L19 0 vs 8, platform L15 0 vs 0 (c fails); CPU
  stairs probe level 19 passed-riser 0.0 at 27,500 and 28,000 (`runs/stairs_diag/v21_d40_{027500,028000}.json`) vs 0.375 (d fails). The
  v21 +250/+500/+1000 probes already showed it (L16 0.17 -> 0.08 -> 0.0). Verdict: v21 made stair refusal worse on every measure;
  its one improvement is ramps (ramp L19 20/20 vs 15).
- Rule alone -> stop. User override (2026-11-14 "Keep SSR training and find the cause"): continue to 40,000 on the v20 flags.
  queue40 (`runs/queue40.sh`, tmux `ssr40`, log `runs/queue40.log`) armed 23:03, `runs/DECISION40` = `extend_v20`. When queue39 / r7 end
  at 30,000: keep `eval/export/ckpts/r7_v21_final_<S>.pt`, export it (`eval/export/r7_v21_final_<S>/`, = export dry-run), pre-flight
  (flags md5 = `runs/v20_flags.md5`, loader iter >= 29,990 and warm_until <= 29,990), resume r7 `--iters 40000 -- $(cat r7_flags_v20.txt)`
  (as queue38 ran v20; sup `runs/queue40_r7_v20x.sup`; `runs/v20x_STARTED`), gates +150/+350/+500, probes `stairs_diag/v20x_p*.json`,
  exports `eval/export/r7_v20x_<it>/`. Gate fail -> stop, keep, restore the 30,000 save, idle (`runs/SSR_IDLE`). Crash loop -> one restart.
  All v20 flags exist in the trainer (checked); v20's sig_v ramp is absolute (from 600), so it is at 0.1 at once, as v20 ran.
- Leg 5 on v21 29,000 scored by me on the desktop (8 processes, 50 trials; `eval/out/ssr/r7_v21_029000_d40/legs/leg5/`) for the record.
### What went wrong
- v21 (all paper items at once) regressed stair attempts: refusal 42 -> 50 of 50 on leg 5 and 20/20 -> 0/20 on stairs L15. Cause not
  yet known (the per-term diagnosis is another SSR agent's job, FINDINGS); candidates are the items v21 changed: separate advantage
  normalisation, sigma_v 0.1 -> 0.25 (paper), omni_top, beam_hmax, foot_base_frame, no_vy_clamp.

### 2026-11-14 23:10-23:30 UTC (node clock): rollback armed, then held; backup added
- Leg 5 on v21 29,000 (my desktop run, `eval/out/ssr/r7_v21_029000_d40/legs/leg5/`): 0/50 successes, 39 refusals, 11 falls (attempts).
  First v21 checkpoint with refusals below v20's 42, but 0 successes; 27,500 was 50/50 refusals.
- User "SSR rollback to 5,050": `runs/r10_preflight.sh` (CPU) and `runs/queue41.sh` (run r10 = resume of r7_v14_p250 at 5,050 through the
  current trainer, v20 flags + `--warm_until_abs 5250`, `--iters 20050`) were written. Compatibility: every network and optimiser state of
  the 5,050 save loads with shapes identical to v21 (enc/aux/actor/critics/foot/disc); missing, so fresh/defaults: ext_level/ext_type
  (extended-map levels), arm_std_reset flags (reset to 0.2 applied once), warm_until; gate_start None (style ramp as v14 had it); its D
  was trained on amp_v4 (disc_mask_arms) and re-trains on amp_v5 (smoke: r_s 0.84 -> 0.30 in 5 frozen iterations). Pre-flight so far:
  loader, mirror, smoke A (frozen, KL 0) and B (KL 0.18 then 0.03-0.04), export dry-run err 3.1e-6 pass; probes still running
  (`runs/r10_checks/summary.txt`). Walking probe base vx 0.53 alive 1.0; B vx 0.81 alive 0.92; the stairs probe on the 5,050 save
  failed (rc 1: my pre-flight did not put args.json beside the checkpoint) - not chased, rollback cancelled (user 2026-11-15 "good fix it").
- Coordinator (FINDINGS E8c): 5,050's 0 refusals came from 36/50 trials stalling at the ramps; refusal mechanism = separate advantage
  normalisation (+ sigma_v 0.25). Rollback HELD: queue41 killed before it acted (logged in `runs/queue41.log`); r10 not started.
- queue40 re-armed 23:29 with DECISION40 = extend_v20 and a backup step (user "Keep backup of 30000 checkpoints"): before any training,
  the full r7 30,000 save + args + flags (v20, v21) + log + the final export go to `/mnt/filesystem-g4/tracks/ssr/backups/r7_v21_30000/`
  (MD5SUMS inside, read-only, skipped if the folder exists; start refused if the copy does not match r7/last.pt).

### 2026-11-15 00:02-00:11 UTC: early switch to v20x (user "We are wasting time. Start right now"; "good fix it")
- Rollback to 5,050 cancelled (user). queue39 disarmed 00:02 (killed; logged). What went wrong: killing queue39 closed its tmux session and
  the r7 supervisor with it; the trainer survived as an orphan, so `runs/stop_r7_now.sh` stopped it right after its next (atomic) save.
  No iteration lost: stopped at **29,550** (00:05:48). queue40: kept `eval/export/ckpts/r7_v21_final_29550.pt`, export
  `eval/export/r7_v21_final_29550/` DONE 00:06:59.
- **Backup** `/mnt/filesystem-g4/tracks/ssr/backups/r7_v21_29550/` (read-only, 10 files, MD5SUMS inside; r7_last_29550.pt md5
  3d60fda28998abf1f712b6313d5df5d8; args.json, flags v20/v21, r7 log, final export). Never delete or overwrite.
- Pre-flight PASS 00:07:19 (flags md5 = v20, loader iter 29,550 warm_until 26,000, export ok). **v20x started 00:07:27 UTC** from 29,550
  to 40,000 with `runs/r7_flags_v20.txt` verbatim (joint advantage normalisation, sigma_v 0.1, Rudin curriculum, climb_side_noterm,
  climb_creep, extended map ext_frac 0.4 / 9 types / prop routes / stations; no stall penalty exists in the code). Iterations advance:
  29,570 at 00:10:28, 6.6 s/it -> 40,000 at about 19:20 UTC 11-15.

## HOW TO RESUME (current at 00:11 UTC 2026-11-15, node clock)
- **r7 v20x RUNNING** (queue40, tmux `ssr40`, log `runs/queue40.log`, supervisor output `runs/queue40_r7_v20x.sup`, `runs/v20x_STARTED`)
  from 29,550 to 40,000 (~19:20 UTC 11-15). Gates +150/+350/+500 (vx_ratio >= 0.75 x v21 base, ep_len >= 0.7 x) -> fail: stop, keep
  `r7_v20x_failed_<it>.pt`, restore r7/last.pt to the 29,550 save, idle. Crash loop -> one restart, then idle. Stairs probes
  `runs/stairs_diag/v20x_p{100,250,500,1000}.json` (compare v20 0.375 at L19, v21 0.0); exports `eval/export/r7_v20x_<it>/` every 500.
- queue39, queue41 (rollback) not running. Leg 5 of v21: 27,500 refusals 50/50; 29,000 0 successes / 39 refusals / 11 falls.
- Next check: scorer leg-5 refusals on the first r7_v20x exports (30,000 / 30,500) against v20 26,000 (42 refusals, 5 successes).

## HOW TO RESUME (older, 23:30 UTC 2026-11-14, node clock)
- **r7 v21 RUNNING** (queue39, tmux `ssr39`) to 30,000 (~01:05 UTC 11-15, 7.0 s/it). **queue40** (tmux `ssr40`) then switches to v20
  flags to 40,000 automatically (DECISION40 = extend_v20). Check `runs/queue40.log` for "pre-flight PASS", "iterations advance",
  gate lines; `runs/v20x_STARTED`; backup line (md5) in queue40.log. queue41 (rollback r10) HELD, not running; r10 pre-flight
  results in `runs/r10_checks/` for the record. CPU probes `ssrd40` done (status `runs/stairs_diag/v21_d40.status`).
- v20x to 40,000 at ~7 s/it = about 19.5 h, ends ~20:40 UTC 11-15.

## HOW TO RESUME (older, 17:15 UTC 2026-11-14, node clock)
- **r7 v20 RUNNING** (queue38, tmux `ssr38`) until queue39 stops it. **queue39** (tmux `ssr39`, log `runs/queue39.log`, prefix QUEUE39)
  waits for `runs/GO_v21` (written by the pre-flight on PASS) AND `runs/stairs_diag/v20_p250.json` (v20's +250 probe, started 16:57;
  2 h timeout), then stops v20 at the next save, keeps `eval/export/ckpts/r7_v20_final_<S>.pt` (+ CPU export `eval/export/r7_v20_final_<S>/`),
  resumes r7 with `r7_flags_v21.txt --warm_until_abs <S>` to 30,000 (supervisor output `runs/queue39_r7_v21.sup`; `runs/v21_STARTED`).
- **Gates (automatic)**: +150/+350/+500 mean gait.vx_ratio of the last 3 lines >= 0.75 x v20 base and ep_len >= 0.7 x base, else stop, keep
  `r7_v21_failed_<it>.pt`, restore r7/last.pt to the v20 final, run v21b (same gates), then v20 flags. Crash loop -> v20 flags
  (`runs/queue39_r7_v20fb.sup`). Gate lines in `runs/queue39.log`.
- **Next checks**: `runs/queue39.log` for the stop / start / gate lines; stairs probes `runs/stairs_diag/v21_p{100,250,500,1000}.json`
  against `v20_p100/p250.json`; `eval/start/v21_p100.json`; `runs/style_diag/log6_v21_p100.txt`; `level[1]` / levels_by_type.stairs,
  omni_share, to_paper (reaches 1.0 at +300), sig_v in `runs/r7/log.jsonl`; exports `eval/export/r7_v21_<it>/` every 500 for the scorer.
- **Open / owners**: D1 (paper terrain set + amp in paper layout + arm bound loss from it 0, for the from-scratch run after 30,000):
  me, ready by 2026-11-15 06:00 UTC. If 17-20 cm stairs still stall >= 80 % after ~1,000 v21 iterations: stall cost on climb tiles.
- v21 ends at 30,000 (~4,100 its at ~6.6 s/it = about 7.5 h, ~01:00 UTC 11-15); not extended.

## HOW TO RESUME (older, 16:30 UTC 2026-11-14, node clock)
- **r7 v20 RUNNING since 16:27:49 UTC from 25,550** (v19 final kept as `eval/export/ckpts/r7_v19_final_25550.pt`; iterations advance 25,570 at 16:30; GPU 94%) (queue38, tmux `ssr38`, log `runs/queue38.log`, supervisor output `runs/queue38_r7.sup`): GO 16:25:39; stops v19 at the next
  save, keeps `eval/export/ckpts/r7_v19_final_<it>.pt`, resumes to 30,000 with `runs/r7_flags_v20.txt`; `runs/v20_STARTED` once 20
  iterations advance (or `v20_START_SLOW`). Crash loop -> resume with v19 flags to 30,000 (`runs/queue38_r7_v19fb.sup`). queue37 exits
  when v19's supervisor ends (expected, not a fault).
- **Next checks**: +100/+250/+500 stairs probes `runs/stairs_diag/v20_p{100,250,500}.json` (levels 12,16,17,19 at 0.6 m/s; base =
  p25000.json): attempt share (passed_riser) and stall share at 16-19, side-obstacle endings 0; stairs level in `runs/r7/log.jsonl`
  `level[1]` (was 6.25) should rise; walking: `eval/start/v20_p100.json`, `runs/style_diag/log6_v20_p100.txt`, vx_ratio in the log.
  Exports every 500 `eval/export/r7_v20_<it>/` for the scorer (leg 5).
- **If after ~1,000 iterations 17-20 cm still stall >= 80%**: next lever is a stall cost on climb tiles (no progress >= 2 s under a
  walking command), coded behind a new flag; the curriculum change alone needs time to bring robots to levels 15-19.
- Older resume items (v19 / v18, written 03:05 UTC 11-14) follow; v19 ran 06:00-~16:30 from 19,999.

## HOW TO RESUME (older, 03:05 UTC 2026-11-14, node clock)
- **r7 v18 RUNNING** (tmux `ssr36`, since 20:11 UTC 11-13 from 14,850) to 20,000, ends ~06:10 UTC 11-14; exports
  `eval/export/r7_v18_<it>/` every 500. Do not stop it.
- **v19 ARMED, auto**: GO path `/mnt/filesystem-g4/tracks/ssr/runs/GO_v19` (written only by `runs/pf_then_go.sh` on
  `runs/v19_checks/PREFLIGHT_PASS`; content = absolute target 30000; `runs/v19_params.env` V19_TARGET_ITERS=30000 is the default).
  queue37 (tmux `ssr37`) starts v19 after the v18 run has ended. Flags `runs/r7_flags_v19.txt` = v18 + arm std reset/cap 0.2.
- **Pre-flight PASS 03:09 UTC; `runs/GO_v19` written (30000) 03:09.** Smoke: reset 0.4 -> 0.2, KL 0.47 then 0.02, legs std diff 0.002,
  crash-resume no second reset, flags-off old path, export dry-run err 2.6e-6, walking probe vx 0.54 alive 1.00 (style r 0.05; elbows
  still on the stop, as expected). r7 at 18,410 at 03:00, 7.2 s/it -> 20,000 at ~06:12, v19 start expected ~06:15.
- **Next checks** (were): pre-flight result (`runs/pf_then_go.log`, `runs/v19_checks/summary.txt`); ~06:10-06:20 `runs/queue37.log`
  for "GO: starting v19" and `runs/v19_STARTED` (or `v19_START_SLOW`); +100 probes; style r_s in `runs/r7/log.jsonl` (v18 0.15-0.21).
- **Open**: elbow pinned at the stop (E7b) is NOT fixed by v19 (only the noise part); the fix needs the arm bound loss from
  iteration 0 of a run. CPU trial outputs in `runs/v19_checks_wip/` (copies of r7 saves; can be cleared by the coordinator later).
- Older resume items (written 14:25-23:55 UTC 11-13; v17 and v18 runs below are history, helper line still valid):
- **Running** (history): r7 v17 (amp_v5 from paper ref v4) since 13:39:29 from 11,350, to 20,000 (~06:30 UTC 11-14), tmux `ssr35` (queue35 ->
  supervisor -> trainer). Flags `runs/r7_flags_v17.txt` + `--warm_until_abs 11401`, env X2_HF_CHUNK=400. Heartbeat
  `/mnt/filesystem-g4/status/ssr.json`. User rule: no pauses, no stops, no GPU evals; only the crash-loop fall-back may act.
- **Outputs for the desktop scorer**: `eval/export/r7_v17_<it>/` every 500 iterations (CPU, DONE marker), from 11,500 on;
  export_eval.py patched 15:35 UTC (flat-vision record + ray_hit_range); 11,500/12,000 re-exported and PASS the check (FINDINGS E4).
- **Next checks** (read-only, no restart): stairs level per stairs tile level by CPU probe on `r7_v17_*` checkpoints; ext levels
  recovering (were 5-6 before the restart); slow_made_good and share in the log.
- **Open (style, E7) - PREPARED AND ARMED, NOT STARTED (2026-11-13 ~19:40 UTC)**: v18 = v17 flags + `--arm_std_reset 0.3
  --arm_std_cap 0.4` (`runs/r7_flags_v18.txt`), D warm by full resume, ankle-roll cap off. Pre-flight `runs/v18_preflight.sh` ->
  `runs/v18_checks/summary.txt` (PREFLIGHT_PASS / _FAIL). Restart script `runs/queue36.sh` armed 19:34 UTC in tmux `ssr36`
  (log `runs/queue36.log`; it only waits for GO, then refuses unless PREFLIGHT_PASS exists and r7 is training). **GO file: `/mnt/filesystem-g4/tracks/ssr/runs/GO_v18`** - created by the coordinator only on the user's yes;
  queue36 then waits for the next r7 save, STOPs r7 v17 there, keeps it as `eval/export/ckpts/r7_v17_final_<it>.pt`, starts
  `supervise_node.py --name r7 --iters 20000 --num_envs 8192 -- $(cat runs/r7_flags_v18.txt)` (output `runs/queue36_r7.sup`),
  probes +100, exports every 500 as `eval/export/r7_v18_<it>/`; crash loop -> resume with v17 flags. GO after r7 ended: starts
  nothing; the next run takes the same two flags. Probes in `runs/style_diag/`.
- **Open**: 16,384 robots (USER_RULES 2026-11-13 "double ... at the next planned restart") conflicts with the brief's v14 flags and the
  one-restart rule; kept 8,192 (untested memory with chest_rs + walls, a failure would need a second restart) - for the coordinator.
  Paper items of "SSR back to the paper" still not on this line (v15/v16 fell back); `--ext_home` (clean per-type levels) not on.
  code_pre/ and simcheck_v15/ still on the node.
- **Helper**: ssh `~/bin/ssr_ssh.sh <timeout_s> '<cmd>'` (stdin not passed: upload as base64). CPU work from `code/`:
  `env CUDA_VISIBLE_DEVICES= X2_PLANT=mc_20261024 X2_EXCL_PELVIS_HIPROLL=1 X2_HF_CHUNK=400 nice -n 10 taskset -c 16-23 ../env/venv/bin/python ...`.

### 2026-11-14 23:05-23:45 UTC: why v21 refuses - per-term and per-group check (fresh agent, CPU only; FINDINGS E8c)
- Probe `runs/refuse_diag/refuse_attempt.py <ckpt> v20|v21 <out> [per levels modes]` (cores 16-23, ~15 min); jsons `ra_v20.json` (v20 final
  26,000), `ra_v21.json` (v21 27,500). Stairs levels 15-19, 0.6 m/s, natural + pushed attempts, flat 0.15/0.2/0.35/0.6.
- Table (10-s value from the stair foot, r_l units, as PPO weighs groups; natural v20 behaviour, pushed v21 behaviour in brackets):
  | config | refuse | climb | fall | attempt - refuse (nat 12.5 % falls) | break-even fall rate |
  | v20 (joint, sigma 0.1) | 16.7 | 20.0 | 9.3 | +2.0 | 31 % (18 %) |
  | v21 (separate, sigma 0.25) | 24.7 | 24.7 | 11.7 | -1.6 | 0.3 % (4.6 %) |
  | v21 + `--adv_norm joint` | 18.6 | 21.4 | 10.1 | +1.4 | 25 % (14 %) |
  | v21 + stall penalty w 1.0 / 3 s (fallback) | 18.4 | 24.2 | 11.7 | +4.2 | ~46 % |
  Change v20 -> v21 = -3.6: style group (separate norm) -1.64, foothold group -1.31, sigma_v -0.65; foot base frame 0; others not in the stair reward.
  Raw per alive s refuse vs climb: r_l 69 vs 98 (sigma 0.1), r_f 44 vs 30, r_s 16.4 vs 7.9. Group std (log): l 0.175, f 0.072, s 0.032.
- Stall penalty (FALLBACK, not applied anywhere live): `runs/refuse_diag/patch_stallpen.py <code_dir>` adds `--stall_w/--stall_s/--stall_vmin`
  (default off): -w per step once a moving command (|cmd| >= 0.3) has made < max(0.10 m, 0.25 |cmd| stall_s) progress for stall_s s; timer
  reset on progress, command change, reset, stand / turn commands (stations hold stand: free). Tested on a patched COPY
  (`runs/refuse_diag/code_stall`, `stall_test.json`, v21 27,500, w 1.0, 3 s): refusers -31.5 /s (71 % of steps), climbers -2.4 /s, falls 0,
  flat walkers at 0.15/0.2/0.35/0.6 m/s 0; a penalised refuser still earns r_l +45.8 /s (> 0, so no pay for falling; w must stay < ~1.3).
- 5,050 -> 7,110 refusals: 5,050 (r7_v14_p250) stalled at "ramps, front side" 36/50 (never reached the stairs); 7,110 = r8_v15_p100, the first
  separate-normalisation run (28/50 at the stair foot). Not the extended map. Ext suspect not tested: fatal wall touch on ext stairs routes.
- What went wrong: E8's check omitted the imagined-swing foothold part (stairs_diag3 calls env.step without the foot net); v21_d40 probes'
  reward columns have lin_vel 0 (sig_v_to 0 in v21 args). Nothing restarted; queue40 / r7 untouched.

## HOW TO RESUME (23:45 UTC 2026-11-14, node clock)
- r7 v21 RUNNING to 30,000 (queue39), then queue40 (tmux `ssr40`) runs `extend_v20` (v20 flags = joint norm, sigma 0.1) to 40,000 - consistent
  with E8c. Open user decision: keep extend_v20, or v21 flags with `--adv_norm joint` (paper sigma 0.25 kept; margin +1.4 vs +2.0).
  Fallback only: stall penalty patch above. Next: leg-5 split on the first v20x exports; ext-stairs fatal wall check (owner SSR agent).
