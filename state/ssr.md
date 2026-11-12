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
- Inputs copied to the shared filesystem: only inside `tracks/ssr/code` (`data/amp.npz`, 47 MB, the motion priors built on the
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
| Code and configs (with my changes) | `tracks/ssr/code/` | me, by scp; the change log is in this file |
| Checkpoints | `tracks/ssr/runs/<run>/last.pt` (every 50 iterations, about every 5 min), `prev.pt` (the one before), `ckpt_NNNNNN.pt` every 1,000 | `ssr.train` |
| Training logs | `tracks/ssr/runs/<run>/log.jsonl` (every 10 iterations), `runs/<run>.log` (process output), `runs/supervisor.log`, `runs/<run>/args.json` | `ssr.train`, supervisor |
| Benchmarks | `tracks/ssr/runs/bench.sh`, `runs/bench.log` | me |
| Motion prior | `tracks/ssr/code/data/amp.npz` (63-D frames per terrain set) | built on the desktop by `scripts/build_amp.py` |
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
`state/ssr_chest_rs_camera_check.png`, also `renders/camera_check/` on the shared filesystem): 0.55 m line at 51.5 % of the
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

Code changes since the copy (all in `tracks/ssr/code`): `ssr/train.py` (flags `--rew_scale --init_std --adv_norm --adv_floor
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
  gait probe in `eval/<name>/`. Exact commands: `tracks/ssr/code/run_commands/queue4.sh`, `queue5.sh`.
- Stop everything: `touch /mnt/filesystem-g4/tracks/ssr/runs/QUEUE5_STOP /mnt/filesystem-g4/tracks/ssr/runs/STOP`.
- Watch: `python3 /mnt/filesystem-g4/tracks/ssr/code/scripts/w.py <run> 10` (speed ratio, support shares, touchdown air
  time, progress, gate, stage ramp, reward, episode length, tilt share, action std).
- **Motion files** (coordinator, user rule "Motion data and the repo", 01:15): `tracks/ssr/code/MOTIONS.md` lists every
  source and built prior (path, bytes, sha256, clips and seconds per set, origin, build command, filters, licence line);
  full numbers in `eval/motion_inventory.json` (`scripts/motion_inventory.py`). The prior files moved out of `code/` to
  `/mnt/filesystem-g4/tracks/ssr/data/priors/` (01:12 UTC); `code/data/amp*.npz` are symlinks to them, so every command with
  `--amp_file data/amp_v2.npz` still works. To restore: copy the files back over the links.

## Status (2026-11-12 02:00 UTC): r6 running

- **r6 training since 01:59:46 UTC** (tmux `ssr6`, `runs/queue6.sh`, supervisor, heartbeat run "r6"). Command (from
  `tracks/ssr/code`): `../env/venv/bin/python scripts/supervise_node.py --name r6 --iters 20000 --num_envs 8192 -- --stage_a 150
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
  platform), 6,219 flat. First iteration: 61 % of commanded speed. Command (from `tracks/ssr/code`):
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
