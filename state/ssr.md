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

## Status

(see the end of this file; updated as runs progress)

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
- Standing does not need terrain, side obstacles or the style reward to appear... not yet proven: d4 tests it.

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
   normalisation (style 44 % of the gradient), plant `mc_20261024`, camera `torso60`. The motion prior is the paper's own
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

## Cameras (user spec 2026-11-11: ground from about 10 cm behind the toe tips to at least 2 m ahead, feet in view, range 3 m)

Measured with the ray march on flat ground, plant `mc_20261024`, 36 x 36 image, distances from the front edge of the soles
(`eval/model_mc_20261024.json`; frames in `renders/camera_check/`: flat, a 4x4 at 1.5 m and at 0.3 m, a stair foot at 0.3 m,
standing and a kinematic mid-stride pose, PNG upscaled and the raw 36 x 36 arrays):

| Option | Mount | Tilt below horizontal | Square field of view | Depth range | Height | Ground seen, standing | Width at 0.5 m | Legs in image (standing / mid-stride) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `head` (runs so far) | `head_pitch_link`, (0.0576, 0, -0.0484) | 40 deg | 54.9 deg | 0.1-2.0 m | 1.13 m | 0.61 to 1.56 m | not visible | 17 % |
| `chest` | `torso_link`, (0.155, 0, 0.20) | 75 deg | 54.9 deg | 0.1-2.0 m | 1.00 m | -0.03 to 1.21 m | 1.13 m | 8 % / 12 % |
| `torso60` (new, from r5 on) | `torso_link`, (0.0488, 0, 0.4378) **provisional** | 60 deg | 64 deg | 0.1-3.0 m | 1.22 m | 0.05 to 3.06 m; the rays below that land on the feet (unoccluded they would reach 0.13 m behind the toes) | 1.63 m | 6 % / 7 % |

`torso60`: camera frame x forward, y left, z up in the parent body, rotated about y by the tilt; rays on a regular 36 x 36
tangent-plane grid; depth is distance along the optical axis, clamped to the range and scaled to 0-1; no crop. The mount point
is the Hiking track's G1 torso-camera offset placed on the X2 torso link with y centred (for mirror symmetry); it is to be
replaced by the X2 offset the Hiking track fixes. A real sensor needs 64 deg in the image's vertical direction (a D435i's
58 deg does not reach; a Gemini 335 at 65 deg or a D435i turned to portrait does). In the frames the feet and hands are
visible; a 4x4 at 1.5 m is one to two pixel rows, at 0.3 m about three. The paper's waist camera pose and field of view are
not given in the paper, so its footprint cannot be computed for comparison; `head` and `chest` remain selectable.
