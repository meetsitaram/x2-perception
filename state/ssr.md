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
