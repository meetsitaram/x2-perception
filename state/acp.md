# ACP track state

Updated 2026-11-11 21:27 UTC (track agent). **Stage 1 is training** on node acp: run `s1_v1`, tmux session `acp`, 36,000 envs since iteration 750; backup loop in tmux session `acp-backup`; see section 4.

## 0. Answer to the question that decides the track

**The paper is RL (PPO), and its RL policy is itself a tracker trained on reference motions.** ACP does not put its contact
planner on a separate pretrained motion tracker; it trains its own "contact-conditioned tracker" with PPO, and in stages 1-2
(and on part of the environments in stage 3) that training uses retargeted reference motions through the reward, the resets,
the critic and the contact labels. The policy's inputs never contain the reference. Quotes (arXiv 2610.31608v1, HTML):

- Abstract: "using motion data strictly as a training prior. An explicit, self-supervised, attention-based contact planner
  predicts future body-environment contacts, while a contact-conditioned tracker translates these plans into coordinated
  whole-body motion."
- Materials and Methods, "Three-Stage Training Pipeline": "The first stage trains the tracker alone on a curated
  reference-motion dataset covering climbing, stair traversal, walking on rough ground, and stepping-stone gaits, with the
  simulation terrain spatially aligned to each reference clip. The tracker receives ground-truth contact positions read
  directly from the clips in place of planner output".
- "Reward Functions": "Stage 1 supervises the tracker with ground-truth contact and reference-proprioception targets from the
  motion-capture corpus" (Table 3: track_base_pos 3.0, track_base_vel 1.0, track_joint_pos 2.5, track_joint_vel 0.5 in all
  three stages, reference environments only).
- "Contact-Informed Whole-Body Tracker": "contact commands provide sufficient guidance for executing the demonstrated
  whole-body skills without additional reference kinematics as tracker inputs."
- "Motivation": "reference motions act solely as training guidance and are entirely eliminated at deployment".
- "Training": "We train all policies in Isaac Lab. The contact tracker is trained using a customized Proximal Policy
  Optimization (PPO) algorithm, while the contact planner is trained with supervised learning on rollout data collected from
  the tracker."

So "no SuperMimic" is fully compatible with the paper: SuperMimic is the user's model-based trainer (world model + gradients
through imagined rollouts, MuJoCo Warp). The paper-faithful setup replaces that trainer with PPO and keeps what the paper
also has: X2 reference clips on clip-aligned terrain with nine-body contact labels, used as reward/reset/critic guidance.
It cannot be done "without any reference motion": the paper's stages 1 and 2 need them.

## 1. The paper's pipeline, stage by stage

Robot in the paper: Unitree G1, 23 controlled joints. Simulator: Isaac Lab. Policy 50 Hz, joint PD 200 Hz, action = 23
joint-position offsets x 0.25 + default pose.

**Data (before stage 1).** Human motion retargeted with a contact-aware IK tool (joint limits, foot non-penetration, minimum
distance to obstacles, temporal smoothing); contacts of nine bodies (two feet, knees, hands, elbows, pelvis) recorded per
frame "from the simulated contact geometry", with world contact points. Sources (Table S1): box climbing (OmniRetarget),
stairs / ladder / sit down (Mixamo), walk / get up (LAFAN), vault (Kimodo), sparse terrains (rollouts of an AME policy).
123 reference configurations. Clip count and hours are not stated.

**Stage 1: tracker alone, RL (PPO), oracle contacts.**
- Actor (468): 5-step proprioceptive history (375: base angular velocity, projected gravity, joint positions, joint
  velocities, previous actions), 9 x 7 height scan at 0.1 m (0.8 x 0.6 m), 9 contact targets (27, body-yaw frame, zero when
  invalid), goal command (3: bounded horizontal direction d/max(|d|, r) + relative heading; r = 0.3 m; target = the reference
  0.5-1.5 s ahead). No reference kinematics in the actor.
- Critic V_m (353, privileged): noise-free current proprioception, 17 x 11 scan, base linear velocity, 52-d reference
  kinematics, 27-d reference contact-body positions, 9 reference contact flags.
- Network: MLP 512-256-128 ELU for actor and critics.
- Rewards (Table 3; per step dt * sum): track_base_pos 3.0, track_base_vel 1.0, track_joint_pos 2.5, track_joint_vel 0.5,
  track_contact_label 5.0, track_contact_pos_base 2.0, track_contact_pos_global 1.0; action_rate -0.1, action_rate_2nd -0.05,
  joint_torque -1e-5, torque_limits -1e-3, dof_pos_limits -10, dof_vel_limits -1, contact_acc -0.01. Kernels: sigma_p 0.5,
  sigma_r 1, sigma_v 1, sigma_w 3, sigma_q 2, sigma_qd 10, sigma_c 1.
- Resets: Curriculum State Initialization: start phase 0.9, -0.05 after an episode reaches phase 0.99, +0.05 otherwise,
  clipped to [0, 0.99].
- PPO (Table S6): 12,000 envs per worker (3 GPUs, about 1 day), 24 steps, 5 epochs, 8 minibatches, clip 0.2, value coef 1.0,
  grad norm 1.0, gamma 0.99, lambda 0.95, lr 1e-4 KL-adaptive (target KL 0.02), entropy 0.006.
- Terrain: reference-anchored (aligned to each clip).

**Stage 2: planner (supervised) + tracker co-trained (PPO), same terrains.**
- Planner input: 5-step proprioceptive history, goal (3), 17 x 11 scan of 3-D points (1.6 x 1.0 m). Output per body: a
  distribution over the 187 scan points (attention, d = 96, 32 heads, FiLM body queries, conv encoder 1-16-48 5x5; 786,993
  parameters; top-k decode, k = 1) and an occupancy probability (used if > 0.5, else target zeroed). Horizon 1.5 s.
- Planner targets: hindsight labels from the tracker's own successful rollouts (not from the dataset): next contact onset per
  body (vertical force > 1 N, segments >= 10 steps), within 1.5 s and within the footprint [-0.8, 0.8] x [-0.5, 0.5] m.
  Success for reference environments: started in the first 10 % of the clip and reached 90 %. Loss: masked cross-entropy to a
  Gaussian (0.1 m) over scan points + BCE on occupancy. Adam 1e-4, 10 updates per PPO iteration, 4,096 samples, after 1,000
  replay entries.
- Tracker: contacts in the input switch from reference to planner with probability eta_obs per env and step; later
  eta_rew moves the reward: reference-contact weights x (1 - eta_rew), track_planned_contact 1.5 x eta_rew (linear ramps,
  Eq. 3; start iterations and length not given). Mimic terms unchanged. Initial phase U(0, 0.98). 4,800 envs, target KL
  0.01, entropy 0.005. About 1 day 21 h on 2 GPUs.

**Stage 3: RL fine-tuning on procedural goal terrains + planner refined from successes.**
- Adds goal-reaching terrains with no reference (stairs, walled stairs, platforms, rough ground and slopes, gaps, beams,
  pallets, stones, boxes, get-up, ladders, vaults; 32 configurations, 10 difficulty levels); reference environments stay,
  "through proportional sampling from the original dataset" (ratio not given).
- Goal environments: goal command from the terrain goal (r = 0.5 m); rewards move2goal 40, standatgoal 20,
  track_planned_contact 1.5, flat_orientation -10, stumbling -10, slippage -10, ang_vel_xy -0.1, plus the regularisers;
  no mimic terms. Speed cap 0.8 m/s on stairs, 2.0 elsewhere. Episode 15 s.
- Second critic V_fs (298: proprioception, base velocity, scan, goal, goal distance, dz, normalised time, planned contacts),
  fixed lr 1e-3, 500-iteration warm-up (goal samples excluded from the actor loss); advantages normalised separately per
  environment type.
- Curriculum: promote on success with EMA (0.8 s + 0.2 I) > 0.5; demote when the closest approach is > 1.0 m.
- Planner keeps training from successful rollouts (goal success: within 0.5 m horizontally and 0.1 m vertically), buffer
  partitioned per tile, quality = low positive mechanical work per metre.
- 8,400 envs on 4 GPUs, about 2 days 6 h; target KL 0.01, entropy 0.002.

**Domain randomisation (Table S2, all stages):** friction U(0.3, 1.2), restitution U(0, 0.5), torso mass U(-2, 4) kg, leg
mass x U(0.85, 1.15), torso CoM +-0.025 / 0.05 m, joint zero bias +-0.01 rad (ankle +-0.03), armature and PD gains x
U(0.9, 1.1), delay U(0, 20) ms; observation noise (joint vel +-1.5 rad/s, joint pos +-0.01, ang vel +-0.05, gravity
+-0.0125, contact targets +-0.1 m, goal +-0.2); tracker scan U(-0.1, 0.1) m, planner scan N(0, 0.03) m; odometry delay
50-100 ms and drift. Reset joint noise +-0.2 rad, base +0-0.05 m, goal yaw +-0.4 rad.

**Reported results:** contacts-only tracker >= 98.5 % on reference categories; eight composite courses 92.5 % average (MGMT
58.8 %); evaluated with noise, physics randomisation and delay disabled. Not given in the paper: termination thresholds,
iteration counts, eta schedule numbers, reference / goal mix in stage 3, no code release found (not searched beyond the paper).

## 2. Inventory of `gwbc-v6` (read only; nothing there was changed)

SuperMimic there = `gear_sonic/supermimic/train.py`: collect in MuJoCo Warp -> fit a world model -> policy and critic by
backpropagation through imagined rollouts. The issue log's own root-cause list blames that trainer for several stage-3
failures ("Goal reaching cannot be optimised by SuperMimic's analytic gradients", world-model divergence with walls, goal
rewards scaled to 0.125, planner frozen in stage 3).

**Reusable (method-neutral or matches the paper):**
- X2 model: `gear_sonic/data/assets/robot_description/mjcf/x2_ultra.xml`, URDFs, `envs/manager_env/robots/plant_config.py`
  and `x2_ultra.py` (PD gains, armature, effort), `supermimic/assets.py` (`build_x2`: 5 ms, PD servos, 4 substeps),
  `kinematics.py`.
- Simulator layer: `supermimic/sim.py` (batched MuJoCo Warp, DR, pushes), `terrain.py` (box props, height raster, walls,
  9 x 7 scan), `motion.py` (clip library at 50 Hz), `contacts.py` (nine-body labels, 1.5 s horizon), `goal.py` (Table 3 goal
  terms; its sigmoid relaxations and reward scale are SuperMimic adaptations, to be removed), `planner_contacts.py` (eta
  schedules, decode), `self_clear.py`. These sit in the supermimic package but are not the SuperMimic method.
- Data tools: `scripts/acp/data/ref_labels.py` (contact labels straight from references: the paper's oracle contacts),
  `gen_stairfit*.py` (Bones-SEED stair clips fitted to generated flights, never the apparatus' sizes), `synth_stairs.py`;
  `loco-manip-apparatus/sim/pipeline` (`stairfit`, `contacts`, `hmap`, `walls`, `props`, `terrain_park`). (`stage2/make_walls.py`, named in the issue log, is not in the repo: corridor walls to be rewritten.)
- Data (desktop, `loco-manip-apparatus/build/train`, 730 MB): kinematic clip sets with props (`x2_loco_*`, `x2_chains_far`,
  `pico_2026-10-29_2x2_*`, `tp_stairs_*`, `tp_ramp_*`, `tp_t4x4u_y20`, `gst_v1`, `gs2_v1`, `gs2dn_v1`) and sampling weights.
- Stage 2: `stage2/planner_data.py` (hindsight labelling, the paper's rule), `stage2/planner.py` (the paper's MLP baseline,
  feet only: the attention planner for nine bodies has to be written).
- Stage 3: `stage3/terrain_gen.py`, `stair_world.py` (procedural tiles, 10 levels), `park_roam.py`.
- Eval and renders (for my own quick checks; the eval monitor scores): `eval/contact_metrics.py`, `foothold_match.py`,
  `scorecard.py`, `rollout_contacts.py`, `render_full*.py`, `render_crowd.py`, `variants/*`, `stage3/route_nist.py`,
  `route_full.py`.
- Lessons carried over (ISSUES_LOG): tall geometry as wall boxes, not heightfield columns (catapult); walls must be in the
  scan and in training; fall test over the lowest ground under root and feet; no reference-relative termination in goal
  runs; goal length encodes speed; >= 6 repeats (GPU physics not deterministic); MuJoCo Warp degrades on large heightfields
  and every distinct box size is a slot in every world; sigma_c = 1 m gives no landing pressure (kept as in the paper for
  the first run, listed as a known weakness); no GPU job beside training; curriculum state in the checkpoint; turning in
  place and side-stepping were never trained.

**Depends on SuperMimic, not reused:** `supermimic/train.py`, `networks.py` (world model, policy with xi context),
`replay.py`, `mdp.py` reward relaxations, all `launch/*.sh`, every checkpoint in `logs_sm/`; `data/acp_labels.py`,
`pico_rollout*.py`, `screen_*.sh` and the `acp_*_contacts.pkl` / `acp_*.pkl` sets (labels and physics screening made from
SuperMimic checkpoints' rollouts); `eval/pico_rollout_planner.py` (loads SuperMimic policies; the rollout logic will be
rewritten around the new policy); `planner_v*.pt`.

**To build:** PPO trainer with asymmetric actor and two critics as Table S6; the paper's reward set without relaxations;
CSI resets; attention planner (Table S3); hindsight replay with online planner updates in stages 2-3; reference contact
labels for all clip sets with `ref_labels.py`.

## 3. Decisions taken without the user (reason)

- Reference motions are used as the paper uses them (reward, reset, critic, oracle contacts); only the SuperMimic trainer and
  its checkpoints and rollout-derived labels are dropped. Reason: section 0.
- Simulator for the first run: MuJoCo Warp through the existing X2 sim layer, with a new PPO trainer; the paper uses Isaac
  Lab. Reason: the X2 model, props, scan and data already run there, and the common eval only counts a number that holds in
  MuJoCo. Gate: the smoke run must reach a useful PPO throughput at >= 4,096 envs on the node; otherwise port to Isaac Lab.
  MEASURED 2026-11-11: 59,306 env-steps/s at 4,096 envs and 109,234 at 12,000 (physics + observations, random actions);
  about 102,000 with the PPO updates in the real run (2.8 s per iteration). Gate passed, no Isaac Lab port.
- Paper values everywhere they are given; where the paper is silent (terminations, eta schedule, stage-3 mix) the values are
  written here when chosen.
- Known differences from the paper so far: X2 instead of G1 (31 hinges; controlled-joint set to be fixed at setup);
  MuJoCo Warp instead of Isaac Lab (no restitution); data = Bones-SEED X2 retargets and Pico captures on stairs, ramps,
  4x4 stepovers and blocks, not the paper's ladders, vaults or get-ups; one GPU instead of 2-4.

- Install-time licence and confirmation prompts on the node (NVIDIA, conda, apt, pip, simulator EULAs) are accepted without
  asking (user via coordinator, 2026-11-11). Covers installs on this node only: no purchases or subscriptions.

## 4. What runs now (stage 1, PPO, oracle contacts)

- Node: `ssh -i ~/.ssh/nebius_corpus -o IdentitiesOnly=yes -o BatchMode=yes <user>@<acp-node>`; tmux session `acp`.
- Run folder: `/mnt/filesystem-g4/tracks/acp/runs/s1_v1` (`train.out`, `log.jsonl`, `supervisor.log`, `ckpt_<it>.pt` every 250
  iterations, `latest.pt`). Newest checkpoint: see `newest_checkpoint` in `/mnt/filesystem-g4/status/acp.json` (heartbeat,
  rewritten every 10 iterations, about 30 s).
- Started 2026-11-11 20:02 UTC with 12,000 envs; stopped at checkpoint 750 (20:40) for the GPU-use benchmark below and
  restarted 20:51 from `latest.pt` (iteration 750) with **36,000 envs, njmax 450, 12,000 iterations in total** (about 6.6 s per
  iteration, about 21 h left). Resume from a checkpoint is thereby verified (learning rate, normalisers and optimizer
  restored; the per-environment reset phases were refilled with their mean, 0.28, because the env count changed).
- Start / resume (the supervisor resumes from `latest.pt` by itself, restarts on exit or on 15 min without a log line, waits
  if another process holds the GPU, gives up after 20 restarts):
  `tmux new -d -s acp "bash /mnt/filesystem-g4/tracks/acp/code/supervise.sh /mnt/filesystem-g4/tracks/acp/runs/s1_v1 36000 12000"`
  (njmax 450 is the supervisor default; `NJMAX=...` overrides it)
- Stop for good: `touch /mnt/filesystem-g4/tracks/acp/runs/s1_v1/STOP`.
- Watch: `tail -f /mnt/filesystem-g4/tracks/acp/runs/s1_v1/train.out | grep "^it "`.
- Environment: venv `~/acp_env` on the node's local disk (python 3.12, torch 2.11.0+cu128, mujoco 3.17.0, mujoco-warp 3.17.0,
  warp-lang 1.18.0: the desktop's versions); lock file `tracks/acp/env/requirements.lock.txt`. Rebuild:
  `python3 -m venv ~/acp_env && ~/acp_env/bin/pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128 && ~/acp_env/bin/pip install mujoco==3.17.0 mujoco-warp==3.17.0 warp-lang==1.18.0 numpy joblib pyyaml scipy`.

### Where every file is, and how to restore (backup rule, 2026-11-11)

Node names only here; addresses and the login are in the git-ignored `config/nodes.local.json`.

| What | Where (all under `/mnt/filesystem-g4/tracks/acp/` unless said) | Written by | Restore |
|---|---|---|---|
| Checkpoints | `runs/<run>/ckpt_<it>.pt`, `latest.pt` | trainer, every 250 iterations, straight to the shared filesystem | supervisor resumes from `latest.pt` |
| Training logs | `runs/<run>/train.out`, `log.jsonl`, `supervisor.log` | trainer / supervisor, shared filesystem | read in place |
| Code and configs | `code/` (master copy); a snapshot per run in `runs/<run>/code_snapshot/` | me (rsync from the desktop scratchpad); snapshot by the backup loop | copy the snapshot back to `code/` |
| Clip sets, props, ramp rasters | `data/clips/` | copied once from the desktop (`loco-manip-apparatus/build/train`) | re-copy from the desktop, list in "Copies made" |
| Reference-derived contact labels | `code/cfg/labels_stage1_sets.json.pt` (cache) | `acp/data.py` at first start | delete it: it is regenerated from the clips in about a minute |
| X2 model files | `/mnt/filesystem-g4/data/robots/x2/` and `code/gear_sonic/...` | copied once from gwbc-v6 | re-copy, see its README |
| Benchmark and diagnostic outputs | `backup/tmp_outputs/` (from the node's `/tmp/*.out`) | backup loop | read in place |
| Anything a job writes on the node's local disk | `backup/local/` (mirror of `~/acp_local` on the node; empty so far) | backup loop | rsync back |
| Python environment | NOT copied (node local `~/acp_env`); pinned list `env/requirements.lock.txt`, refreshed by the backup loop | | rebuild with the commands in "What runs now" |
| Heartbeat | `/mnt/filesystem-g4/status/acp.json` | trainer | |
| Manifest of the last backup pass | `backup/MANIFEST.txt` | backup loop | |

Backup loop: `code/backup.sh`, tmux session `acp-backup` on node acp, one pass every 20 minutes; the supervisor starts it if
it is not running (`tmux new -d -s acp-backup "bash /mnt/filesystem-g4/tracks/acp/code/backup.sh"` by hand). Each pass mirrors
the local-disk items above, snapshots the code next to every run, prunes checkpoints to the newest two plus every multiple
of 1,000 plus `latest.pt`, and rewrites the manifest. Checked 2026-11-11 21:24 UTC: first pass ran, `ckpt_000250` and
`ckpt_000500` pruned, `ckpt_000750` and `ckpt_001000` kept. Only on the node's local disk now: the Python environment and
the throw-away benchmark run folders in `/tmp` (their outputs are mirrored). Not a second copy: runs, logs, code and clips
exist once, on the shared filesystem (plus the desktop sources for clips and model files); the shared filesystem itself is
the backup the rule asks for.

### GPU use (user: "make sure the GPUs are well used"), measured 2026-11-11 20:41-20:50 UTC

Full training loop (physics + observations + PPO update), fresh policy, 20 iterations each, njmax 300:

| Envs | Env-steps/s | vs 12,000 | GPU memory | GPU utilisation |
|---|---|---|---|---|
| 12,000 (the run until iteration 750) | 93,000-103,000 | | 19.7 GB | 84-88 % |
| 24,000 | 123,337 | +25 % | 32.0 GB | 81 % |
| 36,000 | 131,286 | +34 % | 45.6 GB | 92 % |
| 48,000 | 134,899 | +38 % | 58.9 GB | 88 % |
| 36,000, njmax 450 | 127,976 | +30 % | 46.9 GB | |

In the real run after the restart (trained policy, njmax 450): 116,000-117,000 env-steps/s, GPU 100 %, 46.9 GB: about +20 %
over the 12,000-env run, less than the fresh-policy benchmark. The card's compute, not its memory, is the limit: the physics
already saturates it, so the 49 GB still free cannot be turned into speed, and 48,000 adds only 3 % over 36,000. The 24 CPU
cores stay idle because simulation and learning are both on the GPU (nothing in this pipeline is CPU work).

Chosen: 36,000 envs = the paper's total (12,000 x 3 workers). Kept fixed: 24 steps per iteration, 5 epochs, 8 minibatches,
initial learning rate 1e-4 with the KL-adaptive schedule (target 0.02), clip 0.2, entropy 0.006. Rollout batch 288,000 ->
864,000 transitions (= the paper's 3 x 288,000); minibatch 36,000 -> 108,000 (the paper: 36,000 per worker, gradients
presumably averaged over 3 workers = the same 108,000 effective). Iteration budget cut from 30,000 to 12,000 so the total
sample budget stays in the same range (about 10 billion transitions).

njmax raised 300 -> 450: MuJoCo Warp printed "nefc overflow - please increase njmax beyond 300" 102 times in the 12,000-env
run (constraint rows dropped in some worlds); none in the first iterations at 450.

### Stage 1 so far (iteration 1,010, 21:24 UTC)

| Iteration | Reward per step (max 0.30) | Start phase (mean) | Finished | Base pos error | Joint error |
|---|---|---|---|---|---|
| 200 | 0.100 | 0.73 | 0.52 | 0.11 m | 0.179 rad |
| 700 | 0.077 | 0.33 | 0.57 | 0.145 m | 0.178 rad |
| 1,010 | 0.098 | 0.03 | 0.76 | 0.138 m | 0.171 rad |

The reset curriculum reached the clip starts (phase 0.03; 10 % of envs at 0.0, 90 % below 0.1) at about iteration 1,000, so
episodes are now whole 10 s segments; since then reward and finish rate rise and the errors fall slowly. Orientation error
0.20 rad, contact-point error 0.087 m, contact agreement 93.7 %. Finish rate per set at iteration 1,010: Pico 2x2 0.96,
ramps 0.87, flat 0.85, blocks 0.83, tp stairs 0.79, 4x4 0.77, chains 0.69, fitted stair climbs 0.67, **stair descents 0.10**.
Weak: stair descents (38 clips, 3.1 min, 5 % share); joint tracking is loose (reward term 0.64 of 1). To judge at iteration
2,000: if descents stay under 0.3, raise their share and look at the failures; if joint error stays near 0.17 rad, tighten
the termination (paper-silent) rather than the paper's reward kernels.

### Code (`/mnt/filesystem-g4/tracks/acp/code`; the node is the master copy, the desktop copy is only in a session scratchpad)

- Rebuilt for this round: `acp/data.py` (clip sets, 10 s segments, reference-derived nine-body contact labels),
  `acp/env.py` (MuJoCo Warp env: paper observations, Table 3 stage-1 rewards, CSI resets, Table S2 randomisation, contact
  forces from the solver), `acp/train.py` (PPO, Table S6), `supervise.sh`, `tools/diag.py`, `cfg/stage1_sets.json`.
- Reused unmodified from gwbc-v6 (commit df16c157): `gear_sonic/supermimic/{assets,kinematics,motion,terrain,robot_cfg}.py`
  (X2 model build, FK, clip library, props / raster / height lookup), `plant_config.py`, `config/robot_plant/*.yaml`,
  `x2_ultra.xml`, `urdf/x2_ultra/`. No SuperMimic trainer, network, checkpoint, rollout or label file is on the node.

### Copies made (2026-11-11, from the desktop, rate-capped)

- `/mnt/filesystem-g4/data/robots/x2/` (112 MB, with README): `mjcf/x2_ultra.xml`, `urdf/x2_ultra/` (URDFs + meshes),
  `robot_plant/*.yaml` + `plant_config.py`, from gwbc-v6 df16c157. No USD.
- `/mnt/filesystem-g4/tracks/acp/data/clips/` (93 MB) from `loco-manip-apparatus/build/train`: the sets below plus
  `x2_loco_flat_test`, `x2_loco_blocks_test`, `pico_2026-10-29_2x2_test` (held out, not trained on), `tp_ramp_hmap*.npz`.

### Clip sets in training (all kinematic; labels re-derived from the references by `acp/data.py`)

| Set | Clips (10 s segments) | Minutes | Sampling share | What |
|---|---|---|---|---|
| x2_loco_flat_train | 419 | 49.8 | 0.20 | Bones-SEED locomotion retargets, flat |
| x2_loco_blocks_train | 419 | 49.8 | 0.20 | same clips with block props |
| x2_chains_far | 210 | 28.1 | 0.15 | generated chains over boxes and stairs |
| tp_stairs_x10_sh2 | 88 | 8.2 | 0.10 | terrain-park stair captures, shifted variants |
| gst_v1 | 113 | 9.4 | 0.10 | Bones-SEED stair climbs fitted to generated flights (not the apparatus' sizes) |
| gs2dn_v1 | 38 | 3.1 | 0.05 | stair descents on generated flights |
| tp_ramp_sh2 | 330 | 43.9 | 0.10 | terrain-park ramp captures (moved +30 m in x onto the ramp raster) |
| tp_t4x4u_y20 | 56 | 3.6 | 0.07 | 4x4 stepover captures |
| pico_2026-10-29_2x2_train | 10 | 0.8 | 0.03 | Pico captures over a 2x2 |

Total 1,683 segments, 196.7 min. Feet in contact 72 % of frames; knees 0.3 % (label noise on locomotion); hands, elbows,
pelvis never: this data has no whole-body contacts.

**Left out:** the apparatus and NIST route sets (`x2_route.pkl`, `x2_route_pass.pkl`, `x2_route_tcrs.pkl`,
`x2_nist_route_pass.pkl`: whole-layout routes, held out by the eval rule); every `acp_*.pkl` and `*_contacts.pkl` (screened
or labelled from SuperMimic rollouts); `gs2_v1` (kinematic, but 239 distinct box sizes = 239 prop slots in every world: to
be re-quantised before use); the older `tp_*` variants superseded by `_sh2`; `x2_chains.pkl` (same clips as `_far`).
Apparatus parts present as re-arranged pieces: a 4x4 stepover, stair flights (other sizes), a ramp cell. Not verified: that
the terrain-park ramp raster is not a 1:1 copy of the apparatus' crossing-ramp cell (a single cell would be allowed anyway),
and how the `tp_*` and `x2_chains` kinematic fits were originally produced (they were taken as kinematic from their names
and the README; no SuperMimic file is read for them here).

### Checks done (`tools/diag.py`, 512 envs, references replayed open-loop through the PD targets)

Total vertical contact force on the nine bodies 389 N (robot about 39 kg): the contact-force read-out is right. Measured vs
labelled contact agree on 95.4 % of body-frames. Observations finite, actor 588, critic 393.

### Differences from the paper in this run (reason)

1. X2 Ultra, 31 actuated joints (the paper: G1, 23); action scale per joint from the X2 plant file, not 0.25.
2. MuJoCo Warp instead of Isaac Lab (coordinator decision); no restitution randomisation (MuJoCo has none).
3. 36,000 envs on one GPU from iteration 750 (the paper: 12,000 per worker on 3 GPUs); 12,000 before that.
4. Data: locomotion, stairs, ramps, stepovers; no ladders, vaults, sitting or get-ups, so non-foot contacts are untrained.
5. Paper-silent choices: early termination at 0.5 m base position error or 1.0 rad orientation error against the reference;
   joint soft limits = MJCF range; action std init 1.0; running observation normalisation; clipped value loss; CSI phase kept
   per environment; goal lookahead drawn per episode; clips cut into 10 s segments so an episode can reach phase 0.99.
6. Tracker scan noise: the paper's text (per-step Gaussian + per-episode offset) and Table S2 (U(-0.1, 0.1) m) are read as
   per-episode offset U(-0.1, 0.1) m + per-step N(0, 0.03) m.
7. Not implemented: odometry delay and drift (covered only by the +-0.1 m contact-target noise).
8. Contact "measured" = solver normal force > 1 N at the end of the control step (not over a sensor history window).
9. MuJoCo Warp prints "height field collision overflow (>= 50)" on the ramp raster (4 cm cells, coarsened from 2 cm) when a
   large body lies on it; contacts beyond 50 are dropped. Not fixed.

### Code release check

None found (2026-11-11): the arXiv page says only "under review", no code or project link; one web search found no
repository. Not exhaustive.

## 5. Next

1. Watch stage 1: CSI phase should fall from 0.9 toward 0, finish fraction per set, tracking errors. First checkpoint at
   iteration 250.
2. Meanwhile write stage 2: attention planner (Table S3), hindsight replay from successful rollouts, eta schedules.
3. Then stage 3 goal terrains with walls, in-place turns and the re-arranged apparatus parts.

Blockers: none.
## 6. The paper's own evaluation, and ours of it on the X2

What the paper evaluates (Results 0.3 "Simulation-based Evaluations", Figure 7, supplementary "Evaluation Details"):

- **Terrain groups:** (1) reference-anchored: "The same 123 reference configurations used for training"; (2) goal-reaching: the
  32 stage-3 configurations at difficulty levels sampled from the training range; (3) unseen: compositions (two goal-reaching
  terrains in sequence), distractors (extra obstacles around the support area), roughness (bumps of 5-15 cm); plus eight
  curated composite courses (Figure S1). Groups 1 and 2 each have four categories: dense foot-only, sparse foot-only, dense
  whole-body, sparse whole-body.
- **Success:** "success requires reaching within 0.5 m of the active goal horizontally; for sitting, standing, and get-up
  transitions, the criterion is instead a vertical error below 0.1 m." Composite courses also require the target height and
  staying in the lane and out of the pit. Invalid policy outputs count as failures.
- **Trials:** 3,000 episodes per method and category over three seeds; 1,000 per radar axis and per composite course.
- **Second metric:** contact placement error = "the Euclidean distance between commanded and realized contact positions at
  contact onset, averaged over valid contact events".
- **Intervals:** Wilson 95 % for success, episode-clustered 95 % for contact error.
- **Conditions:** "Sensor noise, physics randomization, and action delay are disabled".
- **Comparisons:** stage-1 tracker with contacts only vs contacts + reference kinematics (Fig. 7A); attention vs MLP planner
  (7B, 7D); stage 2 vs stage 3 (7C); MGMT baseline (7E); a qualitative table against PHP and LLP (Table 1).
- **Numbers the paper states in text** (the rest are bars in Figure 7): stage-1 contacts-only tracker "at least 98.5% success
  in every category" on reference-anchored terrains; stage 3 vs stage 2 on sparse whole-body goal terrains +54.8 points;
  composite courses 92.5 % average vs 58.8 % for MGMT. Data length in Table 1: 13.2 min (ours: 196.7 min).

Our build for the X2: `code/tools/paper_eval.py CKPT OUT.json [episodes per set per seed]`, results in
`/mnt/filesystem-g4/tracks/acp/eval/paper_eval/`. Reference-anchored group, stage 1: every clip set, 334 episodes per set
per seed x 3 seeds (9,018 episodes, about 1,000 per set), each from the start of a 10 s segment with the oracle contacts,
deterministic actions, noise / randomisation / delay off. Success = not ended early AND within 0.5 m (horizontal) of the goal
= the reference root at the end of the segment. Reports per set and pooled as the paper's "dense foot-only" category: success
with Wilson 95 % interval, share ended early, contact placement error at onset with a 95 % interval over episode means. It
needs the GPU, so it runs only while training is paused at a checkpoint (a few minutes) or between stages: one GPU job at a
time. It is a track report, not the team's apparatus score.

Cannot be reproduced now, and why: the sparse foot-only, dense whole-body and sparse whole-body categories (no X2 reference
data for stepping stones, beams, box climbs, get-ups, ladders, vaults; several are whole-body skills the X2 data never had);
the goal-reaching and unseen groups and the composite courses (they need stage 3 and its terrain suite; planned with our
terrain families, so the numbers will be for our suite, not the paper's 32 configurations); the MGMT baseline and the
MLP-planner and kinematics-conditioned ablations (extra training runs; not planned unless asked); real-robot results.
Stricter than the paper in one point: an episode that ends early under my failure conditions fails even if it was within
0.5 m of the goal (the paper does not say how early terminations are counted).

Results (filled in as checkpoints are evaluated; paper figure beside ours):

| Checkpoint | Category | Our success (95 % interval), episodes | Paper | Our contact placement error | Paper |
|---|---|---|---|---|---|
| (pending: first run at the next checkpoint pause) | dense foot-only, reference-anchored | | >= 98.5 % (stage 1, every category) | | bars only (Fig. 7C) |

## 7. Log legend (`runs/<run>/metrics.jsonl`, one JSON row per PPO iteration, on the shared filesystem)

`log.jsonl` (every 10 iterations, older and shorter) and `train.out` (console) are kept as before. `metrics.jsonl` starts at
the iteration of the logging restart; earlier iterations exist only in `log.jsonl` (raw term values, no termination causes).
Every value is a mean over all environments and the 24 steps of the iteration unless said.

| Name | What it measures | Paper (Table 3 unless said) | Weight |
|---|---|---|---|
| `raw/track_base_pos` | exp(-chi_mov |r - r*|^2 / 0.5^2 - theta_e^2 / 1^2), base position and orientation vs reference | track_base_pos | 3.0 |
| `raw/track_base_vel` | exp(-|v - v*|^2 / 1^2 - |w - w*|^2 / 3^2) | track_base_vel | 1.0 |
| `raw/track_joint_pos` | exp(-|q - q*|^2 / 2^2), 31 joints | track_joint_pos | 2.5 |
| `raw/track_joint_vel` | exp(-|qd - qd*|^2 / 10^2) | track_joint_vel | 0.5 |
| `raw/track_contact_label` | mean over 9 bodies of c_meas (2 c_ref - 1) | track_contact_label | 5.0 |
| `raw/track_contact_pos_base` | exp(-sum_b |p_exec,B - p_ref,B|^2 / 1^2) | track_contact_pos_base | 2.0 |
| `raw/track_contact_pos_global` | exp(-chi_mov sum_b |p_exec,W - p_ref,W|^2 / 1^2) | track_contact_pos_global | 1.0 |
| `raw/action_rate` | |a_t - a_t-1|^2 | action_rate | -0.1 |
| `raw/action_rate_2nd` | |a_t - 2 a_t-1 + a_t-2|^2 | action_rate_2nd | -0.05 |
| `raw/joint_torque` | sum_j tau_applied^2 | joint_torque | -1e-5 |
| `raw/torque_limits` | sum_j |tau_computed - tau_applied| | torque_limits | -1e-3 |
| `raw/dof_pos_limits` | sum_j joint-range violation (rad), capped at 30 | dof_pos_limits | -10 |
| `raw/dof_vel_limits` | sum_j min(1, [|qd| - 0.9 qd_max]+) | dof_vel_limits | -1 |
| `raw/contact_acc` | sum_b |acceleration of the contact point|, capped at 1e4 | contact_acc (style) | -0.01 |
| `weighted/<term>` | dt x weight x raw = that term's share of the per-step reward (dt = 0.02 s) | "per-step reward is dt sum w_i r_i" | |
| `reward_per_step`, `reward_max_per_step` | sum of the weighted terms; the value if every positive term were 1 (0.30) | | |
| `err/base_pos_m`, `err/base_rot_rad` | |r - r*|, theta_e | Table 2 symbols r_b, theta_e | |
| `err/joint_pos_rad_mean_abs` | mean |q - q*| per joint | | |
| `err/contact_pos_base_m` | mean per-body |p_exec,B - p_ref,B| | | |
| `err/contact_placement_at_onset_m`, `contact_onsets` | distance between the commanded target and the contact point when a measured contact starts; how many such events | Results 0.3 "Contact placement error" (Fig. 7C, 7D) | |
| `contact/label_agreement` | share of body-steps where measured contact equals the reference label | | |
| `contact/feet_ref_frac`, `contact/feet_measured_frac` | share of foot-steps in contact, reference vs measured (double-support check) | | |
| `speed/base_xy_mps`, `speed/ref_xy_mps` | actual vs reference horizontal base speed (standing-still check from the shared lessons) | | |
| `episodes`, `end/finished_phase099` | episodes ended this iteration; share that reached phase 0.99 of the clip without failing | Table S4 "Completion threshold 0.99" | |
| `end/timeout_before_phase099` | share cut at the clip end below phase 0.99 (should be 0) | | |
| `end/term_base_pos_gt_0.5m`, `end/term_base_rot_gt_1rad`, `end/term_diverged` | share ended early by each cause: base > 0.5 m from the reference, orientation > 1 rad off, physics diverged | "configured failure conditions" (not given in the paper; ours) | |
| `csi/phase_mean`, `_p10`, `_p50`, `_p90` | Curriculum State Initialization start phase over environments (0.9 at the start, 0 = clip start) | Table S4 "Stage-1 initialization curriculum" | |
| `finish/<set>`, `episodes/<set>` | finish share and episode count per clip set | | |
| `ppo/lr`, `ppo/kl`, `ppo/value_loss`, `ppo/policy_loss`, `ppo/action_std`, `ppo/value_mean`, `ppo/return_mean` | optimiser state | Table S6 (KL-adaptive rate, target KL 0.02) | |
| `iter_s`, `env_steps`, `env_steps_per_s`, `gpu_util_pct`, `gpu_mem_gb`, `envs` | throughput and GPU use (GPU read every 10 iterations) | | |

Paper terms not logged yet: the stage-2 and stage-3 terms (`track_planned_contact`, `move2goal`, `standatgoal`,
`flat_orientation`, `stumbling`, `slippage`, `ang_vel_xy`, eta_obs, eta_rew, the planner's affordance and occupancy losses,
the stage-3 terrain levels and replay-buffer sizes): they do not exist in stage 1 and get the same treatment when those
stages are written. Per-body contact placement error and per-category success are in the paper-eval output, not in the
per-iteration log. Success with noise and randomisation off is only in the paper eval (training metrics include them).

## 8. Plan for the user's extended training map (`/mnt/filesystem-g4/data/terrains/extended_v1/`, exported by the Hiking track)

Not used yet (2026-11-11). How it fits the paper's method, stage by stage:

- **Stage 1 (now): not used.** Stage 1 needs a reference clip aligned to its terrain; the extended tiles have none, so stage 1
  stays on the clip-aligned props. Exception worth doing later: tiles for which we have matching captures (stairs, wedges,
  4x4s) could host those clips, which is what the terrain-park sets already do.
- **Stage 2: only indirectly.** The paper's stage 2 runs "on the same reference-motion terrains" and takes its hindsight
  labels from successful reference-following rollouts. Hindsight labels can come from the extended map only from rollouts
  that succeed there, i.e. goal-reaching rollouts, and those exist from stage 3 on. So: no extended tiles in stage 2.
- **Stage 3: this is where it goes.** The extended tiles and their checked routes become goal-reaching terrains beside the
  paper's families (stairs, walled stairs, platforms, slopes, gaps, beams, pallets, stones), with the paper's goal rewards,
  second critic, 10-level curriculum and the online planner updates: every successful rollout on an extended tile enters the
  planner's hindsight buffer (partitioned per tile, as the paper does), which is how the planner learns contacts next to
  walls, rails and in corridors. The route waypoints give goal sequences, including the dead ends (turn in place), backward
  and sideways moves: goals with a heading different from the travel direction, which the paper's goal command (direction +
  relative heading) already expresses and the earlier X2 work never trained.
- **Held-out rule:** tiles must be re-arranged parts, never the whole apparatus or NIST layout or their routes; I list the
  tiles I use here when I use them.

What I need from the export:
1. Per tile: a height raster (`x0`, `y0`, `res`, `grid` in metres, as the apparatus rasters) with **tall geometry removed**
   (walls, rails, posts, door frames), and those as a separate list of oriented boxes (`center`, `half`, `quat` wxyz): a
   MuJoCo height field pushes a body that touches a tall column upward (the "catapult" of the earlier work).
2. A second raster or flag for **what the robot should see**: tall geometry included, so the height scan shows walls while
   the physics uses the boxes.
3. Tiles small: MuJoCo Warp degraded on a 30 m height field and crashed at 60 m in the earlier work, and a foot-sized box
   over 2 cm cells overflows its 50-collision limit. Tiles of about 8-10 m, cell size 4 cm or coarser where the surface is
   smooth, or the tile's steps given as boxes (preferred for stairs and 4x4s: exact edges, no raster stair-casing). Few
   distinct box sizes (every distinct size is a slot in every world).
4. Tile metadata: family, difficulty level 0-9 (or the parameters to derive one), spawn pose(s) on flat ground, goal patches
   (position, heading, standing height above ground), and for routes the ordered waypoints with heading and a flag for
   in-place turn / backward / sideways segments; which side is open (so goals are not placed inside or against obstacles).
5. A README with units, frames (x forward, z up, metres), the source commit and which apparatus parts each tile copies.
