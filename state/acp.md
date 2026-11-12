# ACP track state

Updated 2026-11-12 00:30 UTC (track agent). **Stage 2 is training** on node acp: run `s2_v1`, tmux `acp`, 36,000 envs, measured gains (plant `mc_20261024`), from stage-1 checkpoint 2,250; see section 13 first (it supersedes the run commands in section 4). Stage 1 (`s1_v1`) ended at 2,250, paper eval 91.2 %.

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
10. Data filters from the stage-2 switch (2026-11-12, iteration 2,250 on): the paper trains on its whole reference set; we
    leave out (a) chain clips that contain a single step higher than 0.30 m (`exclude_step_over: 0.30` in
    `cfg/stage2*_sets.json`; 50 of 96 source clips, 87 of 210 segments remain): 39-41 cm step-ups that the X2 failed in every
    try and that neither apparatus needs; (b) the 38 stair descents `gs2dn_v1` (feet walk down while pelvis faces up the
    stairs: a stair-fit fault) and one Pico clip with a flipped leg (`exclude_leg_flip`); (c) from iteration 2,500 the ramp,
    terrain-park stair and 4x4 clips whose stance soles sit more than 1.5 cm inside the terrain are lifted by their median
    stance gap (`ground_stance`; 99 of 474 clips, at most 4.3 cm). Reason for each: reference faults, not paper choices.
11. Stage-2 schedule numbers (eta_obs from 200 iterations after the stage-2 start over 1,500; eta_rew from 1,700 over
    1,500) are ours: the paper gives Eq. 3 without numbers. The replay buffer is in memory and starts empty after every
    restart (refills to about 430,000 samples in about 50 iterations).

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
| `s1_v1` 1,250 (stage 1, in training) | dense foot-only, reference-anchored | 86.6 % (85.9-87.3), 9,018 episodes; per set in section 9 | >= 98.5 % (stage 1, every category) | not valid yet (see section 9) | bars only (Fig. 7C) |

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

## 9. Planner + tracker: how the earlier run did it, what is ported, the path to an apparatus score

Correction recorded (user, 2026-11-11): ACP is a planner (contact point estimator) plus a tracker (executor); the tracker is
scoreable on a route as soon as something gives it contact targets. The earlier pipeline had all three stages and scored
13 of 50 (26 %) on our apparatus on 2026-11-02.

### How the 2 November run produced contact targets on the apparatus

- **Planner:** the feet-only MLP of `gwbc-v6/scripts/acp/stage2/planner.py` (the paper's MLP baseline, Table S3, sized to our
  inputs). Input: 17 x 11 height scan in the heading frame (x -0.3..1.3 m, y +-0.5 m, minus root height), the 4-number goal
  command, and one step of "history" (both ankles, root height over ground, root velocity, gravity: 13 numbers; root velocity
  is privileged). Output per foot: weights over the 187 scan points, occupancy, time to contact, contact now.
- **Trained on:** hindsight labels (`stage2/planner_data.py`, the paper's rule) from rollouts of the SuperMimic stage-1
  tracker `acp1_v4_gst` checkpoint 1000 on the clip sets (`stage2/collect_v4.sh`).
- **Weights:** `gwbc-supermimic/logs_sm/acp_artifacts/planner/planner_v2_hist1.pt` (also `planner_v1*.pt`, `planner_v2.pt`).
  The tracker was `gwbc-supermimic/logs_sm/acp3_v2/ckpt_001000.pt` (stage 3, planner frozen). Not in `loco-manip-apparatus`
  (its `build/planner/heatmap.pt` and `regress.pt` are an older, separate footstep planner of 2026-10-27 that the route run
  did not use).
- **On the route:** `eval/pico_rollout_planner.py` (launched by `eval/crowd/route50.sh`): every control step it builds the
  planner input from the simulated robot, decodes the planner with three inference-only fixes (`PLAN_DECODE=mode`: choose the
  strongest scan point instead of the weighted mean; `STEP_MASK=0.03`: drop candidates on edges; `PLAN_LATCH=0.3`: hold the
  target near touchdown), and feeds the feet targets to the tracker. Goals: the 22 hand-placed waypoints of
  `loco-manip-apparatus/sim/x2_route.py` ROUTES['mobility'] with 1 s station holds; physics: walk raster + 344 wall boxes.
- **SuperMimic-derived, so not allowed in this round's training:** both networks' weights (the tracker was trained by the
  SuperMimic trainer; the planner on SuperMimic rollouts), the rollout files, the `acp_*_contacts.pkl` labels. **Method-neutral
  and reused:** the planner architecture and loss, the hindsight labelling rule, the decode fixes, the route and eval logic,
  the stage-3 terrain generators.

### Port status of the existing stage 2 and stage 3 code (this round's tree: `tracks/acp/code`)

| Piece (gwbc-v6) | Status | Where now |
|---|---|---|
| `stage2/planner_data.py` hindsight labels | ported (same rule; vectorised; nine bodies; from this round's rollouts) | `acp/planner.py label` |
| `stage2/planner.py` MLP planner + Gaussian-over-scan-points loss | ported to the paper's interface (nine bodies, tracker's own 5-step history, no privileged velocity, scan centred on the base, top-1 decode) | `acp/planner.py` `MlpPlanner` |
| attention planner (Table S3) | new (did not exist before); 802,353 parameters (paper: 786,993 for the G1's sizes) | `acp/planner.py` `AttnPlanner` |
| `stage2/collect_v4.sh` rollout collection | replaced: the paper-eval rollouts are saved (`*_rollouts.pt`) | `tools/paper_eval.py` |
| `eval/pico_rollout_planner.py` closed loop | reference-terrain closed loop ported (`tools/pair_eval.py`, CPU); the route part is the eval monitor's job with the hand-over below | |
| decode fixes (choose / edge mask / latch) | choose = the paper's top-1 decode, built in; edge mask and latch NOT ported (eval-side options; see below) | |
| `supermimic/planner_contacts.py` eta schedules, co-training of the tracker on planner contacts | to port into `acp/env.py` + `acp/train.py` (planner in the loop, eta_obs / eta_rew, `track_planned_contact`): after stage 1 | not started |
| `stage3/terrain_gen.py`, `stair_world.py`, `supermimic/goal.py`, curriculum in `sim.py` | reusable after removing the SuperMimic relaxations (sigmoid indicators, reward scale 0.125, detached goal rewards); second critic and online planner updates are new | not started |

### Shortest path to a scoreable pair, and what is honest to call it

1. **Now (22:20 UTC, MLP planner only; the attention planner is not ready): tracker `s1_v1` checkpoint 1,250 + planner v1** (MLP and attention), both trained only on this round's
   data: the planner on hindsight labels from 2,587 successful rollouts of that same checkpoint (413,298 samples). This is the
   paper's stage-2 planner at its start: **the tracker has not yet been trained on planner contacts**, on goals other than
   "the reference 0.5-1.5 s ahead", on walls or rails, or on turning in place; it has seen only training-terrain rollouts.
   Label for any number from it: "interim: stage-1 tracker + first planner, no co-training, no stage 3".
   The retrained feet-only MLP planner is a legitimate early pairing: the paper trains exactly this MLP as its baseline with
   the same supervision (Fig. 7B, 7D), and reports it below the attention planner mainly on sparse and distractor terrain.
2. **About 23:25 UTC: tracker checkpoint 2,000 + planner v2** relabelled from its rollouts (the 3-minute paper-eval pause
   collects them; label + train on the CPU takes about 10 minutes).
3. **After stage 1 ends: the paper's stage 2** (tracker co-trained on planner contacts, eta schedules) and **stage 3** (goal
   terrains incl. the extended map, walls, in-place turns, online planner updates). Estimate: code about half a day; training
   as the paper's ratios suggest about 1.5-2 days on this one card. This is the pair that should be compared with 13 of 50.

### Interim, eval-only contact sources on the route (must not feed training)

- The paper has no non-learned contact estimator: its planner is always the learned one. So the paper-conform interim is the
  pair above (1), not a hand-made source.
- A geometric foothold picker is feasible on the eval side (e.g. next foothold = the flat scan point nearest to a nominal
  step ahead along the goal direction, alternating feet, held until touchdown), about 2-3 hours of eval-adapter work. I do not
  recommend it over (1): the tracker would get targets with a timing and gait it never saw, and the result would measure the
  picker. If wanted, it must be labelled "interim, geometric contact source, eval only".
- The earlier SuperMimic-based planner (`planner_v2_hist1.pt`) as an eval-only comparison: technically feasible in about
  2 hours (it needs its own inputs incl. root velocity, and its outputs reduced to the 27 target coordinates), but it was
  trained on another tracker's gait, so a low score would not say much. Not used anywhere in training. The coordinator puts
  that choice to the user.
- Two of the earlier inference fixes (edge mask, target latch) raised the earlier route score from 0/6 to 2/6 and are pure
  decoding; the eval may apply them to this planner's outputs as labelled options. They are not in the ONNX.

### Hand-over to the eval monitor

- `tracks/acp/eval/handover_s1_v1_001000/` (plumbing proof, from `ckpt_001000`): `actor.npz` (588-512-256-128-31, via the
  eval's `export_actor.py`), `adapter_card.json` (every observation term in order incl. the nine contact targets' definition
  and frame, joints, gains, scan grid, goal command), `obs_check.npz` (50 steps, one robot, MuJoCo Warp on the CPU, with the
  clip's prop boxes), `meta.json`.
- `tracks/acp/eval/handover_pair_s1_001250_planner_v1/`: the same four files for checkpoint 1,250, `planner_mlp.onnx` and
  `planner_attn.onnx` (inputs `proprio_history` 495 = the tracker's history block, `goal_command` 3, `scan_17x11` 187 = terrain
  height minus root height on a 17 x 11 grid at 0.1 m, x -0.8..+0.8, y -0.5..+0.5 in the body-yaw frame, index ix * 11 + iy;
  outputs `contact_targets` 9 x 3 in the body-yaw frame relative to the root, zero where occupancy <= 0.5, and `occupancy` 9;
  the 27 numbers go straight into the tracker's contact-target slot), and per planner a folder with `obs_check_pair.npz`
  (50 planner-driven steps) and `pair_eval_cpu.json`.

### Results so far (track reports, not the apparatus score)

Paper evaluation, reference-anchored terrains, `s1_v1` checkpoint 1,250 (9,018 episodes, noise and randomisation off,
oracle contacts; file `tracks/acp/eval/paper_eval/s1_v1_001250.json`):

| Set | Success (Wilson 95 %) | Ended early |
|---|---|---|
| Pooled "dense foot-only" | **86.6 %** (85.9-87.3), paper: >= 98.5 % at the end of stage 1 | 13.4 % |
| Pico 2x2 | 98.3 % (97.3-98.9) | 1.7 % |
| flat locomotion | 93.3 % (91.6-94.7) | 6.7 % |
| ramps | 92.7 % (90.9-94.2) | 7.3 % |
| blocks | 90.6 % (88.7-92.3) | 9.4 % |
| 4x4 stepovers | 90.3 % (88.3-92.0) | 9.7 % |
| terrain-park stairs | 88.2 % (86.1-90.1) | 11.8 % |
| fitted stair climbs | 87.9 % (85.8-89.8) | 12.1 % |
| box and stair chains | 74.3 % (71.5-76.9) | 25.8 % |
| stair descents | 63.7 % (60.7-66.6) | 36.3 % |

The contact placement error in that file (0.37 m) and in `metrics.jsonl` (`err/contact_placement_at_onset_m`, 0.49 m) is
**not valid yet**: the reference's target for a foot switches to the following step at the reference's own touchdown, so a
robot that lands one frame late is compared with the next step's target (about a step length away). To fix at the next
pause: compare with the target that was commanded before the switch.

Planner v1 on held-out episodes (a tenth of the rollouts): MLP target error median 6.0 cm, 90th percentile 17.5 cm, feet
occupancy accuracy 90.0 % (760,932 parameters, 2.7 minutes on the CPU). Planner-driven closed loop on the reference terrains
(CPU, 108 episodes, 12 per set, same clips and starts; `handover_pair_s1_001250_planner_v1/mlp/pair_eval_cpu.json`): tracker 1,250
with oracle contacts 93.5 % (87.2-96.8), with the MLP planner's contacts **46.3 %** (37.2-55.7). Read with care: the tracker has
never trained on planner contacts, and this check still ends an episode when the robot is 0.5 m from the reference, which
punishes a planner-driven robot for being early or late (the earlier work's "descent collapse" lesson); a goal-based
failure rule is needed before stage 2 numbers mean much. The attention planner is still training on the CPU (2.8 min per
epoch; after 5 of 30 epochs its target error is 31 cm against the MLP's 6 cm at 30, so it is not handed over yet;
`runs/planner_v1/train.out`). The 1e-4 learning rate is the paper's; it may simply need many more updates than 30 epochs.

Pauses of stage 1 so far: 20:40-20:51 (GPU benchmark), 21:54:37-21:57:21 (logging switch + paper eval, 2 min 44 s).

## 10. Stage 2 port, switch criterion, weak sets (2026-11-11 evening)

Coordinator decisions applied: stage 2 co-training ported now; stage 1 is not run to 12,000 by default; no geometric picker
and no comparison with the old SuperMimic planner on my side; the eval is a ladder (single-tile skills, reviewed routes,
apparatuses) and scores the pair on step 1 first.

### Stage 2 co-training: ported (in `tracks/acp/code_next`, installed into `code/` at the checkpoint-2,000 pause)

`python -m acp.train --stage 2 --init <stage-1 checkpoint> --run runs/s2_v1 ...` (new run folder; stage-1 path unchanged):
- **Planner in the loop** (`acp/env.py`): every control step the planner (attention by default, `--planner-arch`) reads the
  tracker's own noisy history, the goal and the 17 x 11 scan with N(0, 0.03) m noise; each environment independently gets
  the planner's complete contact command with probability eta_obs and the reference's otherwise (paper: not interpolated).
- **Rewards** (Table 3, stage 2): `track_contact_label`, `track_contact_pos_base`, `track_contact_pos_global` x (1 - eta_rew);
  `track_planned_contact` = exp(-sum_b m_b |p_exec - p_hat_b|^2 / (sigma_c^2 max(1, sum m_b))) with weight 1.5 eta_rew; mimic
  and regularisation terms unchanged. Logged raw, weighted and with the current weight (`weight/<term>`).
- **Schedules** (Eq. 3, linear, the reward one later): the paper gives no numbers; mine: eta_obs starts 200 iterations
  after the stage-2 start, eta_rew at 700, both ramp over 1,000 iterations (`--u-obs --u-rew --n-ann`).
- **Hindsight replay, online** (`acp/planner.py` `Recorder`, `Replay`): the first 3,000 environments start in the first 10 %
  of their clip and are recorded (at most 500 steps, Table S5); an episode that reaches 90 % of the clip without failing is
  labelled once per PPO iteration (the paper's rule: onset after segments >= 10 steps, 1.5 s horizon, footprint
  +-0.8 x +-0.5 m) and pushed to a 1.5 M-sample ring buffer; ten planner updates of 4,096 samples per PPO iteration with
  Adam 1e-4, from 1,000 entries on (Table S5). Other environments start at U(0, 0.98) (Table S2, stage 2).
- **PPO**: target KL 0.01, entropy 0.005 (Table S6, stage 2); 36,000 envs kept (the paper: 4,800 on 2 GPUs).
- **Logged per iteration in addition:** `stage2/eta_obs`, `stage2/eta_rew`, `planner/loss_affordance`,
  `planner/loss_occupancy`, `planner/updates`, `planner/new_samples`, `planner/replay_size`, `planner/recorded_episodes`,
  `planner/recorded_successes`, `planner/fed_to_tracker_share`, `planner/target_vs_reference_m`.
- **Checked:** on the CPU (64 robots, from checkpoint 1,250): schedules, weights, planned-contact reward, planner feeding and
  logging run end to end; the recorder / replay / planner-update path: see "CPU smoke" below. NOT yet run on the GPU.
- **Still to do for stage 2:** a planner-driven paper eval with a goal-based failure rule; checkpoint export of the pair as
  one hand-over; the per-tile partition of the buffer (only needed with stage-3 goal terrains).
- Also changed with this code: the contact placement error now compares the realised contact with the nearer of the
  current commanded target and the one before its last change (the command switches to the following step around
  touchdown); a CPU fallback for the physics loop (small checks beside training).

### When stage 1 hands over to stage 2

Criterion (coordinator): paper eval pooled >= about 95 % with stair descents and chains above 85 %, or a plateau (< 1 point
per 1,000 iterations), whichever first once stage 2 is ready. What the paper says about stage-1 length: only wall-clock,
"Stage 1 requires approximately 1 day using 3 NVIDIA RTX 4090 GPUs" with 12,000 envs per worker (Table S6: "Stage-1
distributed totals are not reported"); no iteration count, no stopping rule. At our 115,000 steps/s one day is about 10
billion transitions = about 11,500 of our iterations, so the paper's stage 1 was probably about as long as my 12,000 budget;
that does not argue against a criterion, it only says not to expect 98.5 % early.

### Weak sets at checkpoint 1,250 (`eval/paper_eval/s1_v1_001250_weak_sets.txt`; from the saved paper-eval rollouts)

Failures are clip-specific, not random: the same clips fail on every try (evaluation is deterministic up to the clip draw).
- **Stair descents (38.9 % ended early; 27 of 38 clips fail at least once, 13 more than half the time).** Ends are spread
  over the whole clip; the robot is as often ahead as behind the reference; yaw error 0.37 rad (the largest of any stair
  set). The always-failing clips are the ones with 21-22 cm risers and the ones with a side wall (a 2.76 m box beside the
  flight): 12/12, 10/10, 8/8, 5/5. Treads there are 30 cm boxes, so the 24 cm foot fits; the short-tread explanation is not
  what the data shows for this set. The set is small (38 clips, 3.1 min, 5 % share), so each clip is seen often: coverage is
  not the cause either. Most likely: steep descents need the heel on the tread edge and a precise knee bend that the loose
  tracking (0.16 rad joint error) does not deliver, plus contact with the side wall, which the 9 x 7 scan shows only as a
  tall column.
- **Fitted stair climbs (12.3 %; 19 of 105 clips, all 19 always).** 78 % of the failures end in the middle of the clip with
  the root 0.42 m too low: the robot does not get up the flight. The always-failing clips are jogging climbs (0.46 m/s) on
  22 cm risers with 23-27 cm treads: here the 24 cm foot against a 23-27 cm tread is plausible (toe against the next riser).
- **Chains (23.4 %; 38 of 164 clips, 33 always).** 97 % end with the robot behind the reference, 63 % in the middle
  quintile: slow clips (0.14-0.28 m/s) stepping onto a 39-41 cm platform or a low stair; the robot stalls at the step-up
  and the reference walks away (0.5 m limit).
- Across all sets the robot ends *behind* the reference (59-97 % of failures), never ahead: it is slower than the
  reference, which fits a goal command that says only direction, not distance, beyond 0.3 m.
What I will do: nothing to the paper's rewards; at the switch decision, if these three stay low, (1) raise the share of the
failing clips by sampling clips in proportion to their failure rate (the paper has no such rule; recorded as ours),
(2) re-check the reference clips themselves on 22 cm risers (they may penetrate the riser or float), before anything else.

### CPU smoke of the recorder / replay / planner updates

(filled in below when the run finishes)

## 11. X2 gains: what the simulator applies, and the switch to plant `mc_20261024` (user decision, 2026-11-11)

Read back from the live MuJoCo Warp model (`tools/plant_report.py`, CPU; full table per joint in
`tracks/acp/eval/plant_report_s1_v1_001250.json`). Timestep 0.005 s, 4 substeps, control 0.02 s, implicitfast. The actuators
are PD position servos built by `assets.build_x2` (gain = kp, bias = -kp q - kd qd, force range = +-plant effort); the MJCF's
own `<motor>` actuators are deleted at build time, so their `ctrlrange` (118 for hips and knees, 2.2 wrists, 2.6 / 0.6
head) is **not** applied: the plant's effort is (120, 6, 3 / 1).

**Until iteration 2,000 (plant `vendor_20260823`, armature rule), live = plant file exactly:**

| Joints | kp | kd | armature | effort | action scale | measured (`mc_20261024`) kp / kd | measured / live |
|---|---|---|---|---|---|---|---|
| hip pitch / roll / yaw, waist yaw | 114.8 | 7.31 | 0.0291 | 120 | 0.261 | hips 100 / 4 (pitch), 3 (roll, yaw); waist yaw 40 / 8 | hips 0.87 / 0.41-0.55; waist yaw 0.35 / 1.09 |
| knee | 114.8 | 7.31 | 0.0291 | 120 | 0.261 | 150 / 5 | 1.31 / 0.68 |
| ankle pitch, shoulder pitch / roll | 34.9 | 2.22 | 0.0088 | 36 | 0.258 | ankle pitch 40 / 3; shoulders 30, 20 / 1 | 1.15 / 1.35; 0.86, 0.57 / 0.45 |
| ankle roll | 12.37 | 0.787 | 0.0031 | 24 | 0.485 | 30 / 2 | 2.43 / 2.54 |
| waist pitch / roll | 12.37 | 0.787 | 0.0031 | 24 | 0.485 | 40 / 5 | 3.23 / 6.35 |
| shoulder yaw; elbow, wrist yaw | 12.37 | 0.787 | 0.0031 | 24 | 0.485 | 20 / 1; 50 / 1 | 1.62 / 1.27; 4.04 / 1.27 |
| wrist pitch / roll | 16.78 | 1.068 | 0.0043 | 6 | 0.089 | 20 / 1 | 1.19 / 0.94 |
| head yaw / pitch | 16.78 | 1.068 | 0.0043 | 3 / 1 | 0.039 / 0.009 | 3.4 / 0.11; 10 / 0.8 | 0.20 / 0.11; 0.60 / 0.75 |

Default pose (rad): hip pitch -0.31, knee 0.67, ankle pitch -0.36, shoulder pitch 0.2, shoulder roll +-0.2, elbow -0.6, all
others 0. Armature, effort, default pose and action scale are the same in both plant files (checked: action scale
identical per joint).

**Randomisation coverage:** kp, kd and armature are each x U(0.9, 1.1) per environment (the paper's Table S2). That range
covers **none** of the measured gains: the nearest is hip kp (x0.87), the farthest waist pitch / roll (kp x3.2, kd x6.4),
ankle roll (x2.4 / x2.5) and elbow / wrist yaw kp (x4.0). A policy trained on the armature-rule gains was therefore never
shown the robot's actual stiffness on any joint.

**Are soft ankle roll and waist behind the weak sets?** Per-joint tracking error from the checkpoint-1,250 paper-eval
rollouts (mean |q - q_ref|, rad):

| Set | hip | knee | ankle pitch | ankle roll | waist yaw | waist pitch / roll | arms, head |
|---|---|---|---|---|---|---|---|
| flat | 0.156 | 0.210 | 0.141 | 0.096 | 0.109 | 0.096 | 0.163 |
| blocks | 0.161 | 0.218 | 0.144 | 0.096 | 0.113 | 0.096 | 0.166 |
| chains | 0.137 | 0.260 | 0.141 | 0.080 | 0.116 | 0.128 | 0.156 |
| terrain-park stairs | 0.148 | 0.251 | 0.164 | 0.069 | 0.090 | 0.179 | 0.112 |
| fitted stair climbs | 0.168 | 0.394 | 0.205 | 0.102 | 0.172 | 0.070 | 0.130 |
| **stair descents** | **1.153** | 0.559 | 0.321 | 0.172 | 0.100 | 0.120 | 0.152 |
| ramps | 0.128 | 0.284 | 0.198 | 0.090 | 0.088 | 0.134 | 0.114 |
| 4x4 stepovers | 0.151 | 0.343 | 0.129 | 0.092 | 0.118 | 0.170 | 0.116 |
| Pico 2x2 | 0.180 | 0.270 | 0.145 | 0.078 | 0.154 | 0.170 | 0.103 |

Ankle roll (0.07-0.10 rad) and waist pitch / roll (0.07-0.18) are not worse on the weak sets than on the strong ones, and
they are the best-tracked joints after the arms; the knee is the worst everywhere. So this data does not show soft ankle
roll or waist as the cause of the weak sets. (It does not test what stiffer gains would do; the comparison after the
switch will.)

**The stair-descent set's own references are broken.** Hip error 1.15 rad led to the clips: in all 38 `gs2dn_v1` clips a hip
yaw joint sits at its limit (left +3.43 rad, right -3.43 rad, i.e. the leg turned about 196 degrees) and the clips contain
frame-to-frame joint jumps of up to 1.1 rad at 50 Hz. No other set comes near (hip yaw within +-1.9 rad; the largest jumps
are 0.77 rad in the fitted stair climbs and 0.59 in locomotion, which also want a look). The descents' 63.7 % is therefore
mostly a data fault (a retargeting / stair-fit artefact), not the policy, the foot length or the gains. Recommended and
planned: keep the set through the checkpoint-2,250 comparison so the plant effect is seen on unchanged data, then take it
out of training (weight 0) and replace it with clean descents (the terrain-park stair set has descents; or re-fit these
clips with the hip yaw unwrapped). This set was in the earlier SuperMimic runs too ("0 / 38 varied descents clean" in the
old issue log is probably the same fault).

**The switch (armed, automatic):** at checkpoint 2,000 the run pauses (heartbeat says why), the paper eval runs on that
checkpoint under the old plant (the "before" number, beside 1,250), and training resumes from that checkpoint with
`ACP_PLANT=mc_20261024` (stiffness and damping from the measured file; armature, effort, default pose, action scale
unchanged; randomisation x U(0.9, 1.1) now around the new values). Right after, the gains are read back from the live model
into `eval/plant_report_002000_live_mc_20261024.json`. At checkpoint 2,250 (first after the switch) a second pause runs the
paper eval under the new plant and saves rollouts; on the CPU a new planner is trained from them and the eval hand-over is
regenerated with the new gains in the card (`eval/handover_pair_s1_002250_mc_20261024_planner/`). Planner v1 and the first
hand-over (vendor gains) are stale from then on.

## 12. Checkpoint 2,000: paper eval, gains switched (2026-11-11 23:37-23:40 UTC)

Pause 23:37:30-23:40:08 (2 min 38 s; heartbeat stated the reason). In it: the staged code was installed (stage-2-capable
trainer, fixed contact placement metric, plant selection; smoke test passed), the paper eval ran on checkpoint 2,000 under
the gains it was trained with (`vendor_20260823`), and training resumed from `ckpt_002000.pt` with `ACP_PLANT=mc_20261024`.
Resume command now: `tmux new -d -s acp "ACP_PLANT=mc_20261024 bash /mnt/filesystem-g4/tracks/acp/code/supervise.sh /mnt/filesystem-g4/tracks/acp/runs/s1_v1 36000 12000"`
(without `ACP_PLANT` the old gains come back: do not omit it).

**Live read-back after the restart** (`eval/plant_report_002000_live_mc_20261024.{txt,json}`, from the MuJoCo Warp model):

| Joints | kp | kd | armature | effort | action scale |
|---|---|---|---|---|---|
| hip pitch | 100 | 4 | 0.0291 | 120 | 0.261 |
| hip roll, hip yaw | 100 | 3 | 0.0291 | 120 | 0.261 |
| knee | 150 | 5 | 0.0291 | 120 | 0.261 |
| ankle pitch | 40 | 3 | 0.0088 | 36 | 0.258 |
| ankle roll | 30 | 2 | 0.0031 | 24 | 0.485 |
| waist yaw | 40 | 8 | 0.0291 | 120 | 0.261 |
| waist pitch, roll | 40 | 5 | 0.0031 | 24 | 0.485 |
| shoulder pitch / roll / yaw | 30 / 20 / 20 | 1 | 0.0088 / 0.0088 / 0.0031 | 36 / 36 / 24 | 0.258 / 0.258 / 0.485 |
| elbow, wrist yaw | 50 | 1 | 0.0031 | 24 | 0.485 |
| wrist pitch, roll | 20 | 1 | 0.0043 | 6 | 0.089 |
| head yaw / pitch | 3.4 / 10 | 0.11 / 0.8 | 0.0043 | 3 / 1 | 0.039 / 0.009 |

Every joint reads exactly the `mc_20261024` file (ratio 1.00 for kp and kd on all 31); armature, effort, default pose and
action scale unchanged. Randomisation: kp, kd, armature x U(0.9, 1.1) around these values.

**Paper eval, reference-anchored terrains, oracle contacts, 9,018 episodes each (old gains for both):**

| Set | 1,250 | 2,000 | Change |
|---|---|---|---|
| **Pooled** (paper: >= 98.5 % at the end of stage 1) | 86.6 % | **91.0 %** (90.4-91.6) | +4.4 |
| Pico 2x2 | 98.3 | 100.0 | +1.7 |
| ramps | 92.7 | 96.8 | +4.1 |
| 4x4 stepovers | 90.3 | 93.7 | +3.4 |
| flat locomotion | 93.3 | 92.9 | -0.4 |
| blocks | 90.6 | 92.0 | +1.4 |
| stair descents | 63.7 | 91.2 | +27.5 |
| fitted stair climbs | 87.9 | 88.7 | +0.8 |
| terrain-park stairs | 88.2 | 87.4 | -0.8 |
| box and stair chains | 74.3 | 75.9 | +1.6 |

Contact placement error at onset with the fixed metric: 0.166 m pooled (0.11 Pico 2x2 to 0.25 terrain-park stairs); the
paper shows this only as bars. The stair descents recovered although their references are faulty (hip yaw at the joint
limit): the policy learned to follow them; the set is still to be replaced. Chains are now the weak set (75.9 %, 24 % ended
early) and barely moved in 750 iterations; flat, fitted climbs and terrain-park stairs are flat within their intervals.
Against the switch criterion: pooled 91.0 % (< 95 %), chains 75.9 % (< 85 %), gain 4.4 points over 750 iterations (not a
plateau yet), so stage 1 continues; the decision is retaken at the next evals under the new gains.

Training metrics at iteration 2,000 (old gains): reward 0.123 of 0.30, base position error 0.123 m, joint error 0.156 rad,
contact agreement 94.9 %, finish 0.87. Seven iterations after the gains switch (2,007): reward 0.087, joint error 0.171 rad,
base error 0.150 m: the expected drop; how fast it recovers is the first thing the 2,250 eval shows.

**Planner v2** (hindsight labels from the checkpoint-2,000 rollouts, old gains; CPU job `runs/planner_v2_s1_002000_vendor`,
hand-over `eval/handover_pair_s1_002000_vendor_planner_v2/` with plant, kp and kd in the card): running at the time of
writing. It is the last pair on the old gains; the first pair on the new gains comes from the 2,250 pause (armed:
`eval/handover_pair_s1_002250_mc_20261024_planner/`).

Hand-over cards: `handover_s1_v1_001000` and `handover_pair_s1_001250_planner_v1` now carry `"plant": "vendor_20260823"`
(the eval monitor's check of the pair passed: planner identical, tracker input within 6e-08, action within 3e-06). The export
writes the plant name always.

Checkpoints on disk at 23:41: 1,000, 1,500, 1,750, 2,000 (the backup loop prunes to the newest two plus every 1,000); 1,250 survives only as
`eval/handover_pair_s1_001250_planner_v1/tracker_ckpt_001250.pt`.

Planner v2 result (23:51 UTC): MLP planner from the checkpoint-2,000 rollouts, held-out target error median 5.8 cm (90th
percentile 16.8 cm), feet occupancy accuracy 89.9 %. Planner-driven closed loop on the reference terrains (CPU, 108
episodes, same clips and starts, old gains): tracker 2,000 with oracle contacts 92.6 % (86.1-96.2), with planner contacts
**44.4 %** (35.4-53.9); v1 at 1,250 was 46.3 %: no change, as expected without co-training. Per set with planner contacts
(12 episodes each): Pico 2x2 0.92, terrain-park stairs 0.67, stair descents 0.67, fitted climbs 0.50, 4x4 0.42, ramps 0.33,
blocks 0.25, chains 0.17, flat locomotion 0.08. Flat locomotion is the worst: those clips turn, stop and strafe, and a
planner-driven robot that leaves the reference's timing is ended 0.5 m from the reference (the failure rule of this check),
so the flat number says more about the rule than about walking. Hand-over:
`eval/handover_pair_s1_002000_vendor_planner_v2/` (card with plant `vendor_20260823`, kp, kd and the planner definition).

Chains at 2,000 (`eval/paper_eval/s1_v1_002000_plant_default_weak_sets.txt`): 21.9 % ended early, 31 of 164 clips fail more
than half the time, the same clips as at 1,250: slow (0.18-0.28 m/s) steps onto a 39-41 cm platform; 97 % end with the robot
behind the reference in the middle of the clip. A 40 cm step-up is at the limit of what the X2 can do without hands; these
clips may be kinematically possible only in the retarget. To check next (CPU): replay those references against their boxes
for penetration, and decide whether to drop them.

Recovery after the gains switch: reward 0.087 at iteration 2,007, 0.111 at 2,080 (0.123 before the switch); joint error
0.171 -> 0.161 rad (0.156 before).

## 13. Checkpoint 2,250 and the switch to stage 2 (2026-11-12 00:13-00:22 UTC)

**What runs now: stage 2**, run `/mnt/filesystem-g4/tracks/acp/runs/s2_v1`, tmux `acp`, since 00:21:32 UTC, from stage-1
checkpoint 2,250 (`runs/s2_v1/init_from_s1_v1_002250.pt`), 36,000 envs, budget to iteration 6,500, plant `mc_20261024`,
data `cfg/stage2_sets.json`, attention planner trained online from scratch, eta_obs from 200 iterations in over 1,500,
eta_rew from 1,700 over 1,500 (coordinator's gentle ramp). Start / resume (the supervisor resumes from `latest.pt` once the
first stage-2 checkpoint exists, before that from the init file):
`tmux new -d -s acp "ACP_PLANT=mc_20261024 DATA=cfg/stage2_sets.json EXTRA='--stage 2 --init /mnt/filesystem-g4/tracks/acp/runs/s2_v1/init_from_s1_v1_002250.pt --planner-arch attn --u-obs 200 --u-rew 1700 --n-ann 1500' bash /mnt/filesystem-g4/tracks/acp/code/supervise.sh /mnt/filesystem-g4/tracks/acp/runs/s2_v1 36000 6500"`
Stop: `touch runs/s2_v1/STOP`. Stage 1 (`runs/s1_v1`) ended at iteration 2,260; its last checkpoint is 2,250.

First stage-2 numbers (iteration 2,270, eta still 0): 71,400 env-steps/s (12.1 s per iteration; stage 1 was 108,000: the
planner runs for every robot every step), GPU 99 %, **81.9 GB** of 96; reward 0.123, joint error 0.148 rad, finish 0.91;
replay 428,692 samples from 2,714 successful recorded episodes of 3,050; planner loss 4.76 affordance / 0.455 occupancy
(start of training). To do at the next needed restart: run the planner in chunks to bring GPU memory back under 80 GB
(the live-evaluation guard rail) and see whether the planner step can be made cheaper.

**The 2,250 pause** (00:13:46-00:18:24, 4 min 38 s; heartbeat stated the reason): code update + smoke test, paper eval under
the new gains with rollouts, stage-2 code at full scale for 8 iterations (passed: 36,000 envs, 67,000-70,000 steps/s,
schedules, recorder, replay, planner updates). **The switch** (00:21:31, a stop of about one second plus start-up) followed
the rule "first checkpoint where the paper eval under the new gains is back to 2,000's level": 91.2 % >= 91.0 %.

**Paper eval, reference terrains, oracle contacts, 9,018 episodes each:**

| Set | 1,250 (old gains) | 2,000 (old gains) | 2,250 (measured gains, 250 iterations after the switch) |
|---|---|---|---|
| **Pooled** (paper >= 98.5 %) | 86.6 | 91.0 | **91.2** (90.6-91.8) |
| Pico 2x2 | 98.3 | 100.0 | 100.0 |
| ramps | 92.7 | 96.8 | 95.6 |
| stair descents (faulty clips) | 63.7 | 91.2 | 93.2 |
| flat locomotion | 93.3 | 92.9 | 92.7 |
| 4x4 stepovers | 90.3 | 93.7 | 92.5 |
| blocks | 90.6 | 92.0 | 91.6 |
| fitted stair climbs | 87.9 | 88.7 | 89.6 |
| terrain-park stairs | 88.2 | 87.4 | 88.9 |
| chains | 74.3 | 75.9 | 76.9 |

Contact placement error 0.163 m (0.166 at 2,000). The measured gains cost nothing after 250 iterations and gained nothing
visible yet: every set is within its interval of 2,000. Files: `eval/paper_eval/s1_v1_002250_plant_mc_20261024.json`,
`_weak_sets.txt`, `_rollouts.pt`.

**Data changes at the switch (`cfg/stage2_sets.json`; filters in `acp/data.py`):**
- `gs2dn_v1` (38 stair descents) left out. My attempt to repair it failed: re-solving the legs for the same foot poses
  (`tools/repair_legs.py`) ends on the flipped branch again (hip yaw 3.34 rad, foot error up to 35 cm), which means the
  FOOT poses themselves point backward relative to the pelvis: the clips have the pelvis and upper body facing up the
  stairs while the feet walk down. That is a fault of the stair fit (root heading), not an angle wrap, and needs the fit
  re-run on the desktop (`loco-manip-apparatus/sim/pipeline/stairfit.py`, `gwbc-v6/scripts/acp/data/gen_stairfit2.py`) with
  the root turned; I cannot fix it from the joint angles alone. Until then there is no dedicated descent set; its 5 % share
  went to the terrain-park stairs. The policy had learned to follow these clips (93 % at 2,250), i.e. it was being rewarded
  for walking down stairs with the legs turned around.
- one Pico 2x2 clip with the same flipped leg left out (9 of 10 remain).
- chains: clips containing a single step higher than 0.30 m left out (50 of 96 source clips; 87 of 210 segments remain).
  Checked first: the always-failing chain clips are not penetrating their boxes (sole 3 cm at most, shins none) and have no
  limit dwell; they are simply 39-41 cm step-ups, which neither apparatus needs.
- fitted stair climbs (`gst_v1`): re-solving changed nothing (foot error 0.1 cm, jumps still 0.40-0.51 rad per 50 Hz frame),
  so those jumps are in the motion itself (fast jogging swings, 20-25 rad/s at the joint, above the 11.9 rad/s speed limit),
  not branch flips. Kept; flagged as a data-quality item (the clips ask for leg speeds the robot does not have).
- Total now 1,521 segments, 176.7 min.

**Clip-quality table** (`tools/clip_quality.py`, part of the data build; `eval/clip_quality/stage1_sets.json`; flags: leg
joint at a limit > 10 % of frames; jump > 0.35 rad per 50 Hz frame; |hip yaw| > 2 rad; sole > 3 cm inside prop or floor on
> 5 % of frames; shin > 2 cm inside a box on > 2 % of frames):

| Set | Clips | Any flag | Limit dwell | Jump | Leg flip | Sole penetration | Shin | Max jump (rad) | Max abs hip yaw | Worst sole penetration |
|---|---|---|---|---|---|---|---|---|---|---|
| flat locomotion | 419 | 192 | 145 | 80 | 0 | 30 | 0 | 0.82 | 1.85 | 11 cm |
| blocks | 419 | 192 | 145 | 80 | 0 | 30 | 0 | 0.82 | 1.85 | 11 cm |
| chains | 210 | 175 | 46 | 170 | 0 | 47 | 0 | 0.48 | 1.22 | 8 cm |
| terrain-park stairs | 88 | 88 | 88 | 0 | 0 | 53 | 0 | 0.16 | 0.89 | 17 cm |
| fitted stair climbs | 113 | 113 | 113 | 98 | 0 | 0 | 0 | 0.77 | 1.76 | 2 cm |
| stair descents | 38 | 38 | 38 | 38 | 38 | 0 | 0 | 1.10 | 3.43 | 3 cm |
| ramps | 330 | 205 | 55 | 0 | 0 | 205 | 0 | 0.35 | 0.99 | 16 cm |
| 4x4 stepovers | 56 | 29 | 23 | 0 | 0 | 8 | 0 | 0.23 | 1.16 | 10 cm |
| Pico 2x2 | 10 | 3 | 2 | 0 | 1 | 2 | 0 | 0.14 | 3.43 | 4 cm |

Reading it: "limit dwell" is mostly the knee at 0 (straight leg) and the ankles at their short limits, so it over-flags;
the real findings are the 39 flipped clips, the jumps in 80 locomotion, 170 chain (stitch points) and 98 stair-climb clips,
and soles inside the ground on the ramps (205 of 330 clips, up to 16 cm: the shifted capture variants do not sit on the
ramp raster) and the terrain-park stairs (53 of 88, up to 17 cm). The ramp and stair penetrations are not fixed yet.

**Removed files (my mistake in procedure, nothing lost):** at 00:12 I deleted
`data/clips/gs2dn_v2_resolved.pkl` and `data/clips/pico_2026-10-29_2x2_train_v2_resolved.pkl`: the outputs of the failed
repair above, written ten minutes earlier by `tools/repair_legs.py`, deleted so nobody would train on them. The originals
were never touched. Rule since then: nothing on the shared filesystem is deleted; unwanted files go to
`tracks/acp/_trash/<date>/` with a line in `NOTES.txt` (the checkpoint pruning and the code roll-back copy now move there
too: `_trash/2026-11-12/pruned_checkpoints/`, `code_prev_*`). `/tmp` on the node is cleaned freely.

**No more evaluation pauses.** The supervisor never had a periodic pause; the three pauses so far were one-off scripts
(1,250, 2,000, 2,250) and none is armed now. What replaces them: (1) rollouts for planner labels: the stage-2 trainer
records and labels them in-process (3,000 recorder robots; `planner/*` fields in `metrics.jsonl`); (2) success-style
metrics every iteration from live episodes: `end/finished_phase099`, per-set `finish/<set>`, termination causes,
`planner/recorded_successes`; (3) the paper eval: at restarts that are needed anyway, at most about every 1,000 iterations,
or live beside training if the test below holds; (4) the planner-driven check: on the node's CPU (108 episodes), now with
a fall-based failure rule beside the old one.

**Live evaluation beside training (to test at the first stage-2 checkpoint, 2,500):** `tools/eval_live.sh` (separate
process, starts only after the checkpoint file is complete, lock against a second eval, robot count capped so the total
stays under 80 GB, `nice` + four CPU cores, 15-minute limit, aborts if free GPU memory drops under 10 GB, measures training
speed before and during from `metrics.jsonl`). Open problem: stage 2 itself uses 81.9 GB, so the 80 GB cap refuses to start
until the planner runs in chunks.

**Planner-driven check at 2,250 (measured gains; CPU, 108 episodes, same clips and starts; MLP planner from the 2,250
rollouts: target error 5.8 cm class, 432,342 samples from 2,729 successful rollouts;
`eval/handover_pair_s1_002250_mc_20261024_planner/`, card with plant `mc_20261024`):**

| Contacts | Old rule (ended 0.5 m from the reference) | Fall rule (ended only by a fall: root under 0.35 m over the ground, or tilt over 60 degrees) |
|---|---|---|
| oracle | 92.6 % (86.1-96.2), 7.4 % ended early | 96.3 % (90.9-98.6), 0.9 % fell |
| planner | 50.9 % (41.6-60.2), 49.1 % ended early | **58.3 %** (48.9-67.2), **1.9 % fell** |

Success in both = not ended early and within 0.5 m of the segment's end point when the clip time is up. With the fall rule
the planner-driven robot almost never falls (2 of 108); the 40 % that fail are standing or walking somewhere else when the
time is up: they do not keep up with, or do not follow, the route the reference takes (flat locomotion 2 of 12, chains 4
of 12, ramps 6 of 12; stairs and descents 9-10 of 12). So the gap to close in stage 2 is following the goal under planner
contacts, not balance. Earlier pairs under the old rule: 46.3 % (1,250), 44.4 % (2,000).

## 14. Restart at checkpoint 2,500, soles in the ground, stair descents v2, live eval (2026-11-12 00:50- UTC; new ACP agent)

The previous ACP agent stopped when its usage ran out; this agent resumed at 00:50 UTC from this file, without restarting the healthy run.

**What runs now:** stage 2, run `runs/s2_v1`, tmux `acp`, resumed at 01:13:37 UTC from `ckpt_002500.pt`, data
`cfg/stage2b_sets.json`, budget to 6,500. Resume command (also in `code/ops/README.md`):
`tmux new -d -s acp "ACP_PLANT=mc_20261024 DATA=cfg/stage2b_sets.json EXTRA='--stage 2 --init /mnt/filesystem-g4/tracks/acp/runs/s2_v1/init_from_s1_v1_002250.pt --planner-arch attn --u-obs 200 --u-rew 1700 --n-ann 1500' bash /mnt/filesystem-g4/tracks/acp/code/supervise.sh /mnt/filesystem-g4/tracks/acp/runs/s2_v1 36000 6500"`

**The 2,500 restart** (`code/ops/restart_s2b.sh`: the previous agent's `restart_s2.sh`, with the STOP flag moved to `_trash/<date>/stop_flags/` instead of deleted): STOP seen 01:13:04, the throw-away resume check passed (4,096 robots, iterations 2,500 to 2,503; planner, planner Adam state and stage-2 start 2,250 restored), real run back at 01:13:37. Training was down for under 2 minutes.
- **Planner in chunks of 4,096 robots** (`acp/planner.py` `DECODE_CHUNK`): GPU memory **51.6 GB** (82-84 before);
  **70,300-71,000 env-steps/s**, 12.2-12.4 s per iteration (70,700 before). CPU check: chunked and unchunked outputs are
  identical (max difference 0.0), so nothing learned changes.
- The replay buffer is in memory and starts empty after a restart: it was at 127k samples by 2,512. Memory with a full
  buffer (1.5 M) is to be re-checked.
- **Planner rate:** the paper (by section 1 above) switches contacts per env and per step and gives no slower planner
  rate, so the planner runs every 50 Hz control step. Not changed; a lower rate needs the coordinator. (The paper text
  itself was not re-read; this rests on my pipeline notes.)

**Soles inside the ground: cause and fix** (`tools/ground_check.py`, `tools/pen_cause.py`, `tools/pen_shift.py`; outputs
in `eval/clip_quality/`). Measure = deepest of the four sole corners against the clip's own terrain.
- Per clip, the median stance gap is small everywhere: terrain-park stairs -0.3 cm, ramps -0.7 cm (11 % of ramp clips
  under -3 cm), 4x4 -0.2 cm. Base and shifted capture variants are alike (ramps base -0.7, shifted -0.7 cm). So the large
  figures (16-17 cm) are tails, not whole clips sunk in.
- Where the deep frames are: stairs, chains and 4x4 have 73-76 % of them with the foot **straddling an edge** (more than
  3 cm of terrain-height difference under the foot: the toe or heel corner inside the next riser or box). Ramps have 64 %
  on flat-under-foot stance before the lift: a per-clip vertical offset plus the sole not following the slope.
- Prop offset? Shifting the terrain horizontally under each clip (grid +-16 cm; the score counts deep frames plus stance
  frames floating more than 3 cm) gives **no common best shift**. The most common best shift is (0, 0) for stairs and 4x4;
  the rest scatter to the grid edges in every direction. So it is **not a prop offset**. It is the fit of the retargeted X2
  foot: the 22 cm sole placed where the human foot was overhangs riser edges, and on ramps the foot stays flatter than the
  slope. Ramps also carry a per-clip vertical offset.
- **Fix in the data build** (`acp/data.py` `_ground`, flag `ground_stance` in `cfg/stage2b_sets.json`; in training since
  2,500): a clip whose median stance gap is below -1.5 cm is lifted by it. 99 of 474 clips lifted, at most 4.3 cm.

| Set (clip_quality rule: corner > 3 cm inside on > 5 % of frames) | Flagged before (stage2) | After (stage2b) | Worst before / after | Deep foot-frames before / after |
|---|---|---|---|---|
| ramps (330) | 205 | 177 | 16.4 / 14.2 cm | 36,439 / 14,619 |
| terrain-park stairs (88) | 53 | 53 | 17.1 / 17.1 cm | 3,862 / 3,820 |
| 4x4 stepovers (56) | 8 | 8 | 10.4 / 10.4 cm | 220 / 220 |
| chains (87 after the step filter) | 46 | 46 | 8.3 cm | 1,565 |
| flat / blocks (419) | 30 | 30 | 11.2 cm | 2,205 (half in swing: retarget through the floor) |

  What remains is edge straddling and slope, which a vertical lift cannot fix. The fix would be a foot-placement re-fit,
  and I have not done it: low value for training, see the next point.
- **How the rewards treat a reference contact below the surface:** nothing clamps or projects it. `track_contact_pos_base`
  and `_global` compare the robot's nine contact points (one sole point per foot, 3.8 cm ahead of the ankle at sole height,
  not the corners) with the reference's points, through exp(-sum |d|^2 / sigma_c^2) with sigma_c = 1 m (the paper's
  value). A reference point 5 cm below the floor costs exp(-0.0025): 0.25 % of the term. The robot cannot follow it into
  the ground; it loses a little reward, and the mimic terms (root height, joints) carry the same offset. The contact
  targets in the tracker's input come from the same reference points (`data.next`) and are fed as they are. The planner's
  labels come from the robot's own rollouts, so they are always on the surface.

**Stair descents v2** (desktop CPU; `code/data_build/gen_stairfit_dn.py`). Cause of v1: `gen_stairfit2.py` fits a descent time-reversed as a climb, and
`stairfit.tread_targets` turns every tread foot "straight up the stairs" while the body faces down, so the hip turned about 180
degrees. v2 replaces `tread_targets` for descents: the foot faces down the stairs, heel 5 cm short of the next riser (a
2 cm gap let the heel cut the riser around touchdown: swing clearance -13 to -17 cm in the first trial). Acceptance as
before plus |hip yaw| <= 1.5 rad and swing clearance >= -3 cm. Result: **57 of 108 fits accepted** (v1: 38). |hip yaw|
median 0.45 rad, max 0.74 (v1: 3.43); largest joint jump 0.78 rad per frame (v1 1.10; the fitted climbs gst_v1 0.77);
swing clearance median -1.2 cm, worst -3.0. The node's `clip_quality.py` on `cfg/stage2c_sets.json` agrees: no leg flip,
no sole-penetration flag. 55 of 57 have a jump over 0.35 rad, the same class as the climbs already in training. Prop slots
are the same as v1 (26). Files: `data/clips/gs2dn_v2{.pkl,_props.json,_report.json}` (sha256 in `code/MOTIONS.md`).
**Staged, not in training yet:** `cfg/stage2c_sets.json` (= 2b + gs2dn_v2 at 0.05, terrain-park stairs back to 0.10;
label cache already built). Swap in at the next needed restart (`DATA=cfg/stage2c_sets.json`).

**Motion recipe:** `code/MOTIONS.md` (coordinator / user rule): every clip and label file with size, sha256, clips and
minutes, source, builder and command, filters and licence. The upstream desktop builders are copied as text under
`code/data_build/upstream/` (with their repo commits), and ours under `code/data_build/`. No motion data in `code/`.

**Live evaluation test** (armed: `code/ops/live_test.sh 2750`, tmux `acp-evaltest`): right after `ckpt_002750`, two
paper-eval sizes one after the other (100 and 334 episodes per set per seed, 3 seeds, oracle contacts), each through
`tools/eval_live.sh` (flock, 80 GB start check, nice 10, cores 0-3, `EVAL_TIMEOUT` 1,500 s, self-abort under 10 GB free).
Training speed before / during come from `eval_live.sh`, after from `live_test.sh`. Results go to
`eval/paper_eval/live_test_2750.log` and `live_s2_v1_002750_per*.live.json`, then here.

**Memory with the replay buffer full** (1.5 M samples from iteration about 2,591): 51.6 GB as logged by the trainer
(52.8 GB in nvidia-smi), 69,700-70,500 env-steps/s. Verified.

**Live evaluation test, checkpoint 2,750** (02:05-02:14 UTC; `eval/paper_eval/live_test_2750.log`, `live_s2_v1_002750_per{100,334}.{json,live.json,gpu.txt,out}`):

| Eval size | Robots | Eval time | Training env-steps/s before / during / after | Loss during | Peak total GPU memory |
|---|---|---|---|---|---|
| 100 per set per seed | 900 | 146 s | 69,930 / 44,345 / 69,596 | **36.6 %** | 53.9 GB |
| 334 per set per seed (full paper eval, 8,016 episodes) | 3,006 | 190 s | 69,596 / 39,278 / 69,695 | **43.6 %** | 56.6 GB |

Both evals finished (rc 0), with no abort, no OOM and no new trainer warnings. Training was back to full speed by the next iteration.
**Verdict: not accepted** under the coordinator's rule (at most about 20 % loss while running). The GPU is time-sliced between
the two processes, so `nice` (CPU only) cannot protect the trainer, and a 3x smaller eval only lowers the loss from 44 to
37 %. Cost per full eval is about 85 s of training (about 7 iterations), less than an eval inside a stop (3-5 min). The
choice is the coordinator's / user's. Until then, evaluation stays in needed restarts. The `acp-evaltest` tmux session has
ended; nothing is armed.

**Paper eval at 2,750 (stage 2, oracle contacts, data stage2b, 8,016 episodes, run live):** pooled **92.7 %** (mean of the
eight equal-size sets): Pico 2x2 100.0, ramps 98.1, flat 93.5, chains 93.4 (the 40 cm step-ups are now filtered out; 76.9
at 2,250 with them), 4x4 93.2, blocks 93.3, fitted climbs 90.1, terrain-park stairs 89.7. Not directly comparable with
2,250's 91.2 % (nine sets including the faulty descents, chains unfiltered). **Defect found:** every paper eval so far ran
MuJoCo Warp with `njmax` 300 (training uses 450) and printed nefc overflows (35-290 lines). Fixed in
`tools/paper_eval.py` (default 450, `ACP_NJMAX`) for future evals; the numbers above and before carry the defect.

**Ladder step 1 at 2,250 (eval monitor, coordinator's message):** stairs up level 1 (10 cm risers, 30 cm run) 23/50, all
27 falls at the first riser, while 14 and 18.6 cm passed 50/50. Riser heights in the training props
(`data/clips/*_props.json`): fitted climbs gst_v1 16-22 cm (none lower), terrain-park stairs two levels of 14 and 16 cm,
descents v2 14-22 cm, 4x4 8 cm, block steps mostly 4 cm and 10 cm, chains 10-11 cm stairs (46 clips) and the 40 cm step-ups
that are filtered out. So **no fitted stair flight has a riser under 14 cm**; 10 cm appears only as chain stairs and single
blocks. Action: low-riser flights (risers 10-15 cm, treads 25-32 cm, climbs and descents) are being fitted on the desktop
(`code/data_build/gen_stairfit_low.py`, same rules and screens as the descents v2) for the stage-2c data. Not done: the
planner's foothold on a 10 cm riser in a closed loop (my CPU check runs on reference terrains; the eval monitor's synthetic
stairs are not in my tools yet).
