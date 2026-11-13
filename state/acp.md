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
12. Arm clearance data fix (stage 2d, staged for the next needed restart; coordinator 2026-11-12): the references put an
    arm inside its thigh or the torso on 28-94 % of frames (any penetration) and the simulator collides the robot with
    itself. We add outward shoulder roll per frame (then elbow extension and outward shoulder yaw where the roll alone,
    capped at 0.6 rad, is not enough) for at least 1 cm clearance (0 for the shoulder links against the torso), smoothed
    and rate-limited. The paper uses its references as retargeted. Reason: arm poses the robot reaches only by hitting
    itself.
13. Planned for stage 3 (user's eval rule): goal radius 0.2 m for arrival and goal scaling in the goal environments (the
    paper: 0.5 m success) and routes with sharp heading turns right after a waypoint. Reference environments keep the
    paper's 0.3 m (Table S4).
14. Stage 3 v1 (section 15): six goal families built from the existing prop-slot library (the paper: 32 procedural
    configurations incl. gaps, stones, ladders, vaults); goal share 0.5 (paper: not given); planner buffer not
    partitioned per tile and no work-per-metre quality score yet; 36,000 envs on one GPU (paper: 8,400 on 4).

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

**Coordinator's decision on live evaluation (02:30 UTC):** accepted. Limits: at most one full paper eval per 1,000
iterations (the planner-driven check with it if it fits), started right after a checkpoint, guard rails unchanged, no
per-250 evals; the training-speed cost of each eval is logged here. Armed: `code/ops/live_every.sh 4000 5000 6000` (tmux
`acp-evallive`), with one line per eval in `eval/paper_eval/live_costs.log`. Costs so far: 2,750, 900 robots: 146 s at
-36.6 %; 2,750, full: 190 s at -43.6 % (table above).

**All paper evals before 2026-11-12 02:20 UTC (checkpoints 1,250, 2,000, 2,250, and the two live ones at 2,750) carry the
nefc overflow** (MuJoCo Warp `njmax` 300 in the eval, 450 in training; 35-290 warning lines per eval). From then on the
eval uses 450 (`ACP_NJMAX`).

**Low-riser flights `gs2lo_v1`** (desktop CPU, `code/data_build/gen_stairfit_low.py`, `DN_GAP=0.05 python gen_stairfit_low.py <out>/gs2lo_v1 3`):
risers 10-15 cm, treads 25-32 cm (tread capped so the one wall size covers the flight), 3 random flights per capture,
climbs with the stock foot targets and descents with the down-facing ones. Accepted: **39 climbs** of 63 fits and **43
descents** of 81 (14 and 17 of them with risers of 12 cm or less); |hip yaw| at most 0.64 / 0.73 rad, joint jump at most
0.71 / 0.60 rad, swing clearance worst -1.7 / -2.7 cm. clip_quality: no leg flip, no sole-penetration flag. Box sizes are
the same seven as the other fitted flights, so no new prop slots. sha256 30db5030...b6e1c1.

**Planner footholds on low risers, before the swap** (`tools/riser_check.py`; CPU closed loop, tracker 2,250 + MLP planner
2,250, the eval monitor's pair; fall rule; gs2lo climbs vs gst_v1 climbs; output
`eval/handover_pair_s1_002250_mc_20261024_planner/riser_check_002250.json`):

| Risers | Episodes | Success, oracle contacts | Success, planner contacts | Planner foot targets within 3 cm of a riser edge (oracle: about 0) |
|---|---|---|---|---|
| 10-12 cm | 20 | 65 % (7 fell) | **35 %** (4 fell) | **14.6 %** |
| 12-15 cm | 28 | 96 % | 68 % | 12.0 % |
| 16-22 cm (trained) | 48 | 79 % | 85 % | 4.5 % |

Reading: on 10-12 cm risers both halves are weak. The tracker falls 7 of 20 times even with the reference's own contacts
(it has never trained on them), and the planner puts three times as many foot targets on a riser edge as on the trained
heights. Target heights sit on the surface (median 0 cm above). This agrees with the eval monitor's 23/50 at 10 cm and
50/50 at 14 and 18.6 cm. Fix: train on them (stage 2c below). In stage 2 the planner learns online from the tracker's own
rollouts on these terrains. Small sample (20-48 episodes per bin).

**Stage 2c = 2b + gs2dn_v2 (0.05) + gs2lo_v1 (0.05); terrain-park stairs 0.15 -> 0.08, gst_v1 0.10 -> 0.07**
(`cfg/stage2c_sets.json`; label cache built; clip_quality in `eval/clip_quality/stage2c_sets.txt`). It goes in through a restart
that is needed for the data change, at checkpoint 3,000 (armed: `restart_s2b.sh 3000` with `DATA=cfg/stage2c_sets.json`;
code unchanged). The resume command in `code/ops/README.md` now says stage2c.

**Restart at checkpoint 3,000 (data change to stage 2c), done:** STOP at about 03:00:0x UTC, resume check passed
(iteration 3,000 to 3,003, stage-2c data on the GPU), real run back at 03:00:45; training was down for under 2 minutes. Data: 1,660 segments,
187.6 min, **prop slots 94** (69 before: the walled flights of gs2dn_v2 / gs2lo_v1). Speed **65,000-65,700 env-steps/s**
(70,000 before, -7 % from the extra prop slots), GPU memory 52.1 GB. First live finish rates: descents v2 0.85-0.89, low
risers 0.91-0.96. **What runs now:** stage 2, `runs/s2_v1`, data `cfg/stage2c_sets.json`, from 3,000. Resume command as
above with `DATA=cfg/stage2c_sets.json`. Next: live paper eval at 4,000 (armed, `acp-evallive`), then 5,000 and 6,000; budget
to 6,500. eta_obs reaches 1 at iteration 3,950; the eta_rew ramp runs from 3,950 to 5,450 (the s2_start of 2,250 + 1,700 + 1,500).

**Self-penetration checks (Hiking port audit, 2026-11-12 03:20 UTC; `code/tools/self_pen.py`, output
`eval/clip_quality/self_pen_stage2c.{txt,json}`; training model's collision geoms, every 5th frame of the stage-2c clips):**
- Self-collision is **on** in our simulator: 130 of 132 arm-thigh/torso geom pairs and all 465 leg pairs generate contacts.
- Arm inside thigh in the references (share of frames over 1 cm / over 2 cm, worst): terrain-park stairs **91 % / 83 %,
  8.0 cm**; chains 42 % / 28 %, 9.6 cm; 4x4 39 % / 26 %; Pico 2x2 33 % / 24 %; ramps 25 % / 16 %; flat and blocks
  23 % / 11 %, 9.8 cm; fitted stairs 16-18 % / 6-9 %. Arm inside torso: flat, blocks and chains 7-9 % over 1 cm (max 10 cm);
  none elsewhere.
- What the tracker is paid for: `track_joint_pos` covers all 31 joints (arms included) and the contact-point terms include
  the wrists, so the reference asks for arm poses the robot reaches only by pressing the arm into its thigh. The reward at
  stake is small (an arm 0.2 rad off on two joints costs about 1 % of `track_joint_pos`; sigma_c = 1 m makes the wrist-point
  terms nearly flat). The contact forces in the rewards are robot-terrain only, so self-contact is never penalised. The
  real cost is physical: the arm PD pushes against the leg in swing, and hands knock the thighs on hardware.
  Proposed data fix, not done: in the data build, add shoulder roll per frame until arm-thigh clearance is at least 1 cm.
  It would go in at the next needed restart.
- Legs in the MJCF: no overlap. Leg geoms are at least 5 cm apart at the default pose and at hip yaw +-0.2 and -0.4; at
  +0.4 (toes out) the closest pair is the two feet at +0.4 cm. The URDF-hull overlaps found for Hiking do not occur in our
  model.

**Arm clearance fix, built and verified, staged as stage 2d** (`code/tools/arm_clear.py` -> `code/cfg/armfix_stage2d_sets.pt`;
`acp/data.py` `_armfix` in `code_next`, applied when the cfg has `"armfix"`; `cfg/stage2d_sets.json` = stage2c + armfix;
label cache `labels_stage2d_sets.json.pt` built with the fix; table `eval/clip_quality/arm_clear_stage2d.txt`). Per frame and
arm: the smallest outward shoulder roll (bisection, at most 0.6 rad) giving at least 1 cm between that arm's geoms and the
thigh / torso / pelvis geoms the simulator collides (0 cm for the shoulder pitch / roll links against the torso: they sit
0-7 mm from it at every pose). Where 0.6 rad is not enough (hand in front of the belly, shoulder yaw +-1.3-1.5): elbow
extension, then outward shoulder yaw. Smoothed (max filter + Hann over +-10 frames, never below the need) and
rate-limited; joint velocities follow by central differences. Two earlier builds are in `_trash` (the first asked 1 cm of
the shoulder links; the second left 3.7 % of flat frames).

| Set | Frames | Arm short of clearance by > 1 cm, before -> after | Worst shortfall before -> after (cm) | Mean / max roll added (rad) | Frames with elbow / yaw used | Max added joint step (rad per frame) |
|---|---|---|---|---|---|---|
| flat locomotion | 149,379 | 43.8 % -> 0.1 % | 11.1 -> 6.8 | 0.10 / 0.60 | 7,124 / 1,316 | 0.065 |
| blocks | 149,379 | 43.8 % -> 0.1 % | 11.1 -> 6.8 | 0.10 / 0.60 | 7,124 / 1,316 | 0.065 |
| chains | 33,953 | 63.5 % -> 0 | 11.1 -> 0 | 0.13 / 0.60 | 528 / 0 | 0.050 |
| terrain-park stairs | 24,530 | 93.5 % -> 0 | 9.0 -> 0 | 0.14 / 0.20 | 0 | 0.009 |
| fitted climbs | 28,231 | 36.0 % -> 0 | 9.4 -> 0 | 0.07 / 0.41 | 0 | 0.031 |
| descents v2 | 13,033 | 27.5 % -> 0 | 8.9 -> 0 | 0.07 / 0.28 | 0 | 0.018 |
| low risers | 19,643 | 34.3 % -> 0 | 10.1 -> 0 | 0.06 / 0.35 | 0 | 0.014 |
| ramps | 131,615 | 35.2 % -> 0 | 8.7 -> 0 | 0.04 / 0.28 | 0 | 0.025 |
| 4x4 | 10,868 | 57.1 % -> 0 | 7.8 -> 0 | 0.07 / 0.17 | 0 | 0.012 |
| Pico 2x2 | 2,259 | 48.6 % -> 0 | 7.4 -> 0 | 0.06 / 0.17 | 0 | 0.014 |

("Short by > 1 cm" = any penetration for the arm-thigh pairs.) The largest added joint step is 0.065 rad per 50 Hz frame
(3.3 rad/s), far under the joint speed limits. Independent check through the staged loader (`self_pen.py` on stage2d): arm
in thigh by more than 1 cm on 0 % of frames in every set (before: 16-91 %); worst remaining 5.5 cm in the 0.1 % of flat /
blocks frames; arm in torso at most 2.4 cm (was 10 cm). 472,336 of 562,890 frames changed, largest offset 0.80 rad.
**Swap-in:** at the next restart that is needed for another reason (not a restart of its own): `restart_s2b.sh <it>` with
`DATA=cfg/stage2d_sets.json`. The staged code (`code_next/acp/data.py`) is checked by the throw-away resume run before the real run
resumes. The 1,000-iteration live paper eval will show whether tracking changed.

**Goal convention and exact arrival (eval monitor at 2,250 on our apparatus: 0/200, 71 of 108 stalls standing 0.20-0.40 m
from a waypoint):**
- What training uses (and the adapter card states, field `goal_command`, r_training 0.3): the goal is the reference root
  0.5-1.5 s ahead (per-episode lookahead, `LOOKAHEAD` 25-75 frames), (dx, dy) = d / max(|d|, 0.3) in the body-yaw frame,
  dpsi = goal heading minus robot heading. The 0.3 m is the paper's Table S4 value for reference following. In stage 2 the
  goal therefore almost never comes to rest inside 0.3 m except at clip ends. A waypoint fed as a fixed target makes
  the policy treat anything inside 0.3 m as arrived. The card matches training; the monitor's re-run with a lookahead past
  non-station waypoints is the right adapter for this policy.
- Stage 2 has no goal environments (the paper's stage 2 is reference-only), so 0.2 m arrival and turns after a waypoint
  cannot be trained there without leaving the paper. Plan: **stage 3** (the paper's goal environments) with
  goal radius 0.2 m (arrival disc and goal scaling; the paper uses 0.5 m success, a listed departure for the user's rule),
  routes with sharp heading turns (+-70-90 degrees) right after a waypoint, station waypoints with the stand-at-goal
  reward, on the extended-map tiles (section 8). Stage 3 starts at the stage-2 end (budget 6,500, about 12 h at the current
  speed) or earlier on the switch criterion. Its code is the next CPU work item, written and CPU-tested before then. Until
  stage 3, any pair handed over keeps r = 0.3 and the reference-lookahead convention in its card.

## 15. Stage 3: code written and CPU-tested, switch armed (2026-11-12 03:40-04:00 UTC)

**Approved by the coordinator:** stage 2d at the next needed restart; the stage-3 plan (departure 13). Stage-3 code is
staged in `/mnt/filesystem-g4/tracks/acp/code_s3` (plain-text mirror in `code/staged/s3/`; `code/staged/README.md`).

**Goal terrains: why boxes from the existing slot library.** The user's extended map (`data/terrains/extended_v1`) has 60-346
distinct box sizes per tile type and meshes for wedges. In our simulator every distinct prop-box size is a slot in every
world, and a large height field degrades MuJoCo Warp. So v1 builds goal tiles **only from box sizes, and at most as many
of each, that the stage-2 reference props already use**: tread slabs (7 each of three widths), landing slabs, two
4.6 m walls, one 4x4 log, one 2x2 log, the small blocks. Heights vary by sinking a box into the floor. Result: **prop
slots stay at 94** (checked by `acp/goals.py` `check_slots` and by the data build), so no extra GPU cost. Six families,
each re-arranging extended-map ideas: stairs up, stairs down, logs (4x4, plus 2x2 from level 4), corridor (two walls, a
4x4 inside from level 3), blocks, flat turns (turn in place, backward, sideways). 10 levels x 12 seeds = 720 tiles
(`data/goals/goal_tiles_v1.json`, `python -m acp.goals ... 12`). Levels: stair risers 8 -> 22 cm (never within 6 mm of
both apparatus riser and tread), log height 3 -> 8.9 cm, corridor gap 1.3 -> 0.8 m, block count. Every route ends with a
station (1 s stand). The heading turn right after a waypoint grows to +-70-90 degrees. No block or obstacle within
0.35 m of a waypoint or the spawn. Tiles sit at x = -40 m, away from the shared ramp raster. Bringing in the real
extended-map tiles (raster tiles plus their tall boxes) is v2. It needs a GPU test of a larger height field and a
size-quantised wall library; not done.

**Goal environments (`acp/env.py`, gated by `goal_env`, so stage 2 is unchanged):**
- Share: `--goal-share` 0.5, spread over the env ids, so the recorder's first 3,000 envs hold both kinds (paper: ratio not
  given). Families round-robin; each env has its own level.
- Command: towards the current waypoint, (dx, dy) = d / max(|d|, 0.2) in the body-yaw frame; dpsi = waypoint heading
  minus robot heading. A pass-through waypoint is done on entering the 0.2 m disc; a station after 1 s inside; the last
  waypoint is the goal, and success is latched once it has been held 1 s. Planner contacts only.
- Rewards (Table 3, stage 3; G in env.py): move2goal 40, standatgoal 20 (at stations and the goal), track_planned_contact
  1.5, flat_orientation -10, stumbling -10 (foot |F_xy| > 5 F_z), slippage -10, ang_vel_xy -0.1, and the regularisers.
  No mimic terms.
- Ends only by a fall (root under 0.35 m over the lowest ground under root and feet, or tilt over 60 degrees) or 15 s.
- Curriculum (Table S4): EMA s = 0.8 s + 0.2 I; promote on a success with EMA > 0.5; demote when the closest approach
  to the goal was over 1 m.
- Planner: goal successes enter the same hindsight buffer (recorder success = goal reached and held). Not done yet: the
  paper's per-tile partition of the buffer and its quality score (low mechanical work per metre).

**Trainer (`acp/train.py`, `--stage 3`):**
- Second critic V_fs (input 322: noise-free proprioception, base velocity, 17 x 11 scan, goal, goal distance, height
  error, normalised time, planned contacts), fixed lr 1e-3, its own normaliser.
- Advantages normalised separately per environment type.
- 500-iteration warm-up with goal samples left out of the actor loss.
- Target KL 0.01, entropy 0.002 (Table S6).
- The eta schedules continue from the stage-2 start.
- Checkpoint adds V_fs, its optimiser, the per-env curriculum levels and EMA, s3_start.
- Metrics: `goal/success`, `goal/fall`, `goal/level_mean`, `goal/level/<family>`, `goal/waypoint_index_mean`,
  `raw|weighted|weight/goal/<term>`, `ppo/value_loss_goal`.

**CPU tests:**
1. Training smoke: 64 envs from `ckpt_003000`, 4 iterations, stage 3 end to end (data with 720 tiles, slots 94, V_fs,
   per-type advantages, warm-up switch, checkpoint with stage 3 fields).
2. Closed loop of the stage-2 pair at 3,000 (tracker + online planner, planner contacts) on the goal tiles,
   `tools/goal_check.py`, 4 episodes per family and level, levels 0 / 4 / 9 (`eval/goal_check_s2_003000.json`).
   Results, success out of 12: logs 12/12; stairs up 6/12; flat turns 3/12; stairs down 1/12; corridor 0/12; blocks 0/12.
   **No falls in 72 episodes.** Where it fails, the robot does not move off: corridor (walls never seen in training), blocks,
   stairs down (descents were out of training until 3,000), flat turns with backward and sideways moves. That is the
   expected starting point (the planner has never seen these scans). The tiles, goal command and waypoint logic work
   (logs and stairs up succeed to 4-15 cm).

**Switch (armed, tmux `acp-s3switch`, `code/ops/to_stage3.sh`, log `runs/s2_v1/stage3_switch.log`):**
- Criterion (ours; the paper gives only wall-clock): not before eta_rew = 1 (iteration 5,450); from checkpoint 5,500,
  switch at the first checkpoint where the stage-2 finish rate (mean of the last 250 iterations) gained under 1 point over
  the 250 before; at the latest 6,500.
- The switch waits for a live eval to finish (shared lock), stops stage 2, installs `code_s3` into `code` (previous code
  to `_trash`), checks stage 3 in a throw-away GPU run (4,096 envs, 3 iterations), then starts stage 3 as `runs/s3_v1`
  from that checkpoint, 36,000 envs, budget to iteration 14,000 (`S3_ITERS`; paper: about 2 days 6 h on 4 GPUs).
- On a failed check the previous code is restored and stage 2 resumes.
- `FORCE_IT=<it>` switches at a chosen checkpoint.
- The stage-3 data cfg includes the arm fix, so stage 2d arrives with stage 3 at the latest.

Not verified: stage 3 on the GPU (speed, memory, warp behaviour with goal envs). The switch's own check covers it.

## 16. Resume check by a new ACP agent (2026-11-12 04:55 UTC; nothing restarted)

- Run `s2_v1` healthy: iteration 3,517 at 04:55, 64-65k env-steps/s, GPU 85% / 53 GB, newest `ckpt_003500.pt` 04:51
  (written straight to the shared filesystem). Backup loop `acp-backup` alive, last pass 04:42:32 UTC (`backup/MANIFEST.txt`).
- `acp-evallive` (`live_every.sh 4000 5000 6000`) alive and waiting for `ckpt_004000`; `live_costs.log` does not exist yet,
  its first line is written at 4,000 (the 2,750 cost is in `live_s2_v1_002750_per334.live.json`: 190 s, 43.6% slowdown).
- `acp-s3switch` (`to_stage3.sh`) alive, `bash -n` clean, armed 03:58:41. Plateau function dry-run on the live metrics works
  (keys `it`, `end/finished_phase099` present). Finish rate is already flat: 94.29 -> 94.39% at 3,250, 94.56 -> 94.57% at
  3,517 (gain under 1 point), so the criterion will almost certainly fire at the first allowed checkpoint, 5,500 (about
  11:45 UTC at 12.4 s/iteration). If it switches at 5,500, the 6,000 entry of `acp-evallive` never fires (harmless; the
  watcher then just waits).
- Earliest apparatus-scorable stage-3 checkpoint: after the 500-iteration warm-up, i.e. about s3 6,000-6,250 if the
  switch is at 5,500 (about 14:00-15:00 UTC). Proposed to the coordinator, not done: switch earlier (FORCE_IT=4000, right
  after the 4,000 live eval), since stage 2 has not gained since 3,000; against it: eta_rew reaches 1 only at 5,450.

### 16a. Early switch at 4,000 approved by the coordinator on conditions (2026-11-12 ~05:10 UTC)

**The 2,750 figure, reconciled.** Section 14 says "pooled 92.7 %"; the file `eval/paper_eval/live_s2_v1_002750_per334.json`
says 93.92 % pooled and 93.92 % sets-weighted-equally (8,016 episodes, Wilson 93.4-94.4). The mean of the eight per-set
values listed in section 14 itself is 93.9 %, so **92.7 was an arithmetic slip in this file; 93.9 % is the value**.
From now on **"the paper eval" = `category_dense_foot_only.success_pooled` of the `*_per334.json` file** (334 per set
per seed, 3 seeds, eight stage-2 sets). Caveat for like-for-like: 2,750 ran with `njmax` 300 (nefc overflows, section 14);
4,000 runs with 450. A small rise at 4,000 may be the fix, not learning.

**Conditions and what the code says:**
1. Plateau: finish rate (`end/finished_phase099`, 250-iteration means) since 3,000 under +1 point, and the 4,000 paper eval
   at most 94.9 % (93.9 + 1). Checked after the 4,000 eval.
2. Reward ramp: stage 3 does **not** cut it. `acp/train.py` (code_s3) computes eta from `it - s2_start`, and `s2_start` is
   loaded from the checkpoint (`ckpt_003500.pt` holds `s2_start` 2250, checked). At 4,000: u = 1,750, eta_obs 1.0,
   eta_rew = (1750 - 1700) / 1500 = 0.033, then the same ramp to 1 at 5,450 inside stage 3. Nothing to change.
3. Roll-back: checkpoints are written straight to the shared filesystem (`runs/s2_v1/ckpt_004000.pt`), and the switch keeps
   `code_prev/`. Stage-2 resume from 4,000: restore code (`cp -r code_prev/acp code_prev/tools code/`), then
   `tmux new -d -s acp "ACP_PLANT=mc_20261024 DATA=cfg/stage2c_sets.json EXTRA='--stage 2 --init /mnt/filesystem-g4/tracks/acp/runs/s2_v1/init_from_s1_v1_002250.pt --planner-arch attn --u-obs 200 --u-rew 1700 --n-ann 1500' bash /mnt/filesystem-g4/tracks/acp/code/supervise.sh /mnt/filesystem-g4/tracks/acp/runs/s2_v1 36000 6500"`
   (the supervisor resumes from `latest.pt` = 4,000; check that `latest.pt` is the 4,000 one first).
   **Divergence in the first 250 stage-3 iterations, fixed before the start:** over iterations 4,150-4,250, any of:
   reference-env finish rate under 90 % (stage 2: 94.5 %; reference-only = (finished x episodes - goal/success x
   goal/episodes) / (episodes - goal/episodes), since `end/finished_phase099` mixes both env kinds in stage 3);
   reference termination over 10 %; goal/fall over 25 %; any `diverged` / NaN; value loss over 5x its stage-2 mean;
   env-steps/s under 50 % of stage 2 (64k). Then roll back to stage 2 at 4,000 and report.
4. GPU idle: the switch stops stage 2, runs the 3-iteration check (300 s limit) and starts stage 3; under 10 minutes.
   One training job at a time (the check runs only after the stage-2 job has left the GPU).

### 16b. latest.pt at least every 30 minutes (coordinator, 2026-11-12)
Numbered checkpoints every 250 iterations = about 55 min, over the 30-minute rule. `code_s3/acp/train.py` (and the mirror
`code/staged/s3/acp/train.py`) now has `--latest-minutes` (default 20): `latest.pt` is also written when 20 minutes
have passed since the last save, with a `[save] latest.pt at iteration N (time-based ...)` log line; numbered
`ckpt_*.pt` stay every 250 (retention: `backup.sh` keeps the newest two plus every 1,000 and moves the rest to `_trash`).
Pre-patch copy: `_trash/2026-11-12/pre_latest_minutes/code_s3_train.py`. The running stage-2 trainer cannot change this
without a restart. The trainer needs CUDA (`gear_sonic/.../kinematics.py` at import), so there is no CPU test; the
switch's 3-iteration GPU check runs with `--latest-minutes 0.0001` and requires the time-based save line.

### 16c. Stage 3 is not the only plan (user rule "keep iterating and find ways", 2026-11-12)

**Checks (times assume the switch at about 06:45 UTC and about 13 s per stage-3 iteration; corrected at the switch):**
- **s3 +1,000 (`runs/s3_v1/ckpt_005000.pt`, about 10:30 UTC):** to the eval monitor for ladder step 1 and our apparatus;
  ours: `tools/goal_check.py` on CPU (8 per family and level, levels 0/4/9, 0.20 m disc, 1 s stand).
  "Working" = all of: training `goal/success` >= 50 % with `goal/level_mean` >= 3; goal_check level 0 >= 80 % in every
  family, flat turns (in place, backward, sideways) >= 80 %, corridor >= 50 %, falls <= 5 %; reference-env finish >= 92 %
  (stage 2: 94.5 %). Apparatus: any trial reaching more waypoints than the 3,500 stage-2 pair (0/200, stalls short).
- **s3 +2,000 (`ckpt_006000.pt`, about 14:15 UTC):** apparatus above the baseline 15/200 (aim >= 40/200); ladder 10 cm
  stairs >= 45/50, flat turns 50/50; goal_check level 4 >= 70 % per family.
- Missing the +1,000 marks starts the fallback below at once (not at the 14,000 budget end).

**Fallback experiments (if stage 3 does not beat 15/200), with the change each makes:**
- (a) **Planner gap** (coordinator's figure: planner contacts reach the end 58 % vs 93 % with oracle contacts; I have not
  found the 58 % in this file, to re-measure): measure planner-vs-oracle on the same pair on CPU (`tools/pair_eval.py`),
  then fine-tune the attention planner offline on hindsight labels from successful goal and reference rollouts
  (`python -m acp.planner label/train`, as in `ops/example_planner_job_run.sh`), loaded into the next restart. Needs
  rollouts from a recent checkpoint (the live eval files at 2,750 have none; record them on CPU with `pair_eval`).
- (b) **Data for what the ladder fails:** 10 cm stair clips (training props hold 14-22 cm only, section 14), heading turns
  of +-70-90 degrees at waypoints, log corridor. Stage-3 v1 tiles already cover 8-22 cm risers, sharp turns and a
  corridor; if they are not learned, add stage-2 reference clips of 8-12 cm stairs and turn-in-place clips (data build on
  CPU, enters at a needed restart).
- (c) **Stage 3 v2:** extended-map tiles (`data/terrains/extended_v1`) plus re-arranged single apparatus parts, allowed by
  the held-out rule; parts to copy: the 4x4 log (step-over height and width), the slanted 4x4, the wedges, the stair
  flight (riser and tread), the doorway width, the corridor walls. Needs a size-quantised box library and one GPU test of
  a larger height field at a needed restart.
- (d) **Goal command:** stage 3 already trains arrival in a 0.20 m disc and a 1 s stand (stations and goal). If the
  apparatus still stalls, train with the eval's exact route semantics: waypoint switching on 0.20 m entry, stand only at
  stations, the eval's goal lookahead, 180 s routes made from tile chains.

**CPU work started now (no GPU):** goal_check of the stage-2 pair at 3,500 on all six families at levels 0/2/4/6/9,
8 episodes each (tmux `acp-cpu`, `CUDA_VISIBLE_DEVICES=`, nice 19, cores 8-23, 3 h limit; out
`eval/goal_check_s2_003500.json`). It is the stage-2 baseline that the s3 +1,000 numbers are compared with. Next on CPU after
it: (a) planner-vs-oracle measurement with `pair_eval`, then (b) the data build. (c) and (d) need a restart.
- (e) **Tracker under-trained** (coordinator, 2026-11-12): stage 1 stopped at 2,250 (about 1.5 billion samples; the
  paper's stage 1 is about 10 billion / 11,500 of our iterations and ends at >= 98.5 % per category). Paper eval with
  reference contacts, per set, 2,250 -> 2,750: flat 92.7 -> 93.5, blocks 91.6 -> 93.3, 4x4 92.5 -> 93.2, ramps 95.6 -> 98.1,
  terrain-park stairs 88.9 -> 89.7, fitted climbs 89.6 -> 90.1, Pico 2x2 100 -> 100, chains 77.0 -> 93.4 (filtered, not
  comparable); mean of the seven comparable sets 93.0 -> 94.0 (+2.0 points per 1,000). Flat walking at 93.5 % says the
  tracker itself is short of the paper, not only the obstacle sets. **Test:** per-clip failure concentration on CPU (if a few
  clips hold most failures: clip defects, repair or drop; if spread: training) plus the slope at 4,000 / 5,000. **Change:**
  longer reference-only training (stay in stage 2, or stage 3 with a larger reference share than 0.5) until every set is
  >= 98.5 % or the slope is flat.
- **Switch at 4,000 is on hold** (coordinator, 2026-11-12): report the 4,000 eval with the per-set table and a
  recommendation first; the armed 5,500 / 6,500 switch stays untouched until the coordinator decides.

### 16d. Pair export per checkpoint, automatic (eval monitor's request, 2026-11-12 05:40 UTC)
- `code/ops/export_pair.sh RUN IT` (CPU, CUDA hidden): `eval/pair_<run>_<it>/` with `tracker_ckpt_<it>.pt`, `actor.npz`
  (eval's `export_actor.py`, goal r 0.3 for s2, 0.2 for s3), `planner_attn.pt` / `planner_attn.onnx` (the checkpoint's own
  online planner, `ckpt['planner']`), `adapter_card.json` (stage-specific label), `obs_check.npz` (`tools/handover.py`,
  MuJoCo Warp on the CPU = the training simulator), `DONE`. 3,500 done in 33 s; ONNX vs torch max difference 1.2e-7.
- `code/ops/export_every.sh` in tmux `acp-export`: every numbered checkpoint of `runs/s2_v1` and `runs/s3_v1` from 3,500
  on, 60 s after it is written, cores 8-23, nice 19; log `eval/export_every.log`.

### 16e. Goal-tile baseline of the stage-2 pair at 3,500 (CPU, 8 per family and level, 240 episodes, 0.20 m disc)
Success / 8 at levels 0/2/4/6/9: stairs up 8/8/8/7/6, stairs down 6/5/5/6/1, logs 8/8/8/8/8, corridor 0/0/2/0/0,
blocks 4/1/0/0/0, flat turns 1/2/4/4/1. Total 119/240; falls 6/240 (stairs up levels 6 and 9). At 3,000: logs 12/12,
stairs up 6/12, flat turns 3/12, stairs down 1/12, corridor and blocks 0/12. Corridor, blocks and flat turns (backward,
sideways, turn in place) are what stage 3 must teach. `eval/goal_check_s2_003500.json`.

### 16f. Stage 3 disarmed; base first (user via coordinator, 2026-11-12: "we might get the base more stable first probably")
- 05:27 UTC: tmux `acp-s3switch` killed (it was waiting for `ckpt_005500`; it had changed nothing). Line in
  `runs/s2_v1/stage3_switch.log`. `code/ops/to_stage3.sh`, `code_s3/` and the stage-3 cfg are kept ready. Stage 3 starts only
  on the coordinator's go, after the **base gate**: tracker paper eval with reference contacts >= 98 % on flat and blocks,
  >= 95 % on every other trained set, and the planner-driven closed loop within 5 points of the oracle one. Stage 2 trains on.
- **Per-clip failure analysis** (`code/tools/fail_clips.py`, CPU, on the only saved rollouts: 2,250, stage-1 sets, 334
  per set, out `eval/paper_eval/s1_v1_002250_fail_clips.txt`). Terminations are ours (paper silent): 0.5 m root position
  or 1.0 rad orientation error against the reference; success also needs the end within 0.5 m of the segment end.
  - flat: 25 fail of 334, all early ends, 18 of them lagging > 0.3 m behind the reference, none low (no falls).
    Clips with "jump" in the name fail 14 of 14, "crouch" 7 of 13; together about 21 of the 25 failures. Failure rate by
    reference speed: < 0.6 m/s 11 / 300, 0.6-0.9 m/s 13 / 32; by peak yaw rate: < 1 rad/s 1 / 149, > 2 rad/s 11 / 35.
    Without jump and crouch clips flat would be about 4 / 308 failures (98.7 %).
  - blocks: 29 / 334; jump 13 / 13, crouch 5 / 11; without them about 11 / 310 (96.5 %); the rest are fast (> 0.6 m/s) or
    sharp-turn (> 2 rad/s) clips.
  - fitted climbs gst_v1: all 36 failures are jog-up clips (36 of 53 jog episodes fail), all with the root low (falls);
    without jog clips 0 / 281.
  - terrain-park stairs: 43 / 334, 8 of 10 failing clips hold the gap, 20 low (falls); 5 clips fail every time.
  - chains (since filtered): slow (< 0.3 m/s) clips lag and terminate, 63 / 163; 21 clips fail every time.
  - 4x4: 3 of 6 failing clips hold the gap (slow clips, lag).
  - Reading: the cap near 93-94 % is mostly specific clip kinds (jumps, crouches, jog-ups, a few stair and 4x4 clips),
    not walking in general; fast and sharp-turn walking is the part that is training, not data.
- `tools/paper_eval.py` now also writes `<out>_episodes_seed<k>.pt` (clip, success, failed, set per episode, every seed;
  small, inside a try) so the 4,000 live eval gives the same analysis for the current tracker and data 2c (copy before the
  change: `_trash/2026-11-12/pre_episode_record/`).
- **Data set 2e (proposal):** 2d arm fix plus: drop jump and crouch clips from flat and blocks; drop jog-up clips from gst_v1;
  drop the clips that fail every time in the 4,000 per-episode record (stairs, 4x4, chains) after a look at each (bad
  retarget, prop mismatch, sole or arm penetration via `tools/clip_quality.py` / `ground_check.py`). Expected: flat about
  98.5-99 %, blocks about 96-97 %, gst about 99 %, terrain-park stairs about 97 % on the remaining clips. Gate numbers are
  reported on the 2e sets and, beside them, on the old sets (so a rise is not a change of data sets alone), with the
  excluded clips listed.
- **When:** no restart is due now (stage 3 is off). Proposal: one short stop (under 2 minutes, as at 2,500 and 3,000) right
  after the 4,000 live eval and its per-clip check, if the check confirms the same clip kinds.
- Planner gap at 3,500: `tools/pair_eval.py` on CPU, 40 per set, oracle vs planner contacts (tmux `acp-cpu`, out
  `eval/pair_s2_v1_003500/pair_eval_40*`); running.

### 16g. Decision: data set 2e = 2d arm fix + style filter, restart armed right after the 4,000 live eval (ACP agent, 2026-11-12 05:40 UTC)
**Fact (coordinator's count):** `x2_loco_flat_train.pkl` (351 clips) is SuperMimic's general-tracking "flat" file (flat
ground, no style filter): bump 87 clips, walk 80, injured 79, idle 32, crouch 22, jump 20, turn 15, back 12, other 4.
`x2_loco_blocks_train` is the same clips over blocks. SSR and Hiking filter it by name.
**Check, 2,250 rollouts (paper eval, reference contacts), fail / episodes by style (first matching word):**
flat: bump 1/106, injured 2/66, walk 0/66, idle 1/42, turn 0/20, jump 14/14, crouch 7/13, back 0/5, other 0/2 ->
kept styles 1/135 (99.3 %) against 25/334 (92.5 %) on the old set. blocks: bump 6/112, injured 2/66, walk 2/63, idle 1/46,
turn 0/14, jump 13/13, crouch 5/11, back 0/4, other 0/5 -> kept 3/132 (97.7 %) against 29/334 (91.3 %).
**Decision and reasons:** the apparatus needs plain walking, turns, slow walking, stops and starts, stands and stepping back.
- Drop jump and crouch: they hold most of the failures and the apparatus has neither.
- Drop injured and bump: they track fine (3/172 and 7/218), so they cost no success. But they are about 47 % of the flat
  data, and the stage-2 planner learns its contact plans from hindsight labels of successful rollouts. Limping and
  stumble-into-obstacle footholds on open ground are what the planner should not propose; the apparatus pair falls on
  open ground (street 31/50, amphitheatre 0/50).
- gst_v1: drop the jog-up clips (all 36 failures there, all falls).
Kept on flat and blocks: 143 of 351 clips each (62,704 frames, about 21 min); gst_v1 86 of 113. Set weights unchanged, so
plain walking is sampled more.
**Built on CPU:** `code/cfg/stage2e_sets.json` (as 2d plus `exclude_name`), `cfg/armfix_stage2e_sets.pt` (`tools/arm_clear.py`:
every set 0.0 % of frames short of 1 cm clearance after), `cfg/labels_stage2e_sets.json.pt`. 1,157 clips, 384,715 frames.
Loader: `exclude_name` / `exclude_clips` in `code_next/acp/data.py` and `code_s3/acp/data.py` (pre-change copies in
`_trash/2026-11-12/pre_2e_filter/`). `tools/paper_eval.py` takes `ACP_EVAL_DATA=<cfg>`, so the old-set numbers can be
measured beside the 2e ones.
**Restart:** tmux `acp-restart2e`, `code/ops/restart_s2e.sh 4000` (= restart_s2b plus a wait for the 4,000 live-eval line and
the eval lock). It stops stage 2, installs `code_next/acp`, runs the 3-iteration resume check on 2e, and resumes `s2_v1` from
4,000 with `DATA=cfg/stage2e_sets.json`, budget 10,000 (base first; stage 3 only after the base gate). If the check fails,
the previous code is restored and 2c resumes. GPU stop expected under 2 minutes. Log `runs/s2_v1/restart.log`.
Resume command after it: `tmux new -d -s acp "ACP_PLANT=mc_20261024 DATA=cfg/stage2e_sets.json EXTRA='--stage 2 --init /mnt/filesystem-g4/tracks/acp/runs/s2_v1/init_from_s1_v1_002250.pt --planner-arch attn --u-obs 200 --u-rew 1700 --n-ann 1500' bash /mnt/filesystem-g4/tracks/acp/code/supervise.sh /mnt/filesystem-g4/tracks/acp/runs/s2_v1 36000 10000"`.
**Not in 2e yet:** clips that fail every time on terrain-park stairs, 4x4 and chains. They wait for the 4,000 per-episode
record (`_episodes_seed*.pt`) and a look with clip_quality / ground_check, then go in through `exclude_clips` at the next
restart.
**Expected:** reference-contact eval on 2e about 99 % flat, 97-98 % blocks, about 99 % gst. Old-set numbers are run beside
them with `ACP_EVAL_DATA=cfg/stage2c_sets.json` at the 5,000 eval.

### 16h. Hip-roll hull finding: decision (c), one restart, armed (ACP agent, 2026-11-12 06:30 UTC)
**Finding** (`state/ssr_port_audit.md`; coordinator's shared-model decision): the pelvis / hip-roll convex hulls overlap by up
to 10.9 mm; the real meshes keep about 5 mm. The eval measured 753-799 N per side on that pair for the 3,500 pair while
walking, and on the corrected model the 3,500 pair falls 3 of 3.
**Reproduced in our builder:** `gear_sonic/supermimic/assets.py` now has `X2_EXCLUDE_PAIRS=hip` (same name as the eval's
switch): it adds `exclude pelvis / left_hip_roll_link` and `pelvis / right_hip_roll_link`. Default off, so running tools
are unchanged; pre-change copy in `_trash/2026-11-12/pre_hip_exclude/`. Hull penetration without it: -3.9 mm at 0.12 rad
inward and -10.9 mm at 0.235 rad (matches the audit). With it: no hip-roll contact at any of these angles.
**Our references ask for the blocked range** (2e data, CPU): inward hip roll over 0.05 rad (where the hull contact starts)
on 41.7 % (left) and 41.0 % (right) of all frames; over 0.12 rad on 17.0 % / 15.9 %; the most inward is 0.258 / 0.255 rad,
past the 0.235 limit. So the tracker could never follow the reference hip roll on 40 % of frames. That fits its lag on
turns and fast clips, where the step width changes.
**Finish-rate slip 94.5 -> 94.0 -> 93.2 % (3,401-3,650 / 3,651-3,900 / 3,901-3,932):** only flat (94.8 -> 93.6 -> 91.9) and
blocks (94.2 -> 92.9 -> 91.0) drop; every other set is flat within 0.5. eta_obs = (it - 2,250 - 200) / 1,500 reaches 1 at
3,950, so over these iterations planner contacts replace the last 20 % of the reference contacts in the tracker's input.
They do it worst on open flat ground, the same place where the apparatus pair falls. Not a data defect: the planner gap
on flat ground.
**Decision: option (c).** One restart, no 2e-only stop first (the armed 2e restart was killed at 06:24:56, unused):
- Run `s1_v2` = the paper's stage 1 (oracle contacts, CSI starts, stage-1 KL / entropy) on the corrected body, data 2e
  (style filter + 2d arm fix), initialised from s2_v1 `ckpt_004000` (tracker weights, Adam state, iteration count go on
  from 4,000). `latest.pt` every 20 min (`--latest-minutes`, now in `code_next/acp/train.py`); every checkpoint records
  `x2_exclude_pairs`, and `tools/paper_eval.py` sets the body from it.
- Why stage 1 and not stage 2: the base gate is on reference contacts, the user wants the base stable first, and stage 2's
  planner contacts are what slipped on flat ground. The s2_v1 planner at 4,000 is kept for the next stage 2
  (`--planner-init`).
- Why not (b) at once: a fine-tune keeps everything else the tracker learned (stairs, logs, ramps: 50/50 on step-1 tests).
  It is tried first under a fixed tripwire, and (b) starts by itself if it fails.
- No contact-softness ramp: MuJoCo Warp has no cheap per-iteration pair softness. The ramp is the stage-1 CSI curriculum
  (starts near clip start, phase rises on success) plus early termination.
**Tripwire** (`code/ops/tripwire_s1v2.sh`, tmux `acp-trip`, started by the restart; log `runs/s1_v2/tripwire.log`).
Stage-1 finish rate (`end/finished_phase099`, CSI starts, mean of the last 50 iterations) was 86.9 % at the end of
s1_v1. Marks: >= 60 % at 4,300 and >= 78 % at 4,600, else option (b) starts automatically as `runs/s1_v3` (fresh stage 1,
corrected body, 2e, no init). Recovery target: >= 87 % by 5,000.
**Restart** (`code/ops/restart_s1v2.sh`, tmux `acp-restart`, log `runs/s1_v2/restart.log`). It waits for s2_v1
`ckpt_004000` and the 4,000 live-eval line plus the eval lock, ends the s2_v1 live-eval watcher, copies the init, stops
s2_v1 and installs `code_next/acp`. Then a 3-iteration check: 4,096 envs, corrected body, 2e, the time-based save line
required. Then it starts `s1_v2` (budget 12,000). On a failed check: previous code back, s2_v1 resumes on 2c. GPU stop
expected under 10 minutes.
**Live paper eval for s1_v2:** `code/ops/live_every_run.sh` in tmux `acp-evallive2` at 5,000 / 7,000 / 9,000 on the 2e sets
and at 6,000 / 8,000 / 10,000 on the old 2c sets (`ACP_EVAL_DATA`). One eval per 1,000 as the rule allows, so the old- and
new-set numbers alternate.
**Up-weighting:** dropping 208 of 351 flat clips at the same set weight samples each kept clip (walk, turn, back, idle,
stop/start: the sharp-turn, fast and slow clips) 2.45 times as often.
**Not done / not verified:** whether the wrist / thigh contacts are physical (the audit's mesh-distance check on those pairs;
next on CPU). No apparatus scoring of s1_v2 (stage-1 pairs have no planner; the apparatus is scored again once stage 2
resumes on the corrected body). The door-corner stall (sideways goal at a standstill): it needs turn-in-place and
sideways data or the stage-3 flat-turn tiles; 2e keeps and up-weights the turn and back clips.

### 16i. 4,000 eval, restart outcome, s1_v2 running (2026-11-12 07:10 UTC)
**4,000 live paper eval** (s2_v1, standard body, reference contacts, njmax 450, 334 per set x 3 seeds, 10 sets, 10,020
episodes): pooled 93.0 %. Cost: 240 s, training 41 % slower while it ran, 64.1k env-steps/s after.
Per set, 2,250 / 2,750 / 4,000 (%): flat 92.7 / 93.5 / 90.9; blocks 91.6 / 93.3 / 92.5; 4x4 92.5 / 93.2 / 84.3; ramps
95.6 / 98.1 / 96.1; terrain-park stairs 88.9 / 89.7 / 93.0; gst_v1 89.6 / 90.1 / 84.5; Pico 2x2 100 / 100 / 100; chains
77.0 / 93.4 / 90.1 (filtered from 2,750); gs2dn_v2 - / - / 99.4; gs2lo_v1 - / - / 99.4. Mean of the seven sets in all three:
93.0 / 94.0 / 91.6, so -1.6 per 1,000 since 2,750. Stage 2 has made the tracker worse on reference contacts. Training
moved its input to planner contacts (eta_obs reached 1 at 3,950).
**Per clip at 4,000** (`_episodes_seed*.pt`, out `eval/paper_eval/live_s2_v1_004000_fail_clips.txt`):
- flat: jump 35/36 fail, bump 21/312, walk 13/201, crouch 9/36, turn 8/54, idle 3/118, injured 1/215, back 0/22. Styles kept
  in 2e: 93.8 % of 403 (2,250: 99.3 % of 135). Plain walking itself got worse in stage 2.
- blocks: kept styles 97.5 % of 402.
- gst_v1: jog 109/156; the rest 94.6 %.
- Clips failing every time (seen >= 3 times): 4x4 3 clips hold 44 of 157 failures (88.2 % without them); chains 4 hold 48 of
  99 (94.7 %); terrain-park stairs 1 holds 20 of 70 (94.9 %); ramps 4 hold 12 of 39. These are the `exclude_clips`
  candidates for 2f, after a look at each.
**Restart:** the scripted check "failed" only because the `[save]` log line was missing from `code_next` (the line had been
added to `code_s3` only). The check run itself reached 4,003 on the corrected body. My script then restarted s2_v1, but its
STOP flag was still in place (it moved the flag after the restart, unlike restart_s2b), so s2_v1 did not resume and the GPU
stood idle from 06:50:22 to about 06:52. I added the line, installed `code_next`, and started `s1_v2` by hand at 06:52 UTC.
Lesson: move the stop flag before starting the next job.
**s1_v2 recovery** (stage-1 finish rate, CSI starts): 4,001 0 %, 4,011 12 %, 4,021 36 %, 4,031 58 %, 4,041 76 %, 4,051 83 %,
4,061 89 %, 4,067 89.5 % (s1_v1 ended at 86.9 %). Root position error 0.198 -> 0.110 m. 118.6k env-steps/s (stage 1 has no
planner). The tripwire (`acp-trip`) will pass 4,300 and 4,600 if this holds.
**Shared model:** our builder's MJCF is the shared `x2_ultra.xml` (sha256 4f6a4f5b...). `X2_EXCLUDE_PAIRS=hip` adds the same
two excludes as `x2_ultra_hipexclude.xml`. Adapter cards now carry `trained_model` (`ops/export_pair.sh`; existing s2_v1
cards say the standard model).
**Planner gap at 3,500** (CPU closed loop `tools/pair_eval.py`, 40 per set, 10 sets, the same clips and starts; standard body;
caveats: njmax 300 overflow and height-field collision overflow warnings; small n):
- All sets: oracle contacts 95.0 % (400), planner contacts 74.3 %.
- Flat: oracle 97.5 %, planner 42.5 %. Blocks: oracle 90.0 %, planner 27.5 %.
- With a falls-only rule: flat 100 / 60, blocks 92.5 / 52.5.
- So on open ground the online planner's contacts cost about 55 points. This is fallback (a) with numbers. The base gate's
  "planner within 5 points of oracle" is far off, and it is the main lever for street and amphitheatre after the base.
  Next stage 2 on the corrected body: retrain the planner from the s1_v2 tracker's successful rollouts (hindsight labels,
  `acp.planner label/train`, CPU) before the co-training ramp, and measure flat and blocks separately.

### 16j. Decisions 07:30 UTC (ACP agent, 2026-11-12)
- **s1_v2 is kept; option (b) is not needed.** Finish rate 95.4 % at 4,251. Per set: flat 97.4, blocks 97.3, gst 98.2,
  Pico 98.9, gs2lo 97.1, stairs 95.9, ramps 95.8, gs2dn 95.0, 4x4 91.7, chains 88.5. 110k env-steps/s, 7.4 s per
  iteration. `acp-trip` stays as a guard.
- **Live paper eval brought forward:** `acp-evallive2` now at 4,500 / 5,500 / 6,500 / ...; still one per 1,000. Sets
  alternate: 4,500 on 2e, 5,500 on the old 2c sets, and so on. 4,500 is expected about 07:55 UTC, 5,500 about 10:00 UTC.
- **Return to stage 2 (mark):** the first live eval on the 2e sets with flat >= 98 %, blocks >= 98 % and every other set
  >= 95 % (reference contacts, corrected body). The old-set numbers are reported beside it and do not gate: they keep the
  excluded jump, crouch and jog clips by design. The restart then goes to stage 2 on the corrected body:
  - kept planner (`--planner-init` from s2_v1 4,000) and a new eta ramp (s2_start = the restart iteration);
  - data 2e plus `exclude_clips` (2f) for the always-failing 4x4, chains, stairs and ramp clips once each is checked.
  If 4,500 passes: restart about 08:00, first stage-2 pair on the corrected body at +250 (about 09:00 UTC). If not: the
  next 2e eval is at 6,500 (about 12:05 UTC).
- **Corrected-body pair available now,** without waiting for stage 2: `eval/pair_s1_v2_004250/`. It is the s1_v2 tracker
  (trained_model `x2_ultra_hipexclude.xml`) plus the kept s2_v1 4,000 planner, labelled interim (the tracker was not
  co-trained with this planner). `obs_check` was recorded on the corrected body: `export_pair.sh` now sets
  `X2_EXCLUDE_PAIRS` from the checkpoint. `acp-export` exports every s1_v2 checkpoint the same way (every 250
  iterations, about 31 min).
- **Arms against the real meshes** (2e with the 2d arm fix, 600 random reference frames, arm visual-mesh vertices to the
  same-side thigh and knee visual meshes, vertex spacing 0.3 mm; `eval/arm_real_mesh_2e.txt`): left: min 0.1 mm,
  < 5 mm on 7.5 % of frames, < 10 mm on 20.8 %; right: < 5 mm on 9.0 %, < 10 mm on 24.5 %. **2d does not clear the real
  meshes.** It cleared the training hulls by 1 cm, and the hip-roll collision mesh is 0.9-scaled in y. The no-fix
  comparison is still running (`acp-cpu2`). Fix for 2f: arm_clear against the real visual meshes (or a 2 cm hull target),
  checked with this script before the stage-2 restart.
- **Arm comparison done** (same script, same 600 frames): without the 2d fix the arms are within 5 mm of the real leg
  meshes on 47.7 % (left) / 40.7 % (right) of frames, touching (0.0 mm) at the 1st percentile; with 2d 7.5 % / 9.0 %, min
  0.1 mm. 2d removes about 80 % of the near-contacts on the real meshes; the rest is the 2f item above.

### 16k. 4,500 eval, gate, planner plan, 2f (ACP agent, 2026-11-12 08:30 UTC)
**4,500 live paper eval** (s1_v2, corrected body, data 2e, reference contacts, 10,020 episodes): 97.7 % pooled.
Per set: flat 99.0, blocks 98.3, ramps 99.1, terrain-park stairs 97.5, gst_v1 100, gs2dn_v2 100, gs2lo_v1 98.2, Pico 2x2 100,
chains 94.9, 4x4 89.8. For comparison, 4,000 on the old body / 2c: 93.0 % pooled. The `.out` confirms
`X2_EXCLUDE_PAIRS=hip`. `paper_eval.py` now writes `data`, `x2_exclude_pairs` and `trained_model` into every result
file, and the 4,500 file has been annotated.
**Gate: not passed.** Flat and blocks pass. Chains are 0.1 under 95 and 4x4 is 5.2 under. Stage 1 continues.
**Failing clips at 4,500** (episode records, `tools/clip_quality.py` on 2e, `eval/clip_quality/clip_quality_stage2e.json`):
- chains: 4 clips fail every time (46 of 51 failures; 99.5 % without them). They carry retarget joint jumps of 0.36-0.39
  rad per 50 Hz frame (flag `jump`), 3 of them also `limit_dwell`. Defects: excluded in 2f (plus #434, 4 of 9, same flags).
- 4x4: failures spread over 8 of 56 clips (1110: 28/29, 1136: 20/22, 1124: 18/18, 1108: 16/27, ...).
  - Clips 1110, 1136, 1108, 1109 and 1135 have no quality flag: that is the step-over itself, a skill, and those clips stay.
  - 1122 and 1106 have soles 9-10 cm inside the log on 15-21 % of frames (`sole_penetration`): defects, excluded in 2f.
  - 1124 (sole penetration 9.8 cm on 1.7 % of frames) stays.
  - Since both 4x4s are mandatory, 2f raises the 4x4 weight from 0.07 to 0.12 (ramps, at 99.1 %, go from 0.10 to 0.05).
- terrain-park stairs: no clip fails every time; 460 and 467 are partial (`limit_dwell` 16 %). They stay.
**Why the planner fails on open flat ground** (3,500 closed loop, 40 per set, planner vs oracle contacts): flat 42.5 vs
97.5, blocks 27.5 vs 90, ramps 62.5 vs 100, Pico 2x2 80 vs 100. But stairs 90 vs 95, gst 90 vs 85, chains 85 vs 92.5 and
4x4 85 vs 90. The planner holds where the terrain has features and fails where the scan is flat or smooth. Reading:
- (1) Decode is top-1 over the 0.1 m scan grid (`planner.decode`). On a flat scan all 187 points have the same height, so the
  affordance argmax has nothing to anchor on: targets snap to grid cells (up to 7 cm off) and can jump between cells or to
  the scan edge. A random history on a flat scan decodes to (0.8, 0.3), the grid corner.
- (2) The labels on flat ground were multimodal: 47 % injured and bump styles, plus jump and crouch. 2e removes them.
- (3) The tracker was co-trained only on planner contacts after eta_obs reached 1, so its reference-contact tracking slid
  (plain walk 99.3 -> 93.8 %).
**Changes for the next stage 2 (code in `code_next`, CPU tests running):**
- a. **Planner pre-trained on corrected-body hindsight labels.** tmux `acp-roll`: CPU paper eval of s1_v2 4,500 with the
  rollout dump (100 per set) -> `acp.planner label` -> `acp.planner train --arch attn`. Output in
  `runs/planner_s1v2_004500/`. Stage 2 starts from whichever of it or the kept s2_v1 planner scores better on the CPU
  closed loop (flat and blocks reported separately).
- b. **Decode test.** `ACP_DECODE_K` in `acp/planner.py`: k > 1 = affordance-weighted mean of the top-k points; default 1,
  unchanged. Being tested on CPU (tmux `acp-dec`: s1_v2 4,500 tracker with the kept planner, k = 1 and k = 4, 20 per
  set). If k = 4 closes the flat gap, stage 2 trains with it (departure: the paper uses top-k decode; k is our choice).
- c. **Reference-contact floor:** `--eta-obs-max 0.8` (20 % of the tracker's contact input stays reference contacts;
  departure: the paper ramps to 1). This keeps the tracker trained on reference contacts.
- d. **Slower ramp:** `--n-ann 3000` instead of 1,500.
- e. **Per-set gate on the ramp:** `--ramp-gate 2`. While flat's or blocks' finish EMA is more than 2 points under its
  value at the ramp start, the ramp holds (`eta_hold`, logged as `stage2/eta_hold`, saved in the checkpoint).
**Data set 2f** (`cfg/stage2f_sets.json`): 2e plus `exclude_clips` (5 chains segments, 2 4x4 clips) and the 4x4 / ramp weights
above. The arm fix is rebuilt with 2 cm hull clearance (`ARM_CLEAR=0.02`) and checked against the real meshes (running).
The loader now matches `exclude_clips` against segment names (`s<set>/<clip>#<seg>`); the first build, before that fix,
is in `_trash`.
**Return to stage 2:** at the restart that installs 2f, which is a needed restart (data change). Condition: the 5,500
live eval (old 2c sets) confirms the tracker holds on the old sets, and the planner pre-training (a) and the decode test
(b) are done. Then stage 2 from the newest s1_v2 checkpoint: corrected body, 2f, planner from (a) or kept, b-e on.
Expected: restart about 10:10 UTC, first corrected-body stage-2 pair at +250 about 11:10 UTC. Until then the interim pairs
(s1_v2 tracker + kept planner) are exported every 250 iterations. Every export now also runs a 4-per-set planner closed
loop and writes `pair_eval_40/obs_check_pair.npz` (backfilled for 4,250 and 4,500 by `acp-cpu3`).

### 16l. Interim pairs drift away from the frozen planner: return to stage 2 sooner (ACP agent, 2026-11-12 08:25 UTC)
**Eval monitor** (corrected model, our apparatus, 50 seeds): the interim pair at 4,250 scored 17/50 and at 4,500 5/50.
Walking speed fell 0.42 -> 0.342 m/s. Falls moved to the 4x4s and the landing; there were 13 stalls on the ramps.
Stage-1 training moves the tracker away from what the frozen 4,000 planner produces, so interim pairs are not a measure of
progress and every stage-1 iteration widens the gap.
**Decision:** the 5,500 eval is no longer a condition. Stage 2 restarts as soon as (a) the corrected-body planner
pre-training and (b) the decode test are done and the 2f arm check passes, expected about 09:45-10:00 UTC. The 4x4 skill
(89.8 %) keeps training in stage 2: 20 % reference-contact floor and 4x4 weight 0.12. The restart is needed anyway
(data 2f, stage change).
**Fixes:**
- `tools/paper_eval.py` runs on the CPU when no GPU is visible (the handover's CPU physics). The first `acp-roll` run had
  failed on a CUDA-only load; it was restarted at 08:20 with 80 per set.
- Interim pair cards now say the planner is byte-identical to the s2_v1 4,000 planner and was not trained with that
  tracker (`planner_iteration` 4000). The 4,250 and 4,500 cards are corrected; `export_pair.sh` does this for every
  PLANNER_FROM export.
- `pair_eval_40/obs_check_pair.npz` exists for 4,250 and 4,500 (backfilled 08:1x UTC) and is written by every new export.
  Planner closed loop, 4 per set, falls-only rule: 4,250 planner 77.5 % vs oracle 95 %; 4,500 planner 82.5 % vs oracle 95 %.

### 16m. Station stop, decode result, arm 2f result, restart script (ACP agent, 2026-11-12 09:00 UTC)
**The pair does not stop at the last station** (eval: it walks 0.4-0.7 m past button 5 and over the landing's 1.12 m drop).
- Stage 2 has no hold after arrival: an episode ends when its clip ends.
- Stage 3's stand-at-goal term needs the goal environments (stage 3, held until after the base).
- So the stop moves into stage 2 as data: `tools/hold_clips.py` writes `<set>_hold.pkl`, where every clip whose last
  frame is in double support (both feet in labelled contact) gets a 1.5 s stand appended (last frame held: zero velocity).
  Clips that got the stand: flat 103 of 351, blocks 103/351, terrain-park stairs 83/88, ramps 190/264, 4x4 44/56,
  gs2dn 30/57, gs2lo 31/82, Pico 6/10. Chains and gst end mid-climb, so none of theirs (their original files stay in 2f).
- In stage 2 the goal is the segment end, so these clips train arrive -> stop -> hold with the goal near zero. The paper
  eval's "not ended early" now includes the 1.5 s hold.
- 2f is rebuilt with the stands: 1,161 segments, 426,366 frames, arm fix and labels rebuilt; the pre-stand 2f is in
  `_trash/2026-11-12/cfg_2f_pre_hold/`. The ramp gate uses the `_hold` set names.
**Decode test (refutes reading (1) of 16k)** (s1_v2 4,500 tracker + kept planner, CPU closed loop, 20 per set):
- top-1: planner 67.5 % (flat 55, blocks 40, ramps 45, chains 30, 4x4 60) vs oracle 99 %.
- top-4 weighted: 65.5 % (flat 55, blocks 40).
- The decode is not the cause; it stays top-1 (as the paper). The gap is the planner's training: labels and data from the
  old body and old styles, and a tracker not co-trained with it. Hence (a), re-labelling on the corrected body.
**Arm fix 2f against the real meshes** (2 cm hull clearance, same 600-frame check):
- Arms within 5 mm of the same-side thigh / knee: left 0.2 %, right 0.2 % of frames; min 2.7 / 1.3 mm.
- For comparison, 2d (1 cm hulls): 7.5 / 9.0 %, min 0.1 mm; no fix: 47.7 / 40.7 %. The real meshes now clear.
**Restart script ready:** `code/ops/restart_s2v2.sh` (`PLANNER=<file>`).
- Stops s1_v2 (stop flag moved to `_trash` before anything starts) and ends the s1_v2 watchers.
- Restarts `acp-export` so it picks up `runs/s2_v2`, and installs `code_next/acp`.
- Runs a 3-iteration stage-2 check (4,096 envs) with the full flag set, then starts `s2_v2` from s1_v2's `latest.pt`:
  corrected body, 2f, planner-init, u_obs 200, u_rew 1700, n_ann 3000, eta_obs <= 0.8, ramp gate 2 points on
  flat / blocks; budget 14,000.
- Arms the live eval at the next x,500 and then every 1,000. On a failed check, s1_v2 resumes.
- Planner choice: the corrected-body pre-trained planner (`runs/planner_s1v2_004500/planner_attn.pt`, labelling and
  training on CPU now, `acp-roll`) if it beats the kept s2_v1 4,000 planner on the same CPU closed loop (flat and blocks
  first); otherwise the kept one.

### 16n. Hold at goal: what the policy gets, and the new CPU check (ACP agent, 2026-11-12 09:30 UTC)
**Goal command:** (dx, dy) = d / max(|d|, 0.3) in the body-yaw frame and dpsi = goal heading minus robot heading (`env.py`
336-340). In stages 1-2 the goal is the reference at the segment end, so the policy sees a goal near zero only at clip
ends (where the episode ends at once), in idle clips, and now in the 1.5 s stands of 2f. With nothing after the last
station, our adapter convention is goal = that station's point and heading. A nonzero dpsi there asks for a turn in
place, which stages 1-2 hardly train (the door-corner finding).
**Stations next to a drop in training:** stage 2 has stair-top landings in the terrain-park and gs2 clips. Their 2f stands
put holds on 0.3 m tops (the gst climbs to 1.02 m end mid-climb and get no stand). Stage-3 stairs-up tiles end with a
station on a landing up to 1.07 m high (level 4) with a drop on three sides.
**New CPU check** (`code_s3/tools/goal_check.py`, plus a `PLANNER=` option for stage-1 trackers). From the step the final
station is first held, the next 3 s: pelvis travel, peak horizontal speed and falls. Pass = travel < 0.20 m, speed < 0.15
m/s, no fall; the station height is reported. Rows are written to the json.
**First result** (s1_v2 5,000 + kept 4,000 planner, 4 per family, levels 0 and 4, `eval/goal_check_s1v2_005000_hold.json`):
- 18 holds checked, 13 pass, 0 falls. Median travel 1.6-6.2 cm; median peak speed 0.07-0.14 m/s (close to the 0.15
  limit); speed is what fails.
- On stair-top landings 0.87-1.07 m high (drop edge): 3 of 5 pass, none fall.
- Corridor, blocks and flat turns did not reach their final station.
**Reading:** in our own goal tiles the policy stops and holds next to a 1 m drop, with the goal fixed and nothing after it.
At button 5 the eval sees it walk 0.58 m and turn left. So the difference is more likely the eval's command after the
last station (heading, lookahead or goal source) or the landing's rail posts in the scan than the policy's stand itself.
For the eval monitor: log the (dx, dy, dpsi) fed at button 5 during the second after arrival.

### 16o. Stage 2 on the corrected body started: run s2_v2 (ACP agent, 2026-11-12, node clock 10:13 UTC)
**Stand-at-the-drop set** (eval: at button 5 the policy cannot hold a stand 0.3 m from the landing's 1.12 m drop; 60 % on
the apparatus when that station is moved inward). `tools/stand_edge.py`:
- Source: the 26 plain idle clips of the flat set that drift < 0.10 m (10 others drift more and are skipped).
- Each is lifted onto a 1.5 x 1.5 x 0.3 m landing slab (a size gs2 already uses, so no new prop slot) with its top at
  1.1 m. The front edge is 0.25-0.45 m ahead of the pelvis, facing it; two corner variants also put the left or right
  edge 0.30-0.40 m to the side.
- 78 clips: `data/clips/stand_edge_v1.pkl` + `_props.json`, set weight 0.05 in 2f (flat 0.20 -> 0.15).
- Checked on CPU: feet in labelled contact on 99 % of frames (min 95 %), pelvis 1.75-1.79 m (on top of the slab); arm fix
  clean.
**Planner choice** (CPU closed loop, s1_v2 5,000 tracker, 20 per set; oracle 99 %):
- kept s2_v1 4,000 planner: 74.5 % (flat 55, blocks 45);
- scratch planner on corrected-body labels: 63.0 % (flat 30);
- kept planner fine-tuned 15 epochs on those labels (`runs/planner_s1v2_004500/planner_attn_ft.pt`): held-out target
  error median 0.062 m (scratch 0.237); its closed-loop number is still running.
- Decision: start stage 2 with the fine-tuned planner and not wait for that number. It starts from the kept one and fits
  the corrected-body labels 4 times better, and every stage-1 iteration widened the gap.
**Restart:** `restart_s2v2.sh`, stop 10:12:51, check passed 10:13:19 (stage 2, planner start 5,500, u_obs 200, u_rew 1700,
N_ann 3000), s2_v2 started 10:13:19 from s1_v2 5,500, data 2f (defect clips out, 4x4 0.12, 2 cm arm fix, 1.5 s stands,
stand-at-drop set), eta_obs <= 0.8, ramp gate 2 points on flat / blocks. GPU idle about 1.5 min.
**First iterations:** 5,522: finish 94.9 %, flat 98.0 %, stand-at-drop 44 -> 65 % (new, rising), 71k env-steps/s,
GPU 82 % / 53 GB. eta_obs 0 until 5,700, then 3,000 iterations to 0.8.
**Exports:** every s1_v2 pair since 5,000 has `pair_eval_40/obs_check_pair.npz`. The script writes it before `DONE`, and an
export takes about 6 minutes after the folder appears, so a folder without `DONE` is still being written.
**Next:** first co-trained corrected-body pair `pair_s2_v2_005750` at about 11:10 UTC; it is still nearly tracker-only
because eta_obs is only 0.017 at 5,750. Live paper eval at 6,500 (2c sets) and 7,500 (2f). Watchers: `acp-evallive2`,
`acp-export`, `acp-backup`. `acp-trip` is ended (option (c) kept at 5,000).
**Fine-tuned planner, closed loop** (same test, 20 per set): planner contacts 83.0 % (kept 74.5 %, scratch 63.0 %; oracle
99 %); falls-only rule 92.0 %. Per set: flat 55 (falls-only 80), blocks 45 (65), chains 70, ramps 85, 4x4 90, stairs 95,
the rest 90-100. The choice holds. Flat and blocks are still the gap; under the falls-only rule most of their failures
are early ends (lag), not falls. The ramp gate in s2_v2 watches exactly these two sets.

### 16p. 5,750 -> 6,000 apparatus drop (57.0 % -> 39.5 %): diagnosis (ACP agent, 2026-11-12, node clock 13:40 UTC)
**Planner share in training:** eta_obs 0.016 at 5,750, 0.100 at 6,000, 0.228 at 6,386; eta_hold 0 (the flat / blocks gate
never fired: flat finish 97.9 -> 99.3 %). Planner contacts are switched per env and per control step with probability
eta_obs (`env.py`), so at 6,000 training still sees 90 % reference contacts. No training env runs planner-only, and no
training number measures what the apparatus pair does.
**Cross test** (CPU closed loop on the reference clips, 16 per set, tracker x planner at 5,750 / 6,000;
`eval/xcheck_5750_6000/`). Planner-driven success:
- T5750+P5750 81.3 %, T5750+P6000 83.5 %, T6000+P5750 83.0 %, T6000+P6000 81.8 %. Oracle 98.9-99.4 %.
- 4x4 81.2 % in all four; ramps 50-75 %; flat 44-56 %.
- So on reference clips neither the tracker nor the planner got worse between 5,750 and 6,000. The apparatus drop (4x4
  approach and ramp stalls) does not show where the goal is the reference's segment end. It needs waypoint goals:
  goal-tile check (logs, flat turns, stairs) at both checkpoints running, `acp-g005750` / `acp-g006000`.
**Gate change written** (`code_next`, not yet running):
- `--probe-envs N`: N envs that always use planner contacts. Their per-set finish is logged as `probe/finish/<set>`.
- `--gate-signal probe`: the ramp gate watches the probe finish on the gated sets. Base = the best EMA after 100
  updates. Hold the ramp below base - gate; back off 3 iterations per iteration below base - 2 x gate.
- Proposed sets: 4x4, ramps, flat, blocks.
**Goal-tile check, 5,750 vs 6,000** (waypoint goals, 8 per family and level, levels 0 / 4 / 9): the same at both.
- logs (4x4 at the higher levels) 24/24 and 24/24, 0 falls; stairs up 21 / 22; stairs down 15 / 14; blocks 7 / 7;
  corridor 1 / 0; flat turns 3 / 3.
- My CPU checks therefore do not reproduce the apparatus drop. It is in apparatus geometry and approach (the front 4x4 with
  the course around it, the ramps' front side), not in the tracker or planner on our tiles or clips.
**6,500 live paper eval** (s2_v2, corrected body, data 2f, reference contacts, 11,022 episodes; the alternation rule
put 6,500 on 2f, not 2c): 99.46 % pooled.
- Per set: flat 99.4, blocks 98.8, chains 100, terrain-park stairs 98.2, gst 100, gs2dn 100, gs2lo 99.8, ramps 99.1,
  4x4 98.7, Pico 100, stand-at-drop 100.
- **Base gate passed on 2f.**
**Decision, done:** the ramp is not held at the 5,750 share. The cross test shows neither component got worse with the
share, and holding at 1.7 % would leave the tracker untrained on its own planner. Instead s2_v2 was restarted at 6,500
(13:49:36-13:50:01 UTC, `code/ops/restart_probe.sh`) with 720 planner-only probe envs (2 %). The ramp gate watches their
finish on 4x4, ramps, flat and blocks: hold under best - 2 points, back off under best - 4.
- First probe numbers (planner-only finish, 6,500-6,548): flat 82.3 %, blocks 82.1, chains 86.7, terrain-park stairs
  92.9, gst 95.8, gs2dn 95.8, gs2lo 99.1, ramps 86.9, 4x4 93.7, Pico 99.3, stand-at-drop 100.
- eta_obs 0.28, no hold so far.
- This is the first training number that measures what the eval pair does.
**Stairs approach** (the robot arrives still turning and hits the first riser): stage-2 data has no turn right before a
flight. The stage-3 stairs tiles have exactly that (a heading turn of up to +-70-90 degrees after the waypoint before
the flight), so it is the first reason for stage 3, which is now open since the base gate passed. Until then, an eval
waypoint square at the foot of the flight (a station with the heading of the flight) would be the cheap test that it
is the approach.

### 16q. The eval's three traced gaps at 6,000: counts, why, fixes (ACP agent, 2026-11-12, node clock 14:40 UTC)
**Counted in the training data** (2f, reference pose = robot pose; `/tmp/situ.py`, method in the script header). Three
situations:
- (1) goal (segment end) under 0.5 m away with |dpsi| > 2.5: 2,219 of 454,116 frames (0.5 %), in 24 segments: ramps 11,
  flat 6, blocks 6, 4x4 1. Goal tiles: flat_turns has turn-in-place and backward legs, but stage 3 is not running.
- (2) a ~0.1 m step down started from a stand (still 0.5 s, then the root drops 0.06-0.16 m within 1.5 s): 45 segments
  (stairs 22, chains 12, ramps 11).
- (3) a heading change > 0.5 rad within the 0.5 m before the first rise of a flight: 0 on every stairs set (11 only on
  ramps). The diagonal approach that turns onto a flight next to the first riser is not in the training data at all.
**Why 6,000 lost what 5,750 had:** the planner share rose 0.016 -> 0.100. More planner-driven steps feed the online planner's
hindsight buffer, and a skill seen in 0.5 % of frames (1) can be overwritten in 250 iterations. Gap (1) is the planner
forgetting; gap (2) is the tracker drifting on a rare start. The probe envs (2 %, planner-only, reference clips) would
not have caught either: they sample the same rare situations, and a per-set finish rate averages them away. They guard
against general planner-driven decline, not these.
**Fixes:**
- (1) Data: `turn_place_v1` (turn > 2.0 rad with < 0.6 m travel, from the kept flat styles). Only 4 clips / 18
  segments / 162 frames qualify, so data alone cannot fix it.
  - The training fix is stage 3's flat_turns goal tiles: turn in place, backward and sideways with goals behind.
  - Adapter rule, eval side: the current command is a poor way to ask for a turn. A target 0.36 m straight behind, plus
    the 0.3 m lookahead, becomes a full-length goal (-0.95, +-0.3) with dpsi near pi, which training has almost never
    shown.
  - The rule I specify for ACP pairs: when the bearing to the next waypoint exceeds 1.5 rad and it is within 1.0 m, no
    lookahead; command (dx, dy) = 0.3 m in the direction of the bearing clipped to +-1.0 rad, with dpsi clipped to
    +-1.0 rad, until the bearing is under 1.0 rad; then the normal rule. That is a turning walk, which the data has in
    every turn clip.
- (2) Data 2g: 1.0 s stands prepended to every clip that starts in double support in the stairs, gs2, ramps, 4x4, chains
  and gst sets (`*_hs.pkl`; ramps 253 of 264, stairs 83/88, gst 86/113, chains 34/96, 4x4 44/56, gs2lo 41/82,
  gs2dn 7/57). Count (2) rises from 45 to 49 segments, but every one of those clips now starts from a standstill
  before its first step.
- (3) Not trainable in stage 2 (count 0). Stage 3's stairs tiles have a +-70-90 degree heading turn after the waypoint
  before the flight. The eval's foot-station test: 13 of the 15 that stood square at the foot climbed the whole flight.
  So it is the approach, and stage 3 (or an eval waypoint square at the foot) is the fix.
**Done:** s2_v2 restarted on data 2g (`cfg/stage2g_sets.json`, `tools/mk_stage2g.py`, arm fix and labels rebuilt; flat weight
0.15 -> 0.12, turn_place 0.03). Stop 14:38:42 at latest 6,698; back 14:39:08; 39 iterations lost. Probe gate unchanged
(set names `_hs` for 4x4 and ramps).
**Regression set (eval's offer c): yes.** Freeze `pair_s2_v2_005750` (57.0 %) as the reference pair. Rerun on each new
pair: the turn-around seeds (1001, 1005, 1036), the ramps-trough seeds (1019, 1022) and the stairs seeds. Gaps (1) and
(3) will not move until stage 3 or the adapter rule; (2) should move with 2g from about 6,950 on.
**Next decision:** stage 3 is the training answer for (1) and (3), and the base gate is passed (16p). Stage-3 code
(`code_s3`) predates the eta ceiling, ramp gate and probe envs; merging those is the next CPU job. Switch after the
6,500 / 6,750 apparatus results.

### 16r. Three falling apparatus scores: planner frozen at 5,750 (ACP agent, 2026-11-12, node clock 15:08 UTC)
**Apparatus, strict rule, 200 seeds:** 5,750 57.0 % -> 6,000 39.5 % -> 6,250 36.0 %. At 6,250, 52 of 74 stalls are the
turn-around after button 1. The stairs improved: on the 45 stairs seeds 6,250 has 14 successes and 4 stair falls.
**a) Decision, done: freeze the planner at 5,750, keep training the tracker.**
- `--freeze-planner <ckpt>` (`code_next/acp/train.py`): loads that checkpoint's planner and skips the online planner updates.
  The hindsight buffer can no longer overwrite rare skills, and the tracker keeps adapting to one fixed planner as the
  ramp continues (ceiling 0.8, probe gate on).
- Source: `eval/pair_s2_v2_005750/tracker_ckpt_005750.pt` (the run's 5,750 checkpoint was pruned to `_trash` by the
  retention rule; the pair folder holds the full copy).
- Restart `code/ops/restart_freeze.sh`: stop 15:07:16 at latest 6,750, check passed ("planner frozen" in the log),
  back 15:07:38; 0 iterations lost.
- Why this and not a cap or a replay share: the eval traced gap 1 to the planner (the 6,000 planner proposes no step;
  5,750 turns). A cap would leave the tracker unadapted to its planner, and a replay share needs counts of situations
  that are 0.5 % or absent (16q).
- **One check:** the turn-around seeds (1001, 1005, 1036) on the first frozen pair, 7,000 (about 15:55 UTC). Expected: the
  5,750 behaviour (turns and walks on), with the tracker's later gains (stairs) kept.
**b) Eval cores:** first the swap test (planner 5,750 + tracker 6,250, and the reverse), `turnwalk` and the three
regression seed sets. 6,500 and 6,750 are from the forgetting regime, so only the regression seed sets for them, not
200 + 200. The next full 200 + 200 is the first frozen pair at 7,000.
**c) Stage-3 rule, in numbers.** "Stable" means the apparatus.
- Stage 3 starts when two consecutive frozen-planner pairs (250 apart) each score at least 57 % strict on our apparatus
  (the 5,750 reference). The paper eval with reference contacts must stay >= 98 % on flat and blocks and >= 95 % on
  the rest, and planner-only probe finish must be >= 90 % on flat, blocks, 4x4 and ramps.
- Second route to stage 3: if two consecutive frozen pairs score under 57 % and the regression sets show that >= 70 % of
  their failures are the turn-around (gap 1) and stair approach (gap 3) — both absent from stage-2 data, so not
  trainable there — stage 3 is the fix.
- If neither holds after 7,500: stop and diagnose before anything else.

### 16s. Swap test: the turn-around stall follows the tracker; fallback prepared (ACP agent, 2026-11-12, node clock 15:46 UTC)
**Swap TEST** (eval, 50 seeds each). Successes / turn-around stalls:
- pair 5,750: 21 / 0
- pair 6,250: 18 / 8
- planner 5,750 + tracker 6,250: 27 / 4
- planner 6,250 + tracker 5,750: 27 / 0
The stall follows the tracker. The planner's proposal depends on where the tracker puts the feet. The stairs improve with
either newer half.
**Stand pose after button 1** (my trace with the eval harness, corrected model, seeds 1001 and 1005;
`acp_pose_trace.py` and `pose_*.npz` in the coordinator's shared scratchpad):
- Stance width at the stand: 5,750 0.199 / 0.250 m; 6,250 0.199 / 0.165 m. Foot yaw within +-0.4 rad for both. No
  systematic stance difference.
- What differs is the planner's occupancy for one foot at that stand: 6,250 seed 1001 has 0.34 on the left foot (under the
  0.5 validity line), so no left step is proposed. Yaw change in the first 3 s: 0.48 rad (6,250) against 2.89 (5,750).
- All four traces finished in my 90 s runs (6,250 seed 1001 in 88.8 s, slowly).
- Reading: the newer tracker's pose and history at the stand push the planner's foot occupancy across 0.5 for one foot. It is
  a co-adaptation edge, not a clearly different stance.
**b) Decision rule for the 7,000 check** (read against planner 5,750 + tracker 6,250: 27/50, 4 stalls):
- If the turn-around seeds (1001, 1005, 1036) or the 50-seed set show turn-around stalls at 7,000, the tracker restarts
  from 5,750 as well: `code/ops/restart_t5750.sh` (prepared, not run). It starts a new run s2_v3 from
  `pair_s2_v2_005750` (tracker and planner), planner frozen at that checkpoint, data 2g, probe gate, same ramp, and adds
  s2_v3 to the export watcher. The stairs gain of the newer halves is the cost; 2g's prepended stands are meant to cover
  the step-down part of it.
- If 7,000 has no turn-around stalls: s2_v2 continues frozen.
**c) Hold-out gate:** yes. The CPU checks cannot reproduce this (16p), so the gate on tracker checkpoints is the eval's
three turn-around seeds plus the two ramps-trough seeds, run on every export before its 200 + 200. A pair that stalls
on any of the three turn-around seeds is not kept as a candidate.

### 16t. Decisions received; extended map in stage 3 (ACP agent, 2026-11-12)
- **Stage-3 second route allowed** (coordinator's recommendation under the user's rule of 2026-11-12, on the user's todo
  list, can be overturned): two consecutive frozen-planner pairs under 57 % strict, with >= 70 % of their failures at the
  turn-around and the stair approach. "Neither route by 7,500: stop and diagnose" stays.
- **User confirmed:** strict stand with corrective steps allowed (the 1 s counts after the step); button 5 stays where it is.
- **Extended map** (user: "i was expecting our extended terrain map be part of these training runs"). For ACP it goes into
  stage 3, and stage 3 does not start on re-arranged props alone without a date for the real tiles.
  - **What exists:** `data/terrains/extended_v1/`. 330 tiles of 8 x 8 m, 9 types, 10 levels, 650 checked routes; per type a
    `.npz` with rasters, boxes and meshes, and a `.json` with spawns, routes and targets.
  - **v2 plan, as they are:** the tile rasters and their boxes, with the routes as goal-env waypoints (stations as given).
    - Types first: `single_beams`, `beam_fields` (the 4x4 skills), `stairs` (including the turn onto a flight, gap 3),
      `log_corridors`, `blocks`, `street_front`, `homes`, `tight_spots`. Then `wedges`: their meshes become raster
      heights (they are ramps).
    - Routes with a waypoint behind the robot train the turn-around (gap 1).
  - **What blocks it beyond the prop-box version:**
    - (1) One combined height field for 330 tiles of 8 x 8 m in MuJoCo Warp, beside the existing ramp raster. The CPU loop
      already prints height-field collision overflows, so this needs a GPU memory and speed test at a restart.
    - (2) Box slots: every distinct box size is a slot in every world, and the tiles have 60-346 distinct sizes per type.
      They need quantising to a size library (5 cm steps, capped count), checked against the slot limit.
    - (3) Meshes: our props are boxes only, so wedge meshes go to the raster.
    - (4) Walls and posts collide but must not end episodes near them (SSR's lesson in `SHARED_LESSONS.md`: an
      obstacle-margin termination taught its policy to stand still beside walls). Ours ends only on falls and tilt, and
      that stays.
  - **Date:** v2 tile import, size library and CPU checks (tiles load, slots under the limit, spawn and route clearances,
    one CPU closed loop per type) by 2026-11-13 12:00 UTC. The GPU height-field test runs inside the stage-3 switch
    restart. If the switch comes earlier, stage 3 starts with the v1 tiles and v2 goes in at the next needed restart,
    at the latest 24 h after the switch.
  - Copied apparatus parts stay single, re-arranged parts, listed per tile in the import.

### 16u. Export failure since 2g, fixed (ACP agent, 2026-11-12, node clock 16:31 UTC)
- Every hand-over since data 2g failed. `tools/handover.py` looked up the stair set as "gst_v1", which 2g renamed
  `gst_v1_hs`, so 6,750 and 7,000 had no `actor.npz` and no `DONE`, and the eval's 7,000 gate waited about 25 minutes.
- Fix: the set is now found by prefix (`gst_v1*`), with the first set as fallback. No other hard-coded set names remain in
  tools/ or acp/.
- `pair_s2_v2_007000` was redone by hand on CPU and is DONE at 16:30:55. It matches `pair_s2_v2_006500`: same files,
  actor shapes, gains, `trained_model` and obs_check shapes, plus `pair_eval_40/obs_check_pair.npz`. Its planner ONNX is
  byte-identical to the 5,750 one (the freeze holds).
- 6,750 is redone by the restarted watcher.
- `ops/export_every.sh` now writes each attempt's result to `/mnt/filesystem-g4/status/acp_export.json` (done, running
  or FAILED, with the reason). A failed folder gets `EXPORT_FAILED` and is skipped until the file is cleared. The watcher
  also covers `runs/s2_v3`.
- Lessons:
  - A data-set rename breaks every tool that names a set; look sets up by prefix or from the config.
  - `pkill -f <script>` from an ssh command line also kills that command (my first rerun killed itself).

### 16v. 7,000 gate passed; fallback skipped (2026-11-12, eval finished 16:32:42 node clock)
`pair_s2_v2_007000` (planner byte-identical to the frozen 5,750 one): turn-around seeds 1001 / 1005 / 1036 all succeed,
22/22 waypoints, standing at button 5. For comparison: 5,750 two successes and a stairs fall; 6,250 a ramps fall, a
success and a door-side fall; 6,000 three stalls. No turn-around stall, so by 16s the tracker-from-5,750 fallback is
skipped and s2_v2 continues frozen. The 50-seed set (against the swap line, 27/50) is still to come from the eval.
Next CPU work: the v2 import of the extended map (16t).

### 16w. Extended map v2 import: built, first CPU check has an open defect (ACP agent, 2026-11-12, node clock ~17:30 UTC)
**Built:**
- `code/tools/import_extended.py` (run with 1 tile per type and level) writes `data/clips/hmap_ext_v2_40mm.npz` and
  `data/goals/goal_tiles_v2.json`.
- One height raster with the user's tiles as they are: each tile's floor raster plus every box and mesh that is not in it
  (walls, posts, rails, chairs, tables, raised 4x4s, door leaves, stones). Door headers are left out so doorways stay open.
  2 cm -> 4 cm by max.
- 9 types x 10 levels (per level the first variant that has routes), 197 routes as goal-env routes (final waypoint = station);
  the ramp raster is pasted in unchanged. Raster 2,260 x 1,819 = 4.1 M cells, x 29.7-120.0, y -0.7-72.0 m, max 4.54 m.
- No new prop slots: the v1 tiles stay inside the 2g slot library (`goals.check_slots`: 0 over).
- Waypoint and spawn heights are taken from the raster: 0 of 605 waypoints differ by more than 6 cm. The generator's z
  was off on wedge slopes and over non-raster beams.
- Config `cfg/stage3v2_sets.json` (in code and code_s3): 2g reference sets + goal tiles v1 + v2 + the new raster.
- `goal_check.py` takes `ACP_NCONMAX`.
**First CPU closed loop** (7,000 pair, 2 per family, levels 0 / 4 / 9; `eval/goal_check_ext_v2_007000*.json`):
- v1 families as before (stairs up / down and logs 2/2 at every level).
- Every extended type falls 2/2 at levels 0 and 9 within about a second, with 0 waypoints. At level 4 there are no falls;
  stairs L4 2/2 and tight_spots L4 1/2 succeed.
- Same result with nconmax 200 (one overflow line), so the falls are not the contact buffer.
- A 4-env diagnostic run of the same tiles did NOT reproduce it: single_beams L0 stood for 750 steps, while blocks L0
  fell at 0.64 s with a clear spawn. So it depends on the env count or ordering and is not plainly the terrain.
- Open: tile/world assignment in the goal env with many families, or a CPU-physics artefact with the large hfield.
  Next: replay goal_check's exact env order with a per-env fall log, then the GPU test at the stage-3 switch.
**Also seen:** the running s2_v2 log has 5,228 "height field collision overflow (>= 50 contacts per pair)" lines on the
existing ramp raster (no broadphase overflow). Contacts beyond 50 per geom pair are dropped there today. Worth a look
before the larger raster goes in (coarser raster, or an hfield on the feet only).
**Date:** the 2026-11-13 12:00 UTC target for the v2 CPU checks stands. The open defect above is the first item.

### 16x. 7,000 on the 50-seed set: rule revised, continue; next loss = 4x4 falls (ACP agent, 2026-11-12)
**Eval** (seeds 1000-1049, strict + retry, planner frozen): 7,000 26/50 against pair 5,750 22/50, swap (planner 5,750 +
tracker 6,250) 27/50 and pair 6,250 18/50.
- Failures: 18 falls (4x4s 13, ramps 2, stairs 3) and 6 stalls (4x4s 3, ramps 3). Turn-around stalls 1 (6,250 had 8,
  the swap 4).
- Stair falls 3 (5,750: 19). Button 5 held 26/26. Gate seeds: turn-around 3/3, ramp trough 1/2.
**Decision: do not fire the 5,750 restart; the rule is revised.** As written, one turn-around stall in the 50 would trip it.
But 7,000 beats 5,750 on the same seeds (26 vs 22), has a sixth of its stair falls (3 vs 19) and 1 turn-around stall
against 8 at 6,250. A restart from 5,750 would give all of that up for one stall.
- New rule: the tracker restarts from 5,750 only if a frozen pair stalls on any of the three turn-around gate seeds, OR
  has >= 4 turn-around stalls in the 50-seed set (the swap line), OR scores below 5,750's 22/50 there.
- 7,000 meets none, so s2_v2 continues.
**Next loss to attack: falls at the 4x4s** (13 of 24 failures at 7,000; ramp trough 1022 also fell going back over the
front 4x4). Asked the eval for the traces split by phase: approach, over the front 4x4, going back over it, the slot.
2f/2g already raised the 4x4 weight to 0.12, and the paper eval has 4x4 at 98.7 % with reference contacts. So the gap is
likely the apparatus approach geometry or the planner's footholds at the log, as at the front-4x4 stall. The traces decide.
**Dropped height-field contacts in s2_v2:**
- 5,407 "height field collision overflow (>= 50 contacts per pair)" lines over about 1,980 logged iterations: about 2.7 per
  iteration. Warp prints one line per kernel launch in which any of the 36,000 worlds overflowed, and an iteration is 24
  steps, so about 11 % of steps have at least one world over 50 contacts with the ramp raster.
- 50 contacts per pair needs a large geom lying on many 4 cm cells (a body hull of a fallen or kneeling robot). The sole
  spheres touch a few cells each.
- So the overflow is most likely in falling robots on the ramp raster, which end within a second, not in normal stance.
- I see no link to the apparatus ramp falls and stalls: those come from the eval's own MuJoCo model, which has no height
  field there and showed the 11 cm trough step-down (16q).
- Not verified per world; a per-world contact count on the ramp set (CPU) would confirm it.

### 16y. 7,000 at 200 trials; jammed-foot check in the training physics (ACP agent, 2026-11-12, node clock ~18:00 UTC)
**Apparatus** (eval, 200, strict + retry): 7,000 125/200 = 62.5 % (55.6-68.9), against the 5,750 reference 57.0 %; better
but not separable at 200.
- Failures: falls / stalls at the 4x4s 37 / 14, ramps 4 / 8, stairs 9 / 2 (5,750: 45 stair falls).
- Turn-around stalls 11 in 200 (2.75 per 50, under the rule's 4). Button 5 held 125 of 126.
- This is the first frozen pair at or above 57 % (stage-3 rule: a second consecutive one is needed; 7,250 next).
- 4x4 losses are not acted on until the eval's own floor-physics artefact is settled (eval: 2 cm height-field floor, soft
  contact, spheres jam sideways in cells with 4-20 kN).
**Same check in our training physics** (`code/tools/sole_contacts.py`).
Setup: CPU mujoco_warp, the training model (corrected body, data 2g), the 7,000 tracker with reference contacts,
deterministic, 6 clips per set, 250 control steps = 1,000 physics steps each. Model facts: 24 sole spheres + 2 terrain
foot boxes, geom solref 0.02 s / damping ratio 1, dt 5 ms, ramp raster cells 4 cm.

| where | contacts | depth p50 / p99 / max (mm) | normal force p50 / p99 / max (N) | near-horizontal normals |
|---|---|---|---|---|
| flat floor (plane), spheres | 88,826 | 0.3 / 1.9 / 10 | 22 / 85 / 362 | 0 % |
| box props (gs2dn stairs), spheres | 44,244 | 0.4 / 7.9 / 13.9 | 43 / 368 / 756 | 0.5 % (max 489 N) |
| ramp raster (height field), spheres | 43,812 | 0.5 / 8.6 / 29.6 | 20 / 265 / 18,650 | 3.2 % (p99 450 N, max 4,845 N) |
| ramp raster, terrain foot boxes | 420 | 0.8 / 86 / 254 | 29 / 9,180 / 18,650 | - |
| box props, terrain foot boxes | 117 | 2.2 / 8.4 / 8.5 | 34 / 183 / 184 | - |

**Reading:**
- On the flat plane and on box props the foot physics is clean: mm penetration, forces up to about 1.8x body weight.
- On the height field the sole spheres mostly behave, but a few percent jam sideways in cells. The terrain foot box (meant to
  catch edges) sinks up to 25 cm into the field with 9-19 kN at the 99th percentile: the same artefact the eval found, worse
  with a box than with spheres.
- So yes: the tracker has been learning partly on jammed-foot physics, but only on the ramp raster set (weight 0.05 in
  2g, 0.10 before) and in a small share of its steps there (420 box contacts in 6,000 env-physics-steps).
- The flat, block, stairs, 4x4, chains, gst and Pico sets are box props or plane, and clean.
- The 5,407 overflow lines fit this: the deep box-in-field contacts produce the >= 50 contacts per pair.
- **Consequence for the extended map v2:** it is all height field, so this artefact would be in every tile. It may also be
  the cause of the unexplained v2 falls (16w).
**Fix to test on CPU before any training change** (not yet applied):
- (A) take the terrain foot boxes out of height-field collisions (spheres only on the field; the boxes still meet props);
- (B) stiffer sole contact on the field (solref 0.005-0.01 s);
- (C) 2 cm field cells (cost: 4x cells).
- Measure with `sole_contacts.py`, then rerun the v2 goal check. The training change goes in with the stage-3 switch or the
  next needed restart.

### 16z. The 4x4-section failures at 7,000 (eval traces, provisional while the floor jam is present)
Eval: 16 failures in the 4x4 section of 50.
- 8: east drift into the ramp cells during or after the turn-around at button 1.
- 3: falls crossing the front 4x4 going back west; the right foot hits the ramp-end face, and seed 1022 crosses 143
  degrees off square.
- 4: at the door side after the crossing.
- 1: trunk on the wall before button 1.
**What trains "turn in place, walk back the way you came, cross a beam square":**
- Stage 2: next to nothing. Turn-in-place frames are 0.5 % of the data (`turn_place_v1`: 4 clips). The 4x4 clips approach
  the log already walking, mostly square, and never right after a turn in place.
- Stage 3 v1: the logs family crosses a log after a waypoint with a heading turn of up to +-70-90 degrees, and flat_turns
  trains turn in place and walking backward. Neither has a station, then a 180-degree turn, then a crossing back over the
  same log.
- Extended v2: beam routes cross beams, but not after a turn-around.
- Missing piece, cheap in `acp/goals.py`: a "logs_back" family. Walk over the log to a station 0.4-0.8 m past it, turn
  135-180 degrees, cross back square, with the station on the side the apparatus has (a wall or wedge beside it). Planned
  for the stage-3 tile set.
**East drift, planner or tracker:**
- The goal is right: after button 1 the target is west, behind the robot.
- The planner proposes steps where the robot already is, with high occupancy: stepping in place, not toward the beam.
- The tracker turns by stepping in place, with the body yawing and slipping sideways, and drifts east because it turns in
  a wide arc, not in place.
- I read it as the turn-in-place execution of the pair (tracker plus a planner trained on that tracker's own hindsight
  steps), both from the same data gap, not the goal handling.
- The adapter rule for a target behind (16q: no lookahead, 0.3 m goal along the bearing clipped to +-1.0 rad) would turn
  it into a controlled turning walk; it is worth testing on these 8 seeds.
- Training fix: stage 3 (flat_turns and logs_back).

### 17a. 7,250 gate failure, tracker drift, checkpoint selection, stage-3 rule in legs (new ACP agent, 2026-11-12, node clock 18:10 UTC)
Handover: the previous agent's context passed the limit; this agent started from 16p-16z, USER_RULES 2026-11-12, FINDINGS B8/D.
All apparatus numbers below are provisional (B8 floor jam; the eval's corrected-floor comparison runs until about 21:00).

**Inputs.** Legs (50 each) at 7,000: 49 / 39 / 46 / 50 / 43, product 0.605; full route 125/200 = 62.5 %. 7,250 hold-out gate:
turn-around 1001 fell (back 4x4, 8/22), 1005 success, 1036 fell (door side, 7/22); ramp trough 1019, 1022 success. 7,250 legs
(partial at 18:05): leg 1 43/44, leg 2 38/46 (7 falls), leg 3 44/50, leg 4 49/50, leg 5 not started.

**a) What 7,250 means, decision.**
- The 16x rule (restart from 5,750) does not fire: it names stalls on gate seeds, and 7,250 fell, it did not stall. The
  intent of the gate (16s: "not kept as a candidate") does apply: 7,250 is not a candidate until its legs say otherwise.
- Two falls on three single seeds are weak: one re-rolled seed flips the count. 7,250's partial leg 2 (38 of 46 done, 8
  failures) is at 7,000's rate (39/50, 11 failures), not worse. So no evidence yet that the tracker regressed.
- Decision: **s2_v2 keeps training; no restart now.** Evidence needed: 7,250's full legs (about 18:20) and 7,500's gate and
  legs (exported 17:43; about 19:00). Restart rule, written now: if 7,250 AND 7,500 both score below 7,000 by legs (product
  < 0.55, or leg 2 < 39/50 and leg 5 < 43/50), restart the tracker from 7,000 with `code/ops/restart_t7000.sh` (prepared,
  not run; run s2_v3, planner frozen at 5,750, gate hold-only with a fresh base, see b). Otherwise continue.

**b) Is the tracker drifting with the planner frozen?** (logs on the node, `runs/s2_v2/metrics.jsonl`, +-50 iterations)
- Probe (planner-only) finish EMA: 4x4 0.958 (6,750) -> 0.939 (7,000) -> 0.909 (7,250) -> 0.922 (7,500); ramps 0.949 ->
  0.932 -> 0.891 -> 0.903; flat 0.85-0.87 and blocks 0.84-0.87 flat. About 11 4x4 and 5 ramp probe episodes per
  iteration, so the 7,250 dip is about 3-4 standard errors: real, small, partly recovered by 7,500.
- Same sets with reference contacts: 4x4 0.984-0.986, ramps 0.983-0.985, flat 0.988-0.991 throughout. Live paper eval:
  7,500 on 2c 96.2 % pooled (flat 92.7, blocks 92.7, gst 84.4) against 96.5 % for s1_v2 5,500 on the same 2c; 6,500 on 2f
  99.5 %. No 7,000 / 7,250 live eval (alternation is every 1,000). So the tracker's reference tracking does not drift; what
  moves is how it works with the frozen planner (the co-adaptation edge of 16s), worst around 7,250, which matches the gate.
- **Defect found in the probe gate:** eta_obs has fallen 0.35 (7,000) -> 0.26 (7,250) -> 0.17 (7,500) -> 0.15 (7,580);
  eta_hold 265 -> 1,293 (about +2 per iteration: back-off). Gate bases in the checkpoint: flat 0.879, blocks 0.869, the
  ramp and 4x4 sets 0.954 / 0.962. The base is the MAX of a noisy EMA (5-11 episodes per iteration), so it sits about 2 points
  above the set's real level and the gate backs off on noise as well as on real dips. With the planner frozen, backing off
  also removes the cause it was built for (the hindsight buffer, 16q) and moves the tracker away from the contacts the
  eval uses, and eta_rew falls with it. Fix written (`code/acp/train.py` = `code_next`, default behaviour unchanged):
  `--gate-backoff N` (eta_hold per iteration under base - 2 gate; 1 = hold only) and `--gate-rebase` (drop the base carried in
  the checkpoint). It goes in at the next restart (restart_t7000.sh uses both); I do not restart for it alone now, because
  the 7,250 / 7,500 evidence arrives within the hour and two restarts would cost more than an hour at eta 0.15.
- **Selection rule (replaces "train on and hope"):** the candidate pair (the champion) is the frozen-planner pair with the
  highest leg product that passes the hold-out gate; within 0.05 of product, the one with fewer legs under 45/50 wins, then
  the earlier one. The champion is what the eval scores on 200, what goes to stage 3, and what a restart starts from.
  Newest is not best (7,250 gate). Champion now: **7,000** (only pair with legs). Pair folders keep the tracker copy
  (`eval/pair_s2_v2_<it>/tracker_ckpt_<it>.pt`), so retention pruning in `runs/` does not lose a champion.

**c) Stage-3 rule in leg scores** (replaces 16r c; the user's order: full route only after all five legs pass 45/50).
- Base condition unchanged: paper eval with reference contacts >= 98 % flat / blocks and >= 95 % the rest on the cleaned set
  (2g), with the old 2c beside it; probe finish >= 90 % on 4x4 and ramps (flat / blocks probe stays ~85 %: the planner-only
  flat number is the planner's limit, D2, and is reported, not gated).
- Route 1: two consecutive frozen-planner pairs each pass all five legs at >= 45/50. Their full route (200) is then run, and
  must be >= 57 % (the 5,750 reference) as a cross-check.
- Route 2 (untrainable gaps): two consecutive frozen-planner pairs with legs 1, 3, 4 at >= 45/50, and >= 70 % of the failures
  in legs 2 and 5 at the turn-around (gap 1) or the stair approach (gap 3), which stage-2 data lacks (16q). Stage 3 then starts
  from the champion.
- Why legs, not the full-route 57 %: the full route multiplies five legs, so it cannot say which section stops it, and the
  user ordered legs first. 57 % stays as the cross-check once a full route is run.
- 7,000 by route 2: legs 1/3/4 49, 46, 50; leg 2 failures 11 of 11 at the turn-around (waypoint 0-1 of 5, front 4x4); leg 5
  failures 4 of 7 at the stair approach (d) plus 2 stalls at 0/7 leaving button 4 and 1 at the landing: 15 of 18 = 83 %.
  7,000 qualifies; the second pair decides (7,250 if its legs qualify and it is the champion or within 0.05, else 7,500).
  Stage-3 code still needs the merges of 16q (eta ceiling, probe gate, freeze, now gate flags) and the logs_back family (16z).

**d) Leg 5 (button 4 -> button 5) at 7,000:** 4 falls (1021, 1024, 1028, 1037), all with the "foot of the stairs" waypoint
reached (4 of 7), heading "up the stairs", 16-18.5 s, pelvis 0.58-0.74 m (on the first treads), 3-10 stair side hits, nearest
box 1-12 cm, adapter yaw command 0.87-1.0 rad (saturated) at the fall. That is gap 3: the robot turns onto the flight next to
the first riser and hits the riser while still turning; the same failure as 29 of 45 stair failures at 5,750. Not the climb.
2 stalls (1016, 1040) at 0 of 7 just after the spawn at button 4 ("ramps, front side", commanded 0.7 m/s forward with 0.6 rad
yaw): a start-from-stand stall like the turn-around. 1 stall at the landing (1032) with 27 stair hits and a 13 kN peak
(possible B8 artefact). Trace wanted from the eval: for the 4 falls, per control step from waypoint 3 to the fall, heading
against the flight axis, the adapter command, the planner's proposed footholds (occupancy, distance to the riser edge), the
foot that hits the riser (toe or side) and the B8 flags; and leg 5 with a square station at the stair foot (the 16q test), 50
trials, to confirm the approach. For 1016 / 1040: the planner's occupancy per foot at the spawn.

**Done this turn:** gate flags in `code/acp/train.py` and `code_next/acp/train.py` (backup in `_trash/train.py.pre_gatebackoff.*`);
`code/ops/restart_t7000.sh` prepared (bash -n checked, not run). No restart, nothing stopped.

**What went wrong since the last report:** (1) 7,250 failed the hold-out gate with the planner frozen: the tracker alone
moves the pair; addressed by the selection rule. (2) The probe gate has backed eta_obs from 0.35 to 0.15 in 580 iterations,
partly on noise (max-of-EMA base): fix written, goes in at the next restart. (3) Leg 5 loses to the stair approach as at
5,750: stage 3 (stairs tiles with the turn before a flight).

### 17b. 7,500 worse: tracker restarted from champion 7,000 with the gate fix (ACP agent, 2026-11-12, node clock 18:54 UTC)
**Evidence** (eval, current floor, provisional). Legs, failures out of 250 trials: 7,000 49/39/46/50/43 (23 failures, product 0.605); 7,250
49/41/44/49/44 (23, 0.610); 7,500 45/36/41/49/43 (36, 0.448). Gate: 7,000 5/5 turn-around + 1/2 ramp, 7,250 1/3 + 2/2, 7,500
1/3 + 0/2.
**What I treat as real:** the audit's identical-physics control flipped 21 of 50 outcomes, with a net difference of 5 of 50, so one
leg differing by 3-5 trials is noise. Across all five legs (250 trials, about 23 failures) the binomial SD of the failure count
is about 4.7 per pair, about 6.7 for a difference. 7,500's +13 failures is about 2 SD, and it points the same way as three other signals:
the gate (1/5 against 6/7 at 7,000), the probe finish dip on the 4x4 and ramp sets from 7,250 on, and eta_obs falling 0.35 -> 0.15. 7,250 against 7,000
(0 difference) is a tie. Rule from now on: a pair-to-pair difference counts only at >= 10 failures over the five legs, or
when legs and gate and probe agree.
**Decision, done:** my 17a rule asked for both pairs worse; I fired it on 7,500 alone because the gate defect (17a b) keeps
backing eta off every iteration, so waiting for 7,750 / 8,000 would score more pairs from the same failure mode. Restart
`code/ops/restart_t7000.sh` launched 18:53:51 (tmux `acp-restart`): s2_v2 stops at about 7,907, new run `runs/s2_v3` from
tracker 7,000, planner frozen at 5,750, data 2g, `--gate-backoff 1 --gate-rebase`. s2_v2's checkpoints stay as they are
(7,750 is exported and can still be scored; it is not a candidate unless its legs beat 7,000 by >= 10 failures).
Champion stays 7,000 (tie with 7,250; 7,250 failed the gate).
**Stair-square test** (eval): 38/50 with a 1 s squaring stop at the stair foot against 43/50 without it, no control: no help. The
route stays as it is (user). The stair approach has to be trained.
**New assets (user's request):**
- `tp_stairs_all_v1` (44 Pico stair takes, 6 turn-then-climb clips): enters as data **2h**, at a second restart once built.
  Build on CPU (same pipeline as 2g: `tools/mk_stage2g.py` shift / prepended stand / arm fix / labels), drop the 7 takes that fail the
  2 cm contact check (also 114308 and 114330 out of the existing stairs set), then a CPU closed loop with the 7,000 tracker on reference
  contacts per clip (our replay check in place of the SuperMimic rollouts), keeping clips that it finishes or nearly finishes. Weight:
  the stairs share of 2g split, turn-then-climb clips upweighted. Target: built and checked by 2026-11-13 06:00 UTC, restart then.
- `extended_v1_2_stairs_inset` (60 tiles, 700 routes with `kind`): replaces the extended_v1 stair tiles in the v2 import for
  stage 3 (16w); routes of every kind kept, sampled evenly by kind.
**Wall contact:** nothing in ACP ends an episode on a wall touch. Terminations are base position error > 0.5 m, base rotation
> 1 rad and divergence only. Wall contacts are not penalised by body part either (gap against the user's rule; candidate for
stage 3's reward, logged, not acted).
**B8 fixes A-C** for our height field (16y): not started. Next CPU item after 2h.

### 17c. Height-field fixes done, v2 fall defect found, v2b import with the new map assets (new ACP agent, 2026-11-12, 22:25-23:00 UTC)
Priority from the user (22:30): the extended map in all runs. ACP's stage 2 is unchanged; the map is made ready for stage 3. CPU only.
**Drop test** (`code/tools/drop_test.py`, results `eval/drop/*.json`, summary `eval/drop/summ.py`): SSR's test on OUR raster. A 10 kg
foot with the training model's left-foot collision geoms (12 sole spheres + the terrain foot box, contact params copied), training
options, MuJoCo Warp CPU; dropped at 1-4.5 m/s with 1 m/s sideways and a 150 N pull 0.1-0.3 s, at 2 flat, 2 slope and 2 edge sites
(edge = > 8 cm jump under the foot). Bar (SSR): no contact deeper than 1 cm for more than 0.07 s.

| raster, variant | box sink max (mm) | time > 1 cm, flat / slope / edge (s) | other |
|---|---|---|---|
| ramp 4 cm, as trained (base) | 209 | 0.11 / 0.06 / 0.06 | overflow lines; box jam reproduced |
| ramp 4 cm, A (box off the field) | 0 | 0.09 / 0.055 / 0.05 | |
| ramp 4 cm, A+B (solref 0.01, priority 1) | 0 | 0.015 / 0.01 / 0.025 | PASS; solref 0.005 identical (clamped) |
| ramp 2 cm (C), base / A / A+B | 221 / 0 / 0 | A+B: 0.015 / 0.01 / 0.13 | C is worse at edges: not used |
| v2 raster one field, as trained | 7,472 | - | **foot thrown 17-240 m** |
| v2 raster chunked, A+B | 0 | 0.02 / 0.01 / 0.06 | PASS, no ejection |

- **Second defect, the cause of the 16w "unexplained CPU falls":** MuJoCo Warp mis-collides on one 2,260 x 1,819 height field. On a flat
  1.66 m platform a sphere got a 9 m deep contact and the foot was thrown 240 m (`tools/hf_crop_diag.py`): MuJoCo C on the same model
  8 mm, Warp on crops of 51-681 cells 8 mm. Location-dependent, so it hit levels 0 and 9 and not level 4, and not a 4-env run.
- Single-step deep sphere contacts at cliffs (up to 26 cm for one step as a foot slides off an edge) remain with A+B; the time over 1 cm
  stays <= 0.06 s and the pull moves the foot, so no trap. Planes and boxes (the eval's fix) are not needed for release.
**Fix, in code** (`gear_sonic/supermimic/terrain.py` `add_hfield`, backup `backup/terrain.py.pre_hffix_20261112`; shared by code,
code_next, code_s3 via symlink): (1) the raster is split into fields of at most 400 x 400 vertices sharing border vertices (always on;
a raster that fits one field is one geom "course" as before, so the stage-2 ramp raster is unchanged; surface checked: 0 of 2,000
cell-interior rays off the raster); (2) `ACP_HF_FIX=1` = A + B. Default 0, so the running s2_v3, the live eval and the exports keep
their physics. **Decision:** ACP_HF_FIX=1 goes into the data-2h restart (one planned restart, user rule) and into stage 3.
**v2b import** (`tools/import_extended.py`, old outputs in `backup/goal_tiles_v2a.json`, `backup/hmap_ext_v2a_40mm.npz`; cfg
`cfg/stage3v2b_sets.json` = stage3v2 with a fresh label cache):
- stairs: `extended_v1_2_stairs_inset` tiles replace the extended_v1 stair tiles; per level one straight, one L, one U, plus the
  extended_v1 amphitheatre (not in the inset set): 4 rows, 40 tiles, 366 routes in 12 kinds (straight, side_left/right, oblique,
  far_straight, cross, L, C, S, descend, side_down, base).
- single_beams, beam_fields, blocks, wedges, log_corridors: + `extended_v1_1_prop_routes` (kinds prop_*). tight_spots, homes,
  street_front: base routes only (no extra set exists).
- At most 4 routes per kind per tile (seeded), so kinds are sampled about evenly; each route keeps `kind` and `check_note` in `info`.
  Routes flagged by the generator's checker defect (start above 1 m read as blocked) are kept: the defect is in the checker.
- Result: 884 extended routes (was 197) + 720 v1 tiles; raster 2,260 x 2,419 = 5.5 M cells, 30 fields, max 4.85 m.
**Ramp question (C8):** ACP uses the height map only (`tp_ramp_hmap_x30_40mm.npz`, max-pooled from `tp_ramp_hmap_x30.npz`; the ramp
set has no props file), never `props/ramp.json`. The tracker finished the ramp set at 99.1 % on that map (2f note), so the map and the
motions agree well enough for ACP (not re-checked per clip tonight). The mirrored ramp set comes with its own mirrored map: in 2h the two maps must share one raster
(place the mirrored one beside the original and shift its clips by the same offset).
**v2b goal check** (CPU closed loop, pair 7,000 = tracker 7,000 + planner 5,750, 2 per family at levels 0 / 4 / 9, same tiles in all
runs; `eval/goal_check_ext_v2b_007000_{single,nofix,fix}.json`, table `eval/gc_v2b_table.txt`):

| run | extended families: success / falls of 54 | v1 families: success / falls of 36 | overflow lines |
|---|---|---|---|
| v2a (16w, one field, no fix) | 3 / 36 | 20 / 0 | 13 |
| v2b one field, no fix | 0 / 46 | 14 / 4 | 32 |
| v2b chunked, no fix | 6 / 24 | 14 / 4 | - |
| v2b chunked + A+B (ACP_HF_FIX=1) | **9 / 22** | 14 / 4 | 1 |

- Chunking removes the defect (46 -> 24 falls); A+B adds a little (24 -> 22). The remaining falls are spread over levels and families
  (level 0: 5 of 18 fall) on terrain this stage-2 pair never trained on; I read them as the policy, which stage 3 trains, not physics.
- **Open, cheap:** the v1 stair families fall 4 of 36 under the v2b config in all three runs (0 in v2a, same tiles), so not the field.
  Difference between the runs: the config copy `stage3v2b_sets.json` with a freshly built label cache and 884 instead of 197 goal
  clips. To check before stage 3: v2b tiles with the old label cache, or v2a tiles with the new one.
**Can stage 3 start the moment a pair qualifies?** The map is ready (cfg `stage3v2b_sets.json`, env ACP_HF_FIX=1). Still needed first:
(1) merge the stage-2 additions into `code_s3` (env.py differs, train.py 148 diff lines: gate, data 2g/2h features), (2) the v1-stair
question above, (3) logs_back family (16z, optional). All CPU, a few hours; target before the 2026-11-13 12:00 check. The base gate
(16f) is not met yet (champion 7,000: full route 57 %), so the map is not the bottleneck.
**Data 2h** (restart target 2026-11-13 06:00 UTC, user's batching rule: one planned restart, then none for 2,000 iterations unless a
crash or failed trial): use the CORRECTED Pico sets `tp_stairs_all_v1_1.pkl` and `pico_2026-10-29_2x2_train_v1_1.pkl` (the old 2x2 set
in 2g has take `pico_20261029_211326_0_2x2` with a retarget artefact), plus the mirrored sets (`*_v1_1_mir`, `tp_t4x4u_y20_mir`,
`tp_ramp_base_x30_mir` with `tp_ramp_hmap_x30_mir.npz`, each with its `_mir_props.json`), all in
`/mnt/filesystem-g4/data/motions/ssr_prior_sources/terrain_park_train_sets/`; ACP_HF_FIX=1. Not built yet.
User rules to respect in 2h / stage 3 (USER_RULES last sections): backward travel paid less than forward or turning in place; turn
in place learned, also beside walls and on slopes.

## HOW TO RESUME (current at 2026-11-12 23:00 UTC node clock)
- Node: `config/nodes.local.json` key `acp`, user `NODE_USER`, `ssh -i ~/.ssh/nebius_corpus -o IdentitiesOnly=yes -o BatchMode=yes`.
  tmux: `acp` (run **s2_v3**, started 18:54:48 from tracker 7,000, planner frozen at 5,750, data 2g, probe gate `--gate-backoff 1
  --gate-rebase`, to 14,000; iteration 8,208 at 22:51), `acp-evallive2` (live paper eval of s2_v3), `acp-export` (pairs every 250 ->
  `eval/pair_s2_v3_<it>`, status `/mnt/filesystem-g4/status/acp_export.json`), `acp-backup`, `acp-gcv2b` (finished goal checks; can be
  closed). One GPU job per node; exports and checks on CPU. No git commits or pushes.
- Champion pair: 7,000 (full route 114/200 = 57 % on the final floor). Automatic scorer (`eval/autoscore/results.csv`): 7,250 gate PASS
  4/5, legs 49/41/43/50/44 (0.608); 7,750 gate FAIL 2/5 (turn-around 1/3). Selection, noise and stage-3 rules: 17a b), 17b, 17a c).
- Height field (17c): B8 fixed in `gear_sonic/supermimic/terrain.py` (chunked fields always; `ACP_HF_FIX=1` = foot box off the field
  + solref 0.01). Drop test passes (time > 1 cm <= 0.06 s). Running s2_v3 keeps the old physics (flag off).
- Extended map v2b (17c): `tools/import_extended.py` -> `data/goals/goal_tiles_v2.json` (884 extended routes, stairs inset, prop
  routes, kind-balanced), `data/clips/hmap_ext_v2_40mm.npz`; cfg `cfg/stage3v2b_sets.json`. Goal check with fix: 9 / 22 falls of 54.
- Next, in order: (1) build data 2h (17b + 17c list: corrected Pico v1_1 sets, mirrored sets, mirrored ramp map sharing one raster,
  ACP_HF_FIX=1) and restart s2_v3 with it at 2026-11-13 06:00 UTC; then no planned restart for 2,000 iterations. (2) v1-stair falls
  under the v2b config (17c, cheap). (3) Merge stage-2 additions into `code_s3`; logs_back family (16z). (4) v2 check due 2026-11-13
  12:00 UTC: the 17c goal check is it; repeat with the stage-3 code after (3). (5) Stage 3 only after the base gate (16f), old and
  new data sets side by side.
- Asked of the eval: the leg-5 traces of 17a d).
