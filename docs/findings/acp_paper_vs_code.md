# ACP track: paper vs live code audit

Independent audit, 2026-11-13 (node clock about 04:00 UTC). Paper: He et al., "Humanoid Mobility via Attention-based Contacts",
arXiv 2610.31608v1 (HTML), main text plus Supplementary Methods ("Training Details": data source, domain randomisation,
terrains, networks, observation encoding and curricula, planner optimisation and hindsight replay, contact and reward
annealing, PPO; "Evaluation Details"); Tables 1-3, S1-S6, Eq. 1-3. No code release.
Code: `/mnt/filesystem-g4/tracks/acp/code/` (same as `runs/s2_v3/code_snapshot`, checked with diff) and `code_s3/`.
Copies are in the coordinator scratchpad `acp_audit/`. Running job (s2_v3, iteration 9,761 at 03:53):
`acp.train --data cfg/stage2g_sets.json --envs 36000 --iters 14000 --njmax 450 --stage 2 --init .../init_from_s2_v2_007000.pt
--planner-arch attn --u-obs 200 --u-rew 1700 --n-ann 3000 --eta-obs-max 0.8 --ramp-gate 2 --probe-envs 720 --gate-signal probe
--ramp-gate-sets tp_t4x4u_y20_hs,tp_ramp_sh2_hs,x2_loco_flat_train_hold,x2_loco_blocks_train_hold
--freeze-planner .../pair_s2_v2_005750/tracker_ckpt_005750.pt --gate-backoff 1 --gate-rebase`, env `ACP_PLANT=mc_20261024
X2_EXCLUDE_PAIRS=hip` (ACP_HF_FIX unset). Live metrics: eta_obs 0.588, eta_rew 0.088, eta_hold 2,297, stage-2 iteration 4,261,
planner updates 0 (frozen), replay full at 1.5 M, fed-to-tracker share 0.60.

Status legend: **M** = MATCH; **DJ** = DEVIATION, justified (paper silent, verified / impossible on X2 or MuJoCo / user rule);
**DN** = DEVIATION, paper specifies, not followed; **A** = ADDED (not in paper). File references: `env.py` = `code/acp/env.py`,
`s3env.py` = `code_s3/acp/env.py`, `s3train.py` = `code_s3/acp/train.py`.

## Part 1: what the paper says about stage 3 and Eq. 3

**Stage 3 (Methods "Three-Stage Training Pipeline", "Terrains and Curriculum", "Hindsight Contact Labeling"; Tables 3, S2, S4-S6;
Supplementary "Terrains", "Contact and reward annealing", "Evaluation Details").**
- Goal terrains: procedural, 32 goal-reaching configurations, 10 difficulty levels set by step height, gap width or stone spacing
  (per-level values not given). Families (Supp. "Terrains"): dense foot (stairs, walled stairs, walk-up platforms, rough ground
  and slopes); sparse foot (gaps, balance beams, pallets, double stones, discrete platforms); dense whole-body (box up/down,
  get-up, uneven climbing terrain); sparse whole-body (ladders 50-65 deg, 20-30 cm rungs, 1-2 m; vaults 1.2-2 m wide,
  0.16-0.22 m thick, 0.47-0.53 m high, up to 5/3/10 deg roll/pitch/yaw). A reachable goal patch is sampled at reset.
- Mix: reference training continues "through proportional sampling from the original dataset"; **no ratio or env count is
  given** (verified silent). Stage-3 envs: 2,100 per worker x 4 GPUs = 8,400 (Table S6).
- Planner refinement: still from the hindsight replay, 10 Adam updates of 4,096 samples per PPO iteration after at least
  1,000 entries (Table S5). Success for the buffer: reference envs start in the first 10 % of the clip and reach 90 % of it;
  goal envs come within 0.5 m horizontally and 0.1 m vertically of the goal at some point. Goal segments are truncated at the
  first step within 0.2 m of the goal; at most 500 recorded steps per episode; buffer partitioned by terrain tile with
  quality-based eviction (lower positive mechanical work per metre of initial horizontal goal distance, floor 0.5 m); sampling
  uniform over terrain types, then tiles, then timesteps. Buffer capacity is not given (verified silent).
- Rewards (Table 3, Supp. "Contact and reward annealing"): goal envs get move2goal 40, standatgoal 20, track_planned_contact
  1.5, flat_orientation -10, stumbling -10, slippage -10, plus regularisers (and ang_vel_xy -0.1); no mimic terms. Reference envs
  keep the mimic terms; reference-contact terms are disabled and the planner-contact weight is 1.5; contacts are planner
  predictions only. Goal tolerances: 0.5 m horizontal, 0.2 m vertical, 0.5 heading; v_max 0.8 m/s on stairs, 2.0 elsewhere.
- Terminations: 15 s maximum, earlier under "configured failure conditions" (not listed: verified silent).
- Curriculum (Table S4): s <- 0.8 s + 0.2 I_success per episode; promote after a success with s > 0.5 (success = reached
  the goal at the required height, no early termination); demote when the minimum qualifying distance (measured only while the
  vertical error is < 0.1 m) exceeds 1.0 m. Initial phase fixed at 0 in stage 3 (Table S2); reset yaw +-0.4 rad for goal envs.
- Critics: second critic V_fs for goal envs, fixed lr 1e-3, 500-iteration warm-up during which goal samples train only the
  critic (out of the actor loss) and its value clipping is off; advantages normalised per env type. Target KL 0.01, entropy
  0.002. Duration about 2 days 6 h on 4 RTX 4090; iteration count not given.

**Eq. 3 (Supp. "Contact and reward annealing").** eta_xi(u) = 0 for u <= u_xi, (u - u_xi)/N_ann in between, 1 for
u >= u_xi + N_ann, xi in {obs, rew}; u = PPO iteration; shared N_ann; u_rew > u_obs. Input selection per env per control step,
coordinates not interpolated. Reference-contact weights 5.0/2.0/1.0 x (1 - eta_rew), planner-contact 1.5 eta_rew. **The values
of u_obs, u_rew and N_ann are not given anywhere** (no table row; verified). Our values (200 / 1,700 / 3,000) are ours.

## 1. Data and retargeting

| item | paper (ref) | code | status |
|---|---|---|---|
| Robot | Unitree G1, 23 controlled joints (Methods, Tracker) | X2 Ultra, 31 hinges (`assets.py`, `ACP_PLANT=mc_20261024`) | DJ (X2) |
| Retargeting | own constrained IK: landmark tracking, joint limits, foot unilateral, min-distance to obstacles, temporal reg. (Methods) | Bones-SEED X2 retargets, Pico captures, stair fits (`code/data_build`, cfg `stage2g_sets.json`) | DJ (different robot/data) |
| Skills (Table S1) | box climb, stairs, ladder, sit, walk, get-up, vault, sparse terrains (AME rollouts) | flat/blocks loco, chains, stairs, gst, gs2dn, gs2lo, ramp, 4x4, Pico 2x2, stand_edge, turn_place (`cfg/stage2g_sets.json`) | DN (no ladder, vault, sit, get-up, sparse; apparatus scope) |
| Data length | 13.2 min (Table 1) | about 196.7 min (state/acp.md section 6) | DN (more and different data) |
| Reference contact labels | per frame "from simulated contact geometry", 9 bodies (Methods) | kinematic: gap < 4 cm feet / 2 cm others, speed < 0.5 m/s, runs < 4 frames absorbed (`data.py:31,187-222`) | DJ (thresholds silent) |
| Oracle target | next contact location (Methods, Stage 1) | next onset, lowest-gap point within 25-frame settle, z = ground (`data.py:223-239`); no footprint clip | M (+settle A) |
| Clip sampling, stages 1-2 | not given | per-set hand weights shared by duration (`data.py:105-112`) | DJ (silent) |
| Data filters | whole reference set used | step-over > 0.30 m excluded, descents gs2dn_v1 out, leg-flip, ground_stance lift, 1 s stands prepended, arm-clearance fix (`armfix_stage2g_sets.pt`) | A |
| Clips cut into 10 s segments | not in paper | state/acp.md departure 5 | A |

## 2. Observations

| item | paper (ref) | code | status |
|---|---|---|---|
| Actor proprio history | 5 steps: ang vel, gravity, q, qd, prev action (Methods Obs., Table S4) | `env.py:23,308-320` (5 x 99 for 31 joints) | M |
| Tracker scan | 9 x 7, 0.1 m, 0.8 x 0.6 m (Table S4) | `env.py:183` x from -0.2 to 0.6 m (forward-biased) | M (placement silent) |
| Contact targets | 9 x 3, body-yaw frame, zero when invalid | `env.py:330-348` | M |
| Goal command (ref envs) | bounded direction d/max(abs d, r), r = 0.3, relative heading; target 0.5-1.5 s ahead (Table S4) | `env.py:30-31,334-340`; lookahead drawn per episode | M (draw rule silent) |
| Goal command (goal envs) | r = 0.5 m (Table S4) | `s3env.py:42,383` r = 0.2 m | DJ (user eval rule; paper value overridden) |
| Planner occupancy gate | target kept if p > 0.5 (Table S5) | `planner.py:229-230` | M |
| V_m critic | noise-free proprio, 17x11 scan, base lin vel, 52-d ref kin, 27 ref contact pos, 9 flags (Methods Obs.) | `env.py:375-379` | M |
| V_fs critic | proprio, base vel, scan, goal, d_g, dz, normalised time, planned contacts | `s3env.py:426-430` | M |
| Planner inputs | 5-step history, goal, 17x11 3-D points in body-yaw frame, 1.6 x 1.0 m | `env.py:184,351-355`; `planner.py:265-272` (noisy history and goal) | M |
| Observation normalisation | not given | running mean/var, clip 10 (`train.py:29-48`) | DJ (silent) |
| Measured contact | Fz > 1 N "in the sensor window" (Table 2) | Fz > 1 N at end of control step (`env.py:406`) | DJ (window length not given) |

## 3. Actions and control

| item | paper (ref) | code | status |
|---|---|---|---|
| Action | 23 offsets x 0.25 + default (Methods Tracker) | per-joint scale from X2 plant (`env.py:84,386`) | DJ (X2) |
| Rates | policy 50 Hz, PD 200 Hz | 5 ms implicitfast, decimation 4 (`assets.py:6,36,121-122`) | M |
| PD gains | not given (G1) | X2 plant `mc_20261024` | DJ (X2) |
| Action clamp | not given | +-20 (`env.py:385`) | A |
| Joint friction loss | not given | 0.3 on all hinges (`assets.py:4`) | A |
| Action std init | not given | 1.0 (`train.py:26`) | DJ (silent) |

## 4. Tracker rewards (Table 3)

| item | paper (ref) | code | status |
|---|---|---|---|
| track_base_pos 3.0 | exp(-chi_mov e_p^2/sp^2 - theta^2/sr^2) | `env.py:414` | M |
| track_base_vel 1.0 | exp(-dv^2/sv^2 - dw^2/sw^2) | `env.py:415` | M |
| track_joint_pos 2.5 | exp(-wrap(dq)^2/sq^2) | `env.py:416` (no wrap; hinges) | M |
| track_joint_vel 0.5 | exp(-dqd^2/sqd^2) | `env.py:417` | M |
| track_contact_label 5.0 | (1/9) sum c_meas (2 c_ref - 1) | `env.py:418` | M |
| track_contact_pos_base 2.0 | exp(-sum dp_B^2/sc^2) | `env.py:419` | M |
| track_contact_pos_global 1.0 | exp(-chi_mov sum dp_W^2/sc^2) | `env.py:420` | M |
| track_planned_contact | exp(-sum m_b dp^2/(sc^2 max(1, sum m_b))), 1.5 eta_rew / 1.5 | `env.py:429-430`, `env.py:481-488` | M |
| Regularisers | action_rate -0.1, 2nd -0.05, torque -1e-5, torque_limits -1e-3, dof_pos_limits -10 (cap 30), dof_vel_limits -1 (0.9 qd_max, cap 1), contact_acc -0.01 (cap 1e4) | `env.py:26-28,421-427` | M |
| Soft joint limits | "configured soft limits", values not given | MJCF range (`env.py:35`) | DJ (silent) |
| Kernels | sp 0.5, sr 1, sv 1, sw 3, sq 2, sqd 10, sc 1 | `env.py:29` | M |
| ang_vel_xy in stages 1-2 | -0.1 listed without stage breakdown (Table 3, style group, not marked goal-only) | absent in stages 1-2, goal envs only in stage 3 | DJ (paper ambiguous) |
| Reward scaling | r = dt sum w_i r_i | `env.py:432` | M |

### Stage-3 goal rewards (Table 3, goal-only terms)

| item | paper (ref) | code | status |
|---|---|---|---|
| move2goal 40 | (1-g_xy) g_v + g_xy g_z; g_z = 1 except get-up terrains | `s3env.py:517` (g_z = 1, no get-up tiles) | M (radius below) |
| g_xy radius | 0.5 m | 0.2 m (`s3env.py:42,506`) | DJ (user eval rule) |
| g_v | v_to-goal > 0.1 and speed <= v_max (0.8 stairs, 2.0) | `s3env.py:507`, `goals.py:34` | M |
| standatgoal 20 | g_xy g_h g_psi exp(-(d_foot+d_tilt+d_q)/3), g_h 0.2 m, g_psi 0.5 | `s3env.py:510-518`, extra factor `station` | A (station gate) |
| flat_orientation -10 | T_ori torso gravity xy^2 (mask unspecified) | pelvis, no mask (`s3env.py:520`) | DJ (mask silent) |
| stumbling -10 | sum over feet 1[F_xy > 1.1 F_z + 1] | any foot 1[F_xy > 5 F_z] (`s3env.py:521`) | DN |
| slippage -10 | sum over feet 1[contact > 0.026 s] 1[v_f > 0.5 m/s] | sum v_f^2 x contact (`s3env.py:522`) | DN |
| ang_vel_xy -0.1 | torso, world frame | pelvis, body frame (`s3env.py:523`) | DJ (minor) |
| Goal envs: no mimic terms | Table 3 | `s3env.py:524-526` | M |

## 5. Planner architecture and losses (Methods Planner, Table S3, S5, Eq. 1-2)

| item | paper (ref) | code | status |
|---|---|---|---|
| Point embedding 3->16 ELU; conv 1->16->48 5x5 pad 2 ELU; proj 64->96 | Table S3 | `planner.py:254-256` | M |
| Context 96->64->64 pointwise + max pool | Table S3 | `planner.py:257,271` | M |
| Proprio-goal-context encoder ->128->128->96 ReLU | Table S3 (442 in) | `planner.py:258` (562 in for X2) | M |
| 9 body embeddings 96, FiLM 96->64->192 | Table S3 | `planner.py:259-260,273-274` | M |
| Cross-attention d 96, 32 heads, 1 layer | Table S3 | `planner.py:261-280` | M |
| Affordance = attention weights | "distribution of affordance weights" | mean over the 32 heads (`planner.py:281`) | DJ (head pooling silent) |
| Occupancy head 160->512x3->1 sigmoid | Table S3 | `planner.py:263,282` | M |
| Top-k decode, k = 1 | Table S5 | `planner.py:220-228` | M |
| Eq. 1 soft 3-D Gaussian target, sigma 0.1 m | Eq. 1 | `planner.py:290-291` | M |
| Eq. 2 masked CE + BCE over 9N | Eq. 2 | `planner.py:293-295` | M |
| Adam 1e-4, betas (0.9, 0.999), eps 1e-8, wd 0 | Table S5 | `train.py:151` (defaults) | M |
| 4,096 samples via 8 accumulated sub-batches | Table S5 | one batch of 4,096 (`train.py:302`) | M (equivalent) |
| MLP baseline input | 939 = history + goal + 187 x 3 points | heights only (`planner.py:238`) | DN (baseline only, unused) |
| Planner input normalisation | not given | from tracker normalisers or data (`train.py:147-150`) | A |
| Offline planner pre-training | "after planner initialization" (no procedure) | `--planner-init planner_init.pt` (s2_v2, `restart_t7000.sh:27`) | A |

## 6. Stage 1

| item | paper (ref) | code | status |
|---|---|---|---|
| Tracker alone, oracle contacts | Methods Pipeline | `env.py:330-349` (planner None) | M |
| CSI: 0.9, -0.05 on reaching 0.99, +0.05 otherwise, [0, 0.99] | Table S4 | `env.py:32,457-458` (per env) | M |
| Envs | 12,000 per worker (x3 GPUs, Table S6) | 12,000, then 36,000 on 1 GPU from 750 | DN |
| Target KL 0.02, entropy 0.006 | Table S6 | `train.py:25,110` | M |
| Duration / iterations | about 1 day, 3 x 4090; iterations not given | s1 to 2,250 / 5,500 | DJ (silent) |

## 7. Stage 2 and Eq. 3

| item | paper (ref) | code | status |
|---|---|---|---|
| Planner co-trained online | planner trained jointly, 10 updates per iteration (Methods, Table S5) | **frozen** at the 5,750 planner (`--freeze-planner`, `train.py:154-156,299`; 0 updates live) | DN |
| Per-env per-step switch, no interpolation | Supp. annealing | `env.py:358-362` | M |
| Eq. 3 form | linear, shared N_ann, u_rew > u_obs | `train.py:208-210` | M |
| u_obs, u_rew, N_ann | not given | 200 / 1,700 / 3,000 (flags) | DJ (silent, verified) |
| eta_obs ceiling | ramps to 1 | `--eta-obs-max 0.8` (`train.py:209`) | DN |
| Ramp gate / hold | not in paper | probe gate, eta_hold 2,297 (`train.py:194-208`) | A |
| Probe envs always on planner | not in paper | 720 (`env.py:359-360`) | A |
| Initial phase U(0, 0.98), CSI off | Table S2, S4 | `train.py:138`, `env.py:267` | M |
| Recorder envs start in first 10 % | admission rule only | first 3,000 envs (`env.py:268-269`) | A |
| Envs / minibatch | 2,400 x 2 = 4,800; effective minibatch 14,400 (Table S6) | 36,000; minibatch 108,000 | DN |
| Target KL 0.01, entropy 0.005 | Table S6 | `train.py:110` | M |
| Mimic terms unchanged | Table 3 | `env.py:481-488` | M |

## 8. Stage 3 (code_s3, not yet run)

| item | paper (ref) | code | status |
|---|---|---|---|
| Goal terrains | 32 configurations, 4 groups incl. gaps, stones, beams, pallets, ladders, vaults (Supp. Terrains) | 6 families (`goals.py:33`) + extended map v2b 884 routes (`cfg/stage3v2b_sets.json`) | DN (scope) |
| 10 levels | Table S4 | `s3env.py:563` clamp 0-9 | M |
| Goal = patch at reset | Methods | multi-waypoint routes, stations, sharp turns (`goals.py`, `s3env.py:527-535`) | A (user rule) |
| Mix ref/goal | not given | `--goal-share 0.5`, interleaved (`s3env.py:219-226`) | DJ (silent, verified) |
| Ref envs sampled proportionally | Methods | `clip_w` set weights | M |
| Initial phase 0 in stage 3 | Table S2 | goal envs 0 (`s3env.py:307`); **ref envs U(0, 0.98)** | DN |
| Reset yaw +-0.4 rad (goal) | Table S2 | none; fixed spawn yaw per tile (`goals.py:80-141`) | DN |
| Episode 15 s | Table S4 | `GOAL_T 750` | M |
| Goal-env terminations | "configured failure conditions" | falls only: root < 0.35 m over ground, tilt > 60 deg (`s3env.py:495-539`) | DJ (silent) |
| EMA 0.8/0.2, promote on success with s > 0.5 | Table S4 | `s3env.py:558-563` | M |
| Success at required height (0.1 m) | Table S4 | horizontal only, 0.2 m held 1 s (`s3env.py:529-532`) | DN |
| Demotion: qualifying distance (vertical < 0.1 m) > 1.0 m | Table S4 | horizontal closest approach (`s3env.py:537,562`) | DN (minor) |
| Contacts planner-only | Supp. annealing | goal envs always (`s3env.py:405`); ref envs eta_obs | M |
| Ref-contact terms off, planned 1.5 | Supp. annealing | via eta_rew = 1 | M |
| eta continuity at the switch | n/a | `s3train.py:191-194` ignores eta_hold and eta-obs-max: eta 0.59/0.09 jumps to 1/1 | A (risk) |
| Planner refined | Table S5 | updates resume (`s3train.py`, no freeze) | M |
| V_fs lr 1e-3 fixed, 500 warm-up, goal samples out of actor loss | Table S6 | `s3train.py` (`--s3-warmup 500`, `pw`) | M |
| V_fs value clipping off in warm-up | Table S6 | always clipped (`s3train.py:297-298`) | DN (minor) |
| Separate advantage normalisation | Table S6 | `s3train.py` | M |
| KL 0.01, entropy 0.002 | Table S6 | `s3train.py` | M |
| Envs | 2,100 x 4 = 8,400 | 36,000 | DN |
| Launch script | n/a | `ops/to_stage3.sh:65` uses `cfg/stage3_sets.json` (v1), `--n-ann 1500`, no `X2_EXCLUDE_PAIRS=hip`, no `ACP_HF_FIX` | A (stale; risk) |

## 9. Hindsight relabelling (Methods, Table S5)

| item | paper (ref) | code | status |
|---|---|---|---|
| Force threshold 1 N | Table S5 | `env.py:37,406` | M |
| Min segment 10 steps | Table S5 | `planner.py:24,86-88` | M |
| Horizon 75 steps | Table S5 | `planner.py:24,104` | M |
| Footprint +-0.8 x +-0.5 m | Table S5 | `planner.py:26,104` | M |
| Backward scan for next onset | Methods | `planner.py:94-106` | M |
| Target = position at onset | Methods | body point 5 steps after onset (`SETTLE`, `planner.py:24,96`) | A |
| At most 500 steps per episode | Table S5 | `Recorder.MAXT` | M |
| Every step labelled | implied | every 2nd step (`planner.py:141`) | A |
| Ref admission: start in first 10 %, reach 90 % | Methods | `env.py:268-269,475` | M |
| Goal admission: 0.5 m horiz and 0.1 m vert | Methods, Table S5 | `g_reached` (0.2 m, 1 s, no vertical) (`s3env.py:587`) | DN |
| Goal segment truncated at 0.2 m of goal | Table S5 | not implemented | DN |
| Buffer per tile, quality eviction (work per metre) | Methods, Table S5 | one FIFO ring (`planner.py:110-133`) | DN |
| Balanced sampling types -> tiles -> steps | Methods | uniform (`planner.py:131-133`) | DN |
| Capacity | not given | 1.5 M (`--replay`) | DJ (silent) |
| Buffer persistence | not given | in memory, empty after restart | A |

## 10. Terminations

| item | paper (ref) | code | status |
|---|---|---|---|
| Reference envs | not given ("configured failure conditions") | base > 0.5 m or > 1 rad from reference, diverged (`env.py:33-34,433-439`) | DJ (silent, verified) |
| Timeout at clip end | not given | `env.py:453` | DJ (silent) |
| Fall rule for planner-driven evals | not given | `env.py:434-438` | A |

## 11. Domain randomisation (Table S2, Methods)

| item | paper (ref) | code | status |
|---|---|---|---|
| Robot friction U(0.3, 1.2), ground fixed 1.0 | Table S2 | ground 0.3 in training, max-combine, so effective 0.3-1.2 (`env.py:134,148`) | DN (ground value) |
| Restitution U(0, 0.5) | Table S2 | none | DJ (MuJoCo) |
| Torso mass -2..+4 kg, leg mass x0.85-1.15, CoM +-0.025/0.05, zero bias 0.01/0.03, armature and kp/kd x0.9-1.1 | Table S2 | `env.py:148-166` | M |
| Delay 0-20 ms | Table S2 | 0-4 substeps of 5 ms (`env.py:165,209-227`) | M |
| Proprio noise (0.05, 0.0125, 0.01, 1.5) | Table S2 | `env.py:314-319` | M |
| Contact-target noise +-0.1 m | Table S2; "all actor observations" | reference targets only; **planner targets fed without noise** (`env.py:343` vs `361`) | DN |
| Goal noise +-0.2 | Table S2 | `env.py:344` | M |
| Planner scan N(0, 0.03) | Table S2 | `env.py:353` | M |
| Tracker scan U(-0.1, 0.1) | Table S2 (text: per-step noise + per-episode offset) | per-episode U(+-0.1) offset + per-step N(0, 0.03) (`env.py:294,347`) | DJ (ambiguous; see notes check) |
| Odometry delay 50-100 ms, XY drift 0.05 m disk, Z +-0.05 m | Table S2 | not implemented | DN |
| Reset joint +-0.2, base +0-0.05 m | Table S2 | `env.py:274-275` | M |

## 12. PPO and optimiser (Table S6)

| item | paper (ref) | code | status |
|---|---|---|---|
| 24 steps, 5 epochs, 8 minibatches, clip 0.2, value coef 1, grad norm 1, gamma 0.99, lambda 0.95 | Table S6 | `train.py:24` | M |
| lr 1e-4, KL-adaptive | Table S6 | `train.py:25,277-283` (x1.5 rule, bounds 1e-5..1e-2) | M (bounds silent) |
| Advantage normalisation over the rollout | Table S6 | `train.py:265` | M |
| Value clipping | implied (disabled only for V_fs warm-up) | `train.py:288-289` | M |
| Time-out bootstrap | not given | `train.py:223` | DJ (silent) |
| Batch size | per stage, Table S6 | 36,000 envs throughout (stage 2: 7.5 x the paper's rollout batch) | DN (listed under stages) |

## 13. Networks

| item | paper (ref) | code | status |
|---|---|---|---|
| Actor, V_m, V_fs: 512-256-128 ELU | Table S3 text | `train.py:51-52,58`; `s3train.py` (mlp) | M |
| Input sizes 468 / 353 / 298 | Table S3 text | larger for 31 joints | DJ (X2) |
| State-independent log std | not given | `train.py:59` | DJ (silent) |

## 14. Simulation settings

| item | paper (ref) | code | status |
|---|---|---|---|
| Simulator | Isaac Lab | MuJoCo Warp | DJ (coordinator decision; restitution impossible) |
| GPUs | 3 / 2 / 4 x 4090 | 1 GPU | DJ (resources) |
| Terrain | meshes in Isaac Lab | boxes as mocap prop slots + 4 cm height raster (`terrain.py`); `ACP_HF_FIX` off in the live job (contact overflow > 50 known) | DJ (MuJoCo) |
| Self-collision | not given | hip pairs excluded (`X2_EXCLUDE_PAIRS=hip`) | A |
| dt, solver, contact params | not given | 5 ms implicitfast, defaults; solref 0.01 only with ACP_HF_FIX | DJ (silent) |

## 15. Evaluation protocol (Results 0.3, Supp. "Evaluation Details")

| item | paper (ref) | code | status |
|---|---|---|---|
| Groups: reference-anchored (123 configs), goal-reaching (32 configs, even levels), unseen (composition, distractors, roughness 5-15 cm), 8 composite courses | Supp. Evaluation | `tools/paper_eval.py`: reference sets only | DN (stage 3 not reached; partly not built) |
| 3,000 episodes per category, 3 seeds; 1,000 per radar axis / course | Results 0.3 | 334 per set per seed x 3 (`paper_eval.py:13,70`) | M |
| Success 0.5 m horizontal (0.1 m vertical for sit/stand/get-up) | Results 0.3 | `paper_eval.py:138`; early termination also counts as failure | M (+A) |
| Noise, DR, delay off; Wilson 95 % CI; contact error at onset | Results 0.3 | `eval_mode=True`, `wilson()`, `env.py:440-450` (nearer of current or previous target) | M (+A) |

## 16. Other

| item | paper (ref) | code | status |
|---|---|---|---|
| Deployment: LiDAR neural map, DLIO, Orin | Methods Deployment | not applicable (common apparatus eval) | DJ (out of scope) |
| Contact placement metric | Results 0.3 | `env.py:440-450` | M |

## Status counts

Rows in the tables above: **M 75, DJ 33, DN 25, A 18** (151 rows; rows marked "M (+A)" or "M (+settle A)" count as M).

## Check of the "paper silent" notes in state/acp.md

| note (location) | verdict |
|---|---|
| Terminations not given (lines 154, 305, 429, 1339; `env.py:33`) | Correct: only "configured failure conditions" and the 15 s limit. |
| Eta schedule numbers not given (154, 322, 610) | Correct: Eq. 3 has symbols only. Note the notes disagree with each other: line 610 says u_rew 700 over 1,000, line 322 says 1,700 over 1,500; the live job uses 200 / 1,700 / 3,000. |
| Stage-3 mix not given (154, 334; `--goal-share` help) | Correct ("proportional sampling" only). |
| Iteration counts not given, only wall-clock (pipeline summary, 1210) | Correct. |
| Soft joint limits not given (`env.py:35`) | Correct. |
| Action std, obs normalisation, value clipping, CSI per env, lookahead draw (305) | Correct; value clipping is implied by Table S6 (it is turned off only for the V_fs warm-up). |
| How early terminations count in eval (388) | Correct; only invalid outputs are named as failures. |
| Planner SETTLE "ours" (`planner.py:24`) | Correct. |
| **"Clip count and hours are not stated"** (pipeline summary, Data) | **Wrong in part**: Table 1 gives the data length, 13.2 min (section 6 of the same file already has it). Clip count is indeed not given. |
| **Tracker scan noise read as offset U(+-0.1) + per-step N(0, 0.03)** (306, `env.py:345-347`) | **Not silent**: Table S2 gives U(-0.1, 0.1) m for the tracker scan and N(0, 0.03) m for the planner scan. Only the size of the per-episode offset is open. The code puts the planner's Gaussian onto the tracker scan. |
| **Stumbling recorded as "foot F_xy > 5 F_z"** as the Table 3 term (1179) | **Wrong**: Table 3 gives 1[F_xy > 1.1 F_z + 1] summed over feet. Slippage also differs (paper: indicator on contact > 0.026 s and foot speed > 0.5 m/s). Not flagged as departures. |
| Goal yaw +-0.4 rad (96) | Recorded as paper value, but not implemented in `code_s3`. |
| Friction (94) | The summary omits Table S2's "ground friction fixed at 1.0"; the code uses 0.3 for the ground. |

## Ranked: deviations most likely to matter for apparatus success

1. **Planner frozen at 5,750 in stage 2, eta_obs capped at 0.8 and held (eta_hold 2,297)** (DN). The paper co-trains the planner
   on the tracker's own successes and ramps to 100 % planner contacts. The apparatus runs on planner contacts only, while the
   tracker still gets reference contacts 40 % of the time (fed share 0.60), and the planner does not adapt to the tracker.
2. **Stage-3 switch path** (A, risk): `code_s3` ignores eta_hold and eta-obs-max, so eta jumps from 0.59/0.09 to 1/1 in one
   iteration. `ops/to_stage3.sh` is stale: v1 config, n-ann 1,500, no `X2_EXCLUDE_PAIRS=hip` (that changes self-collision
   physics at the switch), no `ACP_HF_FIX`.
3. **Planner targets fed to the tracker without the +-0.1 m contact-target noise** (DN). The tracker learns robustness only on
   reference targets; on the apparatus every target comes from the planner.
4. **Stage-3 replay rules missing**: no vertical check in goal success or admission, no 0.2 m truncation, no per-tile partition,
   quality eviction or balanced sampling (DN). Easy families and inefficient gaits would dominate planner refinement on stairs and
   platforms.
5. **Stumbling and slippage formulas** (DN): F_xy > 5 F_z is far more lenient than 1.1 F_z + 1 for toe-stubbing on risers,
   which is the apparatus stair failure mode.
6. **Batch size** (DN): 36,000 envs against 4,800 (stage 2) or 8,400 (stage 3). That is about 7.5 x fewer PPO updates per
   environment step with the same epochs and minibatch count.
7. **Ground friction 0.3 instead of 1.0** (DN): the effective friction range reaches 0.3, below the paper's setting. This can
   make gaits conservative or cause slips on treads.
8. **Odometry delay and drift not simulated** (DN). This matters if the apparatus eval or the robot uses estimated state.
9. Stage-3 reference envs not starting at phase 0, and no reset yaw +-0.4 rad (DN; low).
10. Goal radius 0.2 m and stations instead of 0.5 m (DJ, user rule). This is stricter than the paper and is in line with the
    apparatus scoring, so it is intended, not a defect.
