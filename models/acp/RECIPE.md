# ACP recipe: the current best as one uninterrupted run from scratch

Written 2026-11-13 (node clock ~13:00 UTC) from the run history in `models/acp/LEDGER.md`. "Current best" = the lineage
that produced pair 8,000 (station legs 231/250) and the running s2_v4 (12,250: 229/250). Stage 3 has never run: its part
is the plan, not a tested recipe. Tags: [paper] = He et al., arXiv 2610.31608; [fix] defect fixed; [user rule]; [ours].
Code: `/mnt/filesystem-g4/tracks/acp/code` (one folder; repo copy `models/acp/code/`). Every number below is a value read
from the running command line, the code defaults or `agentic/state/acp.md`; section references are to that file.

## 1. Hardware, environment, seeds

| Setting | Value | Tag |
|---|---|---|
| GPU | 1 x RTX PRO 6000 Blackwell (96 GB). The trainer is single-GPU; there is no multi-GPU code path | ours |
| Simulator | MuJoCo Warp 3.17.0, mujoco 3.17.0, warp-lang 1.18.0, torch 2.11.0+cu128, python 3.12; lock file `tracks/acp/env/requirements.lock.txt` | ours (paper: Isaac Lab) |
| Envs | 36,000 (`--envs 36000`) | ours (paper: 12,000 x 3 workers stage 1; 4,800 stage 2; 8,400 stage 3; user kept our count, 2026-11-13) |
| Constraint buffers | `--njmax 450` [fix: 300 overflowed], `--nconmax 64` (default) | fix / ours |
| Seed | `--seed 0` (default; one seed only) | ours |
| Throughput | stage 1 ~110-117k env-steps/s (7.4 s/iteration); stage 2 ~75k (11.5 s); GPU memory ~53 GB stage 2 | measured |
| Checkpoints | numbered every 250 (`--ckpt-every 250`), `latest.pt` also every 20 min (`--latest-minutes 20`) | user rule (backup) |
| Heartbeat | `/mnt/filesystem-g4/status/acp.json` every 10 iterations | ours |
| Supervisor | `code/supervise.sh RUN ENVS ITERS`: resumes from `latest.pt`, restarts on exit or 15 min without a log line, gives up after 20 restarts, waits while another process holds the GPU | ours |

## 2. Body, plant, physics (all from iteration 0)

| Setting | Value | Tag |
|---|---|---|
| Robot model | `x2_ultra.xml` (shared, sha256 4f6a4f5b...) with `X2_EXCLUDE_PAIRS=hip` (pelvis / hip-roll hull contact excluded = `x2_ultra_hipexclude.xml`) | fix (FINDINGS A1) |
| Plant | `ACP_PLANT=mc_20261024` (measured kp / kd; armature, effort, default pose, action scale from the plant file) | user rule |
| Control | dt 0.005 s, 4 substeps, 50 Hz policy, implicitfast, PD position servos built by `assets.build_x2` | ours |
| Height field | `ACP_HF_FIX=1`: fields chunked to <= 400 x 400 vertices (always on), foot terrain box off the field, field solref 0.01 priority 1 | fix (B8, B10) |
| Simcheck | fault detector (`/mnt/filesystem-g4/tools/simcheck/patches/acp.diff`): a body jump truncates the episode with zero reward; `sim_fault_share` in the heartbeat | user rule (TRACK_RULES) |
| Randomisation | Table S2: kp, kd, armature x U(0.9, 1.1); `ACP_PAPER_DR=1`: planner contact-target noise U(+-0.1 m) on targets fed to the tracker, tracker scan per-step U(+-0.1) + per-episode offset, friction effective (r+1)/2 with robot r ~ U(0.3, 1.2) and ground 1.0, odometry drift per episode (XY 0.05 m disc, Z U(+-0.05)) | paper |
| Not modelled | restitution (MuJoCo has none), odometry delay 50-100 ms | DN (paper specifies delay) |
| Terminations | base position error > 0.5 m or orientation error > 1.0 rad against the reference, divergence, simcheck fault; no wall-contact termination | ours (paper silent) |

## 3. Data (reference clips; all paths under `/mnt/filesystem-g4/tracks/acp/data/clips/`)

Config `code/cfg/stage2h_sets.json` (data 2h, 1,469 clips, 558,062 frames), raster `tp_ramp_hmap_x30_2h_40mm.npz` (original
ramp + mirror 6 m beside it, one field). Arm fix `cfg/armfix_stage2h_sets.pt` (2 cm hull clearance) and labels
`cfg/labels_stage2h_sets.json.pt` are built by the data tools and checked by sha256 in `code/MOTIONS.md`.
Use the SAME data from iteration 0 in the single run (history switched data 7 times, see section 9).

| Set (file) | Weight | Filters | Tag |
|---|---|---|---|
| x2_loco_flat_train_hold (Bones-SEED retargets) | 0.11 | exclude names injur, bump, crouch, jump; 1.5 s end stand | user rule (BONES-SEED stays for ACP) + ours |
| x2_loco_blocks_train_hold | 0.20 | same | ours |
| x2_chains_far_hs | 0.15 | step > 0.30 m out; 5 defect segments out; 1 s start stand | fix |
| tp_stairs_x10_sh2_hs | 0.03 | ground_stance; takes 114308, 114330 out | fix |
| gst_v1_hs (fitted climbs) | 0.07 | exclude jog | ours |
| gs2dn_v2_hs (descents v2) | 0.05 | exclude leg flip | fix |
| gs2lo_v1_hs (10-15 cm risers) | 0.05 | exclude leg flip | ours |
| tp_ramp_sh2_hs | 0.035 | shift (30, 0), ground_stance | fix |
| tp_t4x4u_y20_hs | 0.08 | ground_stance; 2 defect clips out | fix |
| pico_2026-10-29_2x2_train_hold (old) | 0.0 | kept only for set indices | - |
| stand_edge_v1 (stand at a 1.1 m drop) | 0.05 | | ours |
| turn_place_v1 | 0.03 | | ours |
| pico_2026-10-29_2x2_train_v1_1_hold / _mir_hold | 0.02 / 0.01 | exclude leg flip | user rule (user's captures) |
| tp_stairs_all_v1_1_hs / _mir_hs | 0.03 / 0.02 | ground_stance | user rule |
| tp_t4x4u_y20_mir_hs | 0.04 | ground_stance; mirrors of the 2 defect clips out | user rule |
| tp_ramp_base_x30_mir_hs | 0.015 | shift (0, 6), ground_stance | user rule |
| tp_stairs_turn_v1_1_hs (turn / approach subset) | 0.01 | ground_stance | ours |

Sources of the corrected Pico and mirrored sets: `/mnt/filesystem-g4/data/motions/ssr_prior_sources/terrain_park_train_sets/`.
Held out (never trained): `x2_route*.pkl`, `x2_nist_route_pass.pkl`, `*_test` sets, every SuperMimic-derived file.

Stage-3 goal terrain: `cfg/stage3v2b_sets.json` = the 2g reference sets + goal tiles v1 (`data/goals/goal_tiles_v1.json`,
720 tiles, 6 families, 10 levels) + extended map v2b (`data/goals/goal_tiles_v2.json`, 884 routes from
`/mnt/filesystem-g4/data/terrains/extended_v1/` + `extended_v1_2_stairs_inset` + `extended_v1_1_prop_routes`, raster
`data/clips/hmap_ext_v2_40mm.npz`) [user rule: extended map in all runs; groups ours, no ladders / vaults]. NOTE: stage3v2b
lists the 2g reference sets, not 2h; a 2h-based stage-3 config does not exist yet (open item 3).

## 4. Networks, PPO

| Setting | Value | Tag |
|---|---|---|
| Actor / critic inputs | actor 588, critic 393, 31 actions; action clamp +-20 in `env.step` | paper (X2 sizes ours) |
| Planner | attention (`--planner-arch attn`, 802,353 parameters), top-1 decode over the 17 x 11 scan, decode in chunks of 4,096 | paper; fix (memory) |
| PPO stage 1 | 24 steps / iteration, 5 epochs, 8 minibatches (108,000), lr 1e-4 KL-adaptive target 0.02, clip 0.2, entropy 0.006, action std init 1.0, clipped value loss, running obs normalisation | paper (Table S6) / ours (silent items) |
| PPO stage 2 | target KL 0.01, entropy 0.005 | paper |
| PPO stage 3 | target KL 0.01, entropy 0.002; second critic V_fs (lr 1e-3); advantages normalised per env type; 500-iteration warm-up without goal samples in the actor loss | paper |
| Planner training | hindsight replay 1.5 M samples, 3,000 recorder envs, 10 updates x 4,096 per iteration, Adam 1e-4, from 1,000 entries | paper (Table S5) |

## 5. Stage schedule (iteration numbers, single run)

History reference: the tracker had ~3,750 stage-1 iterations (s1_v1 0-2,250 + s1_v2 4,000-5,500) and ~8,700 stage-2
iterations before the current best. The schedule keeps those amounts but puts them in paper order.

| Phase | Iterations | What runs | Tag |
|---|---|---|---|
| Stage 1 | 0 -> S1_END (rule 6a; expected ~3,500-4,500, cap 6,000) | `--stage 1`, oracle (reference) contacts, CSI resets (phase rises on success) | paper |
| Stage 2 start | S2 = S1_END | `--stage 2 --init <stage-1 ckpt>`, planner online from scratch, `--u-obs 1000 --u-rew 1700 --n-ann 3000 --eta-obs-max 0.8 --ramp-gate 2 --probe-envs 720 --gate-signal probe --ramp-gate-sets tp_t4x4u_y20_hs,tp_ramp_sh2_hs,x2_loco_flat_train_hold,x2_loco_blocks_train_hold --gate-backoff 1` | paper + ours |
| eta_obs ramp | from S2 + u_obs, linear at 1/6,000 per iteration (N_ann 6,000) to 1, no cap, no gate / hold (USER_RULES "ACP fixed stage lengths", 2026-11-13; s2_v5 runs it from the checkpoint values 0.781 / 0.281 at 12,604) | user rule (paper: Eq. 3, numbers not given) |
| eta_rew ramp | from S2 + 1,700, linear at 1/6,000 to 1 (s2_v5: 1 at 16,918) | user rule |
| Planner | co-trained online throughout (no freeze, since s2_v5 2026-11-13); replay sampling balanced data sets -> clips -> frames | paper (+ balanced sampling, ours) |
| Stage 2 end | 2,300 iterations after eta_rew reaches 1 (10,000 for a run from scratch; s2_v5: 19,218) | user rule |
| Stage 3 | exactly 15,000 iterations, switched automatically (`ops/auto_stage3.sh`; holds in stage 2 if the stage-3 CPU check fails): `--stage 3 --s3-paper 1`, cfg `stage3v2h`, flags `ops/stage3_flags.env`, eta 1 carried (anchor at the switch), planner co-trained, `--goal-share 0.5`, `--s3-warmup 500`; paper rules: success 0.5 m / 0.1 m, 0.2 m cut, per-tile replay with work-per-metre eviction, types -> tiles -> steps, yaw +-0.4, phase 0, Table 3 stumbling / slippage; waypoint / station radius 0.2 m and 1 s stations kept | paper + user rules; s2_v5 -> s3_v2 at 19,218 |

## 6. Automatic rules that replace human decisions

Measures and where they are computed:
- `finish/<set>`, `end/finished_phase099`, `probe/finish/<set>`, `stage2/eta_obs|eta_rew|eta_hold`, `sim_fault_share`: per
  iteration in `runs/<run>/metrics.jsonl`, written by `code/acp/train.py` / `acp/env.py`.
- Paper eval (reference contacts): `code/tools/paper_eval.py`, result key `category_dense_foot_only.success_pooled` and per
  set, 334 episodes per set x 3 seeds, njmax 450, body from the checkpoint; live beside training with `code/tools/eval_live.sh`
  (lock, 80 GB start check, 15 min limit; costs ~4 min at -40 % speed).
- Planner closed loop: `code/tools/pair_eval.py` (oracle vs planner contacts, falls-only rule beside the 0.5 m rule).
- Pair export: `code/ops/export_pair.sh RUN IT` via `ops/export_every.sh` (status `/mnt/filesystem-g4/status/acp_export.json`;
  no DONE when pair_eval fails).
- Off-node, after the run: station legs and full route on the apparatus (`eval/autoscore/autoscore.py`), and the training-
  terrain courses (`eval/train_terrains.py`).

a. **Stage 1 -> 2 (base gate)** [user rule "base first" + ours]: every 1,000 iterations from 2,000, one live paper eval.
   Switch at the first eval with flat >= 98 %, blocks >= 98 % and every other set >= 95 % (history: met at 4,500 except 4x4
   89.8 / chains 94.9, fully at 6,500 in stage 2). Else switch when pooled gains < 1 point over the previous 1,000
   iterations and iteration >= 3,000 (plateau), or at 6,000 (cap).
b. **Planner start in stage 2** [paper]: online from scratch; it trains on hindsight labels from the start but feeds the
   tracker only from S2 + u_obs (u_obs = 1,000 so the planner has ~1,000 iterations of labels from the same tracker before
   any contact reaches it; ours, replaces the offline pre-training of history).
c. **Planner freeze** [ours; RETIRED 2026-11-13, s2_v5 co-trains with balanced replay]: at S2 + 1,250 (eta_obs ~0.08). Reason: when eta_obs passed ~0.1 the online planner forgot
   rare skills (turn-around, 0.5 % of frames) within 250 iterations and the apparatus fell 57 -> 39.5 % (`agentic/state/acp.md` 16q-16r).
   Code-checkable alternative (better, not built): freeze at the last checkpoint before probe finish on flat or blocks
   drops > 2 points under its best.
d. **Ramp gate** [ours; RETIRED 2026-11-13: fixed schedule, user rule]: hold eta (eta_hold += 1 per iteration) while the probe-finish EMA of any gated set is under its base
   - 2 points; base = best EMA after 100 updates, relearned at every start (`--gate-rebase` only needed when resuming).
   No back-off (`--gate-backoff 1`): back-off 3 fired on noise (17a).
e. **Stage 2 -> 3** [REPLACED 2026-11-13 by the user's fixed lengths: switch 2,300 iterations after eta_rew = 1, automatic; nothing stops on eval results; text below is history]: all of (1) last paper eval: flat, blocks >= 98 %,
   others >= 95 %; (2) probe finish (250-iteration mean) >= 0.90 on 4x4 and ramps; (3) `end/finished_phase099` gained
   < 1 point over the last 1,000 iterations; (4) iteration >= S2 + 4,000. Cap S2 + 7,000. The station-leg rule of 17a c
   (two consecutive pairs with all legs >= 45/50) cannot run on the node; the in-distribution rule replaces it in a single run.
f. **Stage-3 curriculum** [paper Table S4]: per env EMA s = 0.8 s + 0.2 I; promote on success with EMA > 0.5; demote when the
   closest qualifying approach (vertical error < 0.1 m) was > 1 m. Goal success (curriculum, replay admission) = within 0.5 m
   horizontally and 0.1 m vertically of the final waypoint [paper, `--s3-paper 1`]; waypoints and stations keep 0.2 m / 1 s [user rule].
g. **Fall-backs after any phase switch** (as `code/ops/watch_2h.sh`, base window = the first 100 rows AFTER the switch, not
   before it, 17f lesson) [ours]: at +150 and +400 iterations: no more than 3 trainer starts and a metrics row within 45 min;
   throughput >= 0.70 x base; `sim_fault_share` <= 0.05 (warn 0.02); mean finish on the sets common to both phases >= base -
   0.015; probe-gate mean >= base - 0.03; no common set > 0.04 under its own base. Fail: resume the previous phase from its
   last checkpoint and continue there.
h. **Crash / divergence** [ours]: supervisor resumes from `latest.pt`; on NaN / `diverged` resume from the previous numbered
   checkpoint (not automated today: open item 6).
i. **Checkpoint selection** [ours + user rule]: keep every numbered checkpoint of the run (no pruning: the retention rule
   pruned the 5,750 champion). Export every pair. On-node ranking = mean of per-set finish and probe finish over the 250
   iterations up to the checkpoint, plus the last paper eval. Off-node, after the run: station legs (250 trials) on every
   export from the top 10 by the on-node ranking; champion = highest leg sum; a difference counts only at >= 10 failures
   over the five legs (17b); the 5-trial hold-out gate is information only (user, 2026-11-13).
j. **Live eval budget** [user rule]: at most one paper eval per 1,000 iterations, right after a checkpoint.

## 7. Command line (stage 2 as it runs today, for reference)

`env X2_EXCLUDE_PAIRS=hip ACP_PLANT=mc_20261024 ACP_HF_FIX=1 ACP_PAPER_DR=1 python -m acp.train --data cfg/stage2h_sets.json
--run <run> --envs 36000 --iters <end> --njmax 450 --resume --stage 2 --init <ckpt> --planner-arch attn --u-obs 200
--u-rew 1700 --n-ann 3000 --eta-obs-max 0.8 --ramp-gate 2 --probe-envs 720 --gate-signal probe --ramp-gate-sets
tp_t4x4u_y20_hs,tp_ramp_sh2_hs,x2_loco_flat_train_hold,x2_loco_blocks_train_hold --freeze-planner
/mnt/filesystem-g4/tracks/acp/eval/pair_s2_v2_005750/tracker_ckpt_005750.pt --gate-backoff 1 --gate-rebase`
(launched through `code/supervise.sh` with `DATA=` and `EXTRA=`). Stage 3: `CK=<ckpt> code/ops/to_stage3.sh` (refuses until
`code/STAGE3_READY` exists; it does not exist on 2026-11-13).
**Since 2026-11-13 13:39 (s2_v5)** the stage-2 line is `runs/s2_v5/launch_flags.txt` (`--n-ann 6000 --eta-anchor-it 12604 --eta-obs0 0.781
--eta-rew0 0.281 --replay-balance set_clip --popt-reset --probe-envs 720 --gate-signal probe --ramp-gate 0`, no freeze, no cap); stage 3 is
started by `ops/auto_stage3.sh` with `ops/stage3_flags.env` (to_stage3.sh is superseded).

## 8. Open items that would change the recipe

1. **Stage 3 built to the paper on 2026-11-13 (`--s3-paper 1`, CPU tests pass, agentic/state/acp.md 17k); first run s3_v2 from 19,218.** Old text: (17d list: goal success / admission with 0.1 m vertical, 0.2 m cut,
   per-tile replay with work-per-metre eviction, balanced sampling, goal yaw +-0.4, phase 0, V_fs clipping off in warm-up,
   Table 3 stumbling / slippage formulas, logs_back family). `code/STAGE3_READY` absent.
2. **Planner co-training vs freeze**: the paper co-trains to eta 1; we freeze and cap at 0.8 (audit rank 1). The user moved
   co-training and eta to 1 to the stage-3 switch. Whether a from-scratch stage 2 with a late freeze (rule 6c) reaches the
   history's planner quality is untested.
3. Stage-3 config `stage3v2b_sets.json` holds the 2g reference sets, not 2h; it needs rebuilding on 2h.
4. Data 2h under paper DR: stairs sets dropped (gst 0.978 -> 0.882, stairs-turn probe 0.36) and old-set finish is flat at
   ~0.956 since 10,407. If this plateau holds, weights or DR scale may change.
5. Single seed (0); RNG state on resume not checked; no evidence of seed variance.
6. Divergence auto-rollback (6h) and the in-run stage switch (6a, 6e) exist only as separate scripts / human steps today;
   a single run needs them in one supervisor.
7. Height-field overflow on the ramp raster (~2.7 warning lines per iteration) and the v1 stair families falling 4/36 under
   the v2b config (17c) are unexplained.
8. Odometry delay not modelled (DN); MuJoCo Warp instead of Isaac Lab; 36,000 envs (7.5x fewer PPO updates per sample than
   the paper's stage-2 / 3 batch).
9. Multi-GPU: the trainer is single-GPU. A run "on more GPUs" needs data-parallel PPO (and a decision whether 36,000 per GPU
   or 36,000 total), which has never been tested.

## 9. Where today's run deviates from this recipe because of history

- Stage order: stage 1 (0-2,250) -> stage 2 (2,250-4,000) -> stage 1 again (4,000-5,500) -> stage 2 (5,500-). The recipe has
  one stage 1 and one stage 2.
- Body: iterations 0-4,000 trained on the false pelvis / hip-roll contact; recipe uses `X2_EXCLUDE_PAIRS=hip` from 0.
- Plant: 0-2,000 on `vendor_20260823` gains; recipe uses `mc_20261024` from 0.
- Data: stage1 -> 2 -> 2b -> 2c -> 2e -> 2f -> 2g -> 2h (seven switches); faulty stair descents (gs2dn_v1, legs turned 180
  degrees) trained 0-2,250; jump / crouch / injured / bump styles 0-4,000; arms into thighs 0-4,000; recipe uses 2h from 0.
- Physics: HF fix, simcheck and paper DR only since 10,356; before that foot boxes jammed in the ramp height field.
- Envs: 12,000 for 0-750; njmax 300 for 0-750.
- Planner: the frozen 5,750 planner = s2_v1 online planner (trained 2,250-4,000 on the old body, old styles) fine-tuned
  offline on s1_v2 4,500 rollouts (P4), then 250 iterations online. Recipe: online from scratch, freeze at S2 + 1,250.
- Ramp: two eta schedules (s2_start 2,250 then 5,500); eta_obs reached 1 in s2_v1, then capped at 0.8 from 5,500; back-off 3
  between 6,500 and 7,000 pushed eta_obs from 0.35 down to 0.15; eta_hold is 4,409 at 12,447 (eta_rew only 0.28).
- Tracker restarted from 7,000 at 7,907 (s2_v2 7,000-7,907 discarded) and s2_v3 10,356-10,428 discarded by the fall-back.
- Evaluations before 2026-11-12 02:20 ran with njmax 300 (overflow); these numbers are not comparable with later ones.
