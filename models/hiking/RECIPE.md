# Hiking recipe: the current best as one uninterrupted run from scratch

Status 2026-11-13 ~13:00 node clock. "Current best" = the configuration training now (v5b, `models/hiking/LEDGER.md` row 31),
written so that one run from iteration 0 gets every setting without a person. It has NOT been run from scratch: today's
policy reached it through 16 restarts (section 6). Apparatus success of every Hiking checkpoint so far: 0% (goal 99%), so this
is the best-known recipe, not a proven one. Open items in section 5 change it; the reference re-retarget (5.1) blocks a big run.

Tags: **[paper]**, **[fix]** (defect fixed), **[user rule]**, **[ours]**. Paths: code in this repo `models/hiking/x2_port/`
(node: `/mnt/filesystem-g4/tracks/hiking/code/x2_port/`, installed into InstinctLab by `apply.sh`; one source,
`files_instinctlab`); data on the shared filesystem only (no paper or BONES-SEED data in the repo).

## 1. Job

| Setting | Value | Tag |
|---|---|---|
| Code base | Hiking in the Wild release (InstinctLab + instinct_rl), patched files from `files_instinctlab` via `apply.sh`; per-run code snapshot | [paper] / [user rule] one code folder |
| Task | `Instinct-Parkour-ExtendedFixed-Amp-X2-v0` (`config/x2/extended.py`, `X2ParkourExtendedFixedEnvCfg`) from iteration 0 | [ours] (history: paper-only `...Fixed...` task until 4,500; see 6) |
| Command | `run_train.py --headless --task=<task> --num_envs 8192 --max_iterations 30000 agent.save_interval=125` (no `--resume`) | |
| `HIKING_ITER_OFFSET` | 0 (schedules read offset + steps / 24) | |
| Robots | 8,192 (one GPU). User approved 16,384 at the next planned restart (5.4) | [ours] (paper 4,096); [user rule] pending |
| Seed | 42 (agent and env), passed explicitly as `--seed=42` since v7 (supervisor `--extra=--seed=42`) | [paper] default |
| Sim | dt 0.005 s, decimation 4 (50 Hz policy), episode 20 s | [paper] |
| Iterations | 30,000 (paper G1 run length; ~9.8 s/it at 8,192 robots -> ~82 h on one GPU) | [paper] |
| Checkpoints | every 125 iterations, all kept (~68 MB each, ~16 GB) | [user rule] (every <= 30 min) |
| GPU | one training job per node; eval runs on CPU (MuJoCo) or beside training under the measured limit | [user rule] |

## 2. PPO and networks (`params/agent.yaml` of the live run)

WasabiPPO, `EncoderMoEActorCritic` (depth Conv2dHeadModel encoders), `init_noise_std` 1.0, `num_steps_per_env` 24 (196,608
samples / iteration), lr 1e-3 adaptive (`desired_kl` 0.01), AdamW, gamma 0.99, lam 0.95, clip 0.2, entropy 0.006, 5 epochs, 4
mini-batches, `max_grad_norm` 1.0, clipped value loss, value coef 1.0; discriminator reward coef 0.25, type quad, AdamW;
one discriminator. All [paper] (user 11-13: "Keep the paper's single judge").

## 3. Environment settings (env vars; every value is what the live trainer has, read from its environment)

Robot and port (all in code, no flag): pelvis root, 29 commanded joints, head joints held; implicit PD no delay, gains
`mc_20261024` [user rule]; default pose = reference mean (pelvis 0.679 m, spawn 0.75 m) [fix]; G1 per-joint-type action scale
[fix]; 25 filtered self-collision pairs (model `x2_ultra_hipexclude.xml`) [fix]; soft joint-limit factor 1.0 [fix];
`pelvis_orientation_l2` on `torso_link` [fix]; root-height termination 0.39 m [ours]. Camera: chest depth camera on
`torso_link` (0.131, 0, 0.259), 61.6 deg down, 87 x 58 deg, 36 x 32 after crop, 0-3 m, 8 frames [user rule];
`HIKING_X2_CAM_RAND=1` mount randomisation per reset (height -10.5/+5 cm, fwd +-2 cm, pitch +-4, roll/yaw +-2 deg) [user rule].

| Env var | Value | Meaning | Tag |
|---|---|---|---|
| `HIKING_X2_OPTS` | `a,c,d,f,r,s,t,k,y,o,z,w,g,v,j` | option letters (ledger Note 1) | see rows |
| `HIKING_X2_STAGE` | `500,1500` | early values held to 500, linear to final at 1,500 (a, d, s, f, r) | [ours] |
| `HIKING_X2_ALIVE_EARLY` | 0.3 | a: `is_alive` x0.3 early; `action_rate_l2`, `feet_slide`, `dof_acc_l2`, `feet_flat_ori` x0.2 early | [ours] |
| `HIKING_X2_WAIT_EARLY` | 2 | d: `dont_wait` x2 early; `stalled` reset (moving cmd, < 0.1 m/s after 4 s) until 1,500 | [ours] |
| `HIKING_X2_STYLE_EARLY` | 0.3 | s: style coef 0.075 early -> 0.25 at 1,500 | [ours] |
| `HIKING_X2_FALL_PENALTY` | 8 | f: -8 per fall (base_contact / bad_orientation / root_height), linear to 0 at 1,500 | [ours] |
| `HIKING_X2_REF_START` | 0.5 | r: share of resets from a reference frame, to 0 at 1,500 | [ours] |
| `HIKING_X2_AIR_TD_WEIGHT`, `_AIR_TD_CAP` | 140, `ref` | t: per touchdown clamp(air - 0.05, 0, cap) x 140 x dt, moving cmd only; cap = 1/f_ref(|v_cmd|) - 0.05, f_ref 1.4 / 1.9 / 2.0 steps/s at 0.3 / 0.5 / 0.7 m/s (`variants/staged.py.v6`). Live since v7 (11-13) | [ours] / [fix] (see 5.2) |
| `HIKING_X2_VX_MIN` | 0.45 | walking command floor | [paper] |
| `HIKING_X2_STAND_SHARE` | 0.10 | exact stand commands | [ours] (paper 0.05) |
| `HIKING_X2_TURN_SHARE` | 0.15 | y: turn-in-place periods (4-6 s, target >= 1.5 m away), then stand | [ours] |
| `HIKING_X2_OVERSPEED_WEIGHT` | -3.0 | o: x max(vx - cmd, 0) for fwd cmd > 0.1 | [ours] |
| `HIKING_X2_STOP_WEIGHT`, `HIKING_X2_STOP_ANY_YAW` | -2.0, 1 | z: planar speed whenever the linear cmd is 0 | [ours] |
| `HIKING_X2_ROUTE_SHARE` | 0.7 | w: robots on tile columns follow the map's checked routes | [ours] on [user rule] map |
| `HIKING_X2_SIDESTEP_SHARE`, `_SIDESTEP_V`, `_ROUTE_VY` | 0.3, 0.3, 0.2 | g: sideways route walking on tight spots / homes; lateral cmd near waypoints | [ours] |
| `HIKING_X2_SLOW_FWD_SHARE`, `_SLOW_TURN_SHARE`, `_SLOW_VX` | 0.25, 0.4, 0.1 | v: fwd cap U(0.1, 0.45) for 25%, yaw cap U(0.1, 0.4) for 40% | [ours] |
| `HIKING_X2_TURN_DRIFT_WEIGHT` | -2.0 | v: drift while turning; `turn_effort` replaces `dont_turn_wait` | [ours] |
| `HIKING_X2_VX_BACK` | 0 | no backward commands | [user rule] (11-12: no backward steps) |
| `HIKING_EXTENDED_TERRAINS` | `/mnt/filesystem-g4/tracks/hiking/data/extended_v4k` | user's extended map: extended_v1 + inset stairs + prop routes (725 routes in last check) | [user rule] |
| `HIKING_X2_MOTION_DIR`, `_MOTION_YAML` | `/mnt/filesystem-g4/tracks/hiking/data/motion_v7`, `parkour_motion_without_run_x2_v4.yaml` | paper reference retargeted to X2, v4 (torsoBase form read correctly; user approved 11-13), 18,982 frames, 50 fps; waist pitch at its +18 deg limit in 2,926 frames (stairs / blocks, accepted by the user); copy of `data/motions/hiking_release_x2/parkour_motion_without_run_x2_v4.npz` (CC BY-NC, shared filesystem only). Live since v7 (11-13) | [user rule] / [fix] |
| `HIKING_X2_LEVELS_OUT`, `_LEVELS_IN` | `<run>/_supervisor/terrain_levels.pt` | k: level save / restore | [fix] |
| `HIKING_X2_SIMFAULT` | 1 (default) | simfault: terminations ignore envs with a sim fault | [user rule] (no sim artefacts) |

Terms fixed in code (no flag): extended task adds `feet_stumble` -2.0, foot clearance -0.5, `wall_contact_graded` (trunk / legs
1 + min(F/100 N, 3) per body; arms free < 30 N and 0.5 s, then 0.25x), `base_contact` ends only for torso > 50 N mostly
vertical [ours]; j style weights legs / waist 1.0, shoulders / elbows 0.3, wrists / head 0 (550 discriminator inputs)
[user rule; user asked head 0.1, code has 0 - check]. Columns: 20 paper terrain columns, 10 extended tile columns, 2 velocity
columns (fwd 0..0.8 with `VX_BACK=0`, side +-0.3, yaw +-1).

## 4. Automatic rules that replace a human

| Rule | Condition (code-checkable) | Action | Measure, file |
|---|---|---|---|
| R1 stall | no trainer output for 900 s | kill, resume newest checkpoint | `supervise_x2.py` / `v5/supervise_x2_v5.py --stall_s 900` |
| R2 crash | trainer exits with error twice from the same checkpoint | stop; resume the previous checkpoint once with the same config; if it fails again, stop and alert (history: fell back to an older config) | `supervisor.log` "run ended: exit", `crashguard_v4.sh` |
| R3 levels across a resume | always | levels saved every 300 s; restored at start; curriculum moves undone for 1,500 control steps (`HIKING_X2_LEVELS_PIN_STEPS`) | `keep.py` level_keeper, `Curriculum/terrain_levels` |
| R4 after-resume health | 80 iterations after any resume: level >= 0.8 x and episode >= 0.5 x the pre-stop means | flag in `gates.txt`; if flagged twice, alert | `crashguard_v4.sh` (args = pre-stop level, episode) |
| R5 sim fault | per-body velocity jump > 20 between steps or speed > 50 (`sim_fault` defaults) | episode not counted as a policy failure; share in heartbeat | `simfault.py`, `simcheck` alarm |
| R6 pre-flight (before GPU start and before any resume with changed code) | CPU checks pass: option letters allowed, motion reader (joint names, shapes, quaternions), terrain / route files, final cfg term signatures (42 terms), installed tree == source | do not start | `v6/precheck_v6.py`, `v5b/cfgscan.py`, `apply.sh` tree check |
| R7 construction check | env builds, routes >= 100, required terms in reward table | do not start | `check_x2_env.py`, `launch.txt` |
| R8 export | every new checkpoint | ONNX + card + `run_params/` to `eval/x2_parkour_v4_<N>/`; failure logged and alerted | `auto_export_v4.sh`, `export_for_eval.py` |
| R9 scoring | every exported checkpoint, newest first | flat gate (walk, stop-stand, turn in place, turn-and-return, 50 each), apparatus legs 1-5, product | `eval/autoscore/autoscore.py`, `eval/ladder.py`, `docs/EVAL.md` -> `eval/autoscore/results.csv` |
| R10 checkpoint selection [ours, proposed] | best = highest apparatus product; ties: sum of legs, then flat gate count; a checkpoint with any flat test < 45/50 is not selectable; keep the frozen reference (best so far) and re-score it beside each candidate | hand-over to the user's review | `results.csv` |
| R11 gait watch [ours, proposed] | every 500 iterations: cadence within 30% of reference cadence at 0.3 / 0.5 / 0.7 m/s, torso pitch within 10 deg of reference | alert (no automatic change) | `eval/gait_stats.py`, `docs/findings/gait_vs_reference.md` method |
| R12 progress watch [ours, proposed] | mean terrain level and route arrival flat for 2,000 iterations after 5,000 | alert | `metrics.jsonl` |

Gates used in history only at restarts (trial 75 iterations, marks over the last 30: level >= 0.8 x, episode >= 0.4 x, turn
rate >= 0.5 x; `continue_v6.sh`) are not needed in an uninterrupted run; keep them for any forced resume with a config change.

## 5. Open items that would change the recipe

0. **v9 (armed 2026-11-14, ledger row 35; user decisions "Hiking v9 decisions").** If its trial passes, the recipe changes to:
   `HIKING_X2_OPTS` without `t` (no touchdown term; `HIKING_X2_AIR_TD_*` unused) [paper]; `HIKING_X2_STAND_SHARE` 0.05 [paper];
   `HIKING_X2_TURN_SHARE` 0, `HIKING_X2_SLOW_FWD_SHARE` 0, `HIKING_X2_SLOW_TURN_SHARE` 0 [paper]; `HIKING_X2_V9=1`,
   `HIKING_X2_PAPER_STAND=1` (Rough (Stand) column vx 0 / vy 0 / wz +-1, per-type curriculum log) [paper]; FAIR kept [fix];
   `HIKING_X2_STUMBLE_SCALE` / `_CONTACT_SCALE` 1.0 (flag for a pending user decision) [user rule]. Code `code/x2_port/v9/`.
   Update section 3 when the trial line "v9 is training" is in launch.txt and the first +500 rows are scored.

1. **Motion reference v4: DONE 11-13 (v7, `motion_v7`).** Was blocking: v1-v3 retargets of the paper's reference read the torsoBase source wrongly (waist bend
   applied twice, pelvis ~4.4 cm high, torso leans back on terrain). Re-retarget v4 from the correctly converted source is in
   progress (`agentic/state/STATUS.md` 13:25). The recipe's `HIKING_X2_MOTION_DIR` / `_YAML` change to v4 once it passes per-category
   checks and the user's G1-vs-X2 review video. Do not start a big run before. [fix]
2. **Option t form.** t at weight 140 with a fixed 0.30 s cap pays most at ~2.9 steps/s at every speed (gait finding). Prepared:
   `HIKING_X2_AIR_TD_CAP=ref` (cap = 1/f_ref(|v|) - 0.05, f_ref 1.4 / 1.9 / 2.0 steps/s at 0.3 / 0.5 / 0.7 m/s), CPU-tested,
   in training since v7 (11-13). Alternatives in the finding: weight 0-10 (paper), per-second pay. [fix]
3. **Order of additions from scratch.** Every option after a,c,d,f,r,s was added to a policy that already walked. Untested:
   t, y, z, o, w, g, v, j and the extended map from iteration 0. Either test a from-scratch run to 2,000 first, or add
   iteration schedules (e.g. t from 1,000, extended columns / routes from ~4,500) in code so one run can replay the order.
4. **Robots 16,384** (user approved for the next planned restart). Untested: PhysX contact drops were seen at 12,288 with the
   task's buffer sizes; needs larger PhysX buffers and a construction check. Changes samples / iteration and wall time.
5. Style weights: user asked wrists and head 0.1, code has them out (0) - confirm.
6. Camera field / pitch (user decision, `agentic/docs/USER_TODO.md` 5): torso lean was partly attributed to buying view range; with the
   reference fixed (5.1) recheck the lean before changing the camera.
7. Shoulder roll sits at its -0.061 rad limit in every reference frame (v2, v3): open in the retarget.
8. Stations next to drops and in slots (ACP's fix) and gap widths scaled to the X2 leg: planned, not built.
9. Paper's specialists (stairs, platforms, 40,000 more iterations each): not planned yet.

7. **v8 candidate (armed 11-14, not in the recipe until kept).** Curriculum: an episode with any stand, turn-in-place or slow-forward command is neither promoted nor demoted; all other episodes use the paper's `tracking_exp_vel` (0.3 / 0.6) unchanged (`v8/fair.py`, `HIKING_X2_FAIR=1`) [fix]. Option t pays touchdowns x forward progress under walking commands (`HIKING_X2_AIR_TD_PROGRESS=1`) [ours]. From scratch these belong from iteration 0 (with slow / turn / stand commands present, the paper's promotion rule cannot see refusals). Evidence and numbers: ledger row 34, agentic/state/hiking.md 2026-11-14.

## 6. Where today's run deviates from this recipe because of history

Today's policy (`x2_parkour_v4`, v5b) came from scratch at 03:19 on 11-12 with:
- Iterations 0-4,500 on the paper-only `...Fixed-Amp-X2-v0` task; extended map from 4,500, routes from 5,000, inset stairs and
  graded contact from 6,000, prop routes, slow commands and no-backward from 7,500.
- t from 1,000 (save interval 250 before 1,000); walking floor 0.6 m/s until 1,750; no exact stands until 1,875 (option c
  alone); y and k from 1,750; o from 3,250; z (any yaw) from 3,875; j and simfault from 7,500.
- Motion: BONES-SEED + Pico mix v2 (400 s) to 6,000, plus Pico sideways / turning stairs (v3) to 7,500, paper reference v2
  since 7,500. The discriminator was re-initialised at 7,500 (input size 330 -> 550); policy, value and optimiser kept.
- 16 restarts in run 4 (ledger rows 10-31), each restarting the adaptive LR at 1e-5 and resetting all robots together; levels
  went to 0 at 1,000 and 1,750, dropped at 2,750; pinned from 3,250 on. Each restart left the GPU idle about 6-24 min (`launch.txt`
  "GPU idle about N s"), plus the failed attempts and the 12:12-12:54 crash loop.
- Schedules (a, d, s, f, r) ran on the iteration counter; resumed segments passed `HIKING_ITER_OFFSET` = start checkpoint.
- Run 4's `max_iterations` is 22,500 from 7,500 (30,000 total), same end as the recipe.
- v7 (11-13 13:32, from model_11125): motion v2 -> v4, t cap -> `ref`, explicit seed 42; discriminator, policy, value and
  optimiser carried over (same 550 inputs). The last planned restart: the run goes to 30,000 with no planned stops (user 11-13).
