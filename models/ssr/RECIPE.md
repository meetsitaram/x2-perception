# SSR recipe: the current best line as one uninterrupted run from scratch

Written 2026-11-13 (node clock 12:58) from `models/ssr/LEDGER.md` and its sources. "Current best" = the **v14 line** (run `r7`,
flags `/mnt/filesystem-g4/tracks/ssr/runs/r7_flags_v14.txt`, effective values from `runs/r7/args.json`), the only SSR line that
kept walking. The two paper-aligned attempts (v15 = r8, v16 = r9) fell back to a shuffle and are listed under open items.
r7 reached v14 by 30 resumes with flags added along the way; this file states the same content as an iteration schedule.
Tags: [paper], [fix], [user rule], [ours]. Apparatus success of every SSR checkpoint so far is 0 % (goal 99 %), so this is the
best *recipe we have*, not one that meets the goal.

## 1. Launch

```
cd /mnt/filesystem-g4/tracks/ssr/code
env X2_HF_CHUNK=400 ../env/venv/bin/python scripts/supervise_node.py --name <run> --iters 20000 --num_envs 8192 -- <flags of section 3>
```
- One code folder `models/ssr/code/` (user rule 2026-11-13; v15/v16 behind `--paper_terrain --simcheck`, defaults = v14,
  bit-identical check in `runs/merge/equiv/`). Snapshot the code folder per run (user rule: running jobs protected by a per-run
  code snapshot).
- Environment: Python venv `tracks/ssr/env/venv` (torch 2.11.0+cu128, mujoco / mujoco-warp 3.17.0, warp-lang 1.18.0).
- Env vars: `X2_HF_CHUNK=400` (height field split into <= 400 x 400 fields; default in `ssr/x2.py` is now 400) [fix, FINDINGS B10].
  For CPU probes also `X2_PLANT=mc_20261024 X2_EXCL_PELVIS_HIPROLL=1`. Not used on the v14 line: `X2_TERRAIN_SOLREF`,
  `X2_TERRAIN_PAPER` (paper mode only).
- Supervisor `scripts/supervise_node.py`: resumes from `last.pt` after a crash (restart limit 200), heartbeat
  `/mnt/filesystem-g4/status/ssr.json`, STOP file `runs/STOP`. Checkpoints: `last.pt` + `prev.pt` every 50 iterations
  (`--save_every 50`), `ckpt_NNNNNN.pt` every 1,000 (`--ckpt_every 1000`); log every 10 (`runs/<run>/log.jsonl`).

## 2. Compute

| Setting | Value | Tag |
|---|---|---|
| GPU | one GPU per node, one GPU job per node (evals beside training only through `scripts/eval_guard.sh`, <= 256 robots) | user rule |
| Robots | 8,192 (paper 4,096). User decided 2026-11-13: 16,384 at the next planned restart (not tested on this recipe) | ours / user rule |
| Throughput | about 6.5-6.8 s per iteration, about 30k samples/s at 8,192 (r7 v14 log; GPU 24-34 GB). 20,000 iterations is about 37 h. At 16,384 the 2026-11-11 bench gave 2.65x the iteration time for +51 % samples/s (head camera; chest_rs costs 9 %, physical walls unmeasured on GPU) | measured |
| PPO rollout | 24 steps, 5 epochs, 4 mini-batches (each mini-batch 2x the paper's at 8,192) | paper |
| Seeds | `--seed 0` (terrain); `--torch_seed 20261113` (torch / numpy / random, from r7 11,351 on; before: unseeded) | fix |
| Checkpoint state | from v17 code: `ext_level` / `ext_type` per robot saved and restored (extended-map levels were reset to 0 at every resume before) | fix |

## 3. Settings (effective values of the v14 line)

### Robot, sim, observation
| Flag / setting | Value | Tag |
|---|---|---|
| `--plant` | `mc_20261024` (`config/robot_plant/x2_ultra_mc_20261024.yaml`, same as `/mnt/filesystem-g4/data/robots/x2/`) | user rule |
| Action scale | plant rule 0.25 x effort / kp (0.26 / 0.26 / 0.49 rad per unit); `--action_scale_paper` off (parked, REVIEW_LATER) | user rule |
| `--excl_pelvis_hiproll` | on (pelvis vs hip-roll hull contacts excluded) | fix |
| `--camera` | `chest_rs`: torso_link (0.126, 0, 0.16), 59.5 deg down, 87 x 58 deg, 36 x 36, 0.1-3.0 m, mount randomised per episode (fwd +-2 cm, lat +-1, height +-5, pitch +-4 deg, roll/yaw +-2, FoV +-1) | user rule |
| Physics | MuJoCo Warp, 200 Hz physics, 50 Hz policy, implicitfast, sole spheres r 2 cm on a 2.5 cm height field, `nconmax` >= 30 with walls (150 default) | ours (sim choice) |
| Self-occlusion | 20 limb spheres, N(0, 0.1 m) noise on occluded pixels | ours / paper (noise) |
| History, maps | history 5, base height map 17 x 10 (paper 18 x 10; user: "Don't change for now") | ours / user rule |

### Rewards
| Item | Value | Tag |
|---|---|---|
| Groups | locomotion 1.0 / foothold 0.25 / style 0.2 (`--w_style 0.2`), each with its own critic | paper |
| `--rew_scale` | 0.02 (x dt) | ours |
| `--adv_norm` | joint (paper: per group, "separate"), `--adv_floor 0.002` | ours (DEV-P) |
| Table 6 terms | lin_vel 1.0, ang_vel 0.8, orientation 0.5, ang_vel_xy 0.25, base_height 0.4 (0.68 m), action_rate -0.12, smoothness -0.06, dof_vel -0.96, dof_torque -0.6, dof_dev -1.8, limits -1.5/-6/-6, stand_still -0.12, single_support 0.2, impact -1.3, slip -0.2, air_time -2.0, stumble -2.0, lateral 0.08 | paper |
| Velocity kernel sigma | 0.25 until 600, ramp to 0.1 by 900 (`--sig_v_to 0.1 --sig_v_from 600 --sig_v_ramp 300`) | ours (DEV-P) |
| Swing height | `--swing_h_w 0.5 --swing_h_target 0.10` (sum over swing feet of min(h/0.10, 1) x min(|v_xy|, 1), moving commands) | ours |
| Creep | `--creep_w 0.5` (share of command made good, 0.10-0.35 m/s commands) | ours |
| Drift | `--drift_w 0.5` (planar speed under commands with |v| <= 0.15) | ours |
| Backward | `--back_scale 0.5` (lin_vel x 0.5 while vx < -0.05; no backward commands on slope tiles) | user rule |
| Side graze | `--side_soft 2.0` (0.06 m band = -2.0 per step; only limb overlap ends) | ours |
| Physical walls (ext routes) | `--ext_phys 24`: forearm/hand free to 40 N and 0.5 s then -0.5/step; other bodies -2 x (1 + F/100 N, cap 5)/step; end only > 400 N | user rule |
| Off on this line | feet-air reward (`--feet_air_w 0`), relative kernel, progress gate, stuck termination, gate prior, imitation | - |

### Commands, starts, curriculum
| Flag | Value | Tag |
|---|---|---|
| Tile commands | forward U(0, 1.2) m/s (slopes U(-1, 1) but no backward on slopes), heading per type; vy clamp +-0.4 (paper: none) | paper / ours |
| `--omni_frac 0.3 --omni_turn 0.5 --omni_slow 0.3` | 30 % omnidirectional robots from the start on the open flat tile; of them 50 % turn in place (|wz| 0.3-1.2), 30 % of the rest slow forward 0.1-0.4 (paper: omni only after the top level) | ours |
| `--resample_s 8` | new command every 8 s (paper: one per episode) | ours |
| `--edge_stop 0.3` | 30 % of tile episodes: zero command for 2 s at the first down edge >= 0.15 m, 0.45 m ahead | ours |
| `--rsi_frac 0.5 --rsi_min_speed 0.15` | half the walking episodes start from captured frames moving >= 0.15 m/s | ours |
| `--init_q_noise` | 0.05 | ours |
| Curriculum | 20 levels, promote at the tile end (paper: past half the tile), demote as legged_gym; foothold guidance latched on when mean level > 2 | paper / ours |
| Terrain | ramp, stairs, beam, platform, flat lane, side walls / posts / door frames (vision raster); no gap / discrete (paper) | ours |
| Extended map | `--ext_frac 0.4 --ext_variants 2 --ext_types tight_spots,log_corridors,homes,street_front,stairs,wedges,single_beams,beam_fields,blocks --ext_stairs_inset --ext_prop_routes --ext_stations 1 --ext_slope_turns 2 --ext_hold_r 0.10 --ext_phys 24`; route robots follow the eval harness law (yaw 1.5 x error cap 1.0, forward clip(d, 0.15, vmax) x cos(err), vmax U(0.4, 1.0)), hold within 0.10 m, sidestep law on segments < 0.39 m | user rule (map) / ours (shapes) |

Data:
- Terrain: `/mnt/filesystem-g4/data/terrains/extended_v1/`, `extended_v1_2_stairs_inset/`, `extended_v1_1_prop_routes/` (paths
  in `ssr/terrain.py` lines 160-162); paper-style tiles generated from `ssr/terrain.py` with seed 0. No apparatus layout or route.
- Motion prior (from r7 11,351 on, 2026-11-13, v17): `models/ssr/code/data/amp_v5.npz`, built by `scripts/build_amp_v5.py` (CPU, log
  `runs/amp_v5_build.log`, sizes in `data/amp_v5_meta.json`) from the paper reference **v4** (`hiking_release_x2/paper_ref_x2_v4.pkl`,
  categories by the v4 time ranges, checked 8/9/14/17 clips = 24.0/16.8/9.3/9.4 s) + the user's Pico takes; no BONES-SEED [user rule].
  Seconds (unique, mirrored on top): ramp 261.3 (fwd 78.4, back 0 -> turn 52.3, ramp 130.7), stairs 295.8, beam 174.3 (gap slot;
  Pico logs 59.9 + 2x2 13.7 + 4x4 13.7 traversal), platform 181.6, flat 136.0 (discrete slot's size; fwd 68.0 / turn 47.6 / side 20.4)
  [ours: v14 layout has a flat group, the paper a discrete one]. `--disc_part_w 1.0,0.3,0.0` (legs/pelvis/waist, shoulders/elbows,
  wrists; no head feature) instead of `--disc_mask_arms` [user rule]. Window flags unchanged: `--prior_min_speed 0.05 --prior_turn_min 0.3
  --prior_balance`. Route types map to a group (wedges -> ramp, beams -> beam, blocks -> platform, tight_spots/homes -> beam).
  Before 11,351: `amp_v4.npz` (BONES-SEED-heavy amp_v2 flat + terrain-park/Pico stairs, beam, ramp) with `--disc_mask_arms`.

### PPO / networks
| Setting | Value | Tag |
|---|---|---|
| gamma / lambda / clip | 0.99 / 0.95 / 0.2 | paper / ours (clip silent) |
| entropy, value coef | 0.005, 1.0 | paper |
| lr | 5e-4 Adam x 4 optimisers, KL-adaptive (desired 0.01), `--fixed_lr 0` | paper |
| action std | init 1.0 (`--init_std`), min 0.2 | ours / paper |
| arm action std (pending, not yet kept) | v18 (armed 2026-11-13, not started): arm std reset to 0.3 once, capped at 0.4 (`--arm_std_reset 0.3 --arm_std_cap 0.4`). A from-scratch run with arms visible to D from the start (paper) may not need it; with the history of masked arms it does (FINDINGS E7) | fix |
| arm action std 0.2 (pending, not yet kept) | v19 (armed 2026-11-14 for 20,000-30,000): `--arm_std_reset 0.2 --arm_std_cap 0.2`, the paper's minimum std (Table 9) as the arm std. Arm action-bound loss (AMP/ASE-style squared excess of the action means outside the joint range; the SSR paper is silent on action bounds, and env.step already clips the PD target to the joint range, which is why a mean past a stop is free) is in train.py default-off: for a from-scratch run it belongs in the recipe from iteration 0 (`--arm_bound_coef 1.0 --arm_bound_margin 0.05`), untested from scratch; on the warm r7 line it broke walking (FINDINGS E7b, agentic/state/ssr.md 2026-11-14) | paper / ours |
| Networks | mirror-equivariant encoder (depth CNN + MLP -> GRU 256), estimator, 5-expert MoE actor, 3 critics, foothold model, 5 discriminators (GP 20, wd 1e-4), mirrored samples in every update | paper |
| Guards | actor mean 8 tanh(x/8), log-ratio clamp, non-finite updates skipped | ours |

## 4. Schedule (iteration numbers; one run)

Built-in schedules (already iteration-driven in the code, work from scratch):

| Iteration | What | Flag | Tag |
|---|---|---|---|
| 0-150 | only tracking + posture terms (lin_vel, ang_vel, orientation, ang_vel_xy, base_height, lateral, dof_dev, stand_still) | `--stage_a 150` | ours |
| 150-400 | remaining Table 6 terms ramp linearly to full | `--stage_b 400` | ours |
| 0-550 | every tile robot on the open flat type | `--flat_until 550` | ours |
| 550-1,050 | obstacle-type share 25 % -> 80 % (robots switch at episode end, level 0) | `--obst0 0.25 --obst_ramp 500` | ours |
| 600-900 | velocity kernel sigma 0.25 -> 0.1 | `--sig_v_*` | ours |
| 1,000-1,500 | style weight 0 -> 0.2 | `--style_from 1000 --gate_ramp 500 --style_low 0` | ours |
| when mean level > 2 | foothold guidance on for good | built-in | paper |

Items r7 switched on later by resume (there is **no per-flag schedule in the code**; see section 6). Iteration at which r7 got
them, which is the tested order:

| r7 iteration | Item added | r7 version |
|---|---|---|
| 1,150 | command mix + 8 s resampling | v3 |
| 1,750 | swing-height reward | v4 |
| 2,100 | omni 0.3 / turn 0.5 | v5 |
| 2,850 | edge stops | v7 |
| 2,950 | prior slow and turn windows (0.05 / 0.3) | v8 |
| 3,250 | side graze band | v9 |
| 3,550 | extended map, 3 types, routes >= 0.39 m clear | v11 |
| 3,800 | physical walls; the 5 original ext types again (tight_spots, log_corridors, homes, street_front, stairs) | v12 |
| 4,800 | amp_v4, 9 ext types, ext_frac 0.4, inset stairs, prop routes, wall stations, slope turns, back_scale, drift, creep, hold 0.10 | v13 + v14 |
| 5,550 | height-field split (in a new run: from 0, it is a bug fix) | env |

Recommended single-run form: implement a `--<flag>_from N` schedule (or one `--schedule` file) and use the r7 iterations above,
rounded: commands 1,150; swing 1,750; edge 2,850; prior windows 2,950; side 3,250; ext map 3,550 (physical walls and all nine
types at once, ext_frac 0.4); amp_v4 from 0 (it only adds clips to three groups; untested from 0); v14 terms at 4,800;
`X2_HF_CHUNK=400` from 0. Each scheduled switch should start a ramp, not a step, to avoid the resume-wave seen in r7 (N8 of the
ledger). This is untested as one run.

## 5. Automatic rules that replace a human decision

Measures and the files that compute them (all under `models/ssr/code/scripts/` unless said):
- Training log (`ssr/env.py`, `ssr/train.py`, every 10 its): `gait.vx_ratio` (signed world-x speed / command, walkers commanded
  >= 0.3 m/s, omni robots excluded), `done_share` (tilt, diverged, ...), `ep_len_s`, `samples_per_s`, `level`. Read with `w.py` /
  `d.py`. **Never use `vx_ratio` alone for keep / fall-back with 8 s resampling** (rule 2026-11-12; it also scores a toppling robot
  as speed).
- Deterministic CPU probes on a checkpoint (base and candidate in the same job, same seeds): `probe_start.py` (64 robots, 1 s
  stand then v; started = mean speed of last 4 s of 8 s >= 0.5 v), `probe_turn.py` / `probe_turn2.py` (yaw from a stand),
  `probe_gait.py` (speed, survival, single support, air, swing clearance, step length per command), `probe_step.py` (step-downs),
  `probe_wall2.py` (stalled share beside a wall at 0.40 / 0.45 m), `probe_approach.py` / `probe_stop.py` (harness-law approach,
  final distance). Outputs `tracks/ssr/eval/start|turn|speed_bins|post/*.json`.
- Queue scripts (`runs/queue15.sh` .. `queue34.sh`) apply the rules below; verdicts in `runs/queue*.log` and `runs/*.verdict`.

Rules (code-checkable):
1. **Crash**: supervisor resumes from `last.pt`; `runs/crash_guard.sh crash_loop`: more than 3 `crash(..) at iter None` lines
   since the start -> kill only that run, fall back. (T2 looped 92 times without it.)
2. **Physics tripwire**: from +80 iterations after any switch, mean of the last 10 log lines of done-share tilt + diverged >
   base (mean of the last 20 lines before the switch) + 0.08 -> fall back. (The older absolute 0.15 misfired at v13.)
3. **Throughput**: samples/s < 0.8 x base (60 % for the extended-map switch; 18,000 absolute in v13) -> fall back.
4. **Relative marks at +100 and +250 after a switch** (probes, base = checkpoint before the switch; a mark is used only for a
   property the base passes): start share at 0.35 / 0.5 / 0.6 m/s >= base - 0.15; yaw from a stand at +-0.8 rad/s >= 0.4 both sides;
   swing clearance at 0.5 m/s >= base - 0.015 m; survival >= base - 0.1; wall stall at 0.40 / 0.45 m <= base + 0.15. Not used:
   0.8 / 1.0 m/s starts (flaky on the base), training speed share.
5. **Paper-transition gates** (queue32/33, for v15/v16): +60 samples/s < 0.8x or tilt > base + 0.05; +350 and +500 ep_len < 0.7 x
   base or vx_ratio (mean of 10 lines) < 0.75 x base -> fall back.
6. **Walking** is decided only by the deterministic probe: survival >= 90 %, signed speed >= 50 % of command at 0.5 m/s, single
   support >= 0.5.
7. **Milestone gates** (r7, report not fall-back): by 550 walks at 0.5 / 0.6 (>= 70 %, survival >= 90 %), zero command stands;
   by 1,500 slow 0.2 / 0.35 >= 70 % and yaw +-0.4 >= 0.3, +-0.8 >= 0.6 from a stand; by 2,000 swing clearance median >= 0.10 m,
   stairs and ramp level >= 4, beam >= 6. A gate missed by more than half -> diagnosis before a new flag.
8. **Exports**: `scripts/export_eval.py` (ONNX + `adapter_card.json` + `obs_check.npz`, max torch/ONNX error < 1e-4) every 250
   iterations at probe points; `runs/export_dryrun.sh` must print `EXPORT_DRYRUN PASS` before any launch.
9. **Pre-flight before any launch or restart** (user rule): shared physics pre-flight (`tools/simcheck`), equivariance and mirror
   tests (`tests/test_equivariance.py`, `tests/test_mirror.py`), env smoke 17/17, a 2-iteration CPU train smoke of the exact
   command (`CUDA_VISIBLE_DEVICES= ... --iters 2 --num_envs 64`).
10. **Checkpoint selection for the eval**: the newest checkpoint that passed rule 4 against its predecessor and has the lowest
    wall stall; the autoscorer (`eval/autoscore/`) then ranks by the station legs (250 trials). On an uninterrupted run there is
    no fall-back, so rules 2-4 must become selection filters over the 250-iteration exports instead.

## 6. Where today's run deviates from this recipe because of history

- r7 is one log but 30 segments: every switch reset all episodes at once (resume wave: tilt spikes, ledger N8) and the
  flags were added in the order problems were found, not designed.
- The misplaced-route and impassable-route segments (3,560-4,220) were discarded by restarting from 3,550, so they are not in
  today's weights; but r7 trained 0-5,550 with the single large height field (10.8 % of ends were sim blow-ups from v12 on, likely
  teaching caution on edges), which a new run avoids by setting `X2_HF_CHUNK=400` from 0.
- r7 ran 0-4,800 on amp_v2 and switched to amp_v4 at 4,800; its flat group is still the BONES-SEED-heavy amp_v2 flat set.
- Duplicate flags in the file (`--omni_frac`, `--omni_turn`, `--prior_min_speed`, `--ext_frac` twice; argparse takes the last).
- Discarded / never-kept tries still present in the weights' history only as fall-backs (rel kernel, edge 1, v13, v15, v16, T1).

## 7. Open items that would change the recipe

1. **User rule "SSR back to the paper" (2026-11-13) is not met.** The paper-aligned flag set (`runs/r7_flags_v15.txt`: amp_v5
   Table 11 prior, separate advantage normalisation, sigma 0.25, single stage, omni only after the top level, promote at half the
   tile, no vy clamp, slope backward, base-frame foothold targets, paper terrain with discrete, `--disc_part_w 1.0,0.3,0.0`, no
   swing_h / prior_balance / resampling / walking starts) fell back twice to a shuffle (v15 at +150, v16 at +350). T1 ruled out
   normalisation and sigma; whether fresh amp_v5 discriminators alone cause it (T2) is unanswered (crash loop). The paper items
   are therefore not in this recipe.
0. **Done 2026-11-13 (v17, r7 11,351)**: items 2-4 below are applied on the v14 line (amp_v5 rebuilt from v4, no BONES-SEED, weighted
   style input). Items 1, 5-8 stay open; stairs / platform `levels_by_type` mix route robots' stale tile levels (v14 layout has no
   `--ext_home`), so they are not a clean stairs measure (agentic/state/ssr.md, 2026-11-13 13:39 entry).
2. **(old) amp_v5 was defective**: the paper reference retarget v1-v3 has the torso / waist read twice (pelvis 4.4 cm high, torso leaning
   back; STATUS 2026-11-13 13:25). v15/v16 trained on it. A v4 re-retarget is in progress; any paper-aligned retry must wait for it.
3. **BONES-SEED**: user 2026-11-13 "completely ignore the boneseed motion set" for SSR style; amp_v4 still contains it (flat group
   and parts of others). Following the rule means a new prior (Pico + paper reference v4) = a prior change.
4. **Discriminator input weighting** (user rule): legs / pelvis / waist 1.0, shoulders / elbows 0.3, wrists / head 0; this recipe
   still masks arms (`--disc_mask_arms`).
5. **Gait quality**: r7_v14 steps at 3.5-5.1 steps/s vs the reference 1.3-1.5, steps a third as long; ranked fixes in
   `docs/findings/gait_vs_reference.md` (separate norm, sigma 0.25, drop swing_h and creep, unmask arms) are the same items that
   failed as a package in v15/v16.
6. **16,384 robots** (user decision) at the next restart: changes mini-batch size and iteration time; untested.
7. Paper deviations still open: base height map 18 x 10 (deferred by the user), action scales (parked), gap terrain, paper
   reset / promotion rules; walls as MuJoCo boxes only on route robots; foothold base frame.
8. Seeding: torch unseeded.
9. **2026-11-14 (E8b)**: v20 curriculum (`--cur_rudin`, climb-tile side no-term, climb creep) and v21 paper items (separate norm, sigma 0.25,
   no vy clamp, base-frame foothold, omni after the top level, Table 5 omni draw, beam heading +-15) are on trial in r7; they enter this
   recipe once v21 passes its +500 gate (queue39). The paper terrain set (D1) is for the from-scratch run
   (`docs/findings/ssr_paper_vs_code_status_20261114.md`).
