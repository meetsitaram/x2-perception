# Unapplied paper deviations, ACP and Hiking (2026-11-14)

Asked by the user's rule "Fix every unapplied paper deviation" (USER_RULES, 2026-11-14, after SSR E8a). Read-only check, no run
or node touched. Sources: `docs/findings/acp_paper_vs_code.md` (audit 2026-11-13), `agentic/state/acp.md` 17j-17m, `models/acp/LEDGER.md`
rows 15-20, `models/acp/RECIPE.md`; for Hiking `models/hiking/RECIPE.md`, `models/hiking/LEDGER.md` rows 32-34c,
`agentic/state/hiking.md` (2026-11-14 entries), `docs/FINDINGS.md` H7-H8, the X2 port configs in `models/hiking/x2_port/files_instinctlab/`,
the user's G1 audit `hiking-in-the-wild/docs/PAPER_VS_CODE.md` and the paper text `models/hiking/x2_port/paper_v2.txt`
(arXiv 2601.07718 v2).

**Caveat (affects both tracks):** the repo copies `models/acp/code` and `models/hiking/x2_port` were last synced 2026-11-12 21:07
PST (11-13 05:07 UTC). They hold neither ACP's `ACP_PAPER_DR` / `--s3-paper` code nor Hiking v6-v8. Everything after that date
is taken from the state files and ledgers, not from the code. `agentic/scripts/sync_track_code.sh --commit` has not brought those changes
in since then (that is a coordinator item).

## ACP

### Counts
- **DN (25):** user override 8; applied since 15; still deviating 2 (odometry delay, which is partly applied, and the MLP baseline, which is never used).
- **A (18):** user rule or fix, kept 4; retired or moot now 6; still ours and active 8.
- Since the audit, items added that the paper does not cover: goal share 0.5, slippage foot speed taken as horizontal, per-tile replay
  share = capacity / tiles present, balanced replay in stage 2 (the paper's principle applied to reference sets) (17k).

### DN rows
| # | Audit row | Status | Evidence |
|---|---|---|---|
| 1 | Skills (no ladder, vault, sit, get-up, sparse) | user override (data choices; "no ladders, vaults") | USER_RULES 2026-11-13 |
| 2 | Data length 196.7 min vs 13.2 | user override (data choices; BONES-SEED stays for ACP) | USER_RULES 2026-11-13 |
| 3 | Stumbling formula | **applied** s3_v2 (`--s3-paper 1`): sum over feet 1[F_xy > 1.1 F_z + 1] | 17k, ledger 19 |
| 4 | Slippage formula | **applied** s3_v2: sum over feet 1[contact > 0.026 s] 1[v_f > 0.5] | 17k |
| 5 | MLP baseline input | still deviating; baseline is never used, no impact | - |
| 6 | Stage-1 envs 36,000 | user override (larger env counts) | USER_RULES item 18-21 |
| 7 | Planner frozen in stage 2 | **applied** s2_v5 12,604: co-trained, 10 x 4,096 per iteration | 17j-17k, ledger 18 |
| 8 | eta_obs ceiling 0.8 + hold | **applied** s2_v5: Eq. 3 to 1 on the fixed schedule, no gate | 17k |
| 9 | Stage-2 envs / minibatch | user override | USER_RULES |
| 10 | Goal terrains (32 configs) | user override (own families + extended map) | USER_RULES 2026-11-13 |
| 11 | Stage-3 initial phase 0 | **applied** (all envs) | 17k B1 |
| 12 | Goal reset yaw +-0.4 | **applied** | 17k B2 |
| 13 | Success at required height 0.1 m | **applied** (0.5 m / 0.1 m latch on the final waypoint) | 17k B3 |
| 14 | Height-aware demotion | **applied** | 17k B5 |
| 15 | V_fs clipping off in warm-up | **applied** | 17k |
| 16 | Stage-3 envs 36,000 | user override | USER_RULES |
| 17 | Goal admission 0.5 m / 0.1 m | **applied** | 17k |
| 18 | Goal segment cut at 0.2 m | **applied** | 17k B4 |
| 19 | Buffer per tile, work-per-metre eviction | **applied** (TileReplay) | 17k A1-A7 |
| 20 | Balanced sampling types -> tiles -> steps | **applied** | 17k |
| 21 | Ground friction 1.0 | **applied** s2_v4 onward (`ACP_PAPER_DR=1`; effective (r+1)/2) | ledger 15, recipe 2 |
| 22 | Contact-target noise on planner targets | **applied** (`ACP_PAPER_DR`, also on goal-env planner targets) | ledger 15, 17e |
| 23 | Odometry delay 50-100 ms, drift | **partly applied**: drift (XY 0.05 m disc, Z +-0.05) on; **delay not modelled** | recipe 2, open item 8 |
| 24 | Eval groups | user override (deferred: "extend the eval later") | USER_RULES |
| 25 | PPO batch size | user override | USER_RULES |

### A rows
| Row | Status |
|---|---|
| Data filters (step-over, leg-flip, stands, arm fix) | kept: fixes / user data choices |
| standatgoal station gate; multi-waypoint goal routes | kept: user rule (stations, 0.2 m) |
| Hip self-collision pairs excluded | kept: fix (FINDINGS A1) |
| Ramp gate / eta_hold | retired s2_v5 |
| eta jump at the stage-3 switch | fixed (anchor at the switch, eta 1.0 / 1.0 from row 1, 17l) |
| Stale `to_stage3.sh` | fixed (moved to _trash; `auto_stage3.sh`) |
| Offline planner pre-training | history only (the live planner descends from it; undoing it would need a run from scratch, which the user ruled out) |
| Probe envs on the planner | moot at eta 1 (log only) |
| Recorder envs start in the first 10 % | moot in stage 3 (phase 0 for all envs) |
| Clips cut into 10 s segments | still ours |
| Action clamp +-20 | still ours |
| Joint friction loss 0.3 on all hinges | still ours (no reason recorded) |
| Planner input normalisation | still ours |
| Hindsight target 5 steps after onset (SETTLE) | still ours |
| Every 2nd step labelled | still ours |
| Replay buffer kept in memory only | still ours; **empties at every resume** (refills in ~100 iterations, 17m) |
| Fall rule for planner-driven evals | eval only |

### Ranked: still deviating, not a user override (ACP)
ACP is now on the paper for every item the paper specifies, apart from the user's overrides. What is left is small:
1. **Replay not saved across a resume** (A, paper silent on persistence). The approved spawn-fix GO resumes s3_v2. For about
   100 iterations the planner then refines on whatever fills the buffer first, which is uneven across tiles. Impact: low to
   medium, and it is relevant now. Fix: save TileReplay with `latest.pt`, or hold planner updates until every type is present.
   Owner: ACP agent. Due: before GO_spawnfix, or at the latest at the next resume.
2. **Odometry delay 50-100 ms not modelled** (DN; drift is modelled). Impact on the simulated apparatus is low if the eval feeds
   the true base state. It matters on the real robot. Owner: ACP agent. Due: 2026-11-15, CPU implementation queued for the next
   planned resume (none is planned; the user's no-stop rule applies).
3. Joint friction loss 0.3 and action clamp +-20 (A, no reason recorded). Impact unknown, likely low. Owner: ACP agent. Due:
   2026-11-15. Write down the reason, or check on CPU that removing them leaves the rollout unchanged.
4. SETTLE target, labelling every 2nd step, 10 s segments, input normalisation (A). Impact low; record them as ours in the recipe.
5. MLP baseline input (DN). Unused, no action.
- To verify (owner: ACP agent, at the next check): the s3_v2 launch environment has `ACP_PAPER_DR=1 ACP_HF_FIX=1
  X2_EXCLUDE_PAIRS=hip`. 17k and 17m say the flags come from `ops/stage3_flags.env` but do not list the environment variables
  for stage 3. If `ACP_PAPER_DR` is missing, DN 21-23 are not applied in stage 3.

## Hiking (X2 recipe v8 vs the paper)

"Paper" means the v2 text where it gives a number. Where the text is silent, it means the release config (G1), which the user's G1
reproduction uses. The X2 task inherits `ParkourEnvCfg` (terrains, rewards, curriculum term, DR, PPO) and changes only what is
listed below. Table 2 of the paper is a single generalist policy, not the stair or platform specialists (paper text). So the
generalist is the right comparison.

### Counts
39 rows: user rule 10, X2 necessity or port fix 6, ours 12, unexplained 1, matches the paper or is no longer active 8, inherited
release-vs-text gaps 2. **Still deviating and not an override: 13** (ours plus unexplained; ranked below).

### Rewards
| Item | Paper | X2 v8 | Why | Likely climbing impact |
|---|---|---|---|---|
| Touchdown air time (option t) | none (only feet_air_time 0.5) | +140 x clamp(air - 0.05, 0, cap_ref(v)) per touchdown, x forward progress (v8) | ours | **high**: about 20 per episode against linear tracking about 30 (was 26 before v8); pays stepping, including at the foot of an obstacle |
| feet_stumble | none | -2.0, F_xy > 3 F_z, scaled 0.2-2 per foot | user rule ("INCLUDE THEM") | **high**: a toe on a riser or platform face costs up to 2x tracking per step, so attempting costs more than standing |
| wall_contact_graded (replaces undesired_contacts) | -1.0, indicator over non-foot bodies | -1.0 x (1 + min(F/100, 3)) per body; arms free under 30 N / 0.5 s | user rule (contact) | **medium**: shin or knee on a 0.39 m platform edge costs up to 4 per body, not 1 |
| foot_clearance | none | -0.5, swing foot under 6 cm | user rule | low (helps step height) |
| overspeed | none | -3.0 x max(vx - cmd, 0) | ours | low |
| stop_motion | none | -2.0 planar speed under a stand command | ours | low (stand commands only) |
| turn_effort / turn_drift (replace dont_turn_wait) | none | -0.5 / -2.0 | ours | low |
| a, d, s, f, r schedules (alive x0.3, dont_wait x2, style x0.3, fall -8, reference starts 0.5) | paper values | at paper values since 1,500 (stalled termination off) | ours, history | none now |
| feet_at_plane height offset | 0.035 | 0.073 (X2 sole) | X2 | none |
| pelvis_orientation on torso_link | pelvis | torso (X2 root is the pelvis; both bodies penalised as on the G1) | port fix | none |
| Style discriminator inputs | all joints equal | legs and waist 1.0, shoulders and elbows 0.3, wrists and head 0 | user rule | low |
| Style coef 0.25, reward quad | 0.25 | 0.25 | match | - |

### Commands
| Item | Paper | X2 v8 | Why | Impact |
|---|---|---|---|---|
| Stand share | 0.05 | 0.10 | ours (v8b: 0.05) | **high together with the next three rows**: per two-resample episode only about 33% are all-walking (paper about 90%) |
| Turn in place | only the Rough (Stand) column (5% of terrain) | 0.15 on every column, 4-6 s then stand | ours, for apparatus stations (v8b: 0.10) | same |
| Slow forward | vx >= 0.45 | 25% capped U(0.1, 0.45) | ours, slow final approaches (v8b: 0.12) | same |
| Slow yaw cap | wz +-1 | 40% capped U(0.1, 0.4) | ours | low (stair and gap targets are straight ahead) |
| Far target >= 1.5 m | flat patch | redrawn until >= 1.5 m | ours | none on stairs or gaps (fixed target at 3.7 m) |
| Rough (Stand) column | vx 0, random wz | held random vx 0-0.8, vy +-0.3, wz +-1; plus 2 velocity columns | ours | low |
| Backward commands | none | none (VX_BACK 0) | user rule | none |
| vx floor 0.45, k_v / k_w 2.0, resampling 8-12 s, goal 0.4 m | release | same | match | - |
| Routes and sidesteps on extended columns | n/a | 70% / 30% | ours on the user's map | none on paper columns |

### Curriculum and terrain
| Item | Paper | X2 v8 | Why | Impact |
|---|---|---|---|---|
| Promotion rule | tracking_exp_vel > 0.6 up, < 0.3 down, every episode | the same rule, but episodes with any stand, turn or slow command neither promote nor demote (FAIR) | fix (restores the paper's signal, H8) | medium: correct signal, but only about 33% of episodes move levels |
| Level keeper and pin across restarts | none | on | fix | none |
| Gap level stuck at rows 0-1 (eval L2 passes 20/20) | - | open, unexplained | unexplained | **high for gaps** |
| Paper terrain share | 20 of 20 columns | 20 of 32 (62.5%); stairs-up 15% -> 9.4%, platform-up 10% -> 6.25%, gaps 10% -> 6.25% | user rule (extended map, one third new) | medium: about 0.46x the paper's curriculum-eligible stair episodes per iteration, even with 2x robots |
| Rows, geometry, walls 0.3, perlin overlay, max init level 5 | release | same (checked 11-14 16:10) | match | - |
| Gap widths for the X2 leg | 0.1-0.7 m | the same | paper (user: the X2 is capable) | - |

### Observation, robot, PPO, AMP, episode
| Item | Paper | X2 v8 | Why | Impact |
|---|---|---|---|---|
| Camera | head, about 48 deg down, 32 x 18 crop, 0-2.5 m | chest RealSense 1.055 m, 61.6 deg, 87 x 58 deg, 32 x 36 (all rows), 0-3 m, mount randomised | user rule | low to medium (view from the toe line to 1.66 m) |
| Depth noise (range noise, artifacts, OOD) | specified in v2 A.4.2 | absent (inherited: the release lacks it too) | inherited release gap | low in simulation, high on the robot |
| Proprio obs, history 8, noise | Table 3/4 | same (29 joints) | match | - |
| Root-height termination | 0.5 m (G1 torso) | 0.39 m (pelvis) | X2 | none |
| base_contact | torso > 1 N ends the episode | torso > 50 N mostly vertical | user rule (contact) | low |
| Default pose, action scale, soft limit 1.0, implicit PD, gains mc_20261024, filtered pairs | G1 values | X2 equivalents | X2 / fix / user rule | none known |
| Sim faults end as time-outs | none | on | user rule | none |
| PPO (lr 1e-3 adaptive, KL 0.01, entropy 0.006, 5 x 4, 24 steps, clip 0.2) | release | same | match | - |
| Robots | 4,096 | 8,192; the user approved 16,384 "at the next planned restart", not taken into v7 or v8 (PhysX buffers untested) | ours; **unapplied user decision** | likely neutral to positive |
| Episode 20 s, 30,000 iterations | release | same | match | - |
| Motion reference | paper reference (G1) | the paper reference retargeted to X2 v4; waist at its +18 deg limit in 2,926 frames | user rule / X2 | low |
| MoE gate linear, logit weight decay 1.0, MaxPool no-op, eps 1e-6 | text differs from release | release values | inherited release gap | low |

### Ranked: still deviating, not a user override (Hiking)
1. **Command mix on the paper columns** (stand 0.10, turn 0.15, slow forward 0.25; slow yaw 0.4). Fewer full-speed climbing
   episodes, and only about 33% are curriculum-eligible under FAIR. v8b (0.05 / 0.10 / 0.12) is armed and is the user's named
   alternative. Owner: Hiking agent. Due: 2026-11-14 19:00 check (v8b if rows 5 and 8 show no progress). Paper-pure would mean
   turn and slow-forward only on the Rough (Stand) and velocity columns, not on the paper columns.
2. **Touchdown reward t (weight 140)**, not in the paper. It is still the second-largest positive term (about 20 per episode).
   Owner: Hiking agent. Due: 2026-11-15. CPU check of the per-episode term table at the stair foot (refuse against attempt), then
   weight 0-10 or off at the next restart.
3. **Gap curriculum stuck at rows 0-1** (unexplained, H8 open item 3). Owner: Hiking agent. Due: 2026-11-14 19:00 (per-row gap
   terminations).
4. Isaac-to-MuJoCo disagreement: Isaac trains at stairs rows 6-7 while MuJoCo refuses row 5 (not a recipe item, but possibly the
   largest unknown; H8 open item 2: a 0.5 m/s command and a one-wall test). Owner: Hiking agent. Due: 2026-11-14 19:00.
5. 16,384 robots approved and not applied. Owner: Hiking agent. Due: next restart (v8b or later), with a construction check of the
   PhysX buffers first.
6. overspeed -3, stop_motion -2, turn_effort / turn_drift, Rough (Stand) column velocities, far-target rule, slow yaw cap: low.
   Record them as ours, no action now.
7. Inherited depth-noise gap (paper text): low in simulation; needed before the robot. Owner: Hiking agent. Due: before any
   deployment.

**User-rule items with likely climbing impact (the user decides; not ranked above).** These are feet_stumble -2, the graded wall
contact (shins and knees up to 4x the paper's cost), and the 62.5% paper-terrain share. The user's 2026-11-14 priority is to close
the gap to the paper's own terrains first. The first two price an attempt at a riser or platform above standing still at its
foot, which matches the observed refusals (stalls, no falls). Recommendation: a CPU replay of v8 on stairs_up and platform_up
L5 / L8 logging the per-term cost of an attempt against a refusal. If stumble plus contact decide it, ask the user to scale them
down on the paper columns.
