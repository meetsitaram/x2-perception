# SSR: every DEV-P / ADDED row of the audit against the running code (2026-11-14, SSR agent)

Source rows: `docs/findings/ssr_paper_vs_code.md` (audit of 2026-11-13, line numbers given as "l.").
"v20" = the run live at 16:50 UTC 2026-11-14 (r7 from 25,550; trainer command line read from `/proc/<pid>/cmdline` on the node =
`runs/r7_flags_v20.txt`). "v21" = the one resume made in this turn (`runs/r7_flags_v21.txt`, `runs/r7_flags_v21.md`, queue39).
User overrides (USER_RULES 2026-11-13 "SSR back to the paper"): 8,192 envs; 17x10 height map; style input weighted legs/pelvis/waist 1.0,
shoulders/elbows 0.3, wrists/head 0; backward rule; graded wall contact; extended map instead of gap terrain.

## Counts (26 DEV-P rows)

| Status | Rows | n |
|---|---|---|
| Applied before this turn (v17 prior, v20 curriculum, phases over, code fix) | l.81, 123, 157, 160, 279, 280, 281, 328 | 8 |
| User override (kept as the user ruled) | l.42, 173, 188, 246, 292 | 5 |
| Fixed now (v21 resume) | l.95, 133, 143, 148, 151, 152, 306 | 7 |
| Deferred to the next from-scratch run (D1, owner SSR agent, ready by 2026-11-15 06:00 UTC) | l.147, 167, 172, 244, 282, 283 | 6 |

## DEV-P rows

| l. | Item | Paper | v20 (live) | Status | v21 | Expected impact on the apparatus (evidence) |
|---|---|---|---|---|---|---|
| 42 | Base height map | 18 x 10 | 17 x 10 | user override ("Don't change for now") | 17 x 10 | none measured |
| 81 | `--action_scale_paper` waist | 0.2 | fixed in code since the v15 merge (`x2.py:211`, waist in the 0.2 group); flag unused (action scales parked by the user, plant rule) | applied (code) | same | none (flag off) |
| 95 | Velocity kernel sigma_v | 0.25 | 0.1 (`--sig_v_to 0.1`) | **fixed now** | 0.1 -> 0.25 over 300 its from the resume (`--to_paper_ramp 300 --sig_v_start 0.1`) | E8: neutral at the stair foot (0.09 vs 0.10 per step, ours; 0.29 vs 0.32 paper); softer speed penalty, more room for style and careful steps |
| 123 | Staged schedule | single stage | `--stage_a 150 --stage_b 400` in the flags, no effect past it 400 | applied in effect (inactive since it 400) | same flags (no-ops) | none now; omit on the fresh run |
| 133 | Advantage combination | per-group normalisation, then weights ([12],[46]) | joint | **fixed now** | `--adv_norm separate`, blended joint -> separate over the same 300 its | style share of the policy gradient 3.7 % -> about 14 %, foothold 10 -> 17 % (v16 checks 11-13): better foot placement on risers and style; T1 (11-13) cleared it of the v15/v16 shuffle |
| 143 | Projected vy clamp | projection only | clamp +-0.4 | **fixed now** | `--no_vy_clamp` | ramp tiles at large heading (80 deg) get the true sideways command; small |
| 147 | Discrete speed / heading | U(0,1.2), +-30 | no discrete type | deferred D1 | - | the paper's precise-placement terrain; matters for stepping on blocks / logs |
| 148 | Gap speed / heading | U(0,1.2), +-15 | beam (the gap stand-in, amp_v5 gap slot 174.3 s) at +-30 | **fixed now** | `--beam_hmax 15` (new flag, default 30) | logs crossed nearer square; CPU check: heading max 29.98 -> 14.99 deg |
| 151 | Omni stage: when | only after the top level | 30 % of all robots from the start (live omni share 0.35) | **fixed now** | `--omni_top` (graduates walk omni on the open flat tile next episode; level 19 share 0.063 now) | about 30 % more robots on terrain tiles (stairs, platform); RISK: less open-floor turn-in-place practice (route stations still train it) - watched by the scorer's flat tests on the 500-exports |
| 152 | Omni ranges | Table 5 | 50 % turn-in-place, 30 % slow forward | **fixed now** (slow share) / user rule (turn) | `--omni_slow 0`; `--omni_turn 0.5` kept: user rule 2026-11-12 "turn in place trained explicitly" | small |
| 157 | Curriculum | Rudin [30], half tile | `--cur_rudin` (v20) | applied (v20, 16:28) | same | E8: stairs level 6.25 (25,120) -> 8.98 (25,820); +100 probe: level 16 passed riser 12.5 % -> 58 %, level 19 0 % -> 21 %, stall 100 % -> 75 % |
| 160 | Terrain mix | uniform over types | flat phase / ramp over since 1,050; obstacle share 0.8 = 1/5 per type | applied in effect (flat type itself: D1) | same | none now |
| 167 | Terrain types | slope, stairs, discrete, gap, platform | ramp, stairs, beam, platform, flat | gap: user override; discrete: deferred D1 | same | see 147 |
| 172 | Discrete obstacle height | 0.05-0.2 m | not built | deferred D1 | - | see 147 |
| 173 | Gap width / depth | 0.1-0.8 / 0.3-0.6 m | not built | user override (extended map, beams) | - | - |
| 188 | Gap foot termination | yes | no gap | user override | - | - |
| 244 | One D per paper type | 5 paper types | 5 incl. a flat D | deferred D1 | - | style on flat comes from a flat-only D instead of each set |
| 246 | Discriminator input | all 63 dims | `--disc_part_w 1.0,0.3,0.0` (since v17) | user override (applied) | same | - |
| 279 | Prior set sizes | Table 11 | amp_v5 (v17): 261.3 / 295.8 / 174.3 / 181.6 + 136.0 in the discrete slot | applied (v17) | same | - |
| 280 | Composition fwd + turn | Table 11 | explicit fwd / turn selection from the paper reference v4 + Pico (`amp_v5_meta.json`) | applied (v17) | same | - |
| 281 | Backward only in slope | Table 11 | no backward windows in any set (none in the sources; no-backward rule) | applied (v17; matches the user rule) | same | - |
| 282 | Traversal types | per paper type | platform set has the paper's step-onto / step-off; no discrete (step) set | deferred D1 | - | see 147 |
| 283 | No separate flat set | none | flat group (136 s) for the flat tile and omni robots | deferred D1 | - | see 244 |
| 292 | Envs | 4,096 | 8,192 | user override | 8,192 | - |
| 306 | Foothold target frame | base frame | heading frame | **fixed now** | `--foot_base_frame` | identical on flat; on stairs / ramps the target follows base pitch; small |
| 328 | Single stage | ~20k its single stage | phases over by 1,500 | applied in effect | same | none now |

**D1 (deferred, owner SSR agent, ready by 2026-11-15 06:00 UTC, for the next from-scratch run after 30,000):** the paper's terrain set
in the code's v14 layout = `--paper_terrain` (exists since the v15 merge: discrete blocks 0.05-0.2 m in the flat slot, beam = gap slot,
open FREE column for the omni graduates, no flat D) plus an amp_v5 build in the paper layout (discrete set = fwd + turn + step takes,
Table 11 136.0 s). Why not warm: every run that switched to the paper layout warm slid into a shuffle within 30-150 policy iterations -
v15 (r8: vx_ratio 0.80 -> 0.45 by +150), v16 (r9: 0.72 -> 0.51 by +50), T1 (r9 repeated with joint norm and sigma 0.1 held: 0.63 at +30);
the runs without it (v17 onwards, v14 layout, warm discriminators) held 0.77-0.80. The CPU check of 2026-11-13 (c) showed frozen
policies do not reproduce it (it appears only under training), so a CPU test cannot clear it; the GPU history is the evidence.
Other from-scratch items recorded with it: arm action bound loss from iteration 0 (E7b, broke walking warm), staged / flat-phase flags
omitted, swing-height / creep / drift tried off (see ADDED).

## ADDED rows (37): kept or dropped, with the failure that justifies keeping

| l. | Item | v20 / v21 | Decision | Reason (measured failure or rule) |
|---|---|---|---|---|
| 47, 213 | Observation noise | on | keep | sim-to-real; paper silent; shape-neutral |
| 48, 255 | Running normalisation (actor, critic, foothold, D) | on | keep | network inputs; cannot change warm |
| 49 | Map memory | off | off | - |
| 59, 224 | Action clamp +-8, bounded mean 8 tanh | on | keep warm; fresh run adds the arm bound loss | E7b: bound loss / mean shift warm broke walking (probe vx -0.08, alive 0.33) |
| 116 | Swing height 0.5 | on | keep warm; fresh run tests without | r7 1,250/1,500 shuffle gait gate; E8: pays climbing (0.05) more than standing (0.02) |
| 117 | Creep 0.5 | on | keep | slow-command fault 2026-11-12 (slow commands made good now 0.98) |
| 118 | Drift under stand / turn | on | keep | drift at stations (v13/v14 station legs) |
| 119 | Side graze -2 | on | keep | user graded-contact rule 2026-11-12 |
| 120, 191 | Wall contact graded / termination | on (not terminal on stairs / platform since v20) | keep | user rule; E8: lane-wall endings killed 42-75 % of attempts |
| 121, 122, 193, 222 | air_bonus / imitate / feet_air, progress gate, stuck term, gate supervision | off | off | - |
| 124 | Reward x 0.02 | on | keep | value scale only (advantages normalised) |
| 134 | Time-limit bootstrap | on | keep | standard, correct |
| 135 | Return logging | on | keep | logging |
| 150 | Flat tile robots | on | D1 | goes with the terrain set |
| 153 | Resample every 8 s | on | keep | stop-and-stand at stations (v13/v14); E8: keeps <= 8 % of refusals |
| 154 | Edge stops | on | keep | ACP's fix for stations at edges (apparatus) |
| 155, 176 | Route robots / extended map | on | keep | user decision 2026-11-12 |
| 161 | Walking starts (RSI) | on | keep | standing-still failure of r1-r6 (the track's first risk) |
| 175, 363 | Side obstacles in terrain and depth | on | keep | apparatus walls / rails; user graded contact |
| 189 | Base too low | on | keep | fall detection |
| 190 | Clearance scrape | on | keep | stands in for link contact |
| 192 | Diverged physics | on | keep | MuJoCo Warp height-field fault guard |
| 212 | Initial joint noise | on | keep | shape-neutral |
| 256 | Mirrored reference data | on | keep | symmetry |
| 257 | Reference window filter | on | keep | standing frames taught standing (r3) |
| 258 | Reference speed balance | on | keep | same |
| 308 | Non-finite guard | on | keep | NaN crashes |
| 332 | Foothold off for omni robots | on | keep | follows omni |

## v21 resume and its checks
- Flags: `runs/r7_flags_v21.txt` = v20 minus `--sig_v_to 0.1 --sig_v_from 600 --sig_v_ramp 300`, plus `--omni_slow 0 --adv_norm separate
  --to_paper_ramp 300 --sig_v_start 0.1 --no_vy_clamp --foot_base_frame --omni_top --beam_hmax 15`; queue adds `--warm_until_abs <S>`.
- Code: only `--beam_hmax` is new (`runs/patch_v21.py`, backups `_trash/2026-11-14/{env,train}.py.before_v21`); all other flags exist since
  the v15/v16 merge (default off).
- Pre-flight `runs/v21_preflight.sh` -> `runs/v21_checks/summary.txt`; queue `runs/queue39.sh` (tmux `ssr39`, log `runs/queue39.log`).
  Waits for v20's +250 stairs probe before stopping v20. Quality gate (+150/+350/+500): vx_ratio >= 0.75 x v20, ep_len >= 0.7 x v20,
  else v21b (v21 without the command / env items), then v20. Crash loop -> v20 flags.
