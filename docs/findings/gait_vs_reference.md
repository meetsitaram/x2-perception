# Gait vs reference motion: Hiking and SSR (2026-11-13)

Asked after the user's review of `eval/out/renders_20261113/`: Hiking "makes a lot of tiny steps", SSR "hasn't learned human
style yet". Read-only study (no training touched). Scripts and raw numbers: scratchpad `gait/` (rec.py records, ana.py statistics,
swing.py durations, nn.py / nn2.py feature-space distances); copy them into `eval/` if the eval wants them kept.

## Method

- **Policies:** MuJoCo CPU eval model, flat ground, robot starts standing, constant forward command 0.3 / 0.5 / 0.7 m/s for 14 s,
  3 seeds (2000-2002, nominal protocol), statistics over t = 4 s to the end. Standard harness command for Hiking
  (`HIKING_CMD=standard`) and SSR (`SSR_CMD=standard`). Checkpoints: Hiking `x2_parkour_v4_10375` (v5b run, weighted single
  discriminator on the paper reference), SSR `r7_v14_p250` and `r9_v16_p300`, ACP pair 10,000 (goal-driven: it walks at its own
  pace, about 0.40 m/s, whatever the command, so it has one row).
- **References** (same code, same definitions): Hiking `parkour_motion_without_run_x2_v2.npz` (the file v5b trains on; 52 clip
  breaks found from base-position jumps); SSR `amp_v4.npz` group `flat` (its 63-D frames: feet and pelvis velocity in the base
  frame). Steady flat forward walking: 2 s windows (step 1 s) with mean forward speed in the bin, |vy| < 0.1, mean |yaw rate| < 0.3,
  speed sd < 0.25; for Hiking also pelvis height and lowest-foot height steady (flat). Bins 0.2-0.4 / 0.4-0.6 / 0.6-0.85 m/s.
  Hiking has no explicit flat-walk frame index in `paper_ref_x2_categories.json` (only terrain clip times), hence the filter.
- **Definitions:** contacts kinematic for every source (ankle within 2.5 cm of the lowest foot of the surrounding 1.5 s and moving
  < 0.25 m/s). Check against MuJoCo foot loads (40 N on / 10 N off): cadence agrees within 0.1-0.4 steps/s on every policy row.
  Step length = forward distance between the feet at touchdown (heading frame); width = lateral distance at touchdown; clearance =
  median swing peak of the ankle above its stance height; joint ranges = mean of L/R (p95 - p5); arm swing = shoulder pitch range;
  pitch forward positive.

## Table

| Source | cmd | speed m/s | cadence steps/s | step m | width m | clearance m | double support | knee range deg | hip pitch range deg | arm swing deg | pelvis / torso pitch deg |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Hiking ref** (36 s) | 0.3 | 0.28 | **1.44** | **0.20** | 0.12 | 0.066 | 0.28 | 43 | 36 | 5 | -0.1 / +0.5 |
| Hiking 10,375 | 0.3 | 0.28 | **2.90** | **0.06** | 0.18 | 0.039 | 0.30 | 37 | 24 | 15 | +3.3 / **-17.0** |
| **SSR ref** amp_v4 flat (64 s) | 0.3 | 0.32 | **1.33** | **0.31** | 0.19 | 0.096 | 0.23 | 54 | 32 | 11 | -2.3 / -2.3 |
| SSR r7_v14_p250 | 0.3 | 0.26 | **5.13** | **0.05** | 0.24 | 0.019 | 0.28 | **10** | 8 | 4 | -0.7 / -10.6 |
| SSR r9_v16_p300 | 0.3 | 0.28 | 5.47 | 0.05 | 0.22 | 0.014 | 0.34 | 8 | 8 | 3 | +0.7 / -14.8 |
| **Hiking ref** (28 s) | 0.5 | 0.50 | **1.93** | **0.25** | 0.13 | 0.070 | 0.06 | 44 | 43 | 9 | +0.2 / +1.7 |
| Hiking 10,375 | 0.5 | 0.47 | **2.83** | **0.15** | 0.19 | 0.048 | 0.18 | 40 | 30 | 26 | +3.3 / **-17.1** |
| **SSR ref** (68 s) | 0.5 | 0.48 | **1.47** | **0.34** | 0.17 | 0.097 | 0.07 | 58 | 36 | 10 | -1.4 / -1.4 |
| SSR r7_v14_p250 | 0.5 | 0.41 | **3.50** | **0.11** | 0.20 | 0.103 | 0.04 | 44 | 22 | 5 | -0.2 / +1.3 |
| SSR r9_v16_p300 | 0.5 | 0.48 | 4.33 | 0.12 | 0.17 | 0.040 | 0.07 | 22 | 15 | 7 | +1.3 / -5.4 |
| ACP pair 10,000 (own pace) | 0.5 | 0.40 | 1.10 | 0.31 | 0.15 | 0.099 | 0.03 | 51 | 40 | 7 | +2.9 / +7.0 |
| **Hiking ref** (8 s) | 0.7 | 0.64 | **2.00** | **0.30** | 0.14 | 0.071 | 0.00 | 46 | 50 | 11 | +0.4 / +2.4 |
| Hiking 10,375 | 0.7 | 0.63 | **2.80** | **0.22** | 0.19 | 0.052 | 0.04 | 46 | 38 | 36 | +4.5 / **-15.9** |
| **SSR ref** (56 s) | 0.7 | 0.72 | **1.54** | **0.42** | 0.16 | 0.138 | 0.00 | 65 | 42 | 10 | -0.6 / -0.6 |
| SSR r7_v14_p250 | 0.7 | 0.55 | **3.53** | **0.13** | 0.22 | 0.119 | 0.05 | 50 | 26 | 7 | -0.5 / +0.8 |
| SSR r9_v16_p300 | 0.7 | 0.67 | 4.45 | 0.11 | 0.19 | 0.066 | 0.01 | 32 | 18 | 22 | +0.8 / -8.1 |

Falls: none, except SSR r9_v16_p300 at 0.7 m/s, 1 of 3. Swing / stance durations (median): Hiking 0.24 / 0.44 s (0.3 m/s),
0.28 / 0.42 (0.5), 0.34 / 0.36 (0.7); SSR r7_v14 0.14 / 0.24, 0.28 / 0.30, 0.32 / 0.24. Training logs agree with MuJoCo: SSR r7
`gait.steps_per_s` 8.3 (it 1,500-1,700), 5.2 (3,000), 4.7 (6,000), 4.2 (8,300), 4.7 (10,400-10,600) at about 0.50 m/s.

**In short.** Hiking steps at a fixed 2.8-2.9 steps/s at every speed; the reference goes from 1.4 to 2.0 as it speeds up. Its
steps are 30-70% shorter, 2-3 cm lower and 5-6 cm wider. Its hip swings 12-13 deg less, its arms swing 3x more, and the
torso leans 16-17 deg back (waist on its limit). SSR steps at 3.5-5.5 steps/s against 1.3-1.5. Its steps are a third or less
of the reference length, knee range at 0.3 m/s is 10 deg against 54, and its arms barely move. ACP, which has no style prior,
is the closest to human cadence (1.1 steps/s, 0.31 m steps at 0.40 m/s).

## Why the style reward does not prevent it

### Hiking

1. **A reward that pays per touchdown sets the cadence (track addition, not in the paper).** `feet_air_touchdown_positive`
   (`x2/staged.py:184`, weight 140, option "t") pays `clamp(air - 0.05, 0, 0.30)` at every touchdown. Its rate is
   cadence x min(swing - 0.05, 0.30). With no double support that is maximal at a 0.35 s swing, about 2.9 steps/s, whatever the
   speed. That is exactly the measured cadence: 2.80-2.90 at all three speeds, with swing 0.34 s at 0.7 m/s. At 0.7 m/s the
   policy's cadence earns about 0.83 pay/s against 0.60 for the reference's 2.0 steps/s (its swings of about 0.5 s are capped at 0.30), about +35%. In training
   (iterations 7,500-10,768) it is 0.0243 per step, 22% of the task reward (0.109 per step). That is as large as velocity
   tracking (0.030) and 27x the paper's own air-time term (`feet_air_time` 0.5: 0.0009). The paper's term pays the duration of
   single-stance phases, so it favours longer steps.
2. **The style signal is flat.** Discriminator reference / policy outputs +0.224 / -0.224, style reward 0.625 (quad), coef 0.25:
   0.156 per step, constant since 7,500. Even a perfect fool of the discriminator (D = +0.22) would raise it only to 0.25 x 0.85
   = 0.21 (+0.055 per step). In practice the gradient comes from small D differences between the policy's own samples. The
   discriminator is strongly regularised (gradient penalty 5, logit weight decay 0.04, paper values). The user's G1 run sits at
   the same equilibrium (0.62, +-0.25), so "0.62" shows the regularised equilibrium, not a human gait.
3. **The 0.2 s window cannot see cadence, though it could see posture.** 10 frames x 0.02 s (paper: same) is 20-30% of one
   reference stride cycle (0.7-1.4 s) and 57% of the policy's step. Per-frame joint speeds are the same: hip pitch velocity RMS
   2.18 rad/s (policy, 0.7) against 2.35 (reference), 1.84 against 1.97 at 0.5. Short fast steps and long slow steps give the
   same joint speeds, so cadence is visible only as a longer-scale pattern. Posture is visible in the window. In the
   discriminator's own raw feature space, the policy's 0.2 s windows sit 5-6x farther from the nearest reference window than
   reference windows sit from each other, mostly in joint posture. ACP, never trained on this reference, sits at the same
   distance (4.9-6.1x), so part of that gap is MuJoCo vs Isaac and not style.
4. **The reference contains the backward lean.** In `parkour_motion_without_run_x2_v2.npz`, 2,922 frames (15%; all of the first
   59.5 s, which are the stairs and block terrain clips) have `waist_pitch_joint` pinned at -0.314 rad, the X2's backward
   limit. That is 25% of all reference frames between 0.3 and 0.6 m/s, torso -7.9 deg there. This looks like a retargeting clamp.
   The policy holds the waist at -0.355 rad (torso -16 to -17 deg). Now that the waist is in the discriminator input (weight
   1.0, v5 option j), that pose is "in the reference", and the discriminator is not conditioned on terrain or command.
   (FINDINGS already records the waist on its limit in the policy since 1,000; that the reference itself sits there is new.)
5. **Low speeds are outside training.** The X2 command floor `HIKING_X2_VX_MIN` 0.6 (option "c") trains forward commands of 0.6-0.8
   m/s only (paper: 0.45-1.0 / 0.45-0.8). This explains the 6 cm shuffle at 0.3 m/s. It does not explain the fixed cadence at 0.7.
6. Arms swing 3x the reference (15-36 deg against 5-11). Arms enter the discriminator at weight 0.3 only, and `freeze_upper_body`
   is -0.004 per rad.

### SSR

1. **The discriminator does catch it; the policy does not respond.** Logged D outputs: reference +0.61 to +0.64, policy -0.59 to
   -0.64 from iteration 1,500 to 10,600, style reward 0.32-0.37, flat (flat group 0.36). The discriminators separate the gaits
   well. So the problem is the gradient share, not detection. The style term is 0.2 x 0.33 = 0.066 per step against task terms
   summing to about +2.4 / -0.7. Under joint advantage normalisation (`--adv_norm joint`; the paper normalises per group per
   `ssr_paper_vs_code.md` rank 2) style gets far less than its 0.2 share. Closing the whole gap (0.33 -> 0.85) is worth +0.10 per
   step. That is smaller than what the task terms below charge for bigger strides.
2. **Task terms that make short fast steps cheap:**
   - `dof_dev` -1.8 (paper) on |q - q_default| of all 21 joints is the largest penalty (-0.25 per step). Every degree of
     knee or hip excursion costs. Knee range at 0.3 m/s is 10 deg against 54.
   - `lin_vel` kernel sigma 0.1 (deviation; paper 0.25) punishes the speed oscillation within a long stride.
   - `impact` -1.3 (paper) on vertical foot speed in contact.
   - `single_support` 0.2 (paper) pays while double support stays under 0.2 s.

   None of these is new, but together with weak style they settle on the shuffle.
3. **A 0.1 s window.** 5 frames at 50 Hz (paper: 5 frames) is 1/7 of a reference step (0.65-0.75 s), so cadence itself is not
   visible. Knee and hip ranges, foot heights and foot positions are visible, and the discriminator does use them (point 1).
4. Arms are masked in the discriminators (`--disc_mask_arms`, deviation): arm swing 3-7 deg against 10-11 in the reference.
5. v16 (r9, paper-aligned attempt) is no better: 4.3-5.5 steps/s.

## What the papers do

| Item | Hiking paper (G1 release) | Hiking X2 now | SSR paper | SSR r7 v14 |
|---|---|---|---|---|
| Window | 10 frames x 0.02 s = 0.2 s | same | 5 frames (= 0.1 s at 50 Hz) | same |
| Features | gravity, all joints q / qd (unweighted), base v, w | legs and waist 1.0, shoulders / elbows 0.3, wrists / head out | 63-D: g, v, w, 21 q / qd, hands, feet | same, arms (q, qd, hands) masked |
| Discriminators | 1, [1024, 512], quad, coef 0.25 | same | 5 per terrain, w_s 0.2 | 5 (flat instead of gap / discrete), 0.2 |
| Step-setting rewards | feet_air_time 0.5 (single-stance duration) | + feet_air_touchdown 140 per touchdown (track) | air time -2 beyond 0.4 s, single support 0.2, dof_dev -1.8 | same + swing_h 0.5, creep 0.5 (added) |
| Command | vx 0.45-1.0 (rough) / 0.45-0.8 | 0.6-0.8 floor | sigma_v 0.25 | sigma_v 0.1 |
| Advantage mix | single reward sum | same | per-group normalised (audit) | joint |

## Fixes, ranked per track ("paper" = the paper does this; "deviation" = we would add it)

**Hiking**
1. **paper:** remove `feet_air_touchdown` (option "t"), or cut it to about 10. Expected to free the cadence. Cheapest test: a
   resume with `HIKING_X2_AIR_TD_WEIGHT=0` and gait stats every 500 iterations. Risk: the term was added to get the X2 stepping;
   if stepping stalls, use fix 2.
2. **deviation (if stepping needs a push):** pay air time per second instead of per touchdown, e.g. `feet_air_time` as the paper's
   with a larger weight. Or make the touchdown pay saturate at the reference swing for the commanded speed (about 0.40-0.45 s),
   so long strides earn as much per second as short ones.
3. **paper:** fix the reference: re-retarget or repair the 59.5 s of terrain clips whose waist pitch sits on the -0.314 rad limit.
   The paper's G1 reference has no such clamp. Until then, consider weighting waist pitch 0 in the discriminator (deviation).
4. **paper:** command floor back to the paper's 0.45 (`HIKING_X2_VX_MIN`) so 0.45-0.6 m/s walking is trained. Below 0.45 is
   outside the paper too.
5. **deviation:** a longer discriminator window that spans a stride (e.g. 10 frames at 0.1 s spacing = 1 s), or a second
   discriminator on 1 s windows, so cadence and step length become observable.
6. **deviation:** cadence-vs-speed shaping from the reference (cadence about 1.3 + 1.0 x v steps/s on the X2 reference) as a small
   reward. Last resort: it hard-codes the answer.

**SSR**
1. **paper:** `--adv_norm separate` (per-group advantage normalisation), so style gets its 0.2 share of the gradient. The
   discriminators already separate the gaits (+-0.63); the policy just does not listen.
2. **paper:** velocity kernel sigma 0.25 instead of 0.1.
3. **paper:** drop the added `swing_h` and `creep` terms (they tilt the gait away from the paper's balance; swing_h holds the
   swing at its 0.10 m target).
4. **paper:** unmask the arms in the discriminators (arm swing).
5. **deviation:** raise w_style (0.2 -> 0.4) if 1 to 4 do not move cadence. Or use `dof_dev` relative to the reference pose
   (`--dof_dev_ref`, already in the code) so stride amplitude is not charged.
6. **deviation:** a longer window (as Hiking 5) or cadence shaping (as Hiking 6).

## Checks still open

- Score the Hiking discriminator itself (`tracks/hiking/x2_port/score_discriminator.py`) on reference flat-walk windows at 0.7
  m/s vs the policy's MuJoCo windows. That shows directly whether it rates the 2.9 steps/s gait as reference-like.
- Side finding (eval): the local hand-over `eval/data/handover/hiking_v4_10375/run_params/` is the 4,750 copy (same md5). The
  autoscorer copies `hiking_run_params` when an export lacks it, so `hiking_check.py` reads the 4,750 env.yaml for 10,375.
  The live run's env.yaml differs (weighted style terms, `turn_effort` / `turn_drift`). The adapter itself reads the card, which
  is correct.
