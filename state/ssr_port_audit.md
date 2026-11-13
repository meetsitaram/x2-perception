# SSR paper vs SSR track implementation: port audit (independent reviewer, read only, 2026-11-12)

Scope: the SSR paper (arXiv 2605.30770, HTML version, read in three parts) against the SSR code. I read the synced copy
`tracks/ssr/code/ssr/{env,train,networks,x2}.py` (04:58 UTC) and diffed it against the live node copy (05:55). The live copy
adds only `--sig_v_to/--sig_v_from/--sig_v_ramp`, `--dof_dev_ref`, `--excl_pelvis_hiproll` and `--default_from_ref`; there
are no other reward or physics changes. I also read the run logs (r6b, r6d, t_a, d1, copied read-only), the r6b_002000
handover `eval/data/handover/ssr_r6b_2000/obs_check.npz` and the shared MJCF (`gwbc-v6/.../mjcf/x2_ultra.xml`, the same
file as the track copy). The collision and stand measurements ran on the desktop CPU in MuJoCo 3.15, one core each. The
scripts are in my scratchpad: `coll_census.py`, `hiproll_real.py`, `stand_force.py`. No GPU, no training process and no
shared file was touched.

Marks: **[paper]** read in the paper; **[code]** read in the code; **[meas]** measured by me; **[log]** read from the
training logs or obs_check; **[inf]** my inference. "Paper silent" means the paper does not state it.

## Headline (new findings; none of them is in `state/ssr.md`)

1. **The policy has turned hip roll into a locked joint, held by a convex-hull artefact.** [log, meas] Take r6b_002000,
   deterministic, flat ground, 0.6 m/s, 50 steps (obs_check). The hip-roll action sits at the actor's bound on both sides
   (mean -7.996 / +7.996; the bound is 8 tanh(x/8)). The PD target is clamped at the inner joint limit (-0.235 / +0.235) on
   every step. The joint itself stays at -0.057 / +0.056 (range -0.065 to -0.046) because the pelvis and hip-roll
   collision hulls stop it. The PD pushes about 100 x 0.18 = 18 N.m into that stop all the time.
   - The real meshes do **not** touch (item 5): the stop exists only in the simulator.
   - This explains the r6c result. With the pair excluded, the -0.235 target becomes reachable and the legs scissor
     (knee-knee and hip_yaw-knee contacts up to 31 mm in the eval monitor's run), so the robot falls.
   - It also explains the hip-roll exploration noise in every run: hip-roll std 1.0 -> 1.89 (r6b 2,110; r6d 2,240). Ankle
     roll grows from 1.0 to 1.49, and arms to 0.8-1.0. With the mean at the bound, hip-roll noise changes nothing, so the
     entropy bonus inflates it for free.
   - Lateral balance therefore has no hip-roll control. What remains is ankle roll (noise std 1.49 x 0.485 = 0.72 rad,
     torque-limited at 24 N.m) and stepping. [inf] This fits the short, quick steps.
2. **Action scale differs from the paper's X2 table.** [paper Table 14, code] The paper uses 0.25 rad per unit on every
   leg joint and 0.2 on waist and arms. The code uses 0.25 x effort / kp_armature: 0.261 hips / knee / waist yaw, 0.258
   ankle pitch / shoulder pitch / roll, and **0.485 on ankle roll, shoulder yaw and elbow**. So ankle-roll targets move
   1.94x, and shoulder yaw and elbow 2.4x, as far per unit of noise as in the paper. The scale is fixed by the user's
   plant rule (USER_RULES, "Action scale stays 0.25 x effort / (armature x w^2)"); I only report the difference.
3. **The default pose is the G1's, not the X2's.** [code, meas, inf] The plant default (hip pitch -0.312, knee 0.669,
   ankle -0.363) is the gear_sonic G1 default. On the X2 it puts the pelvis **0.641 m** above the soles. The paper's
   base-height target is **0.68 m** [paper Table 7]. The reference-mean leg pose gives 0.685 m and straight legs 0.675 m
   [meas]. The paper does not give its default pose. Its 0.68 m target fits a much straighter default than ours. Ours costs
   `dof_dev` for every walking pose (reference knee 0.32 vs default 0.669) and starts every episode in a crouch. This is
   the same defect as Hiking audit #1. It is fixed by the user's plant rule, and SSR's step 2 (`--default_from_ref`)
   already prepares it.
4. **Orientation kernel.** [paper, code] The paper prints the plain norm, exp(-||g_xy||_2 / 0.01); the code uses the
   squared norm (a PAPER_NOTES choice). Under the paper's form the upright stander's advantage over a shuffler grows: I
   recomputed from the measured terms that it is 0.27 vs 0.00 per step, against 0.50 vs 0.36 under the squared form. So
   this is not the cause of standing. It is a listed deviation; no change recommended.

## 1. Reward table, term by term (paper Table 6 / 7 against `env._rewards`)

All weights, sigmas and constants match the paper. Every kernel and normalisation matches **except orientation**.

| Term | Paper | Code | Verdict |
|---|---|---|---|
| lin_vel 1.0 | exp(-\|\|v_xy - u\|\|^2 / 0.25) | same; v in the full base frame, u in the heading frame | match (frame nuance: minor) |
| ang_vel 0.8 | exp(-(w_z - u_w)^2 / 0.25) | same; u_w = clip(1.0 x heading error, +-1.2) | match (yaw gain: paper silent) |
| orientation 0.5 | exp(-\|\|g_xy\|\|_2 / 0.01) | exp(-\|\|g_xy\|\|^2 / 0.01) | **differs** (see headline 4) |
| ang_vel_xy 0.25 | exp(-\|\|w_xy\|\|^2 / 0.25) | same | match |
| base_height 0.4 | exp(-(h - 0.68)^2 / 0.01) | same, h = pelvis height above the terrain under it | match; default pose gives h 0.641 |
| action_rate -0.12, smoothness -0.06 | (1/N) \|\|.\|\|^2 | same, on the sampled (noisy) actions | match |
| dof_vel -0.96, torque -0.6 | normalised by limits, 1/N | same | match |
| dof_dev -1.8 | (1/N) \|\|q - q_def\|\|_1 | same, 21 joints including arms | match; q_def is the G1 default (headline 3) |
| pos / vel / torque limits -1.5 / -6 / -6 | indicators, 1/N | same; hard limits | match |
| stand_still -0.12 | dof_dev x (\|\|u\|\| <= 0.15) | same | match |
| single_support 0.2 | 1[\|\|u\|\| <= eps or one foot in contact in the last 0.2 s] | same; contact = foot normal force > 5 N | match (threshold: paper silent) |
| impact -1.3, slip -0.2 | sum v_z^2 c, sum \|\|v_xy\|\|^2 c | same; foot velocity from the ankle-roll body origin | match |
| air_time -2.0 | sum max(t_air - 0.4, 0) c_first | same | match (a penalty only) |
| stumble -2.0, lateral 0.08 | as Table 6 | same | match |
| foothold w_f 0.25 | exp(-(sum rho)^2 / 0.0625); rho = 0 on slopes | same; rho also 0 on the flat / omni tile | match (no flat terrain in the paper) |
| style w_s 0.2 | max(0, 1 - 0.25 (D - 1)^2), one discriminator per terrain | same, but arms masked (`--disc_mask_arms`) and weight scheduled | listed deviations |

Grouping: three critics, advantages weighted 1.0 / 0.25 / 0.2 and normalised jointly [code]. The paper says only "advantage
weight associated with each reward group", which matches weighting advantages; normalisation: paper silent.

Paper silent on all of these: multiplying rewards by dt (code 0.02), clipping to positive values (code none), a termination
penalty (code none), time-out bootstrapping (code bootstraps time limit and tile end).

Added terms not in the paper (listed deviations): the staged schedule, feet_air (+20 x clamp(air - 0.1, 0, 0.3)), the slow
kernel ramp to sigma 0.1 (r6d), and the optional progress gate, command curriculum and stuck termination (off).

**Why a stander does not out-earn a walker in the paper, in the paper's numbers.** It is not structural; the paper relies
on the gait being cheap. [paper] A stander under a moving command c still collects: lin_vel exp(-c^2 / 0.25) (0.85 at
0.2 m/s, 0.24 at 0.6 m/s, about 0.37 averaged over U(0.15, 1.2)), ang_vel 0.8, ang_vel_xy 0.25, base_height up to 0.4,
lateral 0.08, foothold 0.25 and style 0.2 x r_s. That is about 2.4 of locomotion pay plus about 0.3.

A walker gains about +0.63 per step in lin_vel on average and +0.2 in single support: **about +0.83 per step**, minus
whatever its gait costs in smoothness, impact, slip, deviation and torso motion. Nothing in the paper's table removes the
stander's pay (no positive clipping that binds, no progress gating). Walking wins in the paper only if a gait costs under
about 0.8 per step.

What makes it lose here [meas by SSR, terms_probe]: the first reachable gait pays 1.11, not under 0.8 (yaw -0.23,
orientation -0.13, roll/pitch rate -0.11, action_rate -0.15, smoothness -0.15, impact -0.17, slip -0.13, dof_dev -0.13).
The causes are below in ranking items (a). On the suspects named in ssr.md: "posture worth about 1.4" and "foothold 0.25"
are as large in the paper. They are not implementation defects. They become decisive only because our gaits are expensive.

## 2. Commands and curriculum

| Item | Paper | Code | Verdict |
|---|---|---|---|
| Speeds and headings by terrain | slope U(-1, 1), heading +-80; stairs / discrete U(0, 1.2), +-30; gap / platform U(0, 1.2), +-15 [Table 4] | same per type (ramp, stairs, beam, platform, flat: 80 / 30 / 30 / 15 / 30 deg) | match (route types replace the paper's) |
| Omnidirectional stage | after the top level and leaving the terrain: vx +-1, vy +-0.4, yaw +-1.2 [Table 5] | 10 % of robots always omni on the flat tile | listed deviation |
| Zero-command threshold | eps 0.15 on \|\|u\|\| | same (norm includes the yaw-rate command) | match |
| Resampling within an episode | paper silent | none: one speed per episode | |
| Slow-command handling | paper silent; the paper reports no slow-speed result | walking starts only from frames >= 0.3 m/s; prior windows only >= 0.2 m/s | deviation; see ranking (c) |
| Promotion | "as in Rudin et al." (legged_gym: up when distance > half the tile, 4 m) | **up only on reaching the tile end** (start at 0.35-0.9 m, so 7.1-7.65 m within 20 s) | **differs** [code]: a command under about 0.37 m/s can never be promoted |
| Demotion | legged_gym: distance < 0.5 x \|cmd\| x episode length | distance < max(0.5 x \|cmd\| x elapsed, 0.5 m), after 3 s | close; the 0.5 m floor also demotes zero commands |
| Over the top level | legged_gym: random level | random level | match |
| Episode length | paper silent | 20 s | |

## 3. Observations, control, plant, terminations, randomisation, initial state

- Proprioception 72-D (angular velocity, gravity, command, q - q_def, qd, last action), history 5 (paper silent), depth
  36 x 36: matches [code].
- Noise: legged_gym-like values; paper silent.
- Control 50 Hz [paper]; 200 Hz physics, decimation 4: paper silent on sim dt.
- **PD gains, paper Table 14 against mc_20261024** (kp / kd):

  | Joint | Paper | mc_20261024 |
  |---|---|---|
  | hip pitch | 120 / 4 | 100 / 4 |
  | hip roll, hip yaw | 100 / 2 | 100 / 3 |
  | knee | 120 / 4 | 150 / 5 |
  | ankle pitch | 40 / 2 | 40 / 3 |
  | ankle roll | 20 / 1 | 30 / 2 |
  | waist yaw | 100 / 3 | 40 / 8 |
  | shoulders, elbow | 30 / 1 | 30 / 1, 20 / 1, 20 / 1, 50 / 1 |

  The gains are fixed by the user's rule. The one sizeable gap is waist yaw (2.5x softer, 2.7x more damped); I expect it to
  matter little for leg gait [inf]. d3 already showed that the gains do not change how motion starts.
- Action scale: headline 2.
- Default pose: headline 3; paper silent on its values.
- Robot: the paper's X2 has 29 joints, 1.31 m, about 39 kg [paper App. D]. The model has 31 joints and **43.5 kg** [meas],
  11 % heavier: a different variant or extra mass.
- The MJCF gives every joint frictionloss 0.3 N.m; Isaac Gym has no joint friction by default. Minor.
- Terminations: the paper's list is matched, with thresholds paper silent. Code: tilt over 60 deg, pelvis under 0.3 m,
  elbow force over 80 N, clearance spheres, leaving the tile, side obstacles (listed).
- Domain randomisation: Table 10 matched [code]. Mass, CoM, link mass, motor strength, gains, friction 0.2-1.25, delay 0-2,
  push 0.4 m/s every 12 s, occlusion noise; camera mount ranges are wider on the user's chest mount.
- Initial state: paper silent. Code: default pose plus U(-0.05, 0.05) joint noise, 2.4 cm drop, and half the walking
  episodes from reference frames at 0.3 m/s or more.
- Passive stand [meas]: with zero action, under mc gains, the robot falls in 1.41 s both with and without the pelvis /
  hip-roll pair. The ankles cannot hold it (ssr.md has the same).

## 4. The method's own components

| Component | Paper | Code | Verdict |
|---|---|---|---|
| Mirror-equivariant encoder (CNN lift + equivariant conv, equivariant MLP and GRU, three heads) | App. B | `networks.Encoder` with paired-channel layers; `tests/test_mirror.py` passes (SSR) | implemented |
| Mirrored samples in PPO, encoder on the original only | Sec. 3.3 | `M.actor_obs / priv / joints` appended per mini-batch, same old log-prob and advantage | implemented (as in rsl_rl symmetry augmentation) |
| MoE actor, 5 experts [1024, 512, 128], gate 128 | Table 8 | same; gate prior off; mean bounded by 8 tanh(x/8) | implemented. **The bound is where hip roll saturates** |
| Three critics | Sec. 3.1 | 3 MLPs [512, 256, 128] | implemented; joint advantage normalisation (paper silent) |
| Velocity estimator, reconstruction and VAE losses, separate optimiser | Tables 8 / 9 | `Aux`, own Adam | implemented; loss weights paper silent |
| Foothold imagination (s, a) -> Gaussian per foot, NLL on the first future contact; guidance on once the mean level > 2 | Eq. 1-3 | same; targets in the heading frame instead of the base frame; 3 x 3 stencil | implemented (frame: minor) |
| Five discriminators, 5 x 63-D, GP 20, weight decay 1e-4 | Tables 8 / 9 | same; LSGAN loss; mirrored reference data | implemented. **Deviations:** arms masked; reference windows under 0.2 m/s removed; speed-balanced sampling; style weight 0 until 1,000 |
| Motion data | 71 clips, 521 s, own mocap + AMASS; every terrain set reuses flat forward, backward and turn clips [Table 11] | BONES-SEED + Pico captures; flat set 999 s before filtering | different data; slow and standing windows filtered out by design |
| Minimum std 0.2 | Table 9 | clamp at 0.2; init 1.0 (paper silent); state-independent | implemented; std **rises** on hip roll, ankle roll and arms (headline 1) |

Nothing is stubbed. The "switched off or replaced" items are only the listed deviations.

## 5. Collision model (CPU MuJoCo measurements)

**What the paper says: paper silent on self-collision.** Isaac Gym loads each URDF collision mesh as its convex hull. In
legged_gym's default config (`self_collisions = 0`) self-collision is on, and PhysX filters only joint-adjacent links
[inf, not from the paper]. The X2 URDF uses the same meshes, including the 0.9 y-scaled hip-roll mesh. So the paper may or
may not have had this stop; it cannot be determined from the paper.

**Collidable set in the shared MJCF** [meas]:
- 328 robot body pairs (999 geom pairs) collide: all mesh pairs except parent-child.
- One explicit exclusion: head_pitch / torso.
- Waist yaw and waist pitch have no collision geometry, so torso / pelvis is a live pair; it is at least 50 mm apart
  at the default pose.

**At the default pose** (hull distances): pelvis / hip_roll 2.6 mm (L) and 2.7 mm (R); shoulder_roll / torso 7.0 mm
(constant over shoulder roll 0.061-0.2); everything else 50 mm or more.

**Hip-roll sweep, pelvis / left hip_roll** (hip pitch -0.312, -0.8, +0.3; the same at every pitch):

| hip roll (inward negative) | +0.1 | 0 | -0.03 | -0.05 | -0.08 | -0.12 | -0.16 | -0.2 | -0.235 (limit) |
|---|---|---|---|---|---|---|---|---|---|
| hull distance, mm | 7.0 | 2.6 | 1.1 | 0.0 | -1.7 | -3.9 | -6.3 | -8.7 | -10.9 |
| real CAD meshes, min surface distance, mm | 5.2 | 5.0 | 5.1 | 5.1 | 5.1 | 5.1 | 5.2 | 5.4 | 4.4 (3.6 measured both ways, see §7) |

- Real meshes: the visual STL shells, unscaled. Distance is measured from hip-roll vertices to the pelvis surface, signed
  by face normal; no vertex lies behind a pelvis face at any angle.
- The real parts keep about 5 mm clearance across the **entire** hip-roll range. The closest real points are on the
  pelvis underside (z -86 mm), next to the roll axis, which is why the gap does not change.
- The hull contact point is on the same plane but 40-50 mm forward, where the hull of the clevis-shaped hip-roll link
  fills its open side.
- **Verdict: the pelvis / hip-roll contact is an artefact of the convex hulls.** The real mechanical stop is the joint
  limit, -0.235 / +0.235 rad (URDF range; I have no other source for the hardware stop).

**Force it applies** [meas, passive stand, PD targets held]:

| Hip-roll target inward | Hull contact (vertical force, share of 426 N weight) | Torque on each hip-roll joint | Motor torque |
|---|---|---|---|
| 0.05 rad | 8 N | 5.9 N.m outward | 0.3 N.m |
| 0.10 rad | 145 N (34 % of weight) | 34.5 N.m outward | -3.8 N.m |
| 0.20 rad | 383 N (90 % of weight) | 82 N.m outward | -13 N.m |
| single stance, swing leg lifted, stance target 0.1 | 106 N | 22 N.m on the stance hip roll, -7.9 N.m on its hip pitch | |

- In stance the hull, not the motor, carries the hip-adduction moment.
- The eval monitor's 390 N standing / 730 N walking per side (SSR 1,600 and 2,000) is the policy's saturated -0.235 target
  pushing into this stop.
- Removing the pair changes nothing for a robot at the default targets (it falls at 1.41 s either way). It removes the
  stop only for a policy that pushes inward, which is every SSR policy so far.

**Other pairs over a walking sweep** (2,000 random poses: each leg hip pitch -0.9 to 0.5, hip roll -0.235 to 0.3, hip yaw
+-0.4, knee 0.1-1.4; arms at default):
- Pelvis / hip_roll penetrates in 33 % of poses (every pose with hip roll at -0.05 or beyond).
- Leg-on-leg pairs: knee-knee, shank-ankle, ankle-ankle, hip_yaw-knee, 0.1-1.5 % of poses, up to 104 mm.
  - These occur only when both legs adduct or cross, so they are physical: the real legs would collide too, with the hull
    adding a few mm.
- Arms: not in the sweep (default pose). The eval monitor's ACP log shows wrist / hip_pitch, hip_roll and hip_yaw hull
  contacts at 0.5-3.5 mm in 20-48 % of steps.
  - These were not checked against the real meshes. Wrist-on-thigh brushing is plausible on the real robot, so check it
    before excluding.
- Torso / pelvis and thigh / thigh: no contact in any swept pose.

**Correct collision set, and a fix for ACP, SSR and the eval** (shared MJCF; needs the user's or coordinator's decision
because all three use it):
1. Add `<exclude body1="pelvis" body2="left_hip_roll_link"/>` and the same for `right_hip_roll_link`. The joint limit
   becomes the stop, as on the hardware.
2. Keep every other pair: leg-leg contacts are physical.
3. Optional, more faithful: replace the hip-roll hull with a decomposition, or with primitives that leave the 5 mm gap.
4. Apply the same model in the eval and in all trainers. A policy trained with the artefact must not be scored on a model
   without it, nor the other way round.
5. Existing SSR (and Hiking, which shows the same contact on one side, 153-244 N) checkpoints rely on the stop. They need
   retraining or a fine-tune that starts with the hip-roll actions pulled off the bound. A plain resume falls, as r6c
   showed.

## 6. Ranked differences

(a) **Standing from scratch**

1. **Hip-roll hull stop plus large roll exploration** (headline 1). It gives the stander a braced, noise-proof hip. It
   takes the hip abductors out of the policy's hands, which makes every early step attempt (balance in single support)
   costly and noisy, and ankle roll at 0.485 rad per unit cannot replace them.
   - Evidence: action saturation [log]; hull vs real [meas]; std growth in every run [log].
   - How sure: high that the artefact is real and is used; medium that it is the main cause of standing.
   - Test: d1 / d4-style flat runs of 300 iterations with the exclusion, from scratch, against d4 (all Table 6 terms, which
     stood). CPU check first: hip-roll action and target in obs_check of any new checkpoint must not sit at the bound.
2. **Action scale against Table 14** (ankle roll, shoulder yaw and elbow 0.485 instead of 0.25 / 0.2), with init std 1.0.
   - Evidence: about 2x larger noise on the joints that set foot roll and arm swing [code]. Measured cost: the shuffler's
     action_rate and smoothness -0.30, slip -0.13, stumble.
   - How sure: medium.
   - Test: a d4 repeat with Table 14 scales (needs the user's agreement under the plant rule), or init std 0.5 on those
     joints only.
3. **G1 default pose on the X2** (headline 3): pelvis 0.641 m against the 0.68 m target; dof_dev charges walking.
   - How sure: medium-low on its own.
   - Test: the prepared `--default_from_ref` (r7) on flat ground, 300 iterations against d4.

(b) **Short, quick steps**

1. **No hip-roll control** (locked at -0.057): lateral balance is done by quick re-stepping and the 0.72 rad ankle-roll
   noise [inf from log and meas]. Test as (a)1, then watch step rate and swing width.
2. **Training noise**: training rollouts at std 0.8 mean (hip roll 1.9) give the logged 0.11 s air time and 5.5
   touchdowns/s, while the deterministic gait swings 0.18-0.22 s at about 2 steps per foot per second (SSR's per-bin
   table). Part of "quick steps" is exploration noise. How sure: high. Test: none needed; read deterministic probes, not
   training-log air time.
3. **The prior's speed filter and arm mask** weaken the style signal for swing timing. How sure: low.

(c) **Standing at slow commands (0.2-0.35 m/s)**

1. **No slow-walking experience at all**:
   - walking starts only at 0.3 m/s or more;
   - prior windows only at 0.2 m/s or more;
   - promotion needs about 0.37 m/s or more, because it requires reaching the tile end (§2), so slow commands are never
     rewarded by the curriculum;
   - one command per episode, so a slow-command episode always starts and stays near a stand.

   The paper's lin_vel kernel pays the same 0.85 to a stander at 0.2 m/s; the paper reports no slow-speed result
   (paper silent).
   - How sure: medium-high.
   - Test, cheapest first: promotion at half the tile (legged_gym rule); `--rsi_min_speed 0.1`; prior windows down to
     0.1 m/s; resample the command every 5-10 s (legged_gym default 10 s). A 500-iteration resume, with the per-bin probe
     at 0.2 / 0.35 m/s.
2. **Gait cost**: the same hip-roll and noise defects as (a). At 0.2 m/s a walker gains only 0.15 + 0.2 = 0.35 per step
   over a stander, so any stepping cost above that loses. The narrower kernel in r6d treats the symptom.

## 7. Follow-up: wrist / thigh contacts and a hull-vs-mesh scan of every pair (2026-11-12, CPU, MuJoCo 3.15)

Method: the same shared MJCF. Hull distance is `mj_geomDistance` on the collision geoms. Real distance is the unsigned
surface distance between the visual CAD meshes, both directions, vertex to surface. On closed (watertight) meshes a
vertex-inside test gives the depth, shown negative. Read 0.0 as "the surfaces cross"; the depth cannot be measured on
open meshes (hip_yaw, ankle, wrist, torso and pelvis meshes are open).

Scripts (scratchpad): `wrist_real.py`, `wrist_acp.py`, `pel_recheck.py`.

The re-check with this both-way method confirms pelvis / hip_roll: real 5.0-5.1 mm over hip roll 0 to -0.12 and 3.6 mm at
the -0.235 limit, against a hull of -10.9 mm. The artefact verdict stands.

**Arms at the default pose** (shoulder pitch 0.2 +- 0.3, shoulder roll +-0.2 or +-0.1, elbow -0.6; hip pitch -0.6 to
+0.2): every wrist / thigh pair listed below is **50 mm or more apart (hull)**. There is no contact, false or real.

**The ACP 3,500 contacts come from ACP's own arm posture**, not from the hulls. From `eval/data/handover/acp_s2_3500/obs_check.npz`
(50 frames, walking):
- shoulder roll at the inner limit (L -0.047, R +0.063, against default +-0.2);
- shoulder yaw L -0.61 / R +0.53 (default 0);
- elbow -0.26 / -0.29 (default -0.6).

This is the reference-data arm pose (Hiking audit #2; SSR port_audit point 1): the wrists hang beside the thighs.

At ACP's frames, minimum distance in mm (hull / real):

| Pair | hull / real | Verdict |
|---|---|---|
| right_hip_yaw / right_wrist_roll | -0.9 / 0.0 | real touch |
| left_hip_pitch / left_wrist_pitch | -0.5 / -0.5 (inside) | real |
| left_hip_roll / left_wrist_roll | -8.8 / -2.7 | real; hull depth 6 mm too deep |
| left_hip_yaw / left_wrist_roll | -2.3 / +4.3 | false at that frame, real 0.5 at others; hull up to 6.6 mm too close |
| right_hip_pitch / right_wrist_pitch | 1.9 / 2.1 | hull = real |
| right_hip_roll / right_wrist_roll | 4.4 / 1.8 | hull **under**-reports by 2.6 mm |
| right_hip_roll / right_wrist_pitch | 5.2 / 3.0 | hull **under**-reports by 2.2 mm |

The under-reporting comes from the hip-roll collision mesh being scaled 0.9 in y (`col_*_hip_roll_link`): the real part
is wider than its collision proxy.

**Conclusion**: the wrist contacts are real in that arm pose. Do not exclude these pairs. The fix is the arm pose: keep
the arms near the default (arm-masked or arm-cleaned reference, or an arm dof_dev term). No hull change is needed for the
wrists. If the hip-roll collision mesh is replaced (it is now excluded from the pelvis anyway), use the unscaled mesh,
decomposed, so it no longer under-reports wrist contacts.

**Every pair where hull and mesh differ by more than 5 mm in a walking range.** 400 random poses: each leg hip pitch -0.8
to 0.4, roll +-0.15, yaw +-0.25, knee 0.1-1.3; shoulder pitch 0.2 +- 0.3, shoulder roll 0.1-0.2.
1. pelvis / hip_roll (both sides): hull -5.7 mm against real +5.2 mm; **false contact** (now excluded).
2. Leg-on-leg, only with the legs crossed or adducted: knee / knee, ankle_pitch / knee, ankle_pitch / ankle_pitch,
   ankle_roll / ankle_roll.
   - The real meshes touch (0.0, or -2 to -3 mm inside the closed knee mesh), so these are **real contacts**.
   - The hulls overlap 10-57 mm, so the depth and the contact force are exaggerated. This is the "knee-knee 22 mm,
     hip_yaw-knee 31 mm" seen when r6c scissored.
   - Keep these pairs. A decomposed hull or capsules would give realistic depths.
3. Feet: the ankle_roll body collides only through its 12 sole spheres, with no shell. Foot-on-other-ankle contacts that
   are real (real 0.0 at a hull distance of +14.5 mm) are **missed**. This is minor in a normal gait; it matters for leg
   crossing.
4. shoulder_roll / torso: hull = real = 7.0 mm, constant. Torso / pelvis and thigh / thigh: no near contact.

Nothing else came within 15 mm in the walking range.
