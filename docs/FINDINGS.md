# Findings log

Kept by the coordinator (user, 2026-11-12: "the coordinator should take note of all these kind of interesting findings and log
them somewhere for future reference. the agents report to the coordinator what things are going wrong and how they are
addressing them"). One entry per finding: what was found, the evidence, what was done, what is still open, and where the
detail is. Newest findings are added at the end of their section. The running timeline is `state/STATUS.md`; lessons shared
between tracks are in `state/SHARED_LESSONS.md`; this file is the short list a reader starts from.

Status words: **settled** (acted on and confirmed), **acted** (changed, result pending), **open** (no fix yet),
**needs the user** (a decision is waiting).

## A. The robot model

### A1. False contact at the hips (2026-11-12) — settled, confirmed on the real robot
- **Found:** in the shared X2 MuJoCo model the convex collision shapes of the pelvis and each hip-roll link overlap by up to
  10.9 mm; the real meshes keep 3.6-5 mm of clearance over the whole hip-roll range. The real stop is the joint limit
  (0.235 rad); the false contact stops the joint at about 0.057 rad.
- **Evidence:** independent audit (`state/ssr_port_audit.md`, sections 5 and 7); the eval measured 750-800 N per side on that
  pair while ACP and SSR policies walked (the robot weighs about 425 N); with the pair excluded the same policies fell (ACP
  3,500 pair 0/50 on flat tests against 50/50 with it; SSR 2,000 7/50).
- **Effect:** every ACP and SSR checkpoint before about 06:50 UTC stood on a contact the robot does not have. ACP's reference
  clips ask for the blocked hip-roll range on 41% of frames, so its tracker could never follow them there.
- **Done:** corrected model `data/robots/x2/mjcf/x2_ultra_hipexclude.xml` (two `<exclude>` lines; old file kept); all tracks
  and the eval moved to it; every result names the model the policy trained on. Hiking's run 4 had already filtered the pair.
- **Which model file, and is there a newer one (checked 2026-11-12 at the user's request):** the collision shapes are not custom.
  The collision geoms in `x2_ultra.xml` are `type="mesh"` on the vendor's own STL files (`pelvis.STL`,
  `left_hip_roll_link.STL` scaled 0.9 in y, as in the vendor's URDF); MuJoCo collides the convex hull of each mesh, which
  fills the recess the hip-roll link sits in. The file is based on AgiBot's MJCF (model name `x2t2.5`), added 2026-04-16.
  The same file (sha256 4f6a4f5b...) is in `gwbc-v6`, `gwbc-supermimic`, `gwbc-echo` and `ssr-x2`, and the SuperMimic
  trainer (`gear_sonic/supermimic/assets.py`, MuJoCo Warp) loads it "as-is" for collisions, so the SuperMimic runs had the
  same false contact (not tested whether their policies leaned on it). `GR00T-WholeBodyControl` (branch `vla-training`)
  holds an older variant (b24e575a..., no exclusions at all). `gwbc-x2` has no X2 model file in its checkout. The only
  other variants are `data_local/media_shoot/para6/plant_20260813/x2_ultra_20260813*.xml` (a media-shoot plant, one
  exclusion). No newer or corrected X2 model exists in these repos. The SONIC (Isaac Lab) runs load the URDF with
  `enabled_self_collisions=True`, so Isaac builds its own hulls from the same meshes.
- **Pictures, simulator render and real-robot photos together:** `docs/findings/hip_false_contact.md`.
- **Render:** `docs/findings/hip_false_contact.jpg` (MuJoCo, static poses): hull distance +2.7 mm at the standing pose,
  -0.6 mm at 0.06 rad inward, -3.9 mm at 0.12 rad, -10.9 mm at the 0.235 rad limit.
- **Photos of the real robot (user, 2026-11-12):** `docs/findings/hip_real_1..4_*.jpg`, pelvis and hips from the front and
  into the recess, legs hanging apart. They show a visible gap between the pelvis shell and the hip parts at that pose, and
  the hip housing sitting inside a recess of the pelvis (the recess a convex hull fills). They do not show the hip rolled
  inward, so they support the finding but do not prove the clearance at the 0.235 rad limit. **Open:** a photo or feeler
  gauge with the leg pushed inward to its stop, robot unpowered on the hoist.
- **Second set of photos (user, 2026-11-12), leg pushed inward:** `docs/findings/hip_real_5..7_inward_*.jpg`. The white
  shell that closes on the pelvis is part of `left/right_hip_roll_link` (model check: the next link, hip yaw, stays 61 mm
  from the pelvis at the limit), so the photos show the pair in question. The gap is small, a few millimetres at most, and
  from the photos it cannot be told apart from light contact; one arc-shaped rub mark on the pelvis side may be a witness
  mark. The roll angle in the photos is not known. Consequence if they do touch at the stop: the corrected model still stops
  the joint at 0.235 rad through its limit, so it stays the better model as long as the real parts do not touch well before
  the stop (the old model stopped it at 0.057 rad). **Settled by the user's test (2026-11-12):** with the leg held at its inward stop "a
  playing card nicely slides all around without being blocked" (a card is about 0.3 mm thick). The real pelvis and
  hip-roll shell do not touch anywhere in the range; the joint is the stop. The exclusion in the corrected model is right.
- **Open:** leg-on-leg contacts when the legs cross are real but the hulls are 10-57 mm too deep (exaggerated force); feet
  collide only through sole spheres. ACP's wrists resting on the thighs were real (its arm pose), fixed in its data.

### A2. Neither default pose holds a stand with zero action (2026-11-12) — open
- **Found:** with the plant's PD gains and zero action the X2 is on the ground within about 2 s from the plant default pose and
  from the reference-mean pose (SSR CPU test; the audit measured 1.41 s). A from-scratch policy must balance from step 0.
- **Open:** gravity-compensated default targets as a 300-iteration test (queued in SSR); the paper says nothing about it.

### A3. Action scale differs from the SSR paper (2026-11-12) — parked by the user
- Ours is 0.485 rad per action unit on ankle roll, shoulder yaw and elbow (from the plant rule) against the paper's 0.25 legs
  and 0.2 arms; the other joints match. A test with the paper's scales is prepared behind a flag.
- **User, 2026-11-12:** "for now, let it go as is". On the review-later list (`docs/REVIEW_LATER.md`); revisit if SSR's
  sideways balance or arm motion looks jittery on the apparatus or the real robot.

## B. The eval

### B1. The stand check counted robots that were still moving (2026-11-12) — settled (user confirmed 2026-11-12)
- **Found:** a station counted when the pelvis was in the disc at standing height. One "success" was scored while the robot
  was stepping off the landing with zero load on both feet.
- **Done:** strict stand: for the whole 1 s the pelvis is in the disc, the feet carry at least half the weight, the pelvis moves
  under 0.15 m/s, and the robot is upright afterwards; the old count is reported beside it. The final stand may be retried
  like any station (a re-balancing step restarts the 1 s). `docs/EVAL.md`, "Strict stand".
- **Effect:** all earlier apparatus counts were upper bounds. Under the strict check the baseline scores 0/50.

### B2. Earlier baseline numbers cannot be reproduced (2026-11-12) — settled
- **Found:** 15/200, 4/200 and 6/50 for the 2 November baseline pair came from an eval adapter version that was never saved.
- **Done:** retired. With today's checked eval the baseline is 2/50 (old check) and 0/50 (strict). A ten-trial regression
  check now runs at the start of every eval run.

### B3. End-of-route command fault (2026-11-12) — settled
- After the last waypoint the harness sent the ACP adapter its own position with no heading. It affected only the old check's
  final stand. Fixed.

### B4. The last station sits next to a drop (2026-11-12) — route settled by the user; open for SSR and Hiking
- **Found:** button 5 is 0.30 m from the open landing edge (1.12 m drop). ACP's pair held a stand at four stations and never
  at this one: 0/50 on the route, 30/50 with the drop hidden from its terrain view or the station moved (tests, not results).
- **Done:** ACP added 78 training stands next to a 1.1 m drop; its next pair held 114 of 117. The route is unchanged.
- **User's decision (2026-11-12):** the station stays; "that's how the apparatus setup is and we will have to make it work".
  The user says a rail there prevents a fall. In the eval's model the landing (x 9.74-10.96, y to 2.42, deck 1.118 m) has one
  rail per side at 0.22-0.30 m above the deck, 0.11-0.12 m outside the edge; button 5 is 0.31 m from the north edge. A rail
  at shin height does not stop a robot stepping off. **Open:** the real rail's height (asked of the user).
- **Open:** SSR and Hiking have never trained a stand next to a drop.

### B5. Hiking's eval command did not match its training command (2026-11-12) — settled
- The eval kept turning the robot inside the arrival disc; Hiking's training commands zero within 0.4 m and a stand inside.
  The eval now follows Hiking's card exactly. Stop-and-stand went from 6/50 to 45/50 on the same checkpoint family.

### B6. The leaning board at the door is modelled as a solid wedge (2026-11-12) — needs the user
- **Found:** our apparatus has a leaning board / brace at (2.61, 2.10), next to the door side. The course's floor height map is
  made by rays straight down, so the space under the board is filled: the physics and every policy's depth see a solid wedge
  rising to 0.48 m where the real board has open space under it. Route legs 7-11 pass 0.19-0.21 m from it.
- **Effect:** unknown; it is in front-left of SSR at its door-side stalls (199 of 1,296 depth rays hit it). Same for all tracks.
- **Done:** a labelled TEST with the board as a tilted box is queued (SSR's stall seeds, ACP and Hiking for side effects).
- **Open:** whether the held-out course's model should be corrected (the user's decision).

### B7. The edge of a ladder course looks like a drop to a depth policy (2026-11-12) — settled
- **Found:** the ladder courses (steps 1 and 2) end in nothing: beyond the height field there is no ground, so a depth camera
  sees a void. SSR stops 1.1 m before the edge of the corridor course (119 of 1,296 depth pixels read 'far'; none with a test
  ground plane). Stop-stand, which ends 2.0 m from the edge, passes 50/50.
- **Effect:** SSR's corridor 0-1/50 and street 0/50, and Hiking's street stalls, may be this and not the policies. The two
  apparatus are walled rooms and are not affected. ACP's height scan is being checked.
- **Done:** a test with ground beyond the edge is queued; if it confirms, every ladder course is padded with flat floor and the
  affected rows are re-run. Every failed trial now records the nearest obstacle, the distance to the course edge and the command.
- **Confirmed and fixed (16:4x UTC):** with ground beyond the edge SSR's corridor goes from 0/50 to 20/20 and Hiking's street
  from 37/50 to 10/10. Every ladder course now has 3 m of flat floor around it; affected rows are being re-run, old rows marked.
  SSR's street still stalls mid-route (8 of 10), which is a different cause.
- **Lesson:** SSR's and the coordinator's explanations for this stop (slow command, falling command, corridor walls) were each
  wrong; the trace settled it.

### B8. Feet jam inside the floor's height-field cells (2026-11-12) — confirmed by the eval, effect being measured
- **Trigger:** the user asked for a render of a Hiking fall to rule out impractical physics; the coordinator found a 3,289 N
  foot contact in that trial's log. That figure itself is an inflated sum from one impact frame (the eval's force and impulse
  numbers add magnitudes over up to 38 contact points and are not to be quoted).
- **The real artefact (eval's replays):** the walk surface is a height field of 2 cm cells and the feet collide through 2 cm
  sole spheres with a soft contact. A sphere pressed more than about 1 cm into the floor escapes sideways into a cell and is
  squeezed by its neighbours: 100-137 contacts, spheres 19-24 mm deep with horizontal normals, 4-20 kN held for 0.2 s or
  more, on FLAT floor. Found in 7 of 18 kept ACP 7,000 trajectories, 5 of 22 and 5 of 20 for Hiking, 0 of 7 for SSR; almost
  all in the area beside the front 4x4.
- **Effect:** unknown yet. Trials with such an event followed by a fall within 3 s: ACP 5,750 30, 6,000 16, 6,250 22, 7,000 8;
  Hiking 3,000 10, 4,125 13; SSR 0 (indicative).
- **Test result (17:33 UTC, eleven trials):** a stiffer floor contact removes the jams (0 of 11 trials with a sustained load,
  against 7 of 18 and 5 of 20 before); the 4x4s as boxes do not (2 of 11). ACP's 8 failed 4x4 trials: 7 succeed with the
  stiffer floor, 5 with boxes; but at a 62.5% success rate about 5 of 8 would succeed by chance after any physics change, so
  this does not yet show an effect on scores. Hiking's three traced falls fail either way (the policy's own).
- **Decision:** the floor is corrected either way, because the jam is unphysical. A like-for-like 50-trial run with the stiffer
  floor for ACP, Hiking and SSR measures the effect; the correction is chosen when that and the independent audit are in.
- **Running:** a labelled test with a stiffer floor contact and with the 4x4s as boxes; an independent audit of all recorded
  apparatus trials (`docs/findings/sim_artifact_audit.md`). If outcomes change, every apparatus number is re-run.
- **ACP's training, measured (17:58 UTC):** clean on flat ground and on box props (depth under 14 mm, forces under 2 body
  weights). On its ramp height field about 3% of sole-sphere contacts jam sideways, and the foot's terrain box sinks up to
  25 cm with 9-19 kN. Only the ramp set is exposed (5-10% of training). It blocks ACP's import of the extended map, which is
  all height field, and may explain that import's unexplained falls. Three fixes are being tested on CPU.
- **SSR's training:** same class (2 cm spheres on 2.5 cm cells); its CPU check is planned, not yet reported.
- **Isaac (user's question, 2026-11-12) — evaluated by Hiking from its code, measurements pending:** nothing had been measured
  before. Possible: ghost contacts at seams, because each tile is one mesh of closed boxes without a union (stair flights,
  posts under rails, slabs under door frames); tunnelling through plates thinner than about 2 cm. Not an issue: leaning props
  are rotated boxes (not filled underneath); 8.9 cm posts and beams; the depth camera sees the same mesh as the physics.
  From the next restart (iteration 5,000) the training log reports force spikes over 3 and 10 body weights, feet sunk more
  than 1 cm and spawns in contact, per terrain type. Not built yet: a tunnelling check and whether spikes precede falls.
  Also noted: Hiking trains with the foot's convex hull and is scored in the eval on sole spheres.

## C. Training, all tracks

### C1. Slow commands: standing pays (2026-11-11/12) — acted (SSR), partly open
- The speed reward exp(-error^2 / 0.25) pays a stander 70% at a 0.3 m/s command. SSR stood at commands under 0.5 m/s; Hiking
  never trains under 0.45 m/s. Detail and options: `state/SHARED_LESSONS.md`, "Slow commands".
- SSR: command mix with slow and turn commands helped (0.2 m/s: 0.13-0.19 m/s); two tries of a relative-error kernel failed
  their checks (fast walking or slow turning collapsed).

- **Relative-error kernel dropped (SSR, 14:22 UTC):** both tries made the policy lose commands it already handled (stood at
  1.0 m/s) although the kernel there was unchanged: forgetting under a reward change, not a reward balance. Slow commands go
  on through command share and the motion prior.

### C2. A training "speed share" can be a fall (2026-11-12) — settled
- SSR's and Hiking's training speed numbers count lunges and falls as speed. A fall-back rule acted on such a number twice
  (SSR e3's pose choice; SSR's relative-kernel run thrown away at 0.595 against 0.60). Rule: walking is judged only by a
  deterministic probe.

### C3. Hiking: every restart cost half the terrain levels (2026-11-12) — settled
- Restoring levels moved each robot's tile while all robots restarted at once; the curriculum demoted them. A level pin for
  the first 1,500 steps fixes it (verified: 5.53 before, 5.56 after). The pin's first version crashed the logger when it
  released (a log name disappeared); fixed; a guard now falls back after two error exits from the same checkpoint.

### C5. A fall-back mark the base line itself fails (2026-11-12) — acted
- SSR's fall-back tested speed at a 1.0 m/s command from a stand. The unchanged line gives 0.00 at 2,450 and 2,600 and 0.67
  at 2,850 (16 robots), so two relative-kernel runs and one edge-stop run were judged on it wrongly. With the training
  speed share (C2) that is three runs lost to bad marks. Rule asked of SSR: marks relative to the base probed in the same
  job, only marks the base passes, enough robots: adopted 15:12 UTC. A fourth run (edge stops, second try) was mis-judged
  the same way before the rule took effect; edge stops are kept. Starts from a stand, 64 robots: 100% at 0.35-0.6 m/s;
  0.8 m/s 98% at 2,450 and 19% at 2,600; 1.0 m/s 0%. **Open:** starts at 0.8-1.0 m/s. The relative kernel is not ruled
  out after all (C1).

### C4. Fall-back tests must include every skill that matters (2026-11-12) — settled
- SSR lost turning in place during a change whose fall-back did not test turning. Turning, speed at 1.0 m/s, clearance and
  survival are now in every SSR fall-back test; Hiking's trials check the turn rate.

## D. ACP

### D1. Stage 1 was cut short and its data was mostly styled clips (2026-11-12) — settled
- Stage 1 ended at 2,250 iterations with the tracker at 91% (paper: about a day on 3 GPUs, 98.5%). The "flat" set had only
  about 95 plain walk or turn clips of 351 (87 bump, 79 injured, 22 crouch, 20 jump); jump clips failed 14 of 14.
- **Done:** styled and defective clips removed, 4x4 weight raised, arm fix against the real meshes, stands added. Paper eval
  with reference contacts: 93.0% at 4,000 (old body) -> 97.7% at 4,500 -> 99.5% at 6,500 (corrected body, cleaned sets).

### D2. The planner is the weak half (2026-11-12) — acted
- With reference contacts 99%; with planner contacts about 82% (flat 44-56%, blocks 45%). A smoother decode did not help; a
  planner fine-tuned on corrected-body rollouts did (83% against 74.5%).
- No training robot ever ran on the planner alone, so no training number showed what the scored pair does. 2% probe robots
  now do, and the hand-over is gated on them.

### D3. Apparatus results (corrected model, strict check) — open
- Pair 5,750: 114/200 = 57.0%. Pair 6,000: 79/200 = 39.5% (stalls short of the front 4x4). Pair 6,250: 72/200 = 36.0%
  (stairs better: 24 falls against 45; turn-around worse: 52 stalls).
- **The paper eval and the apparatus point opposite ways (2026-11-12):** 93% -> 99.5% on reference clips while the apparatus
  fell 57 -> 39.5 -> 36%. The paper eval cannot be the gate for the next stage on its own.
- The drop does not reproduce on ACP's clips or goal tiles, so it is specific to the apparatus; the eval is logging the
  planner's proposals at the stalls. Stairs: 45 of 86 failures at 5,750, 34 at the foot or first step (arrives still turning).

### D4. Where the pair fails on the apparatus, traced (2026-11-12) — acted
- **Turn-around after station 1 (planner):** the next waypoint is straight behind the robot at 0.36 m (heading change near
  180 degrees). The 6,000 planner proposes no step for 45 s; the 5,750 planner turns and walks on. Cause of most of the
  57% -> 39.5% drop (31 stalls there).
- **Step down on the ramps (tracker):** the 6,000 tracker stands before an 11 cm step down although the planner proposes
  valid steps; the 5,750 tracker takes them.
- **Stairs (approach, not climb):** in a test with a station at the stair foot, 13 of 15 trials that stood square climbed the
  whole flight; 29 of 45 fell on the diagonal approach and turn next to the first riser.
- **In the training data (ACP's count on set 2f):** turn-around goals 0.5% of frames; step-downs from a stand 45 segments;
  a turn onto a flight within 0.5 m of the first riser: none. The planner's share rising from 1.7% to 10% overwrote the
  rare turn-around skill within 250 iterations; per-set probe averages cannot see such a loss.
- **Done:** a 1 s stand before every double-support clip (data 2g, training since 14:39 UTC) for the step-down; an eval
  command variant `turnwalk` for the turn-around (labelled; the present rule stays the headline until the user decides);
  5,750 frozen as the reference pair, the three failing seed sets rerun on every new pair.
- **`turnwalk` on 50 trials:** 5,750 pair 18 (22 without it), 6,500 pair 22 (19 without it): no general gain; kept as a
  labelled variant, not run routinely.
- **First check of `turnwalk` (three trials):** the 6,000 pair turns around in 3 of 3 (stalled in 3 of 3 before); the 5,750
  pair gets worse (falls in 2 of 3, one during the turn). The rule is not a free gain; 50 trials each are queued.
- **Done (15:07 UTC):** the planner is frozen at its 5,750 state and only the tracker trains on, so the planner's online
  updates can no longer overwrite rare skills. Check: the turn-around seeds on the first frozen pair (7,000).
- **Swap test (50 trials each, a test):** 5,750 planner + 6,250 tracker 27/50 and the reverse 27/50, against 21/50 and 18/50
  for the original pairs. The turn-around stalls follow the tracker (0 with the 5,750 tracker under either planner, 4-8
  with the 6,250 tracker), so freezing the planner alone may not remove them; the stairs improve with either newer half.
- **ACP's trace (16s):** the two trackers' stances at the stand do not differ clearly; the newer tracker's pose and history
  tip the planner's occupancy for one foot under its 0.5 line, so no step is proposed: the two halves co-adapted on an edge.
  **Done:** the eval runs the turn-around and ramp-trough seeds on every ACP export as a gate; if the first frozen pair
  (7,000) stalls, the tracker restarts from 5,750 with the frozen planner.
- **First frozen-planner pair (7,000) passes the gate:** all three turn-around trials complete the whole route.
- **Station legs for 7,000 (first use of the user's measure):** 49 / 39 / 46 / 50 / 43 of 50; the weak legs are button 1 to 2
  (the turn-around at the front 4x4) and button 4 to 5 (stairs and landing). Their product, 60.5%, matches the full route.
- **7,250 failed the gate** (2 of 3 turn-around trials fell) with the same frozen planner: the tracker alone can regress
  between checkpoints 250 iterations apart. Newest is not best; selection by legs is being considered.
- **Probe gate works against a frozen planner (2026-11-12 18:15):** the gate lowered the share of planner contacts in
  training from 0.35 to 0.15 after 7,000, partly on noise in its baseline; the planner-only probe dipped at 7,250 (4x4 set
  0.939 to 0.909), matching the gate failure. Fix written (`--gate-backoff 1`, `--gate-rebase`), applied at the next
  restart. Selection is now by leg product among gate-passing pairs (champion 7,000). Detail: `state/acp.md` 17a.
- **Leg 5 falls (7,000):** all four on the first stair steps while the yaw command is at its limit; the robot turns onto
  the flight beside the first riser. Traces asked of the eval. Open.
- **7,000 on 200 trials: 125/200 = 62.5%** (5,750: 57.0%; ranges overlap): stairs 9 falls against 45, 11 turn-around stalls,
  51 of 75 failures in the 4x4 section. Provisional because of B8.
- **Where 7,000 fails in the 4x4 section (16 traced, provisional):** 8 drift east into the ramp cells around the turn-around
  at button 1; 3 fall crossing the front 4x4 on the way back although the step is planned (one 143 deg off square); 4 at the
  door side after the crossing; 1 on a wall before button 1. Mostly the tracker's motion around the turn-around.
- **7,000 on 50 trials: 26/50** against 22 (5,750) and 18 (6,250) on the same seeds; 1 turn-around stall (8 at 6,250), 3 stair
  falls (19 at 5,750). Freezing the planner worked. The largest loss is now falls at the 4x4s (13 of 24 failures).
- **Open:** the turn-around and the stair approach need the terrain stage (stage 3). Detail: `state/acp.md` 16q.

### D5. Extended map for stage 3: import built, one unexplained defect; contacts dropped on the ramp raster (2026-11-12) — open
- The user's tiles are imported as one height raster with 197 routes for ACP's goal environments. In the first CPU closed loop
  every extended type falls at levels 0 and 9 within a second and none at level 4; it does not reproduce with 4 robots, so it
  depends on how many robots run or their order. Being diagnosed; CPU checks due 2026-11-13 12:00 UTC.
- The running stage-2 log has 5,228 'height field collision overflow' lines on the existing ramp raster: contacts beyond 50 per
  pair are dropped in training today. Effect on the ramp clips unknown.

## E. SSR

### E1. Standing from scratch under the paper's rewards (2026-11-11/12) — acted
- From-scratch runs with the paper's plain reward table stand, with or without the hip fix (tests e1, e2). The reward does
  favour walking; learning from a stand settles there. A staged start (speed tracking and posture first) escapes it.
- Run r7 (staged start, corrected model): walks at commanded speed from 0.35 m/s, turns in place, 100% survive;
  `r7_002250` is the first policy of any track to pass the flat gate under the strict check (50/50 on all four tests).

### E2. Short shuffling steps (2026-11-12) — acted
- Feet lifted 2-4 cm. A reward for foot height times forward foot speed raised clearance to 6-9 cm in 500 iterations.

### E2b. Slow commands cost a route (2026-11-12) — open
- `r7_002250` passes log, wedge, stairs up and stairs down 50/50 and scores 0/50 on the corridor: every trial stops 0.27 m
  short of the last station, where the command falls under the speed it walks at (C1 in a result).
- Routes at 2,250 (standard command): home 44/50, street 0/50 (45 stalls short of a waypoint): the same cause.
- **Done:** the eval adds a labelled command variant `floor40` (0.40 m/s floor until the disc, zero inside, yaw at least
  0.6 rad/s); `standard` stays the headline and both are reported (coordinator's decision, needs the user's view). The
  training fix continues through command share and the motion prior.

### E2c. Slow walking was cut from the motion examples (2026-11-12) — acted
- The motion prior's 0.2 m/s filter removed all slow walking and slow turning. Since 15:07 UTC the prior includes windows
  from 0.05 m/s and turning windows (flat prior 656 s -> 853 s). Checks at +100 and +250 against the base checkpoint.

### E2d. First apparatus score: 0/50, stopped by slow turns (2026-11-12) — acted
- `r7_002250`, standard command, strict: 0/50 (49 stalls). It crosses the front 4x4 and holds button 1 in 42 trials, then
  stalls on the next leg (a 90-degree left turn to a waypoint 0.91 m away, near the door wall and the 4x4).
- **Cause (eval traces of five stalls, 15:45 UTC):** not a failed turn. The robot has turned, is commanded forward at
  0.25-0.37 m/s and stands still 0.4 m from the door-frame posts; with a 0.40 m/s floor it still stands there. It does not
  walk toward a wall or post close beside its path. One stall was the slow command 1 cm outside a station. The eval's
  first guess (no turn under 0.4 rad/s) was wrong; SSR's probe showed it turns at 0.97 rad/s when asked for 1.0.
- **Reproduced and explained (SSR, CPU):** on flat ground with obstacles the robot can only see, a wall 0.25 m beside the
  path stops 50% of robots, 0.35 m stops 25%, nothing closer than 0.35 m stops none. Training ended an episode when any
  limb came within 6 cm of an obstacle (17-21% of episode ends), walls only ever lined a straight lane, and slow or
  turning commands were given only on open ground: near a wall the policy expects to fail, and standing is cheap.
- **Done:** the 6 cm band becomes a penalty instead of an episode end (starts after the current check). Then the user's
  extended terrain map with walls as real boxes and routes, which SSR does not use yet.
- **It had grown:** at 3,250 a wall 0.25 m beside the path stops 100% of robots (50% at 2,250).
- **Extended map wired in (16:12 UTC):** 100 tiles and 267 routes (tight spots, log corridors, homes, street front, stairs) for
  21% of robots, commanded as the eval commands; starts on the GPU after the margin check. Walls stay see-only for now.
- **First check of the margin change (3,350):** a wall at 0.25 m stops 75% (100% before); a post or stub close ahead stops
  none (25% and 12% before); foot clearance 10.5 cm, past the 10 cm gate.
- **For the other tracks:** an early-termination margin around obstacles teaches refusal; check Hiking's and ACP's.

### E2e. A speed floor does not end the stops inside the station (2026-11-12) — open
- With a 0.40 m/s floor the corridor is still 1/50: the robot slows from 0.6 to 0.40 and then stands 1.5 cm outside the
  disc, although from a stand it walks at 0.35 m/s. SSR's probe rules out the falling command: on open floor steps from 0.6
  down to 0.30 m/s never stop it. Not the walls either: the eval found the course's edge 1.1 m ahead reads as a void in the
  depth view (B7).

### E2f. First extended-map run trained on misplaced routes (2026-11-12) — settled
- A bug shifted every route 2 m off its tile, so route robots were spawned into walls and kerbs for 460 iterations. Found by
  replaying route ends on CPU; fixed, a start filter added (170 of 267 routes kept), the run discarded and restarted from 3,550.
  The per-terrain successes reported from that run meant nothing. **Lesson:** check a few route start and end points against
  the tile in a picture or a replay before the first GPU run (Hiking's route build did this on CPU).
- **Wall refusal measured properly (64 robots):** at 3,550 a wall at 0.20 / 0.25 / 0.30 / 0.35 / 0.45 m stops 78 / 66 / 55 / 55 /
  39% of robots; 0% with no wall. This table is the measure for the restarted run.

### E3. Still open
- Slow starts (0.2 m/s) and slow turns (0.4 rad/s); clearance gate 10 cm; no apparatus score yet.
- Stands next to a drop: edge stops training since 2,850 (14:17 UTC), but tile drops are at most 0.40 m against 1.1 m.

## F. Hiking

### F1. Port defects (2026-11-12) — settled
- Seven defects found by an independent audit (`state/hiking_port_audit.md`); run 4 has all the fixes.

### F2. Stopping, turning, overspeed (2026-11-12) — acted
- No turn in place (the paper trains it on 5% of robots): 15% turn commands added; it now turns at the commanded rate.
- Nothing in the reward penalised motion under a stand command; exact stand commands (10%) shortened the stop.
- It walks faster than commanded as the terrain gets harder; an overspeed penalty is training since 3,250.

- **The overspeed penalty made it worse (eval about 14:15 UTC):** `model_3500` walks 0.33 / 0.49 / 0.59 m/s at 0.2 / 0.35 / 0.5 commands
  (3,000: 0.20 / 0.34 / 0.48) and turn in place fell from 18/50 to 2/50 (drifts 0.4-0.7 m). `model_3000` stays the reference.
- **Diagnosis (Hiking, 14:20 UTC):** the penalty is off under a 0.1 m/s forward command, so it did not cause the turning drift;
  nothing in the reward charged sliding while turning in place; 0.2-0.35 m/s is below the 0.45 m/s training floor; the
  overspeed began at 3,125, before the penalty. **Done:** a penalty on ground speed whenever the forward command is zero
  starts at 3,875; slow commands (floor 0.15 m/s) and slot / drop stations follow, one change each.
- **Open:** Hiking has no per-command probe in Isaac, so Isaac against MuJoCo at slow commands is unmeasured.

- **The sliding penalty worked (`model_4125`):** stop 0.28-0.29 m, stop-and-stand 50/50, speed within 7% from 0.35 to 0.6 m/s
  (the overspeed is gone), turn in place 32/50 with its own command (18 at 3,000; gate 50). Still open: drift of 0.4-0.5 m
  in slow turns; the eval's plain 'floor' command fails the turn test (1/50) by its shape and is dropped for Hiking.

### F2b. How the walk looks (user's review of `model_4125`, 2026-11-12) — watch
- The user: the walk is good overall, but the torso leans backward while walking. The coordinator also saw feet placed almost
  on one line with the legs near crossing, knees bent throughout and little arm swing. The reference motion leans forward
  (waist about +0.23 rad). **Open:** measured torso pitch and stance width per checkpoint (asked of Hiking and the eval); what
  in the reward allows the backward lean. Watch item, no restart.
- **Measured (eval, `eval/gait_stats.py`, 50 flat-walk trials each; forward lean positive):** Hiking 3,000 torso -16.6 deg,
  pelvis +3.5, stance width 0.230 m; Hiking 4,125 torso -15.5, pelvis +4.4, width 0.211 m; SSR 2,250 torso +3.6, width
  0.221 m; ACP 5,750 torso +11.0, width 0.189 m. Hiking bends back about 20 deg at the waist, unchanged from 3,000 to 4,125.
  No track crosses its feet (0 of 4,803 touchdowns): the coordinator's 'near crossing' reading of the frames was wrong.
- **User's hypothesis:** the policy leans back to tilt the chest camera up and see more terrain ahead. **Supported by the
  camera geometry (eval):** upright, the depth view reaches 1.66 m ahead of the toes (the user's requirement is at least 2 m,
  from about 10 cm behind the toes); at the walking lean it reaches the 3.0 m clip, and the toes leave the image.
- **User's prediction, not confirmed at 3,000:** it does not lean forward near stairs (-18.7 deg in the last 0.4 m before the
  first riser, -15.5 on the flight, -15.4 at the top edge going down). SSR leans slightly forward on obstacles.
- **Confirmed by Hiking (16:25 UTC):** the waist sits on its backward joint limit (-18 deg) in every checkpoint since 1,000,
  also when standing; the reference leans 11.6 deg forward, where the camera would reach only about 1.0 m. Nothing in the reward
  resists the lean (the style reward sees only the 12 leg joints, which is also why the arms barely swing). The trained mount
  matches the eval's numbers. 2 m upright needs about 67 deg of vertical field against the camera's 58; pitching the camera up
  about 5 deg gives toe tips to 2.0 m. **Needs the user** (the real mount): `docs/USER_TODO.md` 5.

### F2c. The extended map enters training (2026-11-12 16:25 UTC) — acted
- v4h continues from `model_4500` on the paper's 20 columns plus 12 extended-map columns (all ten levels) and 2 velocity
  columns; the construction check passed. Hiking ends episodes only on real failures, not near walls. **Open:** the trial at
  +31; the map's checked routes are not yet used as start / target pairs; no stations next to drops or in slots yet.

### F3. Apparatus (2026-11-12) — open
- `model_3000` passes every level-1 obstacle (49-50/50) and the home route (48/50), and scores 0/50 on our apparatus: it
  never gets past the third station, in the slot behind the front 4x4 (narrow slot, 90-degree turn and the log together).

- `model_4125` (before the extended map): every single obstacle 49-50/50 and the street route 50/50, home 31/50, apparatus
  0/50 again; 21 of 35 falls at the turn out of the doorway toward the front 4x4 (3 at 3,000). Better flat walking did
  not help where the course is tight; the extended map's doorways, homes and tight spots are the training answer.
- **Traced (three falls):** it overshoots the waypoint after the door by about 0.35 m into the corner between the south wall
  and the front 4x4, turns on the spot with its feet against the beam (6-8 foot hits, an arm on the wall), then backs over
  the beam and falls. Its stop distance on flat is 0.28-0.29 m. Hiking's first reading (turning drift in the doorway) was wrong.

### F4. The paper's hardware results use specialists (2026-11-12) — open
- Stairs, platforms and running were fine-tuned for 40,000 more iterations each. A general policy may fall short on stairs.

## G. How the team works

- **A silent export failure blocked a time-critical check (2026-11-12):** ACP's hand-over script looked up a data set by a
  name that the 2g data renamed; every export since failed with a line in a log nobody read, while the eval waited 25
  minutes for the 7,000 pair. Export logs are now read at every refresh, and the watcher must report failures.

- **The eval's steps 1 and 2 use the user's extended map; no running training does (2026-11-12).** SSR trains on a
  paper-style generator with see-only walls along straight lanes, Hiking on the paper's open terrains, ACP's stage 2 on clips.
  A policy first meets walls, doorways and slots in the eval. **User, 2026-11-12: the map belongs in the training runs.**
  Hiking's own start condition for it was met at `model_3000` and went uncalled (a coordinator miss). Both tracks now have it
  as their next change.

- **One place stops all three tracks (2026-11-12):** the turn-around right after button 1, in the slot behind the front 4x4.
  ACP's planner proposes no step there, SSR stands still next to the door frame just after it, Hiking stalls or falls in the slot. Turning on the
  spot next to an obstacle belongs in every track's training.

- **Launch incident, SSR 14:14-14:17 UTC (2026-11-12):** a watcher script with a syntax error was started anyway because the
  syntax check and the launch were chained with `;`; killing its tmux session then killed the supervisor and left a trainer
  running alone for a minute. Nothing lost. Rules now in `state/ssr.md`: `bash -n && tmux new-session`; never kill a tmux
  session that holds a supervisor; stop runs only through STOP. Applies to every track.
- Track agents decide in their own track and report what went wrong and how they fixed it (`docs/USER_RULES.md`).
- Agents' hand-back reports arrive once per turn; later ones are lost, so decisions are also written to `state/<track>.md`.
- A rare skill can vanish between two checkpoints while every average holds: keep a frozen reference checkpoint and rerun
  its known failing and passing seeds on each new one (ACP, 2026-11-12).
- Heartbeats do not show crashes; the refresh reads each supervisor log.
- Agents' own clocks drift 10-20 minutes ahead; the node clock (`date -u`) is the reference.

### B9. Independent audit of the eval's physics (2026-11-12) — fixes in progress
Report: `docs/findings/sim_artifact_audit.md`. It corrects B8 in three ways.
- **Mechanism:** the trap depends on touchdown speed (7.15 mm of sink per m/s; caged above about 3 m/s), not on the 4x4s as
  such. ACP's stairs are worst. A foot was held 0.3-0.7 s in recorded trials and 43.5 s in one control run.
- **Readings:** the harness's force figures sum opposing contacts and are wrong by up to 30 times. Quote no force until fixed.
- **Effect on scores:** small and not separable at 50 trials: an identical-physics control flips 21 of 50 outcomes (26 to 31 of
  50), the no-trap floor gives 34. About 3-6 of ACP 7,000's 24 failures are likely affected; Hiking's and SSR's are not.
- **Other defects found:** the landing is modelled twice (weight on feet reads 0.60, which can fail the strict stand); a head
  clamped between two wall boxes scored as a stall; soft joint limits; a brace rasterised as a wedge at (3.4, 1.1).
- **Done about it:** physics floor of planes and boxes (raster kept for scans and depth), net-force readings, automatic flags,
  the landing's duplicate box removed, every test variant run with a control. Then a new regression reference and re-scores.
- **Lesson:** with 50 trials, differences under about 20 points mean nothing; compare by rates with a control.
- **Open:** the same trap is possible in SSR's and ACP's training simulators (height fields with small sole spheres); a drop
  test in each is asked for.

### C6. The extended map's stair routes are almost all straight-on standing starts (2026-11-12) — being fixed
The user asked whether side approaches to stairs exist on the map. Measured on the 121 routes of the 80 stairs tiles (raster and
route files; script in the session scratchpad, `stair_routes.py`):
- Straight, L and U flights (80 routes): every route starts about 0.44 m in front of the first step, already facing it. No side
  approach, no turn before the first step, no descent.
- Amphitheatre (41 routes): 28 straight-on, 2 at 10-30 degrees, 11 at 30-60 degrees; 9 descend. Four turn 45 degrees or more
  within 1 m of the first step, none within 0.5 m.
- On the apparatus the robot reaches the stair foot 75-81 degrees off the flight and must turn and climb at once (ACP's leg-5
  falls, F-section). Nothing on the map trains that.
- **Done about it:** an additive route set (`extended_v1_1_stair_routes`: side left and right, oblique, far straight, descend) is
  being built and checked with `check_routes.py`; tracks load it when it is verified. The apparatus route is unchanged (user).

### C7. Only 8 of the user's 53 Pico stair takes are in the training sets (2026-11-12) — being fixed
The user recorded approach shapes on the props with the Pico (straight, side up and down, cross, C, L, S). Checked on the
shared filesystem and in `C:\Users\PC\projects\pico-captures` (README, 2026-10-30 session):
- **4x4 step-overs (`tp_t4x4u_y20.pkl`, 56 takes):** the variety is there: 17 straight, 19 L turns, 9 sideways, 4 C or U turns,
  3 S, 4 oblique. Ramps (15 base takes): 10 straight, 3 oblique, 2 L. 2x2 logs (10): 6 straight, 2 C, 1 L, 1 oblique.
- **Stairs (`tp_stairs_x10.pkl`):** only 8 base takes of the 53 recorded and retargeted (11:36-11:47): six straight single
  steps of about 0.5 m and two descents with an 83-89 degree turn. No take that arrives from the side and climbs.
- 45 retargeted stair takes are unused; the reason is not yet known (only 7 passed into `x2_physics_v1/stairs`).
- **Done about it:** an agent is finding the reason, classifying all 53 by shape and building `tp_stairs_all_v1.pkl` for the
  tracks. With C6 (map routes) this is the training answer to the stair-approach falls; the apparatus route is unchanged.

### E2g. SSR wall refusal: the map's tight routes taught it (2026-11-12) — fix running
- The probe at 0.25 m asked the impossible (limbs reach 0.34 m from the centre line). At passable distances refusal grew on the
  extended map: stalled share at 0.40 m 0.38 -> 0.56 -> 0.84 over 3,550 / 3,800 / 4,100.
- Cause: on tight_spots and homes 80% of routes pass within 0.34 m of a wall and all within 0.40 m; touching a wall ends the
  episode, so those routes always ended in a termination. The route filter had only checked the route start.
- Fix: run 3 from 3,550 with routes of at least 0.39 m clearance along their whole length (118 routes: log corridors, street
  front, stairs). Homes and tight spots return when walls are physical boxes and contact is a penalty, not an ending.
- SSR's own simulator does not trap feet (drop test at 1-4.5 m/s: at most 0.07 s deeper than 1 cm).
- Open: the same clearance question for Hiking's route training; walls as boxes for SSR.
- **C6 update (19:50 UTC):** two additive sets exist. `extended_v1_1_stair_routes`: 526 routes on the original tiles (side legs 0.1 m
  from the tile border). `extended_v1_2_stairs_inset`: the 60 straight / L / U tiles with the stair foot 2.5-3.0 m inside the
  tile and 700 routes in all shapes at 0.40 m clearance or more; 17 straight tiles lose 1-6 steps per flight to fit. Contact
  sheets: `docs/findings/stair_routes_extra_contact_sheet.png`, `stairs_inset_contact_sheet.png`.
- **Checker defect:** `route_blocked` (`review/new_terrains/check_routes.py`) reads the start surface with a ray from z = 1.0 m, so
  a route starting on a landing above 1 m is read as starting inside the solid and fails with "obstacle at body height" (45 of
  80 descents; all pass on a mesh lowered by the start height). Worked around and flagged per route; the checker fix (cast from
  above the mesh) is open.
- **C7 update (20:00 UTC):** cause found: the stair prop was fitted only at the second of the two recording places, so place 1's 45
  takes never entered a set. `tp_stairs_all_v1.pkl` now holds 44 takes: straight up 10 and down 9, single steps 12, down then
  turn 4, side approach then up 2, C up 2, S up 2, C down 2, S down 1; six clips turn 63-102 degrees on the floor and climb.
  Seven takes fail the 2 cm contact check, two of which are in the current `tp_stairs_x10`. Open: tracker replay on ACP's GPU,
  variants, loading in each track.
- **B9 update (19:00 UTC): fixed.** The planes-and-boxes floor (`PHYS_FLOOR=solid`, `eval/phys_floor.py`) matches the raster to 2.4 mm
  (p99) and removes the trap: trapped-foot time on ACP 7,000's 50 trials fell from 8.3 s over 25 trials to 0.11 s in one. The
  score did not move (27/50 against 26/50 control), so the failures at the 4x4s and stairs are the policy's, not the floor's.
  It is the default floor from now; headline numbers are being re-run on it. Open: joint-limit flag threshold.

### F5. Hiking's route training: contact rule and Isaac artefacts (2026-11-12) — restart v4j being built
- No refusal near walls (levels on tight spots and homes rise), unlike SSR, because only a torso touch ends an episode.
- But contact is not graded: a hand brush costs the same as a hip, and a torso touch on a wall ends the episode. The user's rule
  (graded by body part, sidestepping allowed) goes in with restart v4j.
- Defects: 23.5% of robots on homes tiles spawn touching a prop; 8.6% on tight spots carry a contact over 10 times body weight
  (cause not yet split between wall pressing and mesh seams); a 148 kN force seen on the paper's columns.
- Sunk feet are under 1% except on wedges (16%, probably a measuring effect) and street (3.7%). Tunnelling not yet checked.
- **C6 update (19:25 UTC), other props:** the original routes on beams, beam fields and blocks were one pattern with the turn at the
  tile centre; 21 of 89 crossed no prop. `extended_v1_1_prop_routes` adds 1,116 checked routes (straight, oblique, side, L, C, S,
  along, cross) on 170 tiles. Not possible without new tiles: side entries on the 28 corridor tiles walled on both sides, C and
  S at the long ramp (its foot is 0.4 m from the tile edge). Sheet: `docs/findings/prop_routes_contact_sheet.png`.

### C8. Pico motion catalogue: two data findings (2026-11-12) — open
Catalogue at `terrain-park/catalog/` (135 takes, 6 props, 36 categories, 47 MuJoCo GIFs; uncommitted). It found:
- **Ramp prop mismatch:** `props/ramp.json` (shifted +30 m) does not match the training height map `tp_ramp_hmap_x30.npz`: the
  height map's ramp is about 0.8 m wide against 2.2 m in ramp.json, and its slope direction is turned 11.5 degrees. Whichever
  track trains on the ramp clips with ramp.json geometry is using the wrong prop. To check with ACP and SSR (both use the ramp set).
- **Ramp takes:** 7 of the 15 base takes walk over the ramp side to side (start and end on the floor); only 8 go up or down it.
- **Heading offset:** in forward 4x4 step-overs the travel direction sits a steady 10-20 degrees (median 14) left of the pelvis
  heading on both capture days, while sidestep takes are square: probably a retarget yaw offset. Matters for any track that
  conditions on heading against a prop. Open.

### F6. Hiking v4j did not start: a letter missing from an option whitelist (2026-11-12) — fixed
- The 19:22 restart's construction check failed with `AssertionError: unknown HIKING_X2_OPTS: ['g']` in `staged.py`; the CPU build
  test had passed because it never imports that file (it needs Isaac). The fall-back resumed v4i, but without a crash guard.
- Fixed: `g` allowed; a CPU pre-flight now checks the launcher's options against `staged.py` and the data paths before stopping the
  running job; the fall-back starts a guard. Lesson: a CPU test must exercise the same entry checks the GPU launch will.

### C9. Retarget carry-over bug in the Pico pipeline (2026-11-12) — fixed in v1_1
- The user spotted a stooped robot in a stair GIF (take 113941) while the raw recording shows them upright. Cause: the retarget tool
  ran takes in sequence and started each take's joint solver from the previous take's last pose, so a take could begin (or stay)
  with joints stuck at their limits.
- Affected: stairs 113941 (3.4 s), 2x2 take 211326 (whole take; dropped), all 6 driveway takes (whole take; root jumps of 3-4 m/s).
  All other takes match their recordings within 1-5 degrees of lean.
- Fixed with a fresh solver per take (`terrain-park/tools/fix_v1_1/`); corrected `_v1_1` sets on the shared filesystem. ACP and SSR
  still train on the 2x2 set that contains 211326; they switch at their next data change.

### E2h. SSR's apparatus legs: three refusals and a weak turn on a side slope (2026-11-12) — v13 running
- Leg 1: stops 0.375-0.385 m from the wall at button 1, just short of the 0.385 m the reach radius needs (no training route ends
  near a wall). Leg 2: refuses to step back over the 4x4 after the turn-around. Leg 5: at button 4 on the 16-degree side-sloped
  wedge the turn in place stalls 25-45 degrees short (0.13-0.19 rad/s against 1.0 commanded), the arm hits the wall, the robot backs
  off the course.
- v13 adds wall-station routes, the stairs-inset and prop-route tiles and the user's corrected and mirrored Pico takes. Open: no
  side-slope tiles in training for turning on a slope.
- **B9 update (21:40 UTC):** on the solid floor ACP 7,000 scores 115/200 = 57.5% (old floor 62.5%, overlapping). A new defect of
  the solid floor: ramp-slab sides stand 1.5-2 cm in front of the edge in the scanned height field (slabs reach half a cell past
  their cells), where 10 of 15 shin hits happen. Being trimmed and re-checked; stair risers to be probed.
- **C1 update (22:10 UTC), and a coordinator miss:** the slow-command fault found on 2026-11-11/12 was tried three ways (more slow
  and turn commands: 0.2 m/s went 0.13 -> 0.00; a relative-error reward twice: dropped after the policy forgot fast walking; slow
  windows in the motion prior at 15:07) and then left "open" while the work moved to walls and the map. Nobody tracked the slow-
  command probe afterwards: SSR made 0.18 m/s at a 0.2 command at 2,250 and 0.00 at 4,300, which is why stop-and-stand fell
  50 -> 2 of 50 and why stations are missed by a hair. The eval's `floor40` variant hid it from the headline work. Now: SSR v14
  carries a creep reward for 0.10-0.35 m/s commands; the slow-command probe (0.15 / 0.2 / 0.35 m/s) is a key signal the
  coordinator reads at every refresh for every track, and a drop is an intervention.

### B10. MuJoCo Warp mis-collides on a large height field (2026-11-12) — fixed in ACP, being checked in SSR
- ACP traced its unexplained falls on the extended map to MuJoCo Warp: on one 2,260 x 1,819-vertex height field a flat platform
  produced a 9 m deep contact and threw a foot 240 m. MuJoCo C and any crop of the raster are correct. Fix: split the raster into
  fields of at most 400 x 400 vertices. SSR also trains in MuJoCo Warp on a large map raster and has 6-10% "diverged" episode ends.
- ACP's own training physics also failed the drop test (foot box sank 209 mm, held 0.11 s); fixed by taking the foot box off the
  height field and stiffer terrain contact (`ACP_HF_FIX=1`), in from the 06:00 restart.
- **B10 update (23:10 UTC):** confirmed in SSR: one 7,601 x 3,361-vertex field; 10.8% of SSR's episode ends since v12 were simulator
  blow-ups counted as terminations. Split into <= 400 x 400 fields: 29 of 150 -> 1 of 144 in a CPU replay. SSR resumes with the
  split; ACP has it already.

### F7. Hiking v4k did not start: motion file saved with the wrong joint names (2026-11-12) — fixed
- A reused variable in the motion builder saved the clip names as `joint_names`; the GPU construction check died on the first
  joint lookup but reported exit 0. Second failed start of the day (F6 was a missing option letter). The CPU pre-flight now runs
  the motion reader's own checks on every motion file. Lesson: a check that can exit 0 after a traceback is not a check; the
  launcher must read the error line, as it now does.

### B11. First simulator pre-flight: no build passes yet (2026-11-13) — owners assigned
- MuJoCo Warp still gives contacts 10-96 cm deep at slopes, edges and field borders with 400 x 400 fields (MuJoCo C: about 1 cm);
  chunks of 100 fix foot drops on ACP's extended map. Feet trap at stair edges in SSR's terrain also in MuJoCo C (the B8 trap).
  Hands and wrists sink 55-182 mm into terrain in rollouts. Fixes: terrain solref 0.005-0.01 with priority 1, smaller chunks,
  boxes for ramps and edges where needed; each track at its next planned restart. Reports: `tools/simcheck/reports/`.

### E3. SSR's motion prior ignored the paper's Table 11 (2026-11-13) — coordinator miss, fix planned
- The paper (no code or data released) gives the prior per terrain group: slope 261.3 s, stairs 295.8 s, discrete 136.0 s, gap
  174.5 s, platform 181.6 s; each group = flat forward + turn (+ backward for slope) + its traversal; 71 clips, 521 s unique.
  Our port recorded this, then justified replacing it wholesale with "the paper's data is not released": that explains the
  source, not the sizes or the composition. Result: about 4,700 s, BONES-SEED heavy, a separate flat group. SSR's style reward
  sits at 0.35 since iteration 1,600 (the user's G1 Hiking run climbed 0.39 -> 0.62).
- Fix: amp_v5 to Table 11 (Pico + the paper's Hiking reference retargeted to X2; no BONES-SEED). And a pass over every deviation
  in `state/ssr_port_audit.md`, separating "paper does not specify" from "paper specifies and we did not follow".
- Lesson: "data not released" is never a reason to drop a specification the paper does give.
- **E3 evidence (02:05 UTC):** nearest-reference analysis of eval trajectories: SSR and Hiking policies sit 2.3-2.6x farther from
  any reference window than reference clips sit from each other; walking windows almost never come close (0-1%), turns and
  side-steps not at all. The gap is posture: SSR holds its arms out and barely moves shoulders and hips (0.15-0.26x the reference
  spread); Hiking's waist pitch sits 4.7 reference standard deviations back and nearly frozen (the backward lean the user noted).

### C10. The G1-to-X2 retargeter returns frozen frames on CPU (2026-11-13) — worked around
- `soma-retargeter/batched_g1x2_driver.py` on CPU: per-frame `.numpy()` results share one buffer and the feet stabiliser's `joint_q`
  is not a view of the model, so every output frame was identical. Fixed at run time (`hiking_release_x2/tools/cpu_fix.py`); the
  retargeter repo is unchanged. Anyone running it on CPU must use the fix.
- **F6/F7 follow-up (2026-11-13):** a third Hiking start failed only on the GPU (v5: style-weight wrappers declared `**kw`, which Isaac Lab's
  signature check rejects). Each failure so far was a check the CPU pre-flight did not reproduce (option whitelist, motion joint
  names, observation-function signatures); each is now in the pre-flight. Lesson: a CPU pre-flight must mirror every validation
  Isaac does at construction, not only test the new code on stubs.
