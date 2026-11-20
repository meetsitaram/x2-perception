# Findings log

Kept by the coordinator (user, 2026-11-12: "the coordinator should take note of all these kind of interesting findings and log
them somewhere for future reference. the agents report to the coordinator what things are going wrong and how they are
addressing them"). One entry per finding: what was found, the evidence, what was done, what is still open, and where the
detail is. Newest findings are added at the end of their section. The running timeline is `agentic/state/STATUS.md`; lessons shared
between tracks are in `agentic/state/SHARED_LESSONS.md`; this file is the short list a reader starts from.

Status words: **settled** (acted on and confirmed), **acted** (changed, result pending), **open** (no fix yet),
**needs the user** (a decision is waiting).

## A. The robot model

### A1. False contact at the hips (2026-11-12) — settled, confirmed on the real robot
- **Found:** in the shared X2 MuJoCo model the convex collision shapes of the pelvis and each hip-roll link overlap by up to
  10.9 mm; the real meshes keep 3.6-5 mm of clearance over the whole hip-roll range. The real stop is the joint limit
  (0.235 rad); the false contact stops the joint at about 0.057 rad.
- **Evidence:** independent audit (`agentic/state/ssr_port_audit.md`, sections 5 and 7); the eval measured 750-800 N per side on that
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
- **User, 2026-11-12:** "for now, let it go as is". On the review-later list (`agentic/docs/REVIEW_LATER.md`); revisit if SSR's
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

### B-app1. Our apparatus: the eval's physics geometry does not match terrain-park's CAD export (2026-11-16) - needs the user
- **Found:** the eval scores "ours" on a raster + wall boxes from loco-manip-apparatus `scene_x2_env_mobility.xml`
  (`apparatus/ours/eval/cad_walls.json`); terrain-park's CAD export (`apparatus/ours/sim/apparatus.xml`) differs: 377 of 437 tall
  CAD parts have no wall box within 8 cm; both 4x4s sit ~7.5 cm (3 in) further from the door in the eval (front 4x4 centre
  x 3.04 m) than in the CAD (2.965 m), at a slightly different angle; the CAD's z = 0 is the venue floor, 0.108 m below the
  eval's deck top.
- **Evidence:** ray casts of both models over the deck (deck offset median +0.108 m over 369 points), box matching, beam
  geometry from route.json vs the CAD boxes (models/acp_depth/align_apparatus.py, the sim park in deploy/hiking/sim).
- **Done:** renders lower the CAD dressing onto the physics deck (align_apparatus.py); the sim park uses the CAD as the
  apparatus physics (deploy/hiking/sim/make_scene.py). The eval itself is unchanged.
- **Open (user):** which one is the apparatus as built? If the CAD, every track's apparatus score is measured on stale walls
  and beams: regenerate the raster, walls and route from the CAD, re-score.
- **Resolved 2026-11-17 (user asked how one apparatus gave two models):** the eval is the current CAD; terrain-park's
  `apparatus/ours/sim/apparatus.xml` is a stale export. Both come from the same parametric CAD (`cad/params.py` + `model.py`,
  identical in meetsitaram/loco-manip-apparatus and terrain-park). The eval's front 4x4, (2.985, 0.257)-(3.098, 1.041) m, is
  exactly the current params, (117.5, 10)-(122, 41) in (commit 4d0a92d, 2026-10-07 07:11). terrain-park's export has the 4x4s at
  centre (116.75, 26.0) in, 32.65 in long, yaw 78.5 deg: exactly the params of 7fba58a, (113.5, 10)-(120, 42) in, replaced 71
  minutes later. Neither generated file is in loco-manip-apparatus's git (both were built locally on the Windows PC); the
  stale export was copied into terrain-park on 2026-11-15. model.py changed until 2026-10-28, which likely explains the other
  wall mismatches. The 0.108 m is no discrepancy: the export's z = 0 is the venue floor, the eval's the deck top (deck 4.25 in).
  Consequences: the apparatus scores stand; the sim park (deploy/hiking/sim, A2 routes) and the render dressing used the
  stale model. To do: regenerate terrain-park's export from the current CAD (needs trimesh; terrain-park is public, push on
  the user's go), rebuild the park scene, rerun A2. Final check on the built apparatus: front 4x4 centre 119.75 in from the
  deck's start end (stale export: 116.75 in).
- **Acted 2026-11-17 (user: "It's ok to have either map, just don't have both overlapped together"):** the sim park had both
  stacked (eval raster with its 4x4s under the stale CAD, whose deck sat 0.108 m higher and whose ramp / stair skins are
  visual-only); the renders drew the CAD dressing beside the eval's own 4x4s. Now one map: make_scene.py APPARATUS_MODEL=eval
  (default; CAD only with APPARATUS_MODEL=cad, replacing the eval walls), park rebuilt (old one kept as
  scenes/park_cad_stale); render_showcase.sh / render_vis.py draw the eval's physics unless APPARATUS_DRESS names the eval's
  own source scene (loco-manip-apparatus sim/scene_x2_env_mobility.xml, on the Windows PC). A2 routes rerun on the one-map park.

### B-acp1. ACP eval adapter: on a hold at a waypoint without a station heading the robot spins in place (2026-11-16) - acted (acp_depth only), open (shared adapter)
- **Found:** `eval/adapters/acp.py` faces the bearing of the waypoint while holding; once the robot stands on it the bearing
  swings with every centimetre of sway and ACP turns after it: 200-490 deg in the last 4 s of the ladder clips.
- **Done:** `eval/adapters/acp_depth.py` keeps the arrival heading within 0.15 m (ACPD_HOLD_HEADING=1, default): 0-1 deg.
  Success is decided before the spin, so scores barely change.
- **Open:** the same 6-line fix in the shared adapter (every ACP run) waits for the user.

## C. Training, all tracks

### C1. Slow commands: standing pays (2026-11-11/12) — acted (SSR), partly open
- The speed reward exp(-error^2 / 0.25) pays a stander 70% at a 0.3 m/s command. SSR stood at commands under 0.5 m/s; Hiking
  never trains under 0.45 m/s. Detail and options: `agentic/state/SHARED_LESSONS.md`, "Slow commands".
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
  restart. Selection is now by leg product among gate-passing pairs (champion 7,000). Detail: `agentic/state/acp.md` 17a.
- **Leg 5 falls (7,000):** all four on the first stair steps while the yaw command is at its limit; the robot turns onto
  the flight beside the first riser. Traces asked of the eval. Open.
- **7,000 on 200 trials: 125/200 = 62.5%** (5,750: 57.0%; ranges overlap): stairs 9 falls against 45, 11 turn-around stalls,
  51 of 75 failures in the 4x4 section. Provisional because of B8.
- **Where 7,000 fails in the 4x4 section (16 traced, provisional):** 8 drift east into the ramp cells around the turn-around
  at button 1; 3 fall crossing the front 4x4 on the way back although the step is planned (one 143 deg off square); 4 at the
  door side after the crossing; 1 on a wall before button 1. Mostly the tracker's motion around the turn-around.
- **7,000 on 50 trials: 26/50** against 22 (5,750) and 18 (6,250) on the same seeds; 1 turn-around stall (8 at 6,250), 3 stair
  falls (19 at 5,750). Freezing the planner worked. The largest loss is now falls at the 4x4s (13 of 24 failures).
- **Open:** the turn-around and the stair approach need the terrain stage (stage 3). Detail: `agentic/state/acp.md` 16q.

### D5. Extended map for stage 3: import built, one unexplained defect; contacts dropped on the ramp raster (2026-11-12) — open
- The user's tiles are imported as one height raster with 197 routes for ACP's goal environments. In the first CPU closed loop
  every extended type falls at levels 0 and 9 within a second and none at level 4; it does not reproduce with 4 robots, so it
  depends on how many robots run or their order. Being diagnosed; CPU checks due 2026-11-13 12:00 UTC.
- The running stage-2 log has 5,228 'height field collision overflow' lines on the existing ramp raster: contacts beyond 50 per
  pair are dropped in training today. Effect on the ramp clips unknown.

### D-depth1. ACP on a depth camera: a fine-map student under the unchanged ACP matches perfect height scans on our apparatus with D435-like noise in the loop (2026-11-17) - acted, real-camera check next
- **Found:** a student that predicts ACP's own 10 cm scans (planner 17 x 11, tracker 9 x 7) from chest depth + proprioception
  reached the perfect-map score on clean depth (d1, one DAgger round: 85/100 vs 85/100 paired on fresh seeds) but flattened the
  4x4s to 4-6 cm of 8.9 (an 8.9 cm beam on 10 cm sample points flickers in the labels; the net hedged) and fell to 35/50 with
  stereo-like noise. Isolation tests: the tracker's scan, not the planner's, is where map errors cost success (student to the
  planner only 45/50, to the tracker only 21/50); the map behind the robot is not needed (44/50 with it blanked); a +-3 cm offset
  does not hurt the tracker (30/30), shape errors at edges do.
- **Fix:** the fine-map student (`models/acp_depth/fine_student.py`): depth projected through the camera pose (IMU attitude + leg
  kinematics) into a 2.5 cm bird's-eye grid, 0.5 s of grid memory, a small U-Net -> fine map, ACP's points read off it; trained on
  D435-like noisy depth, two DAgger rounds with noise in the loop.
- **Evidence (our apparatus, seeds 2000-2049, perfect scans 46/50):** fine_v3 46/50 with noise in the loop (clean 41/50); 4x4
  estimated 9.2 cm (tracker) / 7.7 cm (planner); showcase clips all succeed (models/acp_depth/render_showcase.sh).
- **Open:** the real D435 (captures from PC2), Orin CPU timing, clean-depth frames in training, the robot-side node.
  Detail: docs/TODO_depth_to_heightmap.md, models/acp_depth/README.md.

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
- Seven defects found by an independent audit (`agentic/state/hiking_port_audit.md`); run 4 has all the fixes.

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
  about 5 deg gives toe tips to 2.0 m. **Needs the user** (the real mount): `agentic/docs/USER_TODO.md` 5.

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
  running alone for a minute. Nothing lost. Rules now in `agentic/state/ssr.md`: `bash -n && tmux new-session`; never kill a tmux
  session that holds a supervisor; stop runs only through STOP. Applies to every track.
- Track agents decide in their own track and report what went wrong and how they fixed it (`agentic/docs/USER_RULES.md`).
- Agents' hand-back reports arrive once per turn; later ones are lost, so decisions are also written to `agentic/state/<track>.md`.
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
  in `agentic/state/ssr_port_audit.md`, separating "paper does not specify" from "paper specifies and we did not follow".
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

### E3a. First paper-aligned SSR run slowed to a shuffle (2026-11-13) — v14 resumed
- v15 (Table 11 prior from Pico + the paper's reference, separate advantage normalisation, paper kernel and commands) ran 150
  iterations from v14's 7,000: speed made good fell 0.80 -> 0.45 of the command, double support 0.21 -> 0.50; its gate fell
  back to v14. Not yet separated: the new prior (5x smaller, no flat group, slow Pico takes dominate several groups) vs fresh
  discriminators. CPU checks before any v16.

### C11. The paper's G1 reference was read in the wrong form (2026-11-13) — re-retarget in progress
- The user doubted a 21-degree backward waist bend on stairs. Check: the Hiking paper's reference is stored in InstinctLab's
  "torsoBase" form (base pose = torso, waist joints negated); our tools read it as a standard pelvis-rooted G1, applying the
  waist bend twice. Effects: pelvis 4.4 cm too high (wrongly "fixed" as a float in v2), and the torso shown leaning back on
  stairs and blocks when the true G1 leans forward (stairs up +10.6 deg, block step-up +24 deg).
- Affected: X2 retargets v1-v3; Hiking trained on v2 since 02:58; SSR's amp_v5 tests. Fix: one shared reader for the source
  form, re-retarget (v4), review renders, then Hiking and SSR switch to v4.
- Lesson: read a dataset's conventions from the code that wrote it before using it; a "float" that needs a root offset is a
  symptom to explain, not to patch.

### G1. Training-curve findings from the new plots (2026-11-13 13:41 UTC; `eval/metrics/out/training_plots.html`, `python eval/metrics/plots.py`)
- ACP (WITHDRAWN 2026-11-13 14:30: plotting artefact - the plot averaged sets with 0 episodes as 0; episode-weighted, Pico v1_1 0.98 / mirror 0.98 vs 0.995 before; diverged 0.028 = simcheck faults now counted, under the 0.05 limit; plots.py fixed to weight by episodes) at 10,356 (s2_v4, data 2h) Pico-capture finish 0.97 -> 0.65 and planner-probe Pico finish -> 0.45, not recovered by 12,660; diverged-end share 0 -> 0.028 at the same point; GPU use 70-90% since about 7,000. Owner: ACP agent (s2_v5), in its report.
- SSR: flat since about 5,500 (speed made good 0.78-0.80, return about 32, level 12-13); stairs level falling 13 -> 8; extended-map levels reset to about 0 at every restart (7,010 / 7,960 / 8,610). Owner: SSR agent (v4 resume), in its report.
- Hiking: terrain levels stuck at about 6 since about 3,000 (gaps still climbing); speed made good about 0.45-0.5 of command since 7,500; about 97% of episodes end by time-out. Owner: Hiking agent (v4 resume), in its report.
- G1 Hiking, answered by the Hiking agent (2026-11-13 14:00): the curriculum is FULL, not stuck: promotion at velocity-tracking score > 0.6; past level 9 Isaac sends a robot to a random level 0-9, so a policy that passes everything averages 6.0 (per type 5.88-6.85, gaps 5.21 still climbing). Half speed (0.247 at 0.498): route robots make 0.74-0.90 of command; the rest turn toward targets with forward commands on, and nothing pays for closing the gap (tracking reward and promotion give 76% at half speed; overspeed penalty only above command). Flat MuJoCo at 4,125 tracked 0.5 m/s within 2%. v7 changes neither. Fix options (harder rows, promotion on speed, tighter tracking kernel) need a restart: open for the user.
- G2 Hiking on its own extended map (2026-11-13, eval agent; `eval/out/renders_20261113/hiking_extmap_11125/`): 11,125, 56 route types x 10 trials (levels 0-9): strict 380/560 (68%), 13 falls, only 86 clean (no stumble). Stumbles are toe strikes on riser, beam and wedge faces in swing (stairs avg 17.9 hits per trial, about 40% shift the pelvis). Stairs straight 4/10, side 4/10, oblique 3, far straight 1, C 5, S 4, L 7, cross 10; homes 0/10 (stuck on counters/walls with arm contact). 109 failures are an eval artefact: Hiking's adapter stops within 0.4 m (as in training), the eval needs 0.20 m (489/560 without it). Open: end-ring rule (user decision; coordinator recommends the adapter creep to 0.20 m so the eval stays strict); side-hit classes not defined for blocks, homes, street front.

### E4. SSR v16/v17 exports failed the hand-over check: two record/adapter defects (2026-11-13 15:45 UTC) - fixed
- Symptom: `r9_v16_p300` (06:31) and `r7_v17_011500` / `012000` (14:09, 15:31) failed the fetch check on depth only (max 0.206 /
  0.02; frame, action, target at 3e-6). SSR's train/leg scores were blind since v15.
- Cause 1 (the record): the export puts the robot on the flat level-0 tile. In the v14 tile layout (T.PAPER off) the next tile
  (flat level 1, y 4-8 m) has 1.3 m side-obstacle walls in the vision raster, inside the 3 m chest_rs range; the check replays
  on an open plane. v15's record (paper layout, open FREE column) had none.
- Cause 2 (the adapter): training's ray march (38 steps, 0.10 m cap, tmin 0.08) cannot reach a hit beyond 3.88 m and reports
  "no hit" (t = 7.5); the adapter's exact cast found the floor at 3.9-4.4 m (corner pixels, error up to 0.12). v15 only passed
  because its gait never showed floor that far. Affects every SSR score so far (a few far corner pixels read 0.93-0.97 instead of 1.0).
- Residual: rays whose hit lies in the march's last 0.1 m (3.79-3.89 m) end as hit or miss by float32 rounding: a float32
  replay of the record reproduces it 64,767/64,767, but a 2e-6 shift of the ray origin flips 84 of 450 such rays (less than
  the check's own 3e-6 pose error), so no adapter can match them pixel by pixel.
- Fix: (1) `code/scripts/export_eval.py` (node) records over a flattened vision raster (`env.hmap.vz` zeroed, physics unchanged)
  and (2) writes `ray_march_iters / ray_march_step_cap / ray_tmin / ray_hit_range` (3.88) in the card;
  `eval/adapters/ssr.py` treats hits beyond `ray_hit_range` as no hit (older cards unchanged). (3) CHECK CHANGE, for the
  coordinator to accept or revert: `eval/tools/ssr_check.py` lists band pixels as `depth_band` and excludes them from the depth
  max; all other pixels keep 0.02; FAIL if the band exceeds 2% of the image (seen: 1.3% max).
- Result: re-exported 11,500 and 12,000 (old folders in `tracks/ssr/_trash/`): PASS, depth max 3.2e-4 outside the band (band
  max 0.107 / 0.124, share 0.6% median). v15 p100 still PASS; the old v17 11,500 record still FAILs (the band rule does not hide
  the wall). queue35 uses the patched script from 12,500 on. Scorer re-queued 11,500 / 12,000 at 15:40 UTC.
- Open: Hiking `x2_parkour_v4_11250` - every train/flat job "failed" within 1 s at 14:19 (not investigated; eval monitor).
- B12 (2026-11-13 15:50, coordinator's own fault): the first run of `eval/autoscore/cleanup.py` (13:35) removed the planner and obs files of `eval/data/handover/acp_s1v2_4250`, the eval regression reference, so every training-terrain job failed in 1 s from 14:19 (Hiking 11,250 has no training-terrain rows). Restored from the node's `pair_s1_v2_004250`; cleanup now never touches regression and baseline folders; regression check 10/10 bit-identical; Hiking 11,250 re-queued. Note: one regression run before the single-thread rerun showed 3 of 4 baseline trials changed (BLAS threading under load is the suspect, not confirmed).
- E4 accepted by the coordinator (decision without the user): SSR's hand-over check leaves out the depth pixels in the ray march's last 0.1 m (float32 hit/miss flips), limited to 2% of the image; all other pixels keep 0.02; the old v17 record still fails, so the wall defect is not hidden.
- G2a (user, 2026-11-13): "hiking videos look good, except one where it just keeps making steps in place.. that setup is also pretty difficult for the x2 to move.. so may be it will figure out later". Homes 0/10: steps in place against a counter/wall, arm contact (19 in the render), commanded 0.48 m/s toward waypoint 2 of 4. Watch on later v7 exports; no action now (user). Also seen: the render credit line still says "Bones Studio" for Hiking (its reference is the paper's data); open item, owner eval monitor.
- G2b (2026-11-13 17:20): correction to 'homes 0 -> 7/10': all 7 successes are on homes routes 0 (6.4 m) and 2 (7.3 m). Route 1 (2.3 m, big block at the entrance then several small blocks; the user's step-in-place video) still stalls at all three levels tried (1, 5, 8) after the stop fix: 0/3. Coordinator first reported 7/10 without the per-route split.
- G2c (user, 2026-11-13): "i think that route is very difficult to navigate.. keep monitoring progress". Homes route 1: no action; watch it in the extended-map results of later exports.
- E5 (2026-11-13 18:30): SSR v17 12,000 on its training terrain (20 trials each): 20/20 flat, beams 5/10/15/19, ramps 10/15/19, stairs 5, platform 5, ext corridor/single log/stairs down/wedges; stairs 10 19/20; platform 10 15/20; 0/20 stairs 15 and 19, platform 15 and 19; ext stairs up 6/20; **ramp L5 2/20 (5 headline) while ramps 10-19 are 20/20 - suspect course or floor, not the policy; owner: eval monitor at the next agent start.** Renders: eval/out/renders_20261113/ssr_v17_12000/.
- E6 (user, 2026-11-13 18:40, on SSR v17 12,000 renders): "able to navigate the obstacles ... but the arms are oddly straight, and i still see some waist tilt.. may be it hasn't learned the human walk style much yet." Context: 12,000 is 650 iterations after the v4 switch (11,350); discriminators warm from v14; arms weighted 0.3 in the style input (user rule); style about 4% of the policy gradient (SSR agent). Plan: re-render the same three courses at 15,000 and 20,000 and compare arm swing and waist pitch/roll numerically; no restart (user rule).
- E7 (2026-11-13 18:50): **SSR style reward collapsed at the v4 switch and has not recovered:** r_s 0.33 / D(policy) -0.63 before 11,350; 0.10-0.11 / -0.88 flat from 12,350 to 13,790. Hiking holds 0.62 (discriminator carried over). Fresh SSR agent diagnosing on CPU (no restart).
- **E7 diagnosis (SSR agent, 2026-11-13 19:10 UTC, CPU only, run untouched): cause = arm action noise, unmasked at the switch.**
  - Log: the discriminators' input-gradient share on arms went 0.00 (11,000-11,340, `--disc_mask_arms`) -> 0.13 at 11,360 -> 0.37-0.60 from 12,000, in all five groups; style advantage spread 0.0073 -> 0.0037 (4.3% -> 2.1% of the locomotion spread, joint normalisation).
  - The arms' action std had drifted high while the discriminator was blind to them (entropy 0.005, no cost): 13,500 std (x action scale) shoulder pitch 0.68 (0.17 rad), roll 0.97 (0.25 rad), yaw 0.48 (0.23 rad), elbow 0.75 (0.36 rad; it rose from 0.52 after the switch); ankle roll 1.87. Sampled each 20 ms step, this is white noise: arm joint-velocity spread 2.48 vs 0.36 in the reference.
  - CPU probe (`tracks/ssr` node `runs/style_diag/`, 64 robots, flat, 0.6 m/s, v17 13,500 policy and discriminator): sampled actions (as in training) D -0.95, style reward 0.06, 47% of windows below D -1 (reward 0, no gradient) = the training numbers. Mean actions: D +0.05, reward 0.75. Noise on legs only: -0.44 / 0.48; arm std set to 0.3: -0.71 / 0.26 (the pre-switch level). Swapping in reference arms on the sampled windows: -0.95 -> +0.33.
  - Posture (the user's "arms oddly straight"): the mean-action gait holds elbows about 0.59 rad straighter than the default (near 0 rad absolute; reference -0.69 rad) and barely swings (shoulder yaw / elbow spread 0.1-0.2x the reference). Reference arms into the mean-action windows lift D 0.05 -> 0.56. The waist has 1 joint (yaw) in the style input; waist pitch / roll are not among the 21 policy joints (held by PD), so a visible tilt comes from the pelvis: gravity x in the pelvis frame 0.005 vs 0.052 in the reference (about 3 deg pitch difference). This is what the style reward should correct once it has a gradient.
  - amp_v5 has no defect found: q-velocity equals the 50 Hz central difference of q (slope 1.000 in all 352 clips, no clip off by 10%), mirror round-trip exact, feature order and default pose identical to the policy's (q - plant default). One oddity: the reference shoulder roll is almost constant (sd 0.013 rad, about 0.06 rad inward of vertical); a sharp feature but not the cause.
  - Not the input normaliser: its arm spread is 2x the reference's (history of the whole run), no blow-up.
  - Paper: all 63 features always visible to the discriminators, so arm noise is penalised from the start; Table 9 min std 0.2, entropy 0.005, init std silent. Our 11,000 iterations with arms masked are what let the arm std drift. Hiking's single discriminator saw arms from its v5b start.
  - Fix (smallest): at the next planned restart, reset the 8 arm log-stds to log(0.3) and cap them at 0.4 (new flag in train.py; ankle roll optionally capped at 0.6: -0.67 / 0.30 in the probe). No restart now: locomotion is not affected (share of command 0.79, slow commands 0.97), user rule no stops; the style signal stays near zero until 20,000.
- **E7b (SSR agent, 2026-11-13 23:55 UTC, CPU only, run untouched): v18 style plateau (r_s 0.15-0.16, not 0.26) = elbows pinned at the joint stop, plus std at the cap and a re-sharpened D.**
  - Stds: all 8 arm stds at the 0.4 cap from ~15,000 (the 0.3 reset lasted ~150 iterations; the entropy bonus pushes up). Ankle roll 1.90.
  - Groups (log, 16,600): arms still 0.25-0.48 of the discriminators' input gradient (stairs lowest), legs 0.26-0.47, pelvis 0.14-0.27. d_data 0.88 -> 0.75 at the reset, back to 0.83 by 16,000.
  - Probe (16,500 policy + D, flat, `runs/style_diag/out4.json`): sampled r 0.07 (the 13,500 D on the same windows: 0.22, so D re-sharpened); arm noise off 0.37; arm std 0.2 0.22; ankle roll 0.6 0.09 (no gain); leg noise off 0.35; mean actions 0.72. Gradient share on sampled windows: arm q 0.82, arm qd 0.04, legs 0.10.
  - Cause: elbow mean action 6.8 (PD target 3.3 rad past the 0.0 rad stop; 1.31 at 11,350), elbow pinned at the stop in every sample (q - default 0.58, reference -0.09); shoulder roll 0.5 rad the other way from the reference. The policy removed elbow noise by pressing into the stop; no sample leaves it, so PPO gets no reward difference and the D's arm-pose gradient cannot act. This is the user's "arms oddly straight".
  - Next (decided, next planned run after 20,000 ~06:30 UTC 14 Nov, no restart): action-bound loss on arm action means (PD targets kept within joint limits) and arm std reset/cap 0.2 (paper min std 0.2). Code behind default-off flags + CPU pre-flight before the run end. Owner: SSR agent.
- **E8 (SSR agent, 2026-11-14 16:30 UTC, CPU only, run untouched until GO): stair refusal at the apparatus leg 5 = refusal, not attempt-and-fall; cause = our curriculum keeps stairs at level ~6 (11 cm), so 15-20 cm risers are never practised, plus a lane-wall termination (ours) that kills attempts.**
  - Probe (`runs/stairs_diag/stairs_diag3.py`, r7 25,000, own training stairs tiles, 24 robots per level, 20 s, sampled actions; json `runs/stairs_diag/{d25000,p25000,v20probe,plat25000}.json`). 0.6 m/s: level 19 (20 cm, up-first) 24/24 stall 0.17 m before the first riser, 0 falls; level 16 (17.6 cm) stall 50%, passed a riser 12%, the rest ended by side-obstacle touch (42%); level 17 (18.4 cm) stall 17%, side-obstacle 75%; level 12 (14.5 cm) side-obstacle 42%, passed 54%. With side-obstacle ending off: level 12 passed 96-100%, 16/17/19 stall 79-96% (the refusal remains). Platform (33/40 cm): attempt-and-fall at 33 cm (tilt 42%), stall + side-obstacle at 40 cm.
  - Reward per alive step, robots at the stair foot vs on the flight (levels 12-19): ours r_l 1.33 vs 1.27, foothold 0.99 vs 0.91; the paper's Table 6 terms with sigma_v 0.25: 1.52 vs 1.43. Standing at the foot pays as much as climbing under both; posture terms (orientation, yaw/roll rate, base height) dominate and climbing loses them; sigma_v 0.1 vs 0.25 changes both sides alike (0.09 vs 0.10 ours, 0.29 vs 0.32 paper). No added reward term pays the stander (swing_h 0.02 at the foot vs 0.05 climbing; creep and drift 0 at 0.6 m/s).
  - Curriculum (DEV-P, the decisive deviation): ours promotes only at the tile end (8 m in 20 s), demotes when travel < 0.5 v t. Training-command probe: level 8 up 29% / down 54% (ours) vs up 92% / down 4% (Rudin [30], the paper's A.1 rule); level 12 13% / 38% vs 58% / 38%; level 14 4% / 50% vs 100% / 0%. Hence the logged stairs level 6.25 at 25,120 (ramp 14.7, beam 13.2). Refusals at timeout are demoted by both rules (0-8% kept), so the curriculum does not reward refusing; it just never brings the robot to the hard risers.
  - Fix v20 (one resume of r7, user-approved): `--cur_rudin` (paper: up when walked > 4 m, down when walked < 0.5 x mean |cmd| x 20 s), `--climb_types stairs,platform --climb_side_noterm` (lane-wall touch is not terminal on stairs/platform tiles, graze cost -2 stays; the paper has no side obstacles; route robots with physical walls unchanged), `--climb_creep` (ours: progress pay 0.5 x share at any speed on climb tiles; the probe shows it is weak, 0.05 per step on both sides; kept as the only pay for actual progress). Not changed: sigma_v (no stairs effect), swing_h (pays climbing more). Open: if v20 still stalls at 17-20 cm after 1,000 iterations, the next lever is a stall cost on climb tiles. Owner: SSR agent; after-probes at +100/+250/+500 (`runs/stairs_diag/v20_p*.json`).
- G3 (2026-11-14 01:40): Hiking 14,750 on its own extended-map wedge tiles: strict 623/900 (69%), headline 746/900 (83%), 55 falls; 6.7 wedge-face foot/shin hits per trial; arm contacts 0.1, edge landings 13.8 per trial. Apparatus legs 3/4 (10,375 / 11,750): arm contacts 59-173 per trial, ALL on the ramp corridor's side walls at about 0.6 m; edge landings 135-141. Differences: the apparatus ramps sit in a walled corridor (wedge tiles have no walls), slopes 15 deg vs 8-12 deg, cells 1.22 m vs 1.2-2.0 m. Stall fault seen on tiles: side-target turn at the last station (heading frozen while commanded 1 rad/s). Render credit fixed per track in eval/render.py (decision without the user).
- G3a (2026-11-14 01:50, coordinator check of extended_v1/wedges.json after the user asked): the 40 wedge tiles have NO solid walls; 21 have rail fences (posts and rails) beside them, some brace stands and blocks. Slopes rise with level from 8-12 deg (level 0) to 15-25 deg (level 9), so the eval agent's '8-12 deg against the apparatus 15' was wrong (it is level 0 only); 15 deg is inside the training range from level 2 up. Remaining difference: solid side walls of the apparatus ramp corridor (all apparatus arm contacts are on them, about 0.6 m high).

### H7. Hiking v7: "falling reward / missing termination shares" = three NaN crashes plus restart warm-up, not a logging gap (2026-11-14 02:35 UTC) - guard installed 02:39, trigger open
- Symptom (coordinator, 300-iteration tensorboard means): reward 100.6 -> 72.6 and episode 968 -> 748 from 13,500 to 15,000;
  termination shares summed to only 0.81-0.83 at 14,700 / 15,000.
- Cause 1 (crashes): the trainer died three times with `RuntimeError: normal expects all elements of std >= 0.0`
  (ppo.compute_losses -> actor_critic.act), supervisor log "run ended: exit 1" at 11-13 22:44 (ckpt 14,125, last iteration 14,246),
  11-14 01:08 (ckpt 14,875, last 14,921), 01:45 (ckpt 15,000, last 15,050). Each time the CRITIC diverges first: value_loss
  0.3-0.56 -> 9.3e9 -> 2.4e21 -> 6.6e32 -> NaN in four iterations (14,243-14,246; 14,919-14,921; 15,047-15,050), surrogate,
  KL-driven actor and discriminator normal until the NaN reaches the shared update. No such crash anywhere before v7 (11,125-14,242 clean,
  and none in v5b). Intervals shrink: 3,121 / 796 / 176 iterations. Settings: no empirical normalisation, clipped value loss,
  max_grad_norm 1.0, adaptive lr; the trigger (an outlier critic input or reward) is not identified on CPU.
- Cause 2 (bookkeeping): each supervisor resume re-runs the iterations after the checkpoint (metrics.jsonl and tensorboard hold
  220 duplicated iterations 14,125-15,050) and restarts all 8,192 robots at t = 0. For about 40 iterations no robot reaches
  time-out, so the logged termination shares sum to 0.0-0.1 and reward / length are those of early falls (iteration 14,125:
  reward 0.7, length 18); mixing takes a few hundred iterations. Those rows are the "17-19% unaccounted ends". Outside the
  warm-ups every row sums to 1.000 (no unlogged termination term, no simfault / level_keeper / command resets).
- Real policy trend with warm-ups and discarded iterations removed (200-iteration means): reward 100.7-102.0 / length 963-975 at
  12,800-13,999; 92-98 / 909-944 at 14,200-14,799 (each a post-restart mix); last 9 iterations 96.6 / 954. Terrain level flat
  6.11-6.19, speed ratio 0.50-0.52, time-out 0.95-0.96, action std 0.90 -> 0.93, bad_orientation 0.012 -> 0.015-0.019. So a mild
  real drift at most (about -3%), dominated by the restart transients.
- Scorer: 11,750 legs 46/45/6/5 (non-strict), flat turn 28/50; 14,625 (partial, 02:00) legs 42/34/1/0, flat turn 25/50, small_boxes L5
  14 -> 5/20, stairs_down L8 3 -> 20/20. Not better than 11,750.
- Hazard: guard7 (crashguard_v4.sh, FB_START=fallback_v5b.sh) falls back to v5b (pre-v4 motion) on two error exits from ONE
  checkpoint; with crashes 176 iterations apart and checkpoints every 125, a NaN crash loop would silently revert the v7 recipe.
- Proposed (not acted on, crash-level, awaiting coordinator): (1) guard7 must not treat a NaN-divergence exit as a code-error
  loop (or stop instead of falling back); (2) a finite guard in instinct_rl ppo.update (skip the minibatch / restore the last good
  critic when value_loss is non-finite or > 1e3, clamp returns), picked up at the next crash resume with no planned stop;
  (3) dump the rollout's max |reward term| and max |critic obs| when value_loss > 1e3 to find the trigger. Detail: agentic/state/hiking.md.
- Done 02:39 UTC (coordinator decision): (a) finite guard in instinct_rl `algorithms/ppo.py` (one code folder; original in
  `_trash/2026-11-14/`): a minibatch with non-finite loss or value_loss > 1e3 is skipped and logged as `[GUARD] ...` in train.log
  with max |returns| / |values| / |critic obs| / |obs| and the top five critic-obs columns; returns clamped to +-1e4 (logged when
  hit); `Train/guard_skipped_minibatches_max` per iteration; three updates in a row with every minibatch skipped raise (-> normal
  resume). Pre-flight on CPU (`code/x2_port/v7/preflight_finite_guard.py`, log `_supervisor/preflight_finite_guard.log`): PASS -
  normal minibatches bit-identical parameters and losses over 3 updates; forced NaN and forced 1e7 critic-obs minibatches skipped
  (= original without them; original went NaN on the NaN case); in-range returns unchanged. Loaded at the next resume, trainer
  not stopped. (b) guard7 now runs `code/x2_port/crashguard_v7.sh`: never falls back; ALERT line per crash exit in gates.txt and
  ALERTS.txt; restarts v7 with `start_x2_parkour_v7.sh` (same flags as the running supervisor, checked) if the supervisor goes
  without STOP; writes STOP only after 5 crash exits within 2 h. Old crashguard_v4 instance (v5b fallback) killed.

- B13 (2026-11-14 04:35): scorer training-course job timeout 570 s per attempt was shorter than one SSR 20-trial course (depth camera, about 26 min), so SSR r7_v18_016000 train_1 looped 35 times for 8 h; raised to 2,700 s. Coordinator miss: the job sat in the 'running' list at several refreshes without its age being checked.
- G3b (2026-11-14 09:00, correction to G3/G3a): the apparatus ramp corridor is NOT solid walls - it is post-and-rail fences 2.44 m apart, rails at 0.565 and 0.87 m, 1.22 m ramp cells at 14 deg (Hiking agent, from the apparatus files). The extended-map wedge tiles also carry rail fences, so 'solid walls missing from training' was wrong; the arm contacts are on the rails. Parked (user priority: paper gap first).
- H8 (2026-11-14 09:00): Hiking plateau diagnosis: refusals at stairs-up / platform-up / gap rows 5 and 8 are stalls with no fall; our extra commands (stand 0.10, turn 0.15, slow forward for 25%) make the paper's promotion rule promote a refusing robot (simulated: 42% promoted vs 7% with the paper's commands); our touchdown reward (weight 140, not in the paper) pays about 26 per episode for marking time. v8 armed (GO_v8 not created): HIKING_X2_FAIR=1 (episodes with stand/turn/slow commands neither promote nor demote) + touchdown pay times forward progress. Pre-flight PASS.

### A4. ACP stage 3: falling finish_frac is composition plus spawn faults on three extended-map families, not tracker degradation (2026-11-14 15:30 UTC) - open, fix queued for the next restart
- Evidence (runs/s3_v2 metrics, 100-row windows; CPU diag in runs/s3_v2_diag/: diag_1430.txt, fault_diag_lv1.json). Heartbeat finish_frac = all episodes; since 19,218 half the envs are goal envs, so it mixes two measures.
  Reference-clip envs (episode-weighted): s2_v5 last 300 rows 0.957 -> s3 0.925 (first 100 rows, a step at the switch: paper phase-0 starts = whole clips) -> 0.938 now, rising. Largest step: flat 0.965 -> 0.922, blocks 0.964 -> 0.917, stairs turn 0.945 -> 0.910, stairs x10 0.934 -> 0.918, ramp sh2 0.966 -> 0.929; all flat or rising since.
  Goal envs: goal/success 0.26 -> 0.47 -> 0.25 while the curriculum climbs (level mean 0 -> 8.1 of 9); falls 0.04. But 63% of goal episodes end in a simulator fault (end/term_diverged 0.15 -> 0.32 of all episodes; heartbeat sim_fault_share 31.8%). Success among clean goal episodes ~0.43 at level 0 -> ~0.67 at level 8: learning.
- Cause of the faults (CPU replay of latest.pt, 64 goal + 64 reference envs, 480 steps, random levels): 281 goal faults vs 18 reference; 219 of 281 (78%) within 10 steps of reset; 228 in ext_homes 79 / ext_log_corridors 82 / ext_tight_spots 67 (ACP's own six families: 6); causes depth > 5 cm 240 (77 deeper than 1 m, max 2.24 m), qvel 102, speed jump 70; no NaN. = goal spawns inside geometry on those tiles. Reference faults: 18 shallow depth (5-10 cm), as in stage 2.
- Impact: small in steps (a spawn fault costs a few env steps, ~0.2% of the batch); no stop risk (watch_s3 watches crash/stale only; curriculum EMA excludes faulted episodes). Real cost: those tiles never train and finish_frac / sim_fault_share read as alarms.
- Not crash-level, nothing changed in the run (user rule: no stops). Detail: agentic/state/acp.md 17l.
- Fix prepared and ARMED (2026-11-14 16:10 UTC, agentic/state/acp.md 17m): spawn check of all 1,604 tiles: 80 fault (ext_homes 17, beam fields 15, tight spots 9, stairs 8, single beams 7, blocks 6, wedges 6, street 6, log corridors 5, stairs_down 1); goal_tiles_v2c: 62 spawns moved on the same tile, 20 tiles take a clean sibling, none dropped (cfg/stage3v2i_sets.json). Frozen-policy A/B, all tiles x 4 resets x 30 steps: 203 fault events (114 deep) -> 49 (8). Switch armed on runs/s3_v2/GO_spawnfix (resume from latest.pt, only DATA changes); GO is the user's call. Open: log corridors unchanged (7 events); outcome in training after a GO.
- E8a (2026-11-14 17:00, coordinator miss): the SSR audit of 2026-11-13 (docs/findings/ssr_paper_vs_code.md line 157, priority item 9) DID list the curriculum as DEV-P with the paper's rule and the fix (promote at TILE_X/2), but described the impact only as 'slow commands never promote' and nobody applied it, although the user ruled on 2026-11-13 that SSR goes back to the paper on every paper-specified item. No owner or date was set; the coordinator did not track the DEV-P list into v17-v19. Action: check every remaining DEV-P row against the running v20 flags and give each an owner and date.
- **E8b (SSR agent, 2026-11-14 17:00 UTC): all 26 DEV-P rows checked against the live v20 command line (`/proc` of the trainer) and code: 8 applied before (v17 prior Table 11, v20 curriculum, phases over, waist-scale code fix), 5 user overrides, 7 fixed now in ONE resume (v21), 6 deferred (D1).** Table: `docs/findings/ssr_paper_vs_code_status_20261114.md` (also every ADDED row kept / dropped with its reason).
  - v21 = v20 + separate advantage normalisation and sigma_v 0.25 (blended over 300 its from the resume), no vy clamp, base-frame foothold targets, omni commands only after the top level, Table 5 omni draw (`--omni_slow 0`; turn-in-place share kept: user rule), beam (the gap stand-in) heading +-15 deg (new `--beam_hmax`). Flags `runs/r7_flags_v21.txt`; queue39 (tmux `ssr39`) waits for v20's +250 stairs probe, then stops v20 at its next save; quality gate +150/+350/+500 (vx_ratio >= 0.75 x v20, ep_len >= 0.7 x) -> v21b (only norm / sigma / beam heading) -> v20; crash loop -> v20 flags.
  - D1 (owner SSR agent, ready by 2026-11-15 06:00 UTC, for the from-scratch run after 30,000): paper terrain set (`--paper_terrain`: discrete in the flat slot, no flat D, discrete prior set). Not warm because v15, v16 and T1 (all paper layout) slid to a shuffle within 30-150 iterations while every v14-layout run held; CPU cannot reproduce that slide (11-13 check c).
  - Evidence so far for v20 (curriculum): stairs level 6.25 -> 8.98 in 270 its; stairs probe (fixed 0.6 m/s, 24 robots) level 19 passed-riser 0 (25,000) -> 0.21 (+100) -> 0.375 (+250), stall 1.0 -> 0.75 -> 0.625; level 16 0.125 -> 0.58 -> 0.54.
  - v21 pre-flight PASS 17:19 (paper end state at once on CPU: KL 0.02-0.04, walking vx 0.58 alive 1.0, stairs 24 robots no regression); v21 started 17:20:56 from 26,000; gate +150 PASS (vx_ratio 0.752, ep_len 14.07). What went wrong: the pre-flight's 8-robot 8 s stairs probe falsely showed level 12 all stalling (horizon too short for the down-first tile); re-run at 20 s showed 100 % passed.

### H7b. Hiking v9 collapse at 22,113-22,118: one runaway robot poisons the whole update below the finite guard's threshold (2026-11-15 00:55 UTC) - rolled back to 22,000, rollout guard and collapse watchdog armed
- Symptom (coordinator, tensorboard): reward 77.8 -> -1.4, episode 970 -> 40-46, terrain level 5.88 -> 0.0 by 22,135, value loss
  0.2 -> 85-150, no crash, no alert; guard_skipped_minibatches_max 0 in the 25-iteration samples.
- Evidence (metrics.jsonl per iteration, train.log [GUARD] lines): 22,113 [GUARD] returns clamped (max |reward| 5.4e11, max |value|
  4.2e5), update applied (value loss 0.53); 22,114 every minibatch skipped (value loss 2.6e4-3.9e6; critic obs 1.1e8 in the
  action-history columns 713 / 742 / 771 / 800, 29 apart = one robot's last actions growing ~1.6x per step); episode sum of
  action_rate_l2 -9.1e9; 22,115 update applied (value loss 85, surrogate 0.141 vs ~0.005, action std 0.92 -> 0.93); from 22,116
  time-out share 0.94 -> 0.60 -> 0.09, root_height / bad_orientation ends 0.03 -> 0.75 / 0.25, level falling ~0.45 per iteration.
  The same runaway raised the guard at 20,560 (critic obs 1.7e18, three full skips -> resume) and is the likely v7 NaN trigger (H7).
- Cause: actions are unclipped and fed back as observations; one robot's actions run away. The storage normalises advantages over
  the whole batch (rollout_storage.py: (A - mean) / std), so that robot sets the std and every other robot's advantage is ~0: the
  updates at 22,113 and 22,115 trained only on the runaway while their value loss stayed below the 1e3 minibatch threshold.
  Checkpoint 22,125 was saved after the damage (episode 40). Not the level bookkeeping, FAIR, the lr or a terrain reset.
- Done (hiking agent, 00:33-00:45 UTC): STOP; model_22125 / 22250 and the collapsed levels file to
  runs/_collapse_backups/x2_parkour_v4_v9_22114/ (nothing deleted); levels from terrain_levels_prestop_v9.pt (mean 5.84 vs 5.83 in
  the log at 22,000); v9 resumed from model_22000 at 00:36:30 with the same flags (iteration 22,001 at 00:43, level 5.84).
- Guard (safety only, CPU pre-flight PASS, `code/x2_port/v9/guard2/preflight_guard2.log`): instinct_rl ppo.py / wasabi.py skip the
  whole update (PPO and discriminator) when any rollout obs or critic obs > 1e4, action > 1e3, reward > 1e4 or non-finite, log the
  robot, step and columns, and raise after 3 in a row (-> supervisor resume); rollout maxima logged every iteration as
  Train/guard_rollout_*_absmax_max. Normal updates bit-identical (params, losses, stats, discriminator calls). crashguard_v9b adds a
  levels copy per checkpoint and a collapse watchdog (episode < 0.4x and reward < 0.5x the median of the rows -300..-50, >= 350 rows
  after a restart) with automatic rollback to the newest checkpoint <= onset - 25 (once per 3 h, then STOP + ALERT); replayed over all
  23,271 metrics rows it fires only here (first at 22,123, onset 22,116). Loaded by one planned restart at model_22125
  (`v9/guard2/install_guard2.sh`, tmux inst_g2).
- Open: the root cause (unbounded actions) is not fixed; an action clip in the env (e.g. +-100) would change dynamics only on
  outliers - proposal pending the logged normal maxima. Check the guard_rollout maxima after an hour to confirm the bounds never
  touch normal rollouts. Detail: agentic/state/hiking.md 2026-11-15 00:55.

### H7c. Hiking v9: the policy holds waist_pitch at |action| ~18-28 (joint range +-0.31 rad, std 2.92); a +-20 action clip (user, 2026-11-15) would bind on 12-56% of normal steps on that joint (2026-11-15 13:10 UTC, hiking agent) - clip built and pre-flighted, restart held for the coordinator
- Evidence: tensorboard Train/guard_rollout_action_absmax_max, 3,990 iterations since 22,125: median 50, p99 331, 12 iterations > 1e3 (25,169: 6.8e5; 25,691-25,692; 25,858-25,860 full runaway -> guard raise -> resume from 25,750); the guard's top column is always 5 (waist_pitch), then 24 (right_wrist_yaw). Checkpoint 26,000 action std per joint: waist_pitch 2.92, wrist yaw 2.06, wrist pitch 2.56, shoulder roll 1.32, all others 0.06-0.76.
- MuJoCo probe (eval harness, model_26000 export, 3 x 60 s each, `code/x2_port/v9/aclip/probe/`): waist_pitch signed mean -17 to -19, median -17.5 to -20.5, max |a| 20.9-38.4; share > 20: flat 11.9% (mean action) / 28.6% (sampled), apparatus 56.0% / 30.7%. All other 28 joints: max 3.6-18.1, p99.9 <= 10.3, 0% > 20.
- Physics: waist_pitch saturates its 24 N m at |a| > 5.8 for any q in range and |qd| at the limit (kp 40, kd 5, scale 0.439), so a clip at 20 does not change the applied torque; it changes only that joint's last-action observation (from ~-27 to -20 on ~30% of steps) - an input shift at restart. The policy pinning a +-0.31 rad joint at a target 8-12 rad away is itself a defect (no gradient against a saturated action; std grew to 2.92) and is where the runaway starts.
- Built: `code/x2_port/v9/aclip/new/vecenv_wrapper.py` (clamp in InstinctRlVecEnvWrapper.step before env.step: applied target and fed-back obs both clipped, PPO keeps the sampled action; off unless HIKING_X2_ACTION_CLIP; optional per-joint HIKING_X2_ACTION_CLIP_JOINT; [ACLIP] log of clipped share per joint every 25 iterations). CPU pre-flight PASS (`aclip/preflight_aclip.log`). Installer `aclip/install_aclip.sh <clip> [joint bounds]` (one planned restart at the next checkpoint, also makes crashguard_v9b rollbacks keep the clip). Not installed.
- Open (coordinator / user): (a) uniform +-20 anyway (physically inert, waist_pitch obs shift ~30% of steps), or (b) +-20 on 28 joints and +-50 on waist_pitch (never binds in the probe; still bounds the runaway ~20x below the guard's 1e3). Hiking agent recommends (b). Eval adapter: the MuJoCo eval feeds the unclipped action back; if the clip binds (a), the eval adapter should clip the same way (eval monitor).

### H9. Hiking: the touchdown term t (ours) pays marking time at the foot of a row-8 obstacle more than climbing; stumble and graded contact do not decide the refusal (2026-11-14 17:25 UTC) - v9 armed, GO after the 19:00 data
- Evidence: CPU replay of every v8 reward term in MuJoCo (exports 20,000 and 19,250; 4-8 trials per course; table in agentic/state/hiking.md 17:25). Per second: stairs L8 foot-stepping 8.2 (touchdown +4.4), platform L8 7.3 (+3.7), against climbing stairs L5 5.2 and platform L5 6.1-6.6 (touchdown +0.9-1.8). Without t: refusal 3.5-3.8 /s, climb 4.3-4.9 /s. v8's progress weighting does not stop it (vx oscillates, positive at touchdown). Gap L8 edge stepping also pays 2.4 /s.
- feet_stumble and wall_contact_graded (user rules) cost 0.01-0.03 and 0.00-0.10 /s in real row-5 climbs (1-2% of the climb reward); the graded contact charges the row-8 foot-stepping (0.21 /s, knees on the platform face) more than any climb. No evidence that their weights make attempting cost more than refusing; proposal to the user: keep 1.0 (optional flag for 0.5 on the paper columns built, default off). Caveat: hard row-8 strikes are never seen because the policy never attempts.
- Gap rows 0-1: the paper's score is the episode SUM of exp(-err^2/0.25) / max length, so early terminations and edge-stepping (about 0.37) both fail the 0.6 bar once FAIR counts only walking episodes; v9 logs per-type score / up / down / early-termination to separate them.
- Done: v9 = v8 + release command mix (stand 0.05, no turn-in-place off the stand column, no slow forward / slow yaw, stand column vx 0 / vy 0 / wz +-1) + t removed (user decisions "Hiking v9 decisions"); CPU pre-flight PASS 17:20; tmux cont9 armed; the desktop script check1900.sh scores the 12 paper rows on exports 20,500 / 20,625 and then writes GO_v9. Fallback v9 -> v8.
- Open (owner Hiking agent): v9 trial verdict (launch.txt), by_type gap numbers +1 h, 12 rows at +500 / +1,000; 16,384 robots (approved; needs its own restart because the level keeper cannot restore a different robot count; next planned restart); stall penalty (user TODO).

### E8c. SSR v21 refuses the stairs because separate advantage normalisation (paper) blows up the style and foothold groups, both of which pay standing at the stair foot; sigma_v 0.25 adds a little (2026-11-14 23:45 UTC, SSR agent, CPU only, run untouched)
- User: "Why would it refuse? Check what rewards are overpowering the penalties ... changes from previous version." Probe `runs/refuse_diag/refuse_attempt.py` (node; jsons `ra_v20.json`, `ra_v21.json`, log `ra_*.log`): own training stairs tiles levels 15-19 (16.8-20 cm; 17-19 = apparatus-like 18.4-20 cm), 16 robots per level, 0.6 m/s heading 0, sampled actions, every term per alive second over 10 s from arrival at the foot, foothold reward WITH the imagined-swing part (foot net), style r_s from each checkpoint's discriminators, GAE per group with each checkpoint's critics. Behaviours: natural (v20 final 26,000: 16 refuse / 56 climb / 8 fall; v21 27,500: 80/80 refuse) and pushed attempts (base pushed to 0.45 m/s from the foot: v21 57 climb / 23 fall).
- Raw reward per alive second, refuse vs climb (v20 natural): r_l 69.0 vs 97.5 (sigma 0.1) / 78.8 vs 104.8 (sigma 0.25); r_f 44.3 vs 29.8; r_s 16.4 vs 7.9 (D -0.63 vs -0.83). Inside r_l climbing gains lin_vel +24, creep +13.6 (ours), swing_h +11.3 (ours), single_support +6 and loses dof_dev -9.4, impact -2.8, side_graze -2.6, orientation -2.8. So r_l pays climbing; foothold and style pay standing.
- How PPO weighs the groups (log adv_sd = std x w, 25,800-26,000: 0.163 / 0.0157 / 0.0064): joint (v20) per unit raw reward l : f : s = 1 : 0.25 : 0.20; separate (v21, std_g 0.175 / 0.072 / 0.032) = 1 : 0.61 : 1.08. Style pull x5.4, foothold x2.4.
- 10-s value from the foot, r_l units, natural v20 behaviour (12.5 % of attempts fall): v20 config refuse 16.7, climb 20.0, fall 9.3 -> attempting +2.0, break-even fall rate 31 %. v21 config refuse 24.7, climb 24.7, fall 11.7 -> attempting -1.6, break-even 0.3 %. Change -3.6 = style group under separate norm -1.64, foothold group -1.31, sigma_v 0.1 -> 0.25 -0.65 (standing gets exp(-0.36/0.25) = 0.24 of lin_vel instead of 0.03). Base-frame foothold targets: 0.00 (r_f identical both frames). Omni-top / Table 5 omni / vy clamp / beam heading do not enter the stair-foot reward. Same with v21's pushed attempts (29 % falls): break-even fall rate 18 % (v20 config) -> 4.6 % (v21). Own-critic GAE advantage (v20 checkpoint) refuse vs attempt mix: joint 0.18 vs 0.44, separate 0.27 vs 0.29.
- Style check: yes, the style group favours standing at the foot: r_s 0.33 per step standing vs 0.16 climbing (v20 D), v21 D 0.31 at the foot vs 0.21 in forced climbs; r_s rising 0.25 -> 0.32 over v21 is the policy moving to the style-paid stance.
- The 5,050 -> 7,110 refusal jump (coordinator note) is not the extended map: at 5,050 (r7_v14_p250) 36/50 leg-5 trials stalled at "ramps, front side" and never reached the stairs (0 stair refusals for that reason); 7,110 is r8_v15_p100 = the first back-to-paper run (separate normalisation, sigma 0.25, fresh discriminators), 28/50 stair refusals - the same mechanism. Extended-map mechanics were not tested by this probe (own stairs tiles only); the remaining ext suspect is the fatal wall touch for route robots on ext stairs (v20's no-term applies only to stairs/platform tiles); check proposed, map kept (user 2026-11-14).
- What went wrong earlier: E8's per-term check called env.step without the foot net, so r_f lacked the imagined-swing part (0.99 vs 0.91; with it 0.89 vs 0.60) and the foothold pull toward standing was missed; the `v21_d40_*` probes used `sig_v_to` = 0 from v21's args.json, so their lin_vel is 0 (their stall/pass counts are valid, their reward columns are not).
- Proposed (not applied; user decides): smallest fix = `--adv_norm joint` on the v21 flags (keeps sigma_v 0.25 and the other paper items): attempting +1.4 (break-even fall 25 % natural, 14 % pushed). queue40's armed `extend_v20` (v20 flags) gives +2.0 and is consistent with this. Fallback only: stall penalty (patch `runs/refuse_diag/patch_stallpen.py`, default off, tested on a code copy) - see agentic/state/ssr.md 2026-11-14 23:45.
- A5 (2026-11-15 12:00, ACP agent): stage-3 route by checkpoint 21,000 80.5%, 21,500 79.5%, 22,500 87%, 23,000 99%, 23,500 95%, 24,500 84.5%, 25,000 81.5% (pooled 86.7%). Spread beyond chance (chi2 about 59, 6 dof) but no trend and no change in training metrics (goal success 0.55 -> 0.63, std, KL, planner losses smooth; goal levels full at 8.99/9 since 21,000): checkpoint-to-checkpoint jitter; 23,000 is the top of the swing, its true rate probably lower. Failures are falls whose section moves (4x4s at 21,000/21,500/24,500, stairs at 25,000); no hard seeds. Training courses do not predict the route. Actions: re-score 23,000 on 200 fresh seeds (1200-1399, coordinator started 12:00 UTC); fixed final-checkpoint rule proposed (replace the best only if the new lower 95% bound exceeds the best's point estimate; confirm on fresh seeds); weight averaging of neighbouring checkpoints offline is an option for the user.
- X1 (2026-11-15, cross-track hardware note; user: "Even in previous sonic training, the waist pitch joint caused problems because of its limited range and torque capacity"): the X2 waist pitch has a small range (about +-18 deg / +-0.31 rad) and a 24 N*m torque limit. Seen here: the v4 paper reference sits at its +18 deg limit on all stairs/blocks frames; Hiking v9 holds its action at -18 to -28 (torque saturated above |action| 5.8) and every runaway starts there; SSR does not drive the waist at all. Rule for new recipes: treat waist pitch as a constrained joint (clip, saturation penalty or excluded from the action space as SSR does), and check its saturation share in every pre-flight.
- X2 (2026-11-15, coordinator re-read of the SSR paper, arXiv 2605.30770): the policy controls 21 joints - 6 per leg, 4 per arm, and ONE waist joint (waist yaw; App. D Table 14 lists only "Waist yaw", Kp 100, Kd 3, action scale 0.2); waist pitch and roll are not policy-controlled and the paper gives no reason or gains for them. The depth camera is "a forward-facing Intel RealSense D435i ... mounted at the waist" (Sec. 4.1, App. D), not the chest; the paper gives no reason for the mount. Our SSR (and Hiking) eval/training put the camera on torso_link (chest, above the waist pitch/roll joints), so with SSR's PD-held waist (kp 40, 24 N*m on the measured plant) the camera sways with the torso - a deviation not in the audit (missed). User's hypothesis (keep the camera fixed relative to the pelvis so its view of the ground ahead does not change when the torso leans on ramps or stairs) is consistent with the paper's design but is not stated in it. Open: decide the real-robot RealSense mount (pelvis/waist vs chest) before the depth-to-height-map work.
- H7d SUMMARY (2026-11-15, action runaway - cause and fixes in one place): Hiking's actions were unclipped and fed back as observations; a rare robot's actions grew about 1.6x per step to 1e8 (action_rate reward -9e9), and batch-wide advantage normalisation let that one robot dominate updates. Effects: v7 NaN crashes (14,246 / 14,921 / 15,050), guard stop at 20,560, the v9 collapse at 22,114 (episode 970 -> 46, level 5.9 -> 0), a guarded stop at 25,860. Fixes, all live: (1) rollout guard - skip any update with obs > 1e4, action > 1e3, reward > 1e4 or non-finite, 3 skips -> resume from the newest checkpoint (H7b); (2) collapse watchdog crashguard_v9b - episode < 0.4x and reward < 0.5x median -> automatic rollback; (3) action clip +-20 on 28 joints, waist_pitch +-50 (user decision; SuperMimic and ACP clip +-20, SSR +-8), applied in InstinctLab's vec-env wrapper so both the applied target and the fed-back action are bounded, live since 13:19 UTC (H7c). Rule added to TRACK_RULES: every policy clips its actions.
- X3 (2026-11-15, chest camera mount work, `docs/hardware/chest_camera_mount/`): Hiking and SSR trained with different chest camera poses on torso_link - Hiking depth origin (131, 0, 259) mm, 61.6 deg down; SSR (126, 0, 160) mm, 59.5 deg down, 99 mm lower. SSR's offset is marked "PROVISIONAL until Hiking fixes the X2 offset" in `models/ssr/code/ssr/x2.py` and was never updated. Consequence: one bracket cannot serve both (SSR randomised height only +-50 mm; Hiking -105/+50 mm), so there are two printed variants (H, S). At SSR's exact pose the camera would touch the shell by 3.1 mm; the S bracket sits 5 mm further forward (inside SSR's +-20 mm forward randomisation). Open: if SSR training resumes, align its camera pose with Hiking's (owner: SSR track, at restart).
- X3b (2026-11-15, user: the chest below the seam is detachable): the chest front is two mesh pieces; the detachable lower panel (with the sensor window) ends at an arch, z 247 mm on the centreline down to 177 mm at |y| 80. The first brackets bonded 23 of 39 cm2 (Hiking) and all 35 cm2 (SSR) to that panel. Revised: one 50.5 cm2 pad for both, y -60..+45, from 4 mm above the seam to z 292, on the fixed upper shell; SSR hangs its camera about 80 mm below on two rails. Camera poses unchanged. Open: verify the seam on the real robot before taping; SSR rail stiffness and tape strength in falls untested (Hiking is the real-robot candidate).
- X4 (2026-11-19, ACP on the real X2, robot hanging on the gantry, MC zero torque, a 4x4 lying about 0.5 m ahead): the deployed stack (`deploy/acp/robot/perception_view.py`: live chest depth + torso IMU + joints through ACPCore.step and models/student.onnx; subscribe-only, 4 s, 57 frames) with the mount calibration (+8.25 / -3.5 deg) turns the flat floor into a ramp rising forward (about +0.37 m over 1.2 m), and the 4x4 is only a faint edge. Cause: camera pitch. The stack's camera axis is 73 deg below horizontal (IMU torso pitch about +3 deg: pelvis -16.7, waist pitch +20). A floor-plane fit on the same depth gives 85 deg, 0.88 m above the floor. Replaying the captured frames offline with +20.25 deg instead of +8.25 flattens the floor and shows the 4x4 as a bar about 0.35 m ahead. Open: (1) where the 12 deg comes from (mount moved since calibration, torso IMU vs torso_link offset, or the gantry pose; a standing check with MC holding the robot settles it); (2) the launch script's X2P_CAM_PITCH / ROLL default to 0, so set them explicitly; (3) the hanging robot is out of distribution (legs fill the lower half of the frame). Earlier the same day the coordinator wrongly reported "no blocks" from its own plane fit over a hand-picked window. Rule: results come from the stack itself, and the colour image is used only to check them (USER_RULES 2026-11-19).
- X4b (2026-11-19, same day, robot standing under MC, 4x4 ahead): the deployed stack live at +8.25 / -3.5 gives a flat floor and the 4x4 as a bar about 0.4 m ahead, 6.8 cm high (real 8.9 cm; the student smooths small edges). Replaying the standing frames: floor slope forward -0.7 deg at +8.25 (+1.8 at +6, -2.6 at +10), so the pitch calibration holds; sideways slope +2.4 deg (roll 0: +3.6, roll -7: +1.2), small, roll left at -3.5. So X4's 12 deg was the gantry pose (hanging, limp), not the mount. Still open from X4: the launch script's camera offsets default to 0.
- X5 (2026-11-19, first live ACP runs on the real X2, gantry): the leaky assist worked (joint names, signs, default pose, gains OK). Run 1: the policy "went out of control", plug pulled, logs lost in the page cache. Black box added (SONIC's deploy_logger layout, rows from a writer thread, fsync every second). Run 2 (e-stopped after 4.6 s, `log/blackbox/run_20261120_025933` on PC2): the policy ran at 19 Hz instead of 50 (step compute p50 33 ms, p90 90 ms, max 153 ms; gaps up to 161 ms) while the assist ran at 3 ms. The student's run time doubled in the same window (11 -> 24 ms): planner and student both on CUDA from two threads waited on each other. Ankle roll reached 8.2 / 6.7 rad/s while meant to stand still. IMU inputs plausible (gravity in pelvis (0.18, -0.02, -0.98), gyro ~0 at start), scans 27 ms old typical. head_pitch action at the -20 clip on 93% of steps (also in every sim run; the user does not care about the head). Fix (sim-tested, on PC2): planner on the CPU (sim 0.43 ms, max 0.99), student alone on the GPU, planner/tracker ms in the black box, and a guard: 5 policy steps in a row over 25 ms compute or 40 ms apart -> SAFE. Open: the Orin's real planner/tracker times and policy rate at the next live run.
- X6 (2026-11-19, ACP live on the X2 with the timing fixes: ONNX tracker on 1 thread, planner on CPU, student on GPU; gantry; black boxes `run_20261120_031757` and `run_20261120_042635` on PC2): the policy runs at 49.8 Hz (step compute p50 12-13 ms, p99 25-29 ms) and stands, walks a few steps and turns in place, but balance is poor: feet bump, it leans and tips repeatedly (user). Pushes: recovers even from 44.5 deg tilt, but by frantic stepping, the planner's foot occupancy flipping every 0.1-0.2 s; in 57-59 s leg targets jumped up to 1.49 rad in one 20 ms step, joints reached 16.5 rad/s, PD torque demand 150-330% of the motor limit on 9 of 12 leg joints; calm standing already uses 75% of the ankle-pitch limit. Command-to-motion lag measured from target vs measured joint (cross-correlation of their rates): 60-112 ms per leg joint, median 80 ms; the ACP recipe models no actuator, observation or odometry delay (RECIPE.md: "odometry delay 50-100 ms not modelled", paper specifies it). Walking run: 134 s of policy, 19 s commanded motion, tilt p50 4.2 / p90 6.6 / max 19.8 deg on the gantry, leg target jump p99 0.25 / max 2.70 rad, joint speed max 15.5 rad/s. Hypothesis: the policy, trained without delay, overcorrects against legs that answer ~80 ms late. Next: (1) reproduce in the deploy sim with the measured delay injected; (2) if reproduced, fine-tune ACP with randomised actuator delay (about 40-120 ms) and the real torque limits (ACP track, Nebius); (3) robot again only when the delayed sim stands and recovers calmly. Open: split the 80 ms into transport vs motor response (same measurement in sim).
- X7 (2026-11-19, reproducing X6 in the deploy sim; user: "the gantry didn't take any of the robot's weight, I had to hold it all the time"; user's hypothesis: weak real leg joints / kp-kd): the real run's pad commands replayed in the deploy sim (`deploy/hiking/logs/repro/`, `--gain-scale` and `--cmd-delay-ms` sim-only node options). Results vs real (feet apart mean/min, tilt p90/max, joint speed p99, target jump p99): real 0.19/0.08 m, 6.6/19.8 deg, 3.4 rad/s, 0.25 rad. Sim nominal gains: stable, 4.1/5.0 deg, 0.4 rad/s, 0.03 rad. 80 ms command delay: stable (3.8/7.6 deg) - delay alone is NOT the cause (my earlier lead was wrong). Gains x0.85, x0.85 + 40 ms, x0.7: stable. x0.6: 8.7/20.4 deg, 2.2 rad/s, 0.24 rad - matches the real run. x0.55: falls after 103 s; x0.5: falls after 7 s; x0.3: after 4 s. Conclusion: the real legs behave as if they deliver ~0.55-0.6 of the trained stiffness/damping (training: kp x U(0.9, 1.1), MuJoCo implicitfast PD). SONIC hit the same effect: its deploy has per-group kp scales "to recover the loop gain IsaacLab's implicit PD lent training" (gear_sonic_deploy safety.hpp) and clamps leg targets to 0.70 rad from the current angle (max_target_dev_leg); our deploy has neither. Open: measure the real stiffness directly (black box now logs joint effort; not yet on PC2); then either raise the sent kp/kd toward the trained loop gain (within motor limits) and add a leg target clamp, or retrain with kp/kd randomised down to ~0.5x.
- X8 (2026-11-19, real stiffness measured; black box `run_20261120_062203`, 33.5 s of STAND, joint effort logged): measured joint torque fitted against the commanded PD terms (tau = a*kp*(target - q) - b*kd*qd + c, best lag 20 ms, 40 ms ankle roll right). Effective kp factor a / kd factor b (R2): hip pitch 0.79 / 0.91-0.97 (0.90); hip roll 0.94 / 0.64-0.72 (0.95); hip yaw 0.52-0.54 / 0.74-0.78 (0.63-0.69); knee 0.52-0.55 / 0.72-0.86 (0.60-0.63); ankle pitch 0.76-0.77 / 0.19-0.37 (0.83-0.88); ankle roll 0.52-0.55 / 0.18-0.26 (0.61-0.68). Median kp factor 0.65 - matches the sim reproduction (X7: x0.6 reproduces the real run, x0.7 is stable). Not torque saturation: measured ankle torque >= 90 % of the limit on 0-5 % of steps (the commanded PD exceeds the limit on 1-5 %, up to 110 N*m on ankle pitch, 211 on the knee). Caveat: "effort" is the motor's reported torque (likely current-based); R2 0.6-0.95. Ankle damping is the weakest (0.2-0.4 of trained). Next: per-joint real-factor model in the sim to confirm, then test compensation (sent kp/kd divided by the factors, within limits) plus a leg target clamp in that sim before the robot; else retrain with kp x U(0.5, 1.1), kd lower.
- X9 (2026-11-19, command timing on the real X2; user asked to log it as an issue): our deploy node is Python (rclpy; the 50 Hz policy step, the student thread, ROS callbacks and the "500 Hz" command writer share one interpreter lock). Measured on PC2 with the node idling in SAFE (step 4.5 ms): leg commands reach the bus at 208 Hz, not 500; gap p50 4.2 ms, p99 13 ms, max 20.7 ms, 40 gaps over 10 ms in 5 s. Leg state from the robot: 925 Hz (gap max 10 ms). MC publishes ~490 Hz steadily (sequence numbers); SONIC's X2 deploy is C++ with a dedicated 500 Hz writer and an ~8 Hz low-pass on leg targets. During the policy (13 ms steps) the rate is likely lower. Open: whether this is what makes the joints look weak (X8) - only if the motor side does not simply hold the last command (our targets change only at 50 Hz); proposed fix: a C++ 500 Hz writer as SONIC's, then re-measure the stiffness on the robot.
- X10 (2026-11-19, MC scan vs our node, same fit): the May MC scan (`GR00T-WholeBodyControl/mc_motor_scan_1778884089.jsonl`, 33 s, the scan the training plant mc_20261024 came from). MC sends the same command content as our node (kp/kd hip 100/3-4, knee 150/5, ankle pitch 40/3, ankle roll 30/2; velocity 0, feed-forward 0). MC leg commands: 487 Hz, gap p50 2.00 ms, p99 2.31 ms (11 pauses up to 222 ms, also in the state stream = recorder stalls). Same torque fit as X8: MC at 1 kHz median kp factor 0.98 (R2 0.91-1.00, torque follows the command after 1-4 ms); MC sampled at 50 Hz as our black box: median 0.96 (R2 0.76-0.99, best lag 0 ms). Our node (X8): median 0.65 (R2 0.60-0.95, best lag 20 ms). Same robot, gains, command fields and method: the difference is how our commands reach the motors (X9: Python writer at 208 Hz, gaps to 21 ms, plus the delay between the control step and the actual publish). Part of the 0.65 may be the fit misaligning a delayed command at 50 Hz, but either way the torque the legs produce at the right moment is lower than trained. Fix first: a C++ 500 Hz writer (as SONIC), then re-measure the stiffness on the robot before any compensation or retraining.
- X11 (2026-11-19, CORRECTION to X8 and X10, and the C++ relay on the robot): my X8 fit paired the target sent one step earlier with the position from that same earlier step; the torque at sample k comes from the target sent at k-1 against the position at k. Corrected fit (`tau_k = a*kp*(target_{k-1} - q_k) - b*kd*qd_k + c`): old Python writer (run_20261120_062203) median kp factor 0.77 (hip roll 0.95-0.96, hip pitch 0.87, ankle pitch 0.78-0.81, knee 0.74-0.77, hip yaw 0.73, ankle roll 0.65-0.69; R2 0.73-0.96), not 0.65. With the C++ relay (`deploy/acp/robot/cmd_relay`, 500 Hz out, gaps up to 6-14 ms on the Orin; run_20261120_065615): median 0.98 (0.94-0.99 on all leg joints except right ankle pitch 0.86 and right ankle roll 0.89; R2 0.89-0.99) = MC's 0.96-0.98. So the relay restored the delivered stiffness; X8's "0.65" and its per-joint compensation factors are withdrawn (no gain compensation). The user still finds the balance "a little stiffer but not good enough": the remaining cause is something else.
- X12 (2026-11-19, SONIC baseline on the same robot, 2 x 60 s bus recordings with `deploy/acp/robot/bus_record.py`, quiet then pushes): SONIC's leg command stream is 233-234 Hz with gap p99 32-34 ms and pauses up to 338-349 ms - worse than our old Python writer - and it stands; so command timing (X9/X10) was never the cause. Delivered stiffness under SONIC (corrected fit) median 0.97 = ours with the relay (0.98). SONIC sends the armature-rule gains (hip/knee kp 114.8 kd 7.31, ankle pitch 34.9/2.22, ankle roll 12.4/0.79; damping ratio 2), not the MC set ACP trained with (hip 100/3-4, knee 150/5, ankle 40/3, 30/2). Behaviour: SONIC quiet leg speed p99 1.4 rad/s, tilt p90 2.3 / max 9.6 deg, leg target change per 20 ms p99 0.10 / max 0.70 rad; under pushes 1.9 rad/s, 4.7 / 12.7 deg, 0.16 / 0.89 rad. Our ACP with the relay (held by the user): 3.7 rad/s, 7.3 / 20.5 deg, 0.28 / 2.7 rad. Conclusion: the deploy pipeline now matches or beats SONIC's (stiffness, timing); the remaining gap is the ACP policy itself (twice as jumpy at rest, trained with no delay or actuator randomisation and narrow kp/kd range; less damped gain set). Next: (1) deploy-only test of SONIC's leg target low-pass (~8 Hz) and clamp (0.7 rad); (2) if not enough, an ACP fine-tune with wide delay / actuator randomisation, an action-rate penalty and possibly the armature gains.
- X13 (2026-11-19, ROOT CAUSE of the real-robot instability: IMU placement; write-up docs/findings/real_robot_instability_imu_frame.md): `/aima/hal/imu/torso/state` behaves as if it sits below the waist (pelvis side), but our deploy converted it through the waist FK as a torso_link IMU (Hiking's assumption, marked unverified in deploy/hiking/STATUS.md; our sim bridge was switched to `--imu-from torso` to match it, so the sim could not catch it; SONIC uses the raw IMU as its pelvis-frame base). Evidence: on the relay run, the pelvis pitch from the IMU + waist conversion differed from the feet-flat leg kinematics by -10.5 / -6.6 / -6.2 / -3.1 deg at waist pitch +9.9 / +8.0 / +7.5 / +3.8 deg (slope -0.93, corr -0.73; roll differed by +1.5 to +5.8 deg); in sim the same check agrees to 0.0 deg. The policy saw false tilts of 3-10 deg (ACP trains with only +-0.7 deg gravity noise; SONIC with +-3-6 deg) and straightened and splayed the legs (real: left knee ~0 deg, right ~7 deg vs 15-20 deg for SONIC and the sim). Reproduced in sim (bridge `--imu-from pelvis` with the old conversion, the real run's pad commands): the robot straightens one knee (targets 88 / 5 deg), sees a false roll of +13-15 deg and falls after 4 s. Fix: `--imu-on-pelvis` (hiking_deploy/core.IMU_ON_PELVIS: the IMU is used as the pelvis; torso and camera from the pelvis + waist joints): the same sim stands and walks for 219 s with tilt p90 1.7 / max 3.9 deg (the calmest run so far). Training follow-up in docs/TODO_next_training.md (balance-input noise at SONIC's level). Open: verify on the robot; re-check the camera mount calibration (+8.25 deg), which was measured through the old assumption (also explains the 12 deg camera error on the gantry with the waist at +20 deg).
- X14 (2026-11-19, the IMU fix VERIFIED on the real X2; black box `run_20261120_075719`, 156 s of policy, C++ relay, `--imu-on-pelvis`): the user: "super stable now". Quiet standing (53 s): torso tilt p90 2.9 / max 3.2 deg, leg joint speed p99 0.03 rad/s, leg target change per step p99 0.002 rad (SONIC quiet: 2.3 / 9.6 deg, 1.4 rad/s, 0.10 rad; before the fix: 6.4-7.3 / 20.5 deg, 2.5-3.7 rad/s, 0.18-0.28 rad); knees 6.2 / 9.5 deg, hip roll +3.2 / -4.6 deg (both outward; before: about 0 / 7 deg knees, hip roll inward). Under 13 nudges (gyro up to 3.0 rad/s): peak tilt 4.5-9.7 deg, back under 4 deg within 0.2-2.9 s (3 not within 3 s), leg speed up to 11.8 rad/s, target jump up to 0.76 rad; whole run tilt p90 4.3 / max 9.7 deg, leg speed p99 0.82 rad/s, target jump p99 0.050 rad (SONIC under pushes: 4.7 / 12.7 deg, 1.9 rad/s, 0.16 rad). Conclusion: the instability of X5-X12 was our own deploy bug (X13), not the policy, the motors or command timing. Open: knees straighter than SONIC / sim (6-10 vs 15-20 deg).
- X15 (2026-11-19, shoulders feel weak; user: "probably good for our setup, but log it and find out why"): not a deploy fault - the shoulders deliver the commanded stiffness (fit 0.92-1.00, R2 0.54-0.98). It is the trained gain set: ACP trained with the MC plant (mc_20261024 = the vendor MC's STAND_DEFAULT gains): shoulder pitch / roll / yaw kp 30 / 20 / 20, kd 1.0, elbow 50 / 1.0. SONIC sends the armature-rule gains: shoulder pitch and roll kp 34.9 kd 2.22, shoulder yaw / elbow 12.4 / 0.79. So ACP's shoulder roll is 0.57x SONIC's stiffness and its shoulder pitch / roll damping 0.45x; under gravity the shoulder rolls sag 3.2 / 5.7 deg from their targets while standing. Changing them at deploy time would change what the policy was trained with; candidate for the next training run if the arms matter (docs/TODO_next_training.md).
