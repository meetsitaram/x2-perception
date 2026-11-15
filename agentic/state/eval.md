# eval state

Updated 2026-11-11 (eval monitor, phase 1 done: desktop CPU only, nothing run on a GPU, no track tree touched).

## Where things are

- Definition: `docs/EVAL.md` (finalised 2026-11-11 with the user's answers; no open questions).
- Code: `eval/` in this repo (not committed).
  - `apparatus.py` loads an apparatus folder; `check_routes.py` checks and draws the routes; `x2.py` builds the X2 on it;
    `harness.py` runs and scores trials; `render.py` makes the videos; `adapters/base.py` is the adapter interface,
    `adapters/stand.py` the two sanity adapters; `selftest.py` tests the contact scoring; `sheet.py` makes contact sheets.
  - `collect_apparatus.py`, `collect_nist.py`: desktop-only, how the data folders were made (they read `ssr-x2` and
    `loco-manip-apparatus`).
  - `eval/data/` (apparatus copies) and `eval/out/` (runs) are git-ignored; `.gitignore` got those two lines and `__pycache__/`.
- Shared filesystem: inputs in `/mnt/filesystem-g4/data/apparatus/{ours,nist}` (each with a README); eval code copy, results and
  renders in `/mnt/filesystem-g4/eval/` (`code/`, `phase1/`); heartbeat `/mnt/filesystem-g4/status/eval.json`.
- Run it (desktop, or a node's CPU with `PA_APPARATUS_DIR=/mnt/filesystem-g4/data/apparatus PA_ROBOT_DIR=/mnt/filesystem-g4/data/robots/x2_ultra`):
  `python eval/harness.py --adapter <name or module:Class> --checkpoint <path> --apparatus ours|nist --trials 200 --out <dir>`,
  then `python eval/render.py <dir> --track <name>`. Needs python with mujoco (3.15 used), numpy, pillow, imageio-ffmpeg, psutil.

## Baseline (task 2)

- Found: **13 / 50 = 26 %**, `acp3_v2` checkpoint 1000 + planner `planner_v2_hist1`, our apparatus, full mobility route (22
  waypoints), MuJoCo Warp, 2026-11-02. Source: `gwbc-v6/scripts/acp/README.md` lines 25-30 and `ISSUES_LOG.md` lines 326-334.
  No file says "25 %" itself; this is the only full-route result near it with a trial count. Its definition: 0.5 m arrival
  radius, 1 s station holds, success read from the waypoint count (12 of the 13 tipped after the last waypoint), fall = root
  under 0.35 m. Failures stopped at: start 6, door / 4x4s 8, ramps 12, stairs 10, landing 1.
- Files: tracker `gwbc-supermimic/logs_sm/acp3_v2/ckpt_001000.pt` (exists, 225 MB); planner copy
  `gwbc-supermimic/logs_sm/acp_artifacts/planner/planner_v2_hist1.pt` (exists). The raw output of the 50 runs is gone (it was in
  another session's scratchpad); the video `gwbc-v6/scripts/acp/media/race50_acp3_v2.mp4` remains.
- Other numbers seen: SSR route evals all 0 / 50 (`ssr-x2/runs/*/route/summary.json`); `acp3_v5` 3 / 50; no NIST success rate
  anywhere; the G1 has single runs per section only.
- **Not done:** re-measuring it under the common eval. It needs an adapter around the SuperMimic rollout code
  (`gwbc-supermimic`, `scripts/acp/eval/pico_rollout_planner.py`: planner + tracker, 17 x 11 height scan), which runs on
  MuJoCo Warp on a GPU. Decision taken: do it on a node when one is idle (or port to CPU), after the tracks' first checkpoints;
  until then the baseline row reads "26 % (13 / 50) under the old, looser definition; not measured under this one".

## Apparatuses and routes (tasks 1 and 3; corrected by the user 2026-11-11)

- User corrections, all applied: the NIST route missed the pocket behind the upper 4x4 and stopped at the top of the stairs;
  both 4x4s must be stepped over on each apparatus; the door must be open.
- Ours: 22 waypoints, 19.2 m, 5 stations. Five waypoints differ from the source route (1: 104 -> 107 in; 2 and 6: -> 109 in; 7
  and 8: to (2.80, 1.60) and (3.38, 1.78) m so the back 4x4 is crossed 0.27 m from its end, was 0.14 m). It already visited
  buttons 1 and 2 and ended at button 5.
- NIST: 35 waypoints, 22.5 m, 11 stations (was 31, 19.0 m, 9): added the pocket station (8.88, 5.00) with its approach point
  (9.10, 4.35) before and after, a square crossing of the lower beam (9.246, 4.050) -> (8.750, 3.584) replacing two waypoints
  0.14 m from the beam, and the landing end station (1.84, 3.35).
- `route.json` now carries `beams` and `door`. `check_routes.py` checks corridor, steps, beam crossings (0.25 m from the ends),
  arrival discs against beams, and the doorway (0.60 m minimum). Both PASS: tightest wall clearance 0.25 m (ours), 0.28 m (NIST);
  doorway 0.87 m (ours, leaf open 90 deg) and 0.85 m (NIST, leaf open 94.5 deg, added to the boxes as one box).
- The harness records `beams_crossed` per trial and fails a finished trial that did not cross both ("went round a 4x4").
  Tested: carrying the pelvis through the gap between our beams counts nothing; over the beam counts one.
- NIST walk raster and wall boxes did not exist; made here with the rule that reproduces our walk raster exactly. The first
  set of NIST boxes (MuJoCo's own box per mesh piece) was wrong: it closed the doorway. Replaced by boxes fitted to the vertices.
- NIST data is derived from CAD with no licence: stays on the shared filesystem, not in git, not published.
- Not known: where the NIST mobility buttons 1 and 2 physically are (not in the CAD); the pocket station is placed from the
  user's drawing and the rules text.

## Side hits and the stumble penalty (user, 2026-11-11)

- Definition and formula: `docs/EVAL.md`, "Side hits and the stumble penalty". Constants `HIT_*`, `SCUFF_P`, `BEAM_NEAR` in
  `eval/harness.py`; obstacle zones (`zones`) in each `route.json`; the 4x4 class comes from `beams`.
- Every result now has `side_hits` (per class), `side_hit_count`, `stumble_penalty`, `clean`; every summary has
  `clean_successes`, `clean_success_rate`, `clean_ci95`, `mean_stumble_penalty` and `side_hits` (by class, by section).
- `eval/selftest.py` covers it (4x4 side, wedge side, stair riser must count; 4x4 top, ramp slope, tread, flat must not; the
  height-field normals on each face). All pass. The weights are my proposal, not tuned on any policy.
- The floor stays a height field; no explicit box geoms were added (the normals separate faces from walking surfaces).
- Marked render: `eval/out/selftest_sidehit/renders/legs_seed0_t1-2.mp4` (self-test, not a policy).

## Adapters and logs (2026-11-11, after the user's transparency and sharing rules)

- Logs: every run folder gets `trials.csv` and `events.csv` beside `summary.json` and `results.json`; legend in `docs/EVAL.md`
  ("Logs of every eval run"). Run folders for real checkpoints: `/mnt/filesystem-g4/eval/<track>/<checkpoint>/<apparatus>_<protocol>/`.
- `eval/adapters/common.py` (numpy MLP, normaliser, history, heading-frame scan grid, bounded goal), `eval/adapters/acp.py`
  (from the ACP stage-1 code in `tracks/acp/runs/s1_v1/code_snapshot`), `eval/tools/export_actor.py` (a track runs it; needs torch).
  Tested only with random weights (`eval/out/_acp_plumbing`): shapes and plumbing, not behaviour.
- What each track must hand over is written in `agentic/state/SHARED_LESSONS.md` ("What the eval needs from each track"): the policy
  as npz or ONNX, an adapter card, and a 50-step `obs_check.npz` the adapter must reproduce.
- The desktop's Python has mujoco, numpy and onnxruntime, no torch. Installing torch would be a large download (needs the
  user's okay), so policies come as npz or ONNX.
- No address or login is in any eval file; `eval/node.sh` reads them from the git-ignored `agentic/config/nodes.local.json`.
- When can the first policy be scored: as soon as one track hands over a policy with its export, card and obs check. ACP
  can be scored from stage 1 on IF a contact source exists for its tracker (its own planner, stage 2; or an interim source);
  SSR r3 stands still; Hiking X2 is early (iteration 250 of 30,000). From hand-over the adapter check takes me under an
  hour and 200 trials about 30 to 60 minutes of one CPU core per apparatus and protocol (estimate).

## The 2 November baseline: how it ran, and re-running it under the current rule (2026-11-11)

Correction (user): ACP "has a planner contact point estimator and the tracker or executor ... we already tried acp on our
apparatus and that's where we got 25%". Source code: `gwbc-v6/scripts/acp/` (same files in `gwbc-supermimic/scripts/acp/`,
where the checkpoints are); read only.

**How contact targets reached the tracker on the route** (`eval/pico_rollout_planner.py`, run by `eval/crowd/route50.sh`):
- Every 20 ms the planner (`stage2/planner.py`, MLP 204 -> 512 -> 512; weights `planner_v2_hist1.pt`) gets a 17 x 11 height
  scan in the heading frame (x -0.3 .. 1.3 m, y +-0.5 m, minus root height), the goal (direction to the active waypoint
  capped at 0.5, and the heading still to turn, or the station's facing), and 13 numbers of the current state (ankle
  positions, root height over the ground, velocity, the root's z axis). No reference motion.
- Per foot it outputs weights over the 187 scan points, an occupancy, a time to contact and "in contact now". The eval
  decoded with `PLAN_DECODE=mode` (strongest point, refined within 0.11 m), `STEP_MASK=0.03` (no point whose surroundings
  step by more than 3 cm) and `PLAN_LATCH=0.3` (target frozen in the world from 0.3 s before touchdown).
- That becomes the tracker's 54-number contact input: nine bodies x (target xyz in the heading frame relative to the root,
  time / 1.5 s, valid, in contact); only the two feet are filled, the other seven say "none".
- The tracker (`acp3_v2/ckpt_001000.pt`, SuperMimic policy: encoder 680 -> 32 on a zeroed goal block, decoder on 990 numbers
  of 10-step proprioception + 63 scan + 54 contacts + 30 goal command) outputs 31 joint targets at 50 Hz.
- The waypoints were the 22 of `crowd/route_roam_h1.json`, reached within 0.5 m, stations held 1 s; MuJoCo Warp on the GPU,
  walk raster plus wall boxes, scans on the same raster.

**What a like-for-like re-run needs** (all on the desktop CPU, nothing to download: the `sm` virtual environment already
has torch for the export):
1. Export the tracker to npz (done for the planner: `eval/out/baseline_export/planner_v2_hist1.npz`, 2 MB; the tracker is
   about 9 M weights).
2. Port the tracker's input to an adapter: the 990-number proprioception (10 steps, term-major, the deploy joint order),
   the zeroed goal block, the 9 x 7 scan, the 54 contacts (the `FootMlpPlanner` source in `eval/adapters/acp.py` already
   produces them; `contact_feat = 6`), the 30-number goal command, action order and scale, the `vendor_20260823` plant.
3. Check it: run the original simulator code on CPU for 50 steps of one robot and require the adapter to reproduce the
   planner input and output, the tracker input and the action. Without this check the number would not be trustworthy.
4. Run 200 trials per protocol on our apparatus (and NIST, never measured before) under the current rule.
- **Feasible on CPU: yes.** My estimate: 3 to 5 hours of work for steps 1 to 3, then about 1 hour of one core per 200 trials
  (4 runs: two apparatuses x nominal and randomised). Not started; waiting for the coordinator's go.
- **What it will and will not be:** same policy pair and decode, but MuJoCo on CPU instead of MuJoCo Warp, 2 cm sole
  spheres, the corrected route (5 waypoints moved), 0.20 m arrival, both 4x4s required, 1 s stand at the end, 180 s, and
  the stumble score. So the result is "the baseline policy under the current rule", expected to be lower than 26 %; it
  is not a reproduction of the 13 of 50 run (its raw output is gone and GPU physics is not repeatable). `route50.sh` sets no
  path lookahead (`ROAM_LOOKAHEAD` defaults to 0), so the goal aimed at the waypoint itself, as the adapter does.
- **Go given by the coordinator (2026-11-11); steps 1 to 3 done.** Adapter `eval/adapters/acp_baseline.py`; export and
  recording `eval/tools/baseline_dump.py` (run in the `sm` environment on CPU, CUDA hidden, Warp device "cpu"); check
  `eval/tools/baseline_check.py`; result `eval/out/baseline_export/check_result.txt`.
- **Step 3 result: PASS.** The original rollout code ran 50 steps from two start states on our apparatus (route start; 0.4 m
  short of the front 4x4 with a station waypoint). From each recorded state the adapter recomputed the planner's contact
  input, the 1,817-number tracker input and the action: largest difference 3.6e-06 (route start) and 7.1e-06 (before the
  4x4) in both the tracker input and the action; 0 of 900 planner valid / in-contact flags differ in each; default pose and
  action scale agree to 3e-08. One real mismatch found and fixed first: the original terrain code never returns a height
  below its floor (0); our raster has -0.108 m outside the deck, which moved a planned foothold by up to 9 mm near the lane
  edge. Not covered by the check: waypoint switching and holds (the harness's, under the current rule) and the physics.
- The trials start in the original's start pose (first frame of the route reference; adapter hook `start_q`), at the eval's
  start position and heading. Runs: `bash eval/run_baseline.sh ours|nist nominal|randomised 200` (one core, low priority,
  6 GB cap, 9-minute slices that continue where they stopped); results in `eval/out/baseline_20261102/<apparatus>_<protocol>/`.
- Tested so far: the exported planner, on the standing robot at three places of our route, plans both feet 0.2 to 0.4 m
  ahead on the floor, short of the 4x4, and one foot onto the first stair tread at the stair foot. Not compared with the
  original code's outputs yet (step 3).

## Results so far and the eval ladder (2026-11-11, evening)

- **Baseline pair on our apparatus, current rule, MuJoCo CPU, nominal protocol, 200 trials (seeds 1000-1199): 15 / 200 = 7.5 %**
  (95 % interval 4.6-12.0 %); clean 0 / 200; `eval/out/baseline_20261102/ours_nominal/` (summary, results, trials.csv, events.csv,
  renders of seeds 1156 best, 1061 median, 1136 failed). 153 of 185 failures end in the 4x4 section (78 heading back over the
  front 4x4, 36 at button 1, 21 at the back 4x4); 13 stalls on the ramps. Plant `vendor_20260823`.
  In THIS run the goal command shrank inside 0.5 m of every waypoint (`ACP_GOAL=shrink`), which the original run never
  did on the way to a non-station waypoint (it switched at 0.5 m). The adapter's default is now to keep the goal length
  except at stations; the 200 trials have NOT been re-run with it yet. Randomised protocol and NIST: not run.
- **Impulse numbers of that run are not to be quoted.** They were summed over whole contact episodes and over every contact
  point: 130 of 2,552 hit events lasted over 1 s and hold 87 % of the sum; the largest (2.8 million N s) is a left foot pressed
  into the 16 cm end face of the front ramp cell next to a brace post for 45.7 s in a stalled trial, with contact forces of
  3.5 to 4 kN per contact point (a jammed foot in the solver, not a unit error). Since then the impulse is taken over the
  first 0.3 s only. Still open: a foot pinched over a beam edge gets opposing contact forces, so "impulse" of the face contact
  is not the net impulse on the body, and most hits reach the penalty's cap; the penalty is in effect a hit count. Foot loads
  over 2 kN occur in 1 to 2 % of frames (worst 31 kN): the contact stiffness on the height field needs a look.
- **Ladder step 1 with the baseline pair (50 trials per test, nominal, on the training map):** `eval/out/baseline_20261102/ladder/step1.txt`.
  Flat: walk, stop-and-stand, turn in place, turn and return, backward, sideways all 50 / 50. One level log 32 / 50; slanted log
  43 / 50; wedges 25 / 50; stairs up 44 / 50; stairs down 48 / 50; log corridor with walls 16 / 50, with rails 20 / 50. Clean
  success 0 on every obstacle test (every crossing logs side hits).
- Ladder code `eval/ladder.py`, runner `eval/run_ladder.sh` (9-minute slices that continue). Step 2 is not built (route list
  to be fixed). Tiles are those of the training export; a held-out set is still to be generated.
- Bash `timeout` does not always kill a Windows python child: long runs stop themselves (`--wall-limit`, `LADDER_WALL`).
- Body velocities use `mjOBJ_XBODY` (the Hiking agent's finding; `mjOBJ_BODY` is the inertial frame). The baseline adapter
  reads the free joint directly and its 50-step check covers the gyro (3.6e-06). The ACP hand-over adapter had the wrong
  call and failed its check until fixed; it now passes (6e-08 input, 3e-06 action).
- Hand-overs: ACP `handover_pair_s1_001250_planner_v1` passes its observation check but is refused until its card names its
  plant; Hiking `x2_parkour_v1_250`: card has no plant either, adapter not written (its port to reuse:
  `tracks/hiking/code/x2_port/sim2sim/`); checkpoint 500's depth encoder outputs zero per the Hiking agent: not to be scored.
- Depth views: `eval/depth_views.py` -> `eval/out/depth_views_ours/` (12 points, flat-ground check, README); chest camera on
  `torso_link` (0.131, 0, 0.259), 61.6 deg down, 87 x 58 deg. Flat check: toes at 98 % of the image height, the 1.0 m floor line
  at 25.3 % from the top (rule 27 +- 3 %): met.

## Harness (task 4): what was verified

- Model: X2 loads on both apparatuses; 20 s of simulation takes about 2 s on one core.
- `stand` (hand-written ankle balance, a test fixture): stood 12 s on both apparatuses, feet loaded about 200 N each, pelvis
  0.62 m, no contact other than the soles. It tips after about 30 s, so it is not a long-run fixture.
- `zero` (default pose, PD only): tips forward and is scored as a fall at 1.4-1.5 s, 0 waypoints, in 5 / 5 trials.
- Kinematic sweep (pelvis carried along the route, no physics): 22 / 22 and 35 / 35 waypoints in order, all sections completed,
  each beam crossed (ours: front twice, back once; NIST: upper twice, lower once).
- `selftest.py`: stub into the 4x4, edge landing on it, clean landing on flat deck, arm and trunk against a wall: all pass.
- Time limit, stall and the randomised draw (friction, mass, payload, pushes) each ran once and gave the expected outcome.
- Renders looked at (stills from every video): chase and legs views of the stand run, the zero-policy fall and the two sweeps.

## Not verified / not there yet

- No learned policy has been through the harness: no real adapter exists (ACP, SSR, Hiking adapters come with checkpoints).
- `View.depth` and `View.height_scan(with_walls=True)` are written but untested against any track's training sensor.
- MuJoCo against the tracks' simulators (contact model, PD) is not compared; sim-to-sim gaps will show with the first policy.
- The randomisation ranges are my choice (typical values), not taken from a track's training config.
- Trials run one after another on one core; 1,000 trials of 60-180 s will need the worker option that is not written yet.
- The chase view near the door still shows faint wall layers over the robot; the legs view shows floor spikes where the walk
  raster keeps the feet of braces (they are real in the physics).

## Decisions taken without the user (2026-11-11)

- Arrival radius 0.20 m (baseline used 0.5 m); success needs 1 s standing after the last waypoint.
- Time limit 180 s and "button presses do not count; stations are 1 s stands" were confirmed by the user on 2026-11-11. Stall cut-off 45 s is mine.
- Touching a wall with an arm or the trunk is counted, not a failure; trunk or head on the floor is a fall.
- Sole spheres 2 cm in MuJoCo (5 mm in the MJCF) for sane height-field contact normals.
- Headline protocol is the randomised one; the nominal one is reported beside it.

## Resumed by a new eval monitor (2026-11-12 00:50 UTC)

- The previous monitor stopped (usage ran out). Found running: `eval/run_chain.sh` (bounded ladder slices, 9 x 9.5 min) on the
  ACP pair tracker 2,250 + planner (`handover_pair_s1_002250_mc_20261024_planner`, local copy `eval/data/handover/acp_2250`,
  plant `mc_20261024`), full step 1 (six flat tests + five skills x three levels), started 00:46 UTC. Left running; no second job.
- Re-ran the 2,250 hand-over check: `ALL PASS` (pair recording: actor input 5.9e-08, action 4.2e-06; reference-contact
  recording: 1.2e-07, 9.4e-07). The card has plant `mc_20261024`, kp and kd per joint.
- ACP pair 1,250 (plant `vendor_20260823`) step 1 is partial: 8 of 21 tests (`eval/out/acp/handover_pair_s1_001250_planner_v1/ladder/step1.txt`):
  flat walk / stop-stand / turn in place / sideways 50/50, turn-and-return 39/50 (11 stalled), backward 29/50 (21 stalled),
  log L1 49/50, log L2 43/50. Superseded by 2,250; finished only if the CPU is free after step 2.
- Fixes: three baseline step-1 summaries lacked the plant name (run before the adapter wrote it) -> filled in
  `vendor_20260823` with a note; the step tables now state the simulator and the plant of every row (flags mixed plants).
- Step 2 time limit: 180 s per route (`TIME_LIMIT_STEP2`), the apparatus limit; the routes are 7 to 15.5 m with stations and
  turns on the spot, and the baseline walks about 0.3 m/s on flat ground, so step 1's 60 s would cut them short. Stall 45 s.

## ACP pair 2,250 on ladder step 1 (2026-11-12 01:35 UTC, reported to the coordinator)

ACP tracker s1 2,250 + planner_mlp (plant `mc_20261024`), MuJoCo 3.15 CPU, nominal, 50 trials per test, held-out skill courses
(`eval/out/acp/handover_pair_s1_002250_mc_20261024_planner/ladder/step1.txt`; copy on the shared filesystem `eval/acp/...`).
- Flat gate PASSED (six flat tests 50/50).
- Profile L1/L2/L3: log 50/48/47 (level 3); wedge 50/50/44 (level 2); stairs_up 23/50/50 (none, in-order rule); stairs_down
  50/50/48 (level 3); corridor 50/50/49 (level 3). Weakest: stairs_up. Baseline for comparison: log 50/19/13, wedge 50/50/49,
  stairs_up 50/44/50, stairs_down 50/46/35, corridor 50/50/39.
- 29 falls (27 at the first 10 cm riser of stairs_up L1), 9 stalls, 3 time limits. Clean success ~0 on obstacles.
- `eval/profile_table.py` -> `eval/out/profile_table.{txt,csv}` (all runs with a profile).
- Next: step 2 baseline (running since 01:33 UTC, `eval/out/baseline_20261102/ladder/step2.*`), then ACP 2,250 on all eight routes.

## Coordinator decisions (2026-11-12, after the 2,250 step-1 report)

1. Skill profile = highest level passed, plus any lower level that failed (e.g. ACP 2,250 stairs_up "L3 (L1 failed)"); route
   eligibility on the levels actually passed. Implemented in `eval/ladder.py` (`collect`, `eligible_routes`): a route is held
   back only if a needed skill has no level passed at its tag or above, or (stairs) its riser is lower than the lowest level
   passed while a lower level failed. ACP 2,250: eligible street x2, corridors x2, stairs_U_turn (15 cm risers, L2 14 cm passed),
   home; NOT wedge_field_loop (wedge L3 44/50) nor amphitheatre_loop (12.5 cm risers, between the failed 10 cm L1 and the passed
   14 cm L2). Baseline: eligible street_out_and_back and wedge_field_loop only (stairs_up L2 failed at 44/50). All eight are run
   for both pairs anyway, eligibility is reported beside them.
2. After step 2: ACP 2,250 on our apparatus, 200 trials, nominal (`eval/run_acp_apparatus.sh`), beside the baseline's 15 and 4
   of 200, with per-section reach and where trials end; then renders (chase + legs: best, median, typical failure) of
   step-1 stairs_up L1 and of the apparatus run, to the coordinator for review.
- Render labels now say "MuJoCo, not the training simulator" and carry the protocol and trial count.
- The ACP hand-over adapter's goal is d / max(|d|, 0.3 m): it shrinks inside 0.3 m by construction (its training convention).

## Resumed again (2026-11-12 01:41 UTC): baseline step 2 done, ACP 2,250 apparatus running

- Baseline pair (plant `vendor_20260823`) on the eight step-2 routes, 50 trials each, MuJoCo CPU nominal (CHAIN DONE 02:25 UTC,
  collected; `eval/out/baseline_20261102/ladder/step2.txt`, copied to the shared filesystem): street_out_and_back 50/50,
  home_dead_end_and_back 49/50, stairs_U_turn 32/50, corridor_rails 11/50, wedge_field_loop 11/50, amphitheatre_loop 10/50,
  corridor_walls 4/50. Clean: street 50, home 49, amphitheatre 10, others 0. Falls dominate every failure.
- **street_doorway_out_and_back 0/50 is a COURSE DEFECT, not a result.** A 0.33 m "block" box (top 0.17-0.27 m above the floor) at
  (-1.20, 1.51), plus a rock and a plant (meshes as their boxes) further on, stand on the route's centre line. They are not in the
  walk raster that the height scans and the route check's step test use, and the wall test only looks 0.30-1.30 m above the
  ground, so the check passed it and the policy cannot see it; all 50 trials fall at the block (x -1.45 to -1.28 m, 5-12 s). No
  other step-2 route has a box on its path (checked all eight). Flagged in `eval/ladder.py` `STEP2_DEFECT` (printed under the
  row). Fix to choose: reroute the waypoints round the three objects, or put low boxes into the raster; the route check also needs
  a low-obstacle test. Not done yet.
- Running: ACP 2,250 pair on our apparatus, 200 trials nominal, started 02:26 UTC, bounded loop of `run_acp_apparatus.sh` slices
  (about 50 s wall per trial -> about 3 h). Log `eval/out/acp/handover_pair_s1_002250_mc_20261024_planner/ours_nominal.log`.

- **ACP 2,250 pair on our apparatus, 200 trials nominal (done 03:21 UTC): 0/200** (95 % 0.0-1.9 %), clean 0/200; 108 stalled,
  89 fell, 3 time limit; reach: door 84 % completed, 4x4s 20.5 %, ramps 6 %, stairs 0 %. 71 of 108 stalls stand 0.20-0.40 m from
  their waypoint: the adapter's goal shrinks inside 0.3 m (goal_r) and the eval needs 0.20 m. Reported to the coordinator 03:23.
- Coordinator's decisions (03:25 UTC): (1) after ACP step 2, run the variant "adapter goal convention matched to the eval radius":
  `ACP_GOAL_R=0.2 ACP_LOOKAHEAD=0.3 RUN_SUFFIX=_goalr02_look03`, 200 trials same seeds; then goal_r 0.2 alone
  (`RUN_SUFFIX=_goalr02`) to separate the two; report after (1). (2) Fix street_doorway: scan and route check must see every
  obstacle in the physics; re-check all eight routes; re-run that route for the baseline. (3) Renders.
- Done for (1): `eval/adapters/acp.py` env options `ACP_GOAL_R`, `ACP_LOOKAHEAD` (default off; hand-over check still ALL PASS,
  same differences); `run_acp_apparatus.sh` takes `RUN_SUFFIX`.
- Done for (2), code: `check_routes.py` has `check_floor_obstacles` (every floor-standing box that is not a 4x4, its part below
  0.30 m; blocked under 0.18 m (legs), "near" under 0.25 m). `adapters/base.py` height_scan reads `app.scan_height` when a course
  has one. The `ladder.py` change (tile `via` detour (-1.6, 1.0), (-0.6, 1.0) instead of the export point (-1.5, 1.5); `scan_boxes`
  for all eight step-2 routes; summaries record the scan) is in the scratchpad patch `patch_ladder.py`, to apply only when no
  ladder run is going (the running ACP step-2 chain re-imports ladder.py every slice). Tested on a copy: all eight routes PASS
  the new check, the doorway route is 13.6 m / 16 waypoints, the scan reads 0.416 m at the block.
  If the scratchpad is gone: re-do these three edits by hand from this description.
- Found by the new check on OUR APPARATUS (held-out route, not changed): leg 10 (button 2 -> ramps, back side) passes 0.16 m from
  the lower part of a brace at (3.61, 1.48) (point (3.65, 1.72), 0.05-0.30 m up), which the walk raster does not hold (reads 0);
  legs 7-11 pass 0.19-0.21 m from it and from a leaning board at (2.61, 2.10). NIST: PASS. To the coordinator.
- The training-map export's own street_front_L4_v0 route runs through the block: a map defect for the Hiking agent (exporter).
- Running: ACP 2,250 on the eight step-2 routes (chain started 03:22 UTC, `.../ladder/chain2.log`).

### Remaining list (as given by the coordinator)
1. (done) collect step 2 for the baseline.
2. The ACP 2,250 pair on our apparatus, 200 trials, nominal; repeat until it prints DONE: `RUN_LABEL="ACP pair tracker 2250 + planner, plant mc_20261024, current rule, MuJoCo CPU" bash eval/run_acp_apparatus.sh eval/data/handover/acp_2250 handover_pair_s1_002250_mc_20261024_planner ours nominal 200`. Then `python eval/report.py eval/out/acp/handover_pair_s1_002250_mc_20261024_planner/ours_nominal`, beside the baseline's `eval/out/baseline_20261102/ours_nominal_goalshrink` (15/200) and `ours_nominal_goalkeep` (4/200): success and clean success with trial counts and intervals, per-section reach, where trials end (falls and stalls separately), side hits by class (counts only; no force figures). This is the number the user is waiting for: report it to the coordinator as soon as it exists.
3. The ACP 2,250 pair on step 2 (launch like the baseline chain: `ACP_HANDOVER_DIR=eval/data/handover/acp_2250`, adapter `acp:ACPHandover`, step 2, 50 trials, out `eval/out/acp/handover_pair_s1_002250_mc_20261024_planner/ladder`, `LADDER_LABEL` set); note route eligibility per the profile rule in `docs/EVAL.md`.
4. Renders with `python eval/render.py <run dir> --seeds <best> <median> <failure> --track acp`: step-1 stairs_up L1 (`.../ladder/step1/stairs_up_L1`, a typical failure such as seed 1013) and the apparatus run (picks in its summary). Chase view and legs close-up, labelled "MuJoCo, not the training simulator" with plant, protocol and trial count; look at each yourself and check it against `trials.csv` and `events.csv`; send the coordinator the file paths and what each shows; nothing goes to the user without the coordinator's review.
5. Copy the changed `eval/ladder.py`, `eval/render.py`, `eval/run_acp_apparatus.sh` to `/mnt/filesystem-g4/eval/code/`, and the step-2 and apparatus results to `/mnt/filesystem-g4/eval/...`; update `/mnt/filesystem-g4/status/eval.json` and `agentic/state/eval.md`. (ladder.py and the baseline step-2 results copied 02:28 UTC.)

## Resumed (2026-11-12 04:54 UTC, new eval monitor session)

- Found: no desktop python eval alive. The bash process `wait_iter4.sh` is the coordinator's Hiking heartbeat watcher (not an eval).
  The ACP 2,250 variant `_goalr02_look03` had stopped at its wall limit after 39 / 200 trials (variant active: seeds differ
  from the plain run). Restarted as one background chain (scratchpad `variants.sh`): `_goalr02_look03` to 200, then `_goalr02`.
- ACP 2,250 step 2 (CHAIN DONE 04:38 UTC) ran with the walk-raster-only scan on ALL eight routes (`scan` absent in every
  summary); the ladder fix was applied after it (ladder.py 04:40 UTC). Only street_doorway is affected (no other route has a
  floor box on it). Results in `eval/out/acp/handover_pair_s1_002250_mc_20261024_planner/ladder/step2.txt`.
- street_doorway: the old 0/50 folders of both pairs moved to `step2/street_doorway_out_and_back_before_fix`; re-run of both
  pairs on the fixed course running (scratchpad `doorway.sh`). The course does NOT go through a doorway: it runs from the
  street to (3.2, 2.6), 0.25 m before the front-door opening in the house wall at x 3.45 (tile edge clip), and back; that
  tile has no door leaf. Only home_dead_end_and_back passes a doorway (0.78 m clear, its leaf swung clear).
- `check_routes.check_doorways` (new, called from `check`): every opening under a `door_header` box that the route crosses,
  clear width with every physics box at body height (walls, jambs, open leaf) >= DOOR_MIN 0.60 m and the route >= 0.25 m
  from either jamb. Negative test: a 0.28 m box in the home doorway -> "clear width 0.50 m is under 0.60 m" and "0.11 m from
  a jamb". All eight step-2 routes PASS the full check (walls, steps, floor boxes, doorways); NIST PASS; ours still FAILS on
  leg 10 (brace foot at (3.61, 1.48), 0.16 m), as reported before (held-out route, not changed).
- SSR adapter `eval/adapters/ssr.py` (SSRHandover, SSR_HANDOVER_DIR; limb-sphere occluders in the depth; SSR_CMD=keep
  variant), check `eval/tools/ssr_check.py` -> PASS on r6b_001600 (`eval/data/handover/ssr_r6b_1600/check_result.txt`):
  history frame 1.9e-06, depth 3.2e-04 (without the occluders 0.22: they matter), action 0, GRU 8e-07, targets 1.3e-06.
  In MuJoCo r6b 1600 does not start walking from standing below 0.6 m/s (0.2-0.5 -> 0 m/s; 0.6 -> 0.41; 0.8 -> 0.77) and
  stops dead when the harness's command slows to 0.36 m/s. Flat tests running both ways (standard, keep 0.6 m/s).
- Hiking adapter `eval/adapters/hiking.py` (HikingHandover, HIKING_HANDOVER_DIR; own body in the depth; command = card's
  training generator to the waypoint, vmax 0.7), check `eval/tools/hiking_check.py` (needs pyyaml: run in the `sm` venv with
  a COPY of the port) -> PASS on x2_parkour_v4 model_500: ONNX 1.2e-07; every proprio term 0 and depth 1.2e-07 against the
  track's own MuJoCo port on 60 perturbed states. The card's obs_check is synthetic: nothing here is checked against Isaac.
  Flat tests running.
- ACP stage 2: newest backup `runs/s2_v1/ckpt_003500.pt` (copied to `eval/data/handover/acp_s2_3500_src/`, with its code
  snapshot). Actor 588 inputs (same layout as stage 1), but the checkpoint carries its own stage-2 planner (`planner_arch`
  attn) in `ckpt["planner"]`: a pair needs that planner exported (ONNX) with a card and obs check. Asked of the ACP track
  via the coordinator (CPU only).
- No commits or pushes (user, 2026-11-12, via the coordinator).

## 2026-11-12 06:20 UTC
- ACP 2,250 apparatus, goal_r 0.20 + lookahead 0.30 (200 trials, seeds 1000-1199): 3/200 (1.5 %, 0.5-4.3), clean 0; falls 102,
  stalls 91, time limit 4. Ends: door 5 F / 6 S; 4x4s 80 F / 61 S; ramps 15 / 17; wall 0 / 6; stairs 2 / 1. 61 of 91 stalls
  still stand 0.2-0.4 m from their waypoint; 42 at "door side" (waypoint 8, a 90 degree corner), turning command given
  (yaw -0.78 rad/s) but the robot does not turn. Renders (credit line now in render.py): seeds 1141, 1034 (successes), 1095
  (stall at door side) in `.../ours_nominal_goalr02_look03/renders/`. A 4x4 fall (1002) cannot be rendered: apparatus runs keep
  trajectories of the picked seeds only.
- `eval/speed_table.py` (per-command-speed table, coordinator's request): SSR 1600 and 2000 do not walk below 0.6 m/s;
  Hiking 500 does not walk at any command; ACP 3500 walks 0.48 m/s whatever the command (goal-driven, no speed input).
- Self-collision (SSR's question): eval and SSR use the same MJCF (sha256 4f6a4f5b...); all mesh pairs collide except
  parent-child and head-torso. Pelvis - hip_roll_link convex hulls are 2.6 mm apart at the default pose and close at about
  -0.05 rad hip roll (adduction); the URDF range goes to -0.235 rad. With SSR 1600 they touch in 92 % of physics steps
  standing (about 390 N each side) and 90 % walking (about 730 N). Likely a convex-hull artefact (a phantom stop like the
  head-torso one); not verified against the real hip geometry.
- ACP pair s2_v1 3500 (`eval/data/handover/acp_s2_3500`): ALL PASS (pair recording: input 6e-08, action 3.2e-06; reference
  recording: 2.4e-07, 1.5e-06). Ladder chain running (scratchpad `acp3500.sh`).
- SSR r6b 2000: check PASS (same numbers as 1600); flat tests running.

## 2026-11-12 07:25 UTC
- ACP pair s2_v1 3500 on our apparatus (200, nominal, goal_r 0.20 + lookahead 0.30, current model): 51/200 = 25.5 % (20.0-32.0),
  clean 0; 120 falls, 29 stalls. Sections completed: door 100, 4x4s 83.5, ramps 64, wall 64, stairs 42, landing 28.5 %.
  Fails: 4x4s 30 F / 3 S, ramps 15 / 24, stairs 42 / 2, landing 27 / 0 (+5 falls in the end stand).
  15 of the 33 landing / end falls end at the open landing edge by button 5: button 5 (10.414, 2.108) is 0.30 m from a 1.12 m
  drop at y 2.42 (rail 0.42 m away at body height, open below). Raised with the coordinator (held-out route, not changed).
  Renders in `eval/out/acp/pair_s2_v1_003500/ours_nominal_goalr02_look03/renders/` (1188 best full; 1022 stairs fall; 1009 landing
  fall; 1002 4x4 fall; 1001 ramps stall), looked at.
- ACP 2250 goal_r 0.20 alone: 2/200. Best goal handling: 0.20 + lookahead 0.30 (3/200 for 2250).
- Corrected X2 MJCF (two pelvis-hip_roll excludes) in `eval/out/x2_model_corrected/` (sha256 55dcb669...). Eval second model:
  X2_EXCLUDE_PAIRS=hip (labelled in every result via x2.model_note). Every result records `policy_trained_on`.
- Corrected-model flat (side numbers): ACP 3500 0/0/0/0, SSR 2000 7/0/0/0. Hiking 1000 (main number = corrected model, card
  command): 50/48/0/0; floor variant 50/49/0/42; current model 50/43/0/0. SSR 2000 current model: harness cmd 49/0/50/50,
  constant 0.6: 50/32/50/50. SSR 1600: harness 0/0/50/0; constant 0.6: 50/15/50/50. (flat_walk/stop_stand/turn_in_place/turn_and_return)
- Speed tables: `eval/out/<track>/<ckpt>/speed*/speed_table.txt`.
- Hiking adapter: HIKING_CMD=floor variant. Harness keeps 5 failing trajectories per section and kind.

## 2026-11-12 07:55 UTC (idle)
- ACP interim pair s1_v2 4250 + s2_v1 4000 planner (`eval/data/handover/acp_s1v2_4250`, trained on x2_ultra_hipexclude.xml):
  tracker check PASS on the corrected model (1.2e-07 / 1.2e-06); planner NOT checked (no pair recording; ACP asked to add it).
  Corrected model (main): flat walk 50, stop_stand 50, turn_in_place 46 (4 stalls), turn_and_return 48 (2 stalls) of 50:
  flat gate NOT passed. L1: log 50, wedge 50, stairs_up 49, stairs_down 40 (10 stalls), corridor 50. Speed 0.42 m/s at any command.
  Our apparatus first look, 50 trials (seeds 1000-1049, goal_r 0.20 + lookahead 0.30): 17/50 = 34 % (22.4-47.9), clean 0;
  25 falls, 8 stalls; 10 of the failures are at the end (7 with 22/22 waypoints fall in the 1 s end stand near x 9.7-10.0).
- ACP 3500 step 2 (current model, no goal variant): street 50, doorway 3, corridor_walls 46, corridor_rails 39, wedge_field 36,
  stairs_U 49, amphitheatre 0 (46 stalls at 1.7/13 waypoints), home 21 (27 stalls).
- Heartbeat: `bash eval/tools/heartbeat.sh STATE RUNNING RESULTS` at every job start and end.
- Results copied to /mnt/filesystem-g4/eval/ (acp/pair_s2_v1_003500, pair_s1_v2_004250, hiking, ssr, x2_model_corrected, raster_check, selfcontact).
- Open: SSR asked whether its r6b depends on the hip hulls (answered: yes, falls on the corrected model); button 5 landing-edge
  question with the coordinator; ACP 3500 corrected-model apparatus side number not run (flat 0/50 already).

## 2026-11-12 08:06 UTC (new eval-monitor session)
- ACP pair `s1_v2` 4500 (`eval/data/handover/acp_s1v2_4500`, copied from `tracks/acp/eval/pair_s1_v2_004500`, exported 07:57):
  tracker check PASS on the corrected model (history 1.2e-07, actor input 1.2e-07, action 1.2e-06); planner NOT checked: again no
  pair recording (`mlp/obs_check_pair.npz`). Planner files are byte-identical to the 4250 pair (md5 29319fb9...): it is the kept
  s2_v1 4000 planner, although the card says "trained beside tracker iteration 4500" (card wording wrong).
- Hiking v4 `model_1500` (`eval/data/handover/hiking_v4_1500`, run params from the 06:02 continuation; only the air-time reward
  differs from 1000's env.yaml): check PASS (onnx 6e-08, proprio 0, depth 1.2e-07) against the port copy.
- New `eval/turn_table.py`: constant (0, 0, wz) for 8 s from a stand; achieved yaw rate, drift, falls (Hiking: HIKING_CMD=standard).
- Running (scratchpad `job4500_hk1500.sh`, 6 single-core jobs, started 08:07, all on the corrected model X2_EXCLUDE_PAIRS=hip):
  ACP 4500 flat x4 + L1 x5 ladder, speed table, apparatus 50 (goal 0.20 + lookahead 0.30, `ours_nominal_nohip_goalr02_look03`);
  Hiking 1500 flat x4 card and floor, speed table, turn table. Expected done about 09:00 UTC.
- (08:06 desktop clock; the 08:06/08:07 above were ahead, the job started about 07:59.) Done so far, all corrected model:
  Hiking 1500 speed table (pass-through command): 0 / 0.2 / 0.35 / 0.5 / 0.6 / 0.8 -> 0.026 / 0.263 / 0.408 / 0.50 / 0.555 / 0.616 m/s,
  no falls; does NOT hold a stand at zero (drift 0.43 m in 8 s). 1000 stood at zero and did not walk below 0.5.
  Hiking 1500 turn table (0, 0, wz): +0.4 -> -0.089, +0.8 -> -0.008, -0.4 -> -0.152, -0.8 -> -0.147 rad/s; drifts 0.07-0.66 m: no turn in place.
  Hiking 1500 floor variant flat: walk 50, stop_stand 8, turn_in_place 0, turn_and_return 0 of 50 (1000: 50/49/0/42); failures
  "left the course": reaches the station at 4 s, coasts about 1 m past it under a zero command, then the corrective (0, 0, 1)
  command makes it walk a circle instead of turning (traj 1000.npz).
  ACP 4500 speed table: 0.342 m/s at every command, holds a stand at zero (4250: 0.42).
- Running since 08:02: ACP 4500 step 2, eight routes, goal 0.20 + lookahead 0.30, corrected model (scratchpad `step2_variant.sh`,
  out `eval/out/acp/pair_s1_v2_004500/ladder_nohip_goalr02_look03`).
- ACP 4500 apparatus first look (08:15): 5/50 = 10 % (4.3-21.4), clean 0; 25 falls, 20 stalls; same seeds 4250 17/50.
  Ends F/S: 4x4s 9/2, ramps 4/13 (10 stalls heading to "ramps, front side"), wall 0/3, stairs 0/1, landing 11/1, end stand 1/0.
  Landing-edge (x > 9.4, y > 2.2) failures: 4500 3, 4250 4, 3500 17 (of 200). Fail map script: scratchpad `failmap.py`.
  Interim report sent to the coordinator 08:17.
- Hiking 1500 card command flat (08:21): walk 50, stop_stand 10 (33 stalled, 7 left the course), turn_in_place 0, turn_and_return 0 of 50
  (1000 card: 50/48/0/0). Regression in stopping at a station; never turns in place.
- ACP 4500 flat gate NOT passed: walk 50, stop_stand 50, turn_in_place 40 (10 stalls), turn_and_return 50; L1 log 50 (clean 6),
  wedge 50, stairs_up 50 (clean 0). So no 200-trial run for 4500 (rule: flat gate or stage-2 pair).
- Renders of the best corrected-body apparatus run (4250 goal + lookahead, 50 trials) in
  `eval/out/acp/pair_s1_v2_004250/ours_nominal_nohip_goalr02_look03/renders/`: 1035 full (fastest success, 67.5 s), 1006 t58-66
  (fall at the stair foot after two stair-riser hits, falls backward), 1012 t66-75 (reached button 5, fell in the 1 s end stand
  toward the rail / landing edge). Looked at 26 stills. Defects: the chase camera cuts the head off on the landing (1.12 m up);
  the "cmd vx / yaw" label line is the harness command, which the ACP adapter does not receive (it gets the goal).
- FOUND, checked against the physics: in both successes (1035, 1038) the left foot reads 5-20 kN for about 0.25 s next to
  button 5, and the robot backs up about 0.4-0.5 m; in 1035 a foot then goes past the landing's edge at x 9.74 (the drop to
  the floor; h 1.74) with both feet unloaded during the 1 s end stand, which still scored as a success. Replay shows the
  landing deck is in the physics twice: the walk raster at 1.118 m AND a box `wall_202` (centre (10.36, 1.22, 1.09), half
  size 0.61 x 1.22 x 0.03, top 1.118 m) at the same height; the sole spheres sink 1-2.5 cm and touch both, with many
  horizontal-normal height-field contacts (pinching). This is the only doubled walkable surface (all wall boxes checked).
  Share of frames with a foot over 2 kN: landing 4.1 %, elsewhere 0.6 % (4250). Not yet tested whether removing the box from
  contact changes the landing falls (held-out apparatus physics: coordinator's call).
- ACP 4500 step 2 (goal + lookahead, corrected model) so far: amphitheatre 1, home 28, stairs_U 28, street_doorway 45,
  wedge_field 48 of 50; corridors and street running.
- ACP 4500 L1 (08:31): stairs_down 34/50 (14 stalls; 4250 40), corridor 50; turn_in_place stalls stand at the start (0 of 2 waypoints).
- 08:40 coordinator: Hiking restarted at 1,750 (turn in place) and 08:32 from 1,875 (10 % exact stands); score `model_2125` and
  `model_2250` (about 09:10 / 09:30), not 2,000: flat x4 card + floor, speed table, turn table, stop test. Hiking's single-seed
  probe read 0.00 m/s at 0.6 for 1500; re-checked here per seed and second: 0.55-0.56 m/s on the corrected model from the first
  second (4.10-4.14 m in 8 s), 0.27-0.41 on the old model; asked for the probe's set-up (scratchpad `hk_probe.py`).
- New `eval/stop_test.py` (walk 4 s at 0.6, zero command 3 s). Hiking 1500: 0.56 m/s before, never stops: 0.51-0.54 m
  after the zero command, still 0.14-0.16 m in the last 2 s (3 seeds).
- Running: 4250 apparatus TEST variant `_nodeckbox` (APP_NOCONTACT_WALLS=202, new in `x2.py`); ACP 4500 step 2 chain 1.
- ACP 4500 step 2, ALL EIGHT routes with goal 0.20 + lookahead 0.30, corrected model, 50 each (08:37): street 50, street_doorway 45,
  corridor_walls 40, corridor_rails 47, wedge_field 48, stairs_U 28 (21 falls), amphitheatre 1 (34 falls, 8 stalls, 7 left),
  home 28 (16 stalls). `eval/out/acp/pair_s1_v2_004500/ladder_nohip_goalr02_look03/step2.txt`. Running: plain-goal amphitheatre
  and home for the same pair (`ladder_nohip_plaingoal`) to separate the goal handling.
- TEST 4250 with the landing deck box out of contact (`_nodeckbox`, same seeds): 13/50 (15.9-39.6) against 17/50. Identical up to the
  landing (31 of 50 trials the same); 19 differ, all at the end: 8 successes became end-stand falls and 5 falls became successes.
  So the doubled deck is not the cause; the end stand at button 5 is a coin flip either way. The box stays in contact (standard).
- WHY the end stand fails (traj of 1012, 1018, 1021, 1034, 1039 and the successes 1035, 1038): after arriving at button 5
  (10.41, 2.11, facing 90 deg) the robot does not stop: it keeps turning left (yaw 1.6 -> 1.9-2.9 rad) and walks 0.4-0.7 m in -x,
  to x 9.7-10.0, toward the landing's west edge at x 9.74 (floor 0 beyond; post wall_59 at (9.63, 1.79)), and tips there
  (tilt 60+ deg) or, if the 1 s ends first, is scored a success (1035 was airborne at the end). Not the y = 2.42 edge.
- 08:52 coordinator decision: STRICT STAND rule (docs/EVAL.md "Strict stand"), main number from now on; disc-only beside it.
  Implemented 09:05: `STAND_RULE=strict` (harness default) advances a station only on 1 s with feet >= 50 % of weight and
  pelvis < 0.15 m/s inside the disc; end stand the same plus upright 2 s after; `STAND_RULE=disc` = old rule, with the strict
  check on the counted window. Per-station travel in the 1 s after counting in every summary/trials.csv. report.py and the
  ladder tables print the strict number first and each run's rule. Smoke test (ACP 4250, 3 trials): flat_stop_stand 2/3 strict.
  The button-5 y-edge question is withdrawn; the doubled landing deck stays, labelled.
- Running (scratchpad `strict4250.sh`, since ~09:06): 4250 apparatus 50 seeds STRICT (`ours_nominal_nohip_goalr02_look03_strict`)
  and DISC re-run (`_disc2`, must reproduce 17/50); 4500 plain-goal home step 2 (disc) re-run (the first attempt was stopped when
  the harness changed under it). 4500 plain-goal amphitheatre (disc): 1/50 (34 falls, 10 stalls, 5 left) vs 1/50 with the variant.
- RESULT (09:01) ACP 4250 on our apparatus, 50 seeds, goal 0.20 + lookahead 0.30, corrected model:
  STRICT STAND run: 0/50 (0-7.1 %). 42 falls, 8 stalls; 30 of the falls are on the landing heading to button 5 (never a strict
  1 s stand there; falls 3-11 s after arriving, most at x 9.74-9.95 = the west edge), stairs 5, 4x4s 3 + door side 2, ramps 2.
  Stations 1-4 are reached under the strict rule 0.4-0.6 s later than under the disc rule (it does stand there).
  DISC re-run (new code): 17/50, identical to the 08:00 run (reproducible); strict check on its counted windows: 0 of 17
  (all fail already at button 1: moving 17, feet under 50 % 9 of them). Travel in the 1 s after button 5: median 0.58 m, max
  0.70 m, 24 of 24 over 0.2 m; buttons 1-4 median 0.08-0.14 m.
- 4500 plain goal (disc): home 26/50 (14 stalls, 10 falls; with the variant 28/50, 16 stalls), amphitheatre 1/50 (variant 1/50).
  Goal handling makes no difference on these two routes for 4500.
- 09:07 coordinator: (1) separate adapter / edge in scan / policy for the non-stop at button 5; (2) strict + disc, 50 seeds, for
  the 2 November baseline (old model, goal shrink) and the 3500 pair (old model, goal + lookahead); (3) queue as before.
- (1) goal trace (scratchpad `goaltrace.py`, seed 1012, 4250): during the button-5 hold the adapter sends the right thing:
  target button 5, station heading 1.57, dpsi -0.04 at arrival (turn right a little), goal shrinking to the disc. The robot
  turns LEFT against it (yaw 1.61 -> 2.34 while dpsi goes -0.04 -> -0.77) and walks out. Same at button 4: dpsi +0.34 left
  standing (it does not correct heading but it stands). Tracker scan at the button-5 hold start: the 1.12 m drop (-1.76 to
  -1.87 relative to the pelvis) fills the two far rows, 0.5-0.6 m ahead (the y 2.44 edge); never seen at buttons 1-4.
  HARNESS DEFECT FOUND AND FIXED (09:30): after the last waypoint, `command()` returned the pelvis' own position as the target
  with no heading, so the ACP adapter sent goal (0, 0) and dpsi = wrap(0 - yaw) (noisy 0.4-1.7 rad) for the whole end stand
  (and the strict 2 s after). Now the target stays the last waypoint with its station heading. Affects every disc-rule end
  stand before 09:30 (the 7 end-stand falls of 4250 included); strict-rule runs reach it only after button 5 counts (never for 4250).
- Restarted (09:35) with the fix: baseline strict50 / disc50, 3500 strict50 / disc50 (old model), and two TEST ROUTE copies
  (scratchpad `app_b5mid`, `app_b5approach`; NOT the held-out route) for 4250 strict: button 5 moved to (10.35, 1.30) on the
  landing (0.61 m from the x edges, 1.14 m from the y edge: outside both scans), and button 5 heading 83 deg (approach) not 90.
- Baseline pair DISC re-run on seeds 1000-1049 with today's code (09:13): 2/50 (strict check 0/2). The 2026-11-11 run's same seeds had 6/50; only 5 of 50 outcomes match, diverging from the first waypoints: the baseline does not reproduce across the code changes since 11-11 (harness / adapter fixes, e.g. body velocity frame), unlike 4250 which reproduced exactly within today's code. Today's number is the comparison.
- END-OF-ROUTE BUG, scope (09:25): every result made before 09:30 (apparatus, ladder steps 1 and 2, both rules) ran its end stand
  with the bug (goal (0, 0), dpsi = wrap(-yaw) once the last waypoint had counted). It touches only the time AFTER the last
  waypoint counted: disc-rule end-stand falls (4250: 7 of 33 failures; 3500: 6 of 149) may change; strict-rule runs reach it only
  after button 5 counts under the strict rule, which never happened for 4250 (so its strict 0/50 is unaffected). It does NOT
  explain the button-5 non-stop: in the trace (seed 1012, both rules) the robot turns left DURING the button-5 hold, while the
  adapter sends target button 5, heading 1.57, dpsi -0.04 -> -0.77 (asking for a right turn). Home-route stalls are mid-route
  (waypoints 2 and 14 of 16) and its falls happen before the last waypoint: not this bug.
- Process rule (coordinator, absolute): never stop a process not started by me in this session and identifiable by its command line.

### Which configuration each baseline number belongs to (09:45; coordinator asked 09:24)
All: the 2 November pair (acp3_v2 ckpt 1000 + planner_v2_hist1), plant vendor_20260823, the old model (x2_ultra.xml sha256 4f6a4f5b,
`ssr-x2/assets/robot`, file unchanged since 2026-11-08), our apparatus data unchanged since 2026-11-11 20:55 UTC, goal "shrink"
(ACP_GOAL=shrink then and the default now), nominal protocol, MuJoCo 3.15 CPU.
- 15/200 (and its first 50 seeds: 6/50), and 4/200 (ACP_GOAL=keep): run on 2026-11-11 about 21:30-22:31 UTC with an adapter version
  that was never committed: the first commit holding `acp_baseline.py` (406f0d2, 22:16 UTC) already has the scan floor clamp found by
  the step-3 check (finished 22:37 UTC). Re-running seeds 1000-1004: commit 5dc37cc code gives 8 / 4 / 4 / 4 / 6 waypoints (all
  falls) = today's code exactly; 5dc37cc with the floor clamp removed gives 4 / 9 / 4 / 6 / 8 (stall); the 11-11 run had 6 / 4 / 6 /
  8 / success. So that run is not reproducible with any code on record. Proposal: retire 15/200, 4/200 and 6/50 as "pre-check
  adapter, not reproducible".
- 2/50 disc and 0/50 strict (seeds 1000-1049, 09:13 UTC): today's code (adapter as checked at 11-11 22:37, end-of-route fix
  included); these are the baseline numbers to quote, with "today's checked adapter".
- The 3500 pair reproduces exactly: disc re-run 16/50 on seeds 1000-1049, 50 of 50 outcomes identical to its 07:22 run (51/200).
- New regression check `eval/regression.py` (reference `eval/config/regression_ref.json`, recorded 09:20 UTC): ten 12 s trials
  (ACP 4250 corrected model x6, baseline old model x4), run at the start of every harness run (no partial.json) and once per ladder
  run folder; result in the run's `regression.json` and summary `regression`. EVAL_REGRESSION=0 skips it. Re-record only after a
  deliberate, documented change.
- 3500 STRICT 0/50: 43 falls (landing 21, 4x4s 9, stairs 7, ramps 6), 6 stalls, 1 "not a stand" (reached the end, end stand
  failed). DISC 16/50: all 16 fail the strict check already at button 1 (moving).
- BUTTON-5 TEST (4250, strict, 50 seeds, the same up to the landing: 30 reach it in each): original route 0/50 (0 of 30 stand at
  button 5); TEST button 5 moved to the landing middle (10.35, 1.30) 30/50 = 60 % (46.2-72.4), 30 of 30 stand, travel after it
  median 0.01 m; TEST heading 83 deg (approach) 1/50 (2 of 30 counted). So it is not the adapter and not the station heading: it is
  the position by the edge. Running: DIAGNOSTIC with drops deeper than 0.15 m hidden from both scans on the landing only
  (DIAG_SCAN_DROP_CLAMP=0.15, x2.model_note labels it), original route, to separate "sees the drop" from "is near the edge".
- SSR r7 500 obs check: PASS on everything the adapter computes; the depth "FAIL" is the recording's scene (22 % of rays hit
  something 3.4-3.9 m ahead in the training scene that the check's flat plane lacks); near rays agree to 3.2e-4. Flat tests running.
- Hiking 2125: check PASS; flat tests running.
- 4250 DISC with the end-of-route fix (09:47): 20/50 (27.6-53.8) against 17/50 before; end-stand falls 7 -> 4; everything before the end identical; travel after button 5 unchanged (median 0.57 m, 24 of 24 over 0.2 m: it is already moving when button 5 counts). Strict check on its counted windows 0/20.
- HIKING model_2125 (09:35; corrected model, STRICT rule; check PASS; regression OK). Flat (walk / stop-stand / turn in place /
  turn and return, of 50): card 50 / 6 / 19 / 50; floor variant 50 / 6 / 14 / 50 (1500: card 50/10/0/0, floor 50/8/0/0).
  Speed (pass-through): 0 / 0.2 / 0.35 / 0.5 / 0.6 / 0.8 -> -0.002 / 0.14 / 0.30 / 0.44 / 0.51 / 0.62 m/s, holds a stand at zero.
  Turn (0, 0, wz): +0.4 -> 0.48, +0.8 -> 0.88, -0.4 -> -0.52, -0.8 -> -0.87 rad/s; drift 0.07-0.34 m (in place at |0.8|): it now
  turns in place, the right way. Stop test: 0.53 m/s before, 0.34-0.35 m after the zero command, then 0.05-0.06 m in the last 2 s
  (about 0.03 m/s: slow enough for the strict 0.15 m/s).
  Stop-stand failures (trace 1000.npz): the first station counts; at the second it coasts 0.2-0.3 m past the 0.20 m disc under the
  zero command, then the card's corrective (0, 0, 1) command makes it circle about 0.3 m round the station for 40 s (stalled).
- DIAGNOSTIC RESULT (09:58): original route, drops deeper than 0.15 m hidden from both height scans on the landing only:
  30/50 strict (46.2-72.4), 30 of 30 arrivals stand at the real button 5, travel after it median 0.01 m (max 0.02). Same as moving
  the station to the landing middle. So the non-stop at button 5 is the policy reacting to the drop IN ITS SCANS (a 1.1 m drop
  never seen in training), not the adapter, not the heading, not the physical position. Not a result (labelled DIAGNOSTIC).
- HIKING COMMAND (10:15, coordinator decision after the Hiking track): HIKING_CMD=card now follows the card exactly (standard, main
  number): generator to the waypoint, stand within target_stop_radius_m 0.4 of a station (yaw only toward the station heading),
  target behind -> turn on the spot then walk, no stop at non-station waypoints. The implementation used before 10:15 is
  HIKING_CMD=card_v1 (no 0.4 m stop; harness yaw correction inside the disc; pure-yaw up to 1 rad/s after an overshoot = the
  circling). ALL Hiking "card" numbers before 10:15 (1000, 1500, 2125) are card_v1. New variant HIKING_CMD=brake (zero forward
  from 0.35 m before a station's disc, stand after an overshoot while moving, then turn on the spot <= 0.8 rad/s, creep 0.2 m/s).
  Floor variant unchanged.
- 2250: floor flat + tables running (since ~10:02); card (new) and brake flat running since 10:17; the first card_v1 chain was
  stopped (my own processes, by command line) and its folder deleted.
- Stop test timing (for the Hiking agent): 4 s at (0.6, 0, 0) from the eval's start pose, then hold zero 3 s; "travel after zero"
  over the 3 s (2125: 0.34-0.35 m), "drift last 2 s" over the final 2 s (0.05-0.06 m).
- Hiking 2250 (10:06, strict, corrected): FLOOR variant flat 50 / 46 / 30 / 50; speed 0/0.2/0.35/0.5/0.6/0.8 -> 0.007/0.16/0.31/0.45/0.53/0.64, holds stand; turn +-0.4 -> 0.43/-0.59, +-0.8 -> 0.93/-0.95 rad/s (drift 0.17-0.36 m); stop: 0.40 m over 3 s, 0.09-0.10 m in the last 2 s.
- ACP pair s1_v2 5500 (exported ~10:00; same kept 4000 planner, md5 29319fb9): tracker check PASS (input 6e-08, action 2.4e-06);
  planner NOT checked (no pair recording, third time). Its own pair_eval: oracle contacts 40/40, planner contacts 29/40.
  Running since ~10:25 (scratchpad `acp_pair.sh`): flat + L1 ladder then speed; apparatus 50 STRICT then 50 DISC.
- Queued: Hiking 2500 (copied, run params from the 08:27 run) with card / floor / brake + tables; then L1 + street/home with the best variant.
- Hiking 2250 flat (strict): card (new, standard) 50 / 45 / 25 / 50; floor 50 / 46 / 30 / 50; brake 50 / 48 / 0 / 50 (brake's in-disc yaw is the harness's 1.5 x error up to 1 rad/s: turn tests stall). Best on flat: floor -> used for L1 and street/home (started ~10:30).
- SSR r7 500 (corrected model, strict, 10:10). Flat standard command: 50 / 14 / 0 / 0; constant 0.6 m/s: 50 / 50 / 0 / 10.
  Speed 0/0.2/0.35/0.5/0.6/0.8 -> 0.000/0.001/0.198/0.465/0.557/0.714 m/s, holds stand. Turn table: 0.00 rad/s at every
  yaw command (does not turn in place at all). Stop test: stops in 1.0 s within 0.09-0.10 m, still after. Stop-stand standard
  failures: stops 0.20 m short of the first station (x -0.20): the harness command slows to 0.2-0.35 m/s near the disc, below
  which it does not walk. SSR probe agreement: 0.5/0.6/0.8 within 0.02 m/s; at 0.2 and 0.35 it reads 0.09 / 0.29, here 0.00 /
  0.20 (this table starts standing and averages the last 4 of 8 s).
- CLOCK: desktop `date -u` = node clock (both 10:19:0x at 10:19). Some times I wrote in messages/state between 09:45 and 10:35 were
  estimates and ran up to ~20 min ahead; from now on every stamp comes from `date -u`.
- ACP 5500 full pair check (pair recording now in `pair_eval_40/obs_check_pair.npz`; copied to `mlp/`): ALL PASS (pair: input
  2.4e-07, action 4.0e-06, planner occupancy 1.6e-06). The planner file is byte-identical to the 4250 / 4500 pairs' (md5 29319fb9), so
  their planner is now checked too. Rule from the coordinator: read only export folders with DONE; s2_v2 pairs every 250 from 5750
  (~11:10): pair check incl. planner, flat, L1, apparatus 50 strict + 50 disc; 200 + 200 only at >= 10/50 strict or on request;
  report button 5 (strict stands held / trials reaching it) every time.
- ACP 5500 apparatus STRICT 50 (10:21): 0/50; 24 falls (4x4s 19), 26 stalls (ramps 24); button 5: 1 reached its approach, 0 held. Travel after button 1 median 0.20 m (17 of 46 over 0.2 m). Regression OK. DISC 50 running.
- Hiking 2500 FLOOR flat (10:27): 50 / 0 / 0 / 50 (2250 floor: 50/46/30/50): it runs on 0.3-0.5 m past the station under the zero command and the floor variant's pure-yaw correction makes it circle (traj 1000). Card (new) + tables running.
- Hiking 2250 beyond flat, FLOOR variant, strict (10:30): L1 log 41 (9 not a stand at the end, clean 1), wedge 50,
  stairs up 37 (8 falls, 5 left), stairs down 50, corridor 50. Step 2: street 9/50 (35 "left the course" heading to the far
  station (2.90, -1.50): overshoot at the turn-around near the tile edge), home 34/50 (11 falls) (ACP 4500: street 50, home 28).
- ACP 5500 apparatus DISC 50 (10:32): 1/50 (4x4s 25 F, ramps 18 S); button 5: 1 reached, 1 counted (disc). Strict 0/50. Ladder so far (strict): flat 50/50/44/50, log 49, wedge 50, stairs up 35.
- Hiking 2500 CARD (new) flat (10:34): 50 / 0 / 0 / 50. Trace (stop-stand 1000): walks 0.84 m/s under a 0.7 command, runs 0.4-0.5 m past the station after the stand, then turns toward it while drifting (circling 0.3-0.5 m round it); never holds 1 s inside the disc. Regression from 2250 (card 50/45/25/50).
- Hiking 2500 tables: speed 0/0.2/0.35/0.5/0.6/0.8 -> 0.001/0.25/0.42/0.55/0.60/0.65 (overshoots slow commands), holds stand from rest; turn +0.4/+0.8/-0.4/-0.8 -> 0.54/1.05/-0.48/-0.84 rad/s, drift 0.06-0.52 m; STOP: 0.67 m/s before, 0.72 m over 3 s, 0.26 m in the last 2 s (2250: 0.40 / 0.09): it no longer stops after walking.
- 10:38 coordinator: Hiking restarts at 2750 with a ground-speed penalty under stand commands; its test = stop test on model_3000 (~11:45): <= 0.30 m after zero and <= 0.05 m in the last 2 s, flat walk 50/50, speed at 0.6 within 0.45-0.65, stop-stand >= 45/50 (card). Score 3000 first when exported (stop test, tables, four flat tests card); 2250 stays the reference for obstacles/routes; s2_v2 5750 (~11:10) and the next SSR r7 export keep their place.
- ACP 5500 L1 (10:39): stairs down 12/50 (38 "not a stand": the strict end stand at the foot of the flight fails), corridor 50/50; with log 49, wedge 50, stairs up 35. Flat gate NOT passed (turn in place 44).
- Hiking 2625 (10:51): check PASS; FLOOR flat 50 / 0 / 0 / 50 (same stopping regression as 2500). Card + tables running.
- Hiking 2625 card flat 50 / 0 / 7 / 50; tables: speed 0.00->0.0 0.20->0.236 0.35->0.383 0.50->0.496 0.60->0.559 0.80->0.65 ; stop 0.537 m (last 2 s 0.145) 0.533 m (last 2 s 0.141) 0.535 m (last 2 s 0.144) 
- 11:16 ACP pair_s2_v2_005750 (first stage 2 on the corrected body; NEW planner md5 f4da33e8; card stage field says 1): full check ALL PASS (pair: input 1.2e-07, action 3.9e-06, occupancy 2.4e-06). Own pair eval: oracle 44/44, planner 38/44. Running: flat + L1 + speed; apparatus 50 strict and 50 disc in parallel (scratchpad acp_pair_par.sh).
- Hiking 3000 (11:36): check PASS. STOP TEST: 0.55 m/s before, stops in 1.34 s, 0.35 m travelled after the zero command (mark <= 0.30: MISSED by 0.05), 0.027-0.028 m in the last 2 s (mark <= 0.05: met). Flat card/floor + tables running.
- CORRECTION (coordinator 11:36): model_3000 was trained WITHOUT the ground-speed penalty (v4e failed its own check and the run fell back to v4d from model_2750 at 10:52); the stop improvement 0.72 (2500) -> 0.54 (2625) -> 0.35 m (3000) is NOT the penalty. The penalty is retried at about 3000-3125; its test = my stop test on the second checkpoint after that restart.
- ACP s2_v2 5750 ladder (11:44, strict): flat 50 / 50 / 48 / 50 (gate NOT passed); L1 log 22, wedge 50, stairs up 10, stairs down 22, corridor 49. The log / stairs failures are almost all the strict END STAND (log 28, stairs up 36, stairs down 27): the last station is held strictly (travel after it 0.05-0.11 m), then within the next 1 s the pelvis peaks at 0.19-0.28 m/s (a re-balancing step). The end stand has no retry, so in effect it asks for 2 s of continuous standing.
- ACP s2_v2 5750 OUR APPARATUS first look (11:45), seeds 1000-1049, corrected model, goal + lookahead:
  STRICT 21/50 = 42 % (29.4-55.8), clean 0; 23 falls (stairs 19, 4x4s 3, ramps 1), 5 stalls, 1 not a stand.
  BUTTON 5: 22 reached, 22 held a strict stand (the new stands by a 1.1 m drop work). DISC 29/50 = 58 % (44.2-70.6), strict check
  on its windows 0/29. Speed 0.44 m/s at any command. -> >= 10/50 strict: 200 + 200 started (seeds 1050-1199 in four chunks,
  scratchpad `acp200.sh`; merge with `merge200.py`).
- 11:50 coordinator decision: END STAND WITH RETRY (docs/EVAL.md). Implemented behind END_RETRY=1 (default 0 until the 5750 200+200
  chunks, started 11:47 without it, have finished; then flip the default). Then: count the 200's end-stand failures and re-run only
  those seeds with the retry; re-run the 5750 L1 tests with the retry; 200 report with strict + disc, fall/stall map, button 5,
  where on the stairs the falls happen, a render of a typical stairs fall and of the best run.
- 5750 stairs falls (strict 50): flight risers at x 8.65 / 8.85 / 9.05 / 9.25 / 9.55 / 9.75 (0.19 m each); pelvis ends over: floor before the flight 2, first tread 10, second tread 3, third tread 4, upper treads / top step / landing turn 0. So 15 of 19 at the foot of the flight or on the first two steps.
- Hiking 3000 (11:53), strict: flat card 50 / 50 / 18 / 50, floor 50 / 50 / 33 / 50 (2250: card 45 / 25, floor 46 / 30 for stop-stand / turn); speed 0.2/0.35/0.5/0.6/0.8 -> 0.20/0.34/0.48/0.56/0.67 (0.6 within 0.45-0.65), holds stand; turn +-0.4 -> 0.43/-0.49, +-0.8 -> 0.96/-0.92 rad/s (2250 alike); stop 0.35 m / 0.027 m. -> made Hiking's reference checkpoint (turn in place: card worse than 2250, floor better, turn rates equal). Full look started (L1 + street/home, then apparatus 50 strict + 50 disc; card, end-stand retry), one core at a time beside the ACP 200 run.
- Hiking 3125 (12:04): stop test 0.68 m/s before, 0.464-0.469 m after the zero command (OVER the track's 0.45 m watch mark), 0.055-0.057 m in the last 2 s; speed 0.2/0.35/0.5/0.6/0.8 -> 0.30/0.47/0.58/0.64/0.71 (0.64 at 0.6: under the 0.70 mark, but over-speeding at slow commands).
- ACP s2_v2 5750, OUR APPARATUS 200 + 200 (12:06; seeds 1000-1199; merged in `ours_nominal_nohip_goalr02_look03_{strict,disc}_200`):
  STRICT (end stand without retry, as started) 105/200 = 52.5 % (45.6-59.3), clean 0; falls 72, stalls 14, not a stand 9 (all end stand).
  Map F/S: 4x4s 21/5, ramps 4/8, stairs 45/0, landing 2/1, end stand 9. Button 5: 117 reached its approach, 114 held a strict stand
  (travel after it median 0.02 m, max 0.12). Successes 62-117 s (median 71). DISC 113/200 = 56.5 % (49.6-63.2); map 4x4s 21/2, ramps
  3/9, stairs 51/1. Stairs falls (strict 200) by where the pelvis ends: floor before the flight 11, 1st tread 23, 2nd 5, 3rd 4, 4th 2,
  top / landing turn 0. END_RETRY default flipped to 1 at 12:06. The 9 end-stand seeds are being re-run with the retry.
- ACP s2_v2 6000 (DONE; new planner md5 2da8ec5e): full check ALL PASS; own pair eval oracle 44/44, planner 36/44. Running (with retry):
  flat + L1 + speed, apparatus 50 strict + 50 disc.
- 12:06 coordinator: Hiking restarts at 3250 with a penalty on forward speed above the command. Test = model_3500 (~13:15): speed within +-15 % at 0.2-0.6; stop test <= 0.40 m and <= 0.05 m in the last 2 s; flat walk 50/50, stop-stand >= 45/50; turn in place >= 3000's (card 18, floor 33; rates >= 0.4 rad/s); a held stand at zero. Skip 3250 / 3375 unless idle; 3000 stays the reference.
- Renders 5750 (12:10), looked at 12 stills: 1044 best (66 s, stands at button 5 with both feet loaded 129 / 134 N at 69 s); 1003 typical stairs fall: arrives at the stair foot still turning (yaw command -0.33 -> +1.0 rad/s), steps onto the 1st tread with the right foot only, the left foot then hits the 1st riser (452 N), the left knee the stair side, and it falls sideways at 56.98 s (trunk on the ground), matching trials.csv. Label defect: 'iteration 5750 (stage 1)' comes from the card's wrong stage field.
- 5750 end-stand retry re-run of the 9 end-stand seeds (12:10): 9 of 9 succeed -> STRICT WITH RETRY 114/200 = 57.0 % (Wilson in the report); everything else in the 200 identical (the retry only changes what happens after the last waypoint).
- SSR r7 2000 (12:12): check PASS (frame 2.2e-06, depth 3.2e-04, action 0, GRU 7e-07, targets 1.5e-06). Queued: flat std + keep, speed / turn / stop (strict, retry).
- ACP 5750 L1 WITH END-STAND RETRY (12:15): log 50 (clean 16), wedge 50, stairs up 41 (5 not a stand, stalls), stairs down 48 (1 fall), corridor 50 -- against 22 / 50 / 10 / 22 / 49 without the retry: the earlier low numbers were the end stand, not the obstacle. Weakest: stairs up L1.
- ACP s2_v2 6000 apparatus (12:21, 50, retry on): STRICT 18/50, DISC 16/50 (5750: 21 / 29 on the same seeds).
- 6000 strict 18/50 (24.1-49.9): 18 F... map 4x4s 8 F / 11 S, ramps 1/7, stairs 5/0; button 5 18/18 held. >= 10/50 -> 200 + 200 queued after SSR r7 2000.
- Hiking 3000 full look (12:33; card, strict, end-stand retry): L1 log 50 (clean 33), wedge 50, stairs up 49, stairs down 50, corridor 50 -> every L1 skill passed. Step 2: street 37/50 (11 stalled at 'far end: stand 1 s, then turn back': the turn-around station), home 48/50 (2 falls). (2250 floor: street 9, home 34; ACP 4500 variant: street 50, home 28.) Apparatus 50 strict + 50 disc running.
- ACP 6000 ladder (12:35, strict, retry): flat_walk 50, flat_stop_stand 50, flat_turn_in_place 50, flat_turn_and_return 49, log_L1 49, wedge_L1 47, stairs_up_L1 48, stairs_down_L1 50, corridor_L1 50,  , 
- SSR r7 2000 (12:48, corrected, strict + retry): keep 0.6 variant flat 50 / 50 / 14 / 50; speed 0.2/0.35/0.5/0.6/0.8 -> 0.00/0.31/0.51/0.61/0.82 (excellent from 0.35 up; still nothing at 0.2), holds stand; turn table 0.01-0.02 rad/s at every yaw command (still no turn in place); stop 0.11 m in 0.4-0.6 s. Standard flat running.
- SSR r7 2000 standard flat stopped at 12:53 (replaced by 2250, coordinator): walk 50/50, stop-stand 0/50 done; turn tests not run.
- SSR r7 2250 (12:53): check PASS in substance (one pixel in one frame differs: a limb-sphere edge). Queued after ACP 6000 200+200: flat std + keep, tables, then (if walk and turn-and-return pass) L1, street/home, apparatus 50 strict + 50 disc.
- ACP s2_v2 6000 OUR APPARATUS 200 + 200 (13:13, retry on in every chunk): STRICT 79/200 = 39.5 % (33.0-46.4), DISC 65/200 = 32.5 %
  (26.4-39.3). Strict map F/S: 4x4s 27/34, ramps 3/21, stairs 34/1, landing 1/0. Button 5: 80 reached, 79 held. Successes 64-127 s
  (median 79). Significantly below 5750 (114/200, 50.1-63.7): fewer stairs falls (34 vs 45) but 56 stalls (5750: 14), at the 4x4s and ramps.
- SSR r7 2250 tables (13:13): speed 0.2/0.35/0.5/0.6/0.8 -> 0.03/0.36/0.54/0.62/0.77 (probe 0.13/0.38/0.52/0.59/0.73: agrees from 0.35 up), holds stand; TURN +0.4/+0.8/-0.4/-0.8 -> 0.03/0.72/-0.24/-0.69 rad/s in place (probe 0.14/0.70/-0.25/-0.67: agrees; 2000 did not turn at all); stop 0.16 m in 0.7 s. Flat std + keep running.
- SSR r7 2250 STANDARD flat (13:27, strict + retry): 50 / 50 / 50 / 50 -> FLAT GATE PASSED, the first policy of any track to pass it under the strict rule. Beyond-flat look started (L1, street/home, apparatus 50 strict + 50 disc; standard command).
- Hiking 3000 OUR APPARATUS STRICT 50 (13:28, card, retry): 0/50; 29 falls, 21 stalls; map door 3 F, 4x4s 23 F / 16 S, ramps 3 F / 5 S; by target: button 1 14 stalls + 8 falls, door side 7 falls, over the front 4x4 6 falls, button 3 5 stalls. Furthest: button 3 reached by 3 trials; last station never reached (0). DISC running.
- SSR 2250 keep 0.6 variant flat: 50 50 41 50 
- ACP s2_v2 6250 (13:29): check ALL PASS (planner md5 e8ee9b29; own eval planner 38/44). Running (2 cores): ladder + speed; apparatus 50 strict then 50 disc (retry on).
- ACP s2_v2 6250 (13:57): FLAT GATE PASSED (50/50 x4, strict + retry; first ACP pair to pass), L1 log 46, wedge 50, stairs up 48,
  stairs down 47, corridor 50; speed 0.46 m/s at any command. Apparatus 50: STRICT 18/50 = 36 % (24.1-49.9; 4x4s 10 F / 11 S, ramps 2/5,
  stairs 4/0; button 5 18/18 held), DISC 21/50. >= 10/50 -> 200 + 200 queued after Hiking 3500.
- SSR 2250 L1 (14:00, std, strict + retry): log_L1 50, wedge_L1 50, stairs_up_L1 50, stairs_down_L1 50, corridor_L1 0, 
  corridor L1 0/50 is NOT the corridor: all 50 stop at x 2.83, 0.27 m short of the final station (3.1, 0) -- near the disc the harness command drops to about 0.27 m/s, below which SSR 2250 does not walk (speed table 0.2 -> 0.03). Same mechanism would hit any station approached in a straight line; the keep-0.6 variant avoids it.
- 14:02 coordinator: ACP diagnostics on one core: (1) stall traces (scratchpad stalltrace.py; 6000 seeds 1001/1005/1036 front-4x4 stalls + 5750 same seeds; ramps 6000 1019/1022 + 5750 1019): goal, planner targets/occupancy, scans; (2) TEST route copy app_stairfoot (waypoint 18 -> 1 s station at (8.35, 0.533) facing up the flight), 5750's 45 stair-fall seeds. Queued behind the running jobs (waits for a free core).
- HIKING 3500 TEST (14:16; corrected, strict + retry; check PASS): FAILS.
  speed 0.2/0.35/0.5/0.6/0.8 -> 0.33/0.49/0.59/0.63/0.68 (+66 / +41 / +18 / +6 %: mark +-15 % at 0.2-0.6 FAILED except 0.6);
  stop 0.399-0.404 m (mark <= 0.40: borderline, 1 of 3 over), last 2 s 0.004-0.010 m (met); flat card 50 / 50 / 2 / 50, floor 50 / 5 / 0 / 50
  (turn in place mark: card >= 18, floor >= 33: FAILED); turn rates +-0.4 -> 0.45 / -0.36 (mark >= 0.4: -0.4 FAILED), drift 0.42-0.66 m;
  stand at zero held (met).
- ACP STALL TRACES (scratchpad `st_*.npz`, reproduced exactly):
  FRONT 4x4, 6000 seeds 1001 / 1005 / 1036 (all stall 45 s heading to wp5 "square to the front 4x4" (3.302, 0.686), 0.25 m east of
  the beam centre line, i.e. the same side as button 1, 0.36 m north of it): after button 1 (heading -90) the robot stands at
  (3.33-3.42, 0.28-0.34), facing south (yaw -1.2 to -1.9), 0.36-0.42 m from wp5 (outside the 0.20 m disc), with the target directly
  BEHIND: goal (-0.95, +-0.3) at full length (lookahead puts the target 0.3 m further back), dpsi +-2.6 to 3.1 rad. The planner proposes
  no step: one foot's occupancy 0.35-0.50 (under the 0.5 validity line) and the other foot's target where that foot already stands,
  for the whole 45 s. 5750 at the same place with the same goal (seed 1001) turns left (yaw -1.57 -> 2.0 in 3.4 s) and walks on.
  => PLANNER (6000) not proposing a turn-around step for a target straight behind; the route makes it the first move after button 1.
  RAMPS, 6000 seeds 1019 / 1022 (stall heading to wp15 "ramps, front side" (5.44, 0.61)): the robot faces the target (dpsi 0.1-0.25,
  goal straight ahead), stands at (4.67-4.73, 0.18-0.19) on the 0.12 m cell top 0.1 m before its east edge (4.8 -> 4.9: 0.12 -> 0.01 m,
  an 11 cm step down into the trough before the next ramp); the planner proposes valid steps (occupancy 0.6-0.98) 0.1-0.4 m ahead at
  x 4.8-5.1, z 0.01-0.10, but the tracker does not take them for 45 s. 5750 (seed 1019) takes the same proposed steps.
  => TRACKER (6000) not stepping down an 11 cm edge near the lane side.
- SSR corridor stop confirmed (14:21, traj 1000): stands at x 2.828 from t 11 s to the stall, 0.27 m short of (3.1, 0); the harness command there is (0.273, 0, ~0) m/s for the whole 45 s. NEW SSR_CMD=floor40 in eval/adapters/ssr.py (spec from the coordinator: forward max(min(d, vmax), 0.40) x cos(err) outside the disc under 60 deg, harness inside the disc, turns >= 0.6 rad/s over 60 deg; vy-keeping waypoints unchanged). Standard stays SSR's headline; both reported, labelled.
- 14:21 queue: SSR 2250 floor40 (flat x4 + corridor L1, street/home, apparatus 50 strict + 50 disc) after ACP 6250 200; Hiking model_4125 marks: turn in place >= 33/50 floor, turning drift <= 0.3 m, stop <= 0.40 m, stop-stand >= 45/50, walk 50/50, speed table 0.2-0.8 (sliding penalty at zero forward command from 3875); ACP pair_s2_v2_006500 when DONE. Watcher on both.
- ACP s2_v2 6500 (14:22): check ALL PASS; queued (ladder + apparatus 50 strict / 50 disc).
- 14:23 one serial queue (scratchpad master_queue.sh, log master_queue.log): ACP 6250 200+200 (merge) -> ACP 6500 (ladder + 50/50) -> Hiking 4125 if copied by then -> SSR 2250 floor40. Running beside it: SSR 2250 standard routes + apparatus, Hiking 3000 apparatus disc, ACP stair-foot TEST.
- 14:23 heartbeat refreshed (it was 2 h stale: my fault, refreshed only at job starts I scripted). Desktop and node clocks agree to 2 s; message stamps from now on are taken from date -u at sending, not estimated.
- Hiking 3000 apparatus DISC (14:30): 0/50, outcomes {'fall': 27, 'stalled': 23};    door 3 F / 0 S, 4x4s 21 F / 21 S, ramps 2 F / 2 S, wall 1 F / 0 S, stairs 0 F / 0 S, landing 0 F / 0 S, end stand 0 F / 0 S
- STAIR-FOOT TEST (14:31; TEST copy app_stairfoot, NOT a result; 5750's 45 strict stair-fall seeds, strict + retry): 12 succeed,
  1 stall, 32 falls: 29 fall while still heading to the new foot station (8.35, 0.533) (pelvis over the floor 15, over the 1st tread 10,
  2nd 4), 2 fall going up after standing square, 1 on the landing. Of the 15 that stood square at the foot facing up, 13 climbed the flight.
- 14:31 ACP diagnostics reported to the coordinator (front 4x4 = planner on a target-behind goal; ramps = tracker on an 11 cm step down; stairs = approach / turn at the foot).
- 14:41 USER decisions (via coordinator): strict stand rule and final-stand retry CONFIRMED ("fine to take corrective steps, but the
  1 s should count from after the step"); hip contact confirmed false on the real robot. docs/EVAL.md updated by the coordinator.
- ACP reference pair frozen: pair_s2_v2_005750 (114/200 strict). For every new ACP pair, before its 200+200: regression line on seed sets
  turn-around (1001, 1005, 1036), ramp trough (1019, 1022) and the 45 stair-fall seeds, beside 5750's outcomes.
- NEW ACP_GOAL=turnwalk in eval/adapters/acp.py (ACP track's spec): bearing > 1.5 rad and waypoint within 1.0 m -> no lookahead, goal at
  the bearing clipped to +-1 rad (unit length = a 0.3 m goal over goal_r), dpsi clipped to +-1 rad, until the bearing is under 1.0 rad.
  Default path refactored only (regression 10/10 bit-identical after the change). First check: 6000 and 5750 on 1001 / 1005 / 1036.
- TURNWALK first check (14:42, strict + retry, single seeds): 6000: 1001 success, 1005 fall at 6/22 (past the turn-around, heading
  back over the front 4x4), 1036 success -> 6000 turns and walks on in 3 of 3 (standard: 3 stalls). 5750: 1001 success, 1005 fall at 7/22,
  1036 fall at 5/22 during the turn (standard: success, success, stairs fall). So turnwalk unblocks 6000 but costs 5750 two of three.
  -> 50-trial apparatus runs under turnwalk for 5750 and the newest pair (6500) queued after the serial queue.
- SSR 2250 STANDARD routes (14:42): street 0/50 (45 stalls: 31 heading to (-1.90, -2.10), 14 to (-3.20, -2.10); 5 falls),
  home 44/50 (6 stalls at (0.38, -2.08)). Stalls to be checked against the slow-command floor (floor40 run queued).
- SSR 2250 apparatus STANDARD STRICT (14:57): 0/50; 49 stalls (40 heading to 'door side' after the turn-around over the front 4x4, 8 at button 1), 1 fall. Likely mechanism (to confirm with floor40): the harness turns on the spot with yaw 1.5 x error; under about 0.4 rad/s SSR 2250 does not turn (turn table +0.4 -> 0.03 rad/s), so the turn stalls before the error drops under 60 deg and the forward command never starts. Button 1 counted in 42 (travel after it 0.04 m).
- SSR 2250 apparatus STANDARD DISC: 0 /50 {'stalled': 47, 'fall': 3}
- 15:04 running: ACP 6250 200+200, ACP 6500, SSR 2250 floor40; Hiking 4125 not yet exported (skipped by the queue; run separately when it lands).
- ACP s2_v2 6250 200 + 200 (15:05, retry on): STRICT 72/200 = 36.0 % (29.7-42.9), DISC 96/200 = 48.0 % (41.2-54.9).
  Strict F/S: 4x4s 24/57, ramps 5/16, stairs 24/0, landing 1/1. Button 5: 74 reached, 72 held.
  REGRESSION LINE vs 5750 (strict, same seeds): turn-around 1001 / 1005 / 1036: 5750 success / success / stairs fall; 6250 ramps fall /
  success / door-side fall. Ramp trough 1019 / 1022: 5750 success / success; 6250 fall at button 1 / fall over the back hill.
  45 stair-fall seeds (5750: 0 succeed, all fell on the stairs): 6250 14 succeed, 4 fall on the stairs, the rest fail earlier.
- 15:06 SWAP TEST queued (no new exports needed: actor.npz and planner_attn.onnx swapped between the 5750 and 6250 hand-overs; folders eval/data/handover/swap_p5750_t6250, swap_p6250_t5750; md5 checked): 50 strict + retry each, seeds 1000-1049, out eval/out/acp/TEST_swap/. Starts when 2 cores free, ahead of the turnwalk 50s.
- 15:09 ACP queue change (ACP's decision): planner frozen at 5750 from 6750 on. No 200+200 for 6500 / 6750, only the regression
  seed sets (scratchpad regline.sh). Order: swap TEST -> turnwalk 50s -> pair 7000 (~15:55): turn-around seeds first (report at once), then
  regression line, then 200+200 -> reglines 6500, 6750. Scripts: swap.sh, queue2.sh, queue3.sh (logs beside them). Hiking 4125 and SSR
  floor40 keep their places.
- ACP 6500 apparatus STRICT 50 (15:13): 19 /50 {'success': 19, 'stalled': 23, 'fall': 8};    door 0 F / 0 S, 4x4s 4 F / 20 S, ramps 0 F / 3 S, wall 0 F / 0 S, stairs 4 F / 0 S, landing 0 F / 0 S, end stand 0 F / 0 S
- 15:24 Hiking 4125 exported and copied (run 20261112_143622); queued right after the swap TEST starts (check, flat card + floor, speed / turn / stop); its marks in the 14:3x entry.
- ACP 6500 apparatus DISC 50 (15:25): 18 /50 {'success': 18, 'stalled': 16, 'fall': 16}
- ACP 6500 ladder (15:26): flat_walk 50, flat_stop_stand 50, flat_turn_in_place 50, flat_turn_and_return 50, log_L1 50, wedge_L1 50, stairs_up_L1 49, stairs_down_L1 44, corridor_L1 50,  , 
- SSR 2250 STALL TRACE (15:31; traj of the standard strict run + floor40 run + a re-run of seed 1007; depth frames and a render in
  eval/out/ssr/r7_2250/ours_nominal_nohip_standard_strict/diag/ and renders/chase|legs_seed1000_t30-40*). My 14:57 mechanism (slow turns)
  was WRONG; SSR's objection holds: none of the five is turning.
  Door side, seeds 1000 / 1001 / 1002: turned (yaw 1.2-1.5, heading error -0.10 to -0.01 rad), 0.25-0.37 m from 'door side' (2.80, 1.60),
  commanded forward 0.25-0.37 m/s (the harness slows to the distance), speed 0.00-0.03 m/s for 20 s; the door frame posts (2.44-2.51, 1.70)
  are 0.40 m ahead-left. Under FLOOR40 (forward 0.40 m/s, which it walks on flat) seed 1000 still stands there (2.67, 1.30) -> (c): it will
  not walk toward the door frame; (b) adds to it where the command is under 0.3 m/s.
  Button 1, seed 1007 (re-run): facing button 1 (error -0.14), 0.21 m from it (1 cm outside the 0.20 m disc), commanded 0.21 m/s -> (b).
  Button 1, seed 1005: had wandered to (4.42, 0.15) on the ramp edge beside the posts (3.61-3.70, -0.04), facing the target (error -0.34),
  commanded 0.57 m/s and yaw -0.5, does not move -> (c). Not (a) in any; not (d) (the harness command is as designed; it slows near waypoints).
- RASTER CHECK around the door side (15:33; x 2.3-3.3, y 1.0-2.0, walk raster = physics height field and the scans / depth):
  46 cells stand > 5 cm above their 5 x 5 median, in 7 patches. 35 cells (2 patches) are one object at x 2.51-2.74, y 1.78-2.00: a solid
  wedge rising along x from 0.04 to 0.48 m, the leaning board / brace at (2.61, 2.10) already flagged by the route check (legs 7-11 pass
  0.19-0.21 m from it); the CAD tall raster has 0.52 m there, so it is a real CAD part, kept in the walk raster because it is under 0.5 m
  (README: 351 such cells on the deck). As a height field it is SOLID under the board (rays straight down fill the space a real leaning
  board leaves open). The other 11 cells (0.09 m) are the ends of the two 4x4s: real. No spurious cells in this area.
  In the policy's depth: 199 of 1296 rays of seed 1000's frame at t 50 hit that wedge (z 0.06-0.76 m), i.e. the robot at the door-side
  stall faces a ~0.5 m obstacle 0.3-0.5 m ahead-left. Physics and depth agree with each other; the render shows the same cells.
- 15:34 LEANING-BOARD TEST queued (after floor40 and the swap TEST; scratchpad board.sh, course copy app_board): the board is already in the physics as wall box 109 (a tilted 0.6 x 0.6 m panel at (2.61, 2.10, 0.29)); the copy returns the 203 walk-raster cells under it (x 2.53-2.77, y 1.77-2.43, up to 0.49 m) to the 0.04 m floor and keeps the box. Runs: SSR 2250 standard and floor40 on 13 door-side stall seeds (1000-1003, 1006, 1008, 1010-1013, 1015-1017), ACP 5750 seeds 1000-1009, Hiking 3000 seeds 1000-1009; depth frame of seed 1000 at t 50 in the copy. Out: eval/out/TEST_board/.
- SSR 2250 FLOOR40 flat + corridor (15:35): flat_walk 50, flat_stop_stand 50, flat_turn_in_place 50, flat_turn_and_return 50, corridor_L1 1, 
  floor40 corridor L1 1/50: 49 stall 0.215 m from the final station (x 2.885 of 3.10: 1.5 cm OUTSIDE the 0.20 m disc), with the floor40 command 0.40 m/s forward for 45 s (harness 0.22 in the traj) -- the stop does NOT end inside the disc; it decelerates from 0.6 to 0.4 and stops, although from a stand 0.35 m/s walks (speed table). Flat x4 50/50 under floor40.
- SWAP TEST (15:42; TEST, strict + retry, seeds 1000-1049): planner 5750 + tracker 6250: 27/50 (4x4s 7F/5S, ramps 1F/4S, stairs 6F,
  turn-around stalls 4); planner 6250 + tracker 5750: 27/50 (4x4s 10F/1S, ramps 1F/1S, stairs 9F, turn-around stalls 0). Same seeds: pair
  5750 21/50 (22 with the retry: seed 1011 was its one end-stand failure here; stairs 19F, turn-around 0), pair 6250 18/50 (4x4s 10F/11S,
  stairs 4F, turn-around 8). Both swaps beat both pairs on these 50 (intervals overlap: 27/50 = 40.6-67.0 %).
- 15:47 ACP HOLD-OUT GATE (ACP's decision, agreed): on every ACP export first run seeds 1001 / 1005 / 1036 (turn-around) and 1019 /
  1022 (ramp trough), strict + retry, standard goal rule; a stall at the turn-around on any of the three -> "not a candidate", no 200+200.
  Script scratchpad gate.sh (single core). 7000: gate armed (waits for DONE), then its 50 strict on 1000-1049 (against the swap line
  planner 5750 + tracker 6250: 27/50, 4 turn-around stalls). queue3 (old 7000 plan) stopped; its remaining parts (regline for the
  stair seeds, 200+200 only if a candidate, reglines 6500 / 6750) will be restarted after the gate.
- 15:53 STOP RECORDS (coordinator): every failed trial now carries 'stop' in results.json and trials.csv columns stop_nearest_box_m,
  stop_nearest_raised_cell_m (> 0.15 m above the floor under the pelvis, within 1.5 m), stop_course_edge_m (distance to the end of the
  course raster), stop_harness_cmd, stop_adapter_cmd (SSR / Hiking: the velocity command they send; ACP: goal x, y, dpsi).
- COURSE-EDGE DEFECT FOUND (eval, ladder courses): the SSR floor40 corridor stop (pelvis (2.885, -0.001), centred, walls end at x 2.5, nearest
  box 0.75 m, no raised cell) is 1.1 m from the END OF THE 8 m RASTER (x 3.99); beyond it there is no ground, so 119 of 1296 depth pixels
  (the top rows, rays that should hit floor 1.1-3 m ahead) read "far": a void / drop at 1.1 m, which a policy trained on large terrain
  never sees on flat ground. With TEST_GROUND_PLANE=1 (new x2.py TEST option: an endless plane 0.5 mm under the course) 0 pixels read far.
  Ladder tile routes keep points 3.4 m from the tile centre (0.6 m from the edge); flat_stop_stand ends 2.0 m from the edge (SSR passes it).
  Likely affects depth-camera policies (SSR, Hiking) at the far ends of corridor_L1 and the step-2 routes (SSR / Hiking street far-end stalls).
  Test with the plane queued.
- 15:54 coordinator decision: if the ground-plane TEST confirms, pad every ladder course (steps 1, 2) with flat floor >= 3 m beyond its
  edge in physics and depth / scan, record in docs/EVAL.md, bump the regression reference, re-run SSR 2250 and Hiking 3000 corridor, street,
  home (old rows kept, marked "before the edge fix"). Apparatuses untouched. 7000 gate first.
- ACP height scan at a course edge: NOT affected. Apparatus.height / ladder scan_height clamp the index to the raster (np.clip in cell()), so a
  scan point beyond the edge reads the last row's height (flat floor continued), not a void; only ray-cast depth (SSR, Hiking) sees the
  void. ACP rows need no re-run for this.
- 15:56 CORRECTION: my messages stamped 16:03 and 16:15 were sent at about 15:45 and 15:52 (stamps estimated again, not taken from date -u). Hiking 4125 FLOOR flat: 50 / 50 / 1 / 50 (turn-in-place mark >= 33: missed). Card + tables running.
- HIKING 4125 (15:58; check PASS; corrected, strict + retry): flat card 50 / 50 / 32 / 50, floor 50 / 50 / 1 / 50; speed 0.2/0.35/0.5/0.6/0.8
  -> 0.24/0.37/0.49/0.56/0.66 (+20 / +6 / -2 / -7 / -17 %); turn +0.4/+0.8/-0.4/-0.8 -> 0.43/0.95/-0.47/-0.97 rad/s, drift 0.40/0.24/0.53/0.22 m;
  stop 0.28-0.29 m, last 2 s 0.006-0.007 m; stand at zero held. Marks: turn in place >= 33 floor FAILED (1; card 32), turning drift <= 0.3 m
  FAILED at +-0.4 (0.40 / 0.53; met at +-0.8), stop <= 0.40 MET, stop-stand >= 45 MET, walk 50 MET, speed table good from 0.35 up.
- 15:59 Hiking 4125 card full look (L1, street, home, apparatus 50 strict) queued after the ACP 7000 gate + 50, before turnwalk (queue2 re-armed to wait for it). Floor variant dropped from routine Hiking runs (coordinator).
- GAIT POSTURE on flat_walk (new eval/gait_stats.py; recorded trajectories, frames walking > 0.2 m/s; FORWARD LEAN POSITIVE; torso_link and pelvis
  frames are vertical at the default pose, checked) (16:08):
  Hiking 3000: torso -16.6 deg (p5..p95 -18.7..-13.6), pelvis +3.5 (1.4..6.6), stance width 0.230 m, crossing 0 of 1394 touchdowns.
  Hiking 4125: torso -15.5 (-17.7..-13.5), pelvis +4.4 (2.6..6.2), width 0.211 m, crossing 0 / 1250.
  SSR 2250: torso +3.6 (-3.8..11.1), pelvis +0.3 (-0.5..1.6), width 0.221 m, crossing 0 / 1635.
  ACP 5750: torso +11.0 (3.4..17.5), pelvis +4.5 (1.1..7.1), width 0.189 m, crossing 0 / 524.
  -> the user's observation holds: Hiking's torso leans BACK about 15-17 deg from vertical while its pelvis tilts forward 3.5-4.4 deg (about
  20 deg of backward waist pitch); unchanged 3000 -> 4125. Now part of every flat-walk report.
- HIKING LEAN BY ZONE (16:10; torso pitch, forward positive, frames every 0.06 s incl. standing; L1 courses start only 0.7-1.0 m before
  the obstacle, so "more than 1 m before" = the flat walk, -15.5 / -16.6):
  Hiking 3000: log: before -16.7, last 0.75 m -17.6, over the log -16.4, after -16.0. stairs up: flat before -18.5, last 0.4 m -18.7, on the
  flight -15.5, landing -14.0. stairs down: landing -14.3, top edge -15.4, on the flight -15.8, floor after -10.7.
  SSR 2250: log -1.7 / -1.6 / +3.4 / +4.7; stairs up -1.1 / 0.0 / +1.6 / +2.4; stairs down +0.8 / +2.6 / +7.5 / +3.7.
  -> the user's prediction is NOT borne out at 3000: Hiking keeps leaning back 14-19 deg next to and on stairs (at most 3 deg less back on
  the flight); SSR leans forward a few degrees on the obstacles.
  Camera reach (Hiking chest camera, 61.6 deg down, 58 deg vertical FOV, depth clipped at 3.0 m): upright default pose: camera 1.04 m high,
  top row meets the ground 1.66 m ahead of the toes, bottom row at the toe tips (0.02 m). At the walking lean (-16.4 deg): camera 1.11 m,
  top row 3.73 m ahead (ray 4.06 m, beyond the 3.0 m clip, so the top rows read "3 m"), bottom row 0.18 m ahead of the toes (the toes leave
  the image). Mean over the walk: top-row reach 3.75 m.
- SSR 2250 FLOOR40 routes (16:19): home 50/50 (standard 44), street 0/50: 41 stalls, all heading to the LAST station (-3.20, -2.10), 1.03-1.05 m from the course edge where recorded (15.9 of 18 waypoints reached on average; standard stalled earlier at (-1.90, -2.10)); 9 falls. Consistent with the course-edge void; the edge TEST will tell.
- 16:23 ACP 7000 / 6750 hand-overs failed on the node (no DONE); ACP re-exports. Meanwhile queue (scratchpad queue4.sh): edge TEST -> Hiking 4125 beyond flat -> turnwalk 50s; board TEST after floor40; the gate (single core) pre-empts when 7000 lands. Heartbeat refreshed.
- EDGE TEST (16:26): Hiking 3000 street with the ground plane: 10/10 (without: 37/50, all 12 stalls at the far-end turn-around station). SSR runs continue.
- 16:32 EDGE FIX APPLIED (coordinator's decision): eval/ladder.py pads every ladder course with 3 m of flat floor (median border
  height), physics and scans / depth (LADDER_PAD_M=0 = old courses); summaries record course_padding_m; docs/EVAL.md section added.
  Regression reference unchanged: it covers only our apparatus (not padded), so it still applies; checked by its next automatic run.
  Hiking 4125 L1 running across the change: log_L1 and wedge_L1 ran unpadded ("before the edge fix"), later tests padded (field in summary).
  Re-runs to do: SSR 2250 and Hiking 3000 corridor L1, street, home (padded); old rows kept.
- ACP 7000 HOLD-OUT GATE (16:33; re-export with DONE, check ALL PASS, planner = frozen 5750 (md5 f4da33e8); strict + retry, standard goal):
  turn-around 1001 / 1005 / 1036: success / success / success (22/22 each, all end at button 5) -> CANDIDATE. Ramp trough 1019: success;
  1022: fall at 6/22 at (3.10, 0.62) going back over the front 4x4. Next: 50 strict (1000-1049), then 200 (+ disc chunks), stair-seed regline.
- ACP 7000 50 strict (16:56; seeds 1000-1049, retry): 26/50 = 52 % (38.5-65.2); 18 falls (4x4s 13, ramps 2, stairs 3), 6 stalls (4x4s 3, ramps 3); turn-around stalls 1; button 5 26/26 held. Swap line planner 5750 + tracker 6250: 27/50 (4 turn-around stalls); 5750 22/50 (retry), 6250 18/50. 200 strict (+disc chunks) next.
- 16:58 queued: traces of ACP 7000's 16 failures in the 4x4 section (scratchpad fall4x4.sh -> f7000_<seed>.npz), after the 7000 200 starts; analysis by phase (wp5 turn-around 3F/1S, button 1 1F, wp6 back over the front 4x4 6F/1S, wp7 door side 3F/1S), side hits, planner target of the failing step, heading vs the beam axis.
- Hiking 4125 beyond flat (17:06, card, strict + retry): L1 log 50 (before the edge fix), wedge 50 (before), stairs up 49, stairs down 50, corridor 50 (padded); street 50/50, home 31/50 (14 stalls heading to (2.10, 2.25), 5 falls) (padded). Hiking 3000 (unpadded): street 37, home 48.
- TURNWALK 50s (17:09; strict + retry, seeds 1000-1049): 5750 18/50 (standard 22 with retry; turn-around stalls 4 vs 0; 4x4s 13F/7S,
  stairs 11F); 6500 22/50 (standard 19; turn-around stalls 2 vs ~20 stalls at the 4x4s; 4x4s 9F/6S, ramps 2F/8S, stairs 3F).
  -> turnwalk helps 6500 (+3, fewer turn-around stalls) and costs 5750 (-4).
- HIKING 4125 OUR APPARATUS STRICT 50 (card, retry): 0/50; 35 falls, 15 stalls; furthest 12/22 (button 3). By target: 'clear of the door wall'
  21 falls (the turn out of the doorway toward the front 4x4: 3000 had 3 there), button 1 7 stalls, over the front 4x4 5 falls, button 2
  4 stalls + 2 falls, door side 3 falls, button 3 2 + 2. 3000: 0/50, furthest button 3. Last station never reached by either.
- 17:09 HARNESS FIX: partial.json now saved after every trial (was every 5); SSR floor40 apparatus slices (trials of 300-400 s wall each) were killed by the 590 s slice timeout before reaching the 5-trial save, so the strict run restarted from seed 1000 every slice for ~2 h (disc stuck at 10). No result affected, only time lost.
- HIKING 4125 DOORWAY FALLS (17:14; seeds 1002 / 1003 / 1004, all heading to wp2 'clear of the door wall' (2.77, 0.69)): it walks
  south from 'through the door', misses wp2 and overshoots to (2.89-2.94, 0.32-0.37), 0.33-0.36 m past it, 0.27 m from the south wall
  (box 69) and 0.06-0.1 m from the FRONT 4x4's centre line (beam x ~3.0 there); facing west (yaw +-3.0), commanded a turn on the spot
  (wz -1, vx 0) then 0.2-0.3 m/s forward; feet repeatedly hit the 4x4's side (foot_side '4x4' 93-232 N, 6-8 events), the left ARM hits
  the wall (235-367 N); it then moves backward and sideways (lateral -0.4 to -0.96 m/s) over the 4x4 to the east and falls at
  (3.08-3.59, 0.76-0.85). -> not the door sill and not a wide turn at the doorway: an overshoot of wp2 into the 4x4 / wall corner, then a
  turn on the spot with the feet against the 4x4 that ends in a backward-sideways slide over it. Render queued (seed 1002, t 5-8.3).
- 17:15 STATION LEGS built (eval/legs.py; harness Trial.run(prestand=...)); docs/EVAL.md 'Station legs (user, 2026-11-12)' incl. the order rule (full route only when every leg passes, 45/50 proposed). Spawn level test changed to a plane fit (buttons 3 and 4 are on the 15 deg ramps; a flat-spread test rejected every draw there). Smoke test ACP 7000 leg 2: 2/2.
- 17:15 queue5 (scratchpad queue5.sh, log queue5.log; then edgefix_reruns.sh): after ACP 7000's 200 merges: render Hiking 4125 doorway seed 1002; legs ACP 7000 (5 legs in parallel, legs_run.sh); Hiking 4750 flat + gait + tables + L1 + routes (no full route: legs first); legs SSR 2250, Hiking 4125, Hiking 4750; then edge-fix re-runs. Board TEST waits for SSR floor40 apparatus (slow).
- CONTACT CHECK (17:19; coordinator after the user's request): Hiking 4125 seed 1002 'right foot on the 4x4 side, 3,289 N, 1,804 N s':
  replay of the recorded states: the right foot is BESIDE the beam (ankle 0.11-0.15 m west of its centre line; the toe reaches the beam's
  west face at x ~3.02-3.04); one impact frame at t 4.80 with 38 sphere-terrain contacts, deepest 11.7 mm, foot load 5,675 N; afterwards
  0-450 N and <= 2 mm. The 1.2 s "duration" is the event grouping of intermittent contacts; the impulse sums |horizontal force| over every
  contact point (up to 38 per step, pinching ones in opposite directions included), so it is not a momentum (the pelvis dv is 0.29 m/s,
  i.e. about 12 N s net). Force / impulse figures were already "not to be quoted" (2026-11-11 note); now shown why.
  SUSTAINED kN LOADS (the real issue): foot loads > 1,500 N held >= 0.2 s in kept trajectories: ACP 7000 7 of 18, Hiking 3000 5/22, Hiking 4125
  5/20, SSR 2250 0/7 (ACP 200-runs keep no trajectories). Located: all but one at the front 4x4 area (pelvis x 2.84-3.37, y 0.47-0.91; one
  at the stair foot), 4-20 kN for 0.2 s. Replay of two (ACP 1028 t 21.6, Hiking 1006 t 17.9): the foot is on FLAT floor (raster exactly
  0 for 0.3 m around), 100-137 sphere-terrain contacts, sole spheres 19-24 mm deep, with HORIZONTAL normals (+-x, +-y, diagonal).
  Mechanism: the height field is 2 cm cells = triangle prisms 1-2 cm across; a 2 cm sole sphere pressed more than ~1 cm into the floor
  (soft contact, solref 0.02 s) finds a shorter way out sideways through a prism, so neighbouring prisms push it from opposite sides:
  a jammed foot with kN forces that do nothing to the body. An eval physics defect (all tracks, anywhere a foot stamps hard), not the
  4x4 being cells: it happens on the flat deck next to the 4x4.
  TEST running (contact_test.sh): Hiking 4125 seeds 1002-1004 and ACP 7000 4x4-section seeds 1002 1006 1012 1013 1017 1020 1022 1025, each
  with (a) TEST_BOX_4X4=1 (beams as boxes, their cells flattened in the physics only) and (b) TEST_TERRAIN_SOLREF=0.005 (stiffer floor).
- 17:25 SSR r7 3550 copied (eval/data/handover/ssr_r7_3550), check PASS (depth 3.2e-4, action 0); its card's eval_command 'floor40' keeps the harness yaw (1.5 x error, cap 1); the eval's floor40 (coordinator's spec) also floors turns over 60 deg at 0.6 rad/s: the difference is named in reports. Queued (queue6.sh) after Hiking 4750's legs: flat + gait, tables, L1 + routes (padded), legs standard and floor40; full route only if every leg >= 45/50. r7_003800 NOT to be scored (misplaced routes). Hiking's next export comes from the run restarted at 5000.
- ACP 7000 OUR APPARATUS STRICT 200 (17:30; seeds 1000-1199, retry, goal + lookahead): 125/200 = 62.5 % (55.6-68.9), clean 0; 50 falls, 25 stalls. Map F/S: 4x4s 37/14, ramps 4/8, stairs 9/2, landing 0/1. Turn-around stalls 11. Button 5: 126 reached, 125 held. Successes 62-112 s (median 71). Reference 5750: 114/200 (57.0 %, 50.1-63.7). Disc chunks 1050-1199 ran too (disc 1000-1049 not run; disc 200 not merged).
- CONTACT TEST RESULT (17:32; TEST, strict + retry): ACP 7000's 8 seeds that failed in the 4x4 section (1002 1006 1012 1013 1017 1020 1022
  1025): (a) 4x4s as boxes: 5 succeed, 1006 still falls at the 4x4s, 1025 ramps, 1002 stairs; (b) floor solref 0.005: 7 succeed, 1022 stalls on
  the ramps. Hiking 4125 seeds 1002-1004: (a) stall 4/22, falls 4/22 and 2/22; (b) falls 2/22, 3/22, 9/22 (all still fail near the 4x4 / doorway).
  Sustained foot loads > 1,500 N for >= 0.2 s: (b) 0 of 11 trials, (a) 2 of 11 (before: 7/18 ACP, 5/20 Hiking kept traj); single-frame impact
  peaks of 14-18 kN remain in both. CAVEAT: any physics change re-rolls a trial; at ACP 7000's 62.5 % base rate about 5 of 8 failed seeds would
  succeed by chance alone, so (a) 5/8 is not evidence; (b) 7/8 is suggestive only. A like-for-like 50 / 200 with solref 0.005 is needed.
- 17:33 coordinator: the floor jam will be corrected either way; which correction is decided after the solref-0.005 50s and the independent audit (docs/findings/sim_artifact_audit.md). Until then every apparatus number is PROVISIONAL. Running now (2 cores, scratchpad solref50.sh, out eval/out/TEST_solref005/): ACP 7000 and 5750 (goal + lookahead), Hiking 4125 (card), SSR 2250 (standard), 50 each seeds 1000-1049, plus ACP 7000 flat x4. Hiking's three traced doorway falls fail under both variants: the policy's own.
- 17:33 SSR 2250 floor40 APPARATUS paused (my processes, stopped to keep within 6 cores while the legs and the solref TEST run); partial saved, resumes with the same command (scratchpad ssr_floor40.sh app part). The board TEST waits on it.
- 17:33 board TEST waiter stopped (it would have started on the floor40 pause); to be re-queued after the legs, the solref TEST and the edge-fix re-runs (scratchpad board.sh, drop its first wait line).
- ACP 7000 4x4-SECTION FAILURES traced (17:34; 16 seeds re-run with the planner logged, scratchpad f7000_<seed>.npz; pose 0.5 s before the
  end; "side" = metres from the beam line, + = east; PROVISIONAL, the floor-jam defect is in these runs):
  * drift east into the ramp cells during or after the turn-around at button 1 (pelvis x 3.4-3.95, 0.4-0.9 m east of the front 4x4, wedge
    and wall hits, then a fall or stall): 1002 (stall), 1013, 1020 (stall), 1026, 1029, 1030, 1032, 1046 -> 8 of 16. Planner targets there
    are on the east side with high occupancy: it plans steps where the robot already is, not back toward the beam.
  * fall while crossing the front 4x4 going back west (pelvis 0.08-0.19 m from the line): 1022, 1025, 1043. The planner's targets are already
    WEST of the beam (0.06-0.38 m, occupancy 0.94-1.0): the step over is planned, the tracker does not complete it (right foot hits the
    ramp-end face 'wedge' / 'other' 2-4 times, head on the wall). Seed 1022: pelvis (3.23, 0.79), heading 143 deg off the crossing
    direction (it is turned well away from square), targets L 0.38 / R 0.20 m west.
  * door side (after crossing): 1006, 1017 on the beam line (0.04-0.05 m, feet on the 4x4 / wall), 1035 (0.22 m west, arm on the wall), 1028 stall.
  * 1012: trunk on the wall near (2.56, 1.20), heading to button 1 (re-run differs from the 50-trial outcome; single seeds re-roll).
  Sustained floor jams were found in 7 of 18 kept trajectories of this run (see the contact check): these falls stay provisional.

### HOW TO RESUME (17:35 UTC; for a new eval-monitor session)
All job scripts are in the scratchpad C:\Users\PC\AppData\Local\Temp\claude\74893d75b5436cbb\cf3ca1e7-ff31-44d4-97d3-238f0a3bcf48\scratchpad\ (logs beside them).
Running now: solref50.sh (TEST terrain solref 0.005, 50 trials each: ACP 7000, ACP 5750, then ACP 7000 flat x4; Hiking 4125, then SSR 2250;
out eval/out/TEST_solref005/) and queue7.sh (consolidated, log queue7.log): legs ACP 7000 -> Hiking 4750 scoring (hiking_full3.sh) -> legs SSR 2250,
Hiking 4125, Hiking 4750 -> SSR 3550 (queue6b.sh) -> edge-fix re-runs (edgefix_b.sh). Not queued yet: board TEST (board.sh minus its first
wait line, out eval/out/TEST_board/), SSR 2250 floor40 apparatus resume (6 strict / 18 disc saved in partial.json; rerun the 'app' part of
ssr_floor40.sh), ACP 7000 disc 1000-1049 (to merge a disc 200), regression reference with legs. Every harness / legs / ladder run continues
from partial.json when restarted with the same command. If these background jobs die with the session, restart solref50.sh and queue7.sh.
Rules in force: strict stand + end retry (user-confirmed), padded ladder courses, station legs before the full route, ACP hold-out gate
(gate.sh) on every ACP export, apparatus numbers PROVISIONAL until the floor-contact correction is decided.

### 17:38 UTC: what went wrong, process cleanup
- Found my old queues still alive: queue2 (two instances, 06:42 and 07:59 local start) had just restarted the ACP turnwalk 50s for 5750 and 6500 (a
  duplicate of finished runs, 2 cores, about 2 min); queue3, queue6 (which queue7 replaces) and the board waiter were also still polling. I had believed
  queue6 stopped. All stopped by their scratchpad command lines, with the two turnwalk harness children and three stale log tails. Damage: only the two
  turnwalk folders' partial.json were overwritten (6 trials of the rerun); summary.json, results.json and trials.csv (08:44, 18/50 and 22/50) are
  intact. Do not resume those two folders from partial.json.
- Now running exactly 6 single-core jobs: solref 0.005 TEST (ACP 7000 at 12/50, all 12 succeeded; Hiking 4125 at 8/50, falls at waypoint 2 as before)
  and queue7 legs ACP 7000 (legs 1-4 in parallel, then leg 5). Rule for myself: before launching any queue, list every scratchpad process and stop the
  superseded ones first.

### 17:40 UTC: queue8 replaces the rest of queue7 (coordinator)
queue7's main shell stopped (its legs ACP 7000 run continues). queue8.sh (scratchpad, log queue8.log): hold-out gate ACP pair_s2_v2_007250
(seeds 1001/1005/1036, 1019/1022; strict + retry; goal 0.20 + lookahead 0.30) as soon as one core is free -> legs ACP 7250 after the 7000 legs
-> 50 strict on 1000-1049 if the gate passes (PASS = turn-around 3/3 and at least 4/5; ACP 7000 had 3/3 and 1/2 ramp) -> Hiking 4750 scoring
-> legs SSR 2250, Hiking 4125, Hiking 4750 -> SSR 3550 -> edge-fix re-runs. Board TEST stays unqueued until the corrected-floor 50s are in.

### 17:48 UTC: first station-leg numbers (ACP 7000) and the ACP 7250 gate (provisional; floor contact not yet corrected)
ACP 7000 legs (strict + retry, 50 each, prestand 1 s, goal 0.20 + lookahead 0.30): leg 1 49/50 (90-100 %), leg 2 39/50 (65-87 %; 10 falls, 1 stall,
mostly within 4-8 s at 0-1 of 5 waypoints = the front 4x4 turn-around), leg 3 46/50 (81-97 %), leg 4 50/50 (93-100 %), leg 5 43/50 (74-93 %; 4 falls at
waypoint 4 of 7, 16-19 s, and 3 stalls, two at 0/7). Product 0.605 vs full route 125/200 = 0.625: the legs are close to independent. Under the user's
rule (full route only after every leg is at least 45/50), legs 2 and 5 fail.
ACP 7250 gate FAIL: turn-around 1/3 (1001 fall over the back 4x4, 1036 fall door side; 1005 ok), ramp 2/2. ACP 7000 got 3/3 and 1/2. The planner is
unchanged (md5 f4da33e8, the same as 5750 and 7000) and the tracker changed, so this regression is the tracker's. No 50 for 7250; its legs run next (queue8).

### 18:00 UTC: ACP 7250 legs, leg-5 traces, queue9 (HOW TO RESUME: current queue is queue9.sh, log queue9.log)
- ACP 7250 legs: 49 / 41 / 44 / 49 / 44 of 50, product 0.610 (7000: 0.605). Its gate failed (1/3 turn-around).
- ACP 7000 leg-5 falls traced: eval/out/acp/pair_s2_v2_007000/legs_ours/leg5_fall_traces/README.txt. In all four, the robot is still 18-45 deg off the
  flight at the first step-up, after arriving at the stair foot 75-81 deg off. Then come riser stubs, step-edge jams (B8-type, 0.24-1.7 s)
  and contact with its own knee. The 0/7 stalls are the ramp-trough stall, not a stand start.
- eval/legs.py: a waypoint with "test_stop": true keeps its stop but does not split legs (only TEST course copies use it).
- queue9 replaced queue8 (main shell stopped; the 7250 legs had already finished). Order: gate ACP 7500 + stair-square leg-5 TEST (course
  copy scratchpad/app_stairsquare, out eval/out/TEST_stairsquare/) -> legs ACP 7500 -> Hiking 4750 -> legs SSR 2250 / Hiking 4125 / 4750 ->
  SSR 3550 -> edge-fix. No 50 for 7500 (the selection rule is now the gate plus leg product). solref50.sh is still running beside it.

### 18:37 UTC: audit actions (coordinator relay after docs/findings/sim_artifact_audit.md)
- solref50.sh STOPPED (mine). The effective value was 0.0125 s and the run had no control. Its finished trials stay in eval/out/TEST_solref005/ as
  labelled TEST data only. My earlier "5 of 8 failing seeds pass" was not evidence (a control passes 5 of the same 8).
- ACP 7500 gate FAIL: 1/5 (turn-around 1/3: 1001 fell at the front 4x4, 1036 stalled at the ramps; ramp seeds 0/2). Its legs:
  45/36/41/49/43, product 0.448. ACP 7250's legs: product 0.610. ACP 7000's legs: 0.605. All on the current floor, PROVISIONAL.
- Stair-square TEST (leg 5, ACP 7000, 1 s square stop at the stair foot; the held-out course is unchanged): 38/50, against 43/50 without
  the stop. No control was run, and under 20 points of difference means nothing (audit).
- Measurement v2 in eval/harness.py (no physics change):
  - foot load is the z of the summed force vector per foot, wall boxes included (box 202 now counts);
  - event peak and impulse come from the summed vector per contact key and step;
  - auto-flags per trial (flags: trapped_foot_s, limb_clamp_s, joint_limit_s and the max excess; the last 1 s of a fall excluded) are in
    results, trials.csv and the summary.
  - Regression 10/10 bit-identical (note: it needs OMP_NUM_THREADS=1; without it the 4 old-model baseline trials differ, before or after
    the patch).
  - Backup of the old harness: scratchpad harness_before_measure_v2.py.
- New option PHYS_FLOOR=solid (eval/phys_floor.py, wired in eval/x2.py; NOT the default):
  - 630 flat boxes, 8 tilted ramp slabs (15.0 deg plane fits), 2 4x4 boxes, the brace wedge removed, box 202 out of contact;
  - the height field stays drawn and scanned only;
  - checked against the raster at 40,000 random points: p99 deviation 2.4 mm, and every point off by more than 1 cm is at a step edge
    (half-cell shift).
- Running: physfloor.sh (log physfloor.log, out eval/out/TEST_physfloor/): ACP 7000 1000-1049 solid vs control (current floor, v2), 3
  slices each, then flat x4 for both.
- Stopped: hiking_full3.sh 4750 (to avoid mixing measurements). Requeue it after the decision.
- Not queued: legs SSR 2250 / Hiking 4125 / 4750, SSR 3550, edge-fix re-runs.
HOW TO RESUME: rerun physfloor.sh (it resumes from partial.json). The rest of the queue waits for the coordinator's floor decision.

### 18:54 UTC: solid-floor validation (TEST, ACP 7000, seeds 1000-1049, v2 measurement, strict + retry)
- solid 27/50 (40-67 %) against control (current floor) 26/50 (39-65 %). 9 of 50 seeds flip one way and 8 the other: no measurable effect.
- The control reproduces the recorded v1 run on all 50 outcomes, so v2 measurement changes no verdict here.
- Failures, solid / control: 4x4 falls 12/13, 4x4 stalls 2/3, ramp stalls 5/4, stair falls 3/3.
- Trapped foot: solid 1 trial, 0.11 s (seed 1028, at the 4x4s during a fall); control 25 trials, 8.3 s, longest 1.27 s.
- Limb clamp: solid 1 (head, 0.18 s, seed 1044, fall); control 2.
- Joint-limit flag in 47 and 50 trials, about 1 s each in both: ordinary walking, so 0.05 rad is too tight or the soft limits are normal.
  Not yet examined.
- Flat x4: solid 50/50 x4, PASSED. Control 50/50/49/50: one turn-in-place fall on the current floor.
- Waiting for the coordinator's decision on the default floor, the new regression reference and the re-score list. Hiking 4750 scoring
  and the queue tail stay paused.

### 19:00 UTC: solid floor is the default (coordinator decision); new eval session
- physfloor.sh finished 18:53 (eval/out/TEST_physfloor/; aggregator scratchpad pf_agg.py). ACP 7000 seeds 1000-1049, strict:
  solid 27/50 = 54 % (40-67) vs control (old floor) 26/50 = 52 % (39-65). Trapped foot: solid 0.1 s in 1 trial vs 8.3 s in 25 trials;
  limb clamp 0.2 s vs 50.9 s; peak force max 2,682 N vs 16,687 N. Solid has a little more side contact: side hits 1,150 vs 1,011,
  stair side hits 286 vs 237, shin 15 vs 2, knee 90 vs 76. Not looked at per event yet (possible box-edge snag at risers; open).
  Flat x4: solid 50/50 on all four (PASSED); control 49/50 on turn_in_place (NOT PASSED; a one-trial miss on the old floor).
- eval/phys_floor.py: solid is the default; PHYS_FLOOR=heightfield selects the old floor and its runs say "OLD FLOOR" in the simulator
  line. Backup: scratchpad phys_floor_before_default.py. docs/EVAL.md "Robot and simulators" has the note.
- Regression reference re-recorded on the solid floor (OMP_NUM_THREADS=1): 10/10 bit-identical on re-check. Old reference kept as
  eval/config/regression_ref_oldfloor_20261112.json. The old floor differs from the new reference (expected).
- Joint-limit flag (0.05 rad excess) fires in nearly every trial. Distribution (scratchpad jlim_dist.py, traj of 26 solid / 29 control
  trials): the offenders are ankle roll (range +-0.262 rad; excess > 0 in ~5 % of frames, p99 0.05, p99.9 0.09-0.12 rad), then ankle pitch.
  Shoulder roll sits just past its limit in 70-94 % of frames by < 0.01 rad (rest pose at the limit). Trials over threshold (solid / control):
  0.05: 24/26, 29/29; 0.10: 21/26, 27/29; 0.15: 9/26, 14/29; 0.20: 5/26, 5/29. PROPOSAL: flag at 0.15 rad, and report ankle-roll time past
  0.05 rad as its own number (the policies push the 15 deg ankle-roll limit as a habit). Not changed in the harness; do not quote the flag yet.
- Running: scratchpad solidq.sh (log solidq.log, markers solidq.stage1.DONE / solidq.DONE / solidq.FAILED), all solid floor:
  stage 1 in parallel: Hiking 4750 flat (out .../x2_parkour_v4_4750/ladder_solid) + legs (legs_ours_solid); SSR 3550 flat (eval/out/ssr/r7_3550/
  ladder_solid) + legs (legs_ours_solid, standard command); ACP 7750 gate (gate.sh, fetches the export; regline/) + legs (legs_ours_solid).
  stage 2: ACP 7000 full route 200 (eval/out/acp/pair_s2_v2_007000/ours_solid_200/s1000..s1150, 4 x 50), ACP 7000 legs (legs_ours_solid),
  Hiking 4125 apparatus 50 (x2_parkour_v4_4125/ours_solid_50). Old-floor results stay where they are and are "old floor" in tables.
- Not done: the Hiking 4750 old-floor partial outputs (ladder_nohip, speed/turn/stop_nohip from the stopped run) are left as they were.
HOW TO RESUME: if solidq.sh died, check solidq.log for the last "SQ" line; each harness / legs run resumes from its partial.json, so rerun
the missing stream's commands from solidq.sh (do not rerun the whole script while parts of it run: list `ps -ef | grep eval/` first).
Report due to the coordinator when Hiking 4750 and SSR 3550 legs are in (legs.txt in their legs_ours_solid). Heartbeat: eval/tools/heartbeat.sh.

### 21:10 UTC: stage 1 in (solid floor, strict); reported to the coordinator
- ACP 7750 (export fetched 18:57, check ALL PASS, planner md5 f4da33e8): gate 3/5 (turn-around 2/3: 1005 fell going up the stairs;
  ramp trough 1/2: 1022 fell at the ramps front side) -> gate FAIL by the rule. Legs 50/43/43/50/43, product 0.636 (solid; 7000 old floor 0.605).
- Hiking 4750: flat gate NOT PASSED: turn_in_place 12/50 (stalled 37); others 50/50. Legs 22/30/0/0/0, product 0. Legs 3-5 mostly stalls.
- SSR 3550: flat gate NOT PASSED: stop_stand 2/50 (stalled 39, not a stand 9); others 50/50. Legs 28/5/23/34/0, product 0; leg 5: left the
  course 34/50. SSR 2250 (old floor) had stop_stand 50/50.
- Old-floor controls (PHYS_FLOOR=heightfield, 50 each): SSR 3550 stop_stand 11/50; Hiking 4750 turn_in_place 10/50. So the flat failures
  are the checkpoints, not the floor. SSR stop_stand is lower on solid (2/50 vs 11/50, intervals 1-13 % vs 13-35 %): open, worth a look.
- Outputs: <ckpt dir>/ladder_solid, legs_ours_solid, ctrl_oldfloor. No renders made.
HOW TO RESUME (21:10): solidq.sh stage 2 running (ACP 7000 200: slices 1050/1100/1150 done, 1000 running; ACP 7000 legs_ours_solid; Hiking 4125
ours_solid_50). Marker solidq.DONE. Then: merge the four ACP 200 slices (strict success, Wilson interval, section map), report stage 2. Open: joint-limit
threshold proposal (0.15 rad + ankle-roll time) awaiting the coordinator; solid-floor side/shin contact rise (15 vs 2 shin) not examined per event.

### 21:20 UTC: coordinator's 21:15 actions
- Joint-limit flag changed in eval/harness.py (backup scratchpad harness_before_jlim015.py): flag at 0.15 rad (JLIM_EXCESS), new per-trial
  flags.ankle_roll_past_s (ankle roll > 0.05 rad past +-0.262) and summary totals; trials.csv column flag_ankle_roll_past_s. One-trial test OK,
  regression 10/10 bit-identical. docs/EVAL.md "Robot and simulators" has the flag note. The ACP 7000 200 on solid finished before the change
  (its joint_limit_s is at 0.05); ACP 7000 legs also finished before it (20:59). Hiking 4125 50 was mid-run: slices restarted after ~21:10 use 0.15 (mixed flag in that run only).
- ACP 7000 FULL ROUTE 200, SOLID (seeds 1000-1199, strict + retry, goal + lookahead; eval/out/acp/pair_s2_v2_007000/ours_solid_200/s*; merge
  scratchpad agg200.py): 115/200 = 57.5 % (51-64). Old floor: 125/200 = 62.5 % (55.6-68.9). Falls 60, stalls 25. Stops: 4x4s 45, ramps 21,
  stairs 18, landing 1. Sections: 4x4s 155/200, ramps 134/155, stairs 116/134, landing 115/116. Trapped foot 2.0 s in 14 trials.
- Solid-floor extra shin/knee contacts (scratchpad ev_cmp.py, lip_probe.py; physfloor 50 vs 50): shin 15 vs 2 and knee "wedge" 36 vs 12. 10 of the 15
  shin hits are at x = 3.646 m, y 0.48-0.62 (5 seeds), and 4 more at (4.886, 1.70). Probe at y = 0.55: the solid floor rises as a vertical 0.147 m
  face between x 3.64 and 3.65 (ramp slab geom 637), while the raster (what the policy scans) climbs from 3.655 to 3.675 (half-way 3.665):
  the physical face stands ~1.5-2 cm in front of the scanned edge. Likely cause: ramp-slab extents u0/u1 = cell centres +- res/2
  (phys_floor.py line ~89) reach half a cell past the raster's mid-ramp. At (4.885, 1.70) the face and the raster agree. PROPOSAL (not done):
  trim the slab sides to the raster's half-way line, re-probe, re-run the 50 vs 50. Stair risers: knee "stair" 35 vs 20, not probed yet.
- SSR 3550 stop_stand, solid 2/50 vs old 11/50: on both floors it does not stop. Under a zero command it travels a median 0.43 m (solid) /
  0.41 m (old) in the 1 s stand window; no trapped feet or side hits. A checkpoint defect (it does not stop on a zero command), not the floor.
- Queued: scratchpad solidq2.sh (log solidq2.log, marker solidq2.DONE) waits for solidq.DONE, then ACP pair_s2_v3_007250 gate + legs, and SSR
  r7_v12_last legs 1 + 5. SSR r7_v12_last had NO hand-over export at 21:20 (only eval/export/ckpts/r7_v12_last.pt on the SSR node):
  the script waits up to 3 h for eval/export/r7_v12_last/{policy.onnx,adapter_card.json,obs_check.npz}, then fetches, checks and runs.
HOW TO RESUME (21:20): running solidq.sh stage 2 (ACP 7000 legs_ours_solid, Hiking 4125 ours_solid_50) -> solidq.DONE; then solidq2.sh.
Report to the coordinator is due when the ACP 7000 legs are in (200 is in). Open: the ramp-slab face offset (proposal above); stair riser probe.
- 21:13 ACP 7000 LEGS, SOLID: 47/39/49/50/41, product 0.589 (old floor 49/39/46/50/43, 0.605). Reported to the coordinator.
HOW TO RESUME (21:13): solidq.sh still running Hiking 4125 ours_solid_50 -> solidq.DONE -> solidq2.sh (ACP v3 7250 gate + legs; SSR r7_v12_last
legs 1+5 once its export exists). Next report: when solidq2 is in. Open: ramp-slab face offset proposal, stair-riser probe, SSR v12 export.

### 21:20 UTC: ramp slabs trimmed (coordinator approved); queue 3
- eval/phys_floor.py: _fit_side() puts each ramp slab's sides AND ends where the raster (bilinear, checked equal to MuJoCo's hfield) crosses
  half-way between the slab plane and the floor 1.5 cells outside (search up to 3 cells in, outermost crossing; unchanged without a 2 cm step).
  PHYS_FLOOR_TRIM=0 = the untrimmed solid floor. Backup of the untrimmed file: scratchpad phys_floor_untrimmed.py.
- Whole-apparatus edge probe (scratchpad edge_probe.py, 1 cm grid, 764k points, beams excluded; misaligned = |solid - raster| > 2 cm and
  > 0.6 x local step): untrimmed 1,162 (ramp protrudes 1,080, flat holes 82); trimmed 132 (ramp protrudes 46, one 5 mm band at x 6.12,
  y 2.1-2.3; flat holes 86 = the brace wedge removal at (3.40, 1.10), intended, and a 1 cm band at (3.10, 1.41), 9 cm deep, also in the
  untrimmed floor, probably beside a 4x4 where the raster's 4x4 shoulder is flattened; not fixed). Flat boxes (stair risers, deck edge) show no
  protrusion anywhere: risers are aligned, not changed. Outputs: scratchpad edge_probe_before.txt / edge_probe_fit3.txt.
- Regression on the trimmed floor: 10/10 bit-identical against the 18:54 reference (its 12 s trials do not reach the ramps); reference kept.
- STOPPED (mine, verified command lines): solidq.sh (Hiking 4125 50 was mid-run, 18 trials on the untrimmed floor; set aside as
  eval/out/hiking/x2_parkour_v4_4125/ours_solid_50_UNTRIMMED_PARTIAL_stopped) and solidq2.sh (was still waiting). SSR r7_v12_last dropped
  (coordinator: SSR's v13 +250 export about 22:30 replaces it).
- Running: scratchpad solidq3.sh (log solidq3.log; markers solidq3.DONE / FAILED): TEST trimmed vs untrimmed, ACP 7000 1000-1049
  (eval/out/TEST_trim/acp7000_{trimmed,untrimmed}_s*); ACP pair_s2_v3_007250 gate + legs (legs_ours_solid); Hiking 4125 apparatus 50
  (ours_solid_50/s1000,s1017,s1034). Merge: sed the pf_agg.py glob (see agg200.py for the pattern).

### 21:25 UTC: trimmed vs untrimmed in (last action of this eval session)
- TEST ACP 7000 seeds 1000-1049 (eval/out/TEST_trim/; merge scratchpad aggtrim.py, events evtrim.py): trimmed 25/50 = 50 % (37-63) vs
  untrimmed 27/50 = 54 % (40-67); the untrimmed re-run reproduced the 19:00 physfloor solid result seed for seed (deterministic).
  Sections, trimmed / untrimmed: 4x4s 36/50 both; ramps 33/36 vs 30/36; stairs 27/33 vs 27/30; landing 25/27 vs 27/27.
  Shin 7 vs 15; knee on ramp-slab (wedge) contact 23 vs 36; trapped foot 0.5 s in 3 trials vs 0.1 s in 1; ankle-roll-past 42.7 s vs 47.3 s.
  The difference in success is noise; the snags at the slab sides are mostly gone. The trimmed floor stays the default (approved).
- ACP pair_s2_v3_007250 (fetched 21:19, check ALL PASS, planner md5 f4da33e8 = same planner as v2 7750): gate 4/5. Turn-around 3/3,
  ramp trough 1/2 (1019 stalled at the ramps back side, 10/22).

### 21:25-21:55 UTC: eval session 3; automatic scorer started; ACP 7000 200 on the final floor
- eval/autoscore/ (autoscore.py, config.json, README.md, compare.py, port/ = copy of the Hiking sim2sim port for hiking_check).
  No LLM. Polls the three nodes every 10 min; per track priority list, then the newest unscored export; fetch + check ->
  ACP gate (legs only on PASS) -> flat gate 4 x 50 -> legs 5 x 50 -> full route 200 only if every leg >= 45/50. Max 3 processes.
  Outputs eval/out/<track>/<ckpt>/auto/; rows in eval/autoscore/results.csv; dashboard row + prepare.py only for full routes.
- TEST (5 seeds per test/leg, root autotest): ACP v3 7250 gate 5/5 seeds and legs 25/25 seeds bit-identical to regline/ and
  legs_ours_solid/ (both trimmed floor); Hiking 4750 flat 20/20 identical. Hiking 4750 legs differed from legs_ours_solid
  (untrimmed floor); re-run with PHYS_FLOOR_TRIM=0: identical (6/6 compared). So the floor trim changes Hiking legs, the pipeline does not.
- Scorer started 21:46 (pid 44828): activated ACP pair_s2_v3_007750, SSR r7_v12_last (exported after solidq2 gave up; newest
  SSR export), Hiking x2_parkour_v4_6125; all three checks PASS. SSR priority: r7_v13_p250 (legs 1, 2, 5 first). Skipped: SSR
  r7_003800 and older (superseded lines).
- ACP pair_s2_v2_007000 full route 200, FINAL floor (solid, trimmed): 114/200 = 57 % (50-64). 65 falls, 21 stalls; failures
  4x4s 48, stairs 18, ramps 16, landing 4. Untrimmed 115/200, old floor 125/200. Dashboard row acp-s2v2-7000-ours-trim; prepare.py ok.
  Out: eval/out/acp/pair_s2_v2_007000/ours_trim_200/ (agg.txt).
- solidq3 finished 21:43: ACP v3 7250 legs (trimmed) 49/41/43/50/44, product 0.608, gate 4/5 (turn-around 3/3, 1019 stalled
  at the ramps) = PASS. Hiking 4125 apparatus 50 (trimmed): 0/50 (stalled at 4x4s 24, falls door 10, ramps 10, 4x4s 4).
- (a) Floor "defects" probed (scratchpad hole_probe.py, band_profile.py, output hole_probe.txt): NEITHER IS A HOLE IN THE PHYSICS.
  (3.10, 1.41): the solid floor there is at ground level like its surroundings; the raster carries a diagonal 4x4 (phys_beam_639)
  up to 3 cm wider than the real 0.089 m beam box (its 2 cm cell staircase, 89 mm = beam height), so the policy's scan shows the
  beam edge up to 3 cm early on that side. x 6.12, y 2.0-2.3: the raster has a one-cell (2 cm) V-groove between the ramp end
  (phys_ramp_633) and the stair side; the solid ramp top runs about 2 cm on over it. Not changed (no physics hole; a real gap of
  under 2 cm, if any, cannot catch a foot). So no regression run needed.

## 2026-11-13 ~05:40-06:10 UTC: review renders on each track's own training terrain (user request via coordinator)
- New: eval/train_terrains.py (ladder step 20): SSR tiles from the track's own generator (ramp/stairs/beam/platform, levels 5/10/15/19),
  Hiking paper types rebuilt as approximations (pyramid stairs up/down, platform up, slope, large box, small boxes, rough, gap; 3 levels),
  ACP box steps 15/25 cm (other ACP props = ladder step-1 courses). Registered in eval/ladder.py STEPS[20]. eval/render.py gained
  caption=, out_path=, crf= and font scaling with height (defaults unchanged).
- Runs: eval/out/renders_20261113/run_track.sh <track> (3 trials per course, 2 on extended-map tiles, strict, autoscore settings),
  outputs runs/<track>/; compose.py <track> makes <track>_review.mp4, <track>_sheet.png, <track>_clips.json (successes only).
- Checkpoints: ACP pair_s2_v3_010000 (10,000; 010250 exists, not used); SSR r9_v16_p300 (8,250); Hiking x2_parkour_v4_8375.
- What went wrong: SSR r9_v16_p300 hand-over check FAILS on depth only (depth median 0.089, ray_t max 3.7 m; ray origins/dirs,
  frame, gru, target, action all match). Rays identical, hit distances differ -> the check's rebuilt scene differs from v16's training
  scene (likely check-side, not the adapter). r8_v15_p100 passes. Rendered r9 anyway, flagged; SSR should look at obs_check for v16.

- Done ~07:00 UTC: reels eval/out/renders_20261113/{acp,ssr,hiking}_review.mp4 (71 / 70 / 69 s, 3.4 / 3.5 / 4.0 MB) and *_sheet.png;
  per-clip MP4s in <track>/, levels in <track>_clips.json. 3 trials per course (2 on extended tiles): previews, not results.
  SSR: ramp L10-19 3/3 (L5 down-then-up 0/3 stalled in the trough), stairs L5 only (L10+ 0/3, stalls), beam L5-19 3/3, platform L5 only
  (23 cm+ 0/3, stalls before it); extended tiles all but stairs-up. Hiking: rough L1-3, small boxes L1-2, large box / gap / stairs down
  L1 only; stairs up, platform up, slope up 0/3 (stalls one riser / slope short of the goal; approximated courses). ACP: every ladder
  skill at L3 incl. 4x4 from the slot, box 15 and 25 cm 2/3.
- Bugs found and fixed on the way: train_terrains <-> ladder circular import; solid floor (phys_floor.plan) crashes on planar ramps
  outside its 10-20 deg band (open: guard it); render.py frames resized to 368 by imageio (macro_block_size=1 now).

HOW TO RESUME (21:55 UTC, for the next eval session)
- Running: the automatic scorer, eval/autoscore/autoscore.py (pid in eval/autoscore/status.json; log eval/autoscore/autoscore.log).
  It needs no agent. Stop: touch eval/autoscore/STOP. Read results in eval/autoscore/results.csv and status.json. README there.
  Nothing else is running from the eval side; solidq3 and acp7000_trim200 (scratchpad) are DONE.
- Floor: solid + trimmed is the default (PHYS_FLOOR=heightfield old floor; PHYS_FLOOR_TRIM=0 untrimmed). Regression reference
  holds 10/10 on the trimmed floor.
- Headline: ACP 7000 114/200 (50-64 %) on the final floor. Best legs: ACP v3 7250 product 0.608 (trimmed); ACP v2 7000 0.589 and
  v2 7750 0.636 (both untrimmed).
- Open: SSR r7_v13_p250 is a priority in config.json; it is scored when its export appears. Renders are not made by the scorer.
  Hiking exports every ~125 iterations: the scorer takes only the newest when it is free.
- Heartbeat: the scorer writes /mnt/filesystem-g4/status/eval.json every 10 minutes.

### 13:10 UTC 2026-11-13: training-terrain job in the automatic scorer (user rule "Judge training progress on the training terrain")
- What changed: the scorer runs `train` first for every export: the track's courses in `eval/train_terrains.py` TRAIN_SETS (ladder
  step 20), 20 strict trials each, jobs of <= 6 courses (SSR 22 courses / 4 jobs, Hiking 32 / 6, ACP 17 / 3; ramp, slope and rough
  on PHYS_FLOOR=heightfield). Rows: `eval/autoscore/train_results.csv`; `results.csv` column `train_summary`;
  `agentic/dashboard/data/train_terrain.json` (not registered, not published). Hold-out gate: information only, blocks nothing.
- What went wrong and the fix: Hiking's rebuilt stairs-up and slope (review renders, hik_*_L1..L3) were not the training shape.
  Old stairs: 10 risers from r 0.5 m to the end waypoint; training (InstinctLab pyramid_stairs_inv, 8 m tile, 1 m border, step
  0.30 m, platform 2.5 m) has 6 risers from r 1.18 to 2.68 m, then flat. The policy stalled at x 3.09 = after 9 risers, before the
  10th. Old slope never flattened (to 4 m); training slopes from r 1.31 to 2.98 m then flat; the policy stalled at x 2.92-3.11.
  Fix: all nine Hiking types rebuilt from the generator code (rows 2/5/8 of 0-9). Smoke check, Hiking 9625, 2 trials: stairs-up
  row 2 1/2 (old L1 0/3), slope row 2 2/2 (old 10 deg 0/3). Course shape, not only the policy, caused those stalls.
- Exact copies: SSR generator tiles, extended-map tiles, Hiking stairs/platforms up/down (no tread noise), slope, gap (depth 0.5).
  Approximations: Hiking small boxes (mesh generator not ported), boxes and rough = one sampled layout of the same generator;
  all ACP courses (its clip prop scenes are not rebuilt; ladder step-1 obstacles and box steps stand in).
- ACP pair_s2_v4_012250 finished under the old code (legs 48/38/49/50/44) before the restart: no train rows; to score it, stop the
  scorer, delete its entry under `done` in state.json, restart.

HOW TO RESUME (13:10 UTC 2026-11-13)
- Running: automatic scorer (pid in eval/autoscore/status.json), restarted 13:08 with the train job; first: Hiking
  x2_parkour_v4_10375 train_1..6, then its leg5. Stop: touch eval/autoscore/STOP. README: eval/autoscore/README.md.
- Read: `grep TRAIN eval/autoscore/autoscore.log`, eval/autoscore/train_results.csv. Expected runtime per export on 3 cores
  (smoke check: 4-10 s per success, 15-20 s per stalled trial): Hiking about 40-70 min, SSR 20-40 min, ACP about 20-40 min.
- Open: register train_terrain.json on the dashboard (coordinator); ACP 12250 train rows (above).

### ~15:30 UTC 2026-11-13: Hiking on its own extended map, every route type (coordinator request for the user)
- Checkpoint x2_parkour_v4_11125 (newest export; no export at 11,250 or later existed at 14:05 UTC; hand-over check PASS). Terrain:
  the set Hiking trains on, data/extended_v4k (v4j + prop routes), copied to eval/data/terrains/extended_v4k. MuJoCo CPU, desktop.
- 56 route types (9 tile types x route kinds: stairs straight / side L-R / oblique / far straight / cross / L / C / S / descend /
  side down / amphitheatre; prop kinds on beams, beam fields, blocks, wedges, log corridors; homes, street front, tight spots).
  10 strict trials each = one per level 0-9 (ladder.tile(), export route, card command, solid trimmed floor). 560 trials.
- Result: strict 380/560 (68 %); falls 13; failures 180 of which 109 are the known END-RING artefact (stopped 0.20-0.45 m from the
  last station; the adapter's 0.4 m stop radius = training's, eval disc 0.20 m); 58 other stalls / not-a-stand / left course.
  Stumbles (no fall; harness side hit, non-foot body contact or pelvis tilt > 25 deg) in 433/560; clean strict 86/560.
  Side hits are mostly foot toe strikes on riser / beam / wedge faces in swing (stairs 17.9 per trial, 40 % move the pelvis
  >= 0.3 m/s or tilt it >= 5 deg). Side hits are only classified on stair / 4x4 / wedge faces: 0 on blocks, homes, street
  and tight spots means NOT MEASURED there (feet were checked for landings). Homes 0/10 (stuck at walls/counters, arm contact).
- Files: eval/out/renders_20261113/hiking_extmap_11125/ (table.txt, table.json, endring.json, frames_sheet_0..3.jpg, mp4/<type>.mp4,
  scripts). Scratch runs: session scratchpad extmap/out (per-trial results + traj).
- Open: the end-ring artefact keeps biting every Hiking score (harness disc 0.20 vs card 0.40 m): fix proposed again (adapter
  creeps in 0.20-0.40 m, or score Hiking's end with the card's 0.4 m). Side-hit classes for blocks/homes/street not defined.

HOW TO RESUME (15:30 UTC 2026-11-13)
- Nothing from this request is running. The automatic scorer is untouched (see the 13:10 block). Re-run: extmap.py in the folder
  above (python extmap.py run OUT <route types>, env as autoscore BASE_ENV + HIKING_CMD=card + HIKING_HANDOVER_DIR + PA_TERRAIN_DIR).

### 16:45 UTC 2026-11-13: final-station stop rule (user: "don't penalize heavily if it can't stop there"; docs/EVAL.md)
- What changed (CPU desktop only, nothing committed):
  - eval/harness.py: SHORT_R 0.40. Per trial `stopped_short` (not strict; ended stalled / time limit / end stand never held;
    all earlier waypoints and stations fine; last 1 s a strict stand within 0.4 m of the final station), `headline_success`,
    `end_dist_m` (also trials.csv). Summary: `headline_successes`, `headline_ci95`, `stopped_short`. Reasons and strict count
    unchanged. `main_count()` / `strict_count()` helpers. OMP/MKL/OPENBLAS threads default to 1 (also regression.py).
  - eval/adapters/hiking.py card: stop radius HIKING_STOP_R 0.20 (was card 0.4); stand if an overshot station is behind
    within 0.4 m; between 0.2 and 0.4 m forward >= 0.5 m/s (HIKING_RING_VMIN) and a stand after 6 s there (HIKING_RING_T).
    SSR and ACP adapters already command to the 0.20 m disc: unchanged.
  - eval/ladder.py (HEADLINE + STRICT columns; flat gate / skill profile on headline), eval/legs.py (headline, strict,
    stopped short, product and product_strict), eval/autoscore/autoscore.py (main columns = headline; new columns
    flat_strict, legs_strict, product_strict, route_strict, train_strict, stopped_short; train_results.csv strict_successes,
    stopped_short; route eligibility on headline; old rows readable, new columns blank). README and docs/EVAL.md updated.
- Regression: 10/10 OK, bit-identical, single thread, before and after the change. Thread count matters: multi-threaded,
  7/10 (3 baseline-adapter trials change). Now forced to 1 by default.
- Extended map, Hiking 11125, 560 trials (strict/headline): old adapter 380 / 474 (94 stopped short); new adapter 465 / 485
  (20 stopped short); falls 13 / 13. 4 old strict successes lost (3 now stopped short, 1 fall beam_fields side_left L9).
  Files: eval/out/renders_20261113/hiking_extmap_11125/finalstop_20261113/ (compare.json, scripts); per-trial results in the
  session scratchpad ba/full/{before,after}.
- What went wrong: (1) cleanup.py had deleted 11125's onnx files; re-fetched from the shared filesystem (read-only scp).
  (2) First new-adapter try (0.2 m stop only): the policy marks time in place at 0.2 m with a 0.4 m/s command (wedges L1) ->
  added the 0.5 m/s floor and the 6 s give-up. (3) Open: a target beside the robot at ~0.4 m (wedges.generator L1) still
  marches in place without turning: neither strict nor stopped short.
- Scorer: STOP 16:09, "autoscore exit" 16:17, one instance restarted 16:43 (pid in status.json). Old exports not re-scored.

HOW TO RESUME (16:45 UTC 2026-11-13)
- Running: automatic scorer only (new code: headline + strict columns from its next row). Stop: touch eval/autoscore/STOP.
- Next check: first new results.csv row has flat_strict / legs_strict filled; train_results.csv strict_successes filled.
- Open: wedges side-target marching (above); hiking_v4_11125 onnx will be removed again by cleanup.py (re-fetch if needed);
  old rows are strict-only (re-score an export by deleting its `done` entry in state.json if a headline is needed).

### 01:15 UTC 2026-11-14: Hiking on the wedge tiles of its own extended map (user: "evaluate hiking on the wedge routes that are part of the training first")
- Checkpoint x2_parkour_v4_14750 (newest export at 00:50 UTC; fetched read-only to the session scratchpad wedge/h14750, hand-over
  check PASS; 14625 was being scored by the automatic scorer, untouched). MuJoCo CPU, desktop, single-thread BLAS, current adapter
  (HIKING_STOP_R 0.20), strict + headline, time limit 120 s, solid trimmed floor.
- 900 trials: 9 wedge route kinds x levels 0-9 x 10 (trial i of a kind/level = i-th route of that kind at that level, seed 1000+i).
  Strict 623/900, headline 746/900 (123 stopped short), falls 55, stalls 181. Pinwheel tiles (2x2 cells) 326/537 strict, long ramps
  297/363. Best: cross 93, straight 89 strict /100; worst: generator 42 (35 stopped short), side_left 54, side_right 60.
- Apparatus legs 3/4 versus wedge tiles (per trial): arm contacts 100-170 (10375) / 59-69 (11750) vs 0.1; edge landings 135-141
  vs 14 per trial; edge landings per step 0.41-0.59 vs 0.19. The apparatus arm contacts are ALL on the side WALLS (events.csv
  class wall, 8631 leg3 / 5101 leg4 at 10375, z ~0.6 m); the wedge tiles have no walls (rails false). So the wedge tiles do NOT
  reproduce the apparatus failure: no walls, lower slopes (8-12 deg vs 15), larger cells (1.2-2.0 m vs 1.22).
- Files: eval/out/renders_20261113/hiking_wedges_14750/ (wedges.py, full.sh, analyse.py, renders.py, table.txt, table.json,
  mp4/). Per-trial results + traj: session scratchpad wedge/out/wedges/.
- What went wrong: renders.py mkdir without parents (fixed, re-run). Nothing else.
- Failure breakdown (900): falls 55; not-headline non-falls 99 = stalled at the last station within 0.6 m 45 (side-target turn
  fault: e.g. w_oblique_L6_1, 0.395 m from the target, yaw command -1 rad/s held 13 s, yaw stays -142 deg, 80 edge landings from
  stepping in place), left the course 41, stalled mid-route 8, end stand not held 5. The table's "stall" column includes the 123
  stopped-short (headline) trials.
- Renders (pinwheel = closest to the apparatus 2x2 ramp pinwheel; oblique = the diagonal crossing of leg 4): mp4/success/
  (w_oblique_L8_5, strict, 10 side hits, 23 edge landings) and mp4/stall/ (w_oblique_L6_1), chase + legs each, sheet.jpg per
  folder. Every still checked: robot visible, magenta pole floating clear of the wedges, side hits shown match results.json.
- What went wrong: eval/render.py's default footer credits "Bones Studio" motion data for every track; wrong for Hiking (paper
  motions retargeted to X2, CC BY-NC 4.0). Re-rendered with RENDER_CREDIT set. Earlier Hiking renders that used the default
  carry the wrong footer: fix = per-track default credit in render.py (not changed here; coordinator to decide).

HOW TO RESUME (01:35 UTC 2026-11-14)
- Nothing from this request is running; the automatic scorer was not touched. Re-run: full.sh SCRATCH HANDOVER_DIR, then
  analyse.py SCRATCH/out, renders.py SCRATCH/out test:tag (set RENDER_CREDIT for Hiking).
- Open: Hiking footer credit default in render.py; side-target turn fault at the last station (45/900 wedge trials).

## Scorer chunking (2026-11-14, 04:52 UTC restart)
User request: break the scorer's work into small pieces for all three tracks. Done (no commit):
- `eval/ladder.py --chunk k --chunk-size 5` and `eval/legs.py --legs K --chunk k`: run seeds 1000+5k..1004+5k only, into
  `<course|legK>/chunks/c<k>.json` (saved after every trial) + `c<k>.done`. The usual unchunked command merges chunk files (and old
  partial.json) and writes summary/results exactly as before; it then deletes `chunks/`.
- `eval/autoscore/autoscore.py`: jobs `train.<course>.c<k>` / `train.<course>` (merge) and `leg<K>.c<k>` / `leg<K>`; a `regress`
  job runs the regression check once per export, single-threaded, before the chunks (chunks use EVAL_REGRESSION=0). Chunk attempt
  2,700 s, wall 1,800 s. Config `chunk_trials` 5 (`train_job_size` unused). Flat tests, ACP gate, route not chunked (route = 4 jobs).
  Backups of the pre-chunk files: `%TEMP%/autoscore.py.pre_chunk`, `%TEMP%/ladder.py`, `%TEMP%/legs.py` (desktop temp, not repo).
- Identity check (scratchpad chunk_identity.py): SSR r7_v17_12000 ssr_platform_L10 20 trials, 4 chunks vs unchunked: 20/20 trial
  records byte-identical (15/20 both), also identical to the scorer's earlier run; ACP pair_s2_v5_016750 leg 3, 10 chunks vs
  unchunked: 50/50 identical (48/50), also identical to the earlier run.
- Speed-up: SSR course 244 s unchunked -> 93 s in 4 chunks (2.6x; bounded by the slowest chunk, which holds the stalls); ACP leg
  140 s -> 28 s (5x, 10 chunks). A 26-min SSR v18 course should take about 7-10 min on 4 cores.
- Went wrong: one edit put a raw newline in a string in ladder.py for a few minutes (syntax error); fixed before any scorer run
  used it, the identity test's processes had already imported the good copy. The old scorer was drained with STOP (exit 04:51:53).
- Regression check after the change: OK 10/10, 10 bit-identical (single-threaded).

HOW TO RESUME (04:55 UTC 2026-11-14)
- Running: one scorer, pid in eval/autoscore/status.json (started 04:52:18 UTC with nohup; check `ps -ef | grep [a]utoscore.py`
  shows one line). Active: ssr:r7_v18_016000, hiking:x2_parkour_v4_15500, acp:pair_s2_v5_016750 (route_1150 left).
- Next check: autoscore.log for `end ... failed` on chunk jobs and that merge jobs (`train.<course>`, `leg<K>`) end `done`;
  train_results.csv rows for SSR v18 once its 22 courses are merged.
- Open: flat tests (50 trials, one process each) are not chunked yet; chunk them the same way if they become the bottleneck.

## Showcase renders with overlays (2026-11-14/15; user: successes on the apparatus and the extended map for all three, plus failures)
- Folder: eval/out/renders_20261114/showcase/ (capture.py, extsel.py, jobs.py, show.py, render_all.py, ba.py; runs/, mp4/<job>/,
  beforeafter/). Every apparatus trial is a re-run of the scored seed with capture hooks: 12/12 bit-identical qpos to the scored
  trajectory (show.py writes mp4/<job>/check.json). Extended-map trials are new (seed 1000, PHYS_FLOOR=heightfield, 90 s).
- Code changed (no commit): eval/x2.py build(dress=...) visual-only hook; eval/dress.py (apparatus from its own MJCF and
  materials, tiles per primitive class, venue dark blue checker floor, softer light: tracking spot 0.18, headlight 0.38/0.25);
  eval/overlays.py (ACPOverlay: planner 17x11 map tiles, tracker 9x7 dots, side obstacles red / 0.10 m margin pink, planned ring vs
  physics stance square; DepthOverlay: RealSense-style 424x240 87x58 deg chest-camera depth + the policy's actual input thumbnail);
  eval/render.py (app / overlay / dress / speed args; dressed walls see-through or hidden as before; legend line). Old look:
  RENDER_DRESS=0. Pre-change render.py copy: %TEMP%/render.py.pre_dress.
- Went wrong: ACP 016250 hand-over had lost planner_attn.onnx to cleanup.py; re-fetched read-only from the shared filesystem
  (md5 8d4dc438 = autoscore_fetch.json). Robot MJCF tracking spotlight blew out the floor under the robot; dimmed in dress.py.
- Extended-map picks (strict successes): ACP stairs.C_3, wedges.oblique_7, log_corridors.generator_4 (ACP stalled on wedges.C_5);
  Hiking and SSR stairs.C_3, wedges.C_5, log_corridors.generator_4 (SSR stalled on stairs.L_3, not a stand on homes_3).

HOW TO RESUME (2026-11-15)
- Running: automatic scorer only (untouched). Showcase renders: python render_all.py 12 in the showcase folder (DONE line in
  mp4/render_all.log). Re-render one: show.py <job> chase [legs] with X2_EXCLUDE_PAIRS=hip (+PHYS_FLOOR=heightfield for ext jobs).
- Open: coordinator to decide whether dressed rendering stays the default for all renders (it is now: RENDER_DRESS=1).
- 2026-11-15 update: all 21 MP4 jobs rendered and every frame sheet checked (mp4/<job>/sheet_<view>.jpg). Legs view: raised dressed
  geoms and overlay obstacle blocks between camera and legs are now faded (rails and red blocks hid the feet in the first pass);
  depth inset notes "black: no return (< 0.3 m or > 6 m)" (own forearm at 0.10-0.15 m, room beyond 6 m). Two ACP legs MP4s
  re-encoded at crf 29 (7.5-7.8 MB). Open: ACP ext stairs.C_3 shows a 4515 N stair-side hit on the heightfield floor (check
  plausibility before quoting); hiking dressed-scene renders made earlier than this batch keep the old look.

### 2026-11-15: extended-map batch 2 (user: "i asked for renders from the extended map ... these include the setup for the home")
- Went wrong: showcase batch 1 rendered only stairs.C / wedges / log corridors on the extended map, no homes (homes_3 was captured
  but not rendered). Fix: batch 2 per track (ACP pair_s3_v2_022500, Hiking x2_parkour_v4_22000, SSR r7_v20_final_26000), courses
  jobs.py EXT2: homes 0-9 (routes 0/1/2 at several levels), tight_spots 0-5, street_front 0-5, stairs side_left/side_right/L/S,
  single_beams C/S, beam_fields, blocks generator/L. Seed 1000, heightfield floor, 90 s. Running: python jobs.py 14 _ext_
  (runs/jobs2.log, ALL DONE at the end). Then render one chase MP4 per route type (best success, else typical failure).
  jobs.py / show.py backups: *.bak_1115.
- 2026-11-15 results (seed 1000, one trial per course, strict): ACP 22500 34/39 courses (homes 8/10: route 1 1/3 - L1 ok, L5/L8
  fall on the big block; street_front 5/6; tight 6/6; stairs side/L/S 8/8; beams 3/3; beam fields 2/2; blocks 3/4).
  Hiking 22000 31/39 strict (homes 7/10: route 1 0/3 - L1/L8 stall on a small block with turn command not followed, L5 left
  the course; street 5/6; tight 6/6; stairs 6/8; single beams 2/3; blocks 4/4). 17 chase MP4s in mp4/<job>/ (render2.py,
  1-4 MB each), every sheet checked.
- Went wrong: SSR r7_v20_final_26000 hand-over lost policy.onnx to eval/autoscore/cleanup.py (folder mtime 21:28 on 2026-11-14)
  although it is SSR's best training-terrain checkpoint (18/22) - the "best checkpoint keeps everything" rule did not protect
  it. All 30 SSR batch-2 captures FAILED (NO_SUCHFILE). Not re-fetched: this session is desktop-only. Needs a read-only
  re-fetch from the shared filesystem (autoscore_fetch.json lists the files), then: python jobs.py 14 ssr26000_ext_, then
  add ssr26000 jobs to render2.py.
- Render defects seen: homes chase first frame (t 0.02 s) shows the robot behind a solid tall block (only walls are faded);
  Hiking stairs.L_2 side hit 5104 N peak on the heightfield floor (same family as the 4515 N ACP stairs hit - check before
  quoting); ACP blocks.generator_3 route goes between blocks (no block stepped on).

HOW TO RESUME (2026-11-15, batch 2)
- Nothing running. Outcomes: runs/<tk>_ext_<test>/extmap/<test>/results.json. Renders: mp4/<job>/<job>_chase.mp4 + sheet_chase.jpg.
- Open: SSR re-fetch + capture + render (above); cleanup.py protection of the best checkpoint (coordinator/autoscore owner).

### 2026-11-15: README GIFs with minimal captions (user: "cut all the boilerplate text ... one or two liner")
- eval/render.py RENDER_MINIMAL=1 (default off): one caption line on a thin top strip, no footer, no side-hit banners, no legend
  (in-scene markers and ACP map tiles stay); DepthOverlay (eval/overlays.py) draws only the depth view, a "depth" label and the
  0.5/1/2/3 m bar (policy-input thumbnail dropped). RENDER_CHASE_DIST / RENDER_CHASE_ELEV override the chase camera (default
  3.2 / -22). Pre-change copies: %TEMP%/render.py.pre_minimal, overlays.py.pre_minimal.
- Script: eval/out/renders_20261114/showcase/mini.py (window-only chase MP4 in mp4/mini/, then GIF fps 10, 480 wide, 128-colour
  palette, bayer dither, into docs/media/).
- Went wrong: the "L-/C-shaped stairs" clips were straight staircases (stairs.L / stairs.C are approach shapes). Replaced with
  staircases whose own shape is L / U (stairs.json variant_kind): Hiking x_stairs__C_4 (stairs_L4_v2, L) t 3.5-16.5 s and SSR
  x_stairs__S_7 (stairs_L7_v5, U) t 5.5-20.5 s, both strict successes (seed 1000, heightfield floor, 90 s), camera 4.6 m / -34 deg.
  Hiking 22000's policy is no longer on the desktop (nodes down): the Hiking clip uses x2_parkour_v4_26500 (jobs.py hik26500; all
  8 L/U stair courses strict successes, one trial each). SSR 26000 batch 3 (same 8 courses): runs/ssr26000_ext_x_stairs__*.
- GIFs are larger than before: 5.4-6.1 MB for four, 8.7-9.0 MB for the two 13-15 s staircase clips.
