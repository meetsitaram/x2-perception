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
- What each track must hand over is written in `state/SHARED_LESSONS.md` ("What the eval needs from each track"): the policy
  as npz or ONNX, an adapter card, and a 50-step `obs_check.npz` the adapter must reproduce.
- The desktop's Python has mujoco, numpy and onnxruntime, no torch. Installing torch would be a large download (needs the
  user's okay), so policies come as npz or ONNX.
- No address or login is in any eval file; `eval/node.sh` reads them from the git-ignored `config/nodes.local.json`.
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
