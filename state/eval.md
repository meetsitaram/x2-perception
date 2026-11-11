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
