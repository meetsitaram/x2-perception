# The common eval: apparatus walks on the X2

One definition for all three tracks, run by the eval monitor. Version of 2026-11-11 (phase 1: values filled in from the data;
the user's answers of 2026-11-11 are in). The code is `eval/` in this repo; every constant below is a named constant in
`eval/harness.py` and is written into each result file.

## Robot and simulators

- **Robot:** AgiBot X2 Ultra, the same model files for every track (`/mnt/filesystem-g4/data/robots/x2_ultra`; on the desktop
  `ssr-x2/assets/robot`). 31 joints, 43.5 kg.
- **Simulators:** each track's training simulator, plus MuJoCo on CPU as the common cross-check (`eval/harness.py`). A track's
  number counts only if it holds in MuJoCo too. MuJoCo set-up: time step 2.5 ms, implicitfast, PD inside the simulator with
  the gains the policy was trained with (default: the X2 deploy plant, `eval/config/x2_plant.json`), the robot's own collision
  geometry against a 2 cm height field and the wall boxes. The 12 sole spheres per foot are 2 cm in radius instead of the
  MJCF's 5 mm, same lowest point (on a height field the small ones get tilted contact normals on flat ground).
- **Solid floor is the default (coordinator's decision, 2026-11-12 19:00 UTC):** the feet collide with boxes (floor, steps, tilted
  ramp slabs, 4x4s; `eval/phys_floor.py`), the height field is drawn and scanned only. Why: the height field trapped feet (audit,
  `docs/findings/sim_artifact_audit.md`); ACP 7,000 seeds 1000-1049: solid 27/50 vs old 26/50, trapped-foot 0.1 s vs 8.3 s, flat x4 50/50.
  Old floor for controls: `PHYS_FLOOR=heightfield`; results from it are marked "old floor". Regression reference re-recorded on it.
  Ramp slabs trimmed (coordinator, 2026-11-12 ~21:15): each slab side and end sits where the raster crosses half-way (they stood
  1-2 cm proud of the scanned edge); `PHYS_FLOOR_TRIM=0` gives the untrimmed solid floor, results from it are marked "untrimmed".
- **Per-trial flags (measurement v2, `eval/harness.py`):** trapped-foot s, limb-clamp s, joint-limit s (any hinge more than
  **0.15 rad** past its range; 0.05 until 2026-11-12 21:10 UTC, when it fired in nearly every trial), and **ankle-roll-past s** (either
  ankle roll more than 0.05 rad past its +-0.262 rad range) as its own number (coordinator's decision, 2026-11-12). The last 1 s of a fall is excluded.

## Courses (held out of training)

The final result is the walk on both humanoid-challenge apparatuses. Neither goes into training as a whole. Parts (a 4x4
stepover, a crossing-ramp cell, a stair flight, a rail, a brace stand) may be copied into a track's training terrain as
separate, re-arranged, randomised pieces; each track lists in its state file which parts it copied. The full layouts and
these routes are for this eval only. Files and frames: the README in each folder.

| | Ours (`/mnt/filesystem-g4/data/apparatus/ours`) | NIST (`/mnt/filesystem-g4/data/apparatus/nist`) |
|---|---|---|
| Geometry | our CAD: `cad_20mm_walk.npz` + 344 wall boxes | organisers' CAD: `nist_20mm_walk.npz` + 1,920 wall boxes (the open door leaf is one) |
| Frame | metres, x along the lane, deck top z = 0 | metres, lane along -x, deck top z = 0.14 |
| Route | `route.json`: 22 waypoints, 19.2 m, 5 stations (buttons 1-5) | `route.json`: 35 waypoints, 22.5 m, 11 stations |
| 4x4 stepovers | front and back 4x4 (0.79 and 0.78 m long, 8.9 cm tall), a 0.37 m gap between their ends | upper and lower 4x4 of the V behind the door (1.33 and 1.36 m, 10 cm above the floor there) |
| Door | hinge (2.374, 1.651), leaf open 90 degrees toward the start, doorway 0.87 m clear | hinge (9.756, 3.765), leaf open 94.5 degrees toward the start, doorway 0.85 m clear |
| Sections | door, 4x4s, ramps, wall, stairs, landing | door, 4x4s, ramps, counter, stairs, landing |
| Stairs | 6 risers, 0.186 m, landing at 1.12 m | 5 risers, 0.20-0.21 m, landing 1.02 m above the deck |
| Start | (0.762, 0.762), facing +x | (11.90, 4.21), facing -x |

- **Ours, route:** the mobility route of `loco-manip-apparatus/sim/x2_route.py` (door -> front 4x4 -> button 1 -> back over it ->
  back 4x4 -> button 2 -> ramps -> button 3 -> ramps diagonal -> button 4 -> ramps -> round the wall -> stairs -> landing,
  turn left -> button 5 at the landing's far end). Five waypoints were moved after the checks: the door and the slot between
  the door wall and the front 4x4 (1 to 3 inches, wall clearance), and the two either side of the back 4x4, so the leg
  between them crosses the beam 0.27 m from its end instead of 0.14 m. Listed in the folder's README.
- **NIST, route:** the station route `NIST_ROUTE` of `loco-manip-apparatus/sim/pipeline/routepass.py` (keypad, door, bulbs,
  button stand, valves, counter twice, crusher, stairs, landing) with three additions asked for by the user on 2026-11-11.
  That source route was built to reach the manipulation stations, so it left the V of 4x4s over its lower beam only and
  stopped at the top of the stairs. Added: (1) the pocket behind the upper 4x4, the mobility trial's first button (rules:
  "walk through the open doorway and then turn right to press the first button, turn around and proceed to press the
  second ... step over two lengths of 4x4"): in over the upper beam, a 1 s station at (8.88, 5.00), and back out; (2) a
  square crossing of the lower beam, replacing two waypoints that sat 0.14 m from it; (3) the landing walked to its far end
  ("turn left, walk to and press the button"): a turn at (1.80, 4.965) and the last station at (1.84, 3.35).
- **Both 4x4s must be stepped over (user, 2026-11-11).** The route has a waypoint either side of each beam, with a straight leg
  that crosses it at 62 to 90 degrees and at least 0.27 m from its ends (ours: front beam twice, back beam once; NIST: upper
  beam twice, lower beam once). The eval also checks it in every trial: the pelvis path must cross each beam's centre line
  within the beam's length at least once. A trial that reaches the last waypoint without that fails as "went round a 4x4".
  On ours the gap between the two beams' ends makes going round possible; on NIST the pocket and the V are closed by walls,
  so the beams are the only way.
- **The door is open (user, 2026-11-11).** On both, the leaf stands open against the start side and is a wall box in the
  physics; the doorway is clear over 0.87 m (ours) and 0.85 m (NIST); the route passes it 0.35 m or more from either jamb and
  0.43 m (ours) / 0.50 m (NIST) from the leaf. `check_routes.py` fails if the clear width is under 0.60 m (robot width plus 5 cm
  a side). The slanted line outside each start is the start gate, standing open, off the route.
- **Route check (`eval/check_routes.py`, run 2026-11-11, both PASS):** every 2 cm of both routes, the floor never steps more than
  0.25 m along the centre line or 0.10 m to either side, and no wall, post, rail or brace box that overlaps the body (0.30 to
  1.30 m above the ground) is closer to the centre line than 0.25 m, the X2's half width. Tightest clearance: 0.25 m (ours,
  beside the door wall), 0.28 m (NIST). No arrival disc reaches a beam (nearest 0.21 m). The circle test ignores which way the
  robot faces. The X2's hands are 0.28 m from its centre line with the arms at the model's zero pose, so in the tightest spots
  a hand can still touch; that is counted (below), not excluded.
- **NIST, a low panel:** a sloping panel hangs over the corner of the pocket nearest the door wall (x 9.26 to 9.68, y above
  4.75), its lower edge 1.32 m above the deck, about the X2's standing height. The pocket leg and station are kept 0.26 m or
  more to the side of it.

## A trial

- One robot, one attempt at the whole route, from the start pose. The policy gets commands produced from the waypoints only: the
  current waypoint (position, and at a station the heading), and the harness's standard velocity command toward it (forward
  speed up to the track's stated limit, yaw rate up to 1 rad/s, turning on the spot when the waypoint is more than 60 degrees
  off the heading, zero at a station while holding). No reference motion replay and no privileged map: a policy may use what
  its deployed version would have on the robot (proprioception, the command, its depth camera or an on-robot height scan).
  Each track states what its policy observes; the adapter records it in every result file.
- **A waypoint is reached** when the pelvis is within **0.20 m** of it horizontally and within 0.30 m of standing height above
  its ground. At a **station** it must stay inside that disc for **1.0 s** without a break. Waypoints are taken in order. (The
  2026-11-02 ACP runs used 0.5 m; that would count a 4x4 as crossed with the pelvis still over it.)
- **Success:** every waypoint reached within the time limit, both 4x4s stepped over (above), no fall, never off the course,
  and still up 1.0 s after the last waypoint. (In the 2026-11-02 runs a robot that tipped after the last waypoint still counted.)
- **Fall:** the pelvis less than 0.35 m above the lowest ground under the pelvis and feet, or the pelvis tilted more than
  60 degrees, or the trunk or head on the ground.
- **Off the course:** the pelvis more than 1.0 m from the route line, or off the raster.
- **Time limit: 180 s** (confirmed by the user, 2026-11-11). The 2026-11-02 baseline finished in 54 s at best, 59 s
  median, 120 s at worst. A trial also ends, as "stalled", after 45 s without a new waypoint.
- Outcomes reported: success, fall, left the course, time limit, stalled, went round a 4x4; and the section each failure stopped in.

## Size, seeds, randomisation

- At least **200 trials** per checkpoint and apparatus for a tracked number; **500** before saying anything about 99 %; the final
  claim uses **1,000**. Every success count is given with its trial count and a 95 % Wilson interval (0 failures in 200 still
  only shows more than 98.1 %; in 1,000, more than 99.6 %).
- **Seeds:** trial i uses seed 1000 + i, the same for every track, apparatus and checkpoint. The seed fixes all of the below.
- **Randomised protocol (the headline number):** start position +-0.10 m, start heading +-15 degrees; friction 0.5 to 1.0
  (floor, walls and robot); every link's mass scaled 0.95 to 1.05 and 0 to 3 kg added to the torso; a push every 6 to 10 s, a
  velocity change of 0.2 to 0.5 m/s on the pelvis in a random horizontal direction.
- **Nominal protocol (reported beside it):** start +-0.03 m and +-3 degrees, friction 1.0, nominal mass, no pushes. This is the
  closer one to the 2026-11-02 baseline runs.

## What is reported per checkpoint and apparatus

1. Three headline numbers per checkpoint, track and apparatus: **success rate** (count, trials, 95 % interval), **clean
   success rate** and **mean stumble penalty** (both defined under "Side hits and the stumble penalty"); and the time of
   the successes (fastest, median, slowest).
2. Per section: how many trials entered it and how many completed it; where the failures stopped; by outcome.
3. Foot-contact quality, from every contact of every physics step (the feet are always checked; a result says so in
   `contacts.feet_checked`):
   - **side contacts of a foot** (`foot_side`): a foot contact over 5 N against a wall box, or a horizontal-dominant one
     (surface steeper than 60 degrees) where the raster has a face of more than 3 cm within 4 cm; counted as events
     (contacts closer than 0.2 s on the same foot and obstacle class are one), split into swing-foot and stance-foot. The
     ones on a 4x4, a wedge or a stair are the side hits scored below;
   - **edge landings:** at each touchdown (vertical load rising past 40 N after being under 10 N), the ground under the sole
     (22.5 x 10 cm, 21 points) departs more than 3 cm from a plane; reported per landing;
   - **knee, shin and thigh contacts** with the floor or a wall, as events; **arm, trunk and head contacts** with walls too
     (a trunk or head contact with the floor is a fall).
   Totals, per trial, and per section.
4. Renders for the user's review: a chase view and a side-on legs close-up of the best (fastest success), the median success and
   a failed trial (the median one by waypoints reached). From MuJoCo's own camera on the recorded states (and from the training
   simulator's camera for the track's own number). Walls are see-through in the chase view and not drawn in the close-up, so
   nothing hides the robot; the target is a magenta pole floating 0.9 to 1.6 m above the ground, never at foot height; every
   frame is labelled with simulator, track, checkpoint, apparatus and seed and carries the simulator's foot loads and pelvis
   height. The eval monitor looks at every render and checks it against the run data before passing it on.

## Side hits and the stumble penalty (user, 2026-11-11)

"Add negative score for hitting on the sides of the 4x4 as well as the side of the wedges and stairs which will cause
stumbles." The success rate is unchanged by this; two numbers are added beside it.

- **A side contact** is a contact of a foot with a near-vertical face of the floor: its normal is horizontal-dominant
  (|normal z| under 0.5, a surface steeper than 60 degrees) and the raster has a face there (heights within 4 cm of the
  contact point span more than 3 cm). A shin or knee contact with the floor where the raster has such a face is counted the
  same way, whatever its normal. A sole flat on a tread, on a beam top or on a ramp's sloped surface has a near-vertical
  normal and is not a side contact; an edge landing stays its own category.
- **Its class** comes from where it is (`route.json`): **4x4** within 0.12 m of a beam's centre line; **wedge** inside the
  crossing-ramp area (the faces between ramp cells and at their ends, not the 15 degree walking surfaces); **stair** inside
  the stair flight (risers and nosing fronts). Faces elsewhere (door threshold, a brace's foot) are "other": recorded, not
  scored.
- **One event per contact episode:** contacts of the same limb, side and class less than 0.2 s apart are one event. For each
  event: time, place, limb, class, peak horizontal force (N), horizontal impulse (the horizontal contact force summed over the
  episode, N s), whether the foot was in swing (carrying under 40 N or moving faster than 0.3 m/s), and the disturbance it
  caused: the largest change of the pelvis' horizontal velocity and of its tilt within 0.3 s of the contact starting.
- **Hit or scuff:** an event is a **side hit** when its peak horizontal force reaches 20 N or its impulse 0.5 N s; below
  both it is a **scuff**.
- **Penalty of one event:** scuff 0.02. Hit: `0.1 + 0.4 * min(1, impulse / 10 N s) + 0.5 * min(1, max(dv / 0.30 m/s,
  dtilt / 10 deg))`, so 0.1 for a light touch that moves nothing, up to 1.0 for a hard stub that jolts the body (10 N s
  is a 0.23 m/s change of the whole robot's velocity). If a push of the randomised protocol falls in the 0.3 s window the
  disturbance term is left out.
- **Stumble penalty of a trial** = the sum over its events (no cap). **Mean stumble penalty** = its mean over all trials of
  the checkpoint, failed ones included; also given per section (events are booked to the section of the waypoint the robot
  was heading to) and per class.
- **Clean success** = a success with no side hit (foot, shin or knee; scuffs allowed) on any 4x4, wedge or stair. **Clean
  success rate** = clean successes / trials, with its 95 % Wilson interval.
- Also reported: side hits per class (foot, of which in swing; shin or knee; scuffs), per trial and per section.
- **Force and impulse figures are not quoted for now (coordinator, 2026-11-11):** hits are reported as counts by class. On the
  height field a foot pinched over an edge, or pinned between a raster face and a wall box, reads contact forces of several
  kN that are not a net push on the body; until that contact is looked at, peak force, impulse and the impulse term of the
  penalty are logged (events.csv) but not used in reports, and the stumble penalty is to be read as a weighted hit count.
- **In the renders:** each side hit is a ball at the contact point with a thin post above it (red 4x4, yellow wedge, cyan
  stair) for at least 0.8 s, and a red banner with the limb, force, impulse, disturbance and penalty; a scuff is a small
  grey ball.
- **What the geometry can resolve.** The floor is a 2 cm height field, so a vertical face is a ramp one cell wide: an 8.9 cm
  4x4 side is 77 degrees, a stair riser 84 degrees, the 14 to 16 cm faces between wedge cells 82 degrees. Pressing a 2 cm
  sphere into each (`eval/selftest.py`, "normals") gives |normal z| 0.00 to 0.22 on the 4x4 side, 0.11 on a riser, 0.13 to
  0.19 on a wedge face, and 1.00, 1.00 and 0.97 on the beam top, a tread and a ramp surface: the classes separate cleanly, so
  the 4x4s, wedges and stairs are not added as separate box geoms. Limits: a face is placed to within one cell (2 cm); a
  face lower than about 4 cm (under 63 degrees on this raster) is not seen as a side; a toe rolling over a top corner gives
  normals in between and counts only while the normal is horizontal-dominant; overhanging nosings do not exist in a height
  field (the CAD stairs have none).
- **Verified (`eval/selftest.py`, all pass):** the robot set down moving toes-first at 0.6 m/s into a 4x4 side, a wedge side
  face and a stair riser: each logs foot hits in its own class only (e.g. riser: 314 and 377 N peak, 21 and 13 N s, penalty
  1.0 each). Set down on a 4x4's top, a ramp's slope, a stair tread and flat deck: no hit and no scuff. Not yet verified
  with a walking policy: the tests place and slide the robot, they do not swing a leg.

## Logs of every eval run (user, 2026-11-11: "enough logs ... so we have more transparency")

Each run folder on the shared filesystem (`/mnt/filesystem-g4/eval/<track>/<checkpoint>/<apparatus>_<protocol>/`) holds, beside
`summary.json` (the table row, with every constant of the definition) and `results.json` (everything, per trial):

**`trials.csv`, one row per trial**

| Column | Meaning |
|---|---|
| `seed` | the trial's seed (1000 + i); fixes start pose, friction, mass, payload, pushes |
| `outcome` | `success`, `fall: ...`, `left the course`, `time limit`, `stalled`, or `went round a 4x4: ...` |
| `success`, `clean` | 1 / 0; clean = success with no side hit on a 4x4, wedge or stair |
| `time_s`, `sim_time_s` | time the last waypoint was reached (or the trial ended); simulated time in total |
| `waypoints`, `of`, `last_waypoint` | waypoints reached, out of how many, and the name of the last one reached |
| `stopped_heading_to`, `stopped_section` | for a failure: the waypoint it was heading to and that waypoint's section |
| `sections_completed` | the sections whose waypoints were all reached |
| `beams_crossed` | per 4x4, how many times the pelvis path crossed its centre line within its length |
| `side_hits`, `side_hits_4x4`, `side_hits_wedge`, `side_hits_stair`, `scuffs` | counted events (foot, shin or knee) in total and per class; events under the hit threshold |
| `stumble_penalty` | sum of the trial's event penalties |
| `summed_impulse_Ns`, `peak_impulse_Ns`, `peak_force_N` | horizontal impulse over all side hits; the largest single one; the largest horizontal force |
| `landings`, `edge_landings` | touchdowns counted, and those on an edge |
| `foot_side_all` | every foot-side contact event, walls and unscored faces included |
| `shin`, `knee`, `thigh`, `arm`, `trunk`, `head` | contact events of those parts with the floor or a wall |
| `start_dx`, `start_dy`, `start_dyaw_deg`, `friction`, `mass_scale`, `torso_payload_kg` | the trial's randomisation |
| `final_x`, `final_y`, `final_pelvis_z` | where the pelvis was when the trial ended |

**`events.csv`, one row per contact event** (every side hit and scuff, every other foot-side, shin, knee, thigh, arm, trunk and
head contact, and edge landings as kind `edge_landing`; sorted by time within a trial)

| Column | Meaning |
|---|---|
| `seed`, `t_s`, `dur_s` | trial, start of the contact episode, its length |
| `kind`, `side`, `on` | `foot_side`, `shin`, `knee`, `thigh`, `arm`, `trunk`, `head`, `edge_landing`; left / right; `terrain` or `wall` |
| `class` | `4x4`, `wedge`, `stair`, `other`, `wall`, or empty when the floor has no face there |
| `hit`, `penalty` | 1 = side hit, 0 = scuff, empty = not a scored event; its penalty |
| `peak_horizontal_N`, `impulse_Ns`, `peak_normal_N` | peak horizontal force, horizontal impulse, peak normal force of the episode |
| `normal_z`, `in_swing`, `foot_speed` | for foot contacts: |normal z| at the start, whether the foot was in swing, its horizontal speed (m/s) |
| `pelvis_dv`, `pelvis_dtilt_deg`, `push_in_window` | largest change of the pelvis' horizontal velocity and tilt within 0.3 s; 1 if a protocol push fell in that window |
| `x`, `y`, `z` | contact point |
| `heading_to_waypoint`, `section` | the waypoint the robot was heading to, and its section |

The tracks also report their papers' own evaluations (each paper's success criteria as the paper defines them). Those are
track reports; they are kept apart from these common apparatus numbers and never mixed into one table with them.

## Baseline

The user's "around 25 %" matches one written result: **13 of 50 (26 %)**, ACP stage-3 policy `acp3_v2` (checkpoint 1000, planner
`planner_v2_hist1`), our apparatus, full mobility route (the same 22 waypoints before the five moves above), MuJoCo Warp,
2026-11-02 (`gwbc-v6/scripts/acp/README.md` line 25, `ISSUES_LOG.md` line 326). Its definition was looser than this one: 0.5 m
arrival radius, no stand at the end (12 of the 13 tipped after the last waypoint), no randomisation written down, no time
limit beyond the clip length; 95 % interval for 13 / 50 is 16 to 40 %. That policy is a SuperMimic tracker, which this round
does not reuse. No NIST success rate has ever been measured. So under this definition the baseline is **not yet measured**;
it will be once an adapter for `acp3_v2` exists (it needs the gwbc-supermimic code on a GPU node or a CPU port).

## Answered by the user (2026-11-11)

- Time limit: 180 s per trial.
- Button presses do not count: a station is a 1 s stand inside its 0.20 m disc; the heading there is recorded, not required.
- Both 4x4s have to be stepped over; the door has to be open; the NIST route visits the pocket behind the upper 4x4 and ends
  at the landing's far end.

No open questions at the moment.

## Eval ladder (user, 2026-11-11)

"Eval shouldn't go straight to the apparatus for evals, and instead do simpler tests on our terrain map with some of the
routes we reviewed earlier." Every checkpoint is scored bottom-up. Steps 1 and 2 are **on the training map**: they say what a
policy can do per skill, they are never the apparatus result, and their tables and renders are labelled so.

**Step 1: one skill per test** (`eval/ladder.py`, `STEP1`; 50 trials each, nominal protocol, 60 s limit, same 0.20 m arrival,
stumble score and logs as above). Flat tests are on a flat floor; the others on fixed tiles of the export
`/mnt/filesystem-g4/data/terrains/extended_v1` (floor raster + the boxes that are not in the raster), levels 3 to 4 of 9.

| Test | What the robot has to do | Course |
|---|---|---|
| `flat_walk` | walk 4 m straight (2 waypoints) | flat |
| `flat_stop_stand` | walk 2 m, stop and stand 1 s, walk 2 m, stop and stand 1 s | flat |
| `flat_turn_in_place` | on the spot: turn to face back and stand 1 s, turn to face ahead and stand 1 s (the station counts only within 20 degrees of the heading) | flat |
| `flat_turn_and_return` | walk 1.5 m, then back to the start (a 180 degree turn) | flat |
| `flat_backward` | 1.5 m backward, facing ahead | flat |
| `flat_sideways` | 1 m to the left, then 2 m to the right, facing ahead | flat |
| `log_single` | step over one level log (8 cm), square, from 0.9 m before to 0.8 m past it | `single_beams_L4_v1` |
| `log_slanted` | the same over a log with one end raised (14 cm at the top; a box, not in the raster) | `single_beams_L3_v0` |
| `wedges` | 4.9 m across a pinwheel of four wedges (11 to 13 degrees, crest 0.14 to 0.17 m) | `wedges_L3_v1`, route 0 |
| `stairs_up`, `stairs_down` | a straight flight of 6 steps (rise 0.14 m, run 0.31 m) with walls and handrails: from the floor up to the middle of the landing; from there down the far flight to the floor | `stairs_L3_v0`, route 0, first 3.0 m / the rest |
| `log_corridor_walls`, `log_corridor_rails` | 6.9 m along a 1.3 m corridor over four logs, between walls / between rails on posts | `log_corridors_L3_v0`, `_v1`, route 0 |

Tile routes are kept 0.6 m inside the tile (0.3 m for the start) and end at a 1 s station, because the export's routes end as
close as 9 cm to the tile's rim, where one more step is off the course. For backward and sideways the waypoint carries the heading to keep; the standard velocity command then has a backward or
sideways component, and goal-driven policies get the target direction with a zero heading change. A log on the route has to
be stepped over (pelvis path across its centre line), as on the apparatus.

**Step 2: reviewed routes on the map**: a fixed, named subset of the routes the user reviewed (loops with turns, routes
through the main walk area, a dead end with a turn in place), unblocked ones only (every route in the export passed the route
check). The list (`eval/ladder.py` `STEP2`, pictures in `eval/out/step2_proposal/`):
`street_out_and_back` (15.5 m), `street_doorway_out_and_back` (13.3 m, through a doorway), `corridor_walls_out_and_back` and
`corridor_rails_out_and_back` (13.8 m, four logs each way), `wedge_field_loop` (9.4 m round a pinwheel), `stairs_U_turn` (7.4 m,
up a U stair to 1.8 m), `amphitheatre_loop` (10.7 m round curved steps), `home_dead_end_and_back` (14.5 m, through a doorway into
a room and back). The export has no closed loops through the home or street tiles and its corridor and room routes are 5 to
8 m, so those are walked out and back: a 1 s station at the far end, a turn on the spot, and the same way home. All eight
pass the route check (0.25 m corridor, 0.25 m step); tightest wall clearance 0.27 m.

**Step 3: the two held-out apparatuses** (everything above this section): the result against the 99 % goal.

**Gates and the skill profile (user and coordinator, 2026-11-11).** The user: 50 of 50 on the four flat walking tests is
"fair"; on one pass mark for every obstacle: "not sure how you evaluate which obstacle is easy to learn". So:

1. **The flat tests are the only hard gate:** 50 of 50 on `flat_walk`, `flat_stop_stand`, `flat_turn_in_place` and
   `flat_turn_and_return`. Backward and sideways are reported, not gated.
2. **Each obstacle skill is its own ladder of three levels on held-out courses** (`eval/ladder_synth.py`, built here, not
   tiles of the training export), 50 trials each; a level is passed at 45 of 50 and levels count in order:

   | Skill | Level 1 | Level 2 | Level 3 (our apparatus's dimensions, re-arranged) |
   |---|---|---|---|
   | `log` | 5 cm beam, square | 8.9 cm beam, 20 degrees skew | 8.9 cm 4x4 reached along a 0.51 m slot behind a wall, a 90 degree turn to cross it, a post beside the far side |
   | `wedge` | 8 degree hill, crest 0.10 m | 12 degree hill, crest 0.15 m | 2 x 2 pinwheel of 1.22 m cells, 15 degrees, crest 0.16 m, crossed over the 16 cm faces between cells |
   | `stairs_up` | 4 steps of 10 cm, run 30 cm | 5 steps of 14 cm, run 28 cm | 6 steps of 18.6 cm, run 22 cm, 1.1 m wide between walls |
   | `stairs_down` | the same flights, from the landing down | | |
   | `corridor` | 1.3 m wide, 5 m long | 1.0 m wide, two 5 cm logs | 0.87 m wide (the doorway's width), two 8.9 cm logs at 15 degrees |

   The report is a **profile**: the highest level passed per skill, with the success and clean-success counts of all three
   levels, and the weakest skill named (`<run>/step1.txt` and `.json`).
3. **Step 2 routes are tagged with the skills and levels they need;** a checkpoint runs the routes whose skills it has
   passed, so a policy good at stairs but not at wedges still gets its stair routes scored. The eight routes (approved by the
   coordinator 2026-11-11) and their tags (`eval/ladder.py`, `STEP2_NEEDS`): `street_out_and_back` wedge 1; `street_doorway_out_and_back`
   wedge 1 and corridor 3; `corridor_walls_out_and_back` and `corridor_rails_out_and_back` log 2 and corridor 1; `wedge_field_loop`
   wedge 3; `stairs_U_turn` stairs_up 2; `amphitheatre_loop` stairs_up 2 and stairs_down 2; `home_dead_end_and_back` corridor 3.
4. **The apparatuses (step 3)** are run when every skill is passed at level 3, and in any case for the best checkpoint of
   each track once a day, so there is always a current apparatus number.
5. **Difficulty is measured, not assumed:** `eval/out/profile_table.*` keeps success per skill and level for every checkpoint
   and track scored, so the order in which each method learns the obstacles shows; each report names the track's weakest skill.

The baseline pair is run through the whole profile as the reference. The first version of step 1 (one tile of the training
export per skill, the table further up) is kept as `step1_v0` for continuity.

**Known limits of steps 1 and 2 as built:** the step-1 courses are built by hand to the dimensions above, one course per
skill and level (not a random family of them); step 2 uses tiles of the training export. Meshes
that are not in the raster (cylinders, stones, bushes) become their bounding boxes. One tile per skill: a test says "this
log", not "logs in general".
Found 2026-11-12: such boxes are in the physics but not in the walk raster, so neither the height scan nor the route check's step
test sees them, and the wall test starts 0.30 m above the ground. `street_doorway_out_and_back` has a block, a rock and a plant on
its centre line (all baseline trials fall at the block); its rows are marked COURSE DEFECT and are not results until the route is
fixed (`eval/ladder.py` `STEP2_DEFECT`). The other seven routes have no box on their path.
**Fixed the same day:** (a) the route check (`check_routes.py`, `check_floor_obstacles`) tests every physics box that stands on the
floor and is not a 4x4, at every height from 5 cm up to the 0.30 m where the wall band starts. A leg that passes closer than
0.18 m (the legs' half width: feet 0.10 m either side plus half a sole and 2 cm) is blocked; closer than 0.25 m is printed as
"near". (b) On step-2 courses the height scan reads a scan raster: the walk raster raised to the top of every such box
(`ladder.py` `add_scan_raster`, `adapters/base.py`); the physics height field stays the walk raster. Every summary records
`scan`. (c) `street_doorway_out_and_back` detours south of the block: (-1.6, 1.0), (-0.6, 1.0) replace the export point (-1.5, 1.5),
13.6 m. All eight routes pass. Step-2 rows run before the fix used the walk raster only for the scan (their `scan` field is
missing). On the apparatuses the scan is still the walk raster: our route's leg 10 passes 0.16 m from the foot of a brace at
(3.61, 1.48) that the raster does not hold (blocked by (a); legs 7-11 "near"). It is the held-out route, so it is left to the
coordinator.

**Plant (user, 2026-11-11):** a policy is always run under the X2 plant it was trained with, taken from its adapter card (name
and per-joint kp, kd); the harness default is only for test fixtures. Every summary, table and render label states the plant.
A hand-over whose card does not state its plant and gains is refused. The 2 November baseline pair runs under
`vendor_20260823` (its checkpoint's own setting; the exported gains equal that plant's exactly).


## Strict stand (coordinator's decision of 2026-11-12, implementing the user's "a station is a 1 s stand"; confirmed by the user 2026-11-12)

Found by the eval monitor: under the disc-only check a robot that walks slowly through a station's 0.20 m disc, or is airborne
or stepping off the landing during the end stand, was scored as standing (our apparatus, ACP 4250, seed 1035: both feet off
the deck with 0 N load at the instant it was scored a success). The rule is tightened to mean what the user's rule says:

- **A station counts** when, for 1 s without a break: the pelvis is inside the 0.20 m disc (and within 0.30 m of standing
  height), the two feet together carry at least **50 % of the robot's weight** (so never both at 0 N), and the pelvis'
  horizontal speed stays **under 0.15 m/s**. The 1 s counts only while all three hold; the robot keeps the station command
  until it has stood (so entering the disc at walking speed and then stopping is fine).
- **The end stand** (1 s after the last waypoint): the same three conditions for the whole 1 s (when the last waypoint is
  itself a station; on ladder tests that end without a station only "upright" is checked), and the robot still upright
  **2 s after** the end stand (the trial runs on with a stand command). The other stations need no extra check: any fall
  later in the trial fails it anyway.
- **Reported:** the strict-stand success is the **main number** (`strict_successes`, `STAND_RULE=strict`, the harness default
  from 2026-11-12). The old disc-only count (`STAND_RULE=disc`: stations advance on 1 s inside the disc, success = still up
  1 s after the last waypoint) is reported beside it for every apparatus result, from its own run with the same seeds, so
  earlier results stay comparable. A disc run also checks the strict conditions on the window that counted
  (`strict_success` there is a lower bound, since a loose station is not retried). Every table says which rule each number uses;
  results made before 2026-11-12 09:00 UTC are disc-only.
- **Travel after a station** (the "does not stop" measure, in every apparatus and ladder output): the pelvis path length in
  the 1 s after each station counted (`station_travel_1s` in the summary: per station, how many trials reached it, median,
  maximum, and how many moved more than 0.2 m; `station_travel_1s_m` in trials.csv, `!` marks a station that failed the
  strict check in a disc run).
- Constants `STRICT_LOAD`, `STRICT_SPEED`, `STRICT_POST_END`, `TRAVEL_AFTER`, `STAND_RULE` in `eval/harness.py`.
- Course decisions the same day: the question about button 5's distance from the landing's y edge is withdrawn (the end-stand
  falls are at the west edge, x 9.74, after the robot fails to stop); the route is unchanged. The landing deck is in the
  physics twice (walk raster and box `wall_202`, top 1.118 m); it stays, labelled: with the box out of contact
  (`APP_NOCONTACT_WALLS=202`, a test option) ACP 4250 scored 13/50 against 17/50 with no systematic change.
- **End stand with retry (coordinator's decision of 2026-11-12, 11:50 UTC; confirmed by the user 2026-11-12: "i think its fine to take corrective steps, but the 1s should count from after the step").** The end stand is treated
  like a station: it counts once a continuous 1 s strict stand (in the last station's disc, feet carrying at least 50 % of the
  weight, pelvis under 0.15 m/s) is reached at the last station at any time before the trial's time limit (and within 45 s of the
  last waypoint counting), followed by 2 s upright; a re-balancing step restarts the 1 s, it does not fail the trial. Reason:
  stations already work that way, and the user's rule is "a 1 s stand", not "2 s without a correction". Harness: `END_RETRY=1`
  (made the default once the s2_v2 5750 200 + 200 run that started before it has finished); every result records `end_retry`
  and the number of corrections (`end_breaks`). Results before this decision used the end stand without retry.

## Ladder courses padded with flat floor (eval monitor, coordinator's decision of 2026-11-12)

Every ladder course (steps 1 and 2) was an 8 m height field that ended in nothing. A depth camera reads the missing ground as a
drop about 1 m ahead of a station near the course end, which a policy trained on large terrain never sees on flat ground. Found on
SSR r7 2250 (corridor level 1: 0/50, stops 0.21 m short of the last station, 1.1 m from the course edge; 119 of 1,296 depth pixels
read "nothing"). TEST with an endless ground plane under the course: SSR corridor L1 0/50 -> 20/20 (standard and floor40), Hiking
3000 street 37/50 -> 10/10. Fix from 2026-11-12 ~16:40 UTC: every ladder course is padded with flat floor 3 m beyond its edge (the
median height of the border cells), in the physics height field and in the scans / depth, which all read the same grid
(`eval/ladder.py`, `PAD_M`, `LADDER_PAD_M=0` reproduces the old courses). Each ladder summary records `course_padding_m`; rows run
before the fix carry no such field and are marked "before the edge fix" in reports. ACP's height scan was not affected (scan lookups
clamp to the last row of the raster, which continues the floor). The two apparatuses are walled rooms and are unchanged.

## Station legs (user, 2026-11-12; being implemented by the eval monitor)

The user: "instead of just the full route success rate, we should spawn, with some randomness, the robot at each manipulation and
see if it can make up to the next station successfully. This gives better view of the progress." Beside the full route, every
scored checkpoint is run on each leg of an apparatus on its own: start to button 1, button 1 to 2, 2 to 3, 3 to 4, 4 to 5 (the
full route's waypoints between those stations). The robot is spawned standing at the leg's first station with a random position
inside the 0.20 m disc and a heading within 20 degrees of the station's, gets 1 s of stand command, and must reach the next
station and hold the strict stand there. 50 trials per leg, seeds 1000-1049. Reported per leg with falls, stalls and where; the
product of the leg rates is shown beside the full-route rate. The full route remains the result against the 99 % goal. The eval
monitor fills in the final spawn rule and constants here when the first runs are done.

**Order (user, 2026-11-12):** "we should do the full route only after the model runs pass all of the individual stations." The
full route is run only when every leg passes (45 of 50 on each leg, the coordinator's mark, to be confirmed by the user);
otherwise the checkpoint is reported by its leg scores with "full route not run".

## Station legs (user, 2026-11-12)

The user: "instead of just the full route success rate, we should spawn, with some randomness, the robot at each manipulation and see
if it can make up to the next station successfully. This gives better view of the progress." and "May be we should do the full route
only after the model runs pass all of the individual stations." Implemented in `eval/legs.py` (constants at its top).

- **Legs (ours):** start -> button 1, button 1 -> 2, 2 -> 3, 3 -> 4, 4 -> 5, over the full route's own waypoints between those
  stations (NIST: its stations likewise). A 4x4 crossed by a leg's own path has to be stepped over.
- **Spawn:** leg 1 as a full trial (route start, nominal jitter). Legs 2-5: standing at the leg's first station, position uniform in
  its 0.20 m arrival disc, heading = the station heading +- 20 deg, drawn per seed (seeds 1000-1049); a draw is rejected and redrawn
  (up to 200 times) if the ground under the soles is not one plane (residual over a 0.26 x 0.40 m sole area > 2 cm; buttons 3 and 4
  stand on the 15 degree ramps, so "level" means no edge or step under the feet) or a wall / post box is within 0.25 m of the pelvis.
  Share of clean draws at the stations: button 1 51 %, button 2 31 %, button 3 37 %, button 4 66 % (the rest are redrawn). The rest of
  the nominal protocol is unchanged. Before the first waypoint the robot gets 1 s of stand command at its spawn pose (target where it
  stands, its spawn heading, hold), so every adapter's history fills with real standing frames: ACP's proprio history and its
  planner input (history, goal = stand, scan), SSR's 5-frame history and GRU (zero at reset, then 1 s of standing), Hiking's 8-step
  proprio and depth histories. Adapters reset exactly as at a route start.
- **Success:** the leg's last station counts under the strict rule (1 s stand, corrective steps allowed, the 1 s counted after the
  step); for the last leg the end stand as in a full trial. Falls, stalls (45 s) and leaving the course as in a full trial. Time limit
  per leg: 180 s x leg length / route length, at least 60 s, plus the 1 s stand.
- **Report:** per leg successes / 50 with the 95 % interval, falls, stalls and where; the product of the five leg rates beside the
  full-route rate.
- **Order per checkpoint (user):** flat gate and gait numbers, level 1, routes, then the five station legs; the full route (50, then
  200 for a candidate) only when every leg passes. Pass mark (coordinator's recommendation, to be confirmed by the user): 45 of 50 on
  each leg. A checkpoint that fails a leg is reported by its leg scores with "full route not run: leg X at N/50". Exceptions: ACP
  7000's 200 started before this rule finishes; an ACP pair gets 50 full-route trials if ACP's stage-3 rule needs the number.

## Final-station stop (user, 2026-11-13; agentic/docs/USER_RULES.md "Final-station stop")
Every adapter commands toward the final station until within 0.2 m. A trial ending in a stable stand within 0.4 m of the final station
(outside 0.2 m) is a success flagged "stopped short"; reports give the headline count (0.2 m + stopped short) and the strict 0.2 m count.

Implementation (eval monitor, 2026-11-13, live in the scorer from 16:43 UTC):
- `eval/harness.py` `SHORT_R = 0.40`. A trial is `stopped_short` when it is not a strict success, its outcome is "stalled", "time
  limit" or "not a stand: end stand never held", every waypoint before the last was reached, no earlier station failed the strict
  stand, no 4x4 was gone round, and its last 1 s (END_STAND) was a continuous strict stand (feet >= 50 % of the weight, pelvis
  < 0.15 m/s, standing height within Z_TOL, heading within the station's tolerance if it has one) with the pelvis within 0.4 m
  of the final station. Only for routes whose last waypoint is a station. The outcome (reason) and the strict count do not change.
  Per trial: `stopped_short`, `headline_success`, `end_dist_m` (also trials.csv). Per summary: `headline_successes`,
  `headline_ci95`, `stopped_short` beside `strict_successes`.
- Main number: the headline count (ladder tables and flat gate / skill profile, legs and their product, the scorer's flat, legs,
  route, train columns and route-eligibility). The strict 0.2 m count is reported beside it: ladder column STRICT, legs.txt
  "strict", results.csv `flat_strict`, `legs_strict`, `product_strict`, `route_strict`, `train_strict`, `stopped_short`;
  train_results.csv `strict_successes`, `stopped_short`. Results scored before the rule have no headline: their main number
  is the strict count (blank new columns).
- Adapters: Hiking's card command stands within `HIKING_STOP_R` = 0.20 m of a station (was the card's 0.4 m), and also stands
  when an overshot station is behind it within 0.4 m (no turn on the spot). Between 0.20 and 0.4 m it commands at least
  0.5 m/s forward (`HIKING_RING_VMIN`; the policy marks time in place at 0.4 m/s) and stands after 6 s there without reaching
  the disc (`HIKING_RING_T`). `HIKING_STOP_R=0.4` restores the old rule. SSR (harness command, floor40, keep) and ACP (goal
  bounded at goal_r toward the waypoint) already command until the 0.20 m disc: unchanged.
- Thread count: the eval runs single-threaded maths (OMP / MKL / OPENBLAS_NUM_THREADS = 1). A multi-threaded regression run
  changed 3 of the 10 trials (all baseline-adapter); harness.py and regression.py now default these to 1 before numpy loads.
