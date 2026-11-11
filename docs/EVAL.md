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
- Also reported: side hits per class (foot, of which in swing; shin or knee; scuffs), per trial and per section; summed and
  peak impulse; peak horizontal force.
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
