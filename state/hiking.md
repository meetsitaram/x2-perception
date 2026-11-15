# hiking state

Updated 2026-11-12 01:05 UTC (Hiking track agent, second agent, resumed at 00:53 UTC). Node `hiking`; address and login are in
`config/nodes.local.json`.

## Hand-over 2026-11-12 (newest first)

- 22:30 **v4k built, CPU pre-flight PASS, armed at model_6750 (tmux `cont4k`, about 23:11); the last planned restart for >= 2,000 it.**
  - Stall check (eval of 4750, `eval/out/hiking/x2_parkour_v4_4750/`): turn_in_place: all 38 failures ended with the heading
    within 1.4 deg of the goal (adapter yaw 0.04 rad/s) but 0.23-0.39 m off the station: inside the adapter's 0.4 m stop radius
    (so it commands a stand) and outside the harness's 0.20 m arrival disc. Cause = drift while turning (training turn/planar_speed
    0.20 m/s) plus the 0.4 / 0.20 m radius mismatch in the eval (eval monitor's to fix: inside 0.20-0.40 m the adapter should
    creep back). Not a slow-yaw stall. Legs 3 / 4: stalls at buttons 3 / 4 with vx 0 and yaw 0.1-0.5 rad/s (stand turn under the
    trained 0.4 rad/s cost threshold) = slow-yaw dead zone confirmed there. Leg 5: 18 stalls at vx 0.7 below the stairs (other cause).
  - Route count "364" is not a defect: routes on the 100 instantiated cells (v4i 196, v4j 343 in training); v4k adds prop routes.
  - v4k contents: option v (`slow.py.v4k`: forward cap U(0.1, 0.45) for 25%, |yaw| cap U(0.1, 0.4) for 40%, `turn_effort`
    replaces `dont_turn_wait`, `turn_drift` -2.0, metrics slow_fwd_ratio / slow_turn_ratio / turn_drift); VX_BACK=0 (no backward
    commands; backward earns nothing); terrain `data/extended_v4k` = v4j + extended_v1_1_prop_routes (single_beams 29->268,
    beam_fields 30->268, blocks 30->240, wedges 99->484, log_corridors 80->124 routes; spawn filter re-applied); motion
    `data/motion_v4` = v2 + t4x4u_y20 sideways + mir (18), stairs v1_1 turning + mir (24; 113941 fixed in v1_1, class straight,
    not selected), 2x2 v1_1 + mir (18), driveway v1_1 + mir (12): 827 s, 52% new, 0 joint-limit frames. Graded contact,
    sidestep, inset stairs, spawn fix kept. staged.py allows v (backup `_trash/2026-11-12/staged.py.before_v_*`).
  - Scripts: `code/x2_port/continue_v4k.sh`, `v4j/precheck_v4k.py` (PASS, log `v4j/precheck_v4k.log`), `v4j/append_motion_v4.py`,
    `v4j/make_terrain_v4k.py`. Trial bars: level 0.8 x, episode 0.4 x, turn rate 0.5 x; fall-back `fallback_v4j.sh` + guard4j.
  - Resume keeps Adam moments, discriminator + its optimizer and normalizers; the adaptive LR restarts at 1e-5 and re-adapts in
    2-5 iterations. Recovery after the last three restarts (5000, 5500, 6000): level back within 3-5 iterations (level keeper);
    episode length back to 95% in 80 / 119 / 76 iterations (all robots reset at a start, so 30-40 of those are mechanical).

- 20:18 **v4j restart failed at 19:22 (cause found), fixed, re-armed at model_6000 (fifth agent).**
  - What went wrong: the 19:22 construction check died in `staged.py` `__post_init__`: `AssertionError: unknown HIKING_X2_OPTS:
    ['g']` (allowed set was "abcdefrstkyzow"). The env never got built, so no "[X2 routes]" line and no reward table: hence
    "routes none". `test_v4j.py` exec's only the .v4j files, never the staged.py option gate (needs Isaac), so the CPU build
    passed. The fall-back worked (v4i resumed from model_5500 at 19:22:47) but left v4i without a crash guard.
  - Fix: staged.py allows `g` (backup `_trash/2026-11-12/staged.py.before_g_*`); new CPU pre-flight `code/x2_port/v4j/precheck_v4j.py`
    (opts letters vs staged.py, compiles staged.py and the .v4j files, terrain / motion paths and npz from the launcher's env,
    v3 frame count, test_v4j.py) PASS; negative test with the old staged.py reports the `g` failure. Isaac cannot be imported
    without Kit (`pxr` missing), so the full construction check stays a GPU step in the pause.
  - `continue_v4j.sh` now runs the pre-flight after the checkpoint appears and BEFORE stopping v4i (fail: v4i not stopped,
    line in launch.txt); the construction-fail path now starts `guard4i`. Backup `_trash/2026-11-12/continue_v4j.sh.before_precheck_*`.
  - Motion v3 rebuilt without `tp_stairs_113941` (retarget artefact: frozen right hip yaw 2.38 rad, pelvis pitched 46-52 deg):
    12 turning stair takes, 6,065 added frames (121 s), total 521 s, 0 joint-limit frames, seam speed max 1.29 m/s (was 1.14).
    `append_motion_v3.py` has `V3_EXCLUDE` (default 113941); MJCF `data/robots/x2/mjcf/x2_ultra.xml` (x2_ultra/ lacks meshes).
    Old v3 in `_trash/2026-11-12/motion_v3_with_113941_201407/`. No `_v1_1` corrected sets existed at 20:14; v3 uses no 2x2 or
    driveway takes; no `*_mir*` sets added.
  - Running: v4i (5,798 at 20:18); `guard4i` (L0 5.80, E0 878); `cont4j` = `continue_v4j.sh 6000` (model_6000 about 20:55).
  - User rule 2026-11-12 (no backward steps; learn turn in place), code read 20:25, v4j unchanged:
    backward: per-column ranges are floored at VX_MIN 0.45 (staged.py 318-320), routes walk forward (v4j adds sideways);
    only the random-velocity columns (perlin_rough_stand / _velocity, `cmd.ranges.lin_vel_x = (-0.4, 0.8)`, extended.py 189)
    can command backward. Nothing penalises backward motion beyond the velocity-tracking term.
    turn in place: option y (keep.py) turns 15% of non-stand robots on every column (walls, tight spots, homes, slopes) for 4-6 s
    on the spot; train.log turn/share 0.066, rate_along_cmd 0.90 at cmd 0.90 rad/s, BUT planar_speed 0.20 m/s while turning
    (drifts, not in place). No turner is placed on a side-sloped wedge or against a wall on purpose.
    Plan for the restart after v4j: lin_vel_x floor 0 on the random-velocity columns (no backward commands); a drift penalty
    for turners (planar speed > 0.05 m/s); a share of turners spawned on the wedge / ramp tiles (16 deg side slope) and within
    0.5 m of a wall; log turn/planar_speed and rate per tile group.
  - 21:12 **v4j training** (pre-flight pass 20:52, check 20:57: 364 routes, trial 21:09: level 5.73 vs 5.73, episode 453 vs 886
    (0.51 x), turn 0.87 vs 0.92). guard4j running. To check: 364 routes vs about 741 stair routes expected (sampled cells?).
  - Eval model_4750 turn_in_place 12/50 (4125: 32/50): 37 stalls (20 turning to face back, 17 turning to face ahead), 1 time
    limit. Training does not show it: v4i turn/rate_along_cmd per 100 iterations 0.82-0.97 from 4,600 to 5,900 (dips only at
    restarts), planar drift 0.19-0.22 m/s. The training metric measures rate while commanded, not finishing a 180 deg turn.
  - Decision 21:12: turn changes go in v4k at model_6500 (about 22:45), not into v4j now: v4j needs about 500 iterations
    for a clean read of contact / spawn metrics, and a restart costs about 17 min of GPU. v4k = the plan above plus turners
    with targets behind (150-180 deg, as in the eval) and a turn-completion log (share reaching 10 deg in 6 s). Build and CPU
    test on the next resume.

- 19:06 **v4j built, CPU tests all pass, armed: `continue_v4j.sh 5500` in tmux `cont4j` (model_5500 about 19:25 node clock).**
  - Files: `code/x2_port/v4j/` (contact.py, make_v4j_files.py, make_terrain_v4j.py, append_motion_v3.py, test_v4j.py, logs);
    `files_instinctlab/.../x2/{contact,extended,routes,keep}.py.v4j` (live files untouched until the restart; v4i's are backed up
    to `_trash/<date>/v4i_files_<time>/` by the script). Option letter g; sidestep and terrain / motion by env vars (script header).
  - Contact (g): `wall_contact_graded` replaces `undesired_contacts`: trunk (pelvis, waist, torso, shoulders, head) and legs
    1 + min(F/100 N, 3) per body; elbows / wrists free under 30 N and 0.5 s, then 0.25 x; feet stay with stumble / edge terms.
    `base_contact` ends only for a torso force > 50 N with the vertical part dominant. Logs: end/<cause>_<group>,
    end/reached_<group>, c10/<feet|legs|trunk|arms>_<group> (> 10 x weight), c50/... (horizontal > 50 N), tunnel/<group>
    (body origin > 2 cm inside a prop box: pelvis, torso, knees, elbows, wrists, feet), wallspeed/near|far_<group>.
  - Sidestep: 30% of route robots on tight-spot / home tiles walk the route sideways (facing +-90 deg, |v| <= 0.3 m/s);
    every route robot gets vy = clip(lateral offset, +-0.2) for a waypoint within 1 m.
  - Motion v3 (`tracks/hiking/data/motion_v3/`): v2 (400 s) + 9 sideways Pico takes (32 s, lateral share 0.73 as chained) +
    13 turning stair takes (C, S, down-then-turn, side approach; in_set and checks ok), 528 s total; 0 joint-limit frames
    added; pelvis z p50 0.70 vs v2 0.675; seam speed max 1.14 m/s. The 13 straight stair takes not added (v2 has stairs).
  - Terrain (`tracks/hiking/data/extended_v4j/`): stairs v0-v5 from the inset set at the extended_v1 rows (draws unchanged),
    741 stair routes (700 inset by kind + 41 amphitheatre); extended_v1_1 not loaded. Spawn fix: route starts within 0.40 m
    (planar) of a box rising into 0.25-1.5 m above the floor are moved forward along the route to the first point with
    0.40 m; if none, kept at the widest point when >= 0.30 m (tight spots: 10 routes), else dropped (tight spots 4 of 40).
    Spawns near props dropped. Homes starts under 0.34 m: 46.5% -> 0% (54 of 101 moved); tight spots 50% -> 11% (one
    sampled layout). Causes on homes: 'block' furniture > 1 m tall (30) and walls (17); tight spots: low walls (20); the
    wedges hits were mesh bounding boxes over ramps (meshes now left out).
  - Restart path: STOP at 5,500, old crash guard killed, construction check 256 robots (needs routes >= 100 and
    wall_contact_graded in the reward table), trial 31 iterations (level 0.8 x, episode 0.4 x, turn 0.7 x), else
    `fallback_v4i.sh` (restores files and levels, resumes v4i); new crash guard in tmux guard4j (fall-back v4i).
  - Not verified on CPU: anything needing Isaac (the termination / sensor APIs, the box table on the real layout, log
    keys); the spawn-contact figure in Isaac comes from `artefact/spawn_ext_homes` after the restart (was 23.5%).

- 18:55 **Wall contact, route clearance, stairs inset, artefacts, lean (third agent; numbers from `metrics.jsonl`, v4i at 5,313).**
  - What a wall / prop touch does in v4i (code read): episode ends only on `base_contact` = `torso_link` over 1 N (any
    direction, wall or ground), plus `root_height` and `bad_orientation`; the fall penalty (f) is 0 now (schedule ended at
    1,500); `stalled` (d) is off since 1,500 (log 0.0). Every other body over 1 N, feet excluded, pays `undesired_contacts`
    -1 per body per step: a hand brush costs the same as a hip or shin. Feet against a wall: `feet_stumble` -2 (horizontal
    force > 3 x vertical, foot moving). Pelvis contact does not end the episode.
  - So SSR's refusal mechanism (any touch ends the episode) is absent here except for the torso. Evidence of no refusal so
    far: levels since 5,000 rose on tight spots 5.41 -> 6.74 and homes 5.74 -> 6.39 (log corridors 4.57 -> 6.45; stairs fell
    5.48 -> 5.04). Episode ends (all robots, 5,313): time-out 90.0%, root_height 5.5%, bad_orientation 2.6%, base_contact
    2.0%, flat since 4,900. Speed ratio (actual / commanded, all robots) 0.42-0.53, noisy, no trend (0.45 at 4,900, 0.415).
  - Not logged (gap): episode ends and route arrivals per tile type, wall distance, speed near walls, which body touches.
    `route/final_reached` (all route robots) 0.09-0.47, `route/waypoint_index` 0.35-0.93: noisy, rising.
  - Defects seen in the artefact log: (1) tight spots: 8.6% of robots carry a contact force over 10 x body weight (4.1 kN)
    at a step, up from 2.8% at 5,000 (open tiles 0-0.02%); body unknown (max over all bodies, feet included): pressing into
    walls or seam / penetration spikes. Homes 3.3%. (2) spawn contact: share of robots with a non-foot contact over 50 N in
    their first 0.2 s: homes 23.5%, blocks 8.3%, tight spots 7.1%, paper 1.5%: spawns (route starts, possibly with the
    reference-state arm pose) inside props. (3) sunk feet (>1 cm below ground in contact): wedges 16% (ramp faces vs the
    flat ground scanner: likely the measure, not a sink), street 3.7%, others <1%. Running max force 148 kN on paper columns.
  - Lean: torso pitch (Isaac, commanded forward) flat -16.7, stairs -14.1, boxes -14.3, gaps -13.7 deg; waist pitch -17.9
    (at its -18.0 limit) everywhere. Unchanged.
  - Decision: one restart, v4j, at the first 250-checkpoint after the CPU build passes (earliest model_6000, about 21:00
    node clock; 5.4 it/min). Contents, each listed as a deviation:
    1. `wall_contact_graded` replacing `undesired_contacts`: horizontal-force contacts by group. Torso, pelvis, hips,
       thighs, knees, shins: strong (about -2 per step, scaled by force up to 4x at 200 N). Feet: `feet_stumble` stays.
       Elbows, wrists, hands: free under 30 N and under 0.5 s continuous, then -0.2 per step.
    2. `base_contact` ends the episode only for a torso force that is mostly vertical and over 50 N (lying on something);
       a torso against a wall becomes the strong penalty. Falls stay covered by root_height and bad_orientation.
    3. Logs per tile type: episode end causes, route arrival share, contact share and 10x-weight share per body group,
       route-robot speed within 0.5 m of an obstacle-mask cell vs elsewhere (stall near walls).
    4. Spawn fix: route robots get no reference-state start, and a start is skipped (other route) if the tile's mask is
       within 0.40 m of it; check the homes spawn share falls under 3%.
    5. Stairs: `extended.py` loads stair tiles v0-v5 from `extended_v1_2_stairs_inset/stairs.{npz,json}` (same tile ids,
       same arrays, routes with `kind`), amphitheatre v6-v7 from `extended_v1`; `extended_v1_1_stair_routes` not loaded
       (its 526 routes are for the replaced tile versions). Loader needs: the merge by `tile_id`, start z from the route's
       `z` for routes that start on a landing (descend, side_down), and `kind` carried to the route table for a per-kind
       arrival log. Route format is identical (`start`, `waypoints`, `target`, `z`), checked.
    6. Sidestep (vy on route robots in tight spots and homes): only together with the sideways Pico clips in the motion
       reference, otherwise the style reward fights it; find the X2 sideways retargets at the build; if not ready, the
       next restart.
    Path: CPU tests of each term on recorded tensors, construction check with 256 robots in the restart pause, trial at 31
    iterations (level >= 0.8 x, episode >= 0.4 x), fall-back v4i, crash guard.

- 17:25 **Routes (option w) built and armed: `continue_v4i.sh 5000` in tmux `cont4i` (model_5000 about 17:45).**
  - `config/x2/routes.py`: `RouteFarTargetPoseVelocityCommand` (subclass of the turn-in-place command). `extended.py`
    records which tile each terrain cell got (`_CALLS`, generator order: columns, then rows). At each reset, 70%
    (`HIKING_X2_ROUTE_SHARE`) of the robots on the 12 tile columns get one of their tile's checked routes. They start at
    the route start (their env origin moves there, so height terminations use the route's floor), facing the first
    waypoint +-0.3 rad, and follow the waypoints with the eval harness's law: yaw clip(1.5 x error, +-1); forward 0 over
    60 deg, else clip(distance, 0.15, vmax) x cos(error); pass-through inside 0.2 m; final station = zero inside 0.2 m,
    held. Route robots are never turners and keep the route for the whole episode. That is about 26% of all robots
    (12/32 x 0.7).
  - Routes per tile type in the set (checked routes kept): single beams 29, beam fields 30, blocks 30, wedges 99 (38 of 40
    tiles), tight spots 40, log corridors 80, homes 101, street front 120, stairs 121. Every tile type has routes on
    every level; on a sampled layout every one of the 120 tile cells had at least one route.
  - CPU tests (`run4/test_routes.py`): the law's cases; route table from the real set with world coordinates checked
    against the tile; a kinematic robot passes two waypoints without stopping and stops inside 0.2 m of the third;
    assignment at reset (origin, height, heading, flags; nothing outside a reset; paper columns untouched).
  - Restart path: pause at 5,000, then the construction check with option w and 256 robots (all 32 columns get
    robots). It must build a route table of at least 100 routes, else v4h resumes. Then continue with w. Levels come
    from the saved file (same layout). The trial at 31 iterations needs level >= 0.8 x, episode >= 0.4 x and turn
    rate >= 0.7 x, else fall back to v4h. Then the crash guard (fall-back v4h).
  - New logs from v4i: `route/share`, `route/final_reached`, `route/waypoint_index`; artefact watch per tile type
    (below); posture already in.
- 17:25 **Isaac artefact check (user's question): what is known, and the log that goes in with v4i.**
  1. Contact forces: not measured until now. v4i logs, per group (paper columns; each new tile type; velocity
     columns): share of robots with any body force over 3 x / 10 x body weight (1,240 / 4,120 N), running max force, and
     share of feet in contact more than 1 cm below the ground under them (foot height scanners). Whether spikes end in
     falls more often needs per-episode bookkeeping: not built.
  2. Tunnelling through walls, rails, posts, door leaves: not checked. The "sunk foot" share covers feet into floors
     only. A per-robot path check against the tile's boxes is not built.
  3. How props reach PhysX (from the builder code): each tile is a trimesh concatenation of one closed box per prop box
     plus the set's meshes, with no boolean union. So coincident and internal faces remain where boxes touch or overlap
     (seams inside a stair flight, posts under rails, slabs under door frames), and seam ghost contacts are possible.
     Leaning or overhanging props are oriented boxes (quaternion), not filled underneath. Thin boxes survive as they
     are (8.9 cm posts and beams are fine; plates thinner than about 2 cm are tunnelling risks at 200 Hz). A 0.5 m slab
     sits under every tile. The depth camera and the height scanners ray-cast the same terrain mesh PhysX uses.
  4. Spawns on new tiles: the 16:19 check put no robot on a new tile. v4i logs, per tile type, the share of robots in
     their first 0.2 s with a non-foot contact over 50 N, and its construction check puts 8 robots on every column.
  5. Robot side: 25 pairs are filtered in Isaac (pelvis / hip-roll both sides; same-side wrist yaw / pitch / roll vs
     hip pitch / roll / yaw; knee / ankle-roll; wrist-yaw / wrist-roll; torso / head-pitch). Foot-foot, leg-leg and
     arm-torso pairs still collide. No measurement of which pairs carry load in training. The foot is the mesh's
     convex hull in Isaac (the eval uses sole spheres).
  - Fit: the logs come with the v4i restart (no extra pause). Counts per tile type and the worst three examples come
    from the first 100 iterations of v4i.

- 17:40 **Doorway falls re-read from the eval's traces (model_4125, seeds 1002-1004): overshoot of a pass-through waypoint
  into a corner, then a turn with the feet against the 4x4, then a backward / sideways escape and a fall. My slow-turn
  reading was wrong. New order: routes first, turn-rate cap after.**
  - Why it overshoots and misses sideways, from the task as trained:
    1. Every training target is a terminal stand: the command is vx = min(2 x distance, vmax) toward it and drops to zero
       inside 0.4 m, so the policy learned to run at vmax until 0.4 m and coast 0.28-0.35 m. A waypoint it should pass
       through and then turn at is never trained; nothing slows it before a heading change.
    2. Lateral error is corrected only by yaw: vy is always 0 in the paper / run-4 task (`lin_vel_y` range 0), and vx is
       forward-only, so a waypoint slightly to the side at short range is missed sideways and the correction comes after.
    3. Until 4,500 there were no walls or beams beside a turn in training; turning in place happened in open ground.
  - Backward / sideways escape under foot contact: `feet_stumble` (-2, from 4,500) charges a moving foot hit from the side
    (horizontal force over 3 x vertical, foot faster than 0.15 m/s); it punishes the contact, it does not teach a way out.
    Backward and sideways walking is commanded only on the two velocity columns (since 4,500) and has no clips in the motion
    reference, so the escape is untrained either way.
  - Order of the next changes (one at a time, trial / fall-back / crash guard each):
    1. **Routes as start / target sequences** (before the turn-rate cap): robots start at the map's route starts with their
       heading and follow the route's waypoints in order, passing through intermediate waypoints (next target taken inside
       0.2 m, no stop) and standing at the end station; this trains turns beside walls, beams and door frames at the
       distances the routes have, braking before a turn, and the final stand. Needs: per-cell record of the tile instance,
       spawn pose with yaw, a waypoint-sequence command (the current command has one target per period). CPU build and
       tests: about 3 h of work at the next resume; Isaac check in the pause of the restart after it.
    2. Turn-rate cap 0.3-1.0 rad/s (prepared) after that.
    3. Later: lateral command (vy) on route tiles so small side errors are stepped out, not turned out; backward / sideways
       clips (the Pico captures) for the velocity columns.
  - Eval-side, for now (the eval's choice): for a pass-through waypoint followed by a turn of more than 45 deg, slow the
    command to the station rule from about 0.6 m out, because the policy brakes only when the command drops.

- 17:12 **v4h on the map; model_4750 exported for the eval** (`eval/x2_parkour_v4_4750/`, task
  Instinct-Parkour-ExtendedFixed-Amp-X2-v0, trained_model hipexclude, ONNX vs torch 1.9e-6; 4,625 also exported).
  - Training at 4,758: mean level 5.57, episode 889. New columns: single beams 6.38, beam fields 6.34, blocks 6.48, wedges
    6.02, tight spots 4.46, log corridors 3.30, homes 4.29, street front 5.77, stairs 4.04. Posture (Isaac, commanded
    forward): torso pitch flat -17.4, stairs -15.0, boxes -15.8, gaps -13.2 deg; waist pitch -17.7 to -17.8 deg everywhere
    (at its -18.0 deg limit). So the lean is no smaller on stairs or gaps than on flat ground, as the eval found.
  - Doorway turn (4,125: 21 falls "clear of the door wall", 3,000: 3): reading, not yet shown. The eval's turn-in-place
    drift at +-0.4 rad/s grew from 0.15-0.42 m (3,000) to 0.40-0.53 m (4,125), more than the doorway's 0.44 m half-width.
    Training turn commands are clip(2 x bearing error, +-1) and almost always saturate at 1 rad/s, so slow turns are
    untrained; overspeed (o) and sliding (z, any yaw) changed how it turns at speed, and the training map had no walls until
    v4h. Asked the eval for three of the 21 traces (command, yaw rate, planar velocity, contacts in the last 2 s).
  - Next changes inside v4h, one at a time, each with the trial / fall-back / crash guard:
    1. Per-turner yaw-rate cap 0.3-1.0 rad/s (prepared: `patch_turnrate.py`, also adds the per-command speed log), so
       slow turns are trained; restart after model_4750's eval confirms the doorway issue (about 18:30).
    2. The map's checked routes as start / target pairs with start heading: record which tile each (level, column) cell
       got, use the route starts as env origins with yaw, and the route ends (and their waypoints) as targets on those
       cells instead of flat patches. CPU build tonight; Isaac check in the pause of the restart after (1).
  - Turn-in-place commands on tight-spot and home tiles: yes; option y gives 15% of command periods on every
    non-velocity column (all new tile columns included) as turn-in-place periods toward a target at least 1.5 m away.

- 16:25 **Extended task construction check passed (exit 0, 16:19-16:23) and v4h is starting** (continuation of v4g's
  model_4500 on `Instinct-Parkour-ExtendedFixed-Amp-X2-v0`; trial at 31 iterations, fall-back v4g; crash guard after).
  Check log `runs/x2_parkour_v4/_check_extended_161908/`: the X2ParkourExtendedFixedEnvCfg built with all 32 columns, the
  env stepped 150 zero-action steps without a Traceback, depth frames written (32 x 8 x 36 x 32, values 0.10-1.0, no NaN).
  Limits of the check: with 64 robots only columns 0-15 got a robot (2 each), so no robot stood on a new tile in the
  check; the script's printed summary (contacts, episode ends) did not reach check.log (only the npy files), so contact
  and fall counts at spawn on the new tiles are first seen in the v4h trial. The posture log goes in with v4h.

- 16:15 **Torso lean back: the camera explanation holds; watch item, fix belongs to the camera, a user / hardware decision.**
  - Reference clips (v2 motion file, frames at 0.35-0.65 m/s, 7,245 of 20,001): pelvis pitch -1.8 deg, waist pitch +13.3 deg,
    torso about +11.6 deg (forward).
  - MuJoCo, my `posture.py` at 0.5 m/s, t = 3-8 s (forward positive): model_1000 torso -17.5 / pelvis +1.1; 1,500 -17.3 / +1.3;
    2,250 -14.5 / +4.2; 3,000 -14.3 / +4.4; 4,125 -13.8 / +4.6; standing at zero command (4,125) torso -10.2 / pelvis +7.8.
    **Waist pitch -18.3 to -18.4 deg in every checkpoint and standing: the joint sits on its lower limit (URDF -0.314 rad =
    -18.0 deg)**, so the lean is the waist bent fully backward since at least 1,000, not a drift. Eval (state/eval.md):
    3,000 -16.6, 4,125 -15.5; by zone -14 to -18.7, no forward lean near stairs. Isaac: not logged until now; per-group
    posture log (`posture/torso_pitch_<flat|stairs|boxes|gaps>`, `posture/waist_pitch_*`, robots commanded forward) added to
    keep.py with constant keys (CPU test passes) and goes in with the v4h restart.
  - What holds the trunk: `pelvis_orientation_l2` -3.0 on torso_link and `flat_orientation_l2` -3.0 on the pelvis (projected
    gravity squared: a 14 deg lean costs 3 x 0.058 x 0.02 = 0.0035 per step), `freeze_upper_body` -0.004 x |q - q0| (waist
    0.32 rad: 0.00003 per step). The style discriminator sees the 12 leg joints only (run-4 fix a), so nothing in style
    pulls the waist toward the reference's +13 deg. The trunk is almost free; the camera is the only strong reason to choose.
  - Camera as trained (env.yaml of run 4, same in the card): torso_link (0.131, 0, 0.259), 61.6 deg down, 87 x 58 deg,
    64 x 36 raw, crop 16 columns each side only (rows untouched), clip 0-3 m. Footprint on flat ground (my MuJoCo calc,
    centre column): upright torso: top row about 1.5-1.6 m ahead of the toe (eval: 1.66 m), bottom row at the toe tips; at
    the walking lean (4,125): 3.07 m (z-depth 2.96 m, at the 3 m clip), bottom row 0.19 m ahead of the toe; at 1,000-1,500 the
    top row was beyond the clip (3.8-4.0 m). Reference lean (+11.6 deg): about 1.0 m. So the upright view is short of the
    user's 2 m and the policy buys the range by bending the waist to its limit, losing its own toes from the image.
  - What meets the requirement (10 cm behind the toe tips to 2 m ahead, at the standing pose, lens about 1.07 m): it needs
    about 67 deg of vertical field (95 deg down to 28 deg down); the trained field is 58 deg. Options: (1) pitch up about 5 deg
    (56-57 deg down): top row 2.0 m, bottom row at the toe tips (misses "10 cm behind" by about 10 cm); (2) 65 deg vertical
    field (taller image) at about 61 deg down: about -0.05 m to 2.0 m; (3) both. Which is right depends on the real camera's
    depth field and the real mount pitch (the mount was solved from the user's landmarks; a training pitch that differs from
    the hardware makes a sim-to-real gap). Not decided by me: the user's camera mount / sensor field.
  - Cost: same observation size, so a continuation (fine-tune) from the current policy, not a restart from scratch; the depth
    encoder sees a shifted view and must re-learn, so expect a dip like a restart. Not together with v4h (one change at a
    time): after v4h's trial passes. Expect the waist to come off its limit once the upright view reaches far enough; if it
    does not, a waist term (style on the waist, or a stronger upright cost) is the next step.
  - Stance width (MuJoCo, 0.5 m/s): 3,000 0.217 m mean (min 0.083), 4,125 0.200 (min 0.102, 10% of frames under 0.12 m), 2,250
    0.164 (min 0.039, 30%). `feet_close_xy` (+0.4 x Gaussian, threshold 0.12 m) is the only term against narrow feet; foot-foot
    and leg-leg hulls still collide in Isaac (only pelvis / hip-roll and wrist / hip pairs are filtered), so a crossing costs a
    contact, not an episode. Knees bent (pelvis 0.65 m against 0.679 m standing default) and little arm swing: the arms and
    waist are outside the style input, so nothing asks for a swing.

- 16:05 (node clock) **Extended map goes into training now (user, via coordinator). Plan and state.**
  - model_4125 (v4g) eval: card 50 / 50 / 32 / 50, floor 50 / 50 / 1 / 50; turn drift 0.40-0.53 m at +-0.4 rad/s, 0.22-0.24
    at +-0.8; stop 0.28 m; speed +20 / +6 / -2 / -7 / -17% at 0.2 / 0.35 / 0.5 / 0.6 / 0.8. Keep v4g. The floor variant
    (at least 0.45 m/s until the disc) was a workaround for a policy that would not walk slowly; 4,125 tracks 0.2-0.35 within
    20%, and the floor variant spoils turning in place (1/50) by commanding forward speed into the station. Retire it; the card
    is the standard. In Isaac, turners still slide 0.23-0.25 m/s (`turn/planar_speed`, 4,000-4,370, slowly falling).
  - Next change = the extended task (ahead of the walking-floor change; its velocity columns already command -0.4..0.8 m/s
    and turn-only). Prepared but held: per-turner turn-rate caps 0.3-1.0 rad/s and a per-command speed log on the flat column
    (`scratchpad patch_turnrate.py`, not applied).
  - Code: `extended.py` now has `X2ParkourExtendedFixedEnvCfg` (run 4's Fixed task with all HIKING_X2_OPTS, plus the extended
    columns, velocity columns, stumble -2.0, clearance -0.5), registered as `Instinct-Parkour-ExtendedFixed-Amp-X2-v0`. Same
    observations and actions, so v4g's checkpoint continues on it. The 2026-11-11 class (on the run-2 task) is kept, not used.
  - Columns: 32; paper types keep 20 (62.5% of robots); 12 new: single beams, beam fields, blocks, wedges, tight spots, log
    corridors (wall / rail / post sides), homes, street front, stairs x2 (one column each, ~256 robots), and 2 velocity
    columns (forward -0.4..0.8, sideways +-0.3, turn +-1). All ten levels of every tile type are in the grid; robots on new
    columns start where Isaac's curriculum puts new robots and move by the same tracking rule.
  - Near walls: no proximity termination anywhere in this task. Episodes end on torso contact (`base_contact`), tilt > 1 rad,
    pelvis below the floor minimum, leaving the terrain, time-out. Any non-foot contact over 1 N costs -1 per step
    (`undesired_contacts`); feet inside edges cost -4 (`volume_points_penetration`); new: side hits on a moving foot (stumble
    -2), low swing foot (clearance -0.5). So a wall beside the path is free until it is touched (SSR's 6 cm termination is not
    here). Not done: a margin term (cost for a limb within a few cm of a wall).
  - Held-out rule: tiles are the exported set; no apparatus layout or route (the set's README and generator check).
  - Missing, as before: the set's checked routes as spawn / target pairs with spawn yaw (targets are flat patches on each
    tile, spawn is the tile's widest-clearance spawn); backward / sideways clips in the motion reference; drop-edge stations
    and slot stations (to add to the tile set after this first run).
  - Times: `continue_v4h.sh 4500` in tmux `cont4h`. At model_4500 (about 16:17) it stops v4g, runs `check_x2_env.py --task
    Instinct-Parkour-ExtendedFixed-Amp-X2-v0` (64 robots, 150 steps, limit 30 min; result in launch.txt and
    `runs/x2_parkour_v4/_check_extended_*/check.log`); on failure v4g resumes at once. On success it continues v4g's policy on
    the extended task with: paper levels from the logged means, the pin, a trial at 31 iterations (level >= 0.5 x because 12
    new columns start low, episode >= 0.4 x, turn >= 0.7 x), fall-back to v4g; then the crash guard (fall-back v4g).
    Expected training on the extended task by about 16:45. Routes as spawn / target pairs: CPU work after that.
  - Per-command probe without a second GPU job: (1) the eval's MuJoCo tables on the desktop CPU for every checkpoint; (2) at a
    later restart, an in-training log of speed / command per command bin on the flat column (prepared, see above).

- 14:20 (node clock) **model_3500 (v4f, overspeed -3 since 3,250) fails the eval; decision: keep v4f's term, add one change
  (drift under a zero linear command), armed for model_3875 (about 14:32).**
  Eval 3,500: speed at 0.2 / 0.35 / 0.5 / 0.6 / 0.8: 0.33 / 0.49 / 0.59 / 0.63 / 0.68 (3,000: 0.20 / 0.34 / 0.48 / 0.56 /
  0.67); stop 0.40 m (met); flat card 50 / 50 / 2 / 50, floor 50 / 5 / 0 / 50 (3,000 turn in place 18 / 33); turn rates
  0.45 / 1.01 / -0.36 / -0.88 with 0.42-0.66 m drift (3,000: 0.15-0.42). Apparatus (3,000): 0/50, 29 falls, 21 stalls.
  a) Why, checked:
  - Same frame: `overspeed` uses root_lin_vel_b[:, 0] against the command's x, both in the root (yaw) frame the command is
    given in. It fires only when the forward command is above 0.1 m/s, so never during turn-in-place or stands; it cannot
    have taught the turn drift, and nothing else charges drift there (dont_turn_wait charges only a slow turn rate).
  - Training at 0.2-0.35: never. Walking commands are 0.45-0.8 / 1.0 (`HIKING_X2_VX_MIN` 0.45), so the eval's +66% / +41%
    is extrapolation the penalty never saw; I have no per-command Isaac probe, so I cannot say whether Isaac agrees. The
    training monitor mixes all terrains (cmd 0.62, actual 0.22-0.25, ratio 0.35-0.39 from 3,300 to 3,757, below 3,000's
    0.44), so in training the penalty is small (-0.4 to -0.8 per episode) and the robots are slower on average, not faster.
  - Overspeed at 0.5 / 0.6 grew 3,000 -> 3,125 before the penalty existed (0.48 -> 0.58 at 0.5); 3,500 is about where 3,125
    was. So the -3 weight at most slowed the drift; it did not cause it. Turn in place in Isaac is good (rate along command
    0.87-0.91 at 0.93 commanded, 3,400-3,757), so the MuJoCo failure is drift, not rate: an unpenalised behaviour, plus
    whatever sim gap there is (not measured).
  b) Decision: keep v4f (overspeed stays, it is not the cause and costs little), roll forward with ONE change: option z with
  `HIKING_X2_STOP_ANY_YAW=1`: -2.0 x planar root speed whenever the linear command is zero (stands and turn-in-place periods;
  a turner drifting 0.5 m/s pays 0.5 x 2 x 0.02 per step). CPU tests pass (charges turner drift and stand coast, never
  walking; log keys constant, new `turn/planar_speed`). `continue_v4g.sh 3875` (tmux `cont4g`): pin, pre-stop level copy,
  trial at 31 iterations (level >= 0.8 x, episode >= 0.4 x, turn rate >= 0.7 x), fall-back to v4f; then the crash guard
  (two error exits -> v4f with pre-stop levels; after-pin level check at 80 iterations, marks 5.7 / 900).
  Measure on model_4125 (second checkpoint after): flat_turn_in_place >= 33/50 floor (3,000's), turn drift <= 0.3 m, stop
  <= 0.40 m, stop-stand >= 45/50, walk 50/50; Isaac `turn/planar_speed` falling. If speed at 0.2-0.35 is still +40% or more,
  the next single change is slow commands in training (below).
  c) Apparatus needs not trained yet, in the order I will add them (each one change, same restart path):
  1. Slow commands: walking floor 0.45 -> 0.15 with `dont_wait` moved from 0.3 m/s to half the command, so a stander under
     a slow command still pays (SHARED_LESSONS "Slow commands": the paper never trains 0.2-0.4). After v4g passes, about
     4,250.
  2. Station stops next to obstacles and in narrow slots: targets placed 0.3-0.5 m behind a 4x4 / box edge with a 90-degree
     approach, and stand stations next to 0.4-1.2 m drops (ACP's 78 drop-edge stands). Needs terrain or target changes, so
     it belongs to the extended task: CPU build today, first Isaac check at the restart after slow commands.

- 12:55 **v4f crash loop found and fixed.** Traceback (train.log, both crashes): `on_policy_runner.py` line 253 in `log`:
  `KeyError: 'Curriculum/level_keeper/levels_pinned'`. instinct_rl's logger takes its keys from the first step's dict and
  fails when a key disappears; my level keeper stopped returning `levels_pinned` when the pin released (about 62 iterations
  after each start), so every start from 3,250 died before the 3,375 save. The trial check at 12:22 passed (level 5.56,
  episode 408, turn 0.765) because it ran before the release. Fix in keep.py (installed in the InstinctLab copy too): every
  key on every call (`levels_pinned` 0/1; turn values carry the last value when no turner is active; after an internal
  error the same keys with 0). CPU test `run4/test_keep.py`: identical keys over 31 calls through the release at 1,500
  steps, turners on and off, and after a forced error. Killed the third attempt (SIGTERM hung; SIGKILL at 12:54:12);
  v4f restarted 12:54:42 from 3,250 with the fix, levels from the file (3,282: mean 5.56). Cost: no progress from 12:12 to
  about 13:00 (three starts of about 6 min Isaac start-up each, about 9-10 min of training lost each time).
  - 13:12 verified in Isaac: past the pin release without a crash (iteration 3,331, `levels_pinned` 0), level 5.51 (5.4
    before), episode 657 (850 before; still rising after the synchronised start). AFTER PIN check OK.
  - `crashguard_v4.sh` (tmux `guard4`): two error exits from the same checkpoint after it started -> stop, pre-stop levels,
    v4d staged.py, `start_x2_parkour_v4d.sh`; and 80 iterations after its start (after the pin releases) writes level and
    episode length to gates.txt ("AFTER PIN"), flagging level < 0.8 x 5.4 or episode < 0.5 x 850.

- 12:07 **Overspeed: one change armed (v4f at model_3250, about 12:13, tmux `cont4f`).** Eval on model_3125: speed at
  0.2 / 0.35 / 0.5 / 0.6 / 0.8 commands 0.30 / 0.47 / 0.58 / 0.64 / 0.71 m/s (3,000: 0.20 / 0.34 / 0.48 / 0.56 / 0.67), stop
  0.47 m at 0.68 m/s (stopping time 0.69 s, same as 3,000's 0.64 s): the longer stop is the higher speed.
  - Option o (`keep.py` `overspeed`, staged.py.next5): weight -3.0 x max(vx - vx_cmd, 0) for forward commands > 0.1 m/s; 0 at
    or under the command, 0 for stand and turn commands. Per control step (CPU test at -2.0, scaled to -3.0) beside the speed
    reward (+2.0, perfect +0.040): 0.60 under 0.60: 0 (speed reward 0.040); 0.84 under 0.70: -0.0084 (speed reward 0.037);
    0.30 under 0.20: -0.0060 (0.0384); 0.45 under 0.60: 0 (0.0366).
  - Restart path: level pin (1,500 steps) in keep.py; pre-stop level copy; trial after 31 iterations requires level >= 0.8 x
    (5.3 now), episode >= 0.4 x, and the turn-in-place rate >= 0.7 x the value before (0.83 rad/s over the last 20 prints),
    else automatic fall-back to v4d with the pre-stop levels. The pin holds levels during the trial, so the level check mostly
    tests the pin itself.
  - Success measure: eval on model_3500 (second checkpoint after the restart): speed within +-15% of the command at
    0.2-0.6, stop test <= 0.40 m and <= 0.05 m in the last 2 s, stop-stand >= 45/50, flat_walk 50/50, turn in place at least
    3,000's (card 18 / floor 33 of 50; rates >= 0.4 rad/s). Reference checkpoint stays 3,000.

- 11:55 **No restart: v4e retry withdrawn.** Eval on model_3000 (v4d, no stop penalty): flat_walk 50/50 and flat_stop_stand
  50/50 with both the card and the floor command (2,625: stop-stand 0/50; 2,250: 45/50). Stop test: 0.55 m/s before the
  zero command, 0.35 m travelled, 0.03 m in the last 2 s. That meets the bar of 45/50 stop-stand.
  - Why it improved, with numbers. Stop distance divided by speed before the stop (an effective stopping time): 2,250
    0.40/0.48 = 0.83 s; 2,500 0.72/0.60 = 1.20 s; 3,000 0.35/0.55 = 0.64 s. So it is not only slower walking; the stop
    itself got shorter. Most likely the 10% exact-stand periods (on since 1,875) taking hold; not separated from the
    effect of the level dip after the 2,750 restart.
  - Watch for overspeed: training speed before the stop is 0.55 now (0.60-0.67 at 2,500), and eval speed at a 0.6 command
    for 3,000 is still pending. If 3,125 / 3,250 show speed at 0.6 above 0.70 m/s, or the stop test above 0.45 m, the next
    single change is an overspeed term (charge forward speed above the command), not the stop penalty. Both are prepared
    (stop_motion in keep.py; level pin; trial path with the pre-stop level copy).
  - Levels at 3,064: boxes 5.95, slope 5.47, mesh boxes 4.58, rough 5.70, stand 5.40, stairs 7.02, stairs high 5.61, inverted
    5.84, inverted high 5.32, gaps 1.05 (mean 5.28; 5.62 at 3,000).
  - Gaps (1.05) lag every other type by about 4 levels. The paper's gap sizes were made for the G1 (hip about 0.76 m against
    the X2's 0.64 m). Neither apparatus has gaps, so they are not on the path to the goal; left as is. Revisit in the
    specialist stage, or scale the gap widths to the X2's leg length in the extended task.
  - Stand at an edge with a drop: the paper's terrains place targets on flat patches, including platform and box tops (40 cm
    in Table 8) and stair tops, and the 10% stand periods happen wherever the robot is, so stands near 0.2-0.4 m edges occur
    by chance. Nothing trains a stand 0.3 m from a 1.1 m drop. Planned for the extended task: stations next to drops of
    0.4-1.2 m (ACP's fix), with the date of the extended start.

- 11:02 **v4e trial failed on levels; the fall-back (v4d, no new term) drops the same way, so every restart costs levels.**
  v4e (z on), from 5.44: 4.48 at 2,753, 3.55 at 2,759, 2.91 at 2,768, 2.82 at 2,781; episode 96 -> 451. Fall-back v4d from
  model_2750 (10:52), restored 5.42 from a level file I rebuilt at 10:53 from the per-type means at 2,719-2,728 (the saved file
  had already been overwritten by the trial with 2.79; copies in `_supervisor/terrain_levels_prestop_2728.pt` and
  `_trash/2026-11-12/terrain_levels_trial_2782.pt`): 4.43 at 2,751, 3.56 at 2,760, 3.17 at 2,770, 3.08 at 2,780; episode 36 ->
  497. Reading: the drop is the restart, not stop_motion. Mechanism: the restored levels move the env origins but the robots
  are still on their start tiles, and all robots start together; the first episodes end early (35-150 steps; the first wave
  probably on the out-of-bounds check against the moved origin, not verified), and the curriculum score is summed over steps /
  1,000, so every short episode moves a robot down a level, two or three times in 20 iterations.
  - Fix (keep.py, CPU-tested; takes effect at the next restart only): for 1,500 control steps after a restore the curriculum's
    moves are undone for every resetting robot (`HIKING_X2_LEVELS_PIN_STEPS`); afterwards the curriculum runs as usual.
  - Trial path fixed: `continue_v4e.sh` keeps `terrain_levels_prestop.pt` and the fall-back restores from it.
  - Now: v4d training from 2,750 at about level 3.1, recovering by itself (last time 1.8 -> 5.0 took 220 iterations). No restart.
  - Next: retry v4e (z, with the pin) only once levels are back to about 5 (about 3,000-3,125); trial criteria unchanged.

- 10:40 **Stop regression 2,250 -> 2,500: reproduced, diagnosed, one change armed (v4e at model_2750, tmux `cont4e`).**
  - Reproduced with the eval's own `eval/stop_test.py` (3 seeds, 4 s at 0.6 then a stand for 3 s; desktop, standard
    command): 2,250 speed before 0.48 m/s, travel after zero 0.41-0.42 m, last 2 s 0.11-0.12 m; 2,500 0.60 m/s, 0.63-0.64 m,
    0.23 m. Matches the eval (0.40 / 0.72). My own `probe.py` gives 0.73 / 0.71 and is not used for gates.
  - Diagnosis (config read plus trainer log): (1) nothing in the reward charges motion under a stand command. The paper's
    `stand_still` (-0.3) is (sum |q - q0| - 4) under a zero command: a joint-pose term with no velocity in it. The speed
    kernel exp(-v^2/0.25) keeps 70% at 0.3 m/s. (2) v4d's 10% exact-stand periods do include stops from a walk (any period
    after a walking one), but the only cost of coasting is that weak kernel. (3) Walking got faster: speed before the stop
    0.48 -> 0.60 m/s at a 0.6 command (eval: 0.84 under the 0.7 cap). The touchdown air-time reward rose 9.4 -> 12.7 -> 13.8 per
    episode (1,990 / 2,500 / 2,707) and measured air per touchdown 0.08 -> 0.10 s: longer swings at the same cadence are
    longer, faster steps, and overspeed is cheap in the same kernel (+0.2 m/s keeps 85%). More speed means more run-on.
    (4) Overshoot past the 0.4 m radius turns the command back on as yaw-only (turn toward a target behind), which is the
    circling. (5) Levels: flat-terrain robots (perlin_rough) went 7.8 -> 5.8 -> 6.3; no sign that levels drive the speed.
  - **One change, option z:** `stop_motion` = planar root speed (m/s) while the command is a stand (|v| < 0.1 and |yaw| <
    0.1), weight -2.0 (a 0.5 m/s coast for 1 s costs 1.0). keep.py + staged.py.next4; CPU tests pass (`run4/test_keep.py`
    incl. stop_motion cases, `test_t` on next4). Overspeed and overshoot are left for later, one at a time.
  - Restart path with trial: `continue_v4e.sh` waits for model_2750, records level and episode length before the stop
    (5.42, 852), restarts, and after 31 iterations requires level >= 0.8 x and episode >= 0.4 x, else stops and resumes v4d
    from the same checkpoint (logged in launch.txt).
  - **Success measure:** eval stop test on model_3000 (3 seeds): travel after zero <= 0.30 m and last-2-s drift <= 0.05 m,
    with flat_walk 50/50 and speed at 0.6 of 0.45-0.65 m/s; flat stop-stand >= 45/50 (2,250: 45). Reference checkpoint for
    obstacle and route tests stays 2,250 until then.

- 10:10 **No restart; carry on v4d.** Patience rule in full (coordinator 10:01): no restart without a defect shown on a probe that
  reproduces the eval; one change at a time; any restart path tried first in a short throw-away run with automatic fall-back
  if levels or episode length collapse in the first 30 iterations.
  - Levels per type, 1,749 -> 2,398: boxes 4.63 -> 5.43, slope 4.03 -> 5.10, mesh boxes 2.62 -> 3.67, rough 5.94 -> 5.41,
    stand 6.11 -> 5.30, stairs 6.49 -> 6.83, stairs high 6.07 -> 5.39, inverted 3.99 -> 5.69, inverted high 4.57 -> 5.44, gaps
    0.60 -> 0.81; mean 4.43 -> 5.0 (flat at 5.0 since 2,250). Time-outs 69% (same), episode 840 (800).
  - Training speed ratio 0.70 (2,000) -> 0.47 (2,150-2,350) -> 0.37 (2,400) while the mean level rose 1.8 -> 5.0, wait
    penalty -0.6 -> -3.0 per episode: slower on harder tiles, same pattern as 1,250 -> 1,500. Watch: if the eval's flat speed
    table at the next checkpoints drops below 2,125's (0.51 at 0.6), that is a defect.
  - Air-time miss at 2,000 (0.084 s against 0.10): no change. Rising (0.043 / 0.066 / 0.072 / 0.084 at 1,000 / 1,500 / 1,750 /
    2,000), and the eval-side blockers are stop and turn, not air time.
  - Turning in training: rate along command 0.66 rad/s at 0.91 commanded (2,398). Eval at 2,125: 0.48 / 0.88 / -0.52 / -0.87
    rad/s at +-0.4 / +-0.8, flat 50 / 6 / 19 / 50 (walk / stop-stand / turn in place / turn and return).
  - **Probe bug found:** my 08:30 numbers for "1,500" were model_500 (a shell typo `_$1500` expanded to `_500` when copying).
    Re-run on the right files (single seed, starts from standing): 1,500: 0.66 m/s at 0.6, run-on 1.18 m in 1 s, yaw -0.09 /
    -0.07; 2,125: 0.66 m/s, run-on 0.37 m, drift 0.32 m in the next 3 s, yaw 0.73 / 0.85 rad/s. Eval: 1,500 0.56 m/s, about
    1 m coast, no turn; 2,125 run-on 0.34-0.35 m, yaw 0.88 / 0.87, "nearly still" after the stop. Agrees except the drift after
    the stop (mine 0.32 m); until that is resolved against `eval/stop_test.py`, the eval's numbers are the gate.
  - Stop and target behind, as trained: the training command is zeroed within 0.4 m of the target (`target_dis_threshold`
    0.4, card `target_stop_radius_m`), so the policy learned to brake about 0.4 m out and coast 0.3-0.35 m; the eval's
    brake-early variant (zero from about 0.35 m) is the trained rule. A target behind the robot and more than 0.4 m away gives
    vx = 0 (forward-only clamp) and yaw toward it: turn on the spot, then walk; inside 0.4 m the command is zero (stand).
  - Next single change, only if the eval still shows run-on over 0.25 m at the 0.35 m brake on 2,500 / 2,625: stand_still
    weight -0.3 -> -1.0 (shorter stops), tried in a 30-iteration throw-away run first.

- 08:30 **v4c restart at 1,750 collapsed the curriculum; restarted as v4d from model_1875 (08:26).** Measured (trainer log):
  1,749: episode 973, levels 4.43, time-outs 68%. After the restore: 1,751 episode 34, 1,760 levels 1.55, 1,780 levels 0.00,
  root_height falls 60-67% of ends, value loss 3 -> 68 (1,760) -> 0.6 (1,850); by 1,854 episode 798, reward 26 (recovering) at
  level 0 on every type. Cause as far as shown: my forced time-out right after the restore (keep.py) gave every robot a
  1-step episode, whose tracking score (summed over steps / 1,000) is under the 0.3 threshold, so every robot dropped a level
  and all resets were synchronised; the value function then saw states it did not know (value loss x20) and the policy fell
  for about 80 iterations, each fall another level down. Not separated from the other v4c changes (y turners, vx floor 0.45).
  Fix: the forced time-out is off (HIKING_X2_LEVELS_FORCE_RESET, default 0; levels move at natural resets). Since levels were
  already 0, the v4d restart cost no curriculum. Turn signal in v4c: `turn/rate_along_cmd` 0.31 rad/s at 0.85 commanded
  (1,844-1,854), from 0.05 at 1,770.
- 08:30 **Stand regression 1,000 -> 1,500 (eval: zero-command drift 0.43 m / 8 s, stop-stand 49/50 -> 8/50): reasoning.**
  (1) Option c set `rel_standing_envs` to 0 (paper 0.05): since 03:19 no robot has had an exact stand command for a whole
  command period; the only zero commands are the last part of a period after arriving within 0.4 m, and an overshoot there turns
  the command back on (turn and walk back), which is the "corrective circle" the eval saw. (2) The style coefficient ramped
  0.075 -> 0.25 from 500 to 1,500 (schedule s) on a reference with 3.3% standing frames, so standing still scores low style.
  (3) The stand column gets random yaw (+-1), never an exact zero. The air-time reward t does not pay at a zero command
  (gated on |cmd| > 0.1), so it is not the cause. **v4d adds exact stand commands: `HIKING_X2_STAND_SHARE=0.10`**
  (staged.py.next3; paper 0.05) on top of v4c (k, y, vx floor 0.45, t 140, save 125). Fallback: v4c.
- 08:30 Deterministic MuJoCo probe (`perception-agents-work/hiking_sim2sim_x2/probe.py`, desktop CPU, one seed; stop = 4 s at
  0.6 then zero; yaw = [0, 0, +-0.8] 6 s): 1,000: walks 0.11 m/s, falls after the stop; yaw -0.09 / -0.08. 1,500: walks 0.00
  in its last second at 0.6 (eval measured 0.56; my probe starts from standing, one seed), stop 0.07 + 0.01 m; yaw 0.11 /
  -0.05. 1,750: walks 0.49 m/s, stop distance 0.33 m in 1 s, drift 0.31 m in the next 3 s; yaw +0.19 / +0.41 rad/s with
  0.7-1.2 m drift (circles, not on the spot). **Gates for v4d (probe, run on the desktop when resumed): 2,000 and 2,250:
  stop distance <= 0.25 m, drift <= 0.10 m / 3 s, yaw >= 0.4 rad/s both ways with drift <= 0.3 m, no fall.** If the stand
  fails at 2,250: stand share 0.2 and the stand_still weight up; if turning fails: turn share 0.3.

- 07:38 **Decision at the 1,500 gate: carry on with t at 140; no t-280 / kernel experiment.** Gate recorded FAIL on air only:
  measured air per touchdown 0.066 s (mark 0.07) on a rising curve (0.043 at 1,000, 0.063, 0.066), speed 55.7% (mark 50%),
  falls 32% (49% at 1,000), episode 759 (644), while the mean terrain level rose 2.15 -> 3.77 (stairs 4.62, stairs high 7.07).
  A 0.004 s miss while the terrain gets harder is not a defect; the narrower kernel targets slow commands, which run 4 never
  sees (its walking floor is 0.6 m/s). The 2,000 gate (air >= 0.10 s, speed >= 0.6) stands.
- 07:38 **Eval at 1,000 (eval monitor): flat 50/48/0/0 (walk / stop-stand / turn in place / turn and return); never turns in
  place.** Own MuJoCo check (desktop, `perception-agents-work/hiking_sim2sim_x2/yaw_test.py`, model_1000): command fed directly
  as [0, 0, wz], the form the perlin_rough_stand robots train with: wz 0.8 -> turned -0.59 rad in 6 s, wz -0.5 -> +0.53 rad,
  0-4 lift-offs, i.e. it does not turn on the spot under any command form; not an adapter problem. Cause in the task: the only
  zero-forward yaw commands are the stand column (5% of robots, uniform +-1 rad/s), and a stander keeps much of the yaw reward.
- 07:38 **Next restart armed: `continue_v4c.sh 1750` in tmux `cont4c`** (waits for model_1750, about 08:03 UTC). Changes:
  - y (new, `config/x2/keep.py`): 15% of command periods off the stand column become 4-6 s turn-in-place periods: command
    [0, 0, clip(2 x bearing error, -1, 1)] to a target >= 1.5 m away, then stand facing it (the apparatus station); plus
    `dont_turn_wait` -0.5 when commanded > 0.4 rad/s and turning < 0.2 rad/s. Logged as `level_keeper/turn/*`.
  - k (new, keep.py): terrain levels saved every 300 s to `_supervisor/terrain_levels.pt` and restored at every restart; this
    first time from the per-type means at the newest iteration (current: stairs 5.2, stairs high 6.7, inverted 3.3/4.2, boxes
    4.0, mesh boxes 2.3, gaps 0.5), then robots time out once and respawn on their tiles.
  - Walking-command floor back to the paper's 0.45 m/s (`HIKING_X2_VX_MIN`; run 4 raised it to 0.6): the eval walks at 0.5 and
    the policy makes 0.19 m/s there.
  - Kept: a,c,d,f,r,s,t, t weight 140, save interval 125. Fallback: old staged.py + `start_x2_parkour_v4b.sh`.
  - Checked on CPU: `run4/test_keep.py` (stand column never turns, share 0.152, 4-6 s, forward zeroed; dont_turn_wait cases;
    restore from means keeps each type's mean, restore from file exact, time-out after restore, no-op without variables);
    `test_t.py` on `staged.py.next2`; level-means extraction on the live log; `bash -n`. Not checked: construction in Isaac (the
    restart is the check, with the fallback); whether the motion reference has turning-on-the-spot frames for the style reward.
  - Check after the restart: `turn/rate_along_cmd` rising toward `turn/cmd_abs` by 2,250; eval flat_turn_in_place on the first
    checkpoint after 2,000. If turners stay under 0.2 rad/s by 2,250: raise the share to 0.3 and add a yaw-tracking kernel of 0.25.
- 07:38 `trained_model` set in all run-4 cards ("x2_ultra_hipexclude.xml (pelvis / hip-roll pairs excluded)" plus the other
  filtered pairs) and in `export_for_eval.py` for the Fixed / Extended tasks. Isaac has no contact for pelvis / hip-roll at all
  (filtered since run 4).
- Not done: real Isaac observations for obs_check (needs a recording run; next pause that happens anyway).
- Paper A.8 plan, slow commands (0.2-0.4), backward / sideways: unchanged from 06:25 (extended task; specialists after the
  general policy).

- 06:25 **Option-t continuation verified and watched** (started 06:02 by `gates_v4.sh`: air per touchdown 0.043 s at 1,000).
  Trainer environment read from /proc: OPTS a,c,d,f,r,s,t, AIR_TD_WEIGHT 140, `agent.save_interval=125`, resume from
  model_1000; measured monitor in the log (`Curriculum/walk_monitor/air_per_touchdown_s`). One GPU job (pid 131721, 57 GB).
  New log dir `runs/x2_parkour_v4/20261112_060244_from20261112_031923/`. Gates: 750 speed 0.384, stalled 23%, no drift;
  1,000 speed 0.407, falls 49%, episode 644, level 1.55. After restart (1063-1072): speed 0.451, falls 44%, episode 571,
  measured air per touchdown 0.043-0.046 s (flat; touchdowns >= 0.2 s 2.7-3.1%), air-time reward 0.094/ep (was 0.078).
  Terrain levels restarted from 0 on resume (1.55 -> 0.61 at 1030 -> 0.91 at 1072; the curriculum state is not in the
  checkpoint), which also explains the shorter episodes. Decision: keep running.
  - Gates `code/x2_port/gates_v4b.sh` (tmux `gates4b`, record only): 1,250 and 1,750 line only; **1,500 pass = measured air
    >= 0.07 s and speed >= 0.50; 2,000 pass = air >= 0.10 s and speed >= 0.60**; alert if train.log silent > 15 min. Next
    experiment if 1,500 fails: continue from model_1500 with t weight 280 and the speed kernel narrowed 0.25 -> 0.1 (SSR r6d
    form), prepared on CPU before 1,500.
  - Automatic export `code/x2_port/auto_export_v4.sh` (tmux `autoexp`, CPU, niced): every new checkpoint to
    `eval/x2_parkour_v4_<N>/`, log `eval/auto_export_v4.log`. 750 and 1,000 exported (ONNX vs torch 6e-7, 1e-6). obs_check is
    synthetic for all run-4 exports; record real Isaac observations (`--obs`) at the next restart that happens anyway.
  - Eval needs not trained by the paper's task, and who covers them: exact stand on arrival and 1 s stand: adapter (command
    exactly zero inside the disc; the task trains exact-zero stands). 0.2-0.4 m/s: adapter can keep the command at >= 0.45
    until the disc and then zero; real slow walking goes into the extended task (commands 0.15-0.45 with dont_wait from
    0.1 m/s). Turn in place: extended task (zero-forward yaw commands, a share of corner targets). Backward, sideways:
    extended task (commands written; clips missing). Dates: extended task CPU changes 2026-11-12; Isaac construction check
    at the next restart; fine-tune from the general policy at about 3,000 (about 11:30 UTC 2026-11-12) if stairs levels rise.
  - Paper A.8 plan: after the general policy, fine-tune specialists (stairs; platforms) from it, flat plus the target terrain,
    walking command 0.45-0.55 m/s, run one after the other on this GPU; first one when general stairs levels stall.
  - Pelvis / hip-roll: filtered in Isaac since run 4 (fix d, both sides), so Isaac makes no contact for that pair at all; the
    policy never learned to carry load there. MuJoCo eval keeps it (153 N standing at 500). Shared-model question, coordinator.

- 04:57 (node clock; the 05:05 and 05:10 entries below were stamped ahead of the node clock) **Resumed, run 4 healthy, gates
  automated.** Heartbeat 04:54: iteration 577, GPU 82%, 55 GB, pid 102936, one trainer start since 03:19 (no restarts).
  metrics.jsonl lags the trainer by about 20 iterations (newest parsed 554). **545-554:** 22,065 samples/s (8.9 s/iteration),
  speed ratio 0.316 (0.22/0.70 m/s), falls 62%, stalled 22%, time-outs 18%, episode 399, air per touchdown (est.) 0.032 s,
  air-time reward 0.040/ep, both feet down 58%, D-reward 0.51, level 0.48. Against 500: speed 0.26 -> 0.32, episode 370 -> 399,
  air per touchdown 0.029 -> 0.032 s (still under the 0.05 s gate).
  - Option t files re-tested on the node CPU: `run4/test_t.py staged.py.next monitor.py.next` passes (0.25 / 0 / 0.30 cap / 0;
    threshold 0.05 s, default weight 140); `continue_v4.sh` syntax OK, weight 140.
  - **`code/x2_port/gates_v4.sh` running in tmux `gates4`** (CPU only, polls metrics.jsonl once a minute): at 750 writes the
    report line and the drift check (stalled > 40% with speed < 0.2) to `runs/x2_parkour_v4/_supervisor/gates.txt`; at 1,000
    writes the line and, if air per touchdown (est.) < 0.05 s, starts `continue_v4.sh` in tmux `cont4` (stops run 4 through its
    supervisor, continues from model_1000 with t; falls back to plain run 4). Expected: 750 about 05:20, 1,000 about 06:00 UTC.
  - Export of `model_500` made on the CPU: `eval/x2_parkour_v4_500/` (policy.onnx, encoder+actor, adapter_card.json,
    obs_check.npz; ONNX vs torch 7e-7; observation 10,016 = 800 proprio + 8 x 36 x 32 depth; chest camera torso_link 61.6 deg
    down; gains from env.yaml).
  - **Path to ladder step 1:** earliest scorable checkpoint is `model_500` (export ready; `model_250` export also exists). Blocker:
    there is **no `eval/adapters/hiking.py`** in the common eval (only acp, acp_baseline, stand). The adapter card plus
    `code/x2_port/sim2sim/sim2sim_mujoco_x2.py` (MuJoCo observation builder incl. depth, used for the 250 renders) are the sources
    for it. Expected result at 500: fails `flat_walk` (0.27 m/s net in MuJoCo at 250, short shuffles). Owner of the adapter: the
    eval monitor (common eval code), to be asked by the coordinator.
  - Backup: manifest 04:49:36, 2 checkpoints (250, 500), newest model_500 (04:42); next checkpoint 750.
  - 05:05 (coordinator: checkpoints at least every 30 min) **Save interval 125 at the next restart**: `--save_interval 125`
    added to `continue_v4.sh` (generated `start_x2_parkour_v4b.sh`) and to `start_x2_parkour_v4.sh` (fallback / plain resume).
    125, not 150: instinct_rl saves when iteration % interval == 0, so 150 would skip 2,000, 3,000, ...; 125 keeps every 1,000
    (about 18.5 min apart at 8.9 s/iteration). Run 4 keeps 250 until it is restarted (model_1000 is still saved, the gate
    finds it). Retention: the supervisor deletes nothing (all checkpoints kept, decision 2026-11-11); at 125 that is about
    240 x 68 MB = 16 GB over 30,000 iterations. Pruning to "newest two plus every 1,000" is not implemented.
    Backups of the old launchers: /tmp on the node.

- 05:10 **Option t changed to threshold 0.05 s, weight 140** (coordinator; SSR uses 0.10 s): run 4's swings are 0.03-0.06 s, so
  0.10 s gave no gradient. Cap 0.30 s kept. Walker with 0.2-0.25 s swings: 0.07 touchdowns per step x 140 x 0.02 x (0.15..0.20) =
  **+0.029..+0.039 per control step** (with dt: 2.8 per touchdown per second of clamped air; without dt: weight 2.8). A 0.06 s
  swing pays 140 x 0.02 x 0.01 = 0.028 per touchdown (a small gradient). Lunger and stander 0. Walker value with t about 10.6-11.6
  (table below otherwise unchanged). `staged.py.next`, `run4/test_t.py` (pays 0.25 / 0 / 0.30 / 0) and
  `code/x2_port/continue_v4.sh` updated. continue_v4.sh is uploaded but **not started**: it waits for model_1000, then continues
  run 4 with t and the measured-air monitor; it falls back to plain run 4 on failure. Start it only if air per touchdown is still
  under about 0.05 s at 1,000: `tmux new -d -s cont4 "bash code/x2_port/continue_v4.sh"`.

- 05:05 **Option t sized from our own table: weight 200** (coordinator: not SSR's 20, whose rewards are on another scale).
  Isaac Lab adds weight x dt x raw on the touchdown step, so with dt: 200 x 0.02 = **4.0 per touchdown per second of clamped
  air time**; without dt the same pay would be weight 4.0. A walker at 0.6 m/s with 0.2-0.25 s swings: about 3.5 touchdowns/s
  (swing 0.225 s + stance about 0.35 s per foot), i.e. 0.07 touchdowns per control step, each paying 4.0 x (0.10..0.15) =
  0.40..0.60, so **+0.028..+0.042 per control step**, the size of the walker-stander margin (0.077 - 0.042 = 0.035). A shuffler
  with 0.03 s swings, a stander and a lunger (feet planted) get 0. Default set in `staged.py.next` (`HIKING_X2_AIR_TD_WEIGHT`,
  default 200); `test_t.py` passes.

#### Walker, stander, shuffler, lunger per control step at run 4's early values (gamma 0.99), with option t at weight 200

Shuffler = run 4 at iterations 491-500 as measured (task +0.0047 per step excluding the fall penalty: alive 0.019, lin 0.019, yaw
0.013, wait -0.011, heading -0.008, the rest -0.027; style 0.038; episode 370; falls 60% of ends, fall penalty -8 x 0.6 at the
end, discounted 0.99^370 = 0.024). Walker, stander, lunger as in the run-3 table above (the walker is the G1 at 5,000 under X2
early scales; the lunger is run 3 at 278 with the fall penalty of 8).

| | Lunger | Stander | Shuffler (run 4 now) | Walker (0.2-0.25 s swings) |
|---|---|---|---|---|
| task + style per step | 0.066 | 0.042 | 0.043 | 0.077 |
| t per step | 0 | 0 | 0 (swings 0.03-0.06 s) | +0.028..+0.042 |
| steps until the episode ends | 140 (fall, -8) | 200 (stalled reset) | 370 (60% fall) | 1,000 |
| discounted value without t | 3.02 | 3.64 | 4.1 | 7.7 |
| discounted value with t | 3.02 | 3.64 | 4.1 | 10.5-11.9 |

Order now: walker > shuffler > stander > lunger. t widens the walker's lead over the shuffler from 3.6 to 6.4-7.8. The shuffler
earns nothing from t until its swings pass 0.10 s, so it gets no gradient towards longer swings until it is already partly there.
That is the risk with a threshold form. The threshold of 0.10 s is SSR's, unchanged.

- 04:55 **Prepared, not applied: option t (touchdown air time, SSR's form) and a measured air time per touchdown** in
  `config/x2/staged.py.next` and `config/x2/monitor.py.next` (the running files are untouched; both take effect only at a restart,
  e.g. a continuation from `model_1000`). t: on each touchdown of a robot commanded to move, clamp(swing air time - 0.10, 0, 0.30)
  summed over the feet, weight `HIKING_X2_AIR_TD_WEIGHT` (default 20; times dt, so a 0.3 s swing pays 0.08 per touchdown, about
  0.003 per control step at 1.7 touchdowns/s), never negative, beside the paper's `feet_air_time`, not scheduled. Monitor adds
  `air_per_touchdown_s` (mean air time of the swings that end at that step, moving-commanded robots) and `touchdown_air_ge_0p2`.
  CPU tests pass: `code/x2_port/run4/test_t.py` (pays 0.20 / 0 / 0.30 cap / 0 for a standing command; monitor mean and share),
  `test_staged.py`, `test_fall.py`, `test_refstart.py` on the new staged.py. Never built in Isaac.
  **Which air-time form run 4 uses:** InstinctLab's parkour `feet_air_time` (the paper's): in single stance it pays the shorter of
  the stance foot's contact time and the swing foot's air time, every step, gated on a moving command; no threshold is subtracted,
  so it is never negative and short steps are not charged (unlike Isaac Lab's (air - threshold) x first_contact form). It pays
  little for short shuffles (run 4: 0.036 per episode at 500).

- 04:45 **Run 4 at 500** (iterations 491-500): speed ratio 0.26 (0.18/0.70 m/s), both feet down 60%, no support 6.7%, step rate
  16.1 Hz, air time 0.0355/ep, air per touchdown about 0.029 s (estimate from the monitor's shares: (single + 2 x no support) / 2
  over touchdowns per foot; `report_line.py` prints it now), wait -4.18, falls 60%, stalled 28%, time-outs 14%, episode 370, fall
  penalty -4.9/ep, reference starts 0.5 (from here ramping to 0 at 1,500), D-reward 0.50, level 0.42 (perlin_rough_stand 4.49,
  perlin_rough 0.95, mesh boxes 0.54, slope 0.52, boxes 0.41, stairs and gaps 0.00-0.01). Trend 300/350/400/450/500: speed
  0.25/0.28/0.16/0.31/0.26, stalled 38/30/34/31/28%, episode 338/335/336/371/370, air time/ep 0.022/0.027/0.025/0.030/0.036.
  Run 3 at 500: speed 0.52, falls 80%, 348 steps, air time 0.0125. G1 at 1,000 (equal samples): 687 steps, falls 50%, air time
  0.049. Drift check (stalled > 40% with speed < 0.2): not met. `model_500.pt` saved. MuJoCo air per touchdown at 250 (flat):
  mean 0.047 s left, 0.061 s right. Port changes (ImplicitActuator, `run4_player.py`) copied to `code/x2_port/sim2sim/`.

- 04:15 **Run 4 at 250** (iterations 241-250): speed ratio 0.32 (0.23/0.70 m/s), both feet down 63%, no support 9.8%, step rate
  14.9 Hz, air time 0.0195/ep, wait -2.61, falls 67% (root_height 51%, bad_orientation 16%), stalled resets 26%, time-outs 8%,
  episode 279, fall penalty -6.0/ep, reference-start share 0.5 (schedule), style coef 0.075, D-reward 0.48, level 0.14
  (perlin_rough_stand 1.28, perlin_rough 0.44, slope 0.27, mesh boxes 0.24, boxes 0.12, stairs and gaps 0.00). Trend at
  50/100/150/200/250: episode 46/54/85/157/279, falls 103/102/100/66/67%, speed ratio 0.62/0.67/0.42/0.25/0.32, stalled
  0/0/2/31/26%. Run 3 at 250: 148 steps, falls 84%, speed 0.52, air time 0.0038. G1 at 500 (equal samples): 731 steps, falls about 60%,
  air time 0.040. Stop check (episode < 100 and flat, or speed < 0.1 with both feet down > 80%): not met.
- 04:15 **MuJoCo render of run 4 `model_250`** ("MuJoCo, not the training simulator"; explicit PD in MuJoCo, run 4 trained with
  implicit PD; run 4's 20 extra filtered pairs added through `run4_player.py`; the port now accepts ImplicitActuator in env.yaml):
  `perception-agents-work/hiking_sim2sim_x2/x2v4_ckpt250_mujoco_{flat,stairs}_{chase,legs}.mp4`, frame sheets
  `checks/frames/x2v4_ckpt250_*.png`. **Flat: it steps**: upright (pitch +3 deg), 74 lift-offs, short shuffling steps (longest
  air time 0.10 s), 3.74 m to the target in 11.1 s (0.27 m/s net against 0.70 commanded), no fall. Stairs: shuffles at the
  bottom step without climbing, falls forward into the steps at 8.2 s. Export `eval/x2_parkour_v4_250/` (ONNX vs torch 8e-7).

- 03:25 **Run 4 training** since 03:19:06 (launcher: "is training, GPU idle about 375 s"). Iteration 4: reward -12.0 (includes the
  -8 fall penalty), episode 47 steps, bad_orientation 74% / root_height 37% of ends (shares overlap), D-reward 0.26, style coef
  0.075, 9.7 s per iteration, 52 GB GPU memory, 0 Tracebacks. Construction checks seen in the log: "[X2 run 4] filtered 20 more link
  pairs", AMP joint terms 120 = 12 leg joints x 10 frames, `rewards_fall_penalty` -8.0 per fall. Run 3 stopped by its supervisor
  at 03:19 and kept (`model_250`, `model_500`); resume with `start_x2_parkour_v3.sh` if run 4 is abandoned.
  **Run 3 at 500** (iterations 491-496): speed ratio 0.515 (0.36/0.70 m/s), both feet down 64%, no support 16.7%, step rate 11.7 Hz,
  air time 0.0125/ep, wait -1.09, falls 80%, stalled 11%, time-outs 10%, episode 348, style coef 0.075, weighted style 0.033,
  level 0.50 (perlin_rough_stand 4.49, perlin_rough 1.62, slope/mesh boxes 0.83, boxes 0.22, stairs and gaps 0.00-0.06).
  Run 2 at 500: falls 10%, 936 steps, speed ratio 0.006. G1 at 1,000 (equal samples): 687 steps, falls 50%, air time 0.049.

- 03:07 **Run 4 `x2_parkour_v4` built and launching** (coordinator's decision after the port audit `state/hiking_port_audit.md`;
  the iteration-500 continuation of run 3 is cancelled, `continue_v3.sh` is in `_trash`). Task `Instinct-Parkour-Fixed-Amp-X2-v0`
  = `config/x2/fixed.py` (X2ParkourFixedEnvCfg, subclass of the staged task), from scratch, options a,c,d,f,r,s, stage 500,1500,
  fall penalty 8, reference starts 0.5, camera randomisation on, 8,192 robots. Launcher `code/x2_port/launch_run4.sh` (tmux
  `launchrun4`): waits for run 3's `model_500.pt`, stops run 3 through its supervisor, starts run 4, falls back to run 3 on a
  Traceback / trainer restart / supervisor exit / no iteration 2 in 10 min. Resume run 4: `bash code/x2_port/start_x2_parkour_v4.sh`.
  Fixes (each a listed departure where it departs from the paper; CPU checks in `code/x2_port/run4/test_run4.py`, census in
  `perception-agents-work/hiking_run4/census.{py,txt}` on the desktop):
  a. Discriminator joints: the 12 leg joints only (positions and velocities), base terms unchanged; head, arms, waist dropped.
  b. Default pose: old (G1 numbers) hip pitch -0.312, knee 0.669, ankle pitch -0.363, hip roll/yaw 0, ankle roll 0, waist 0.
     New = v2 reference mean, symmetrised: hip pitch -0.136, hip roll -/+0.043, hip yaw +/-0.181, knee 0.379, ankle pitch -0.114,
     ankle roll +/-0.057 (left/right); waist 0 (reference mean +0.229 is the lean); arms, head unchanged. Pelvis 0.679 m above the
     soles (MuJoCo FK; was 0.641); spawn 0.75 m (was 0.71).
  c. Action scale = the G1's per joint type (hip pitch/yaw 0.548, hip roll/knee 0.351, ankles 0.439, ...; was hip/knee 0.261, ankle
     pitch 0.258). Reference gait action-rate raw sum per 50 Hz step: 0.188 -> 0.093 (all joints), legs 0.173 -> 0.082 (x0.48).
     The G1 reference with the G1 scale was not computed.
  d. Self-collision: 20 more filtered pairs: pelvis vs both hip-roll links, and same-side wrist yaw/pitch/roll vs hip
     pitch/roll/yaw. Census over 3,000 reference frames (MuJoCo convex hulls, not PhysX): pelvis/hip_roll in 50-57% of frames (6 mm
     median), wrist/hip 6-38% (4-19 mm median); left/right foot hulls 5.2% (22 mm), ankle/knee cross pairs 0-1%: kept on (real leg
     crossings). Old and new default poses: no overlaps.
  e. soft_joint_pos_limit_factor 1.0 (was 0.9): the reference reaches the hard limits of hip roll, shoulder roll, ankle roll and
     elbow (factor needed 1.0); waist pitch would need 1.11 (the reference exceeds the URDF limit), moot with waist out of the AMP
     input and default 0.
  f. Actuators: implicit PD, no delay (G1 as run); mc_20261024 gains, limits, armature unchanged. `agibot_x2.py` docstring corrected.
  g. Options a,c,d,s as run 3 plus f (P 8) and r (0.5), schedule from iteration 0.
  Installed at 03:06: `staged.py` (with e, f, r; superset, so run 3's resume behaves the same), `fixed.py`, task registration,
  `supervise_x2.py` (moves its STOP flag to `_trash`); old files in `_trash/2026-11-12/` with notes. `launch_next.sh` already
  had the Traceback fallback. Not verified: none of the run-4 config has been constructed in Isaac before this launch; the
  collision census is MuJoCo hulls (PhysX simplifies hulls); the MuJoCo port and `export_for_eval.py` must read run 4's env.yaml
  (they do: both are env.yaml-driven) but implicit PD is not modelled differently in MuJoCo (explicit PD there).

- 02:55 **Reference-state starts.** The paper's G1 task does not use them: every episode starts from the default pose
  (`reset_root_state_uniform` +-0.1 m, +-0.1 rad yaw, +-0.2 velocities; `reset_joints_by_offset` +-0.15 rad), so the share is 0%
  (`InstinctLab/.../parkour/config/parkour_env_cfg.py` EventCfg; the G1 config changes it only in its PLAY variant). Our X2 task
  is the same (run 3 env.yaml: `reset_base`, `reset_robot_joints`). InstinctLab has the machinery (`reset_robot_state_by_reference`,
  used by the shadowing tasks), and the parkour scene has a motion reference (v2 walking clips, start 0-90% into a clip).
  **Option r prepared** in `staged.py.next` (with e and f): a share of resets (HIKING_X2_REF_START, default 0.5, held to
  `hold`, linear to 0 at `end`) takes joint positions/velocities, roll/pitch and body velocities from the reference frame; x, y and yaw
  stay those of the normal reset (nobody placed off the tile or into an obstacle), height = clip pelvis height + 3 cm, velocities
  rotated to the robot's heading. CPU test `code/x2_port/test_refstart.py` passes (stubs: pose, heading, velocity rotation,
  share 0.5/0.5/0.25/0 at 500/1,000/1,500/2,000 with stage 1000,2000; only selected robots written). Not verified: in Isaac
  (reference joint order assumed = robot's as in the shadowing tasks; a mid-stride pose at clip height on rough tiles may touch the
  ground; `ensure_link_below_zero_ground` is off in our reference config).
- 02:55 **Torque saturation of `model_250` in MuJoCo, before the fall only** (`perception-agents-work/hiking_sim2sim_x2/checks/
  torque_sat_v3_250.{py,txt}`): flat (484 physics steps to the fall at 2.42 s): left ankle pitch 3%, right ankle pitch 2%, nothing
  else above 0.5%; stairs (344 steps, fall at 1.72 s): left ankle pitch 8%, right 3%. No hip, knee or waist saturation. The
  clip-wide count (2,305 of 2,800 physics steps with some joint saturated) is almost all after the fall, lying on the ground.

- 03:05 **Prepared, not applied: options e (air time) and f (fall penalty)** in `files_instinctlab/tasks/parkour/config/x2/staged.py.next`
  (the running `staged.py` is untouched). f = Isaac Lab's own `is_terminated_term` on base_contact|bad_orientation|root_height
  (time-outs and `stalled` not penalised, several fall terms in one step clamped to 1), -P per fall, P = `HIKING_X2_FALL_PENALTY`
  (default 8), held to `hold` and linear to 0 at `end`; e now takes `HIKING_X2_AIR_EARLY` (default 4). Tested on the node CPU:
  `test_fall.py` (stub of the Isaac base class, schedule 1 / 0.5 / 0 at 500 / 1,000 / 1,500, stalled and time-out not penalised,
  weight -P/dt = -P per fall) and `test_staged.py` pass; py_compile passes. **Never constructed in Isaac** (the regex term lookup
  and a class-type reward term in this config are untested until the first start; the launcher's fallback catches a Traceback).
  Continuing from `model_500` works with changed options: the supervisor passes `HIKING_ITER_OFFSET` = checkpoint iteration, and
  every schedule (rewards and the style coefficient) reads it. With stage `500,1500` unchanged, all early values would start
  ramping at once, so a continuation should use stage `1000,2000` (every early value held 500 more iterations; continuous at 500).
  To apply: STOP run 3 after `model_500.pt`, move `staged.py` to `_trash/<date>/`, copy `staged.py.next` over it, `apply.sh`,
  start the same run name with `HIKING_X2_OPTS=a,c,d,e,f,s HIKING_X2_STAGE=1000,2000 HIKING_X2_FALL_PENALTY=8 HIKING_X2_AIR_EARLY=4`
  plus run 3's other variables (new start script). At that restart, also swap in `supervise_x2.py.next`.
- 02:50 The MuJoCo port's scripts (`sim2sim_mujoco_x2.py`, `validate_x2.py`, README with the login path replaced by `~`) copied to
  `code/x2_port/sim2sim/` (user allowed it: "It's just a script copy without any secrets"); run-1 versions in
  `_trash/2026-11-12/sim2sim_run1/`. Checked for login, addresses and keys before copying.

#### Walker, stander, lunger per control step under run 3's early values, with and without f (P = 8) and e (x 4)

Per-step reward = task terms (weight x dt x raw) + style (coefficient x D-reward, not times dt). gamma 0.99. Lunger: run 3 measured at
iterations 269-278 (84% falls, 140 steps; per-term episode sums divided by length). Stander: the earlier table's stander under run
3's scales, D-reward 0.57 (run 2 at 499), reset by `stalled` at 4 s (200 steps). Walker: the G1 at 5,000 under run 3's scales,
D-reward 0.51, episode 1,000 steps. Air time for a stepping gait estimated (half the time in single support, mean shorter mode time
0.15 s: 0.5 x 0.02 x 0.075 = 0.0008 per step); run 3 measures 0.00003 per step.

| | Lunger | Stander | Walker |
|---|---|---|---|
| task terms per step | +0.031 (alive 0.019, lin 0.025, yaw 0.017, heading -0.008, wait -0.006, rest -0.016) | -0.001 | +0.039 |
| style per step (0.075 x D) | +0.035 | +0.043 | +0.038 |
| air time per step, x 4 under e | 0.0001 | 0 | +0.003 |
| steps until the episode ends | 140 (fall) | 200 (stalled reset) | 1,000 |
| discounted value now (sum of 0.99^k x reward) | 0.066 x 75.5 = **4.98** | 0.042 x 86.6 = **3.64** | 0.077 x 100 = **7.7** |
| with e | 4.99 | 3.64 | 8.0 |
| with f, P = 8 (discounted 8 x 0.99^140 = 1.96) | **3.02** | 3.64 | 7.7 (a fall after 700 steps costs 0.007) |

Now the ordering is walker > lunger > stander. f with P above about 5.5 ((4.98 - 3.64) / 0.99^140) gives walker > stander > lunger;
P = 8 leaves a margin of 0.6. e alone adds about 0.3 to a walker's value and almost nothing to a lunger's or stander's, so it is
a weak lever by itself (the paper's weight 0.5 x dt x seconds is small; x 4 is still about 4% of a walker's per-step reward).
Caveats: the walker row is the G1, not an X2; the lunger's value assumes the policy keeps its 140-step falls; a larger P pushes
back toward standing (run 2's outcome), which option d's stall reset and wait penalty are there to stop.

- 02:45 **Run 3 at 250** (`report_line.py`, iterations 241-247): speed ratio 0.52 (0.36/0.70 m/s), both feet down 81%, air time
  0.0038/ep, wait -0.85, falls 84% (root_height 65%, bad_orientation 18%), stalled resets 9.6%, time-outs 7.5%, episode 148,
  style coef 0.075, D-reward 0.464, weighted style 0.035, level 0.18 (perlin_rough_stand 1.86, stairs and gaps 0.00). Trend
  50/100/150/200/247: episode 53/74/112/131/148, falls 108/101/94/87/84% (shares overlap, sum can pass 100%), speed ratio
  0.77/0.64/0.58/0.53/0.52. Stop rule hit literally (falls > 70%, episode < 200) and reported; **coordinator: keep running**.
  Their check at 500: continue if episode > about 300 and speed ratio > about 0.3; report at once (no change) if episode
  flat under 200 or speed ratio under 0.1; if air time is still under 0.01 at 500 the next lever is the air-time reward (option e).
- 02:45 **MuJoCo render of run 3 `model_250`** (desktop CPU, "MuJoCo, not the training simulator", nominal camera mount, no
  randomisation): `perception-agents-work/hiking_sim2sim_x2/x2v3_ckpt250_mujoco_{flat,stairs}_{chase,legs}.mp4`, CSV/JSON beside
  them, frame sheets in `checks/frames/x2v3_ckpt250_*.png`. Reading: **a lunge, not a gait**: feet stay planted (4 lift-offs on
  stairs, no step cycle), the body pitches forward and falls flat at 2.4 s (flat) and 1.7 s (stairs) after moving about 1 m;
  peak forward speed 0.9-1.7 m/s during the fall. Same picture as the Isaac numbers (episodes about 3 s, falls 84%, air time a tenth
  of the G1's). Export: `eval/x2_parkour_v3_250/` (ONNX vs torch 1.4e-6, card from the run's env.yaml).
- 02:40 `code/x2_port/export_for_eval.py` replaced by the env.yaml-driven version (the old one hard-coded run 1's gains and camera
  and failed on the current asset file: SyntaxError slicing `agibot_x2.py`); old one in `_trash/2026-11-12/` with a note.

- 02:05 Third Hiking agent resumed. Run 3 training (heartbeat 01:58 iteration 10, 0 Tracebacks, one trainer start since the
  01:50:59 supervisor start). `launch_next.sh` already falls back on a Traceback (checked: it greps the new part of train.log);
  nothing to change there. `supervise_x2.py` STOP handling: patched copy `code/x2_port/supervise_x2.py.next` (moves an old STOP to
  `_trash/<date>/` with a NOTES.txt line instead of `unlink`; py_compile passes). Not swapped in: the running supervisor keeps
  the old code; at the next supervisor restart that happens anyway, move `supervise_x2.py` to `_trash/<date>/` and copy `.next`
  over it, then `bash code/x2_port/start_x2_parkour_v3.sh`. `report_line.py` now also prints style coefficient, D-reward and
  weighted style reward (run 2 has no `style_coef` field: nan there; its weighted style is 0.25 x D-reward).

- 01:48 Correction: the 01:46 launch went back to run 2 after 20 s. Cause: `launch_next.sh` counted 'start pid' lines, and
  the supervisor's own "supervisor start pid" line also matched, so it saw a restart at once. Fixed to count ' start pid N from
  iteration' only. The stale STOP flag was moved to `_trash/2026-11-12/`. Relaunched at 01:47:44; run 3's supervisor started
  training from iteration 0 at 01:48:37. To do at the next restart: a Traceback trigger in the launcher's fallback, and
  `supervise_x2.py` should move its STOP flag to `_trash` instead of `unlink`.
- **01:46 Run 3 `x2_parkour_v3` launched** (`launch_next.sh x2_parkour_v3 a,c,d,s 500,1500 1 0.3 2 0.3`, tmux `launchnext`
  watching). Run 2 stopped cleanly at `model_1000.pt` through its own supervisor at 01:45:38; kept with all its checkpoints. At
  01:46:44 Isaac was still starting up (no iteration yet, no Traceback). The launcher falls back to run 2 if iteration 2 is not
  reached within 10 minutes; result in `runs/x2_parkour_v3/_supervisor/launch.txt`. Resume run 3: `bash
  code/x2_port/start_x2_parkour_v3.sh` (carries every option, `$HOME` path). `supervise_x2.py` now uses `Path.home()`.
- **Departures from the paper in run 3** (all scheduled: early value until 500, linear to the paper's value at 1,500): a)
  `is_alive` x0.3 and `action_rate_l2`, `feet_slide`, `dof_acc_l2`, `feet_flat_ori` x0.2; c) no robots told to stand, minimum
  forward command 0.6 m/s, targets at least 1.5 m away (whole run, not ramped); d) `dont_wait` x2, plus a `stalled` reset
  (commanded to move, average speed under 0.1 m/s after 4 s; off from 1,500); s) style coefficient 0.25 x0.3. Also on: camera
  mount randomisation (first time in Isaac). Plus the run-2 departures (chest RealSense camera, mc_20261024, motion v2).
- `code/x2_port/MOTIONS.md` written (motion references v1 and v2, the G1 release file, sha256s, build command, licences,
  backward and sideways clips still open). Copies of the build scripts are in `code/x2_port/motion/`.
- Isaac evaluation beside training: still not started (plan below), because no checkpoint walks yet. MuJoCo render: no run-2
  checkpoint moves (at 1,000: speed ratio about 0.02), so nothing has been rendered.

- 01:15 Relaunch planned at run 2's `model_1000.pt` (about 01:50 UTC), coordinator's choice (messages 00:58 and 01:10):
  **options a, c, d**, hold 500, end 1,500, camera randomisation on, **early `is_alive` x 0.3, early `dont_wait` x 2** (not x 4):
  `launch_next.sh x2_parkour_v3 a,c,d 500,1500 1 0.3 2`. Why: SSR's test t_b (moving commands only plus a narrow kernel, our b)
  still stood; its stander lives on posture and alive pay. Option b left out (it makes the tracking gradient sparse). A first plan
  with `is_alive` x 0.1 and wait x 4 was dropped: with it a stander's task reward per step is negative, so on task terms alone
  dying early beats standing. If SSR's t_a (staged schedule) shows a gait before the launch, the coordinator is told first.
- `dont_wait` early scale is now set by `HIKING_X2_WAIT_EARLY` (default 4), sixth argument of `launch_next.sh`.
- Stall reset (option d) against a fall: the task has no termination penalty (no `is_terminated` term; nothing in
  `TerminationsCfg` pays), so both simply end the episode and the cost is the lost future reward. No penalty was added.
- Stop at once and report if, at 250, more than about 70% of episode ends are falls and episodes are under 200 steps (run 2 at
  250: falls 37%, length 737).

#### Early reward per control step under run 3's values (before iteration 500), weighted (weight x 0.02 s x raw)

Sources: run 1's stander at 499 and the G1 walker at 5,000 (table "Which terms pay the standing X2"), run 2's raw `dont_wait`
(1.08 over all robots with 77% commanded to move, so about 1.4 for a robot that stands under a moving command), commands
0.6-1.0 m/s under option c. The faller is run 2 at iteration 50 (reward 1.6 over 71 steps, 100% falls, 0.023 per step with full
`is_alive`). **Found while doing this: the style reward is added to every PPO step as 0.25 x D-reward (instinct_rl
`ppo.py` `process_env_step`, coefficient 0.25, type quad) and is NOT multiplied by dt, so it is about 0.13-0.16 per step,
2-3 times all task terms together.** The trainer's printed "Mean reward" leaves it out.

| Term (early scale) | Stander | Walker at 0.6 m/s (G1 at 5,000) | Faller (about 71 steps) |
|---|---|---|---|
| `is_alive` (x 0.3) | +0.018 | +0.017 | +0.018 |
| linear tracking | +0.004 (exp(-v^2/0.25) mean over 0.6-1.0 m/s is 0.09) | +0.022 | |
| yaw tracking | +0.016 | +0.018 | |
| `dont_wait` (x 2) | -0.028 | -0.008 | |
| heading, torso tilt, joint limits | -0.009 | -0.005 | |
| step penalties (x 0.2) | -0.001 | -0.003 | |
| edges, stand_still, the rest | 0 | -0.002 | |
| **task terms in total** | **-0.001 (about zero)** | **+0.039** | **about -0.019** (0.023 - 0.042 for the alive cut) |
| style, 0.25 x D-reward | +0.14 (run 2 stander logs 0.57) | +0.13 (G1 0.51) | +0.09 (run 2 at 50: 0.36) |
| **all, per step** | **+0.14** | **+0.17** | **+0.07 for its 71 steps, then nothing** |
| discounted value (gamma 0.99), task terms only | about -0.1 | about +3.9 | about -1.0 |
| discounted value, with style | about +14 | about +17 | about +3.6 |

**With option s (style coefficient 0.25 x 0.3 = 0.075 early; coordinator 01:20, added to the launch):** stander -0.001 +
0.043 = **+0.042** per step; walker +0.039 + 0.038 = **+0.077**; faller -0.019 + 0.027 = **+0.008** for its 71 steps. Discounted
(gamma 0.99): stander about 4.2, walker about 7.7, faller about 0.4. Walking pays about 1.8 times standing, against 1.2
times without s.

Launch command, final: `launch_next.sh x2_parkour_v3 a,c,d,s 500,1500 1 0.3 2 0.3`. How option s works: staged.py patches
instinct_rl's `WasabiPPO.process_env_step` in the trainer process (installed from the task config's `__post_init__`). Before
each env step it sets `discriminator_reward_coef` = 0.25 x schedule(HIKING_ITER_OFFSET + steps / 24, 0.3). Every staged run
(with or without s) now logs `Mean step style_coef` and `Mean step style_reward_weighted` (coef x D-reward per step), and
these reach `metrics.jsonl` as `style_coef` and `style_reward_weighted`. CPU test: `code/x2_port/test_style.py <staged.py>`
(stand-in algorithm class; coefficient 0.075 early, 0.1625 at 1,000, 0.25 from 1,500; log keys present). Upstream
instinct_rl is not edited.

Ordering walker > stander > faller holds both with and without the style reward. The walker's lead over the stander is
+0.04 per step in task terms and about +0.03 with style (the stander's style reward is higher than the walker's). What
the coordinator called "alive pay" is mostly the style reward: 0.14 per step for being upright, against 0.06 for
`is_alive` at full weight. Cutting `is_alive` moves less than this suggests. Approximations: per-term numbers come from other
runs at other iterations. Terms that only a walker pays on rough terrain (edges, feet) are near zero on flat ground.
The style figures are trainer means, not per-robot values.
- 01:00 Staged task: new environment variable `HIKING_X2_ALIVE_EARLY` (default 0.3) sets option a's early `is_alive` scale;
  `launch_next.sh` takes it as a fifth argument and writes it into `launch.txt` and the generated `start_<run>.sh`.
  `test_staged.py` passes (CPU).
- 00:55 Node login removed from six scripts (`$HOME`): `bench_envs.sh`, `launch_next.sh` (also the generated start script
  carries a literal `$HOME`), `launch_v2.sh`, `pause_and_eval.sh`, `start_x2_parkour_v1.sh`, `start_x2_parkour_v2.sh`.
  `supervise_x2.py` is changed (`Path.home()`) right before the relaunch, not while its process runs. `bench_envs.sh` moves its
  scratch run folders to `tracks/hiking/_trash/<date>/` with a note in `NOTES.txt` instead of `rm -rf`.
- New `code/x2_port/report_line.py <run folder> <iteration>`: one line of speed ratio, double support, air time, wait penalty,
  falls, stalled resets, terrain level per type (mean of the last 10 iterations).
- G1 at equal samples (desktop log, read only; 4,096 robots, so G1 iteration = 2 x X2 iteration): at 500 air-time reward/ep
  0.040, wait -4.27, length 731, level 1.48, time-outs 40%; at 1,000: 0.049, -4.17, 687, 2.06, 50%; at 2,000: 0.038, -1.49,
  477, 1.61, 29%.
- Run 2 at 684: speed ratio 0.019, both feet down 92%, air-time reward/ep 0.011, wait -10.4, falls 8%, level 1.45 (stand
  column 5.8, mesh boxes 4.2, all stairs and gaps 0.06-0.08).

## What is running: run 2 (`x2_parkour_v2`), from scratch since 22:58 UTC

- Task `Instinct-Parkour-TorsoCam-Amp-X2-v0`, 8,192 robots, to 30,000 iterations, tmux `hiking_x2`, supervisor
  `code/x2_port/supervise_x2.py` (stall detection, auto-resume, checkpoint every 250, metrics, GPU log, backup manifest).
  Start or resume: `bash /mnt/filesystem-g4/tracks/hiking/code/x2_port/start_x2_parkour_v2.sh`.
  Stop: `touch /mnt/filesystem-g4/tracks/hiking/runs/x2_parkour_v2/_supervisor/STOP`.
- Run folder `/mnt/filesystem-g4/tracks/hiking/runs/x2_parkour_v2/` (`<start time>/model_<it>.pt`, `metrics.jsonl`, `gpu.csv`,
  `_supervisor/`). Heartbeat `/mnt/filesystem-g4/status/hiking.json`.
- First throughput (iteration 11): 19,600 samples/s, 10.0 s per iteration (collection 8.0 s, learning 2.0 s), 51 GB of 96 GB GPU
  memory. Slower than run 1 (23,000-25,000/s, 35 GB) because the depth input is twice as tall (36 rows instead of 18).
- No checkpoint yet (first at iteration 250, about 23:40 UTC).
- Run 1 (`x2_parkour_v1`) was stopped at 22:54 UTC at iteration about 650; its checkpoints `model_250.pt` and `model_500.pt` are
  kept. GPU idle between the runs: 237 s.

### Standing rules for this run (user, via the coordinator, 2026-11-11)

- No training stop whose only purpose is an evaluation, a recording or a render. Isaac evaluation only inside a restart that is
  needed anyway. `code/x2_port/pause_and_eval.sh` stays unused unless the coordinator says otherwise. Everything else comes from
  the training logs, the MuJoCo port on the desktop CPU, or the eval monitor.
- The run-2 versions of `export_for_eval.py` and the MuJoCo port exist only on the desktop
  (`C:/Users/PC/projects/perception-agents-work/hiking_sim2sim_x2/`): the upload to the shared filesystem was refused by the
  permission system and nobody copies them until the user allows it. The copies on the node (`code/x2_port/export_for_eval.py`,
  `code/x2_port/sim2sim/`) are the run-1 versions; do not export a run-2 checkpoint with them. Run-2 renders are made on the
  desktop and shown from there.
- Nothing under `/mnt/filesystem-g4` is deleted: unwanted files are moved to `tracks/hiking/_trash/<date>/` with a one-line
  note (`/tmp` on the node is free to use). Helpers get the same rule. `code/x2_port/bench_envs.sh` still removes its scratch
  run folders with `rm -rf` (it did so on 2026-11-11 before this rule): change that to a move before it is used again.
- Run 2 runs to iteration 1,000 before any judgement; one line at 500 and 1,000 (signals table below).
- MuJoCo, run 2 `model_250` (desktop, 14 s flat): stands and shuffles in place drifting backwards, 0.54 m net, mean forward
  speed -0.04 m/s against 0.74 commanded, both feet down 87%, no fall; actions did not depend on the depth image.

### Plan: an Isaac evaluation beside the running training (proposed, NOT started; the coordinator decides)

What is known from this node (measured unless marked "estimate"):

- GPU memory: training run 2 holds 56.9 GB of 95.6 GB at iteration 314 (51 GB at start, so it grows), leaving about 39 GB.
  Training-style memory against robot count (run 1 task): 21.0 GB at 4,096, 34.9 GB at 8,192, 49.1 GB at 12,288, i.e. about
  7 GB fixed plus 3.4 GB per 1,000 robots including the PPO buffers. An evaluation has no PPO buffers; estimate 8 GB at 200
  robots and 10-11 GB at 1,000 (never measured: the 64-robot checks did not log memory).
- GPU compute: training already uses the card at 50-98% and is compute-bound (profile above). A second instance takes its share
  for as long as it runs: expect training throughput to drop by roughly the evaluation's share, not for free.
- Start-up: an Isaac start takes 3-5 minutes (terrain generation, URDF conversion, ray-cast mesh build) and briefly used 7.8
  CPU cores; the node has 24 cores, training uses about 1.8, RAM 197 GB free. CPU and RAM are not the limit.
- Two instances on one card: never tried on this node. Each is its own process with its own PhysX GPU context; the shader
  cache (11 GB inside the venv's `isaacsim/kit/cache`) is already warm, so the second instance should only read it, but two
  instances writing it at once is untested. Converted robot USD files go to separate time-stamped folders in `/tmp/IsaacLab`.
- What a failure of the second instance could do to the first: an out-of-memory error normally hits the process that asks for
  memory, so the evaluation would die, but training's own memory use still grows (51 to 57 GB so far) and an allocation of its own would fail if the card is full
  at that moment (the supervisor then resumes from the newest checkpoint, losing up to 250 iterations, about 40 minutes). A
  driver-level fault (Xid) would take both. The supervisor's own scripts are safe: it only refuses to START while another GPU
  job runs, so if training crashes during an evaluation it waits for the evaluation to end and then resumes.
- Things in my own scripts that must not be used with two instances: `pause_and_eval.sh`, `launch_v2.sh` and `restart_at.sh`
  kill every GPU process on the card.

Proposed test (one shot, about 12 minutes, started by hand right after a checkpoint is written so a training crash loses nothing):

1. Wait for `model_<N>.pt` to appear. Record 3 minutes of training throughput from the log (samples/s, iteration time) and GPU
   memory: "before".
2. Start one paper-eval worker on that checkpoint: one terrain, 200 robots, Table 2 only (751 steps), `nice -n 10`, wall-clock
   limit 600 s (`timeout`), its own log, no kill-all anywhere. Refuse to start if free GPU memory is under 25 GB.
3. A watchdog loop every 2 s: if free GPU memory falls under 8 GB, or the training log stops advancing for 120 s, kill the
   evaluation process (by its pid only).
4. Record training throughput and GPU memory "during" (start-up phase and stepping phase separately) and for 5 minutes "after";
   check the training log for PhysX errors and the supervisor log for a restart.
5. Pass if: training never stopped or restarted, no PhysX error, throughput after equals before within 3%, the evaluation
   finished inside the limit. Report the throughput loss during, peak memory, and the evaluation's own start-up and run time.
6. Only if that passes: 1,000 robots, then several terrains in sequence (each is a separate start, so most of the cost is
   start-up; nine terrains would take about 40 minutes of shared GPU).

Until the coordinator decides, Hiking evaluation stays on the MuJoCo port (desktop CPU) and the training logs.

### Next run, ready to launch at run 2's iteration 1,000 (the coordinator names the options before 01:50 UTC)

Task `Instinct-Parkour-Staged-Amp-X2-v0` (`code/x2_port/files_instinctlab/tasks/parkour/config/x2/staged.py`): the run-2 task plus
options that are all off by default. **Never constructed in Isaac**; the schedule function and the two new kernels pass a CPU unit
test (`code/x2_port/test_staged.py`); everything else is first exercised at launch.

Launch (watched, falls back to resuming run 2 if the new task does not reach its third iteration; stops run 2 through its own
supervisor, kills nothing else):
`tmux new -d -s launchnext "bash /mnt/filesystem-g4/tracks/hiking/code/x2_port/launch_next.sh x2_parkour_v3 <options> <hold,end> <camera randomisation 0 or 1>"`,
for example `... x2_parkour_v3 b,c,d 500,1500 1`. It writes `runs/x2_parkour_v3/_supervisor/launch.txt` and a matching
`code/x2_port/start_x2_parkour_v3.sh` for later resumes. Expected GPU idle: about 1 minute plus the 4-5 minute start-up.

Every schedule holds its early value until iteration `hold` (default 500), moves linearly to the paper's value at `end` (default
1,500) and stays there. Schedules survive restarts (the supervisor passes the resume iteration as `HIKING_ITER_OFFSET`).

| Option | What changes | Early value | Paper's value (after `end`) |
|---|---|---|---|
| a | `is_alive` scaled | x 0.3 | x 1 (weight 3.0) |
| a | `action_rate_l2`, `feet_slide`, `dof_acc_l2`, `feet_flat_ori` scaled | x 0.2 | x 1 |
| b | linear-tracking kernel std | 0.25 (zero speed pays 0.002 for a 0.63 m/s command) | 0.5 (pays 0.20) |
| b | yaw tracking multiplied by forward progress v_x / (0.5 x commanded v_x), 0..1, for robots commanded to move | full gate | no gate |
| c | robots told to stand | 0% (whole run, not scheduled) | 5% |
| c | minimum commanded speed | 0.6 m/s (whole run) | 0.45 m/s |
| c | target redrawn until at least 1.5 m from the robot (whole run) | on | targets anywhere, about 15% of commands zero from the start |
| d | `dont_wait` scaled | x 4 | x 1 (weight -0.5) |
| d | termination `stalled`: commanded to move, average speed since episode start under 0.1 m/s after 4 s | on | off |
| e | `feet_air_time` scaled | x 4 | x 1 (weight 0.5) |

- Option c is not ramped back: it is a config value read once. Returning to the paper's command settings means a restart
  without c; say so if it is used.
- Camera mount randomisation (`HIKING_X2_CAM_RAND=1`, last argument of the launch script) can be switched on in the same launch.
- Curriculum (checked 2026-11-12): a robot moves up when its episode's linear tracking score exceeds 0.6 and down under 0.3; a
  zero command scores 1.0 by standing. 25% of robots have a zero command (5% told to stand, 5% stand column, about 15% with the
  target inside the 0.4 m stop radius), and they are what raises the mean level while nobody walks: at iteration 492 the stand
  column is at 7.45, mesh boxes 2.83, slope 2.57, rough 2.36, boxes 1.09, and all stairs, platforms and gaps at 0.05-0.08. Not a
  port defect (no robot-specific number in the rule; same code as the G1 run). Option c removes most of it.
- The three scripts that killed every GPU process (`launch_v2.sh`, `pause_and_eval.sh`, `restart_at.sh`) no longer do.

### What run 2 changes against run 1 (everything else identical; the paper's terrains, commands, rewards, curriculum, PPO)

| Item | Run 2 | Why |
|---|---|---|
| PD gains | plant `mc_20261024` (table below); armature, effort, default pose and action scale unchanged | user: "Use the ones used for supermimic training last iteration" |
| Depth camera | chest RealSense: on `torso_link` at (0.131, 0.0, 0.259), 61.6 deg below horizontal, 87 x 58 deg, all 36 rows kept, the paper's left/right crop (36 x 32 per frame), range 0-3 m | the user's camera; a departure from the paper's camera at the user's direction |
| Depth encoder input | 8 frames of 36 x 32 (run 1: 18 x 32); conv output 4 x 36 x 32 = 4,608 into the same head | follows from the crop |
| Motion reference | v2: standing cut to 3% (run 1 until iteration 500: 21%) | measured defect, see "Standing" |
| Torso tilt term | on `torso_link` from the start (run 1: from iteration 250) | |
| Logging | commanded vs actual speed, support shares, step rate, terrain level per type, from iteration 0 | |

### Camera (run 2): calibration and acceptance

- Reference: the user's infrared frame, `docs/camera/` in this repo. Acceptance: at the standing pose on flat ground the toe tips
  are just inside the bottom edge, a floor line 1.0 m ahead of the toes is at 27% (+-3%) of the image height from the top, and a
  0.55 m line is near 52%.
- Solved for the simulated standing pose (`code/x2_port/chest_solve.py`), lens 3 cm proud of the chest shell (shell front from the
  torso mesh: x = 0.101 m at that height): **lens 1.055 m above the ground, 61.6 deg below horizontal, 1.1 cm ahead of the toe
  tips**. Result: toe line at 99.9%, 0.55 m at 52.1%, 1.0 m at 26.2% from the top; top edge sees 1.66 m ahead of the toes.
  Picture with the lines marked: `state/hiking_v2_camera_check.png` (MuJoCo preview, not a training-simulator frame); more in
  `/mnt/filesystem-g4/tracks/hiking/renders/camera_check/` (colour view, depth 0-3 m, the 36 x 32 policy view, `camera_check.txt`).
- The coordinator's reading (0.90 m, 53 deg, lens 12 cm behind the toes) fits the two main landmarks too but puts the 0.55 m line
  at about 50%; the mesh says the chest shell is only 1.3 cm behind the toe tips in the default pose. The user's measured height
  and angle would settle it; a change is two numbers in `x2_parkour_target_amp_cfg.py` (`X2_TORSO_CAMERA_POS`, `..._PITCH_DOWN`).
- The view reaches 1.66 m, not the 2 m the user mentioned earlier: with a 58 deg vertical field of view, seeing the toes and 2 m
  at once needs the camera about 1.25 m high.
- Earlier candidates, for the record (flat ground, standing, ahead of the toe tips): run 1 head camera, 40 deg, paper crop:
  0.40 to 1.27 m. Paper's G1 camera placed on the X2 at the same height (1.24 m), 48 deg, paper crop: 0.23 to 1.01 m (that point is
  inside the X2's head shell at the G1's 0.049 m forward offset; the head's face is at 0.076 m). Same height, 0.10 m forward,
  60 deg, all rows: 0.01 to 2.00 m. G1 itself (not computed by FK; about 1.24-1.27 m high): about 0.3 to 1.1 m from the camera.
- The SSR track's estimate for the same camera: `torso_link` (-0.026, 0, 0.125), 53.5 deg (toe line 96%, 0.55 m at 49.5%, 1.0 m
  at 27.1%, -0.02 to 1.85 m). At that height (0.92 m) the mesh puts the chest shell front at x = +0.096 m, so that point is
  12 cm inside the torso; the two poses differ by 16 cm forward, 13 cm in height and 8 deg. One of them must be adopted by both.
- Mount randomisation (user: the real lens height is "somewhere in between"): per robot, redrawn at each episode reset, height
  -10.5 to +5 cm (0.95 to 1.105 m), forward +-2 cm, pitch +-4 deg, roll and yaw +-2 deg (`X2_CAMERA_MOUNT_RANGES`, InstinctLab's
  `randomize_camera_offsets`). **Written but OFF in the running process**: it has never run in Isaac, and an unattended
  auto-resume must not pick up untested code. Turn it on at the next watched restart:
  `HIKING_X2_CAM_RAND=1 bash code/x2_port/start_x2_parkour_v2.sh` and watch the first iterations. Until then run 2 trains with
  the fixed pose. Not yet in the adapter card.
- Not done: an Isaac-rendered frame at the chest pose (the frames
  dumped by the pre-launch check are from the superseded brow-height pose and are marked SUPERSEDED in `runs/_smoke_v2/`).
- Row order (checked on run 1's Isaac recording): row 0 is the top of the image (far), the last row is nearest the feet.

### Gains applied in run 2, read back from the Isaac articulation (`runs/_smoke_v2/applied_*.json`)

| Joints | kp / kd applied (mc_20261024) | kp / kd run 1 (armature rule) | armature | effort N.m | velocity rad/s | action scale rad |
|---|---|---|---|---|---|---|
| hip pitch | 100 / 4 | 114.8 / 7.31 | 0.02908 | 120 | 11.94 | 0.261 |
| hip roll, hip yaw | 100 / 3 | 114.8 / 7.31 | 0.02908 | 120 | 11.94 | 0.261 |
| knee | 150 / 5 | 114.8 / 7.31 | 0.02908 | 120 | 11.94 | 0.261 |
| ankle pitch | 40 / 3 | 34.9 / 2.22 | 0.00884 | 36 | 13.09 | 0.258 |
| ankle roll | 30 / 2 | 12.4 / 0.79 | 0.003133 | 24 | 15.08 | 0.485 |
| waist yaw | 40 / 8 | 114.8 / 7.31 | 0.02908 | 120 | 11.94 | 0.261 |
| waist pitch, roll | 40 / 5 | 12.4 / 0.79 | 0.003133 | 24 | 13.09 | 0.485 |
| shoulder pitch | 30 / 1 | 34.9 / 2.22 | 0.00884 | 36 | 13.09 | 0.258 |
| shoulder roll | 20 / 1 | 34.9 / 2.22 | 0.00884 | 36 | 13.09 | 0.258 |
| shoulder yaw | 20 / 1 | 12.4 / 0.79 | 0.003133 | 24 | 15.08 | 0.485 |
| elbow, wrist yaw | 50 / 1 | 12.4 / 0.79 | 0.003133 | 24 | 15.08 | 0.485 |
| wrist pitch, roll | 20 / 1 | 16.8 / 1.07 | 0.00425 | 6 | 20.94 | 0.089 |
| head yaw (held at 0) | 3.4 / 0.114 | 16.8 / 1.07 | 0.00425 | 2.6 | 6.02 | not commanded |
| head pitch (held at 0) | 10 / 0.8 | 16.8 / 1.07 | 0.00425 | 0.6 | 6.28 | not commanded |

- Every applied kp and kd equals the plant file's value (first-match substring rule reproduced); no joint differs from the intent.
- Action scale = 0.25 x effort / (armature x (2 pi 10)^2), from the armature, not from the new stiffness. Actuator type:
  delayed PD, 0-2 physics steps of delay, torque clipped to effort. Default pose: hip pitch -0.312, knee 0.669, ankle pitch
  -0.363, elbow -0.6, shoulder pitch 0.2, shoulder roll +-0.2, the rest 0.
- Run 1 used the armature-rule column; its checkpoints must be played with those gains.

### Model audit (Isaac's conversion of the URDF against the MJCF)

- Total mass: Isaac 41.967 kg = URDF link sum 41.967 kg; **MJCF 43.475 kg**. The whole difference is the pelvis: 3.523 kg in the
  URDF and Isaac, 5.032 kg in the MJCF (+1.509 kg). Every other body agrees within 5 g. Which is right is not known to me.
- Joint limits agree except waist pitch (URDF +-0.314, MJCF +-0.349) and waist roll (URDF +-0.488, MJCF +-0.524).
- Armature per joint: identical in Isaac and the MJCF.
- Fixed links (IMUs, cameras, lidar) are merged into their parents by the converter with their mass; the body masses above
  already include them. Inertia tensors were not compared.
- Feet: Isaac uses the convex hull of the foot mesh, sole 0.073 m below the ankle; the MJCF has 12 spheres of 5 mm per foot.
- Self-collision: on, minus the 5 always-overlapping pairs.
- The URDF used in training is the shared one minus the massless `base_link` and the XML declaration.

## Standing: what was measured (run 1) and what run 2 does about it

- Run 1 at checkpoint 500 stands and does not walk: Isaac Stair Up test 0 of 200 reached, 198 still standing; whole-population
  log at iterations 520-615: commanded 0.63 m/s, actual 0.00 m/s, both feet down 95%.
- MuJoCo run of `model_500` (CPU): stands, leans back 9 deg, no steps, 0.15 m in 14 s. Files and validation of the port:
  `tracks/hiking/renders/ckpt500_mujoco/`, `tracks/hiking/code/x2_port/sim2sim/`.
- At checkpoint 500 the depth encoder's output is exactly zero for all 50 recorded depth stacks: the policy ignored depth.
- Style data: reference v1 was 21% standing (30% under 0.1 m/s; median speed 0.32 m/s); the paper's G1 file is 5% (7.5%; median
  0.41). Reference v2 (`data/motions/hiking_x2/x2_parkour_motion_without_run_v2_retargetted.npz`, 400 s): 3.3% standing kept on
  purpose, median 0.53 m/s, 67% of frames inside the 0.45-1.0 m/s command range. Script `build_x2_motion.py`, numbers by
  `motion_stats.py` in the same folder.
- The G1 log (iterations 0-5,098) has no phase like run 1's: full-length episodes never above 69%, wait penalty never below -7.0,
  air-time reward never below 0.015 after iteration 150. Run 1 at 499: 89%, -9.6, 0.008.
- Reward balance per step (weighted): standing X2 at 499 earns 0.048, the walking G1 at 5,000 earns 0.068; the gaps are tracking
  (-0.010), alive (-0.009) and the wait penalty (-0.006). Standing does not out-earn walking on the task terms; the style reward
  (discriminator reward about 0.5-0.65 per step before its 0.25 coefficient) is paid in full for standing when a fifth of the
  reference stands.
- Ankle action scale and default pose equal the X2's working values in gwbc-v6 (same formula and numbers).
- Run 1 was restarted from `model_500` with reference v2 at 22:28; in the 150 iterations it then ran the speed ratio stayed at 0.
  That is too short to judge the fix, and run 2 starts with v2 from scratch.
- Run 2, iteration 11: speed ratio 0.24, both feet down 51% (untrained policy flailing; not a result).

### Which terms pay the standing X2 (answer to the SSR track's question)

Weighted reward per control step (episode sum / episode length), from the per-term logs:

| Term | X2 run 1 at 499 (stands) | G1 at 500 | G1 at 5,000 |
|---|---|---|---|
| is_alive | +0.0483 | +0.0577 | +0.0568 |
| track_ang_vel_z_exp | +0.0160 | +0.0132 | +0.0180 |
| track_lin_vel_xy_exp | +0.0116 | +0.0201 | +0.0218 |
| stand_still | +0.0000 | +0.0010 | +0.0022 |
| volume_points_penetration (edges) | -0.0000 | -0.0041 | -0.0045 |
| feet_slide | -0.0010 | -0.0038 | -0.0032 |
| dof_acc_l2 | -0.0010 | -0.0023 | -0.0018 |
| feet_flat_ori | -0.0012 | -0.0021 | -0.0023 |
| dof_pos_limits | -0.0021 | -0.0001 | -0.0004 |
| action_rate_l2 | -0.0030 | -0.0050 | -0.0068 |
| pelvis_orientation_l2 (torso tilt on the X2) | -0.0031 | -0.0010 | -0.0009 |
| heading_error | -0.0040 | -0.0069 | -0.0033 |
| dont_wait | -0.0096 | -0.0058 | -0.0039 |
| all task terms | +0.0478 | +0.0569 | +0.0679 |
| style (discriminator) reward per step, before its 0.25 coefficient | 0.646 | 0.436 | 0.511 |

- What pays the stander: staying alive (the largest term by far); yaw tracking, which it earns in full by turning on the spot
  toward the target; and about half of a walker's linear-tracking reward, because the exponential kernel (std 0.5) still pays
  exp(-0.63^2 / 0.25) = 0.2 at zero speed. It also avoids about 0.011 per step of penalties a walker pays (edge penetration, foot
  slide, action rate, joint acceleration, foot tilt). The wait penalty (-0.0096) and lost tracking do not outweigh these by much:
  the stander is at 0.048 against 0.057 for the G1 at the same iteration.
- The style reward is where the X2 stander does better than the G1 ever did (0.65 against 0.44-0.51): consistent with a reference
  that was one fifth standing. How the 0.25-weighted style reward compares in size with the task terms inside the trainer was not
  traced.
- Did the G1 pass through a shuffle? Yes, by its air-time reward: per 100 steps 0.0054 at 500, 0.0071 at 1,000, 0.0088 at 5,000,
  against roughly 0.05-0.06 at 20,000 (0.56 per episode). So for its first 5,000 iterations the G1 moved with very short swing
  phases while falling in 34-71% of episodes. It never stood: the X2 at 499 has 0.0008 per 100 steps, seven times lower again.

### Rest pose with zero actions (answer on the lean-back question)

- Isaac, pre-launch check of the run-2 task (mc_20261024 gains, 64 robots on the training terrain with reset noise, 2.2 s):
  1.05 episodes ended per robot, 63 of 71 by tilt over 1 rad. Run 1's gains under the same check: 1.86 per robot in 3 s; the G1
  task: 2.45. So with zero actions the default pose is not held in Isaac under either plant; it falls rather than settling into
  a lean. Pelvis pitch at rest was not recorded (direction of the fall not logged).
- MuJoCo, stock MJCF with sphere soles, flat ground, no noise (`rest_pose.py`): falls forward at 1.4 s under both plants. In
  the default pose the centre of mass is 0.016 m ahead of the pelvis origin and the sole contact points span -0.083 to +0.121 m
  (centre 0.019 m), so the pose is balanced only if the joints hold it exactly; knees sag 0.044 rad under mc gains.
- This does not reproduce the SSR track's "settles 7.6 deg leaning back" (its model holds the waist stiffly and uses its own
  feet). The 9 deg backward lean of run 1's checkpoint 500 is the policy's choice of stance, not the rest pose.
- Nothing was changed in the default pose.

### Run 2 signals (trainer's printed means over all 8,192 robots; not a score)

| Iteration | Reward | Episode length | Terrain level | Speed ratio (commanded / actual m/s) | Both feet down | Air-time reward | Wait penalty | Episodes ending in a fall | Style reward |
|---|---|---|---|---|---|---|---|---|---|
| 50 | 1.6 | 71 | 0.00 | 0.45 (0.62 / 0.28) | 66% | 0.004 | -0.41 | 100% | 0.36 |
| 100 | 3.1 | 104 | 0.00 | 0.28 (0.61 / 0.17) | 72% | 0.005 | -0.82 | 100% | 0.41 |
| 150 | 10.9 | 282 | 0.09 | 0.17 (0.62 / 0.11) | 79% | 0.009 | -2.25 | 87% | 0.44 |
| 200 | 22.6 | 573 | 0.25 | 0.02 (0.63 / 0.01) | 81% | 0.014 | -6.16 | 58% | 0.47 |
| 249 | 35.1 | 736 | 0.40 | 0.02 (0.62 / 0.01) | 84% | 0.016 | -8.31 | 35% | 0.50 |
| G1 at 500 (equal samples to X2 250) | 43.9 | 731 | 1.48 | not logged | not logged | 0.040 | -4.27 | 60% | 0.44 |
| 400 | 57.7 | 939 | 0.88 | 0.01 (0.63 / 0.01) | 90% | 0.011 | -10.09 | 12% | 0.55 |
| 499 | 63.5 | 950 | 1.20 | 0.01 (0.63 / 0.01) | 91% | 0.009 | -10.70 | 10% | 0.57 |
| G1 at 1,000 (equal samples to X2 500) | 44.5 | 687 | 2.06 | not logged | not logged | 0.049 | -4.17 | 50% | 0.47 |

**Run 2 is going the same way as run 1: as the robots learn to stay up, forward speed goes to zero** (speed ratio 0.45 at
iteration 50 while everything falls, 0.02 from iteration 200 on). So the standing is not explained by the old gains, the head
camera or the standing share of the motion reference alone: all three were changed and the pattern is the same at 250. The early
"speed" at iterations 50-100 is robots falling forward, not walking. Checkpoints: `runs/x2_parkour_v2/20261111_225809/model_250.pt`, `model_500.pt`. At 500 the robots stand through almost the whole
episode (950 of 1,000 steps, 10% falls) at 1% of the commanded speed; air time is falling, not rising (0.016 at 249, 0.009 at 499).

### Does the style reward tell standing from walking? (offline, CPU, `code/x2_port/score_discriminator.py`)

Each checkpoint's discriminator scored on 10-frame windows built from the motion file as the task builds them; reward =
clamp(1 - 0.25 (D - 1)^2, 0); the policy gets 0.25 x this.

| Discriminator | Reference walking (> 0.3 m/s) | Reference standing stretches | Synthetic perfect stand (default pose, zero velocity) | Stand as share of walking |
|---|---|---|---|---|
| X2 run 2, iteration 250 (reference v2) | 0.917 (D +0.44) | 0.811 | 0.720 (D -0.06) | 78% |
| same discriminator on reference v1 | 0.909 | 0.823 | 0.720 | 79% |
| G1, iteration 500 | 0.948 (D +0.55) | 0.888 | 0.875 (D +0.29) | 92% |
| G1, iteration 5,000 | 0.923 | 0.840 | 0.835 | 90% |
| G1, iteration 20,000 (walks stairs) | 0.874 | 0.771 | 0.750 (D 0.00) | 86% |

- The style reward separates standing from walking only weakly on both robots: a motionless robot in the default pose collects
  78% (X2) or 86-92% (G1) of what real walking collects. The discriminator's outputs stay between about -0.1 and +0.6 (targets
  are +1 for the reference and -1 for the policy), and with this reward form D = 0 already pays 0.75.
- **The G1 walked with a discriminator that separates even less than the X2's.** So the style prior is not what makes the G1
  walk and the X2 stand; the difference has to be in the task terms or the robot (staying alive against the wait penalty and
  tracking, and how costly a first step is for each robot).
- Which inputs carry what little separation there is (X2): joint positions (walking windows with stand joint positions drop
  0.917 -> 0.811) and base linear velocity (0.917 -> 0.873); joint velocities, gravity and angular velocity carry almost none
  (0.914-0.924). Same ranking on the G1.
- Logged discriminator terms are alike on both: weight decay 92 (X2 run 1 at 260) against 109-117 (G1), gradient penalty 0.043
  against 0.044, loss 0.36 against 0.21-0.38. The weight-decay figure is not an anomaly.
- The policy's own windows were not scored (no rollout of run 2 available on the CPU yet); the trainer logs 0.50 for them at 249,
  below the synthetic stand's 0.72, so the discriminator does mark the policy's actual standing as less reference-like than a
  perfect stand.
- Approximations: reference velocities are finite differences of the file without the task's interpolation; clip seams skipped.

### Terms that correspond to a "get off the ground" recipe (for carrying one over from SSR as a listed deviation)

| Role | Term here | Value now |
|---|---|---|
| Survival reward | `is_alive` | +3.0 per step (x 0.02 s): the largest term, about 0.06 per step |
| Penalty for not moving when told to | `dont_wait` | -0.5 (about -0.011 per step for a stander) |
| Penalty for moving when told to stand | `stand_still` | -0.3, offset 4.0 |
| Linear tracking | `track_lin_vel_xy_exp` | +2.0, kernel exp(-err^2 / 0.5^2): pays 0.2 at zero speed for a 0.63 m/s command |
| Yaw tracking | `track_ang_vel_z_exp` | +2.0, same kernel; a stander earns it in full by turning on the spot |
| Heading | `heading_error` | -1.0 |
| Swing reward | `feet_air_time` | +0.5, only when commanded above 0.15 m/s |
| Gait penalties a first step pays | `action_rate_l2` -0.005, `feet_slide` -0.4, `dof_acc_l2` -1.25e-7, `feet_flat_ori` -0.4, `volume_points_penetration` -4.0 | |
| Style | discriminator reward | x 0.25 |
| Curriculum | `tracking_exp_vel`, thresholds 0.3 / 0.6 | robots that stand stay on the easy rows |
| Command | forward 0.45-1.0 m/s to a target, 5% told to stand, resampled every 8-12 s | |

**Watch in run 2** (`metrics.jsonl`): `Curriculum/walk_monitor/speed_ratio`, `double_support`, `Episode_Reward/rewards_feet_air_time/sum`,
`rewards_dont_wait/sum`, at 250, 500, 1,000, 2,000. G1 reference at 500 (4,096 robots): air time 0.040, wait penalty -4.3,
full-length episodes 40%, terrain level 1.48.

## Run 1 (`x2_parkour_v1`, stopped 22:54 UTC): record

- **X2 paper-terrain run `x2_parkour_v1`**: task `Instinct-Parkour-Target-Amp-X2-v0`, **8,192 robots** since iteration 250
  (4,096 before), to 30,000 iterations, first started 2026-11-11 20:22 UTC. tmux session `hiking_x2` on the node, under
  `code/x2_port/supervise_x2.py` (stall detection 900 s without log output, kill and resume from the newest checkpoint, one GPU
  job, checkpoint every 250 iterations, metrics and backup housekeeping).
- Heartbeat: `/mnt/filesystem-g4/status/hiking.json` (every 10 s: state, iteration, reward, episode length, terrain level,
  newest checkpoint, GPU, robot count). During a pause for the paper eval the state reads "paper eval ..., training paused".
- Run folder: `/mnt/filesystem-g4/tracks/hiking/runs/x2_parkour_v1/` (one timestamped folder per start or resume with
  `model_<it>.pt`, `params/`; `metrics.jsonl`, `gpu.csv`; `_supervisor/train.log`, `_supervisor/supervisor.log`).
- Newest checkpoint: `checkpoint` in the heartbeat. `model_250.pt` is the only one I have exported so far.
- History of stops: 20:55-21:22 robot-count benchmark (27 min, too long; limit is now 10 min); 21:26-21:36 restart to apply the
  torso-orientation term and the new supervisor (resumed from `model_250`, about 20 iterations redone).
- 22:13-22:18 (291 s): paper-eval plumbing test at checkpoint 500, training restarted by itself from `model_500`.
- Nothing else is queued on the node.

## Run 1 early signals (trainer's own printed means; not a score)

| Iteration | Robots | Mean reward | Episode length (of 1,000 steps) | Terrain level | Velocity-tracking term (0-1) | Episodes ending in a fall |
|---|---|---|---|---|---|---|
| 25 | 4,096 | 2.0 | 66 | 0.00 | 0.03 | 100% |
| 100 | 4,096 | 5.2 | 122 | 0.00 | 0.05 | 100% |
| 150 | 4,096 | 12.5 | 420 | 0.03 | 0.16 | 93% |
| 200 | 4,096 | 34.0 | 728 | 0.16 | 0.30 | 57% |
| 249 | 4,096 | 41.6 | 832 | 0.30 | 0.29 | 39% |
| 364 (after two resumes) | 8,192 | 44.5 | 850 | 0.75 | n/a | n/a |
| 499 | 8,192 | 58.4 | 1,000 | 1.00 | 0.29 | 11% |

- The G1 reproduction's log at the same iterations (desktop run, 4,096 robots): 250: reward 31.9, length 645, level 0.55,
  tracking 0.35; 500: 43.9, 731, 1.48; 1,000: 44.5, 687, 2.06; 3,000: 46.3, 727, 1.93. The X2 curve has the same shape so far.
- After each resume the printed means dip (reward 4.5, length 129) and recover within about 110 iterations (statistics restart,
  robots re-spread over terrain rows). Terrain level right after a resume is not comparable with before it.
- **Warning sign at iteration 500: the robots stay up but do not walk.** Paper-eval test on Stair Up (200 robots, 15 s): 0 of 200
  reached the target 3.7 m away, 198 were still standing at the time limit, 2 fell. Robot 0 was commanded 0.74 m/s and moved
  0.07 m in its first second. In the training log at 499: 89% of episodes run the full 20 s, the `dont_wait` penalty sum is -9.6
  (G1 run at 500: -4.3), `feet_air_time` sum 0.008 (G1: 0.040), tracking reward per step 0.012 (G1: 0.020), terrain level 1.00
  (G1: 1.48). So the X2 is surviving more and stepping less than the G1 did at the same iteration: the standing-still pattern the
  SSR track reported. Too early to call (500 of 30,000; the G1 also took thousands of iterations to climb), but this is the thing
  to watch: if `feet_air_time` and tracking per step are not rising by about iteration 1,500, look at the X2-specific suspects
  first (style data with long standing stretches, action scale on the ankles, the 40 deg camera seeing less ground, 8,192 robots).
- Not measured yet: double-support share, any video; actual speed only for one robot in one second. "Walks" is not established;
  only that robots stay up for most of an episode and the tracking term matches the G1 run's at this stage.
- Numbers at a few thousand iterations: not reached yet (about 8.5 s per iteration; iteration 2,000 around 02:00 UTC 2026-11-12).

## GPU use (user: "make sure the GPUs are well used")

Benchmark at checkpoint 250, same task, 12 iterations each, one job at a time (`runs/_bench/bench.log`, `code/x2_port/bench_envs.sh`):

| Robots | Samples per second | Iteration time | GPU memory | Note |
|---|---|---|---|---|
| 4,096 | 15,400 | 6.4 s | 21.0 GB | paper's count |
| 8,192 | 22,100 (+43%) | 9.0 s | 34.9 GB | chosen; no PhysX errors |
| 12,288 | 24,500 (+59%) | 12.1 s | 49.1 GB | PhysX "collisionStackSize buffer overflow, contacts have been dropped" 6 times: needs larger PhysX buffers, untested |
| 16,384 | not measured | | | stopped during startup to get training going again |

- Running now: 22,800-23,400 samples/s, 8.4-8.6 s per iteration, 35 GB of 96 GB, GPU utilisation 50-95% (`gpu.csv`).
- Kept fixed: 24 steps per robot per iteration, 4 minibatches, 5 epochs, adaptive learning rate (KL 0.01), 30,000 iterations,
  all rewards. Effect on the comparison with the paper's 4,096: each iteration uses twice the samples (minibatches twice as
  large), so iteration counts are not one-to-one with the G1 run; 30,000 iterations take about 71 h instead of 53 h.
- What limits it (py-spy on the live trainer, 25 s, `runs/_bench/pyspy_8192_iter370.raw`): the main Python thread spends about 36%
  in sensor updates (30% in the depth-camera ray-cast, mostly inside `warp synchronize`, i.e. waiting for the GPU), 14% in the
  policy step, 10% rewards, 8% the PhysX step call, 7% actuators, 6% motion reference. The single thread mostly launches GPU work
  and waits for it, so the limit is GPU compute done one kernel after another (ray-casting and physics), then per-step Python
  overhead; not GPU memory, and not something more CPU cores would fix.
- Still open: 12,288 with `sim.physx.gpu_collision_stack_size` and `gpu_max_rigid_patch_count` enlarged (+11% samples over 8,192
  at +34% iteration time); only worth testing inside a 10-minute pause.

## Backup (user: everything a restart needs on the shared filesystem)

Nothing a restart needs is only on the node's local disk: the run writes straight to the shared filesystem.

| Kind | Where (under `/mnt/filesystem-g4/`) | Restore |
|---|---|---|
| Checkpoints (all kept, 59 MB each, every 250 iterations: more than "newest two plus every 1,000") | `tracks/hiking/runs/x2_parkour_v1/<start time>/model_<it>.pt` | `start_x2_parkour_v1.sh` resumes from the newest |
| Training log, supervisor log | `tracks/hiking/runs/x2_parkour_v1/_supervisor/` | read only |
| Metrics, GPU log | `tracks/hiking/runs/x2_parkour_v1/metrics.jsonl`, `gpu.csv` | rebuilt from `train.log` by `log_metrics.py` (delete `metrics.jsonl*` first) |
| Run config as trained | `tracks/hiking/runs/x2_parkour_v1/<start time>/params/{env,agent}.yaml` | |
| X2 port code, supervisor, eval and export scripts, URDF variant | `tracks/hiking/code/x2_port/` | `apply.sh` copies the task files into InstinctLab |
| Upstream code at pinned commits + patches | `tracks/hiking/code/hiking-in-the-wild/` (from `tracks/hiking/hiking-in-the-wild.bundle`) | `env/setup_node.sh` |
| X2 motion reference and its build script | `data/motions/hiking_x2/` | `build_x2_motion.py <mjcf> <flat pkl> <stairs pkl> <out npz>` on the desktop (sources listed below) |
| X2 model | `data/robots/x2_ultra/` | |
| Extended terrain set and its export scripts | `data/terrains/extended_v1/`, `tracks/hiking/code/x2_port/terrain_export/` | `export_extended_terrains.py` (20 s on CPU) |
| Benchmarks, profile | `tracks/hiking/runs/_bench/` | |
| Eval hand-over, paper eval results | `tracks/hiking/eval/` | `export_for_eval.py`, `pause_and_eval.sh` |
| Environment: exact versions | `tracks/hiking/env/freeze.txt` (rewritten every 30 min), `constraints.txt` | `bash env/setup_node.sh` rebuilds `~/hiking-env` (about 15 min) |

- Automatic part: the supervisor rewrites `metrics.jsonl` every 5 min, `gpu.csv` every minute, and every 30 min `env/freeze.txt`
  and `tracks/hiking/backup_manifest.txt` (checkpoint list, file counts).
- Only on the node's local disk, all rebuildable: the venv `~/hiking-env` (29 GB), `~/Datasets` (the released G1 motion,
  unzipped from `data/motions/hiking_release_g1/data_model.zip`), Isaac's converted USD in `/tmp/IsaacLab`.
- Environment pins that differ from the desktop: numpy 1.26.0; `setuptools<81` then `flatdict==4.0.1 --no-build-isolation`;
  instinct_rl installed with `--no-deps --ignore-requires-python`; `mujoco==3.3.7 websockets==12.0`; torch 2.7.0+cu128;
  `isaacsim[all,extscache]==5.1.0` from pypi.nvidia.com; Python 3.11 from uv.

## Commands

```bash
S="ssh -i ~/.ssh/nebius_corpus -o IdentitiesOnly=yes -o BatchMode=yes <user>@<hiking-node>"
$S 'cat /mnt/filesystem-g4/status/hiking.json'                                             # state
$S 'tail -5 /mnt/filesystem-g4/tracks/hiking/runs/x2_parkour_v1/_supervisor/supervisor.log'
$S 'touch /mnt/filesystem-g4/tracks/hiking/runs/x2_parkour_v1/_supervisor/STOP'            # stop (supervisor kills training, exits)
$S 'bash /mnt/filesystem-g4/tracks/hiking/code/x2_port/start_x2_parkour_v1.sh'             # start or resume (newest checkpoint)
$S 'bash /mnt/filesystem-g4/tracks/hiking/env/setup_node.sh'                               # rebuild code tree + env (idempotent)
$S 'tmux new -d -s evalwait "bash /mnt/filesystem-g4/tracks/hiking/code/x2_port/pause_and_eval.sh x2_parkour_v1 <iteration> 540 stair_up,gap 1000"'  # paper eval in a pause
$S 'CUDA_VISIBLE_DEVICES= ~/hiking-env/bin/python /mnt/filesystem-g4/tracks/hiking/code/x2_port/export_for_eval.py <model_N.pt>'  # eval hand-over, CPU
$S 'bash /mnt/filesystem-g4/tracks/hiking/code/x2_port/apply.sh'                           # copy the X2 port into InstinctLab again
# quick env check (not a result); needs the GPU free, so stop training first:
$S 'cd /mnt/filesystem-g4/tracks/hiking/code/hiking-in-the-wild/InstinctLab && OMNI_KIT_ACCEPT_EULA=YES /home/<user>/hiking-env/bin/python ../../x2_port/check_x2_env.py --headless --num_envs 64 --steps 150'
```

## Layout on the node

| Path | What |
|---|---|
| `/mnt/filesystem-g4/tracks/hiking/code/hiking-in-the-wild/` | clone of the bundle (branch `g1-20k-and-extended-terrain`) with `InstinctLab` de6b92f + patch, `instinct_rl` 574c8d5 + patch, `IsaacLab` 37ddf62 (v2.3.2), `instinct_onboard` b6965a4 cloned beside it |
| `/mnt/filesystem-g4/tracks/hiking/code/x2_port/` | the X2 port: `files_instinctlab/` (new files only, copied into InstinctLab by `apply.sh`), `robot/x2_ultra_pelvis_root.urdf`, `supervise_x2.py`, `check_x2_env.py`, `start_x2_parkour_v1.sh` |
| `/mnt/filesystem-g4/tracks/hiking/env/` | `setup_node.sh`, `constraints.txt`, `freeze.txt`, setup logs |
| `/home/<user>/hiking-env` | the Python 3.11 venv (local disk: the shared filesystem takes about 17 ms per small file). Isaac Sim 5.1.0, Isaac Lab 0.54.2 (2.3.2), torch 2.7.0+cu128 |
| `/home/<user>/Datasets` | the released G1 motion file (the G1 task reads `~/Datasets`) |

Do not run Python with `code/x2_port` as the working directory if a folder named `instinctlab` is in it (it shadowed the package
once; the folder is now `files_instinctlab`).

## Copies made to the shared data folder (rsync --bwlimit=3000, 2026-11-11)

| To | From (desktop) | Size |
|---|---|---|
| `data/robots/x2_ultra/urdf/` (x2_ultra.urdf, two sphere-feet variants, meshes) | `gwbc-v6/gear_sonic/data/assets/robot_description/urdf/x2_ultra/` | 112 MB |
| `data/robots/x2_ultra/mjcf/x2_ultra.xml` | `gwbc-v6/gear_sonic/data/assets/robot_description/mjcf/x2_ultra.xml` (identical to `ssr-x2/assets/robot/mjcf/x2_ultra.xml`) | 60 KB |
| `data/motions/hiking_x2/` (`x2_parkour_motion_without_run_retargetted.npz`, `.yaml`, `build_x2_motion.py`) | built on the desktop from `loco-manip-apparatus/build/train/x2_loco_flat_train.pkl` (BONES-SEED X2 retargets) and `terrain-park/data/train_sets/tp_stairs_x10_sh2.pkl` (Pico stair takes) | 3 MB |
| `data/motions/hiking_release_g1/data_model.zip` | `hiking-in-the-wild/data_zip/data_model.zip` (the paper's release) | 12 MB |

Motion credit: BONES-SEED, Motion Data by Bones Studio (bones.studio). No clip names in anything shown.

## The X2 port (task `Instinct-Parkour-Target-Amp-X2-v0`, play variant `...-X2-Play-v0`)

Inherited unchanged from the paper's config (`ParkourEnvCfg`): terrains, commands, rewards and weights, observations, curriculum,
randomisation, terminations (except one threshold), network, PPO and discriminator settings.

Changed, robot only:

| Item | X2 value | Reason |
|---|---|---|
| Model | shared `x2_ultra.urdf` (mesh feet) with the massless `base_link` and the XML declaration removed (`x2_port/robot/`), root = pelvis | Isaac named the root `base_link`; pytorch_kinematics cannot parse the declaration. The G1 task uses a torso-rooted URDF; I kept the X2's own root so the shared model stays as is |
| Actuators | delayed PD (0-2 steps) as on the G1; armature, effort, velocity from the vendor plant (gwbc-v6 `config/robot_plant/x2_ultra.yaml`); kp = armature x (2 pi 10)^2, kd = 4 x armature x 2 pi 10 | same BeyondMimic rule as the G1 task |
| Actions | 29 joints, scale 0.25 x effort / kp; the two head joints are not commanded (held at zero by PD) | head motors are 2.6 and 0.6 N.m and carry the camera |
| Default pose | gwbc-v6 X2 pose (same leg angles as the G1 pose, elbows -0.6), spawn height 0.71 m | pelvis stands at 0.641 m (MuJoCo FK) |
| Self-collision | on, minus 5 link pairs two joints apart whose hulls always overlap (knee/ankle_roll, wrist_yaw/wrist_roll per side, torso/head_pitch) | measured 6-20 kN between them at spawn; it tripped the torso-contact termination every step |
| Depth camera | the X2's head camera: on `head_pitch_link` at (0.0576, -0.0112, -0.0484), 40 deg below horizontal; optics, crop, noise, history as in the paper | deployable mount; the G1's is 48 deg down on the torso |
| Foot numbers | sole 0.073 m below the ankle (feet_at_plane offset); volume points x -0.045..0.12, y +-0.05, z -0.078..-0.038 | X2 sole from the model's contact points |
| Torso tilt term | `pelvis_orientation_l2` points at `torso_link` (from iteration 250) | the G1's root is the torso and the term adds the pelvis; the X2's root is the pelvis, so the term adds the torso |
| Root-height termination | 0.39 m (G1: 0.5 m for its torso root) | same fraction of standing root height |
| Motion reference | one concatenated 405.7 s file at 50 Hz: 322 s flat walking and turning (26 BONES-SEED clips), 83 s stairs (16 Pico takes); head and wrist joints zeroed; clips chained in xy and yaw | the release is one 379.6 s file of walk and climb; a single long file keeps `dataset_exhausted` from ending episodes early |
| Mirror maps | derived from joint names in Isaac order (roll and yaw negate) | checked with MuJoCo FK: 2.5 mm max error; a wrong sign gives 0.28 m |

Apparatus parts copied into training: none (paper terrains only).

## Verified

- G1 released task starts and trains headless on this GPU (256 robots, 3 iterations, 4.0 s per iteration): `runs/_smoke/g1_smoke.log`.
- X2 env: joint order equals `X2_ISAAC_JOINTS` (assert), 29 actions, no contact above 1 N at spawn after the filter, depth image
  shows ground with no self-occlusion blobs. Zero actions for 3 s: 1.86 episodes ended per robot (G1 under the same check: 2.45),
  so standing without a policy is no worse than the G1.

## Not verified

- Nothing is scored: no common eval, no MuJoCo run of an X2 checkpoint yet.
- Feet in MuJoCo are 12 small spheres per foot (the shared MJCF); training uses the mesh foot (convex hull). The sphere-feet URDF
  was not used because 5 mm spheres let terrain edges pass between them.
- Depth camera: pose from the URDF, intrinsics left at the paper's (D435i-like); not compared with the real Gemini 335 stream.
- Head held by PD at zero: on the robot the head pitch must be held the same way.
- Whether the stair clips (0.3 m total rise, two steps) are enough of a climbing prior for the high platforms.

## Decisions (date, decision, reason)

- 2026-11-11: venv on the node's local disk, rebuild script on the shared filesystem. Reason: small-file speed.
- 2026-11-11: Linux pins differ from the desktop (see Backup). Reason: Isaac Sim's own pins.
- 2026-11-11: pelvis root, 29 commanded joints, head camera, mesh feet, filtered pairs: see the port table.
- 2026-11-11: wrote `data/robots`, `data/motions` and `data/terrains/extended_v1` myself, as the coordinator's messages said.
- 2026-11-11: 8,192 robots, not 12,288. Reason: PhysX dropped contacts at 12,288 with the task's buffer sizes; +11% samples
  did not justify an untested buffer change while training was stopped.
- 2026-11-11: `pelvis_orientation_l2` now penalises `torso_link` on the X2 (from iteration 250). Reason: on the G1 the root is the
  torso and that term adds the pelvis; with the X2's pelvis root the unchanged term penalised the pelvis twice and the torso not at
  all (seen in the log: both terms printed identical values). Iterations 0-250 were trained with the duplicate.
- 2026-11-11: all checkpoints are kept (7 GB for the whole run) rather than pruned to "newest two plus every 1,000".
- 2026-11-11: paper eval is run in pauses of at most 10 minutes, a few terrains per pause (each terrain is its own Isaac start).

## Logs: what is recorded per iteration (`runs/x2_parkour_v1/metrics.jsonl`, one JSON object per iteration)

Written by `code/x2_port/log_metrics.py` from the trainer's printed block; names are the trainer's. An iteration that was redone
after a resume appears twice (take the last). dt = 0.02 s.

| Name | What it is | Paper | Weight |
|---|---|---|---|
| `iteration`, `steps_per_s`, `collection_s`, `learning_s`, `Iteration_time`, `Total_timesteps`, `Total_time`, `ETA` | throughput | | |
| `reward` | mean episode return (all terms, weighted, summed over the episode) | | |
| `length` | mean episode length in control steps (max 1,000 = 20 s) | | |
| `action_noise_std` | policy action noise | Table 6 (init 1.0) | |
| `surrogate_loss`, `value_loss`, `entropy`, `total_loss` | PPO losses | Table 5 | |
| `discriminator_loss`, `discriminator_gradient_penalty`, `discriminator_weight_decay`, `logit_weight_decay`, `amp_total_loss` | AMP discriminator losses | A.9 | gradient penalty 5.0 |
| `discriminator_reward` | mean style reward per step, before the 0.25 coefficient | A.9 | 0.25 |
| `Episode_Reward/rewards_<term>/sum` | weighted episode sum of the term | Table 4 | below |
| `Episode_Reward/rewards_<term>/timestep` | weighted mean per step (weight x dt x raw) | | |
| `Episode_Reward/rewards_<term>/max_episode_len_s` | weighted sum divided by 20 s | | |
| `raw/<term>` | unweighted mean per step = timestep / (weight x 0.02) (4-decimal print, so tiny terms are coarse) | | |
| `Curriculum/terrain_levels` | mean terrain row (0-9) over all robots | A.3 | |
| `Metrics/base_velocity/error_vel_xy`, `error_vel_yaw` | command tracking error summed over the command period | | |
| `Metrics/base_velocity/tracking_exp_vel_xy`, `tracking_exp_vel_yaw` | exponential tracking score used by the curriculum | A.3 | |
| `Episode_Termination/<cause>` | share of ended episodes per cause: `time_out`, `terrain_out_bound`, `base_contact` (torso), `bad_orientation` (tilt > 1 rad), `root_height` (pelvis < 0.39 m), `dataset_exhausted` | A.2 | |

Reward terms and weights (the paper's, unchanged): `track_lin_vel_xy_exp` +2.0, `track_ang_vel_z_exp` +2.0, `heading_error` -1.0,
`dont_wait` -0.5, `is_alive` +3.0, `stand_still` -0.3, `volume_points_penetration` -4.0 (foot points inside terrain edges),
`feet_air_time` +0.5, `feet_slide` -0.4, `joint_deviation_hip` -0.5, `ang_vel_xy_l2` -0.05, `dof_torques_l2` -1.5e-7,
`dof_acc_l2` -1.25e-7, `dof_vel_l2` -1e-4, `action_rate_l2` -0.005, `flat_orientation_l2` -3.0 (root tilt),
`pelvis_orientation_l2` -3.0 (on the X2: torso tilt), `feet_flat_ori` -0.4, `feet_at_plane` -0.1, `feet_close_xy` +0.4,
`energy` -5e-5, `freeze_upper_body` -0.004, `dof_pos_limits` -1.0, `dof_vel_limits` -1.0, `torque_limits` -0.01,
`undesired_contacts` -1.0 (any body except the feet over 1 N).

`gpu.csv`: unix time, iteration, GPU memory used and total (GB), GPU utilisation (%), trainer CPU (% of one core), every minute.

Not logged (gaps): terrain level per terrain type (only the mean over all types); commanded speed distribution and the robot's
actual speed; gait statistics (air time per foot, double-support share, step length); the paper's Table 1 and 2 metrics during
training (success rate, reaching time, landing area: these come from the paper eval below); per-expert MoE gate weights.
Plan: a small monitor term for per-type terrain level, actual vs commanded speed and double-support share, added at a restart.

## Paper's own evaluation (Tables 1 and 2) for the X2

- Script: `code/x2_port/eval_paper_x2.py`, generated by `make_eval_paper_x2.py` from the G1 reproduction's `scripts/eval_paper.py`
  (protocol and its 21 assumptions: `docs/EVAL_NOTES.md` in the code tree). Paper figures ("Ours", G1) are read from the paper
  text and printed beside ours: Table 2 success 99.09-100% on all nine terrains, reaching time 5.15-8.03 s; Table 1 landing-area
  fraction 0.77 (stair up), 0.67 (stair down), 0.84 (gap), 0.81 (small box).
- X2 differences: the X2 task and model; landing-area rays cover the X2 sole (0.18 x 0.10 m centred 0.037 m ahead of the ankle,
  sole 0.073 m below it; G1 shoe: 0.18 x 0.06 m); the interaction-range grid hangs under the pelvis; reaching distance is from the
  pelvis; terminations are the X2 task's. Terrain sizes are the paper's Table 8 (20 cm stairs, 40 cm platforms and boxes, gaps),
  unchanged, on a robot whose hip is at 0.64 m instead of the G1's roughly 0.76 m.
- How to run: `bash code/x2_port/pause_and_eval.sh x2_parkour_v1 <checkpoint iteration> <budget s> <terrains> [robots]` in tmux. It
  waits for the checkpoint, stops training, runs one Isaac worker per terrain, restarts training, logs the pause length.
  Each terrain costs about 3 to 5 minutes of start-up, so the nine terrains need four or five 10-minute pauses, or one 40-minute
  pause: to be agreed with the coordinator before a full table.
- Status: plumbing proven at checkpoint 500 on one terrain with 200 robots and a shortened Table 1 run (400 steps): 288 s for
  the terrain including Isaac start-up, tables printed with the paper's row beside ours, `results_*.json` written
  (`tracks/hiking/eval/paper_eval/x2_parkour_v1_500/`). Result at 500: Stair Up 0 of 200 (see Early signals). No full table
  (nine terrains, 1,000 robots, 10,000-step Table 1) has been run. This is a track report, not the team's apparatus score.

## Eval hand-over (`/mnt/filesystem-g4/tracks/hiking/eval/x2_parkour_v1_500/`, and `..._250/` with synthetic observations)

- `policy.onnx` (flat observation 5,408 -> 29 actions), also `0-depth_encoder.onnx` + `actor.onnx` (the paper's deploy layout),
  `adapter_card.json` (50 Hz, joint order, default pose, action scale, PD gains, effort and velocity limits, each observation term
  with offset and size, the command and how to make it from waypoints, depth sensor pose, optics and preprocessing),
  `obs_check.npz` (50 observations with torch and ONNX actions, max difference 7e-7).
- Made on the CPU by `code/x2_port/export_for_eval.py <checkpoint>` while training runs.
- `x2_parkour_v1_500/obs_check.npz` holds 50 observations recorded in the training simulator (robot 0, Stair Up, first second);
  `obs_check_sim_state.npz` holds the matching simulator state per step (root pose and velocities, gravity vector, joint positions
  and velocities, command, default pose, joint names) so an adapter can rebuild each observation and compare. ONNX on those
  observations reproduces the simulator policy's actions to 1e-6.
- Checked on the recording: history entries are oldest first (newest last) for joint positions (error 0.009 = the noise level;
  0.036 the other way round).
- Limits: `model_500` stands and does not walk (plumbing only). The order of the 8 depth frames is not verified. The card's
  depth preprocessing was read from the config, not rebuilt outside Isaac.

## Extended terrain set (`/mnt/filesystem-g4/data/terrains/extended_v1/`, 9.4 MB, CPU export, 2026-11-11)

- 330 tiles of 8 x 8 m, ten difficulty levels per type: single beams 30, beam fields with V pairs and slanted 4x4s 30, blocks 30,
  wedges (pinwheel cells, long ramps, narrow strips) 40, tight spots and dead ends 20, log corridors with wall, rail and post
  sides 40, homes 30, street front 30, stairs 80 (straight, L, U, amphitheatre).
- Format: per type one `.npz` (2 cm height rasters 400 x 400, obstacle mask, boxes with centre, half size, quaternion and role,
  meshes for non-box shapes) and one `.json` (level, seed, parameters, spawn poses, routes); `index.json`, `README.md`,
  `load_example.py`, `preview_top.png`, `preview_closeups.png` (previews, not simulator renders).
- Routes (after the street-front fix of 2026-11-11 23:20): 710 defined, 60 dropped by `check_routes.py`, **650 kept**; 51 kept
  routes had end points pulled back inside the tile. Street front: 120 of 120 kept, four per tile (street to porch over the curb;
  street to garage up the driveway; porch to street over the rock bed and curb; garage to street), least clearance 0.40 m, from a
  generator variant `terrain_export/street_routed.py` (obstacles placed clear of the routes, car moved, garage floor slab added,
  door headers raised with the floor); its tiles differ from the original `street.py` for the same seed.
- Still weak: wedges 21 routes dropped (pinwheel routes into rails or brace stands, or off steep gables), stairs 19 (all
  amphitheatre), homes 19 (kitchen clutter on fixed route lines); causes are the sub-agent's reading, not individually diagnosed.
  The 0.35 m clearance rule is enforced for street only.
- `targets` in the JSON are now only the checked ones; the 44 outside the tile are under `raw_targets_unchecked`.
- Not in it: the paper's own nine terrain types (their generator needs Isaac Sim; `review/terrain/terrain.ply` could be
  rasterised later); roll wedges; rails on amphitheatre aisles.
- Held-out check: the generators read no apparatus file; no tile reproduces an apparatus layout or route. Apparatus-like single
  parts present: 4x4 beams 5-15 cm (flat or one end raised up to 0.20 m), V pairs of 2.6 m beams, 2 x 2 pinwheel ramp cells
  (1.27-1.94 m, 8.8-25 deg), rails on 8.9 cm posts, brace stands, stair flights (risers 8.4-21.6 cm, 3-15 steps), doorways.
  No pinwheel matches our apparatus cell (about 1.2 m, 15 deg) in both size and all slopes.
- Not verified: nothing loaded in a simulator yet; spawns not tested against the X2 footprint.
- Export scripts: `tracks/hiking/code/x2_port/terrain_export/`.

## Next

1. Run 2 signals at 250, 500, 1,000 and 2,000 iterations (see "Watch in run 2").
2. Make `export_for_eval.py` and the MuJoCo port take the camera mount and gains from the task (they still describe run 1's head camera and armature-rule gains): needed before any run-2 checkpoint is exported or rendered.
3. Paper table only when a checkpoint walks, in pauses under 10 minutes.
4. Extended task as a second task name (`Instinct-Parkour-Extended-Amp-X2-v0`): **first version written, never run in Isaac**
   (`code/x2_port/files_instinctlab/tasks/parkour/config/x2/extended.py`). It has: the paper's 20 terrain columns plus 12 new ones
   (one per exported tile type, two for stairs, two velocity-command columns), tile meshes built from the set's boxes and meshes
   (builder tested on the CPU for all nine types: 96 to 9,260 faces a tile, inside 8 x 8 m); velocity-mode commands forward
   -0.4..0.8, sideways +-0.3 m/s, turn +-1 rad/s; `feet_stumble` (weight -2.0) and `foot_clearance` (-0.5) beside the paper's edge
   term. Missing: the set's checked routes as spawn/target pairs (targets are sampled flat patches; spawn yaw not applied),
   backward and sideways clips in the motion reference, graded knee and shin costs. Before any training:
   `check_x2_env.py --task Instinct-Parkour-Extended-Amp-X2-v0` in a pause. What was planned originally: Needed: an Isaac terrain
   importer for `extended_v1` tiles beside the paper's nine types, route spawns and targets, backward and sideways commands with
   matching motion clips, the stumble, foothold and clearance terms (the user: "INCLUDE THEM"). Start only after the paper-terrain run has a checkpoint that walks the paper's terrains.

## Blockers

None. Open: the user's measured camera height and angle; which pelvis mass is right (URDF 3.52 kg or MJCF 5.03 kg).

## HOW TO RESUME (updated 2026-11-13 13:40 node clock; supersedes everything below where they differ)

- v7 = THE LAST PLANNED RESTART (user approved paper reference v4 on 11-13; user rule 11-13: no pauses, stops or planned restarts
  for evaluation; checkpoints go to the desktop, evals run there). Run goes to --total 30000 (fixed end iteration 30,000).
- 13:28 v6 disarmed (tmux cont6 killed; GO_v6 never written; v6 used the flawed v3). 13:31:57 GO_v7 ("now") written,
  tmux `cont7` = `code/x2_port/v7/continue_v7.sh`, restart from model_11125 (v5b was at 11,235).
- v7 content = v6 minus v3: motion `data/motion_v7/parkour_motion_without_run_x2_v4_retargetted.npz` (byte copy of
  hiking_release_x2/..._v4.npz; already in the loader's format: framerate, joint_names (31 X2 joints, same order as v2),
  joint_pos, base_pos_w, base_quat_w; yaml lists it); t cap `AIR_TD_CAP=ref` x 140 (variants/staged.py.v6); `--extra=--seed=42`
  (train.py --seed -> agent and env seed 42; was 42 by default before); all other env vars = v5b (precheck compares).
- Discriminator CARRIES OVER (decision 13:30): same 550 inputs, so the v5 style loader keeps it with its optimiser and
  normalisers. v4 vs v2: leg joints mean |d| 0.0014 rad, waist pitch 0.138 rad (p95 0.63), waist at -18 deg limit 0 frames
  (v2 2,901), at +18 deg 2,926. The leg features stay valid and only the waist must be relearned (tens of iterations);
  a fresh one would give a random style reward for the whole adaptation (v5b's fresh start cost episode 833 -> 610 in the trial).
- Pre-flight 13:30 by hand: `precheck_v7_manual.log` PASS (launcher = v5b except cap and motion; refcap tests; motion v4 checks;
  code tree 32 = installed), `cfgscan_v7_manual.log` 42 terms, touchdown = refcap x 140. The gate repeats both before stopping v5b.
- Trial: 75 iterations, marks over the last 30 (level >= 0.8 x, episode >= 0.4 x, turn >= 0.5 x), else fallback_v5b.sh. Then guard7.
- Nothing pauses training: auto_export_v4.sh (tmux autoexp) exports on CPU (`CUDA_VISIBLE_DEVICES=` , nice) without STOP;
  crashguard stops only after two error exits from one checkpoint; the supervisor restarts only on a stall/crash. No eval on the node.
- Not taken into v7: 16,384 robots (approved for "the next planned restart"): untested PhysX buffers (contact drops at 12,288);
  a trial failure would have cost the v4 switch, and no later planned restart exists now. Stays at 8,192. Open for the user.
- Coordinator questions 13:45 (training plots), answered from code and metrics.jsonl 11,000-11,125:
  1. Levels at about 6 = the curriculum is SATURATED, not stuck. Promotion (paper `tracking_exp_vel`): up if the episode mean of
     exp(-|v_cmd - v|^2 / 0.5^2) > 0.6, down if < 0.3. Isaac's `update_env_origins` sends a robot that passes level 9 to a
     uniform random level 0-9; a robot that passes every level then has mean level sum k(k+1)/sum(k+1) = 330/55 = 6.0. Per type
     now: 5.88-6.85, gaps 5.21 (still climbing). So the 10 rows are exhausted for every type but gaps; time-outs 91-99% per group
     agree. More difficulty needs harder rows (terrain change), not more iterations.
  2. Half speed (walk_monitor speed_ratio 0.496: 0.247 / 0.498 m/s over all robots with a forward command > 0.1 m/s). Route
     robots away from props track 0.74-0.90 of command (stairs 0.56, tight spots 0.66, homes 0.74); so the low mean comes from the
     rest: paper-column target robots get a forward command also while turning toward a target (only routes zero it over 60 deg),
     high-level obstacle tiles, stairs. Nothing pays for closing the gap: the reward and the promotion kernel (std 0.5) keep 76%
     at half of a 0.5 m/s command (above the 0.6 promotion bar), overspeed (-3) charges only above the command. Not separated
     per column for paper tiles (no log); MuJoCo flat eval at 4,125 tracked 0.5 within 2%, so it is not a flat-ground inability.
     (`slow_fwd_ratio` 0.059 is a per-env metric diluted over all robots; not read as a ratio.)
  3. v7 changes neither rule. The t cap `ref` raises the paid swing at 0.5 m/s from 0.30 to 0.48 s (reference cadence 1.9
     steps/s instead of 2.9): speed then rests on step length from the style reference; watch speed_ratio after the trial. Fixes
     (tighter tracking std / promotion on speed ratio, harder rows) need a restart: under the 11-13 rule (no planned restarts)
     they are open for the coordinator/user, not taken.
- TRIAL VERDICT 13:57:05: PASS after 76 iterations (last 30): level 6.153 (before 6.153), episode 877 (before 981, 0.89 x),
  turn rate 1.206 (before 1.201). GPU idle 1,460 s for the restart. Run dir `20261113_133809_from20261113_023937`, trainer
  command has `--seed=42`, params/agent.yaml seed 42; no "discriminator shapes changed" line = discriminator carried over
  (discriminator reward 0.615 vs 0.627 before). guard7 (crash-loop fall-back v5b) running; cont7 finished.
- NEXT: nothing planned on the node; run ends at 30,000 (about 9.6 s/it -> about 50 h, ~16 Nov). Read speed_ratio and the
  desktop scorer's results on the v7 exports (11,250 onward); resume this agent only for a crash-guard event or a finding.

## (older) HOW TO RESUME 2026-11-13 12:42 (v6 below never started; superseded by v7)

- RUNNING: v5b (x2_parkour_v4, run dir 20261113_023937_from20261113_020626), iteration ~10,940 at 12:41; guard5, autoexp.
- ARMED, NO GO: tmux `cont6` = `code/x2_port/v6/continue_v6.sh`, waits for `runs/x2_parkour_v4/_supervisor/GO_v6` (coordinator
  writes it after the user approves; "now" inside = newest checkpoint at once). v6 = v5b + (a) option t with a speed-dependent cap
  (`HIKING_X2_AIR_TD_CAP=ref`, weight 140 kept; `variants/staged.py.v6` = live staged.py + `v6/mk_v6.py`): cap = 1/f_ref(|v|) - 0.05,
  f_ref 1.4/1.9/2.0 steps/s at 0.3/0.5/0.7 m/s (flat outside); CPU estimate: paid optimum moves from 2.86 steps/s at every speed to
  1.40/1.90/2.00 (no double support); at 0.7 m/s 2.9 steps/s now earns 5% less than 2.0 (before: about 40% more); swings < 0.35 s
  pay as before, so feet still lift. (b) paper reference v3 (`data/motion_v6`, copy of hiking_release_x2/..._v3.npz). Slow commands,
  VX_MIN 0.45, backward rule: unchanged (user push-back 2026-11-13: both earlier fixes answered real stalls; do not remove).
- Pre-flight passed by hand 12:40 (`v6/precheck_manual.log`, `v6/cfgscan_manual.log`): launcher = v5b except the two changes;
  refcap tests; v3 motion reader, 0 waist frames at the limit (v2 2,901), changes only frames 0-2974; code tree 32 files =
  installed; cfg scan 42 terms, touchdown term = refcap x 140. Then construction check, 75-iteration trial, fall-back `fallback_v5b.sh`.
- ONE CODE SOURCE: `files_instinctlab` is the source; `apply.sh` installs; precheck_v6 and continue_v6 (`tree_ok`, after apply) fail
  on any difference or extra file. Variants live in `x2_port/variants/` (old .v4j/.v4k/.next files still sit in both trees, identical).
- WHAT WENT WRONG: (1) the reference itself leans back on stairs/blocks: G1 waist -21.3 deg median, past the X2 -18 deg limit (not a
  sign bug); v3 moves the excess to the pelvis, torso error vs G1 2.6 -> 0.1 deg median. (2) exports had no run_params, so the
  autoscorer used the 4,750 copy: `export_for_eval.py` now copies params/ as run_params/ (backup in _trash/2026-11-13), 83
  exports backfilled. (3) X2 shoulder roll sits at -0.061 rad (limit) in every reference frame, v2 and v3: open, not fixed.
- NEXT: when GO_v6 appears the gate runs by itself; check launch.txt for "v6 trial" about 40 min after GO.

## (older) HOW TO RESUME (updated 2026-11-13 02:17 node clock; supersedes everything below where they differ)

- WHAT WENT WRONG (2nd): v5b's construction check failed at 02:06 for the same reason as v5. `simfault.py` `_unless_sim_fault`
  wrapped the termination terms (bad_orientation, root_height, stalled) as `wrapped(env, *args, **kwargs)`, and Isaac Lab's
  term check read args/kwargs as mandatory parameters. My 01:50 signature check covered only the 4 style terms. v4j resumed.
- FIX: `@functools.wraps(func)` (Isaac's inspect.signature follows __wrapped__) in both copies of simfault.py (bad copy in
  `_trash/2026-11-13/simfault.py.v5bad`). New generic gate: `code/x2_port/v5b/cfgscan.py` builds the FINAL env cfg on CPU
  (simulator modules stubbed, launcher env vars, v5b file swaps on an overlay copy via `v5b/cfgscan_v5b.sh`) and applies Isaac's
  rule to every term of observations, rewards, terminations, events and curriculum (42 terms). Proof: on the 02:06 code it
  FAILS on bad_orientation, root_height and stalled (`v5b/cfgscan_0206code.log`); on the fixed code it passes
  (`v5b/cfgscan_fixed.log`). `continue_v5b.sh` runs it after precheck_v5b.py and does not stop v4j if it fails (edited by hand
  after mk_v5b.py generated the file, so do not regenerate it).
- RE-ARMED 02:16:44: tmux `cont5b`, restart from model_7500 (v4j was at 7401 at 02:16, so about 02:33). Content unchanged:
  paper v2 motion alone, plus everything from v5. Trial: 75 iterations; fall-back to v4j.
- NEXT CHECK about 03:00: launch.txt (cfg scan, check exit, trial line).

## (older) HOW TO RESUME (updated 2026-11-13 01:57 node clock; supersedes everything below where they differ)

- WHAT WENT WRONG: v5 never trained. 01:32 construction check died in Isaac Lab's term check
  (`manager_base._resolve_common_term_cfg`: "amp_policy/joint_pos_rel expects mandatory parameters ['asset_cfg','kw']"),
  because `style_weights.py` wrapped the observation functions with `**kw`, which Isaac Lab's signature check counts as a
  mandatory parameter. The CPU pre-flight tested the functions on stubs but not Isaac's signature rule. v4j resumed from
  model_7250 at 01:32 (run dir `20261113_001803_from...`? newest ckpt dir; v4j at it 7344 at 01:55, 9.8 s/it).
- FIX (01:50): `code/x2_port/v5/fix_style_signature.py` gave the 4 wrappers explicit keyword parameters with defaults (same as
  the wrapped functions; bad copy in `_trash/2026-11-13/style_weights.py.v5bad`), both copies (InstinctLab tree and
  files_instinctlab). `v5b/precheck_v5b.py` now runs Isaac's signature rule on the 4 swapped terms on CPU.
- v5b = v5 content (v4k changes, option j style weights, fresh discriminator 330 -> 550 inputs, simcheck) with the motion
  reference = the paper's reference v2 ALONE (379.6 s, 18,982 frames; floating-feet fix): `data/motion_v5b/
  parkour_motion_without_run_x2_v2_retargetted.npz` + yaml. Coordinator correction 01:57: the user's "include my Pico ones"
  was for SSR; Hiking keeps the paper's data only. A Pico-mixed build (01:52, 614 s) was disarmed before it ran; files and
  build script in `_trash/2026-11-13/`. The paper file has 49 planar jumps > 5 m/s (clip seams inside the release, v1 and v2
  alike). Scripts generated by `v5b/mk_v5b.py` from the v5 ones.
- Discriminator: re-initialised (forced: every checkpoint has the 330-input leg-only discriminator; v5's 550-input one never
  existed). Policy, value and optimizer load from the checkpoint.
- Pre-flight `v5b/precheck_v5b.py` PASS at 01:56 on the v2-only file (motion reader steps, 380 s, signature check, discriminator drop).
- ARMED 01:56:50: tmux `cont5b` = `v5b/continue_v5b.sh`, GO_v5b written (user approved via coordinator), restart from model_7375
  (about 02:00). Trial 75 iterations, marks over the last 30; fall-back = v4j (`fallback_v4j.sh`), since v5 never ran.
- NEXT CHECK about 02:30: `_supervisor/launch.txt` (v5b check exit, trial line); then style reward / discriminator logits,
  base_contact share on ext_tight_spots, slow_* and turn metrics vs v4j.

## (older) HOW TO RESUME (updated 2026-11-13 00:42 node clock; supersedes everything below where they differ)

- 00:05 v4k started (693 routes); trial FAILED 00:17 (ep 306 vs 944, turn 0.34 vs 0.91); v4j resumed from model_7000 at 00:18.
- Trial was NOT fair (CPU from train.log, `code/x2_port/v4j/cmp_trial.py`, `restarts.py`): v4j restarted from the SAME
  model_7000 with v4j config scored ep 195, turn 0.17, vx -0.13 in its first 32 iterations, below v4k (184, 0.21, -0.18), and
  recovered by 7044 (ep 520-780, turn 0.67, vx 0.20). Cause: the episode mark is censored, since no episode can time out
  until about 42 iterations after a full reset (v4k end/time_out 0.00 vs 0.78), so it counted falls only. The turn mark already
  excludes the capped robots (caps <= 0.4, metric needs |cmd| > 0.4). Endings in both: root_height about 0.7, bad_orientation
  0.27. One v4k-specific signal: base_contact on ext_tight_spots 12% of ends vs 0.5%; watch it in v5. Open: why the
  7000 restart transient is deeper than earlier restarts (first-32 turn 0.63-0.88 at 3250-6750).
- v5 = ONE combined restart, built and armed (tmux `cont5`, `code/x2_port/v5/continue_v5.sh`): v4k content + paper motion
  retargeted to X2 (`data/motion_paper_x2/`, copied as *_retargetted.npz: the loader reads only that suffix) + option j
  (style input: legs/waist 1.0, shoulders/elbows 0.3, wrists/head out; 550 inputs; checkpoint discriminator dropped on shape
  mismatch, policy kept; `style_weights.py`) + simcheck (`simfault.py`, `v5/supervise_x2_v5.py`). Trial 75 it, marks over the last 30.
  Gate: `touch _supervisor/GO_v5` (content "now" = newest checkpoint at once, else the next 125 checkpoint). Pre-flight
  `v5/precheck_v5.py` PASS at 00:41 on the delivered retarget (18,982 frames, 380 s, 50 fps); FAILs on the 23:11 bad file.

- 23:11 v4k NOT started. What went wrong: the GPU construction check died in `AmassMotion._read_retargetted_motion_file`
  (`ValueError: 'left_hip_pitch_joint' is not in list`, check.log line 153) because `data/motion_v4/append_motion_v4.py` reused
  the variable `names` (joint names, line 13) for the clip keys (line 49) and saved the 72 clip keys as `joint_names`. The
  arrays were right. The CPU pre-flight only counted frames. Fix (23:30, `code/x2_port/v4j/fix_v4k_jointnames.py`): script
  uses `JNAMES`; npz rewritten with the v2 joint names, arrays identical (bad copies in `_trash/2026-11-12/`);
  `precheck_v4k.py` now runs the reader's joint lookup, shapes, finite values and quaternion norms for every yaml file on CPU:
  PASS on the fixed file, FAIL (exit 1) on the 23:11 file. `continue_v4k.sh` writes the check's last error line to launch.txt.
  Not reproducible on CPU: everything from `gym.make` in `check_x2_env.py` line 28 (scene, sim.reset, managers, route table).
- v4j resumed 23:11 from model_6750 (run dir `20261112_231149_from20261112_205728`), guard4j. Re-armed 23:31: tmux `cont4k` =
  `continue_v4k.sh 7000` (about 00:05), content unchanged; same pre-flight, check, trial 31 it and fall-back.
- Next check about 00:30: launch.txt; then Metrics/base_velocity/slow_* and turn_drift, turn/* and episode length vs v4j.
- PENDING (user decision 2026-11-13, via coordinator, docs/USER_RULES.md): keep the paper's single discriminator; replace the
  BONES-SEED + Pico motion mix with the paper's own motion reference retargeted to X2:
  `/mnt/filesystem-g4/data/motions/hiking_release_x2/parkour_motion_without_run_x2.npz` (paper keys, X2 joint names, 50 fps;
  retargeted by another agent). v4k stays unchanged. Goes into the next planned restart after v4k's 2,000-iteration freeze, and
  only once the coordinator confirms the retarget passed its checks. Before that restart: run `precheck_v4k.py`'s motion-reader
  check on it (joint lookup, shapes, quaternions), new yaml, and a CPU pre-flight twin.
- After v4k: NO restart for >= 2,000 iterations (coordinator 22:20) except crash or failed trial.

## (older) HOW TO RESUME (2026-11-12 20:18 node clock)

- Node `hiking` = the host whose `tmux ls` shows `hiking_x2` (login NODE_USER, key ~/.ssh/nebius_corpus; address not written here).
- Running: run 4, v4i (options a,c,d,f,r,s,t,k,y,o,z,w) resumed from model_5500 at 19:22 after the failed v4j check;
  iteration 5,798 at 20:18 (about 5.4 it/min); tmux `hiking_x2`, `autoexp` (exports), `guard4i` (crashguard_v4.sh 5.80 878).
- Armed: tmux `cont4j` = `code/x2_port/continue_v4j.sh 6000` (stdout `_supervisor/continue_v4j.out`). At model_6000 (about
  20:55): CPU pre-flight `v4j/precheck_v4j.py` (log `_supervisor/precheck_v4j.log`; fail = v4i not stopped), then stop v4i,
  construction check 256 robots (routes >= 100, `wall_contact_graded` in the reward table), start v4j, trial 31 iterations
  (level >= 0.8 x, episode >= 0.4 x, turn rate >= 0.7 x), else `fallback_v4i.sh` + guard4i. Results in `_supervisor/launch.txt`.
- Next check (about 21:15): launch.txt pre-flight, check and trial lines; then in metrics.jsonl the v4j keys: artefact/spawn_ext_homes (< 3%
  wanted, was 23.5%), c10/*_ext_tight_spots (which body carries the 4 kN spikes), tunnel/*, end/*_ext_homes and
  _ext_tight_spots, end/reached_*, wallspeed/near vs far; the turn rate and level against v4i.
- Run dir `/mnt/filesystem-g4/tracks/hiking/runs/x2_parkour_v4/`; code `code/x2_port/` (v4j build in `v4j/`), live config
  `code/hiking-in-the-wild/InstinctLab/source/instinctlab/instinctlab/tasks/parkour/config/x2/`.
- Eval: model_4125 0/50 on the apparatus; model_4750 queued (flat gate and station legs first, on the plane-and-box floor).
- Open: lean -16.7 deg (camera explanation; user / hardware decision); camera height and pelvis mass from the user.
- Rules: no commits or pushes; one GPU job; never touch the desktop G1 reproduction.

## 2026-11-14 02:35 node clock: v7 reward drop investigated (coordinator request; CPU only, nothing changed on the node)
- See docs/FINDINGS.md H7. Three trainer crashes (NaN std after critic divergence: value_loss 0.3 -> 1e10 -> 1e21 -> 1e33 -> NaN
  in 4 iterations) at 14,246 / 14,921 / 15,050, resumed by the supervisor from 14,125 / 14,875 / 15,000; run dirs
  20261113_224549_..., 20261114_010924_..., 20261114_014623_... (current, pid 666793, iteration about 15,210).
- The "missing termination shares" are the ~40-iteration warm-ups after each resume (all robots restart at t = 0; no time-outs
  yet), plus 220 duplicated iterations in metrics.jsonl / tensorboard. Clean series: 101 / 970 (13k) -> 92-98 / 910-945 (post-
  restart mixes) -> 96.6 / 954 (15,200). Level 6.1, speed ratio 0.51 unchanged. Not real collapse; mild drift at most.
- Scorer 14,625 partial: legs 42/34/1/0 (11,750: 46/45/6/5). No newer hiking export scored.
- HAZARD: guard7 falls back to v5b on two exit-1s from one checkpoint (FB_START=code/x2_port/fallback_v5b.sh). A NaN loop would
  revert v7. Proposed to the coordinator (not done): disarm that fallback for NaN exits; add a finite guard to ppo.update
  (skip non-finite / >1e3 value-loss minibatches, clamp returns) to load at the next crash resume; log max |reward| / |critic obs|
  when value_loss > 1e3. Scripts used: /tmp/term_hk.py, /tmp/clean_hk.py on the node (read metrics.jsonl only).
- 02:39 node clock (coordinator decision, crash-level): finite guard installed in instinct_rl ppo.py (pre-flight PASS, see FINDINGS
  H7), crashguard_v7.sh running in tmux guard7 (old v5b-fallback guard killed). Trainer pid 666793 untouched; the guard loads at
  the next crash resume. NEXT CHECK: `grep -a "^\[GUARD\]" _supervisor/train.log | tail` and `_supervisor/ALERTS.txt`; the first
  [GUARD] line names the critic-obs column / return size that triggers the divergence -> fix the source term.

## HOW TO RESUME (2026-11-14 02:40 node clock; supersedes the 13:40 block above where they differ)
- v7 training to 30,000 (tmux hiking_x2, supervisor v5; current run dir `20261114_014623_from20261114_010924`, about 15,220).
- guard7 = `code/x2_port/crashguard_v7.sh` (no fallback; alerts; stop after 5 crash exits in 2 h). autoexp exports on CPU.
- instinct_rl ppo.py carries the finite guard (FINDINGS H7); not yet active in the running process.
- Read v7 metrics only with resume warm-ups (~45 iterations after each resume) and re-run iterations removed (/tmp/clean_hk.py
  pattern; see FINDINGS H7). Open: the divergence trigger; scorer results for 14,625 and later.

## 2026-11-14 08:45 node clock: gap to the paper on its own terrains - diagnosis and armed v8 (user priority 11-14)
Scope: the walled-ramp tiles / toe-strike penalty brief was dropped on the user's 11-14 rule ("That gap has to go away first");
nothing of it was built. X2 capability taken as established (user 11-14, SSR paper deploys the X2 on 0.20 m stairs, gaps, platforms).

Check before acting (numbers):
- Eval on the paper's generator rows (eval/autoscore/train_results.csv; courses rebuilt line by line from InstinctLab
  hf_terrains / parkour_env_cfg, difficulty (k+0.5)/10, so rows 8-9 ARE the paper's Table 8 hard settings: stairs 0.20 m at
  row 8, platform 0.39 m, gap 0.61 m, boxes 0.39 m; slope row 8 = 0.595 rise/run, steeper than the paper's 0.40 rad):
  15,500: stairs_up L5 0/20 (14,625 16/20), L8 0/20; platform_up L5 0/20, L8 0/20; gap L5/L8 0/20; slope_up L8 0/20;
  small_boxes L8 5/20, boxes L8 3/20. Rows 2 pass; downhill types (stairs_down, platform_down L8 20/20) pass.
- Every failure of stairs_up L5, platform_up L5, gap L5, slope_up L8 at 15,500 is reason 'stalled' at waypoint 0 (45 s, no
  fall, pelvis 0.67-0.68 m, final x 0.91-0.95 m with the riser at 1.18 m / gap edge at 1.25 m, command 0.7 m/s); platform_up
  logs repeated edge steps at the same x (marking time). A refusal, not a fall.
- Training says the opposite: every type's level sits at the recycling equilibrium (17,285: pyramid_stairs_inv 6.63,
  inv_high 6.77, square_gaps 5.26, slope_inv 6.91; equilibrium 6.0). So the curriculum does not see refusals.
- Cause 1 (curriculum signal): paper rule = episode mean exp(-|v_cmd - v|^2/0.25) > 0.6 up, < 0.3 down, commands U(0.45, 0.8)
  toward a far target, 5% stand. Ours adds stand 0.10, turn in place 0.15 (4-6 s then stand), slow forward cap U(0.1, 0.45) for
  25%, on the paper columns too (slow.py: only the random-velocity columns are exempt). Monte Carlo (200k episodes, two 10 s
  command segments, robot stops at the obstacle foot after 2 s): paper commands 7% up / 59% down; ours 42% up / 21% down.
  A refuser drifts UP under our mix, so rows saturate while the policy refuses rows 5-9. (Half-speed climber under the paper
  rule: 89% up, so the paper rule also tolerates half speed; not changed.)
- Cause 2 (pay for marking time): option t (ours; paper feet_air_time 0.5, episode sum about 1) pays 26.1 per episode at 17,288
  (linear tracking 30.3) for any touchdown under a moving command, including stepping in place at a refusal.
- Not causes: generator rows = eval rows (checked above); the X2 task inherits the paper's terrain cfg, curriculum term and
  thresholds unchanged (x2_parkour_target_amp_cfg.py header); the paper's G1 was scored in Isaac, ours in MuJoCo (a sim2sim
  share of the gap cannot be excluded on CPU; the refusal pattern with no falls points to the policy, not physics).
Decision 08:43 (mine, logged): arm v8 = v7 + two changes, nothing else:
  (1) HIKING_X2_FAIR=1: an episode with any stand / turn / slow-forward command is neither promoted nor demoted; the paper's
      rule unchanged for the rest (about 20-35% of episodes stay eligible: 0.9 x 0.85 x 0.75 per resample). [fix, paper-aligned]
  (2) HIKING_X2_AIR_TD_PROGRESS=1: t pay x clamp(v . c / |c|^2, 0, 1) under walking commands; turn-in-place unchanged. [ours]
  Not taken: harder rows (rows 8-9 already hold Table 8), removing slow / turn / stand commands (needed for the apparatus
  stations), the 0.6 bar or kernel std (paper values), robots 16,384.
Files (node, one code folder): code/x2_port/v8/{fair.py, extended.py.v8 (3-line hook), continue_v8.sh, precheck_v8.sh,
  cfgscan.py, cfgscan_v8.sh, test_fair.py, crashguard_v8.sh}. Installed into files_instinctlab only at GO (v7 untouched now).
CPU pre-flight 08:40-08:44 (`runs/x2_parkour_v4/_supervisor/preflight_v8/`): precheck_v8 PASS (v7 checks; only the two v8 vars
  differ from v5b), cfgscan_v8 PASS (42 terms; touchdown = feet_air_touchdown_refcap_progress x 140; curriculum terrain_levels =
  tracking_exp_vel_paper_cmd; command FairSlowRouteFarTargetPoseVelocityCommand), test_fair PASS, start-command smoke
  (start_x2_parkour_v8.smoke.sh = v7's + the two vars, bash -n OK), export dry-run of model_17250 PASS (onnx vs torch 1.67e-6).
  The GPU construction check (needs [X2 fair] line, routes >= 100) runs at GO while training is stopped.
Expected effect: levels on stairs_inv / inv_high / gaps / platforms / boxes fall from ~6 toward the real frontier within a few
  hundred iterations (only eligible episodes move), then climb only where the robot actually climbs; mean level is no longer a
  saturation number. t episode sum falls from 26 toward 13-20 (speed ratio 0.5); expect speed_ratio to rise. Read the eval rows
  5/8 of the first exports 500-1,000 iterations after the start.
What went wrong since the last report: the 11-14 brief (walled ramps) was on the wrong problem by the user's priority; dropped
  after one CPU look at the apparatus ramp (corridor = post-and-rail fences 2.44 m apart, rails at 0.565 / 0.87 m, ramp cells
  1.22 m, crest 0.15 m = 14 deg; recorded here for later). No crash since 01:46 (17,288 at 08:30, [GUARD] 0 lines).

## HOW TO RESUME (2026-11-14 08:45 node clock; supersedes the 02:40 block above where they differ)
- v7 training (tmux hiking_x2, about 17,300 at 08:30), guard7 = crashguard_v7.sh, autoexp exports on CPU.
- ARMED: tmux cont8 = `code/x2_port/v8/continue_v8.sh`, waits for `/mnt/filesystem-g4/tracks/hiking/runs/x2_parkour_v4/_supervisor/GO_v8`
  (content "now" = newest checkpoint, anything else = the next one). The coordinator writes it only on the user's yes.
  At GO: pre-flight again -> kill guard7 -> STOP v7 -> install fair.py + extended.py.v8 -> apply.sh -> tree check -> GPU
  construction check -> start_x2_parkour_v8.sh -> 76-iteration trial (level >= 0.7 x, episode >= 0.4 x, turn >= 0.5 x) ->
  guard8 (crashguard_v8.sh); any failure -> fallback_v7.sh (v7 files, start_x2_parkour_v7.sh, guard7). All lines in launch.txt.
- Next check after GO: launch.txt trial line; then per-type levels (Curriculum/walk_monitor/level/*), Metrics paper_cmd_episode,
  t episode sum, speed_ratio; eval rows 5/8 of the exports after +500 iterations. Resume this agent at GO+2 h or on a guard alert.

## 2026-11-14 09:05 node clock: v8 did not start (launcher bug), fixed and re-run; v8b armed as its fallback
What went wrong: v8's GO at 08:47 stopped v7 at 17,375, CPU pre-flight passed, but the GPU construction check failed
(FileNotFoundError `/data/extended_v4k/single_beams.npz`, `_check_v8_084921/check.log`): continue_v8.sh defined ENVV on line 8
before T and S on line 9, so every `$T`/`$S` path in ENVV was empty (extended terrains, motion dir, level files). cfgscan_v8.sh
substitutes $T/$S itself and precheck compares text, so the CPU pre-flight could not see it. fallback_v7 resumed v7 at 08:50.
Not a trial failure and not an unfair criterion: v8 never trained. Fixes: T/S defined before ENVV; a path check (ENVV's
terrain and motion dirs must exist) before anything stops; old script in `_trash/2026-11-14/continue_v8.sh.envbug_*`.
Trial criterion reviewed for a curriculum meant to drop levels: under FAIR only all-walking episodes (33-57%) move, at most
one level per episode, about 2 episodes per robot in 76 iterations, so the mean level can drop about 0.5 at most from ~6.15;
the 0.7 x gate (4.3) is not binding on the intended effect. Kept.
Decision 08:58 (mine, logged; user rule 11-14 "don't give up quickly"): re-run v8 on the existing GO_v8 (user's yes stands);
v8b armed behind it.
v8b = v8 + fewer non-walking commands: stand 0.10 -> 0.05 (the paper's rel_standing_envs), turn-in-place 0.15 -> 0.10,
slow-forward share 0.25 -> 0.12 (cap U(0.1, 0.45) unchanged), slow-yaw cap share 0.4 kept (walkers still walk; not
FAIR-excluded). Evidence: all-walking share per resample 0.9x0.85x0.75 = 0.57 -> 0.95x0.90x0.88 = 0.75, per two-resample
episode 0.33 -> 0.57 (1.7 x the curriculum signal and 1.3 x the full-speed commands on stairs/platform/gap); stop-stand 50/50 at
every scored checkpoint since 11,750 (paper share is enough); flat turn-in-place 28/47/12/50 of 50 (11,750/14,625/15,500/16,125)
and apparatus legs 3-4 0-7/50, so turn-in-place keeps 2/3 of its share, not less; slow forward halved (final approaches).
Chain now: cont8 (v8) -> on any failure after GO: fallback_v7 resumes v7, EXIT trap writes GO_v8b "now" -> cont8b (v8b, own
pre-flight, stops v7, trial against v7's pre-v8 numbers in `baseline_v7_for_v8.txt`) -> on failure fallback_v7 (final).
Files: `code/x2_port/v8/{continue_v8.sh (fixed), continue_v8b.sh, precheck_v8b.sh, cfgscan_v8b.sh, crashguard_v8b.sh,
setup_v8b.py}`; start_x2_parkour_v8b.sh is generated at v8b's start (smoke copy in `_supervisor/preflight_v8b/`).
Pre-flight 08:55-08:58 (`_supervisor/preflight_v8b/`): precheck_v8 (fixed) PASS, precheck_v8b PASS (negative test with slow
fwd 0.20 fails as it should), cfgscan_v8b PASS (42 terms, refcap_progress x 140, tracking_exp_vel_paper_cmd, Fair command),
test_fair PASS, expanded ENVV paths of both launchers checked absolute, start-line smoke bash -n OK with the three shares.

## HOW TO RESUME (2026-11-14 09:05 node clock; supersedes the 08:45 block above where they differ)
- tmux cont8 = `code/x2_port/v8/continue_v8.sh` (fixed), re-run 08:58 on the existing GO_v8 from model_17375: CPU pre-flight
  passed 08:59, then stop v7 -> install -> construction check -> start_x2_parkour_v8.sh -> 76-iteration trial -> guard8.
  09:04:46: construction check PASS (exit 0, 768 routes, [X2 fair] line), v8 training started; trial verdict expected about
  09:35 in launch.txt ("v8 trial after ..." then "is training" or "FAILED" -> v8b).
- tmux cont8b = `code/x2_port/v8/continue_v8b.sh`, waits for `_supervisor/GO_v8b`, which only continue_v8.sh's EXIT trap
  writes when v8 does not reach "is training". v8b failure -> fallback_v7.sh (final). Never write GO_v8b by hand while v8 trains.
- Read `_supervisor/launch.txt` (lines "v8 ..." / "v8b ..."), gates.txt for guard lines.
- If v8 trains: judge it on the paper rows (stairs_up / platform_up / gap L5 and L8, eval/autoscore/train_results.csv) of the
  exports 500-1,000 iterations after 08:58 (about 17,900-18,400), plus Curriculum/walk_monitor/level/* (expected to fall from
  ~6 on stairs_inv / gaps / platforms), t episode sum (26 -> 13-20), speed_ratio. If those rows have not moved off 0/20 by
  ~1,000 iterations, v8b is the next experiment: stop v8 by hand (STOP + guard8 first), then write GO_v8b "now" (cont8b still
  armed; its baseline file is older than 3 h by then, so it uses v8's own last numbers).
- Rate about 340 iterations/h: +500 near 10:40, +1,000 near 12:10. Resume this agent at about 12:15 (v8 +1,000, eval rows of
  the ~17,875 and ~18,375 exports), or earlier on a guard alert / a "v8b" line in launch.txt.

## 2026-11-14 12:45 node clock: v8 judgement at about +1,130 iterations (coordinator request 12:30)
v8 trial passed 09:25 (77 iterations: level 6.13 vs 6.11, episode 870, turn rate 0.98, paper-command episode share 0.41);
training since, no crash, at 18,507 by 12:40.
Training side, v7 (last 20 before 17,374) -> v8 (fifths of 17,376-18,507; last 20):
  square_gaps 5.13 -> 4.51 2.71 1.27 0.67 0.60 (0.61): FAIR made the curriculum see the gap refusal; gaps now train from
    rows 0-1 upward (the intended effect, first on gaps).
  pyramid_stairs_inv (stairs up) 6.66 -> 6.81; pyramid_stairs_inv_high 6.79 -> 6.15 (falling); hf_pyramid_slope_inv 6.61 -> 6.72;
    pyramid_stairs / _high (down) 6.0 -> 6.1; boxes 6.22 -> 6.32; mesh_boxes 6.0. Mean terrain level 6.14 -> 5.88.
  speed_ratio 0.51 -> 0.60; touchdown episode sum 24.5 -> 19.8 (progress weighting, as predicted 13-20).
CPU rows of export 18,375 (paper courses, 20 trials, same commands and env as the scorer's train job, low priority on the
desktop; out `eval/out/hiking/x2_parkour_v4_18375/manual_v8_rows/`): see the next entry.
CPU rows, export 18,375 (v8, +1,000) vs v7 15,500 / 17,000 (train_results.csv), 20 trials each:
  stairs_up   L2 20/20 (20, 20)   L5 20/20 (0, 15)   L8 0/20 (final), all stalled at x 1.04 (riser 1.18), pelvis 0.68 (0, 0)
  platform_up L2 20/20 (20, 20)   L5 5/20 (0, 17), 15 stalls   L8 0/20 stalled (0, 0)
  gap         L2 18/20 (20, 0)    L5 0/20 (0, 0) but 14 falls + 6 stalls (v7: all stalls; now it attempts)   L8 0/20, 19 stalls
  slope_up    L2 20/20, L5 20/20 (20/20 both)   L8 0/20 (final), all stalled at x 1.2 (0, 0)
Decision 12:55 (mine, logged; user rule "don't give up quickly"; coordinator's rule: switch only if rows 5 and 8 are still 0/20
AND training shows no climbing): KEEP v8. Rows 5 are not 0/20 (stairs 20/20, platform 5/20), gap went from refusing to
attempting, and the training levels move (square_gaps re-learned from ~0.6). v8b stays armed (cont8b, GO_v8b absent).
Open (owner me, next check ~15:45 at about 19,400): row-8 refusals on stairs / platform / slope are unchanged (stall at the foot,
no fall) while the training levels of pyramid_stairs_inv (6.8) and slope_inv (6.7) sit at saturation; read the per-row level
distribution of those types from terrain_levels.pt and the generator's row-8 geometry vs the eval's (riser 0.20 m) before
choosing between "wait" and v8b. platform_up L5 fell 17 -> 5/20 since 17,000 (stalls): watch.

## HOW TO RESUME (2026-11-14 12:55 node clock; supersedes the 09:05 block where they differ)
- v8 training since 09:04 (tmux hiking_x2, guard8), about 18,500 at 12:40, about 340 iterations/h. cont8b (v8b) armed on GO_v8b;
  create GO_v8b ("now") only after stopping v8 by hand (touch STOP after killing guard8) - or simply write GO_v8b: continue_v8b
  stops hiking_x2 itself, but kill guard8 first so it does not restart v8.
- Manual CPU eval (12:40): per course `eval/ladder.py --adapter hiking:HikingHandover --step 20 --trials 20 --tests hik_<type>_L<row>`
  with autoscore's run_env('hiking', hdir) (+ PHYS_FLOOR=heightfield for slope), hdir = scp -r of the node export; out
  `eval/out/hiking/<export>/manual_v8_rows/`. The desktop scorer also scores v8 exports (score_every 500).
- Next check ~15:45 (export ~19,375): the 12 paper rows again + per-row levels of pyramid_stairs_inv / slope_inv / square_gaps.
  Switch to v8b only if stairs/platform/gap rows 5 AND 8 are 0/20 and no training level climbs.

## 2026-11-14 16:10 node clock: v8 second check (coordinator request); decision KEEP v8
CPU paper rows, 20 trials, low priority (`eval/out/hiking/x2_parkour_v4_{19375,19250}/manual_v8_rows/`), vs 18,375 / 18,000 (scorer):
  stairs_up   L2 20 | L5 19,375: 0, 19,250: 0 (18,375: 20, 18,000: 20) all stalled | L8 0 stalled
  platform_up L2 20 | L5 19,375: 0, 19,250: 18 (5, 20)                              | L8 0 stalled
  gap         L2 20 | L5 19,375: 0 (17 stall, 3 fall), 19,250: 1 (13 falls = attempts) (0, 0) | L8 0 stalled
  slope_up    L2 20, L5 20 | L8 0 stalled (1430 s course time: they stand at the foot)
Training levels (terrain_levels.pt, 17,384 pre-v8 -> 19,520; rows 0-9 counts):
  pyramid_stairs_inv (stairs up) 6.61 -> 6.58, now peaked rows 6-7 (235/280), row 8 124, row 9 16 (was 148/129): FAIR removed
    the recycling at the top; the frontier is row 7-8 (0.18-0.21 m) in Isaac.
  pyramid_stairs_inv_high (platform) 6.91 -> 5.99, peak row 6 (310 of 512; 0.31 m), rows 8-9 15. Still falling slowly.
  hf_pyramid_slope_inv 6.58 -> 6.89, peak row 7, rows 8-9 125. Slowly climbing (tb fifths 6.53 6.75 6.80 6.87 6.90).
  square_gaps 5.11 -> 0.63, rows 0/1/2 = 249/207/50, FLAT at 0.58-0.63 since 18,355 (1,100 iterations).
  speed_ratio 0.60 (flat), paper_cmd_episode share 0.41.
Geometry: generator params (env.yaml of this run) = eval builder (eval/train_terrains.py): stairs step 0.05-0.23 x 0.3 m,
  platform 2.5; high 0.05-0.45 x 1.5 m, platform 4.0; gaps 0.1-0.7, depth 0.4-0.6 (eval 0.5); slope 0-0.7. Training row k
  draws difficulty U(k/10,(k+1)/10), eval uses (k+0.5)/10. Differences: training adds wall_prob 0.3 per side and a perlin
  overlay on stairs/slope; eval has neither. Not a geometry mismatch at row 8.
Decision 16:10 (mine; rule "keep if any hard-row progress or climbing levels"): KEEP v8. platform L5 18/20 at 19,250 (> 5 at
  18,375), gap L5 first success and attempts instead of refusals, slope level still climbing. v8b stays armed (GO_v8b absent).
What went wrong / open (owner me):
  (1) Checkpoint-to-checkpoint oscillation of the refusal: stairs L5 20/20 at 18,000/18,375, 0/20 at 19,250/19,375; platform L5
      20, 5, 18, 0. One export is not a verdict; score at least two neighbours before any switch.
  (2) Train/eval disagreement: Isaac robots sit at stairs rows 6-7 while MuJoCo eval refuses row 5 (stall at the riser foot,
      no fall). Either a sim2sim share or the eval command (0.7 m/s straight) differs from training. Next: a 10-trial
      stairs_up L5 with command 0.5 m/s and with walls on one side, to separate command from scene.
  (3) Gap level stuck at rows 0-1 in training although eval gap L2 (0.25 m) passes 20/20: the promotion test fails on gaps
      for another reason (tracking under gap terrain resets / depth 0.4-0.6 falls?). Read per-row gap terminations next.
  (4) Row 8 unchanged 0/20 on all four types, all stalls.
Next check ~19:00 (about 20,400): the 12 rows on two neighbouring exports + the level files; v8b if rows 5 and 8 have no
  progress across both AND stairs_inv / inv_high / gaps levels are flat or falling.

## HOW TO RESUME (2026-11-14 16:10 node clock; supersedes the 12:55 block where they differ)
- v8 training (tmux hiking_x2, guard8), about 19,550 at 15:30, ~340 it/h. cont8b armed on GO_v8b (kill guard8 first if used).
- Manual rows: scratchpad driver pattern = autoscore names/run_env + hiking_check + eval/ladder.py per hik_<type>_L<row>
  (env TESTS list), out `eval/out/hiking/<export>/manual_v8_rows/`. Level distribution: /tmp/lvl_dist.py on the node
  (node python env, args: [geom] <terrain_levels*.pt>); tb trends /tmp/tb_lv.py (both in /tmp, rewrite if gone).
- Next check ~19:00: open items (1)-(4) of the 16:10 entry.

## 2026-11-14 17:25 node clock: attempt-vs-refusal replay, v9 built and armed (user: "Please fix"; decisions "Hiking v9 decisions")
Replay (CPU, MuJoCo, exports 20,000 and 19,250, 4-8 trials per course (row-8 courses cut by the 50-min wall limit); every v8 reward term recomputed per control step as
weight x 0.02 x raw from the MuJoCo state, the policy's command and action; scripts `replay_terms.py` / `summ_terms.py` and
per-trial traces in the coordinator desktop's session scratchpad, not in the repo). Windows: "stand" = within 0.7 m before the
edge, |vx| < 0.15; "foot-stepping" = within 0.35 m of the edge on a course where every trial stalled (row 8: marking time at the
riser / platform face); "climb" = the same window on row-5 successes. Per second (sum without the unnormalised energy term;
AMP style reward not replayed):

| term /s | stairs L8 stand | stairs L8 foot-stepping | platform L8 stand | platform L8 foot-stepping | stairs L5 climb | platform L5 climb (20k / 19,250) |
|---|---|---|---|---|---|---|
| touchdown (ours, 140) | 0.00 | **+4.36** | 0.00 | **+3.74** | +0.92 | +1.38 / +1.76 |
| track_lin_vel | +0.29 | +0.58 | +0.30 | +0.56 | +0.95 | +1.26 / +1.27 |
| dont_wait | -0.77 | -0.80 | -0.74 | -0.81 | -0.28 | -0.09 / -0.11 |
| feet_stumble (user rule, -2) | 0.00 | 0.00 | 0.00 | 0.00 | -0.03 | -0.01 / -0.02 |
| wall_contact_graded (user rule) | 0.00 | 0.00 | -0.01 | -0.21 | 0.00 | -0.05 / -0.10 |
| is_alive + track_ang + rest | +3.4 | +3.4 | +3.5 | +3.3 | +3.3 | +3.4 |
| sum | 3.64 | **8.17** | 3.79 | **7.33** | 5.22 | 6.11 / 6.63 |

Reading: (1) marking time at the foot of a row-8 obstacle pays MORE per second than climbing a row-5 one (7.3-8.2 vs 5.2-6.6);
the whole difference is the touchdown term t (ours; 3.7-4.4 /s at the foot; v8's progress weighting does not stop it because the
oscillating vx is positive at touchdown). That is the eval's refusal pattern (stall at the foot, steps at the same x). Without
t: stand 3.6-3.8, foot-stepping 3.5-3.8, climb 4.3-4.9 /s, so climbing pays about +0.6-1.0 /s, as in the paper's reward.
(2) feet_stumble and the graded contact cost 0.01-0.03 and 0.00-0.10 /s during real climbs (row 5), 1-2% of the climb reward and
well below the climb advantage; at row 8 the graded contact charges the foot-stepping refusal (knees on the platform face,
0.21 /s), not the climb. In these data they do NOT make attempting cost more than refusing. Caveat: hard row-8 strikes are never
observed (the policy never attempts); a 300 N shin hit costs 0.08 per step (4 x 0.02), a 0.2 s 200 N toe strike about 0.8,
small against losing the climb by a fall (value about 7-8 with gamma 0.99). Proposal: keep both at 1.0 (no evidence for a cut);
if the user wants a test, 0.5 on the paper columns only (flag below).
(3) Gap row 8 (0.61 m): foot-stepping at the edge pays touchdown 2.4 /s; gap L5 attempts (5 falls of 8) pay graded contact 0.17 /s.
Gap block (square_gaps at rows 0-1 since FAIR): the paper's promotion score is sum of exp(-err^2/0.25) / max episode length, so an
early termination scores only its survived fraction and a robot stepping at the gap edge scores about exp(-0.5^2/0.25) = 0.37
(< 0.6). Under v7 the non-walking commands hid this; under FAIR only walking episodes count and they fail the bar. Falls vs
marking time at the edge are both plausible; t pays the second. v9 removes t and logs per type the score, up, down and
early-termination shares of eligible episodes (Curriculum/by_type/*) to separate the two within about an hour.
Decision 17:25 (mine, on the user's decisions 1-2): v9 = v8 (FAIR kept) + release command mix + no touchdown term, nothing else.
  Release values (InstinctLab parkour_env_cfg.py CommandsCfg): rel_standing_envs 0.05; obstacle columns lin_vel_x (0.45, 0.8),
  lin_vel_y 0, ang_vel_z (-1, 1) via the heading rule; perlin_rough (0.45, 1.0); random-velocity column perlin_rough_stand with
  lin_vel_x (0, 0), lin_vel_y (0, 0), ang_vel_z (-1, 1) (the only turn in place); no slow-forward or slow-yaw commands.
  v9 env: STAND_SHARE 0.05, TURN_SHARE 0, SLOW_FWD_SHARE 0, SLOW_TURN_SHARE 0, OPTS without t, AIR_TD_PROGRESS 0, AIR_TD_WEIGHT 0,
  V9 1, PAPER_STAND 1 (stand column back to vx 0 / vy 0 / wz +-1; the extended velocity column keeps vx 0..0.8, vy +-0.3).
  Kept: feet_air_time 0.5 (paper), stumble -2 and graded contact (user rules), FAIR, routes and sidesteps on the user's map.
  Optional flag (default 1.0 = off; user decision pending): HIKING_X2_STUMBLE_SCALE, HIKING_X2_CONTACT_SCALE, HIKING_X2_SCALE_SCOPE
  (paper = robots on the paper's 20 columns only; all). Wiring tested: cfgscan with 0.5 / 0.25 gives feet_stumble_x0.5_paper and
  wall_contact_graded_x0.25_paper and passes Isaac's term check (`preflight_v9/cfgscan_flagtest.log`).
  Not taken: 16,384 robots (approved; the level keeper restores levels only when the robot types match, so a robot-count change
  resets the curriculum; needs its own restart and PhysX check; owner me, next planned restart); stall penalty (user: TODO only).
Files (node, one code folder): code/x2_port/v9/{v9.py, extended.py.v9 (v8's + 3-line hook), continue_v9.sh, precheck_v9.{sh,py},
  cfgscan_v9.sh, test_v9.py, crashguard_v9.sh}. Installed into files_instinctlab only at GO (v8 files untouched now).
CPU pre-flight 17:20 (`runs/x2_parkour_v4/_supervisor/preflight_v9/`): precheck_v9 PASS (every launcher var equals v8's except
  the 12 listed; live staged / fair / extended = v8's), test_v9 PASS, test_fair PASS, cfgscan_v9 PASS (43 terms; no
  feet_air_touchdown; stumble -2.0 and wall_contact -1.0 unchanged; curriculum tracking_exp_vel_paper_cmd_v9 + by_type; command
  V9Command), start-line smoke bash -n OK (OPTS without t, shares 0.05 / 0 / 0, --total 30000), export dry-run of model_20000
  PASS (onnx vs torch 1.91e-6). The GPU construction check (needs [X2 fair] and [X2 v9] lines, routes >= 100) runs at GO.
What went wrong since the last report: (a) t, ours, pays the row-8 refusal (marking time at the foot) 3.7-4.4 /s, more than
  climbing; v8's progress weighting did not close it (found by today's replay; v9 removes t). (b) My first v9 files written on
  the desktop had CRLF line ends (cfgscan_v9.sh failed on $'\r'); stripped, pre-flight re-run and passed. (c) A wrong `cp -r`
  copied the repo into my scratchpad (scratchpad only; nothing in the repo or on the node changed).

## HOW TO RESUME (2026-11-14 17:25 node clock; supersedes the 16:10 block where they differ)
- v8 training (tmux hiking_x2, guard8), about 20,100 at 17:15, ~340 it/h.
- ARMED: tmux cont9 = `code/x2_port/v9/continue_v9.sh`, waits for `_supervisor/GO_v9` ("now" = newest checkpoint). At GO:
  pre-flight again -> kill guard8 and cont8b (v8b superseded by the user's paper-mix decision) -> STOP v8 -> install v9.py +
  extended.py.v9 -> apply.sh -> tree check -> GPU construction check -> start_x2_parkour_v9.sh (generated) -> 76-iteration trial
  (level >= 0.7 x, episode >= 0.4 x, paper-command episode share >= 0.6, by_type lines present; no turn gate: v9 has no
  turn-in-place robots off the stand column) -> guard9. Any failure -> fallback_v8.sh (v8 files, start_x2_parkour_v8.sh, guard8).
  Lines "v9 ..." in launch.txt. The run then continues to 30,000.
- GO is written by the desktop script `check1900.sh` (session scratchpad; log check1900.log, marker check1900.DONE) after it has
  scored the 12 paper rows (stairs_up / platform_up / gap / slope_up L2/5/8, 20 trials, 45-min wall limit per course) on exports 20,500 and 20,625 into
  `eval/out/hiking/x2_parkour_v4_{20500,20625}/manual_v8_rows/` (the 19:00 check data). If that script died: score them by
  hand (ladder.py --step 20 --trials 20 per course with autoscore.run_env('hiking', hdir), PHYS_FLOOR=heightfield for slope),
  then write GO_v9 "now" on the node.
- After v9 starts: Curriculum/by_type/{score,up,down,early,elig}_<type> (square_gaps: early high = falls; early low with score
  ~0.3-0.4 = marking time); per-type levels; paper_cmd_episode (~0.9 expected); speed_ratio; then the 12 rows on two
  neighbouring exports +500 and +1,000 iterations after the start. Resume this agent at GO + 2 h or on a "v9 FAILED" line.

## 2026-11-15 00:55 node clock: v9 collapse at 22,113-22,118 - rollback to 22,000, rollout guard, collapse watchdog (FINDINGS H7b)
What went wrong: v9 collapsed with no crash and no alert (reward 77.8 -> -1.4, episode 970 -> 40, level 5.88 -> 0 by 22,135).
  Trigger (metrics.jsonl + train.log [GUARD]): one runaway robot, its unclipped actions fed back through the action-history
  obs (critic columns 713/742/771/800, 29 apart) growing ~1.6x per step to 1.1e8; action_rate_l2 episode sum -9.1e9; max |reward|
  5.4e11 at 22,113 (returns clamped, update APPLIED, value loss 0.53), 22,114 all 20 minibatches skipped, 22,115 APPLIED (value
  loss 85, surrogate 0.141, std 0.92 -> 0.93). Batch-wide advantage normalisation lets that one robot set the std, so normal
  robots' advantages ~0 and the update trains only on the runaway; the 1e3 value-loss threshold misses it. Same runaway at 20,560
  (critic obs 1.7e18; 3 full skips -> raise -> resume). Not FAIR, levels bookkeeping, lr or a terrain reset.
  My error: the first resume (00:35) loaded model_22250 from the backup folder I had put inside the run dir (supervisor globs
  run/*/model_*.pt); stopped at 00:35:38 before any iteration; backups now in runs/_collapse_backups/ (outside the run folder).
Done: STOP 00:33; model_22125 / 22250 + collapsed levels -> runs/_collapse_backups/x2_parkour_v4_v9_22114/; levels file =
  terrain_levels_prestop_v9.pt (mean 5.84 vs log 5.83 at 22,000); v9 resumed from model_22000 at 00:36:30, same flags
  (start_x2_parkour_v9.sh), 22,012 at 00:45, level 5.84. 22,000 is pre-collapse by the log (rollouts 22,001-22,112 used its
  descendants: episode 920-975, level 5.88); 22,125 was saved at episode 40. MuJoCo check (flat + stairs_up L5, 5 trials) of both
  exports: 22000 flat 5/5, stairs_up L5 5/5; 22125 flat 0/5, stairs_up L5 0/5 (all falls) - eval/out/hiking/x2_parkour_v4_{22000,22125}/collapse_check/.
Guard (safety only; CPU pre-flight PASS, code/x2_port/v9/guard2/preflight_guard2.log): ppo.py / wasabi.py skip the whole update
  (PPO + discriminator) if any rollout obs / critic obs > 1e4, action > 1e3, reward > 1e4 or non-finite; log robot, step, columns;
  raise after 3 in a row (-> supervisor resume); log Train/guard_rollout_{obs,critic_obs,action,reward}_absmax_max and
  guard_rollout_skipped_max each iteration. Normal updates bit-identical. Built by v9/guard2/make_guard2.py from guard2/orig.
  crashguard_v9b (code/x2_port/v9/crashguard_v9b.sh): v9 + levels copy per checkpoint (_supervisor/levels_hist/tl_<it>.pt) +
  collapse watchdog (v9/guard2/collapse_check.py: >= 350 rows since restart, last-10 episode < 0.4x and reward < 0.5x the median
  of rows -300..-50) -> STOP, checkpoints > onset-25 to runs/_collapse_backups/, levels of the newest kept checkpoint, restart; a
  second collapse within 3 h -> STOP + ALERT only. Replay over all 23,271 metrics rows: fires only at 22,116 (first at 22,123).
  Installed by one planned restart at model_22125 (v9/guard2/install_guard2.sh, tmux inst_g2, marker guard2/INSTALL.DONE,
  originals in _trash/2026-11-15/).
Open: root cause (unclipped actions) not fixed; propose an env action clip (+-100) once the logged normal maxima show it never
  binds. Check guard_rollout maxima ~1 h after the restart (normal obs should be O(10), well below 1e4).

## HOW TO RESUME (2026-11-15 00:55 node clock; supersedes the 2026-11-14 17:25 block where they differ)
- v9 training (tmux hiking_x2) from model_22000 (00:36:30), ~11 s/it; at model_22125 inst_g2 restarts it once with the rollout
  guard and starts crashguard_v9b in tmux guard9 (lines "v9 guard2 ..." in _supervisor/launch.txt). If "NOT installed" or
  "import failed": v9 keeps running on the old guard and crashguard_v9 is NOT running - start it (tmux guard9) and fix.
- Rollbacks: automatic (crashguard_v9b), backups in runs/_collapse_backups/; never put backups inside runs/x2_parkour_v4/.
- Next check ~02:00: guard_rollout_* maxima in train.log (no [GUARD] rollout skip on normal rollouts), episode / level back to
  ~950 / 5.9; then the 17:25 block's v9 checks (by_type, 12 paper rows at +500 / +1,000 iterations).
- 01:11 update: guard2 installed 01:05:03 (import OK), v9 restarted from model_22125 (healthy: 00:36 run's episode back to 970,
  level 6.05 at 22,125), crashguard_v9b running (tmux guard9), levels_hist/tl_22125.pt written. First guarded iteration 22,125:
  rollout maxima obs / critic obs / action 28.4, reward 0.415 (bounds 1e4 / 1e3 / 1e4), no skip. Level 5.05 in the first row
  is restart warm-up; check that it returns to ~5.9-6.0 by ~01:45.

## HOW TO RESUME (2026-11-15 13:10 node clock; supersedes the 00:55 block where they differ)
- v9 training (tmux hiking_x2) resumed by the supervisor at 11:51 from model_25750 after the guard raised on a full runaway at
  25,858-25,860 (waist_pitch, column 5); iteration 26,163 at 13:06; crashguard_v9b in tmux guard9. Unchanged by this session.
- Action clip (user 2026-11-15 "Cap at 20 like others"; FINDINGS H7c): built + CPU pre-flight PASS in code/x2_port/v9/aclip/
  (new/vecenv_wrapper.py, preflight_aclip.log, install_aclip.sh, probe/). NOT installed: +-20 binds on waist_pitch in 12-56% of
  normal MuJoCo steps (policy holds it at ~-18..-28; physically inert above 5.8); all other joints never exceed 18.1. Per the
  user's instruction this went to the coordinator before restarting. Recommendation: +-20 with waist_pitch +-50.
- To install once decided: `tmux new -d -s inst_ac "bash /mnt/filesystem-g4/tracks/hiking/code/x2_port/v9/aclip/install_aclip.sh 20 5:50"`
  (or `... install_aclip.sh 20` for uniform +-20); it waits for the next checkpoint (+125), STOPs, installs, restarts with the
  clip, re-arms crashguard_v9b, and writes "v9 aclip installed" to _supervisor/launch.txt. Then 100 iterations later read the
  [ACLIP] lines in _supervisor/train.log (clipped % per joint, window and total) and the ledger row 35c / FINDINGS H7c.
- Eval: if uniform +-20 is chosen, the MuJoCo eval adapter (adapters/hiking.py) must clip a_prev the same way (eval monitor).
