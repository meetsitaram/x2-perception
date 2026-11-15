# Simulator-artefact audit of the MuJoCo CPU eval (our apparatus), 2026-11-12

Independent audit asked for by the coordinator after finding B8 (user: "Make sure there are no sim artifacts like that").
Scope: `eval/harness.py`, `eval/x2.py`, the "ours" course, the newest apparatus runs of the three tracks (ACP pair 7,000 and
5,750, SSR r7_2250, Hiking 4,125 and 3,000). Nothing under `eval/`, `eval/out/` or `docs/` was changed except this file.
All scripts, logs and re-simulations are in
`C:\Users\PC\AppData\Local\Temp\claude\74893d75b5436cbb\cf3ca1e7-ff31-44d4-97d3-238f0a3bcf48\scratchpad\audit_sim\`
(815 MB; `resim/` holds the per-physics-step logs).

**How things were measured.** (1) *Replay*: recorded `traj/<seed>.npz` (qpos at 50 Hz) set into the eval's own model, `mj_forward`:
geometry only (distances, contact normals), no forces. (2) *Re-simulation*: the eval's `Trial` class imported unchanged and run
through a wrapper that logs every 2.5 ms physics step (per foot: contact count, normals, deepest contact, the NET force vector of
the floor on the foot, the sum of the contacts' force magnitudes, the harness's `load`). The wrapper reproduces the recorded
trials bit for bit (Hiking 4,125 seeds 1002 and 1006, ACP 7,000 seed 1028: max |dqpos| = 0 over 417, 950 and 1,150 frames), so
its force numbers are the forces of the recorded trials. (3) *Micro-test*: a 10 kg foot with the X2's sole spheres dropped on a
flat floor (`micro.py`). (4) *Variants*: the same harness with a test option switched on, same seeds.
"Verified" below means measured by one of these; "inferred" is marked.

## 1. The answer in short

1. **There is a real contact artefact, and it is not the one B8 guessed.** A 2 cm sole sphere that goes more than about 1 cm into
   the height field is pushed sideways by the neighbouring 2 cm cells (triangle prisms) instead of up; deeper than its radius
   (2 cm) it is caged: 100-160 contacts with horizontal normals squeeze it with 50-170 kN in total, friction on those contacts
   holds the foot in the floor, and the net force on the foot is a few hundred newtons, at times pointing DOWN (the floor pulls,
   up to -489 N). The foot then creeps out at about 0.1 m/s: 0.3-0.7 s in the recorded trials, 43 s in one test run. It
   happens on **flat** raster (deck, stair treads), not at the 4x4's sloped sides.
2. **What sinks the foot is touchdown speed times the soft contact** (MuJoCo default `solref` 0.02 s): measured 7.15 mm of
   penetration per m/s of vertical foot speed (theory 6.6). Over about 1.5 m/s the first horizontal normals appear, over
   about 3 m/s the sphere centre is under the surface and the foot is trapped. A plane with the same softness sinks just as
   deep and releases in 0.05 s. It clusters at the 4x4s and on the stairs because that is where the policies lift a foot
   high and stamp (touchdowns of 2.7-4.4 m/s there; normal steps are 0.3-1 m/s).
3. **The trigger event of B8 is this artefact, in its mild form**: 22.6 mm deep for 0.10 s (not 1.2 s), net force on the foot
   at most 1,478 N; the 3,289 N is one contact of a squeeze whose net horizontal force was 36-97 N. It did not cause that fall
   (3.6 s later) and the foot was not held.
4. **Every force, load and impulse figure of the eval is wrong in such moments** (by a factor of 15-30): the harness adds
   magnitudes of contacts that cancel. On the landing the opposite happens: the logged foot load is 40 % too LOW (section 5, d).
5. **Effect on results: small for ACP, none for Hiking and SSR, as far as 50 seeds can tell.** With the artefact removed
   (floor contact 0.005 s, no trapping at all) ACP 7,000 scores 34/50; a control run with physically identical contact
   (0.01995 s) scores 31/50; the recorded run 26/50. The control alone flips 21 of 50 outcomes, so differences of this size are
   noise (section 6). Direct count: a trapped foot within 3 s of the end in 3 of 16 kept ACP 7,000 failures, 2 of 16 for 5,750,
   1 of 22 for Hiking 3,000, 0 of 20 for Hiking 4,125, 0 of 7 for SSR. Hiking 4,125 stays 0/20 with boxes for the 4x4s and 0/20
   with a stiffer floor.
6. Other artefacts found: joint limits are soft (ankle roll up to 0.36 rad past its stop); the landing deck exists twice
   (raster and wall box 202); a head clamped between two wall boxes for 50 s; toes and knees 4-5 cm inside stair risers; a brace
   rasterised as a solid wedge with a cut-off cliff. Not found: tunnelling through walls, falling through the floor, energy
   injection, pelvis velocity jumps.

## 2. The trigger: Hiking 4,125, seed 1002, t 4.6-6.0 s (verified)

Per recorded frame (replay; `replay1002.txt`) and per physics step (re-simulation; `resim1002_steps.txt`). Right foot:

| t (s) | terrain contacts (horizontal-normal) | deepest sphere, signed distance | net floor force on the foot x / y / z (N) | sum of contact magnitudes (N) | harness `load` (N) | largest single contact (N) |
|---|---|---|---|---|---|---|
| 4.7000 | 0 | +1.9 mm (foot falling at 2.7 m/s, sideways 2.0 m/s) | 0 | 0 | 0 | 0 |
| 4.7050 | 23 (0) | -5.4 mm | -526 / -17 / 1,382 | 1,675 | 1,556 | 508 |
| 4.7100 | 60 (26) | -15.0 mm | -117 / 86 / 1,346 | 25,487 | 14,292 | 843 |
| 4.7200 | 82 (52) | -21.8 mm | 8 / 59 / 870 | 70,144 | 21,111 | 2,574 |
| 4.7300 | 82 (58) | **-22.6 mm** | 34 / 36 / 722 | **76,002** | 15,177 | **3,291** |
| 4.7600 | 64 (42) | -20.1 mm | 77 / -20 / 589 | 54,953 | 16,083 | 2,593 |
| 4.8000 | 38 (2) | -12.4 mm | 6 / -74 / 427 | 9,317 | 5,675 | 796 |
| 4.8300 | 21 (2) | -5.2 mm | 30 / -39 / 327 | 1,173 | 748 | 238 |
| 4.86-5.10 | 2-29 (2) | -3 to -1 mm | about 30 / -25 / 290 | 550-1,200 | 150-450 | under 250 |
| 5.32-5.94 | 2-30 (0-4) | -2 to 0 mm | | | 150-480 | |

- The left foot has no deep contact in the window (deepest 9.8 mm at t 5.04, no horizontal normals, load 3,421 N logged).
- **Is the 3,289 N contact a penetration artefact? Yes.** It is one of up to 58 horizontal-normal prism contacts on sole
  spheres that are 20-22.6 mm inside flat floor (sphere radius 20 mm: the centre is 2.6 mm under the surface), 0.13 m from
  the front beam's centre line. The contacts oppose each other: while that contact reads 3,289 N, the net horizontal force on
  the foot is 36-97 N and the net vertical force 630-735 N.
- **How deep, how long:** over 10 mm from t 4.7075 to 4.8075 (0.10 s), over 20 mm for 0.05 s, under 5 mm from t 4.83.
- **The logged figures:** `peak_horizontal_N` 3,288.8 = one contact; `impulse_Ns` 1,803.8 = force magnitudes summed over time,
  not a momentum (the net floor impulse on the foot over the impact is 78 N s up, 1.1 and 2.4 N s horizontal); `dur_s` 1.237
  = event grouping (contacts less than 0.2 s apart are one event): after t 4.83 they are light touches (2 contacts, 100-450 N)
  of the toe sphere on the beam's side. Whole trial: the largest NET floor force on a foot is 1,917 N (4.5 x body weight),
  while `load` peaks at 21,153 N.
- **The eval monitor's reading** ("one impact frame at t 4.80, 11.7 mm, 5,675 N") describes the tail of the impact; the peak
  is at t 4.72-4.73 and is twice as deep and four times as high. Its other two points (grouping, summed magnitudes) are right.

## 3. The mechanism (verified)

**Trapped feet in the recorded trials** (re-simulated, bit-identical):

| | Hiking 4,125 seed 1006, left foot, t 17.81 | ACP 7,000 seed 1028, left foot, t 21.53 |
|---|---|---|
| foot speed just before (x, y, z) | -2.7, 0.5, **-4.2** m/s | -4.3, 0.7, **-3.9** m/s |
| place | flat deck (2.81, 0.59), 0.22 m from the front beam | flat deck (2.79, 0.95), 0.29 m from it |
| deepest / time deeper than 1 cm | 23.8 mm / 0.45 s | 25.2 mm / 0.34 s |
| contacts (horizontal-normal), at most | 153 (109) | 160 (135) |
| sum of contact magnitudes, at most | 142,700 N | 173,370 N |
| harness `load`: mean, peak | 8,110 N, 20,013 N | 11,909 N, 26,185 N |
| NET vertical floor force: mean, range | **193 N**, -489 to 1,476 N | **551 N**, 53 to 2,542 N |
| foot speed while held | about 0.1 m/s (rising out 15 mm in 0.4 s) | about 0.15 m/s |

The solver is not at fault (2-11 Newton iterations, never at the 100 limit); the contact model is: a height field collides as
independent prisms, and each prism pushes a sphere out the shortest way, which for a deep sphere is sideways.

**Touchdown speed against depth**, 122 touchdowns of the three re-simulated trials (`impacts.py`):

| vertical speed at touchdown | n | deepest (median / max) | horizontal-normal contacts (median / max) | time deeper than 1 cm (max) | `load` peak (median) | NET vertical force peak (median) |
|---|---|---|---|---|---|---|
| under 0.5 m/s | 46 | 3.5 / 16.2 mm | 0 / 18 | 0.035 s | 622 N | 579 N |
| 0.5-1.0 | 46 | 5.8 / 11.3 mm | 0 / 10 | 0.025 s | 809 N | 707 N |
| 1.0-1.5 | 7 | 10.3 / 14.9 mm | 0 / 16 | 0.033 s | 3,658 N | 903 N |
| 1.5-2.0 | 3 | 12.2 / 19.7 mm | 22 / 50 | 0.065 s | 10,966 N | 1,370 N |
| 2.0-3.0 | 8 | 17.8 / 22.6 mm | 28 / 78 | 0.152 s | 17,943 N | 1,226 N |
| over 3.0 | 2 | 24.5 / 25.2 mm | 122 / 135 | 0.448 s | 23,099 N | 2,009 N |

**Micro-test** (`micro.txt`; 10 kg foot, 12 sole spheres, flat floor, 1 m/s sideways, then a 150 N sideways pull):

| floor and sole | at 3.0 m/s: deepest / time over 1 cm / horizontal normals | at 4.5 m/s: the same | moved by the pull in 0.2 s after a 4.5 m/s impact |
|---|---|---|---|
| 2 cm height field, solref 0.02 (the eval) | 19.9 mm / 0.078 s / 137 | 24.9 mm / **never released (0.79 s, end of test)** / 154 | **0.00 m** |
| plane, solref 0.02 | 18.7 mm / 0.043 s / 0 | 26.4 mm / 0.053 s / 0 | 0.135 m |
| 2 cm height field, solref 0.01 | 7.4 mm / 0 / 0 | 9.6 mm / 0 / 0 | 0.184 m |
| 2 cm height field, solref 0.005 | 2.9 mm / 0 / 0 | 2.7 mm / 0 / 0 | 0.163 m |
| 1 cm height field (finer cells) | 20.7 mm / 0.44 s / 438 | 22.8 mm / never / 451 | 0.00 m |
| 4 cm height field (coarser) | 19.5 mm / 0.063 s / 51 | 29.7 mm / never / 64 | 0.00 m |
| 4 cm sole spheres | 17.2 mm / 0.027 s / 0 | 26.1 mm / 0.053 s / 172 | 0.135 m |
| box sole on the height field | 19.4 mm / 0.098 s / 36 | 27.6 mm / 0.143 s / 50 (the 50-contact cap), floor pulls down up to 201 N | 0.009 m |
| contact margin 5 mm | 23.1 mm / 0.053 s / 144 | 30.0 mm / never / 180 | 0.003 m |
| solimp 0.99 / 0.999 | as the eval | as the eval | 0.00 m |
| the MJCF's 5 mm spheres | caged already at 0.5 m/s: 38 of 45 normals horizontal | | 0.00 m |

**Why at the 4x4 and not everywhere on the deck** (the coordinator's question): impact speed, not foot orientation or sphere
layout. Share of recorded frames with a logged foot load over 1,500 N (which needs a touchdown of about 1.2 m/s or more), kept
trajectories (`loadmap.py`): start and door 0.0-0.2 %; 4x4 area 0.7-3.1 % (all tracks); ramps 0.5-1.3 %; stairs 9-10 % (ACP,
the only track that gets there); landing 0.8-2.5 %. Episodes of 0.2 s or more: ACP 7,000: 7 at the 4x4s, 2 on the stairs;
ACP 5,750: 1 stairs, 1 ramps; Hiking 4,125: 6 at the 4x4s; Hiking 3,000: 4 at the 4x4s, 2 on the ramps; SSR: none (it does
not stamp: 0 s of trapped foot in 7 trajectories). These counts match the eval monitor's (7/18, 5/22, 5/20, 0/7 trajectories).

**The eval monitor's stiff-floor test is weaker than labelled** (verified): `TEST_TERRAIN_SOLREF=0.005` sets the terrain geom
only; MuJoCo averages it with the sole sphere's 0.02, so the contacts run at **0.0125 s** (read from `d.contact[].solref`).
That is still enough to nearly remove trapping (below), but it is not the 0.005 the label says.

## 4. Strong contacts in events.csv, all tracks (verified counts; the forces themselves are squeeze readings)

A = peak force over 1,500 N and longer than 0.2 s; B = peak over 3 x body weight (1,279 N), any duration. `scan_events.txt`.

| Run (strict rule) | trials / falls | A: contacts (trials) | A by part and obstacle | A followed by a fall within 3 s: contacts (trials) | fall rate with A / without | B: contacts (trials) | fall rate with B / without |
|---|---|---|---|---|---|---|---|
| ACP 7,000 | 50 / 18 | 65 (37) | foot: stair 19, 4x4 18, wedge 12, wall 7, other 4; head wall 2; arm, trunk, thigh 1 each | 21 (14) | 15/37 / 3/13 | 120 (43) | 18/43 / 0/7 |
| ACP 5,750 (200) | 200 / 72 | 163 (80) | foot: stair 87, wedge 15, 4x4 11, wall 9, other 6; thigh 17 (stair 16); knee 8; head wall 5; arm 3; shin 2 | 59 (32) | 37/80 / 35/120 | 362 (131) | 57/131 / 15/69 |
| SSR r7_2250 | 50 / 1 | 4 (4) | foot: 4x4 2, wedge 1, wall 1 | 0 | 0/4 / 1/46 | 20 (15) | 0/15 / 1/35 |
| Hiking 4,125 | 50 / 35 | 55 (33) | foot: 4x4 39, other 7, wall 2, wedge 2; knee 4x4 1; arm, head, trunk, thigh 1 each | 18 (12) | 20/33 / 15/17 | 131 (45) | 30/45 / 5/5 |
| Hiking 3,000 | 50 / 29 | 32 (23) | foot: 4x4 14, other 8, wedge 4, wall 3; thigh 2; trunk 1 | 13 (9) | 15/23 / 14/27 | 94 (36) | 24/36 / 5/14 |

- By body part it is almost entirely the foot; "floor" (no face) does not appear because events.csv only logs foot contacts
  that count as side contacts, so the flat-floor squeezes are in it only where a face is within 4 cm (beside a beam, at a
  riser). The stair figures of ACP are squeezes on treads next to risers.
- Trials with such contacts fall more often for ACP (41-46 % against 23-29 %) and less often for Hiking (61 % against 88 %).
  This does not show cause: a stumble produces both the stamp and the fall. Section 6 is the causal test.
- Largest readings: head on wall 35,291 N for 50.6 s (ACP 7,000 seed 1002, see 5a), thigh on a stair 8,415 N (ACP 5,750 seed
  1177, a fall), foot on a wall 5,426 N (Hiking 4,125 seed 1043).

## 5. Artefact classes in the kept trajectories (replay, 85 trajectories, 207,000 frames)

Counts per run: ACP 5,750 (18 trajectories) / ACP 7,000 (18) / Hiking 3,000 (22) / Hiking 4,125 (20) / SSR 2,250 (7).
`s2_*.txt`, `scan2_*.json`, `trapped.py`.

| Class | Found | Count | Worst example (run, seed, time) |
|---|---|---|---|
| (a) penetration over 1 cm held over 0.1 s: foot in terrain | **yes** | 96 / 138 / 134 / 198 / 27 contact episodes (plus 21 / 24 / 30 / 11 / 0 in the last second of a fall) | Hiking 4,125 seed 1011, t 4.70: 21.5 mm at (2.98, 0.83) |
| (a) limbs in wall boxes | **yes, few** | head 0 / 4 / 0 / 0 / 0; arm 0 / 0 / 5 / 3 / 0; thigh 0 / 0 / 2 / 0 / 0; knee 0 / 0 / 1 / 0 / 0; foot 0 / 1 / 0 / 0 / 0 | ACP 7,000 seed 1002, t 13.2-63.8: head 10 and 12 mm inside wall boxes 122 (post) and 129 (frame bar) with opposite normals, pelvis hanging at 0.87 m for 50 s; scored "stalled" |
| (a) self-penetration | yes, few | 0 / 0 / 4 / 3 / 0 (hip-yaw link against the same side's wrist, 14-18 mm) | Hiking 4,125 seed 1010, t 12.04: 17.8 mm (known open point of A1: hulls too deep) |
| (a) knee or thigh mesh in the stair raster | yes, few | knee 2 / 2 / 0 / 0 / 0; thigh 1 / 0 / 0 / 0 / 0 | ACP 7,000 seed 1009, t 75.12: knee 16.9 mm in the riser at (8.84, 0.46) |
| (b) sole sphere centre under FLAT floor (trapped foot) | **yes** | 100 / 328 / 197 / 568 / 0 sphere-frames; episodes of 0.2 s or more in 2 / 5 / 1 / 5 / 0 trajectories, longest 0.70 s | Hiking 4,125 seed 1035, t 21.78: centre 25.6 mm under the deck at (2.81, 0.67) |
| (b) sole sphere over 1 cm inside a face or edge (riser, 4x4 side, brace wedge) | **yes** | 440 / 471 / 489 / 459 / 20 sphere-frames | ACP 7,000 seed 1009, t 75.14: 44.8 mm inside a riser; Hiking 4,125 seed 1003, t 8.66: 48.2 mm inside the brace wedge at (3.39, 1.06) |
| (b) passing through thin geometry between frames (any body origin through any of the 344 wall boxes) | **not found** | 0 in all runs | limit: body origins only, 50 Hz |
| (c) pelvis speed gain over 2 m/s within 50 ms | **not found** | 0 in all runs (largest in the re-simulated trials 0.44 m/s) | |
| (c) foot speed gain over 2 m/s within 50 ms | found, not an artefact | 88 / 112 / 111 / 139 / 63 | the largest ones in the re-simulated trials happen with the foot in the air or in light contact (leg swing); the floor's net work on the feet is negative (-180 to -430 J per trial against +60 to +170 J returned); no energy injection seen. Inferred for the trials not re-simulated |
| (d) feet carried by geometry that should not carry: wall box 202 (the landing deck plate, top 1.118 m = the raster) | **yes** | 16,073 / 11,144 / 0 / 0 / 0 contact-frames (every step on the landing) | ACP 7,000 seed 1018, t 66.78. Effect measured: the harness leaves wall-box contacts out of `load`, so on the landing the two feet read a median 0.59-0.61 of body weight (0.97-0.98 on the deck) and 11-19 % of frames read under the strict rule's 0.5 |
| (d) solid where the real part is open: brace rasterised as a wedge, cut off at 0.5 m | yes | raster at x 3.36-3.46, y 1.07-1.17 rises 0.05 to 0.356 m and drops to 0 in one cell | same kind as B6; one foot 48 mm inside it during a fall (above) |
| (d) falling through geometry (pelvis or foot origin under the raster) | **not found** | 0 | |
| (e) joints past their limits by over 0.05 rad | **yes** | 854 / 785 / 1,172 / 594 / 20 joint-frames outside the last second of a fall; mostly ankle roll (limit +-0.263), then knee and ankle pitch | Hiking 3,000 seed 1045, t 45.02: right ankle roll 0.362 rad past the stop; ACP 7,000 seed 1028, t 21.74: 0.25 rad while that foot is trapped. MuJoCo joint limits are soft (default 0.02 s) |
| (f) horizontal contact normals (under 0.5) on flat raster | **yes** | 212 / 294 / 188 / 408 / 44 foot-frames | Hiking 4,125 seed 1006, t 17.84: 109 of 153 contacts horizontal, 44 deeper than 1 cm, `load` 13,233 N |
| (g) height-field ramps where the part is vertical | **yes, as documented** | see below | |

**(g), measured on the raster** (`cad_20mm_walk.npz`):
- 4x4 side: height goes 0 to 0.089 m between two neighbouring vertices, no intermediate cell: a sloped band **2 cm wide,
  77.3 degrees** (as docs/EVAL.md says). The modelled beam is a trapezoid: flat top 6-8 cm, base 10-12 cm (the part: 8.9 cm
  square). The beam runs 8 degrees off the grid, so its edge is a staircase in plan: a 2 cm jog every 12-14 cm.
- Stair riser: 0.186 m over one cell: **2 cm wide, 83.9 degrees**; the top riser is two steps (0.167 m, then 0.019 m). No
  intermediate cells on the others.
- Deck edge: 0 to -0.108 m in one cell (79.5 degrees).
- These ramps are not what traps feet (the trap is on flat cells), but toes do sink into them: sole spheres up to 37-48 mm
  inside a riser, a beam or the brace wedge (class b).

## 6. Do the artefacts change outcomes? (variants, same seeds, verified)

**Step 4 of the brief (cheap, done).** `eval/x2.py` already has the test option `TEST_BOX_4X4=1` (beams as boxes, their cells
flattened in the physics only), so no copy of the course was needed. Hiking 4,125, seeds 1002 / 1003 / 1004:

| | 1002 | 1003 | 1004 |
|---|---|---|---|
| standard (recorded) | fall, 2/22, door | fall, 2/22, door | fall, 2/22, door |
| 4x4s as boxes | stalled, 4/22, 4x4s | fall, 4/22, 4x4s | fall, 2/22, door |
| floor contact 0.0125 s (`TEST_TERRAIN_SOLREF=0.005`) | fall, 2/22 | fall, 3/22 | fall, 9/22 |
| both | stalled, 12/22, ramps | stalled, 4/22 | stalled, 9/22 |

The outcome changes in detail and not in substance: no success, all still fail between the door and the ramps. (The eval
monitor's box results for these seeds are identical to mine, run independently.)

**Wider, because single seeds mean nothing here:**

| ACP pair 7,000, seeds 1000-1049, strict rule | success | falls / stalls | failures at 4x4s / ramps / stairs | time with a contact deeper than 2 cm, per 100 s |
|---|---|---|---|---|
| standard (recorded) | 26/50 = 52 % (39-65) | 18 / 6 | 16 / 5 / 3 | not logged |
| control: floor contact 0.01995 s (physically the same) | 31/50 = 62 % (48-74) | 11 / 8 | 7 / 7 / 5 | 1.85 s |
| floor contact 0.005 s effective (terrain priority; no trapping) | 34/50 = 68 % (54-79) | 12 / 4 | 6 / 4 / 6 | 0.00 s |
| floor contact 0.0125 s (the monitor's option), seeds 1000-1019 | 14/20 (control on the same seeds 11/20) | 1 / 5 | 2 / 4 / 0 | 0.00 s |

- **The control flips 21 of 50 outcomes** (13 failures become successes, 8 the reverse) with a change of 0.25 % in one contact
  parameter. A trial's outcome is that sensitive; only rates over many seeds carry information.
- Control to stiff floor: 31 to 34 (11 failures become successes, 8 the reverse; not significant). Recorded to stiff floor:
  26 to 34 (15 against 7; p about 0.13). Pooled, 57/100 against 34/50: +11 points, interval includes zero.
- **The eval monitor's test on 8 selected failing seeds** (boxes: 5 of 8 pass; 0.0125 s: 7 of 8 pass) is not evidence: the
  control passes 5 of the same 8 (1002, 1012, 1013, 1020, 1025). Picking failures and perturbing anything gives this.
- One control run shows what the artefact can do at worst: seed 1046, left foot stuck over 2 cm deep in a stair tread at
  (8.72, 0.53) for **43.5 s**, scored "stalled" at 19/22. In the recorded kept trajectories no episode is longer than 0.70 s.

| Hiking 4,125, seeds 1000-1019 | success | falls / stalls | mean waypoints |
|---|---|---|---|
| standard (recorded) | 0/20 | 13 / 7 | 4.4 (4.5 over all 50) |
| 4x4s as boxes | 0/20 | 12 / 8 | 5.2 |
| floor contact 0.0125 s | 0/20 | 19 / 1 (15 at the door turn) | 3.1 |

**Judgement: how many recorded failures are likely affected**

| Track, run | failures | trapped foot (0.2 s or more) starting within 3 s of the end, kept failures | judgement |
|---|---|---|---|
| ACP 7,000 strict 50 | 24 | 3 of 16 kept (seed 1040, fall heading to the ramps, foot trapped 0.70 s at (3.00, 2.26) 1.5 s before; seed 1010, fall at the stair foot, 0.38 s, 0.3 s before; seed 1009, fall on the stairs, 0.20 s, 1.9 s before) plus the head clamp of seed 1002 (a stall) | about 3 to 6 of the 24 (measured 4 of 16 kept; the variants allow 0 to about 8). Not the 13 falls at the 4x4s as a block |
| ACP 5,750 strict (50 kept run) | 29 | 2 of 16 kept (falls on the stairs, seeds 1015 and 1004) | about 3 to 5; for 5,750 the stairs, not the 4x4s |
| Hiking 4,125 | 50 | 0 of 20 (5 trapped episodes, all 24-66 s before the end) | none. It fails without the artefact too |
| Hiking 3,000 | 50 | 1 of 22 (seed 1022, 0.24 s, 0.3 s before the fall) | 0 to 2 |
| SSR r7_2250 | 50 | 0 of 7; its feet are never trapped | none (inferred from the scan; no SSR variant was run). Its failures are stalls |

The artefact also acts in the other direction (a caged foot cannot slip), which is one reason the net effect is small.

## 7. Fixes for the eval, in order of value

1. **Make the measurements right (no physics change, no re-baseline needed).** In `harness.py` `_contacts`: foot `load` = the
   vertical component of the summed world-frame contact forces on the foot, not the sum of `fn * |nz|`; `peak_h` and
   `impulse` from the summed force vector per limb and class (impulse = length of the time integral of the vector); `dur_s` =
   time actually in contact. Count wall box 202 in `load` (or see 3). Until then no force, load or impulse figure from the
   4x4s or stairs should be quoted, and the render banners' forces are wrong by up to 30 times. The "stance" test (40 N) and
   the strict stand's load share use `load` too.
2. **Flag artefact trials automatically.** Per trial: seconds with a sole sphere centre under the surface, the longest
   episode and its time before the end; non-foot bodies with opposing wall contacts over 1 cm for over 1 s (head clamp);
   joints over 0.05 rad past a limit. Write them to trials.csv; report failures with a trapped foot in the last 3 s separately.
3. **Remove the trap.** In order of preference:
   a. *Primitives for the physics floor*: deck and ground as planes or boxes, the two 4x4s as boxes (the test option exists),
      the six stair steps and the landing as boxes (box 202 is already there), the four ramp cells as tilted boxes; keep the
      raster for the height scans, depth and the side-face test. Exact normals, true vertical faces and beam width (also ends
      g and the double landing), no trap at any speed (micro-test: plane). About half a day; needs a new regression reference.
   b. *Stiffer floor contact, cheap*: terrain `solref` 0.005-0.01 s **with `priority = 1` on the terrain geom** (or the same
      value on the sole spheres); without the priority MuJoCo mixes it to 0.0125 s. At 0.005 s effective: 0.00 s of contact
      deeper than 1 cm in 50 ACP trials; micro-test 2.7 mm at 4.5 m/s. Impacts get harder (net peak force 2 to 4 times
      higher), which is closer to rigid ground and to Hiking's trainer (PhysX), further from the 0.02 s ACP and SSR presumably
      train with (inferred). 0.005 s is the smallest value MuJoCo accepts at this time step (2 x dt).
   c. Not worth doing: finer or coarser cells, contact margin, solimp, a box sole on the height field (all tested, none helps
      or worse); 4 cm spheres help only up to about 3 m/s.
   Either a or b changes every trajectory, so earlier numbers stay comparable only as rates; re-run the baselines.
4. **Landing:** take wall box 202 out of contact (`APP_NOCONTACT_WALLS=202` exists) or build the landing from it alone (3a).
5. **Joint limits:** give the limits a stiffer `solreflimit` (or log the excess); ankle roll passes its stop by 10-20 degrees.
6. **Compare checkpoints only by rates over all seeds**, never by re-running chosen failing seeds; when a test variant is
   run, run a control (for example `TEST_TERRAIN_SOLREF=0.0199`) beside it. With 50 trials two rates must differ by about
   20 points to mean anything.
7. **Raster wedges** (B6 and the brace at (3.4, 1.1)): replace by their boxes when 3a is done.

**Training simulators (inferred from `tracks/`, not measured; MuJoCo Warp was not run):** SSR trains on a height field with the
same 2 cm sole spheres plus 2.2 cm guard spheres and no `solref` set (`models/ssr/code/ssr/x2.py`), so the same trap is
possible there. ACP's height field collides with the MJCF's 5 mm sole spheres (conaffinity 1) and a foot box 1.5 cm above them
(`gear_sonic/supermimic/assets.py`, `terrain.py`), 5 ms step; in the CPU micro-test 5 mm spheres on a 2 cm height field are
caged even at 0.5 m/s and a box sole sticks partly. Worth a ten-minute drop test in each track's own simulator: a policy that
learned on sticky treads may rely on them.

## 8. What was verified and what is inferred

- Verified by measurement: sections 2, 3 (three recorded trials, bit-identical), the micro-test, all counts of sections 4 and 5,
  the raster slopes, the landing's load share, the mixed `solref`, all variant outcomes of section 6.
- Inferred: that the stamps cluster at the 4x4s because of high steps (speeds measured in 3 trials, places in all kept ones);
  the per-track numbers of affected failures (kept trajectories are at most 5 per section and kind, not all failures; ACP's
  200-trial runs keep none); SSR "none"; that foot speed jumps outside the 3 re-simulated trials are swings; everything about
  the training simulators.
- Not done: NIST course, randomised protocol, ladder courses, a primitive-floor build, variants for SSR.
- Machine use: at most 2 cores, except about 3 minutes with a third process (the trapped-foot scan overlapped two queues).
  All audit jobs have ended.
