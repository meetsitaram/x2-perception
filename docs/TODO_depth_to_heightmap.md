# Depth camera -> height map for ACP (parked 2026-11-15; started 2026-11-16 on the local RTX 5090)

## Why
ACP (contact planner + tracker) reaches 96.5-99% on our apparatus route and 77.5% on the NIST one, but only with a
perfect height map read from the simulator (planner 17x11 scan, tracker 9x7 scan). On the real X2 the only usable terrain
sensor is an added Intel RealSense depth camera: the X2's built-in LiDAR (chest RoboSense E1R) is too noisy and sees the
robot's own hands (user's tests; neither X2 paper, SSR or CReF, uses it either). So the height scans must come from depth.

## How the ACP paper does it (with LiDAR, arXiv 2610.31608, Materials and Methods: LiDAR-based Neural Mapping, Deployment)
- G1 head-mounted Livox MID-360; DLIO LiDAR-inertial odometry gives the root state and deskewed scans.
- A neural network turns each scan into local grids (3 cm cells); predictions are fused into a global map the
  controller queries. Only non-empty predictions are fused; high-uncertainty cells are inpainted with the neighbourhood
  minimum, then median-filtered.
- Trained in simulation with simulated per-point positional noise and ray-divergence noise from the LiDAR specs.
- About 30 ms per 10 Hz scan (20,000 points) on the Jetson Orin NX CPU, alongside DLIO and the policy.
- The policy's height scan gets per-step Gaussian noise and a per-episode elevation offset in training (mapping drift).

## What the literature does with depth (survey: docs/findings/depth_camera_mounts_survey_20261115.md)
- 18 humanoid works (2024-06 to 2026-10): 11 depth-only, 5 LiDAR-only, 2 both. Of the depth works, 9 feed raw depth to the
  policy; 2 reconstruct a height map:
  - **DPL** (arXiv 2510.07152, 2025-10, TienKung Ultra, Orbbec depth): 5 depth frames + 50 steps of proprioception ->
    cross-attention (proprio query, depth keys) -> GRU memory -> rough map (MSE) -> conditional U-Net refine (L1, sharp
    edges). Sim depth by GPU ray casting against terrain AND the articulated robot (self-occlusion), noise: range-growing
    axial noise, lateral noise, dropout at uncertain and Sobel-edge pixels. Map 1 x 1 m at 5 cm. Real map error 3.25 cm
    (10.07 cm without self-occlusion modelling, 4.48 cm without the noise model). ~30 Hz, ~20 ms. Policy: blind backbone +
    perceptive modulator, distilled from terrain-specific privileged teachers.
  - **Under-base reconstruction** (arXiv 2512.07464, 2025-12, LimX Oli 1.65 m, RealSense D435i under the pelvis pointing
    down): depth -> gravity-aligned point cloud -> raw height grid -> small U-Net (height head L1 + training-only edge head
    BCE+Dice). Single frame, no memory. Map 1.2 x 0.8 m at 5 cm. Error 2.64 cm (3.58 without the edge head). 11 ms per
    frame on a Jetson Orin NX, 50 Hz. Real: 15/20 cm stairs, backward descent, unseen spiral stairs, 46 cm gap.
- Mounts: depth cameras are mostly head-mounted, facing forward, 45-57 deg down (mostly the factory G1 camera); SSR mounts
  a D435i at the waist, 50 deg down; one under the pelvis pointing down; no paper compares mounts or measures torso lean.
- Problems reported: forward-only field of view, leg/arm self-occlusion, stereo holes at step edges going down, sunlight
  and reflections. No paper uses a vertically mounted LiDAR on a humanoid; head MID-360 setups report sparse returns, drift,
  blurred step edges and missed risers.

## Our constraints
- Chest vs waist/pelvis mount: the X2 waist pitch is weak (about +-18 deg, 24 N*m, often saturated - FINDINGS X1/X2), so a
  chest camera tilts with every torso lean; the SSR paper mounts at the waist and does not drive waist pitch/roll. Our SSR
  and Hiking sims put the camera on torso_link (chest) - a deviation from SSR.
- Narrow forward FOV: turning in place at stations next to 4x4s/wedges needs memory of terrain behind/beside the robot.
- ACP needs both a 17x11 planner scan and a 9x7 tracker scan around the robot (incl. under and behind the feet).

## Proposed plan (draft, for the user to decide)
1. Decide the RealSense mount: simulate chest (torso_link, 59.5 deg down) and pelvis (below the waist, about 50 deg down)
   on the same stairs/steps/turn-arounds in MuJoCo on CPU; compare visible ground near the feet, self-occlusion and view
   change with torso lean.
2. Build a RealSense depth simulator: ray casting against terrain + the robot's own meshes, DPL-style noise (range-growing
   axial noise, lateral noise, edge/uncertainty dropout), latency.
3. Train a depth -> height-scan estimator (DPL-style: depth history + proprioception, GRU memory, U-Net refine, edge loss),
   output = ACP's 17x11 and 9x7 scans, fused over time with odometry; supervised by the simulator's true scans on ACP rollouts.
4. Run ACP's policy unchanged on the estimated scans (estimator under the frozen policy); DAgger: roll out with the
   estimator in the loop, label with true scans. Only if needed: fine-tune ACP on estimated scans.
5. Score on our apparatus and NIST with the estimated map; compare with the perfect-map scores (96.5-99% / 77.5%).
6. Needs a GPU node for training the estimator (none free now: ACP to about 9-10 AM PST 16 Nov, Hiking to about 3-4 PM PST
   15 Nov; the SSR node is shut down).

## Status
Parked by the user on 2026-11-15. The full literature review (deep-research run) was started and paused; partial notes
folder research_notes/Depth camera terrain mapping review/ (empty).

## Started 2026-11-16 (user: "start working on ... depth camera to height map teacher student learning phase so we can use
## acp similar to how we have the hiking model working"; "you can use the local 5090 gpu")
Code: `models/acp_depth/` (collect.py, student.py, train.py, run_v1.sh) and `eval/adapters/acp_depth.py`. Data and runs in
`models/acp_depth/runs/` (git-ignored). ACP export: Hugging Face `tinkerbuggy/x2-acp-contact-planner-tracker` in `release/ckpt/acp`.
- **Approach (plan step 3-4, first pass):** the student sits under the unchanged ACP. `acp_depth.py` wraps the eval's view:
  ACP's two height inputs (planner 17 x 11, tracker 9 x 7, height minus root height) are recorded from the simulator
  (teacher labels) or replaced by the student's estimate. ACP and the eval are untouched.
- **Camera:** the chest bracket's pose (Hiking variant H: torso_link (0.131, 0, 0.259) m, 61.6 deg down) plus a per-episode
  offset around +4 deg pitch / -1.5 deg roll, spread +-8 / +-4 deg: covers the first real mount (+8.25 / -3.5, leg
  calibration 2026-11-16) and the trained pose. 87 x 58 deg, 80 x 45, 25 Hz, ray-cast with the robot's own body (self-occlusion).
- **Data:** ACP with perfect scans on terrain-park `extended_v4k` tile routes (levels 0-7, all nine types); the apparatus
  and the eval ladder's courses stay held out. data_v1: 312 routes x 2 seeds.
- **Student:** depth CNN + proprioception MLP -> GRU (memory for what left the view) -> two heads; D435-style noise added in
  training (range-growing axial noise, edge and min-range holes, left strip, patches, repeated frames). L1, 1.3 s sequences.
- **Baseline on this PC:** ACP with perfect scans on our apparatus: 46/50 (92 %, CI 81-97 %; published 96.5-99 %).
- **Next:** score ACP + student on the same 50 seeds (run_v1.sh); then DAgger (student in the loop, relabel with true
  scans), real-depth checks against the D435 captures, Orin CPU timing.
- **Options kept open (user, 2026-11-16: "keep all options on the table"; the local GPU is available for any of them):**
  (a) estimator under the frozen ACP (running), improved by DAgger and ACP-sensitive loss weighting; (b) brief ACP
  fine-tune on estimated scans; (c) end-to-end depth encoder head inside ACP (distillation); memory: GRU (now) vs a fixed
  frame stack vs explicit leg + IMU odometry fusion of the visible patch. Decide on the per-region error (visible / under
  the feet / behind vs time since seen) and the closed-loop apparatus score against the 92 % perfect-map baseline.
- **Finding 2026-11-16 (isolation tests, our apparatus, same 50 seeds; partial counts at 10:52):** true scans 46/50 (92 %);
  planner's rear cells (x < -0.2 m) blanked 44/50 (88 %): the map behind is not needed; student v1 to the planner only
  26/29 (90 %); student v1 to the tracker only 13/27 (48 %); student v1 to both 15/39 (38 %). The tracker's 9 x 7 scan,
  not the planner's, is where the student's errors cost success, although its error is the smallest (0.8 cm flat, 3.4 cm
  terrain). Next: controlled errors on the tracker scan only (2 cm noise; +-3 cm offset per episode) to see which error
  kind hurts; then precision under the feet (DAgger, loss weights) or a brief ACP tracker fine-tune with scan noise and
  offsets (the ACP paper trains its policy that way).
- **Result 2026-11-16 (our apparatus, seeds 2000-2049):** student v1 18/50, v2 35/50 (794 teacher episodes), **d1 46/50 = the
  perfect-map baseline** (one DAgger round: +624 episodes on v2's own scans), d2 38/50 (second round; lower validation error
  but worse closed loop: validation error does not rank students). Tracker-scan errors: +-3 cm offset 30/30, 2 cm noise
  24/30: shape errors at edges hurt, not bias. Confirming d1 vs the truth on 100 fresh seeds (3000-3099), paired.
- **Confirmed 2026-11-16 (100 fresh seeds 3000-3099, paired):** student d1 85/100 = perfect scans 85/100; both 71, only d1 14,
  only truth 14, neither 1. Closed loop on clean ray-cast depth (camera pose randomised). Open: closed loop with the D435-like
  noise in the loop (ACPD_DEPTH_NOISE=1), map accuracy (the 4x4 comes out 4-6 cm of 8.9; stair descents 7-10 cm too shallow;
  a +2-3 cm upward bias everywhere; foot anchoring ACPD_ANCHOR=1 removes part of the bias, not the shape error), real D435
  captures. Edge-weighted loss (train.py --edge-w 10): student_e1, being scored.
- **Renders (user, 2026-11-16):** always `models/acp_depth/render_showcase.sh STUDENT.pt OUT "label" [--noise] [--anchor]`: ACP's
  overlay from the maps ACP was given (tiles, dots, landing rings, stance squares), the chest depth inset bottom right, the
  apparatus with its real textures and a raised chase camera, and the same four scenes (apparatus seed 2007, stairs up / down,
  4x4 log). Check the check_*.jpg frames before sharing.
- **Fine-map student (started 2026-11-16 evening):** the 4x4 (8.9 cm wide) on 10 cm sample points flickers in the labels and
  the grid students hedge at ~half height. `models/acp_depth/fine_student.py`: depth projected through the camera pose (IMU
  attitude + leg kinematics) into a 2.5 cm bird's-eye grid (max / mean height, count, seen), the grids of 4 / 8 / 12 frames
  before as short memory, a small U-Net -> fine map (52 x 44, x -0.40 .. 0.875 m), ACP's 17 x 11 / 9 x 7 read off it. Geometry
  check: projected depth vs the true fine map 0.5 mm median. Data: data_v3 (teacher, beam tiles oversampled) + data_v3d (DAgger
  on d1 with D435 noise in the loop; clean depth recorded). Chain: models/acp_depth/run_fine.sh (train, score clean + noisy on
  seeds 2000-2049, showcase renders, 4x4 height).
- **End-of-course spin (2026-11-16, user: "why is the robot spinning around at the end?"):** on a hold at a waypoint with no
  station heading, eval/adapters/acp.py (line 267) faces the bearing of the waypoint, which swings wildly once the robot stands
  on it: 200-490 deg of turning in the last 4 s of the ladder clips. acp_depth now keeps the arrival heading within 0.15 m
  (ACPD_HOLD_HEADING=1, default): 0-1 deg. Success is decided before the spin, so scores are barely affected. OPEN: the same
  bug is in the shared eval adapter that scores every ACP run (not changed; the user decides).
- **Fine-map results 2026-11-17 (seeds 2000-2049; perfect scans 46/50):** fine_v1 (30 epochs, data_v3 + v3d) clean 41 / noisy
  44; fine_v2 (150 epochs) 43 / 44; **fine_v3 (+ data_v4d, DAgger on fine_v1 with noise) clean 41 / noisy 46 = the perfect-map
  baseline with D435-like noise in the loop.** 4x4 (true 8.9 cm): tracker 9.2 cm, planner 7.7 cm (d1: 4.1 / 5.6). Showcase
  (vis_fine3): all five clips succeed. Clean depth scores lower than noisy: the fine student only ever trained on noisy depth
  (next: a share of clean frames in training). d1 (10 cm grid) for comparison: clean 46, noisy 35.
