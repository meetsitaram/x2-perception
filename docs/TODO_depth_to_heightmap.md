# TODO: depth camera -> height map for ACP (parked 2026-11-15, user: "We will work on it later")

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
