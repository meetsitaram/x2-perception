# Terrain sensors and mounts in perceptive legged locomotion: survey (2026-11-15)

Scope: what terrain sensor and mount recent perceptive humanoid works use (plus key quadruped works), what the
policy consumes, and any stated reasons or reported problems. Sources: arXiv HTML pages (method, hardware and
limitations sections), read 2026-11-15. "Not stated" means the paper text I read does not say; inferences are
marked as such. Dates are the first arXiv submission month unless noted.

## Humanoids

| # | Paper | Date | Robot | Sensor (model) | Mount, tilt, above/below waist joints | Policy consumes | Stated reason / reported problem |
|---|---|---|---|---|---|---|---|
| 1 | [Humanoid Parkour Learning](https://arxiv.org/abs/2406.10759) (CoRL 2024) | 2024-06 | Unitree H1 | Depth, RealSense D435i | Head; sim pitch randomised 0.77-0.99 rad (44-57 deg down). Above the H1 torso yaw joint (H1 has no waist pitch) | Raw depth 48x64 -> CNN latent at 10 Hz | No mount reason. Noise handled by simulated D435 noise, hole filling, smoothing; latency 60-120 ms randomised |
| 2 | [PIM: Perceptive Internal Model](https://arxiv.org/abs/2411.14386) | 2024-11 | H1, Fourier GR-1 | LiDAR, Livox MID-360 (+FAST-LIO); alternative D435 + T265 | Mount not stated (H1's MID-360 is the factory head unit) | Height map from LiDAR (GPU elevation mapping), 96 points over 0.8x1.2 m | Map is "less affected by camera movement or noise" than raw input; LiDAR setup more robust than D435/T265 under rapid/irregular motion |
| 3 | [BeamDojo](https://arxiv.org/abs/2502.10363) | 2025-02 | Unitree G1 | LiDAR, MID-360 (+FAST-LIO) | Mount not stated in paper; G1's MID-360 is factory-mounted in the middle of the head, physically inverted (needs driver flip) | Height map from LiDAR, 15x15 at 0.1 m, 10 Hz | Raw cloud "suffers from significant occlusion and noise"; odometry jitter and map drift are the main deployment problems; occlusion left no under-foot data, causing missteps |
| 4 | [DPL](https://arxiv.org/abs/2510.07152) | 2025-10 | TienKung Ultra | Depth, Orbbec 355L | Not stated | Height map reconstructed from depth (cross-attention), 1x1 m at 5 cm | Why depth: elevation maps need several sensors plus localisation (latency, drift). Problem: "limited field of view and self-occlusion from the robot's body" hide gap bottoms |
| 5 | [PolygMap](https://arxiv.org/abs/2510.12346) | 2025-10 | Unnamed biped | LiDAR + RGB-D + IMU fused | Not stated | Polygonal stair-plane map for a footstep planner (model-based, not RL) | None relevant |
| 6 | [Gallant](https://arxiv.org/abs/2511.14625) (CVPR 2026) | 2025-11 | Unitree G1 | LiDAR, 2x Hesai JT128 | Torso, front and rear (above waist) | Voxel grid 32x32x40 from LiDAR (OctoMap), 10 Hz | Dual LiDAR for wide multi-layer coverage; LiDAR sim ray-casts the robot's own links (self-hits). Problem: >100 ms LiDAR latency limits reaction on stepping stones |
| 7 | [Gait-Adaptive, Under-Base Terrain Reconstruction](https://arxiv.org/abs/2512.07464) | 2025-12 | LimX Oli (1.65 m) | Depth, RealSense D435i | Under the base (pelvis), pointing straight down; below the waist | Height map from a single depth frame (U-Net) | Reason: terrain under the base/feet is what matters for omnidirectional stepping; forward cameras' memory goes stale when slowing or turning. Problem: depth "strongly affected by self-occlusions from the legs", hence the U-Net |
| 8 | [Deep Whole-body Parkour](https://arxiv.org/abs/2601.07701) | 2026-01 | Unitree G1 | Depth, D435i | Head (factory); tilt not stated; above waist | Raw depth (one frame) | RealSense filtering too slow for 50 Hz on Jetson CPU, replaced by GPU inpainting; reflections, motion blur, stereo errors |
| 9 | [Hiking in the Wild](https://arxiv.org/abs/2601.07718) | 2026-01 | Unitree G1 | Depth, D435i (factory) | Head link, "without any hardware modification"; above waist | Raw depth history, 480x270 -> 36x18, 60 Hz | Reason: factory sensor, no modification. Problems: black (occlusion/disparity shadow) and white (matching failure) regions; single forward camera means no backward/lateral motion (main limitation) |
| 10 | [RPL](https://arxiv.org/abs/2602.03002) | 2026-02 | Unitree G1 | Depth, 2x ZED 2i | Torso shell, front and rear (above waist); tilt not stated | Raw depth from both cameras, transformer fusion | Reason: G1 has "mounting holes only at the front and rear of its torso shell", minimal modification, bidirectional walking. Problems: upper-body motion causes "dynamic self-occlusions" (simulated with robot meshes); narrow downward view limits stepping-stone lookahead. Critique of mapping: calibration, state estimation, delay |
| 11 | [Now You See That](https://arxiv.org/abs/2602.06382) | 2026-02 | Custom humanoid; G1 | Depth, Orbbec Gemini 336L; D435i on G1 | Not stated (extrinsics randomised +-5 cm, +-0.1 rad) | Raw depth 24x32 | Stereo holes in occluded/textureless regions; stair descent 13/15 because the foothold is occluded by the step edge |
| 12 | [Perceptive Humanoid Parkour (PHP)](https://arxiv.org/abs/2602.15827) | 2026-02 | Unitree G1 | Depth, onboard (model not stated) | Not stated (inferred: G1 factory head D435i) | Raw depth 58x87 | Sim camera calibrated by matching robot self-visibility sim vs real. Limitation: "short-range, narrow field-of-view camera" at running speed |
| 13 | [Omnidirectional Stairs, Sparse LiDAR Elevation Mapping](https://arxiv.org/abs/2603.07928) | 2026-03 | Unitree G1 | LiDAR, MID-360 (+DLIO) | Head (above waist) | Elevation map from LiDAR, 5-frame history, refined by an edge-guided U-Net | Reason: forward depth cameras leave lateral/rear blind zones and suffer lighting and motion blur. Problems: very sparse, occluded clouds; step-edge "melting"; solid-state LiDAR loses returns at large incidence (stair risers); interpolation noise on flat ground; added a protected zone under the base to keep memory of the blind area |
| 14 | [CReF](https://arxiv.org/abs/2603.29452) | 2026-03 | AgiBot X2 Ultra | Depth, D435i (not the stock Orbbec) | Forward-facing, 50 deg down; location not stated | Raw depth 64x48 | Reason: raw depth avoids a mapping pipeline. Problems: reflection holes, vegetation out-of-distribution. Does not use the X2's built-in LiDAR |
| 15 | [SSR](https://arxiv.org/abs/2605.30770) | 2026-05 | AgiBot X2 | Depth, D435i | "Mounted at the waist", forward, 50 deg down (App. D), FOV 87x58; inferred below the waist joints (not stated) | Raw depth 640x360 -> 36x36, 60 Hz; height map only for the critic | No mount reason. Sim renders robot link meshes for self-occlusion. Problems: fixed forward camera limits coverage; sunlight reflections; grass. Does not use the built-in LiDAR |
| 16 | [GuideWalk](https://arxiv.org/abs/2606.10449) | 2026-06 | Not named | LiDAR MID-360 (elevation map, Miki 2022) + stereo depth (Gemini 335Lg, Fast Foundation Stereo) | Search snippet says MID-360 on the torso; not confirmed in text read | Elevation map + depth image | Limitation: two modalities cost compute; future work derives maps from depth to drop the LiDAR |
| 17 | [Echo in the Steps](https://arxiv.org/abs/2609.28960) | 2026-09 | Unitree G1 | Depth, D435i | Not stated (inferred: factory head) | Raw depth history 18x32 | "Restricted field of view": footholds leave view during sharp turns |
| 18 | [ACP: Humanoid Mobility via Attention-based Contacts](https://arxiv.org/abs/2610.31608) | 2026-10 | Unitree G1 | LiDAR, MID-360 (+DLIO) | Head (above waist) | 9x7 height scan from a neural map (per-cell keep / neighbourhood mean / empty, with uncertainty) | Head mount gives "a panoramic view with sparse hits"; occlusion "considerably less severe" than on the platforms of the depth-based method it adapts. Handles sparsity, per-point and ray-divergence noise, overhangs; no FOV, arm-hit or lean problems reported |

## Quadrupeds (contrast)

| Paper | Date | Robot | Sensor | Mount | Policy consumes | Stated reason / problem |
|---|---|---|---|---|---|---|
| [Miki et al., Learning robust perceptive locomotion in the wild](https://arxiv.org/abs/2201.08117) (Sci. Robot.) | 2022-01 | ANYmal C | LiDAR or depth cameras | Body | Height samples from an elevation map, with a belief encoder | Core point: maps are unreliable (reflection, occlusion, drift), so the policy learns when to trust them |
| [Robot Parkour Learning](https://arxiv.org/abs/2309.05665) (CoRL 2023) | 2023-09 | Unitree A1, Go1 | Depth, D435 | Front of body (head) | Raw depth | Low-rate depth, latency |
| [Extreme Parkour](https://arxiv.org/abs/2309.14341) (ICRA 2024) | 2023-09 | Unitree A1 | Depth, D435 | Inside the head, front-facing | Raw depth 58x87 at about 10 Hz, GRU | Depth is "low-frequency, jittery, and prone to artifacts"; fixed 80 ms latency in training |
| [ANYmal Parkour](https://arxiv.org/abs/2306.14874) (Sci. Robot. 2024) | 2023-06 / 2024-03 | ANYmal D | 6 RealSense depth (2 front, 2 back, left, right) + Velodyne Puck | Body, all sides | Neural voxel reconstruction -> 2.5D elevation map | Heavy occlusion "by obstacles ... or the robot itself", limbs block the side cameras while climbing, blind spot beneath; camera point clouds up to 250 ms late (raw depth cut it to 25 ms); hallucinated stairs in blind spots |

## Summary

**Depth vs LiDAR (18 humanoid works).** Depth only: 11 (61%). LiDAR only: 5 (28%; PIM, BeamDojo, Gallant,
Omnidirectional Stairs, ACP). Both fused: 2 (PolygMap, GuideWalk). Of the 11 depth works, 9 feed raw depth to the
policy and 2 reconstruct a height map from depth (DPL, under-base Oli). Every LiDAR work converts to a height map or
voxel grid, all but Gallant (dual torso JT128) from a single MID-360, and 4 of 5 on the G1/H1 factory head unit.

**Most common depth mount on humanoids: the head, forward-facing, pitched down about 45-57 deg.** Stated
mounts: head 3 (H1 parkour, Deep Whole-body Parkour, Hiking), plus 2 more almost certainly the G1 factory head
D435i (PHP, Echo); torso shell front/rear 1 (RPL); waist 1 (SSR); under-base pointing down 1 (Oli); not stated 3.
The reasons given are practical, not optical: the G1's factory camera is in the head (Hiking: "without any hardware
modification"), and the G1 shell has mounting holes only front and rear of the torso (RPL). The only geometric
argument for a low mount is the under-base paper: the ground under the feet matters, and forward cameras' memory
goes stale. No paper compares head, chest and pelvis mounts, and none reports a problem from a leaning torso moving
the camera (head- and chest-mounted cameras sit above the waist joints, so torso pitch moves the view; the papers
cover it only through extrinsic randomisation of a few degrees). Problems that are reported: forward-only field of
view (Hiking, SSR, PHP, Echo, Omnidirectional Stairs), self-occlusion by legs (under-base, DPL) and by moving arms
(RPL, ANYmal limbs), stereo holes at step edges on descent (Now You See That), sunlight/reflections (SSR, CReF,
Deep Whole-body Parkour). SSR, RPL and Gallant ray-cast the robot's own link meshes in simulation for this reason.

**Body-mounted / vertically mounted LiDAR quality.** No paper found reports a vertically mounted LiDAR on a
humanoid. All successful humanoid LiDAR deployments use a 360 deg MID-360 (head) or JT128 (torso), always with
LiDAR-inertial odometry (FAST-LIO / DLIO) and a temporal map, never raw scans. Even then the papers report
sparse, occluded returns, odometry jitter and map drift (BeamDojo), step-edge melting and lost returns at grazing
incidence on stair risers for solid-state units (Omnidirectional Stairs), >100 ms latency (Gallant), and the need
for a learned map cleaner (Omnidirectional Stairs, ACP). The G1's MID-360 is mounted inverted in the head; that
works but needs driver changes, and developers report SLAM trouble when the head tilts (community reports, not a
paper). The X2's factory LiDAR is a chest RoboSense E1R (solid-state, 120x90 deg FOV, per AgiBot's
[sensor page](https://x2-aimdk.agibot.com/en/dev/about_agibot_X2/sensor_fov.html)). Neither X2 paper (SSR, CReF)
uses it; both add a D435i. This fits the project's finding that it is too noisy.

**What this means for an X2 with an added RealSense.** Both published X2 works use a D435i pitched 50 deg
down feeding raw depth. The only one that names a location (SSR) puts it at the waist. With the X2 waist pitch
weak (about +-18 deg, 24 N*m, often saturated in the project's runs), a mount on the pelvis, below the waist
joints, has three advantages. (a) The view does not move when the torso leans or the waist pitch saturates, so the
learned depth features see the same geometry the paper trained on. (b) It matches SSR, the reference recipe,
including its 50 deg tilt and its use of link meshes for self-occlusion. (c) It sees the near-foot ground the
under-base paper argues matters most. The cost is that the thighs and knees enter the lower image on stairs; SSR
handles this by rendering link meshes. A chest mount (above the waist, like RPL and the head mounts) gives more
lookahead and is the common choice. But its view pitches with the torso: up to about 18 deg on the X2, and a
backward lean pushes the near ground out of view. Arm swing can also occlude it (RPL).

Recommendation: prefer a pelvis/waist mount at about 50 deg down for the X2 (SSR's setup), with robot meshes in the
depth renderer. If the existing chest mount (`torso_link`, 59.5 deg down) stays, three things are needed. The sim
camera must ride `torso_link`, so waist pitch and roll move it as on the robot. The waist-pitch range must be
randomised in the depth randomisation. Arm meshes must be ray-cast. Rendering both mounts in simulation over the same
stair and step terrain would show which loses the near-foot ground under the observed lean. That costs only CPU.
None of the surveyed papers measured this, so this is the open question to settle on the project's own data.
