# SSR paper vs this repo

Taken from the paper as written: observations (72-D proprioception, 36 x 36 depth, 21 joints), the MoE actor, three critics, encoder and
estimator sizes (Table 8), hyperparameters (Table 9), reward terms and weights (Tables 6, 7), command ranges (Tables 4, 5), terrain ranges
(Table 3), domain randomization (Table 10), mirror rules (Table 12), equivariant layers (App. B).

## Different on purpose

| Paper | Here | Why |
| --- | --- | --- |
| Isaac Gym | MuJoCo Warp, 200 Hz physics | Your SuperMimic / ACP stack; Isaac Lab was not needed |
| Waist-mounted D435i, forward | Built-in Gemini 335 on the head, 40 deg down (URDF `rgbd_head_front`) | What the robot has; camera pose is a constant in `ssr/x2.py` |
| Private mocap + AMASS | BONES-SEED walking + your Pico terrain captures | Their data is not released |
| Omnidirectional stage after the last level | 10 % of robots always on the flat tile with omni commands | Same exposure, simpler |
| Contact-force termination on links | Trunk, thighs, head end by terrain clearance; elbows by force | Large spheres or meshes overflow mujoco_warp's heightfield contacts |

## Added beyond the paper, for the apparatus

| Addition | Why | Where |
| --- | --- | --- |
| Side obstacles on the training tiles (lane walls, rails on posts, door frames). Seen in depth and the critic's map; a limb touching one ends the episode | The paper has none in training; the route has walls, rails and a door | `ssr/terrain.py` `_side_obstacles`, `env._touches_obstacle` |
| Rolling terrain map around the robot (64 x 64 at 5 cm, moved by odometry with bias and noise) as a second encoder input | The head camera cannot see the ground under the feet | `ssr/mapcache.py`, `--map_memory` |
| Down-looking chest camera, 75 deg | Same reason. Sees from 4 cm behind the feet to 1.03 m ahead; legs cover 18 % of the image, so self-occlusion is rendered | `ssr/x2.py` `CAMERAS`, `--camera chest` |

| Terrain categories replaced by the route's: ramp, stairs, beam (4x4 / 2x2 crossings), platform, flat lane with doors and rails. One discriminator and one motion set each | The paper's gap and discrete terrains are not on the route | `ssr/terrain.py`, `scripts/build_amp.py` |
| Each gate expert tied to one category by a soft cross-entropy term (weight 0.02), training only | The paper does not say how experts are assigned | `--gate_prior` in `ssr/train.py` |

| Half of the walking episodes start from captured walking frames (reference state initialization, standard for AMP), and a stepping bonus that fades to zero by iteration 2000 | With the paper's rewards alone the robot settled on standing still: 50 of 50 robots upright for 150 s, moved under 3 cm, both feet down (iteration 460) | `env._apply_rsi`, `air_bonus` |

| Early-training schedule, fading to the paper's values by iteration 3000: linear velocity reward x3, joint-deviation penalty x0.25, stepping bonus, low-friction floors and added mass eased in | Still standing at iteration 620 and after, with the additions above | `env.boot` |

| Walking-clip warm-up: episodes started from a captured clip get a joint-tracking reward along that clip, weight 2, fading to zero by iteration 3000 | The steps above still left the robot standing at iteration 900 to 1290 | `--imit_w`, `env._imitation` |

| Training only: an episode ends if a robot told to walk at 0.3 m/s or more covers under 30 % of the commanded distance in 2 s | Robots stood still through iteration 1350 even with the warm-up | `env._done` |

| Actor initialized by supervised fit to the SuperMimic tracker (`chains_ft320k_v1`) on flat walking clips; PPO continues from there | Rewards alone never produced a gait from a standing start. The fitted actor steps (feet off the ground 57 % of the time) but falls after 1.4 s until PPO stabilizes it | `scripts/bc.py` |

## Paper-silent choices

| Item | Value |
| --- | --- |
| Orientation reward | squared norm of g_xy; the table can be read either way, but the plain norm with sigma 0.01 gives about 0 at any real tilt |
| Proprio history h | 5 |
| Support tolerance eps_h | 3 cm |
| Guidance start level | above 5 of 20 |
| Rewards scaled by dt | no |
| Base height map | 17 x 10 at 10 cm |
| Imagined-landing samples | 3 x 3 stencil at -sigma, 0, +sigma |
| PPO clip, action clip, sensor noise | 0.2, +-8, usual legged values |
| Tile layouts | `ssr/terrain.py` |

## Open items for the real robot

- Depth topic name and encoding: confirm with `ros2 topic list` (`--depth_topic`).
- Head pitch is not actuated on current firmware; the camera angle assumes the head sits at 0. Check against the head IMU (about 4 deg at stand, per the camera calibration notes).
- Self-occlusion uses limb spheres. Depth noise and latency are not modelled in training; add them before trusting depth on hardware.
