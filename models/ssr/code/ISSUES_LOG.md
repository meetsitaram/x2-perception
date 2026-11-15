# Issues log

Symptom -> cause -> fix -> verified by.

| When | Symptom | Cause | Fix | Verified by |
| --- | --- | --- | --- | --- |
| 2026-11-08 | Heightfield collision overflow warnings, then an index crash in contact reading | Collision meshes and large spheres touch far too many 2.5 cm cells | Only spheres up to 5 cm touch terrain; trunk, thighs and head end by clearance | No overflow at 4096 worlds |
| 2026-11-08 | About 3 % of worlds diverged at first foot contact, qvel over 150 | 5 mm sole spheres tunnel into heightfield prisms in one 5 ms step | Sole spheres 2 cm radius, same lowest point | `scripts/debug_bad2.py`: 0 diverged in 30 steps on flat and on the tiles, with and without DR |
| 2026-11-08 | Out-of-memory, 56 GB allocation | Broadcast bug in the imagined-landing lookup | Explicit shapes | 4096 worlds run at 6 GB |
| 2026-11-08 | Episodes ended at once, then robots dropped 10 cm at spawn | Spawn height 0.74 m | 0.665 m (sole 2.5 cm above ground) | `scripts/debug_terms.py` |

| 2026-11-09 | PPO step size collapsed to the floor (lr 6e-5, stuck) | The KL check was a crude log-prob difference, so the adaptive schedule kept shrinking the step | True Gaussian KL per mini-batch, as rsl_rl does | training log |
| 2026-11-09 | PC hung; no training for about 8 hours | Two training jobs on the GPU: repeated nvlddmkm 13 / 153 driver resets from 17:37, then a PowerShell process scan hung | One GPU job at a time, stall detection, jobs only between slices (`scripts/supervise.py`) | Heartbeat advancing since the restart |
| 2026-11-09 | Reward terms disagreed with Table 6 (found on re-reading it, and may be why robots stood) | Air-time penalty had the wrong sign; impact applied at first touchdown only; orientation term near 0 with the plain norm | Table 6 air time and impact; squared tilt | Effect on gait not measured yet |

| 2026-11-09 | Terrain-map run went NaN at iteration 780 after a jump in value loss | A diverged world passed extreme proprioception into the encoder, which has no normalizer | Inputs clamped; divergence threshold 60 rad/s; updates skipped on non-finite loss or gradient; non-finite networks are never saved | Rolled back to clean checkpoint 750 |

| 2026-11-09 | All PPO updates skipped from iteration 880 and the terrain-map policy fell apart (policy loss infinite) | Action probability computed after clamping the action, so a drifting mean made log-ratios enormous | Probability of the raw sample; log-ratio clamped; bounded actor mean | Rolled back to checkpoint 875 |

| 2026-11-09 | Robot walked permanently bent over and leaning (first walking render) | Waist pitch and roll are not policy joints and were held at the plant kp of 12, too weak for the torso: they sat on their limits (20 deg, 30 deg), and the chest camera tilted with them | Held at kp 120, kd 6 in sim and on the robot | Re-measure after retraining |
