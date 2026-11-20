# Why ACP could not stand on the real X2: the IMU frame bug (2026-11-19)

Short version: **our deploy fed the policy a pelvis attitude that was off by the waist angle.** The robot's
`/aima/hal/imu/torso/state` behaves as if it sits on the pelvis side of the waist, but our deploy treated it as a `torso_link` IMU
and rotated it back through the waist joints. Every degree the waist bent became a degree of false pelvis tilt. The policy kept
correcting a lean that was not there. Fixed with `--imu-on-pelvis`; the same checkpoint now stands as steadily as SONIC.

Details and numbers: docs/FINDINGS.md X5-X15.

## Symptom

First live ACP runs on the gantry (C++-free Python deploy, then with fixes): the policy stood up after the leaky assist but could
not stay still. Feet bumped into each other, it leaned sideways and forward and tipped over and over; the operator had to hold it
the whole time ("the gantry didn't take any of the robot's weight"). Hips felt weak. Walking a few steps and turning in place
worked, nowhere near stable. In the deploy sim the same checkpoint stood and walked calmly.

## How it was found (including the wrong turns)

| Lead | What we did | Verdict |
|---|---|---|
| Policy loop too slow (19 Hz) | student in its own thread on the GPU, tracker as ONNX on 1 thread, planner on the CPU | real problem, fixed (50 Hz), robot still unstable |
| Command delay (~80 ms measured) | sim with 40-80 ms delay added | wrong: the sim stays calm |
| Weak joints (fit said 0.65 x stiffness) | sim with weak-joint models, gain compensation | wrong: the fit paired the target with the previous step's position; corrected fit gives 0.77 (old writer) and 0.98 (relay) |
| Command stream (Python writer 208 Hz, 21 ms gaps vs MC 487 Hz) | C++ 500 Hz relay | worth keeping, but SONIC itself sends at 234 Hz with 300 ms gaps and stands: not the cause |
| Sensor noise | noise floor at rest | wrong: gyro 0.002 rad/s, joints 0.006 rad/s; the variation was real motion |
| Calibration / IMU mounting offset in hardware | - | dropped: SONIC uses the same sensors and stands |
| **SONIC baseline on the same robot** | 2 x 60 s bus recordings, same metrics | deploy pipeline now equal or better; the difference had to be in what our policy sees |
| **Posture comparison** | real vs SONIC vs sim, joint angles and torques | the policy itself straightened the knees on the real robot (targets 38 -> 3-7 deg) after starting from a pelvis it believed was tilted back 18.5 deg |
| **Pelvis attitude: legs vs IMU** | pelvis pitch from feet-flat leg kinematics vs from IMU + waist conversion | **root cause**: in sim they agree to 0.0 deg; on the robot they differ by -0.93 x the waist pitch (-10.5 deg at +9.9 deg waist) |

## Root cause

- Training (ACP and SONIC alike) uses the **pelvis** as the base: gravity and angular velocity in the pelvis frame.
- SONIC's deploy uses the raw IMU as that base.
- Our deploy (inherited from the Hiking port: "the robot publishes only the torso IMU; rotate it into the pelvis frame with the
  waist FK", marked unverified in deploy/hiking/STATUS.md) converted it through the waist joints.
- Our sim bridge had been switched to `--imu-from torso` to match that assumption, so the sim could never catch it. The bridge's
  own default publishes the pelvis IMU, which is what training used.
- ACP trains with only ±0.0125 gravity noise (about ±0.7 deg); SONIC with ±0.05-0.10 (3-6 deg). False tilts of 3-10 deg were far
  outside anything ACP had seen.

## Proof

- Reproduced in sim: bridge `--imu-from pelvis` (as the real robot) with the old conversion, the real run's pad commands. The
  policy saw a false roll of +13-15 deg, drove the knees lopsided (targets 88 / 5 deg) and fell after 4 s.
- Fixed in the same sim with `--imu-on-pelvis`: 219 s standing and walking, tilt p90 1.7 / max 3.9 deg.
- Verified on the robot (black box `run_20261120_075719`): quiet standing tilt p90 2.9 / max 3.2 deg, leg speed p99 0.03 rad/s;
  under 13 nudges tilt max 9.7 deg, recovered to under 4 deg in 0.2-2.9 s on 10 of them; whole run tilt p90 4.3 / max 9.7 deg vs
  SONIC under pushes 4.7 / 12.7 deg. Operator: "super stable now".

## What changed

- `deploy/hiking/hiking_deploy/core.py`: `IMU_ON_PELVIS`; `--imu-on-pelvis` on the deploy node; the launch script passes it.
- `deploy/hiking/sim/run_sim.sh`: `BRIDGE_IMU=pelvis` to emulate the real IMU placement.
- Kept from the investigation: C++ command relay, black box with measured joint effort, `replay_blackbox.py`, `bus_record.py`.

## Lessons

1. An assumption marked "unverified on hardware" must be checked on the first hardware run, not inherited.
2. A sim that is configured to match our own assumptions cannot test them; prefer the configuration that matches training
   (here the bridge's pelvis IMU default) and check the real sensor against an independent measurement (leg kinematics).
3. Compare against a known-good controller on the same robot (SONIC) early: it separated deploy problems from policy problems.
4. Before trusting a fitted number, validate the fit where the answer is known (the sim's known gains) and check the time
   alignment of every channel.
5. Next training run: balance-input noise at SONIC's level (docs/TODO_next_training.md).
