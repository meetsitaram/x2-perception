# To do for the next ACP training run (sim-to-real)

Items found on the real X2 (2026-11-19, docs/FINDINGS.md X5-X13). Each is added only once the sim or the robot shows it matters.

1. **Observation noise on the balance inputs, at SONIC's level** (user, 2026-11-19: "policy trained to expect an almost perfect
   attitude - add that to the to-do list for the next training run"). ACP trains with gravity noise ±0.0125 (about ±0.7 deg) and
   angular-velocity noise ±0.05 rad/s (`models/acp/code/acp/env.py` `_proprio`). SONIC's X2 line trains with ±0.05 / ±0.2 rad/s (v1,
   `gwbc-x2/gear_sonic/config/manager_env/observations/policy/local_dir_hist.yaml`) and ±0.10 / ±0.4 rad/s (v2,
   `local_dir_hist_v2.yaml`, "doubled for sim2real"): 4-8x wider. Joint position ±0.01 (SONIC ±0.01-0.02), joint velocity ±1.5 rad/s
   (SONIC ±0.5-1.0). Proposal: gravity ±0.05-0.10 and angular velocity ±0.2-0.4 rad/s, or a per-episode attitude bias of a few
   degrees on top, so a residual IMU or mounting error cannot tip the policy.
2. **Arm gains** (user, 2026-11-19: shoulders feel weak; FINDINGS X15). ACP trains with the vendor MC's arm gains (shoulder pitch / roll /
   yaw kp 30 / 20 / 20, kd 1.0); SONIC uses the armature-rule gains (shoulder pitch and roll kp 34.9 kd 2.22). Fine for walking and
   the apparatus; revisit if the arms are needed (contacts with hands, carrying).
