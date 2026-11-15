"""Run-4 wrapper of sim2sim_mujoco_x2.py: adds run 4's 20 extra filtered link pairs (config/x2/fixed.py) to the MuJoCo model,
then runs the normal player with the same command-line options. MuJoCo uses explicit PD; run 4 trained with Isaac's implicit PD.
  python run4_player.py --run assets/run4 --export assets/eval_v4_250 --name x2v4_ckpt250_mujoco_flat --terrain flat"""
import sys
import sim2sim_mujoco_x2 as s

SIDES = ("left", "right")
s.FILTERED_PAIRS = list(s.FILTERED_PAIRS) + [("pelvis", f"{x}_hip_roll_link") for x in SIDES] + [
    (f"{x}_{h}_link", f"{x}_{w}_link") for x in SIDES for h in ("hip_pitch", "hip_roll", "hip_yaw") for w in ("wrist_yaw", "wrist_pitch", "wrist_roll")
]
assert len(s.FILTERED_PAIRS) == 25
s.main()
