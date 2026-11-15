"""Turn in place in MuJoCo with the command the perlin_rough_stand robots train with: [0, 0, wz] fed directly (no target),
run 4's filtered pairs added. python yaw_test.py <wz> <player args incl. --stand --no-video ...>"""
import sys, numpy as np
import sim2sim_mujoco_x2 as s
SIDES = ("left", "right")
s.FILTERED_PAIRS = list(s.FILTERED_PAIRS) + [("pelvis", f"{x}_hip_roll_link") for x in SIDES] + [
    (f"{x}_{h}_link", f"{x}_{w}_link") for x in SIDES for h in ("hip_pitch", "hip_roll", "hip_yaw") for w in ("wrist_yaw", "wrist_pitch", "wrist_roll")]
WZ = float(sys.argv.pop(1))
_orig = s.run_closed_loop
def patched(sim, policy, seconds, target=None, vmax=0.74, cmd_fn=None, verbose=True):
    rec = _orig(sim, policy, seconds, cmd_fn=lambda k, sim_: np.array([0.0, 0.0, WZ]), verbose=verbose)
    q = np.asarray(rec["root_quat"]); p = np.asarray(rec["root_pos"])
    yaw = np.unwrap(np.arctan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2)))
    t = np.asarray(rec["t"]); h = t > 1.0
    print("YAWTEST wz %.2f: turned %.2f rad in %.1f s (mean rate after 1 s %.2f rad/s), drift %.2f m, min pelvis z %.2f" % (
        WZ, yaw[-1] - yaw[0], t[-1], np.polyfit(t[h], yaw[h], 1)[0], np.linalg.norm(p[-1, :2] - p[0, :2]), p[:, 2].min()))
    return rec
s.run_closed_loop = patched
s.main()
