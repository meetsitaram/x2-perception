"""Step 3 of the baseline re-run: does the eval adapter reproduce the original code?

Reads obs_check.npz (tools/baseline_dump.py: the original rollout code on CPU, 50 steps from two start states on our
apparatus). For every recorded step the simulator state is put into the eval's own MuJoCo model, the adapter is called,
and its planner contact input, its tracker input and its action are compared with what the original code computed from
the same state. The adapter keeps its own history and latch state from step to step, as it does in a trial.

    python eval/tools/baseline_check.py [EXPORT_DIR]      (CPU, seconds)
PASS needs, on every step: every block of the tracker input within 0.02 of the original in units of that block's
normalisation scale, the action within 0.02, and no planner valid / in-contact flag different.
"""
import json
import math
import os
import pathlib
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
D = sys.argv[1] if len(sys.argv) > 1 else str(pathlib.Path(__file__).resolve().parents[1] / "out" / "baseline_export")
os.environ["ACP_BASELINE_DIR"] = D
import x2 as X  # noqa: E402
from adapters.acp_baseline import ACPBaseline  # noqa: E402
from adapters.base import Command, View  # noqa: E402
from apparatus import Apparatus  # noqa: E402

z = np.load(os.path.join(D, "obs_check.npz"), allow_pickle=True)
WPS = json.loads(str(z["waypoints"]))
app = Apparatus("ours")
ad = ACPBaseline()
x = X.build(app, ad.plant(X.joint_names_of(X.robot_dir() / "mjcf" / "x2_ultra.xml")))
m = x.model
d = mujoco.MjData(m)
view = View(x, d, app)
BLOCKS = (("goal block", 0, 680, None), ("proprioception", 680, 1670, "prop"), ("scan", 1670, 1733, "scan"), ("contacts", 1733, 1787, "contact"), ("goal command", 1787, 1817, "ref"))
ok_all = True
print("default pose max diff %.2e, action scale max diff %.2e" % (np.abs(z["default_q"] - ad.q0).max(), np.abs(z["alpha"] - ad.scale).max()))
for r, name in enumerate([str(n) for n in z["names"]]):
    wps = WPS[name.split("#")[0]]
    wi, T = 0, z["qpos"].shape[0]
    worst = {b[0]: 0.0 for b in BLOCKS}
    worst_raw = {b[0]: 0.0 for b in BLOCKS}
    worst_a, flips, moved = 0.0, 0, 0.0
    for k in range(T):
        d.qpos[:], d.qvel[:] = z["qpos"][k, r], z["qvel"][k, r]
        mujoco.mj_forward(m, d)
        # the recorded steps all aim at the clip's first waypoint (the dump does not run the original's waypoint switch)
        w = wps[wi]
        tgt = np.array(w[:2])
        dist = float(np.linalg.norm(tgt - d.qpos[:2]))
        cmd = Command(tgt, float(w[2]), (w[4] if len(w) > 3 else None), False, np.zeros(3), dist, 0.0, wi, len(wps), np.zeros((0, 2)), len(w) > 3)
        if k == 0:
            ad.reset(view, None)
        ad.act(view, cmd)
        xo, xa = z["x"][k, r], ad.debug["x"]
        for nm, a, b, key in BLOCKS:
            diff = np.abs(xo[a:b] - xa[a:b])
            worst_raw[nm] = max(worst_raw[nm], float(diff.max()))
            worst[nm] = max(worst[nm], float((diff / ad.norm[key][1]).max()) if key else float(diff.max()))
        # which contact numbers differ (the planner's discrete choices: valid, in contact)
        co, ca = z["contacts"][k, r].reshape(9, 6), ad.debug["contacts"].reshape(9, 6)
        flips += int((np.abs(co[:, 4:] - ca[:, 4:]) > 0.5).sum())
        worst_a = max(worst_a, float(np.abs(z["action"][k, r] - ad.debug["action"]).max()))
        moved = float(np.linalg.norm(z["qpos"][k, r, :2] - z["qpos"][0, r, :2]))
    good = max(worst.values()) < 2e-2 and worst_a < 0.02 and flips == 0
    ok_all &= good
    print("%s clip '%s': %d steps, the robot moved %.2f m" % ("PASS" if good else "FAIL", name, T, moved))
    for nm, _, _, _ in BLOCKS:
        print("     %-15s largest difference %.2e raw, %.2e in normalised units" % (nm, worst_raw[nm], worst[nm]))
    print("     %-15s largest difference %.2e (actions are of order 1; joint target = default + scale * action)" % ("action", worst_a))
    print("     planner valid / in-contact flags that differ: %d of %d" % (flips, T * 18))
print("ALL PASS" if ok_all else "CHECK FAILED")
sys.exit(0 if ok_all else 1)
