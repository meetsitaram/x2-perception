"""Observation check of an ACP hand-over (tracker actor.npz + planner_mlp.onnx) against the track's own recordings.

    python eval/tools/acp_handover_check.py HANDOVER_DIR

mlp/obs_check_pair.npz: 50 steps of the pair in the track's simulator. From each recorded state (qpos, qvel) the adapter
recomputes the proprioception history; the planner ONNX is run on the recorded planner inputs and on the adapter's own
history; the tracker input is assembled and the action computed. Compared with the recorded planner outputs, tracker
input and action. The scans and the goal are taken from the recording (the track's terrain and goal are not in the file),
so the eval's own scan sampling and goal are NOT covered here; obs_check.npz (reference contacts) covers the tracker's
scan through its recorded terrain heights.
"""
import os
import pathlib
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
D = sys.argv[1]
os.environ["ACP_HANDOVER_DIR"] = D
import x2 as X  # noqa: E402
from adapters.acp import ACPHandover  # noqa: E402
from adapters.base import View  # noqa: E402

ad = ACPHandover()
x = X.build(None, ad.plant(X.joint_names_of(X.robot_dir() / "mjcf" / "x2_ultra.xml")))
m = x.model
d = mujoco.MjData(m)
view = View(x, d, None)
ok = True
_pair = next((f for f in ("mlp/obs_check_pair.npz", "pair_eval_40/obs_check_pair.npz") if os.path.exists(os.path.join(sys.argv[1], f))), "mlp/obs_check_pair.npz")
for fname, pair in ((_pair, True), ("obs_check.npz", False)):
    if not os.path.exists(os.path.join(D, fname)):
        print("%s: NOT THERE (pair recording missing): the planner is not checked" % fname)
        ok = False
        continue
    z = np.load(os.path.join(D, fname), allow_pickle=True)
    T = z["qpos"].shape[0]
    worst = dict(history=0.0, planner_targets=0.0, planner_occ=0.0, actor_input=0.0, action=0.0, scan=0.0)
    for k in range(T):
        d.qpos[:], d.qvel[:] = z["qpos"][k], z["qvel"][k]
        mujoco.mj_forward(m, d)
        if k == 0:
            ad.a_prev = np.zeros(ad.nj)
            rec = z["actor_input"][0, :495].reshape(5, 99)
            ad.a_prev = rec[-1, 68:99].astype(float)          # the recording may start mid-episode: take its own first history
            from adapters.common import History
            ad.hist = History(5)
            ad.hist.buf = rec.astype(float).copy()
        else:
            ad.a_prev = z["action"][k - 1].astype(float)
            ad.hist.push(ad._proprio(view))
        hist = ad.hist.flat()
        worst["history"] = max(worst["history"], float(np.abs(hist - z["actor_input"][k, :495]).max()))
        if pair:
            t, o = ad.contacts.run(z["planner_proprio_history"][k], z["planner_goal"][k], z["planner_scan_17x11"][k])
            worst["planner_targets"] = max(worst["planner_targets"], float(np.abs(t - z["planner_targets"][k]).max()))
            worst["planner_occ"] = max(worst["planner_occ"], float(np.abs(o - z["planner_occupancy"][k]).max()))
            t2, o2 = ad.contacts.run(hist, z["planner_goal"][k], z["planner_scan_17x11"][k])
            tgt = (t2 * (o2 > 0.5)[:, None]).ravel()
            scan, goal = z["actor_input"][k, 495:558], z["planner_goal"][k]
        else:
            p, R = d.qpos[:3], d.xmat[1].reshape(3, 3)
            scan = np.clip(z["terrain_under_scan_world_z"][k] - p[2], -2.0, 2.0)
            worst["scan"] = max(worst["scan"], float(np.abs(scan - z["scan"][k]).max()))
            tgt, goal = z["contact_targets"][k], z["goal"][k]
        obs = np.concatenate((hist, scan, tgt, goal))
        worst["actor_input"] = max(worst["actor_input"], float(np.abs(obs - z["actor_input"][k]).max()))
        a = np.clip(ad.actor(ad.norm(obs)), -20.0, 20.0)
        worst["action"] = max(worst["action"], float(np.abs(a - z["action"][k]).max()))
    good = worst["history"] < 1e-3 and worst["actor_input"] < 1e-3 and worst["action"] < 1e-2 and worst["planner_targets"] < 1e-4
    ok &= good
    print("%s %s: %d steps; largest differences: %s" % ("PASS" if good else "FAIL", fname, T, ", ".join("%s %.1e" % kv for kv in worst.items())))
print("ALL PASS" if ok else "CHECK FAILED")
sys.exit(0 if ok else 1)
