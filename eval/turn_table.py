"""Turn-rate table (eval monitor, 2026-11-12; asked by the coordinator for Hiking's turn in place): on flat ground, MuJoCo
CPU, nominal protocol, the robot starts standing and gets a constant command (0, 0, wz) for 8 s: zero forward speed, a yaw
rate only. The command reaches the adapter as the harness Command with vel = (0, 0, wz), hold = False, the target 20 m
straight ahead (so an adapter that builds its own command from the target would walk forward: run command-following
adapters with their pass-through mode, e.g. HIKING_CMD=standard).

    python eval/turn_table.py --adapter hiking:HikingHandover --out DIR [--rates 0.4 0.8 -0.4 -0.8] [--seeds 3]

Per rate: achieved yaw rate over the last half (mean over seeds, rad/s, same sign as commanded = turns the right way), share
of the commanded rate, how far the pelvis moved (m, in place = under 0.30 m), falls. Writes turn_table.json / .txt in --out.
"""
import argparse
import json
import os
import pathlib
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import harness as Hn  # noqa: E402
import ladder  # noqa: E402
import x2 as X  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", required=True)
ap.add_argument("--checkpoint", default=None)
ap.add_argument("--out", required=True)
ap.add_argument("--rates", type=float, nargs="*", default=[0.4, 0.8, -0.4, -0.8])
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--seconds", type=float, default=8.0)
a = ap.parse_args()
ad = Hn.load_adapter(a.adapter, a.checkpoint)
c = ladder.load_course("ladder:flat_walk")
c.waypoints = c.waypoints[:1]
names = X.joint_names_of(X.robot_dir() / "mjcf" / "x2_ultra.xml")
x = X.build(c, ad.plant(names))
tr = Hn.Trial(x, c, ad, x.model.body_mass.copy(), x.model.body_inertia.copy())
rows = []
for w in a.rates:
    rates, falls, drift, total = [], 0, [], []
    for s in range(a.seeds):
        tr.reset(2000 + s, Hn.NOMINAL)
        p0 = tr.d.qpos[:2].copy()
        yaws, fell = [], False
        n = int(a.seconds / (tr.sub * X.SIM_DT))
        for k in range(n):
            cmd = tr.command()
            p, yaw = tr.d.qpos[:2], X.yaw_of(tr.d.qpos[3:7])
            cmd.target_xy = p0 + 20.0 * np.array([1.0, 0.0])
            cmd.dist = float(np.linalg.norm(cmd.target_xy - p))
            cmd.heading_err = 0.0
            cmd.vel = np.array([0.0, 0.0, w])
            cmd.hold = cmd.station = False
            tr.d.ctrl[:] = ad.act(tr.view, cmd)
            for _ in range(tr.sub):
                mujoco.mj_step(tr.m, tr.d)
            yaws.append(X.yaw_of(tr.d.qpos[3:7]))
            if tr.d.qpos[2] < 0.35:
                fell = True
                break
        falls += fell
        if not fell:
            u = np.unwrap(np.array(yaws))
            h = len(u) // 2
            rates.append(float((u[-1] - u[h]) / (a.seconds - h * tr.sub * X.SIM_DT)))
            total.append(float(u[-1] - u[0]))
            drift.append(float(np.linalg.norm(tr.d.qpos[:2] - p0)))
    ach = float(np.mean(rates)) if rates else None
    row = dict(command_wz=w, achieved_wz=round(ach, 3) if rates else None, share=round(ach / w, 2) if rates else None,
               turned_rad=round(float(np.mean(total)), 2) if total else None,
               max_drift_m=round(max(drift), 3) if drift else None, in_place=bool(drift and max(drift) < 0.30), falls=falls, seeds=a.seeds)
    rows.append(row)
    print(row, flush=True)
out = pathlib.Path(a.out)
out.mkdir(parents=True, exist_ok=True)
meta = dict(adapter=ad.name, checkpoint=str(ad.checkpoint), plant=ad.plant_name,
            simulator="MuJoCo %s (CPU)%s" % (mujoco.__version__, " | model: pelvis-hip_roll contact EXCLUDED (corrected model)" if os.environ.get("X2_EXCLUDE_PAIRS") else ""),
            seconds=a.seconds, rows=rows)
json.dump(meta, open(out / "turn_table.json", "w"), indent=1)
with open(out / "turn_table.txt", "w") as f:
    f.write("%s | %s | flat ground, start standing, command (0, 0, wz) for %.0f s, achieved = mean yaw rate over the last half\n" % (ad.checkpoint, meta["simulator"], a.seconds))
    f.write("command wz  achieved wz  share  turned (rad, whole run)  max drift m  in place  falls\n")
    for r in rows:
        f.write("%10.2f  %11s  %5s  %23s  %11s  %8s  %d/%d\n" % (r["command_wz"], r["achieved_wz"], r["share"], r["turned_rad"], r["max_drift_m"], r["in_place"], r["falls"], r["seeds"]))
print(open(out / "turn_table.txt").read())
