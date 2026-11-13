"""Per-command-speed table (eval monitor, 2026-11-12; asked by the coordinator for every track): on flat ground, MuJoCo
CPU, nominal protocol, the robot starts standing and gets a constant forward command for 8 s.

    python eval/speed_table.py --adapter ssr:SSRHandover --out DIR [--speeds 0 0.2 0.35 0.5 0.6 0.8] [--seeds 3]

Per speed: achieved forward speed over the last 4 s (mean over seeds), falls, and for 0 m/s whether the robot holds a stand
(no fall, pelvis moved under 0.10 m over the 8 s; a 1 s stand is the station rule). The command reaches the adapter as the
harness Command: vel = (v, 0, 0), target 20 m straight ahead (so a goal-driven policy such as ACP, which takes no speed,
walks at its own pace: its rows show that pace, not command following), hold = True only for 0 m/s.
Writes speed_table.json and speed_table.txt in --out.
"""
import argparse
import os
import json
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
ap.add_argument("--speeds", type=float, nargs="*", default=[0.0, 0.2, 0.35, 0.5, 0.6, 0.8])
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
for v in a.speeds:
    vx, falls, drift = [], 0, []
    for s in range(a.seeds):
        tr.reset(2000 + s, Hn.NOMINAL)
        p0 = tr.d.qpos[:2].copy()
        xs, fell = [], False
        n = int(a.seconds / (tr.sub * X.SIM_DT))
        for k in range(n):
            cmd = tr.command()
            p, yaw = tr.d.qpos[:2], X.yaw_of(tr.d.qpos[3:7])
            cmd.target_xy = p0 + 20.0 * np.array([1.0, 0.0])
            cmd.dist = float(np.linalg.norm(cmd.target_xy - p))
            cmd.heading_err = float(X.wrap(-yaw))
            cmd.vel = np.array([v, 0.0, 0.0])
            cmd.hold = cmd.station = v == 0.0
            if v == 0.0:
                cmd.target_xy = p0.copy(); cmd.dist = float(np.linalg.norm(p0 - p)); cmd.target_heading = 0.0
            tr.d.ctrl[:] = ad.act(tr.view, cmd)
            for _ in range(tr.sub):
                mujoco.mj_step(tr.m, tr.d)
            xs.append(tr.d.qpos[:3].copy())
            if tr.d.qpos[2] < 0.35:
                fell = True
                break
        xs = np.array(xs)
        falls += fell
        if not fell:
            h = len(xs) // 2
            vx.append(float((xs[-1, 0] - xs[h, 0]) / (a.seconds - h * tr.sub * X.SIM_DT)))
            drift.append(float(np.linalg.norm(xs[-1, :2] - p0)))
    row = dict(command=v, achieved=round(float(np.mean(vx)), 3) if vx else None, share=round(float(np.mean(vx)) / v, 2) if vx and v > 0 else None,
               falls=falls, seeds=a.seeds)
    if v == 0.0:
        row["holds_stand"] = bool(falls == 0 and max(drift) < 0.10)
        row["max_drift_m"] = round(max(drift), 3) if drift else None
    rows.append(row)
    print(row, flush=True)
out = pathlib.Path(a.out)
out.mkdir(parents=True, exist_ok=True)
meta = dict(adapter=ad.name, checkpoint=str(ad.checkpoint), plant=ad.plant_name, simulator="MuJoCo %s (CPU)%s" % (mujoco.__version__, " | model: pelvis-hip_roll contact EXCLUDED (second eval model)" if os.environ.get("X2_EXCLUDE_PAIRS") else ""), seconds=a.seconds, rows=rows)
json.dump(meta, open(out / "speed_table.json", "w"), indent=1)
with open(out / "speed_table.txt", "w") as f:
    f.write("%s | %s | flat ground, start standing, constant command %.0f s, achieved = mean forward speed over the last half\n" % (ad.checkpoint, meta["simulator"], a.seconds))
    f.write("command m/s  achieved m/s  share  falls  holds stand\n")
    for r in rows:
        f.write("%10.2f  %12s  %5s  %d/%d  %s\n" % (r["command"], r["achieved"], r["share"], r["falls"], r["seeds"], r.get("holds_stand", "")))
print(open(out / "speed_table.txt").read())
