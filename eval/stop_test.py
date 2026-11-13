"""Stop test (eval monitor, 2026-11-12; asked by the Hiking track via the coordinator): on flat ground, MuJoCo CPU, nominal
protocol, the robot starts standing, walks WALK s at a constant forward command V, then gets a zero command (a stand:
hold = True, vel = 0) for STOP s.

    python eval/stop_test.py --adapter hiking:HikingHandover --out DIR [--v 0.6] [--walk 4] [--stop 3] [--seeds 3]

Per seed: speed just before the zero command, distance to stop (pelvis travel from the zero command until its speed stays
under 0.05 m/s for 0.5 s), time to stop, drift over the last 2 s of the stop phase, falls. Command-following adapters
should run in pass-through mode (HIKING_CMD=standard). Writes stop_test.json / .txt in --out.
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
ap.add_argument("--v", type=float, default=0.6)
ap.add_argument("--walk", type=float, default=4.0)
ap.add_argument("--stop", type=float, default=3.0)
ap.add_argument("--seeds", type=int, default=3)
a = ap.parse_args()
ad = Hn.load_adapter(a.adapter, a.checkpoint)
c = ladder.load_course("ladder:flat_walk")
c.waypoints = c.waypoints[:1]
x = X.build(c, ad.plant(X.joint_names_of(X.robot_dir() / "mjcf" / "x2_ultra.xml")))
tr = Hn.Trial(x, c, ad, x.model.body_mass.copy(), x.model.body_inertia.copy())
dt = tr.sub * X.SIM_DT
rows = []
for s in range(a.seeds):
    tr.reset(2000 + s, Hn.NOMINAL)
    p0 = tr.d.qpos[:2].copy()
    ps, fell = [], False
    nw, ns = int(a.walk / dt), int(a.stop / dt)
    for k in range(nw + ns):
        cmd = tr.command()
        p, yaw = tr.d.qpos[:2].copy(), X.yaw_of(tr.d.qpos[3:7])
        walking = k < nw
        cmd.target_xy = p0 + np.array([20.0, 0.0]) if walking else p.copy()
        cmd.dist = float(np.linalg.norm(cmd.target_xy - p))
        cmd.heading_err = float(X.wrap(-yaw))
        cmd.vel = np.array([a.v, 0.0, 0.0]) if walking else np.zeros(3)
        cmd.hold = cmd.station = not walking
        if not walking:
            cmd.target_heading = 0.0
        tr.d.ctrl[:] = ad.act(tr.view, cmd)
        for _ in range(tr.sub):
            mujoco.mj_step(tr.m, tr.d)
        ps.append(tr.d.qpos[:2].copy())
        if tr.d.qpos[2] < 0.35:
            fell = True
            break
    ps = np.array(ps)
    row = dict(seed=2000 + s, fell=fell)
    if not fell:
        sp = np.linalg.norm(np.diff(ps, axis=0), axis=1) / dt
        w = int(0.5 / dt)
        row["speed_before_stop"] = round(float(np.linalg.norm(ps[nw - 1] - ps[nw - 1 - w]) / (w * dt)), 3)
        stop_k = None
        for k in range(nw, len(sp) - w + 1):
            if np.all(sp[k:k + w] < 0.05):
                stop_k = k
                break
        row["stopped"] = stop_k is not None
        row["time_to_stop_s"] = round((stop_k - nw + 1) * dt, 2) if stop_k is not None else None
        row["distance_to_stop_m"] = round(float(np.linalg.norm(ps[stop_k] - ps[nw - 1])), 3) if stop_k is not None else None
        row["travel_after_zero_m"] = round(float(np.linalg.norm(ps[-1] - ps[nw - 1])), 3)
        m2 = int(2.0 / dt)
        row["drift_last_2s_m"] = round(float(np.linalg.norm(ps[-1] - ps[-m2])), 3)
    rows.append(row)
    print(row, flush=True)
out = pathlib.Path(a.out)
out.mkdir(parents=True, exist_ok=True)
meta = dict(adapter=ad.name, checkpoint=str(ad.checkpoint), plant=ad.plant_name,
            simulator="MuJoCo %s (CPU)%s" % (mujoco.__version__, X.model_note()), v=a.v, walk_s=a.walk, stop_s=a.stop, rows=rows)
json.dump(meta, open(out / "stop_test.json", "w"), indent=1)
with open(out / "stop_test.txt", "w") as f:
    f.write("%s | %s | flat, %.1f s at %.2f m/s then a zero command (stand) for %.1f s\n" % (ad.checkpoint, meta["simulator"], a.walk, a.v, a.stop))
    f.write("seed  speed before  stopped  time to stop s  distance to stop m  travel after zero m  drift last 2 s m  fell\n")
    for r in rows:
        f.write("%d  %12s  %7s  %14s  %18s  %19s  %16s  %s\n" % (r["seed"], r.get("speed_before_stop"), r.get("stopped"), r.get("time_to_stop_s"),
                                                                 r.get("distance_to_stop_m"), r.get("travel_after_zero_m"), r.get("drift_last_2s_m"), r["fell"]))
print(open(out / "stop_test.txt").read())
