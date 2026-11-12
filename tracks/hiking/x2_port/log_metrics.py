"""Turns the trainer's printed per-iteration block into JSON lines: one object per iteration with every printed scalar.

  python3 log_metrics.py <train.log> <metrics.jsonl>

Incremental (byte offset kept in <metrics.jsonl>.offset), safe to run every few minutes while training writes the log.
Names are the trainer's own. Added per reward term: raw/<term> = <term>/timestep / (weight * 0.02 s), the unweighted per-step
mean (checked: is_alive gives 0.99). Legend: state/hiking.md in the perception-agents repo.
"""
import json
import os
import re
import sys

LOG, OUT = sys.argv[1], sys.argv[2]
OFF = OUT + ".offset"
DT = 0.02
WEIGHTS = {  # the paper's weights (parkour_env_cfg.py), unchanged for the X2
    "track_lin_vel_xy_exp": 2.0, "track_ang_vel_z_exp": 2.0, "heading_error": -1.0, "dont_wait": -0.5, "is_alive": 3.0,
    "stand_still": -0.3, "volume_points_penetration": -4.0, "feet_air_time": 0.5, "feet_slide": -0.4,
    "joint_deviation_hip": -0.5, "ang_vel_xy_l2": -0.05, "dof_torques_l2": -1.5e-7, "dof_acc_l2": -1.25e-7,
    "dof_vel_l2": -1e-4, "action_rate_l2": -0.005, "flat_orientation_l2": -3.0, "pelvis_orientation_l2": -3.0,
    "feet_flat_ori": -0.4, "feet_at_plane": -0.1, "feet_close_xy": 0.4, "energy": -5e-5, "freeze_upper_body": -0.004,
    "dof_pos_limits": -1.0, "dof_vel_limits": -1.0, "torque_limits": -0.01, "undesired_contacts": -1.0,
}
ANSI = re.compile(rb"\x1b\[[0-9;]*m")
IT = re.compile(r"Learning iteration (\d+)/(\d+)")
COMP = re.compile(r"Computation: (\d+) steps/s \(collection: ([\d.]+)s, learning ([\d.]+)s\)")
KV = re.compile(r"^\s*(?:Mean episode |Mean step |Mean )?([A-Za-z_][\w/ ]*?):?\s+(-?\d+(?:\.\d+)?(?:e-?\d+)?)s?\s*$")

start = int(open(OFF).read()) if os.path.exists(OFF) and os.path.exists(OUT) else 0
with open(LOG, "rb") as f:
    f.seek(start)
    data = f.read()
# only whole blocks: up to the start of the last "Learning iteration" header (that block may still be printing)
heads = [m.start() for m in re.finditer(rb"Learning iteration \d+/\d+", data)]
if len(heads) < 2:
    sys.exit(0)
end = data.rfind(b"\n", 0, heads[-1]) + 1
blocks = ANSI.sub(b"", data[heads[0]:end]).decode("utf-8", "replace").split("Learning iteration ")[1:]
rows = []
for b in blocks:
    lines = ("Learning iteration " + b).splitlines()
    m = IT.search(lines[0])
    if not m:
        continue
    row = {"iteration": int(m.group(1))}
    for line in lines[1:]:
        c = COMP.search(line)
        if c:
            row["steps_per_s"], row["collection_s"], row["learning_s"] = int(c.group(1)), float(c.group(2)), float(c.group(3))
            continue
        k = KV.match(line)
        if k:
            name = k.group(1).strip().replace(" ", "_")
            row[name] = float(k.group(2))
    for term, w in WEIGHTS.items():
        key = "Episode_Reward/rewards_%s/timestep" % term
        if key in row:
            row["raw/" + term] = row[key] / (w * DT)
    rows.append(row)
with open(OUT, "a") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
with open(OFF, "w") as f:
    f.write(str(start + end))
print("wrote %d iterations (%s..%s)" % (len(rows), rows[0]["iteration"] if rows else "-", rows[-1]["iteration"] if rows else "-"))
