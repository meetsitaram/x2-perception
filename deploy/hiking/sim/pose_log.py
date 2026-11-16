"""Sim only: log the bridge's ground-truth pelvis pose (ZMQ robot_pose) at 20 Hz to CSV, and summarise a run.

    python3 pose_log.py --out pose.csv                 # until Ctrl-C / SIGTERM
    python3 pose_log.py --summary pose.csv deploy.log  # per state segment: time, distance, speed, yaw rate, min pelvis z
"""
from __future__ import annotations

import argparse
import json
import math
import re
import signal
import sys
import time


def yaw(qw, qx, qy, qz):
    return math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def log(out, port):
    import zmq
    s = zmq.Context.instance().socket(zmq.SUB)
    s.setsockopt(zmq.CONFLATE, 1)
    s.setsockopt(zmq.SUBSCRIBE, b"robot_pose")
    s.connect("tcp://127.0.0.1:%d" % port)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    with open(out, "w", buffering=1) as f:
        f.write("wall,sim_time,x,y,z,yaw\n")
        while True:
            if s.poll(1000):
                p = json.loads(s.recv().split(b" ", 1)[1])
                x, y, z, qw, qx, qy, qz = p["pelvis_qpos_wxyz"]
                f.write("%.3f,%.3f,%.4f,%.4f,%.4f,%.4f\n" % (time.time(), p["sim_time"], x, y, z, yaw(qw, qx, qy, qz)))
            time.sleep(0.05)


def summary(csv, deploy_log):
    rows = [list(map(float, ln.split(","))) for ln in open(csv).read().split("\n")[1:] if ln]
    ev = []   # (wall time, state, cmd)
    for ln in open(deploy_log):
        m = re.search(r"\[(\d+\.\d+)\] \[x2_hiking_deploy\]: (\w+) cmd ([-+\d. ]+?)  step", ln)
        if m:
            ev.append((float(m.group(1)), m.group(2), m.group(3).split()))
    segs, cur = [], None
    for t, st, cmd in ev:
        key = (st, tuple(cmd))
        if cur is None or cur[0] != key:
            cur = [key, t, t]
            segs.append(cur)
        cur[2] = t + 1.0
    print("%-6s %-20s %6s %7s %7s %8s %6s" % ("state", "cmd vx vy wz", "s", "dist m", "v m/s", "wz r/s", "zmin"))
    for (st, cmd), t0, t1 in segs:
        r = [x for x in rows if t0 <= x[0] < t1]
        if len(r) < 2:
            continue
        dist = math.hypot(r[-1][2] - r[0][2], r[-1][3] - r[0][3])
        dyaw = sum((math.remainder(b[5] - a[5], 2 * math.pi)) for a, b in zip(r, r[1:]))
        dt = r[-1][1] - r[0][1]
        print("%-6s %-20s %6.1f %7.2f %7.2f %8.2f %6.3f" % (st, " ".join(cmd), dt, dist, dist / max(dt, 1e-6), dyaw / max(dt, 1e-6),
                                                          min(x[4] for x in r)))
    print("final pelvis z %.3f, xy (%.2f, %.2f)" % (rows[-1][4], rows[-1][2], rows[-1][3]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--port", type=int, default=5570)
    ap.add_argument("--summary", nargs=2)
    a = ap.parse_args()
    summary(*a.summary) if a.summary else log(a.out, a.port)
