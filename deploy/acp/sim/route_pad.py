"""Sim only: drive an eval course's waypoints through the deploy stack as a person on the pad would. Reads the bridge's
ground-truth pelvis pose (ZMQ robot_pose) and sends ordinary pad packets (hiking_deploy/pad.py's format) to the deploy node:
turn stick = bearing to the next waypoint (the ACP node turns it into the goal the eval gives), forward stick when the
bearing is under --walk-bearing, stop for the waypoint's hold (buttons). Reports each waypoint's time and falls.

    python3 route_pad.py [--course ours] [--shift 61.944,-9.222] [--reach 0.20] [--timeout 30] [--eval-goal]

--eval-goal (needs the node's --goal-from-pad): the packets also carry the goal exactly as eval/adapters/acp.py computes it
(the waypoint, 0.3 m look-ahead past non-station waypoints, d / max(|d|, 0.2) in the heading frame, the station heading while
holding), so the stack is driven as the eval drives ACP; without it, the sticks only (as a person would).

--shift: the course's place in the park (deploy/hiking/sim/scenes/park/scene.json, apparatus:eval). Start the stack with
BRIDGE_SPAWN at the course's spawn plus the shift (printed by --print-spawn).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
import time

import zmq

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "eval"))
sys.path.insert(0, os.path.join(REPO, "deploy", "hiking"))
from hiking_deploy.pad import WZ_MAX  # noqa: E402

HZ = 50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--course", default="ours")
    ap.add_argument("--shift", default="61.944,-9.222")
    ap.add_argument("--reach", type=float, default=0.20, help="arrival radius (the eval's RADIUS)")
    ap.add_argument("--eval-goal", action="store_true")
    ap.add_argument("--lookahead", type=float, default=0.3, help="the eval's ACP_LOOKAHEAD")
    ap.add_argument("--goal-r", type=float, default=0.2, help="the eval's ACP_GOAL_R")
    ap.add_argument("--timeout", type=float, default=30.0, help="per waypoint (s)")
    ap.add_argument("--walk-bearing", type=float, default=0.8, help="walk only when the waypoint is within this bearing (rad)")
    ap.add_argument("--bearing-max", type=float, default=1.0, help="the node's --bearing-max")
    ap.add_argument("--port", type=int, default=5690)
    ap.add_argument("--pose-port", type=int, default=5570)
    ap.add_argument("--print-spawn", action="store_true")
    a = ap.parse_args()
    from ladder import load_course
    course = load_course(a.course)
    sx, sy = (float(v) for v in a.shift.split(","))
    sp = course.spawn
    if a.print_spawn:
        print("%.3f,%.3f,%.1f" % (sp[0] + sx, sp[1] + sy, math.degrees(sp[2]) if len(sp) > 2 else 0.0))
        return
    wps = [dict(w, x=w["x"] + sx, y=w["y"] + sy) for w in course.waypoints]

    sub = zmq.Context.instance().socket(zmq.SUB)
    sub.setsockopt(zmq.CONFLATE, 1)
    sub.setsockopt(zmq.SUBSCRIBE, b"robot_pose")
    sub.connect("tcp://127.0.0.1:%d" % a.pose_port)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    ev, seq = {"y": 0, "b": 0, "reset": 0}, 0

    def send(vx, wz, deadman, hud, goal=None):
        nonlocal seq
        pk = dict(vx=vx, vy=0.0, wz=wz, deadman=deadman, estop=False, ev=ev, seq=seq, hud=hud)
        if goal is not None:
            pk["goal"] = [float(v) for v in goal]
        sock.sendto(json.dumps(pk).encode(), ("127.0.0.1", a.port))
        seq += 1

    def eval_goal(w, x, y, yaw, hold):
        """eval/adapters/acp.py ACPAdapter.act: target, look-ahead, bounded goal, heading still to turn."""
        tx, ty = w["x"], w["y"]
        dist = math.hypot(tx - x, ty - y)
        if a.lookahead > 0 and not w.get("hold_s", 0) and dist > 1e-6:
            tx, ty = tx + (tx - x) / dist * a.lookahead, ty + (ty - y) / dist * a.lookahead
        want = math.radians(w["heading_deg"]) if hold else math.atan2(ty - y, tx - x)
        dpsi = math.atan2(math.sin(want - yaw), math.cos(want - yaw)) if dist > 0.05 or hold else 0.0
        c, s_ = math.cos(yaw), math.sin(yaw)
        dx, dy = tx - x, ty - y
        b = (c * dx + s_ * dy, -s_ * dx + c * dy)
        n = max(math.hypot(*b), a.goal_r)
        return (b[0] / n, b[1] / n, dpsi)

    def pose():
        q = json.loads(sub.recv().split(b" ", 1)[1])["pelvis_qpos_wxyz"]
        w, qx, qy, qz = q[3:7]
        return q[0], q[1], q[2], math.atan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))

    for _ in range(HZ):                                  # pad alive, then Y: stand
        send(0.0, 0.0, False, "route: connecting")
        time.sleep(1 / HZ)
    ev["y"] += 1
    t_end = time.monotonic() + 4.0
    while time.monotonic() < t_end:
        send(0.0, 0.0, False, "route: standing")
        time.sleep(1 / HZ)
    t0, reached, fell = time.monotonic(), 0, False
    for i, w in enumerate(wps):
        tw = time.monotonic()
        while True:
            x, y, z, yaw = pose()
            if z < 0.4:
                fell = True
                break
            dx, dy = w["x"] - x, w["y"] - y
            dist = math.hypot(dx, dy)
            if dist < a.reach:
                break
            if time.monotonic() - tw > a.timeout:
                break
            e = math.atan2(math.sin(math.atan2(dy, dx) - yaw), math.cos(math.atan2(dy, dx) - yaw))
            u = max(-1.0, min(1.0, e / a.bearing_max))
            walk = abs(e) < a.walk_bearing
            send(0.5 if walk else 0.0, u * WZ_MAX, True, "route %d/%d %s: %.2f m" % (i + 1, len(wps), w["name"], dist),
                 eval_goal(w, x, y, yaw, False) if a.eval_goal else None)
            time.sleep(1 / HZ)
        ok = dist < a.reach and not fell
        print("%2d %-28s %s  %5.1f s  (%.2f m off)%s" % (i + 1, w["name"], "ok  " if ok else "MISS", time.monotonic() - tw, dist,
                                                         "  FELL (pelvis %.2f m)" % z if fell else ""), flush=True)
        if not ok:
            break
        reached += 1
        if w.get("hold_s", 0) > 0:
            t_end = time.monotonic() + w["hold_s"]
            while time.monotonic() < t_end:
                g = None
                if a.eval_goal:
                    x, y, z, yaw = pose()
                    g = eval_goal(w, x, y, yaw, True)
                send(0.0, 0.0, True, "route %d/%d %s: hold" % (i + 1, len(wps), w["name"]), g)
                time.sleep(1 / HZ)
    t_end = time.monotonic() + 2.0
    while time.monotonic() < t_end:
        send(0.0, 0.0, False, "route: done")
        time.sleep(1 / HZ)
    print("ROUTE %s: %d / %d waypoints in %.0f s%s" % (a.course, reached, len(wps), time.monotonic() - t0, ", fell" if fell else ""))


if __name__ == "__main__":
    main()
