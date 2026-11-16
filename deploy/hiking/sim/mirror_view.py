"""Sim viewer in its own process: mirrors the bridge's state (as depth_pub does) and draws it at 30 fps, so drawing never slows
the physics (the bridge's own --viewer ran the park at 0.56x real time; headless it runs at 1.00x).

    python3 mirror_view.py --scene DIR/scene.xml [--fps 30]

Shows the deploy node's state and hint (/x2_hiking/status) top left and the gamepad (pad_hud.py) bottom left. It takes no commands: key callbacks in this viewer
arrive late and in bursts (they run on MuJoCo's render thread and wait for the GIL), so the pad is pad.py.
"""
from __future__ import annotations

import argparse
import math
import json
import os
import threading
import time

import mujoco
import mujoco.viewer
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from depth_pub import Mirror
import pad_hud

HUD_FLIP = os.environ.get("HUD_FLIP", "0") == "1"   # set_images takes the image top row first (checked 2026-11-15)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--pose-port", type=int, default=5570)
    a = ap.parse_args()
    rclpy.init()
    node = Node("x2_hiking_mirror_view")
    mir = Mirror(node, a.scene, a.pose_port)
    status = ["NO DEPLOY|deploy node not running|", 0.0]

    def on_status(m):
        status[0], status[1] = m.data, time.monotonic()
    node.create_subscription(String, "/x2_hiking/status", on_status, 10)
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    hud_t, cam_t = [0.0, 0.0], [time.monotonic(), 0.0]   # cam_t: last frame, last right-stick use
    with mujoco.viewer.launch_passive(mir.m, mir.d, show_left_ui=False, show_right_ui=False) as v:
        for g in (4, 5):        # terrain and walls (eval/x2.py groups); the viewer shows only groups 0-2 by default
            v.opt.geomgroup[g] = 1
        v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        v.cam.trackbodyid = mir.m.body("pelvis").id
        v.cam.distance, v.cam.elevation, v.cam.azimuth = 3.5, -15.0, 150.0
        while v.is_running():
            t = time.monotonic()
            with v.lock():
                if mir.update():
                    mujoco.mj_comPos(mir.m, mir.d)   # the tracking camera follows subtree_com, which kinematics alone leaves at 0
            st = status[0].split("|") if t - status[1] < 1.0 else ["NO DEPLOY", "deploy node not responding"]
            body = "  ".join(st[1:4])
            hud = st[4] if len(st) > 4 else ""
            try:
                pad = json.loads(hud) if hud.startswith("{") else None
            except ValueError:
                pad = None
            dt = min(t - cam_t[0], 0.1)
            rx, ry = (pad.get("rx", 0.0), pad.get("ry", 0.0)) if pad is not None else (0.0, 0.0)
            rx, ry = (rx if abs(rx) > 0.15 else 0.0), (ry if abs(ry) > 0.15 else 0.0)
            if rx or ry:                     # right stick: orbit the camera around the robot (still tracking it)
                cam_t[1] = t
                with v.lock():
                    v.cam.azimuth -= 120.0 * rx * dt
                    v.cam.elevation = float(np.clip(v.cam.elevation + 60.0 * ry * dt, -85.0, -2.0))
            elif t - cam_t[1] > 1.5 and mir.root is not None:   # idle: ease in behind the robot, as in games
                w, qx, qy, qz = mir.root[3:7]
                yaw = math.degrees(math.atan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))
                err = (yaw - v.cam.azimuth + 180.0) % 360.0 - 180.0
                with v.lock():
                    v.cam.azimuth += err * min(1.0, 1.5 * dt)
            cam_t[0] = t
            if t - hud_t[1] > 5.0:
                hud_t[1] = t
                print("hud: status age %.1f s, fields %d, pad %s" % (t - status[1], len(st), "parsed" if pad else repr(hud[:40])), flush=True)
            if t - hud_t[0] > 1 / 15:       # the pad picture, 15 Hz
                hud_t[0] = t
                img = pad_hud.draw(pad)
                v.set_images((mujoco.MjrRect(12, 12, img.shape[1], img.shape[0]), img[::-1].copy() if HUD_FLIP else img))
            if pad is not None:
                hud = ""
            v.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT, st[0], body),
                         (mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, "", hud)])
            v.sync()
            time.sleep(max(0.0, 1.0 / a.fps - (time.monotonic() - t)))
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
