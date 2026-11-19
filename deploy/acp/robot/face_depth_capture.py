#!/usr/bin/env python3
"""PC2: the chest depth (/x2_acp/chest_depth, 32FC1 metres) and, while the deploy node runs, the student's raw fine map (--fine,
from acp_deploy.node --fine-shm) as a colour mp4 for the robot's face screen.

The face UI on PC3 only plays video files (PlayVideo, looped), as gear-sonic/interact/face_camera_feed.sh uses for the head
camera: so the feed is short clips, a slideshow with a few seconds of delay. Colours as our renders: red near, blue far
(0.3-3.5 m), black = no return.

    python3 face_depth_capture.py --seconds 3 --out /tmp/x2p_face/live_0.mp4

Runs under gear-sonic/venv (cv2 + numpy) with the ROS paths exported; never pip install into that venv.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

NEAR, FAR = 0.3, 3.5
RES, X0, NX, NY = 0.025, -0.40, 52, 44                  # the student's fine map (models/acp_depth/fine_student.py)
# The map as the student outputs it: its own value per cell (height minus the root height), on a continuous colour scale with a
# fixed range, no floor picked and no bands (user, 2026-11-19: banded colours were my interpretation, not what the student sees).
H_LO, H_HI = -0.90, -0.30                               # m below the root: the floor sits near -0.7 standing



def colour(z):
    u = np.clip((z - NEAR) / (FAR - NEAR), 0, 1)
    v = 1.0 - u                                          # near warm, far cool (jet reversed), as eval/overlays._cmap
    rgb = np.stack([np.clip(1.5 - np.abs(4 * v - k), 0, 1) for k in (3, 2, 1)], -1)
    rgb[(z < NEAR) | (z > 6.0)] = 0
    return (rgb[..., ::-1] * 255).astype(np.uint8)       # BGR for OpenCV


class Grab(Node):
    def __init__(self, topic):
        super().__init__("x2p_face_depth")
        self.frames = []
        self.create_subscription(Image, topic, self.cb, qos_profile_sensor_data)

    def cb(self, m):
        self.frames.append(np.frombuffer(m.data, np.float32).reshape(m.height, m.width).copy())


def read_fine(path, max_age=1.0):
    """The deploy node's newest fine map (52 x 44, height minus root height), or None if absent or older than max_age s."""
    try:
        if time.time() - os.path.getmtime(path) > max_age:
            return None
        return np.load(path)
    except Exception:                                    # noqa: BLE001 -- mid-replace or not written yet: depth only
        return None


def fine_img(f, h):
    """Fine map -> BGR (turbo, H_LO..H_HI m relative to the root), forward up, left at left, with a colour bar; h pixels high."""
    u = np.clip((f - H_LO) / (H_HI - H_LO), 0, 1)
    img = cv2.applyColorMap((u * 255).astype(np.uint8), cv2.COLORMAP_TURBO)[::-1, ::-1]   # front at top, left at left
    k = h // NX
    img = cv2.resize(img, (NY * k, NX * k), interpolation=cv2.INTER_NEAREST)
    r = (NX - 1 - int(round(-X0 / RES))) * k + k // 2   # the robot (pelvis) at x = 0
    cv2.drawMarker(img, (img.shape[1] // 2, r), (255, 255, 255), cv2.MARKER_TRIANGLE_UP, 3 * k, 2)
    bar = cv2.applyColorMap(np.linspace(255, 0, img.shape[0]).astype(np.uint8)[:, None].repeat(14, 1), cv2.COLORMAP_TURBO)
    return np.concatenate((img, np.zeros((img.shape[0], 4, 3), np.uint8), bar), 1)


def _bar(img, x, y0, y1, colours, top, bottom, unit):
    """A vertical colour bar (colours: BGR rows, top first) with its end labels and unit."""
    img[y0:y1, x:x + 18] = cv2.resize(colours[:, None], (18, y1 - y0), interpolation=cv2.INTER_LINEAR)
    for t, y in ((top, y0 + 16), (bottom, y1 - 4)):
        cv2.putText(img, t, (x + 24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(img, unit, (x - 30, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


def compose(z, fine, W, H, show="depth"):
    """One full-screen panel: the chest depth (distance from the camera) or the student's raw fine map (height relative to the
    root), each with its own colour bar. The two use different quantities and scales; colours are for the screen only."""
    img = np.zeros((H, W, 3), np.uint8)
    if show == "fine" and fine is not None:
        fm = fine_img(fine, H - 60)[:, :-18]             # without fine_img's own bar (drawn labelled here)
        x0 = (W - fm.shape[1]) // 2 - 40
        img[45:45 + fm.shape[0], x0:x0 + fm.shape[1]] = fm
        cv2.putText(img, "student height map (forward up)", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        ramp = cv2.applyColorMap(np.linspace(255, 0, 256).astype(np.uint8)[:, None], cv2.COLORMAP_TURBO)[:, 0]
        _bar(img, x0 + fm.shape[1] + 30, 45, 45 + fm.shape[0], ramp, "%+.1f" % H_HI, "%+.1f" % H_LO, "m vs root")
        return img
    dw, dh = W - 120, (W - 120) * 9 // 16
    img[50:50 + dh, :dw] = cv2.resize(colour(z), (dw, dh), interpolation=cv2.INTER_NEAREST)
    cv2.putText(img, "chest depth", (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    ramp = colour(np.linspace(NEAR, FAR, 256))
    _bar(img, dw + 20, 50, 50 + dh, ramp, "%.1f" % NEAR, "%.1f" % FAR, "m from cam")
    return img


def write_still(path, z, W, H, frames, fine=None, show="depth"):
    wr = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (W, H))
    img = compose(z, fine, W, H, show)
    for _ in range(frames):
        wr.write(img)
    wr.release()


def pc3_password():
    """PC3's login for the copy: X2_PC3_PASSWORD, else ~/.x2_pc3_password on PC2 (kept out of the repo)."""
    pw = os.environ.get("X2_PC3_PASSWORD")
    if pw is None:
        with open(os.path.expanduser("~/.x2_pc3_password")) as f:
            pw = f.read().strip()
    return pw


def loop(a):
    """Newest frame -> 1 s still -> PC3 -> PlayVideo (looped until the next replaces it), as fast as that round trip goes."""
    W, H = (int(v) for v in a.size.split("x"))
    os.makedirs(a.out, exist_ok=True)
    rclpy.init()
    g = Grab(a.topic)
    threading.Thread(target=rclpy.spin, args=(g,), daemon=True).start()
    dest, slot, n, t0 = "/opt/x2_interact/media", 0, 0, time.monotonic()
    while True:
        if not g.frames:
            time.sleep(0.05)
            continue
        z = g.frames[-1]
        del g.frames[:-1]
        slot = 1 - slot
        local = os.path.join(a.out, "x2p_depth_%d.mp4" % slot)
        show = "fine" if a.toggle_s > 0 and int(time.monotonic() / a.toggle_s) % 2 else "depth"
        write_still(local, z, W, H, max(1, int(a.hold * 10)), read_fine(a.fine) if a.fine else None, show)
        r = subprocess.run(["sshpass", "-p", pc3_password(), "scp", "-o", "StrictHostKeyChecking=no", "-q", local,
                            "run@%s:%s/" % (a.pc3, dest)], capture_output=True)
        if r.returncode:
            time.sleep(1.0)
            continue
        body = json.dumps(dict(video_path="%s/x2p_depth_%d.mp4" % (dest, slot), mode=2, priority=100)).encode()
        try:
            urllib.request.urlopen(urllib.request.Request("http://%s:18080/PlayVideo" % a.pc3, body,
                                                          {"Content-Type": "application/json"}), timeout=3).read()
        except Exception as e:                           # noqa: BLE001 -- the screen is best-effort
            print("PlayVideo failed: %s" % e, flush=True)
        n += 1
        if time.monotonic() - t0 > 10.0:
            print("%.1f updates/s" % (n / (time.monotonic() - t0)), flush=True)
            n, t0 = 0, time.monotonic()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="/x2_acp/chest_depth")
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="clip length; 0 (default) = the newest frame only, as a short still (the screen shows what the "
                         "camera sees now, not a replay seconds old)")
    ap.add_argument("--hold", type=float, default=1.0, help="still length (s) for --seconds 0; the next one replaces it")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--size", default="800x450", help="output size (the 80 x 45 depth scaled up, nearest)")
    ap.add_argument("--out", required=True, help="output mp4 (with --loop: a directory for the alternating stills)")
    ap.add_argument("--loop", action="store_true",
                    help="stay running: keep the newest frame and push it to the face screen as fast as copy + play allow "
                         "(no per-update process start or ROS discovery: the screen lags about the copy + play time)")
    ap.add_argument("--fine", default="/dev/shm/x2p_fine.npy",
                    help="the deploy node's fine map (acp_deploy.node --fine-shm): shown beside the depth while fresh; '' = off")
    ap.add_argument("--toggle-s", type=float, default=10.0,
                    help="alternate the screen between the depth and the student map every this many seconds (the map only "
                         "while the deploy node writes it); 0 = depth only")
    ap.add_argument("--pc3", default=None, help="PC3's address (default: X2_PC3_HOST, else ~/.x2_pc3_host on PC2)")
    a = ap.parse_args()
    if a.pc3 is None:
        a.pc3 = os.environ.get("X2_PC3_HOST") or open(os.path.expanduser("~/.x2_pc3_host")).read().strip()
    if a.loop:
        return loop(a)
    W, H = (int(v) for v in a.size.split("x"))
    rclpy.init()
    g = Grab(a.topic)
    if a.seconds <= 0:                                   # the newest frame: wait for one, then take the latest that came
        t_end = time.monotonic() + 2.0
        while not g.frames and time.monotonic() < t_end:
            rclpy.spin_once(g, timeout_sec=0.02)
        rclpy.spin_once(g, timeout_sec=0.0)
        g.frames = g.frames[-1:] * max(1, int(a.hold * a.fps))
        a.seconds = a.hold
    else:
        t_end = time.monotonic() + a.seconds
        while time.monotonic() < t_end:
            rclpy.spin_once(g, timeout_sec=0.05)
    rclpy.try_shutdown()
    if not g.frames:
        print("no depth frames on %s" % a.topic, file=sys.stderr)
        sys.exit(1)
    step = max(1, len(g.frames) // int(a.seconds * a.fps))
    wr = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (W, H))
    for z in g.frames[::step]:
        img = cv2.resize(colour(z), (W, H), interpolation=cv2.INTER_NEAREST)
        cv2.putText(img, "chest depth", (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        wr.write(img)
    wr.release()


if __name__ == "__main__":
    main()
