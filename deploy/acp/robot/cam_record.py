"""PC2: record the robot's cameras while the stack runs, for reviewing walks next to the black box (user, 2026-11-19: "capture the
whole obstacle course walk"). Each stream an mp4 plus a CSV of frame times (time.monotonic(), the black box's clock):
  stereo_left / orbbec  head cameras: the left stereo IMX900 (2064 x 1552, kept at 516 x 388) and the Orbbec colour (1280 x 720,
        kept at 640 x 360), 10 fps each, from their compressed ROS topics (already JPEG: no USB load added); see --head. Was:
        /aima/hal/sensor/rgbd_head_front/rgb_image/compressed (already JPEG; needs the vendor's orbbec_camera app running)
  chest the chest D435's colour camera, straight from V4L2 at a small size (it shares the D435's USB 2 link with the depth
        stream: 424 x 240 at 15 fps is about 25 Mbit/s; full colour would be about 150)
A new folder per start under --out. Subscribes / reads only; commands nothing. Ends cleanly on SIGINT / SIGTERM.

    python3 cam_record.py --out /home/run/x2-perception/log/video
"""
import argparse
import os
import signal
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage

STOP = threading.Event()


class Stream:
    def __init__(self, folder, name, fps):
        self.path, self.fps, self.wr = os.path.join(folder, name + ".mp4"), fps, None
        self.csv = open(os.path.join(folder, name + "_frames.csv"), "w", buffering=1)
        self.csv.write("frame,t_mono\n")
        self.n, self.last_flush, self.t_last = 0, time.monotonic(), 0.0

    def write(self, img, t):
        """Constant frame rate on the wall clock: frame k of the file is the newest image at t0 + k / fps (an image is repeated
        when frames come slower, skipped when faster). 2026-11-19: the head cameras arrived at ~6 fps into a file marked 10 fps
        and played 1.6x too fast; the CSV keeps each written frame's real capture time."""
        if self.wr is None:
            h, w = img.shape[:2]
            self.wr = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
            self.t0, self.last = t, img
        target = int(round((t - self.t0) * self.fps))       # this image's slot
        while self.n < target:                              # fill the slots before it with the previous image
            self.wr.write(self.last)
            self.csv.write("%d,%.4f\n" % (self.n, self.t_last))
            self.n += 1
        if self.n == target:
            self.wr.write(img)
            self.csv.write("%d,%.4f\n" % (self.n, t))
            self.n += 1
        self.last, self.t_last = img, t

    def close(self):
        if self.wr is not None:
            self.wr.release()
        self.csv.close()


def chest_loop(dev, size, fps, stream):
    W, H = (int(v) for v in size.split("x"))
    while not STOP.is_set():
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
        cap.set(cv2.CAP_PROP_FPS, fps)
        if not cap.isOpened():
            print("chest camera %s not available; retrying in 3 s" % dev, flush=True)
            time.sleep(3.0)
            continue
        while not STOP.is_set():
            ok, img = cap.read()
            if not ok:
                print("chest camera read failed; reopening", flush=True)
                break
            stream.write(img, time.monotonic())
        cap.release()
        time.sleep(1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    # head cameras as name=topic:reduce (JPEG decode at 1/reduce size, cheap). The robot's own topics need AgiBot's DDS profile
    # to be discovered (FASTRTPS_DEFAULT_PROFILES_FILE, set by the launch script). rgb_head_front_center is out of focus (user,
    # 2026-11-19: "head cam is blurry"); the left stereo IMX900 (2064 x 1552) and the Orbbec colour (1280 x 720) are sharp.
    # the Orbbec's view is blocked by our chest D435 (user, 2026-11-19): left stereo only by default; add
    # "orbbec=/aima/hal/sensor/rgbd_head_front/rgb_image/compressed:2" or the right stereo / rear camera with --head
    ap.add_argument("--head", default="stereo_left=/aima/hal/sensor/stereo_head_front_left/rgb_image/compressed:4")
    ap.add_argument("--head-fps", type=float, default=10.0, help="frames kept per second per head camera")
    ap.add_argument("--chest-dev", default="/dev/video15", help="the D435's colour node ('' = no chest recording)")
    ap.add_argument("--chest-size", default="424x240")
    ap.add_argument("--chest-fps", type=int, default=15)
    a = ap.parse_args()
    folder = os.path.join(a.out, time.strftime("run_%Y%m%d_%H%M%S"))
    os.makedirs(folder, exist_ok=True)
    print("recording to %s" % folder, flush=True)
    for s in (signal.SIGINT, signal.SIGTERM):
        signal.signal(s, lambda *_: STOP.set())
    heads = {}
    for spec in a.head.split(","):
        name, rest = spec.split("=", 1)
        topic, red = rest.rsplit(":", 1)
        heads[name] = (topic, {1: cv2.IMREAD_COLOR, 2: cv2.IMREAD_REDUCED_COLOR_2, 4: cv2.IMREAD_REDUCED_COLOR_4,
                               8: cv2.IMREAD_REDUCED_COLOR_8}[int(red)], Stream(folder, name, a.head_fps), [0.0])
    chest = Stream(folder, "chest", a.chest_fps) if a.chest_dev else None
    if chest:
        threading.Thread(target=chest_loop, args=(a.chest_dev, a.chest_size, a.chest_fps, chest), daemon=True).start()
    rclpy.init()
    n = Node("x2p_cam_record")
    def on_head(m, name):
        topic, flag, stream, last = heads[name]
        t = time.monotonic()
        img = cv2.imdecode(np.frombuffer(bytes(m.data), np.uint8), flag)
        if img is not None:
            stream.write(img, t)

    for name, (topic, _, _, _) in heads.items():
        n.create_subscription(CompressedImage, topic, lambda m, name=name: on_head(m, name), qos_profile_sensor_data)
    t_log = time.monotonic()
    try:
        run_loop(n, heads, chest, t_log)
    finally:                                            # always finalise the mp4s (2026-11-19: a broken log pipe skipped this,
        STOP.set()                                      # leaving files without an index; recovered by hand)
        for v in heads.values():
            v[2].close()
        if chest:
            chest.close()
        rclpy.try_shutdown()
        say("stopped: %s, chest %d frames in %s" % (", ".join("%s %d" % (k, v[2].n) for k, v in heads.items()), chest.n if chest else 0, folder))


def say(msg):
    try:
        print(msg, flush=True)
    except Exception:                                   # noqa: BLE001 -- a closed log pipe must never stop the recording
        pass


def run_loop(n, heads, chest, t_log):
    while not STOP.is_set() and rclpy.ok():
        rclpy.spin_once(n, timeout_sec=0.05)
        if time.monotonic() - t_log > 30:
            say(", ".join("%s %d" % (k, v[2].n) for k, v in heads.items()) + ", chest %d frames" % (chest.n if chest else 0))
            t_log = time.monotonic()



if __name__ == "__main__":
    main()
