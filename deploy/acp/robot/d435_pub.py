"""PC2: the chest D435's depth as ACP's depth student expects it, on /x2_acp/chest_depth (sensor_msgs/Image 32FC1, 80 x 45
z-depth in metres, 0 = no return). librealsense fails on PC2's kernel, so the depth comes straight from V4L2 (v4l2-ctl,
Z16 in millimetres), the same way deploy/hiking/tools/d435_live.py streams it.

    python3 d435_pub.py [--dev /dev/video11] [--size 480x270] [--fps 30] [--topic /x2_acp/chest_depth]

Resampling: the value at each of the 80 x 45 pixel centres (the student was trained on rays through pixel centres, with
holes kept as holes; no filling). Exits if the stream stops, so the launch script's check sees it.
"""
from __future__ import annotations

import argparse
import os
import select
import subprocess
import sys
import tempfile
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

W_OUT, H_OUT = 80, 45


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="/dev/video11")
    ap.add_argument("--size", default="480x270")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--topic", default="/x2_acp/chest_depth")
    a = ap.parse_args()
    W, H = (int(v) for v in a.size.split("x"))
    iu = ((np.arange(W_OUT) + 0.5) * W / W_OUT).astype(int)
    iv = ((np.arange(H_OUT) + 0.5) * H / H_OUT).astype(int)
    rclpy.init()
    node = Node("x2_acp_d435")
    pub = node.create_publisher(Image, a.topic, qos_profile_sensor_data)
    tmp = tempfile.mkdtemp()
    # the camera may not be ready just after boot (2026-11-19: v4l2-ctl died at once and the open below waited forever, so the
    # session looked alive and published nothing): open the fifo without blocking, retry the capture, exit if it never streams
    cap = None
    for attempt in range(1, 11):
        fifo = os.path.join(tmp, "z16_%d" % attempt)      # a fresh fifo each try: a dead writer leaves EOF behind
        os.mkfifo(fifo)
        fd = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK)
        cmd = ["v4l2-ctl", "-d", a.dev, "--set-fmt-video=width=%d,height=%d,pixelformat=Z16 " % (W, H),
               "--set-parm=%d" % a.fps, "--stream-mmap", "--stream-to=%s" % fifo]
        cap = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        t_try = time.monotonic()
        while cap.poll() is None and time.monotonic() - t_try < 5.0 and not select.select([fd], [], [], 0.2)[0]:
            pass
        if cap.poll() is None and select.select([fd], [], [], 0)[0]:
            break
        err = (cap.stderr.read() or b"").decode(errors="replace").strip()[-200:] if cap.poll() is not None else "no frames in 5 s"
        node.get_logger().warning("capture attempt %d failed (%s); retrying in 2 s" % (attempt, err))
        if cap.poll() is None:
            cap.terminate()
        os.close(fd)
        time.sleep(2.0)
    else:
        node.get_logger().error("%s never streamed: exiting" % a.dev)
        rclpy.try_shutdown()
        sys.exit(1)
    os.set_blocking(fd, True)
    node.get_logger().info("%s %dx%d @ %d fps -> %s %dx%d (capture attempt %d)" % (a.dev, W, H, a.fps, a.topic, W_OUT, H_OUT, attempt))
    nbytes, n, t0 = W * H * 2, 0, time.monotonic()
    try:
        with os.fdopen(fd, "rb", buffering=0) as f:
            buf = bytearray()
            while rclpy.ok():
                r, _, _ = select.select([f], [], [], 1.0)
                if not r:
                    node.get_logger().error("no depth for 1 s (capture %s)" % ("exited" if cap.poll() is not None else "stalled"))
                    break
                chunk = f.read(nbytes - len(buf))
                if not chunk:
                    node.get_logger().error("depth stream ended")
                    break
                buf += chunk
                if len(buf) < nbytes:
                    continue
                z = np.frombuffer(bytes(buf), np.uint16).reshape(H, W)[iv][:, iu].astype(np.float32) * 1e-3
                buf = bytearray()
                img = Image(height=H_OUT, width=W_OUT, encoding="32FC1", is_bigendian=0, step=4 * W_OUT, data=z.tobytes())
                img.header.stamp = node.get_clock().now().to_msg()
                img.header.frame_id = "chest_depth"
                pub.publish(img)
                n += 1
                if time.monotonic() - t0 > 5.0:
                    node.get_logger().info("%.1f Hz, valid %.0f%%" % (n / (time.monotonic() - t0), 100 * float((z > 0).mean())))
                    n, t0 = 0, time.monotonic()
    finally:
        cap.terminate()
        rclpy.try_shutdown()
    sys.exit(1)


if __name__ == "__main__":
    main()
