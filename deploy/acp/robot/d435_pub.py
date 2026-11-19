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
    fifo = os.path.join(tempfile.mkdtemp(), "z16")
    os.mkfifo(fifo)
    cmd = ["v4l2-ctl", "-d", a.dev, "--set-fmt-video=width=%d,height=%d,pixelformat=Z16 " % (W, H),
           "--set-parm=%d" % a.fps, "--stream-mmap", "--stream-to=%s" % fifo]
    cap = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    node.get_logger().info("%s %dx%d @ %d fps -> %s %dx%d" % (a.dev, W, H, a.fps, a.topic, W_OUT, H_OUT))
    nbytes, n, t0 = W * H * 2, 0, time.monotonic()
    try:
        with open(fifo, "rb") as f:
            buf = bytearray()
            while rclpy.ok():
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
