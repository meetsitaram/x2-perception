"""Chest camera mount calibration: measure how far the real camera is from the trained mount and say how to adjust it.
For whoever mounts the camera (user, 2026-11-16: "a calibration script with visuals for someone to iterate and adjust camera
position when mounting it").

    python3 deploy/hiking/tools/mount_calib.py [--host run@<PC2_IP>]   (or export PC2_HOST=<PC2_IP>)          # window: adjust, Space, repeat
    python3 deploy/hiking/tools/mount_calib.py --once                              # one measurement, report + PNG
    python3 deploy/hiking/tools/mount_calib.py --cap deploy/hiking/logs/d435/cal_1116_0737   # re-analyse a saved capture

Before: the robot stands still on flat, plain floor (MC stand is fine), the camera on PC2, nothing in front of the feet within
~0.8 m. Each measurement reads the robot's joint angles + torso IMU and one depth frame from PC2 over ssh (nothing is installed
or started there) into logs/d435/mount_<time>/, then fits the camera's pitch and roll to the robot's own feet and lower legs (2-15 cm above the floor; hands and arms ignored): the
sim robot in the same pose seen by the trained camera vs the real depth (tools/calib_legs.py; IoU of the leg pixels).
Within +-1 deg pitch and roll the mount matches training (green).

Keys: Space = measure again, A = measure continuously, S = save the window, Q / Esc = quit.
Conventions: pitch + = the camera looks further down than trained (fix: tilt it up); roll + = turned clockwise as seen from
behind the camera looking where it looks (fix: turn it anticlockwise). The overlay is the ground truth if in doubt.
Lens values are nominal (87 x 58 deg) until the librealsense calibration is read; expect ~1 deg from that.
Memory: run as is (it caps itself at 6 GB; an uncapped full SVD once froze the desktop).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import resource
import subprocess
import sys
import threading
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
DEP = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(DEP))
sys.path.insert(0, str(DEP.parents[1] / "eval"))
os.environ.setdefault("PA_ROBOT_DIR", os.path.expanduser("~/Projects/gwbc-x2/gear_sonic/data/assets/robot_description"))
resource.setrlimit(resource.RLIMIT_AS, (6 << 30, 6 << 30))      # never again freeze the desktop

import mujoco  # noqa: E402

import calib_legs as L  # noqa: E402
import compare_mount as C  # noqa: E402
from hiking_deploy import core  # noqa: E402

TOL = 1.0                                      # deg: within this the mount matches training
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", "-i", os.path.expanduser("~/.ssh/id_ed25519")]
STATE_PY = r'''
import json, time, rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
from aimdk_msgs.msg import JointStateArray
rclpy.init(); n = rclpy.create_node("x2_mount_calib_peek"); out = {"q": {}, "groups": set()}
def js(g):
    def cb(m):
        for j in m.joints: out["q"][j.name] = j.position
        out["groups"].add(g)
    return cb
for g in ("leg", "waist", "arm", "head"):
    n.create_subscription(JointStateArray, "/aima/hal/joint/%s/state" % g, js(g), qos_profile_sensor_data)
def imu(m):
    o, w = m.orientation, m.angular_velocity; out["imu"] = dict(xyzw=[o.x, o.y, o.z, o.w], gyro=[w.x, w.y, w.z])
n.create_subscription(Imu, "/aima/hal/imu/torso/state", imu, qos_profile_sensor_data)
t0 = time.time()
while time.time() - t0 < 8 and not (len(out["groups"]) == 4 and "imu" in out):
    rclpy.spin_once(n, timeout_sec=0.2)
out["groups"] = sorted(out["groups"]); print(json.dumps(out))
'''
REMOTE = ("source /opt/ros/humble/setup.bash; "
          "export PYTHONPATH=/agibot/software/common/local/lib/python3.10/dist-packages:$PYTHONPATH "
          "LD_LIBRARY_PATH=/agibot/software/common/lib:$LD_LIBRARY_PATH "
          "FASTRTPS_DEFAULT_PROFILES_FILE=/agibot/software/entry/cfg/ros_dds_configuration.xml; "
          "python3 -c \"$(cat)\"")


def capture(host, dev, out: pathlib.Path):
    """Robot pose + one 640x360 depth frame (full 87 x 58 deg view) from PC2 -> out/. Returns (state, z)."""
    out.mkdir(parents=True, exist_ok=True)
    st = subprocess.run(SSH + [host, REMOTE], input=STATE_PY.encode(), capture_output=True, timeout=40)
    state = json.loads(st.stdout.decode().strip().splitlines()[-1])
    if len(state.get("groups", [])) < 4 or "imu" not in state:
        raise RuntimeError("no joint states / IMU from PC2 (robot on? HAL running?)")
    (out / "x2_state.json").write_text(json.dumps(state))
    cmd = ("pkill -x v4l2-ctl; F=$(mktemp /tmp/mount_calib.XXXX); v4l2-ctl -d %s --set-fmt-video=width=640,height=360,"
           "pixelformat='Z16 ' --stream-mmap --stream-skip=15 --stream-count=1 --stream-to=$F >/dev/null 2>&1; cat $F; rm -f $F" % dev)
    raw = subprocess.run(SSH + [host, cmd], capture_output=True, timeout=40).stdout
    if len(raw) != 640 * 360 * 2:
        raise RuntimeError("depth frame not read (%d bytes): camera on %s? (live viewer still running?)" % (len(raw), dev))
    (out / "depth640x360.raw").write_bytes(raw)
    return state, np.frombuffer(raw, np.uint16).reshape(360, 640).astype(float) / 1000.0


class Fitter:
    def __init__(self, model_dir):
        import x2 as X
        self.m = X.build(None).model             # the X2 on a flat floor (eval/x2.py; floor geom group 4)
        self.d = mujoco.MjData(self.m)
        self.card = core.Card.load(model_dir)

    def fit(self, state, z):
        m, d = self.m, self.d
        L.pose_robot(m, d, state)
        real, valid = L.real_robot_mask(z)
        if real.sum() < 60:
            raise RuntimeError("legs / feet not found in the depth (%d px): is the robot standing, camera looking at the feet?" % real.sum())

        def iou(dp, dr):
            sim = L.sim_robot_mask(m, d, self.card, dp, dr)[0]
            return (sim & real & valid).sum() / max(((sim | real) & valid).sum(), 1)
        _, p0, r0 = max((iou(p, r), p, r) for p in np.arange(-15, 25.01, 2.0) for r in np.arange(-8, 8.01, 4.0))
        _, p1, r1 = max((iou(p, r), p, r) for p in np.arange(p0 - 2, p0 + 2.01, 0.5) for r in np.arange(r0 - 4, r0 + 4.01, 1.0))
        score, dp, dr = max((iou(p, r), p, r) for p in np.arange(p1 - 0.5, p1 + 0.51, 0.25) for r in np.arange(r1 - 1, r1 + 1.01, 0.5))
        return dict(pitch=float(dp), roll=float(dr), iou=float(score), iou_trained=float(iou(0, 0)), real=real, valid=valid,
                    sim0=L.sim_robot_mask(m, d, self.card, 0, 0)[0], simb=L.sim_robot_mask(m, d, self.card, dp, dr)[0],
                    z=z, height=float(L.sim_robot_mask(m, d, self.card, 0, 0)[2][2]), knees=(state["q"]["left_knee_joint"],
                                                                                             state["q"]["right_knee_joint"]))


def advice(r):
    p, rl = r["pitch"], r["roll"]
    lines = []
    lines.append(("PITCH OK (%+.1f deg)" % p, (90, 220, 120)) if abs(p) <= TOL else
                 (("Tilt the camera UP by %.1f deg (it looks too far down)" if p > 0 else
                   "Tilt the camera DOWN by %.1f deg (it looks too far up)") % abs(p), (240, 190, 60)))
    lines.append(("ROLL OK (%+.1f deg)" % rl, (90, 220, 120)) if abs(rl) <= TOL else
                 (("Turn the camera %.1f deg %s (seen from behind the camera)" % (abs(rl), "ANTICLOCKWISE" if rl > 0 else "CLOCKWISE")),
                  (240, 190, 60)))
    return lines


def overlay(sim, real, valid, scale=4):
    img = np.zeros(sim.shape + (3,), np.uint8)
    img[real & valid] = (230, 120, 40)
    img[sim] = (60, 140, 255)
    img[sim & real] = (240, 240, 240)
    img[~valid] = (img[~valid] * 0.4).astype(np.uint8)
    return np.repeat(np.repeat(img, scale, 0), scale, 1)


def depth_rgb(z, far=2.0):
    v = np.clip(z / far, 0, 1)
    img = (np.stack([np.clip(1.6 * v, 0, 1), np.clip(1.2 * v ** 0.7, 0, 1), np.clip(1 - 1.5 * v, 0, 1)], -1) * 255).astype(np.uint8)
    img[~(z > 0)] = 0
    return img


def report(r, folder):
    print("legs/feet fit: pitch %+.2f deg, roll %+.2f deg (IoU %.2f; trained mount %.2f); camera %.2f m up; knees %.2f / %.2f"
          % (r["pitch"], r["roll"], r["iou"], r["iou_trained"], r["height"], *r["knees"]))
    for t, _ in advice(r):
        print("  ->", t)
    from PIL import Image
    W = 160 * 4
    canvas = np.zeros((90 * 4, W * 3 + 20, 3), np.uint8)
    canvas[:, :W] = np.asarray(Image.fromarray(depth_rgb(r["z"])).resize((W, 360)))
    canvas[:, W + 10:2 * W + 10] = overlay(r["sim0"], r["real"], r["valid"])
    canvas[:, 2 * W + 20:] = overlay(r["simb"], r["real"], r["valid"])
    Image.fromarray(canvas).save(folder / "mount_calib.png")
    print("  image:", folder / "mount_calib.png", "(depth | trained mount: orange real, blue expected | best fit)")


def gui(args, fitter):
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame
    pygame.init()
    W, H = 1340, 760
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("X2 chest camera mount calibration")
    big, mid, small = pygame.font.SysFont(None, 40), pygame.font.SysFont(None, 28), pygame.font.SysFont(None, 22)
    state = dict(busy=False, result=None, error=None, history=[], auto=False, folder=None, t=0.0)

    def measure():
        state.update(busy=True, error=None)
        try:
            folder = DEP / "logs" / "d435" / time.strftime("mount_%m%d_%H%M%S")
            st, z = capture(args.host, args.dev, folder)
            r = fitter.fit(st, z)
            state["history"].append((r["pitch"], r["roll"]))
            state.update(result=r, folder=folder, t=time.time())
        except Exception as e:      # shown in the window; the next Space retries
            state["error"] = str(e)[:160]
        state["busy"] = False

    def start():
        if not state["busy"]:
            threading.Thread(target=measure, daemon=True).start()

    def surf(arr):
        return pygame.surfarray.make_surface(arr.swapaxes(0, 1))

    start()
    clock = pygame.time.Clock()
    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key in (pygame.K_q, pygame.K_ESCAPE)):
                return
            if e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE:
                start()
            if e.type == pygame.KEYDOWN and e.key == pygame.K_a:
                state["auto"] = not state["auto"]
            if e.type == pygame.KEYDOWN and e.key == pygame.K_s and state["folder"]:
                pygame.image.save(screen, str(state["folder"] / "mount_calib_window.png"))
        if state["auto"] and not state["busy"]:
            start()
        screen.fill((22, 24, 28))
        r = state["result"]
        if r is not None:
            screen.blit(pygame.transform.scale(surf(depth_rgb(r["z"])), (640, 360)), (20, 60))
            screen.blit(surf(overlay(r["sim0"], r["real"], r["valid"])), (680, 60))
            screen.blit(small.render("real depth (near blue, far yellow)", True, (200, 200, 200)), (20, 425))
            screen.blit(small.render("feet: orange = real, blue = where the TRAINED mount expects them, white = both", True,
                                     (200, 200, 200)), (680, 425))
            y = 470
            for text, col in advice(r):
                screen.blit(big.render(text, True, col), (20, y))
                y += 46
            screen.blit(mid.render("fit: pitch %+.2f deg, roll %+.2f deg   overlap %.2f (trained mount %.2f)   camera %.2f m up"
                                   % (r["pitch"], r["roll"], r["iou"], r["iou_trained"], r["height"]), True, (210, 210, 215)), (20, y + 6))
            screen.blit(small.render("saved: %s   (%.0f s ago)" % (state["folder"].name, time.time() - state["t"]), True,
                                     (150, 150, 160)), (20, y + 40))
            # history of measurements: pitch (amber) and roll (cyan), the +-1 deg band in green
            hx, hy, hw, hh = 680, 470, 640, 200
            pygame.draw.rect(screen, (34, 37, 42), (hx, hy, hw, hh))
            mid_y, sc = hy + hh / 2, hh / 2 / 15.0
            pygame.draw.rect(screen, (40, 70, 50), (hx, mid_y - TOL * sc, hw, 2 * TOL * sc))
            pygame.draw.line(screen, (90, 90, 100), (hx, mid_y), (hx + hw, mid_y))
            hist = state["history"][-20:]
            for k, col in ((0, (240, 190, 60)), (1, (80, 200, 255))):
                pts = [(hx + 20 + i * (hw - 40) / max(len(hist) - 1, 1), mid_y - np.clip(h[k], -15, 15) * sc) for i, h in enumerate(hist)]
                if len(pts) > 1:
                    pygame.draw.lines(screen, col, False, pts, 2)
                for p in pts:
                    pygame.draw.circle(screen, col, (int(p[0]), int(p[1])), 4)
            screen.blit(small.render("history: pitch (amber), roll (cyan), deg; green = within +-%.0f" % TOL, True, (180, 180, 190)),
                        (hx, hy + hh + 6))
        msg = ("measuring ... (robot pose + depth from PC2, then fitting the legs)" if state["busy"] else
               "Space: measure again   A: continuous %s   S: save   Q: quit" % ("ON" if state["auto"] else "off"))
        screen.blit(mid.render(msg, True, (220, 224, 230)), (20, 18))
        if state["error"]:
            screen.blit(mid.render("! " + state["error"], True, (240, 90, 90)), (20, H - 40))
        pygame.display.flip()
        clock.tick(20)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--host", default=os.environ.get("PC2_HOST") and "run@" + os.environ["PC2_HOST"],
                    help="run@<PC2_IP>, or set PC2_HOST (PC2's address on the robot's Wi-Fi)")
    ap.add_argument("--dev", default="/dev/video11", help="the D435's Z16 depth node on PC2")
    ap.add_argument("--once", action="store_true", help="one measurement, print + save PNG, no window")
    ap.add_argument("--cap", help="analyse a saved capture folder (x2_state.json + depth640x360.raw) instead of PC2")
    ap.add_argument("--model", default=str(DEP / "models" / "x2-hiking-in-the-wild"))
    a = ap.parse_args()
    if not a.cap and not a.host:
        ap.error("no PC2 address: pass --host run@<PC2_IP> or export PC2_HOST=<PC2_IP>")
    fitter = Fitter(a.model)
    if a.cap or a.once:
        folder = pathlib.Path(a.cap) if a.cap else DEP / "logs" / "d435" / time.strftime("mount_%m%d_%H%M%S")
        if a.cap:
            st = json.loads((folder / "x2_state.json").read_text())
            z = np.fromfile(folder / "depth640x360.raw", np.uint16).reshape(360, 640).astype(float) / 1000.0
        else:
            st, z = capture(a.host, a.dev, folder)
        t = time.time()
        r = fitter.fit(st, z)
        report(r, folder)
        print("  (fit %.1f s)" % (time.time() - t))
        return
    gui(a, fitter)


if __name__ == "__main__":
    main()
