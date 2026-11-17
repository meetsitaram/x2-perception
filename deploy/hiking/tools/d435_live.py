"""Live D435 depth from the robot's PC2, viewed on this PC. Nothing is installed on PC2: it runs v4l2-ctl and streams raw Z16
frames over ssh (the installed librealsense cannot open the camera on JetPack 6, STATUS.md 2026-11-16).

    python3 deploy/hiking/tools/d435_live.py [--host run@<PC2_IP>]   (or export PC2_HOST=<PC2_IP>) [--dev /dev/video11] [--size 640x480] [--fps 15]

Left: depth in metres (near blue, far yellow, no return black). Right: what the policy would get, roughly: the full frame
resampled to the trained 64 x 36, then the card's crop / blur / clip / scale (core.preprocess_depth), 36 x 32. The camera
FOV is used as nominal 87 x 58 deg (no calibration yet); H toggles the hole fill (fill_holes); on the chest the image should resemble
docs/hardware/chest_camera_mount/expected_depth_hiking.png. Esc / q quits; s saves a PNG and the raw frame.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from hiking_deploy import core  # noqa: E402


def colour(z, far):
    v = np.clip(z / far, 0, 1)
    img = np.stack([np.clip(1.6 * v, 0, 1), np.clip(1.2 * v ** 0.7, 0, 1), np.clip(1 - 1.5 * v, 0, 1)], -1)
    img[~(z > 0)] = 0
    return (img * 255).astype(np.uint8)


def fill_holes(raw, iters=3):
    """Cells with no return at all (inf) take the nearest surface (min) of their finite 8-neighbours, growing up to iters
    cells. Stereo holes at the toes (closer than the ~0.3 m minimum range) and in the occlusion shadows beside the feet
    otherwise read as 'far' (3 m) right in front of the robot, which training never saw: its ray-cast depth has no holes.
    Large no-return areas (sky, a real drop) stay far."""
    out = raw.copy()
    for _ in range(iters):
        bad = ~np.isfinite(out)
        if not bad.any():
            break
        p = np.pad(out, 1, constant_values=np.inf)
        nb = np.min(np.stack([p[1 + di:p.shape[0] - 1 + di, 1 + dj:p.shape[1] - 1 + dj]
                              for di in (-1, 0, 1) for dj in (-1, 0, 1) if di or dj]), 0)
        out[bad] = nb[bad]
    return out


def to_policy(z, card_depth, fill=True):
    """Full frame (H, W) metres -> raw 36 x 64 by min pooling over blocks (nearest surface wins, holes ignored), then the
    card's preprocessing -> 36 x 32 in 0..1."""
    H, W = z.shape
    r, c = 36, 64
    zz = np.where(z > 0, z, np.inf)
    ys, xs = np.linspace(0, H, r + 1).astype(int), np.linspace(0, W, c + 1).astype(int)
    out = np.empty((r, c))
    for i in range(r):
        band = zz[ys[i]:ys[i + 1]]
        for j in range(c):
            out[i, j] = band[:, xs[j]:xs[j + 1]].min()
    if fill:
        out = fill_holes(out)
    return core.preprocess_depth(out, card_depth)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=os.environ.get("PC2_HOST") and "run@" + os.environ["PC2_HOST"],
                    help="run@<PC2_IP>, or set PC2_HOST (PC2's address on the robot's Wi-Fi)")
    ap.add_argument("--dev", default="/dev/video11")
    ap.add_argument("--size", default="640x480")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--model", default=str(HERE.parent / "models" / "x2-hiking-in-the-wild"))
    a = ap.parse_args()
    if not a.host:
        ap.error("no PC2 address: pass --host run@<PC2_IP> or export PC2_HOST=<PC2_IP>")
    W, H = (int(v) for v in a.size.split("x"))
    card = core.Card.load(a.model)
    cmd = ["ssh", "-o", "BatchMode=yes", "-C", "-i", os.path.expanduser("~/.ssh/id_ed25519"), a.host,
           # frames through a FIFO: v4l2-ctl prints progress characters on stdout, which would shift the raw frames
           "F=$(mktemp -u /tmp/d435.XXXX); mkfifo $F; trap 'rm -f $F; kill %%1 2>/dev/null' EXIT; "
           "v4l2-ctl -d %s --set-fmt-video=width=%d,height=%d,pixelformat='Z16 ' --set-parm=%d --stream-mmap --stream-to=$F "
           ">/dev/null 2>&1 & cat $F" % (a.dev, W, H, a.fps)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=0)
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame
    pygame.init()
    S = 2 if W <= 480 else 1                                  # left pane scale
    PW, PH = 32 * 10, 36 * 10                                 # policy pane: 36 x 32 at 10 px per cell
    screen = pygame.display.set_mode((W * S + PW + 30, max(H * S, PH) + 60))
    pygame.display.set_caption("X2 chest D435 (PC2, live)")
    font = pygame.font.SysFont(None, 24)
    nbytes, buf, n, t0, fps = W * H * 2, bytearray(), 0, time.monotonic(), 0.0
    out_dir = HERE.parent / "logs" / "d435"
    fill = True
    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key in (pygame.K_ESCAPE, pygame.K_q)):
                proc.terminate()
                return
            if e.type == pygame.KEYDOWN and e.key == pygame.K_h:
                fill = not fill
            if e.type == pygame.KEYDOWN and e.key == pygame.K_s and n:
                out_dir.mkdir(parents=True, exist_ok=True)
                stem = out_dir / time.strftime("%H%M%S")
                pygame.image.save(screen, str(stem) + ".png")
                np.save(str(stem) + ".npy", z)
                print("saved", stem)
        while len(buf) < nbytes:
            chunk = proc.stdout.read(nbytes - len(buf))
            if not chunk:
                print("stream ended (ssh / v4l2-ctl stopped)")
                return
            buf += chunk
        z = np.frombuffer(bytes(buf[:nbytes]), np.uint16).reshape(H, W).astype(np.float32) / 1000.0    # 1 mm units
        del buf[:nbytes]
        n += 1
        if time.monotonic() - t0 > 1.0:
            fps, n0 = n / (time.monotonic() - t0), n
            t0, n = time.monotonic(), 0
        pol = to_policy(z, card.depth, fill)
        screen.fill((24, 26, 30))
        left = pygame.surfarray.make_surface(colour(z, 4.0).swapaxes(0, 1))
        screen.blit(pygame.transform.scale(left, (W * S, H * S)) if S != 1 else left, (10, 10))
        g = (np.clip(pol, 0, 1) * 255).astype(np.uint8)
        right = pygame.surfarray.make_surface(np.stack([g, g, g], -1).swapaxes(0, 1))
        screen.blit(pygame.transform.scale(right, (PW, PH)), (W * S + 20, 10))
        valid = z > 0
        med = float(np.median(z[valid])) if valid.any() else 0.0
        txt = "%.1f fps  valid %.0f%%  median %.2f m  centre %.2f m   |   right: policy input 36x32 (white = far, 3 m), hole fill %s [H]" % (
            fps, 100 * valid.mean(), med, float(z[H // 2, W // 2]), "ON" if fill else "off")
        screen.blit(font.render(txt, True, (220, 224, 230)), (10, max(H * S, PH) + 20))
        pygame.display.flip()


if __name__ == "__main__":
    main()
