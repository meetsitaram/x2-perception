"""Gamepad HUD for the sim viewer: draws the pad's live inputs (from pad.py's "hud" dict, relayed by the deploy node) as an RGB
image: a rounded pad body, both sticks with a direction triangle per side that lights with the deflection, L1/R1/L2/R2,
A/B/X/Y in Xbox colours, View, the speed limit and the ignition / e-stop state.

    img = draw(hud)          # (H, W, 3) uint8, top row first
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, SS = 440, 230, 2                       # image size and supersampling
BG, BODY, EDGE = (24, 26, 30), (52, 56, 63), (88, 94, 104)
DIM, TXT, ACC = (78, 84, 94), (220, 224, 230), (80, 200, 255)
FACE = dict(A=(70, 190, 90), B=(220, 70, 70), X=(70, 130, 230), Y=(235, 200, 60))
DEADZONE = 0.15


def _font(px):
    try:
        return ImageFont.load_default(size=px)
    except TypeError:
        return ImageFont.load_default()


F_S, F_M = _font(12 * SS), _font(15 * SS)


def _mix(a, b, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(x + (y - x) * t) for x, y in zip(a, b))


def _stick(d, cx, cy, r, x, y, label, axes="xy"):
    s = SS
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(36, 39, 44), outline=EDGE, width=2 * s)
    tri = 0.42 * r                                   # direction triangles just outside the ring
    for dx, dy, val in ((0, -1, y), (0, 1, -y), (-1, 0, -x), (1, 0, x)):
        if (dx and "x" not in axes) or (dy and "y" not in axes):
            col = (44, 47, 53)
        else:
            col = _mix(DIM, ACC, (val - DEADZONE) / (1 - DEADZONE)) if val > DEADZONE else DIM
        tx, ty = cx + dx * (r + tri * 1.15), cy + dy * (r + tri * 1.15)
        if dx == 0:
            pts = [(tx, ty + dy * tri * 0.55), (tx - tri * 0.75, ty - dy * tri * 0.45), (tx + tri * 0.75, ty - dy * tri * 0.45)]
        else:
            pts = [(tx + dx * tri * 0.55, ty), (tx - dx * tri * 0.45, ty - tri * 0.75), (tx - dx * tri * 0.45, ty + tri * 0.75)]
        d.polygon(pts, fill=col)
    kx, ky = cx + x * r * 0.62, cy - y * r * 0.62    # the stick head
    on = abs(x) > DEADZONE or abs(y) > DEADZONE
    d.ellipse([kx - r * 0.38, ky - r * 0.38, kx + r * 0.38, ky + r * 0.38], fill=ACC if on else (120, 126, 136),
              outline=(20, 22, 25), width=2 * s)
    d.text((cx, cy + r + tri * 2.2), label, fill=TXT, font=F_S, anchor="mm")


def _trigger(d, x0, y0, w, h, v, label):
    s = SS
    d.rounded_rectangle([x0, y0, x0 + w, y0 + h], radius=6 * s, fill=(36, 39, 44), outline=EDGE, width=2 * s)
    fh = (h - 4 * s) * max(0.0, min(1.0, v))
    if fh > 1:
        d.rounded_rectangle([x0 + 2 * s, y0 + h - 2 * s - fh, x0 + w - 2 * s, y0 + h - 2 * s], radius=4 * s,
                            fill=ACC if v > 0.5 else _mix(DIM, ACC, v))
    d.text((x0 + w / 2, y0 + h / 2), label, fill=TXT, font=F_S, anchor="mm")


def _pill(d, x0, y0, w, h, on, label):
    d.rounded_rectangle([x0, y0, x0 + w, y0 + h], radius=h / 2, fill=ACC if on else (40, 43, 48), outline=EDGE, width=2 * SS)
    d.text((x0 + w / 2, y0 + h / 2), label, fill=(15, 15, 15) if on else TXT, font=F_S, anchor="mm")


def draw(hud: dict | None) -> np.ndarray:
    s = SS
    im = Image.new("RGB", (W * s, H * s), BG)
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([4 * s, 4 * s, (W - 4) * s, (H - 4) * s], radius=34 * s, fill=BODY, outline=EDGE, width=3 * s)
    if not hud:
        d.text((W * s / 2, H * s / 2), "no pad", fill=TXT, font=F_M, anchor="mm")
        return np.asarray(im.resize((W, H), Image.LANCZOS))
    held = set(hud.get("held", []))
    # shoulders and triggers
    _trigger(d, 22 * s, 14 * s, 40 * s, 44 * s, hud.get("l2", 0.0), "L2")
    _pill(d, 70 * s, 26 * s, 52 * s, 22 * s, "L1" in held, "L1")
    _pill(d, (W - 122) * s, 26 * s, 52 * s, 22 * s, "R1" in held, "R1")
    _trigger(d, (W - 62) * s, 14 * s, 40 * s, 44 * s, hud.get("r2", 0.0), "R2")
    # sticks: left = walk (y forward) and turn (x), right = the viewer's camera
    _stick(d, 92 * s, 128 * s, 34 * s, hud.get("lx", 0.0), hud.get("ly", 0.0), "walk / turn")
    _stick(d, 262 * s, 150 * s, 30 * s, hud.get("rx", 0.0), hud.get("ry", 0.0), "camera")
    # face buttons (diamond)
    fx, fy, g, br = 372 * s, 118 * s, 25 * s, 12 * s
    for k, (dx, dy) in dict(Y=(0, -1), A=(0, 1), X=(-1, 0), B=(1, 0)).items():
        on = k in held
        cx, cy = fx + dx * g, fy + dy * g
        d.ellipse([cx - br, cy - br, cx + br, cy + br], fill=FACE[k] if on else (40, 43, 48), outline=FACE[k], width=2 * s)
        d.text((cx, cy), k, fill=(15, 15, 15) if on else FACE[k], font=F_S, anchor="mm")
    _pill(d, 188 * s, 84 * s, 50 * s, 20 * s, "View" in held, "View")
    # status line: speed limit, ignition, e-stop
    d.text((W * s / 2, 30 * s), "max %.1f m/s" % hud.get("vmax", 0.0), fill=TXT, font=F_M, anchor="mm")
    ign = hud.get("ign", "")
    if hud.get("estop"):
        msg, col = "E-STOP", (235, 70, 70)
    elif ign == "armed":
        msg, col = "IGNITION ARMED - press Y", (90, 220, 120)
    elif ign:
        msg, col = "igniting %s / 3.0 s" % ign, (235, 200, 60)
    else:
        msg, col = "", TXT
    if msg:
        d.text((W * s / 2, 58 * s), msg, fill=col, font=F_M, anchor="mm")
    return np.asarray(im.resize((W, H), Image.LANCZOS))


if __name__ == "__main__":       # preview: python3 pad_hud.py out.png
    import sys
    demo = dict(lx=-0.3, ly=0.9, rx=0.6, ry=0.0, l2=1.0, r2=0.2, held=["L2", "A", "R1"], vmax=0.5, estop=False, ign="")
    Image.fromarray(draw(demo)).save(sys.argv[1] if len(sys.argv) > 1 else "pad_hud.png")
