"""Rectify the user's photo of the chest RealSense infrared frame and draw the measurement overlay.

Usage: python make_overlay.py <photo.jpg> <out.jpg>
The photo is a phone picture of a laptop screen, so the frame corners are picked by hand (CORNERS) and
the result is good to a few percent of image height, not to the pixel.
"""
import math
import sys

from PIL import Image, ImageDraw, ImageFont

# corners of the infrared frame in the phone photo: top-left, bottom-left, bottom-right, top-right (pixels)
CORNERS = [(100, 198), (72, 1248), (1790, 1240), (1812, 292)]
W, H = 1280, 760          # rectified size
PAD = 300                 # right-hand margin for labels

# solve used by the coordinator (docs/USER_RULES.md, "Camera, final pose")
CAM_H = 0.90              # camera height above the floor, m (estimate)
CAM_BACK = 0.12           # lens behind the toe tips, m (estimate)
VFOV = 58.0               # RealSense depth / infrared vertical field of view, deg
TILT = 53.4               # optical axis below horizontal, deg


def frac_from_top(d):
    """Image row (0 = top, 1 = bottom) of a floor point d metres ahead of the toe tips."""
    theta = math.degrees(math.atan2(CAM_H, d + CAM_BACK))      # ray angle below horizontal
    return 0.5 + math.tan(math.radians(theta - TILT)) / (2 * math.tan(math.radians(VFOV / 2)))


def main(src, dst):
    img = Image.open(src).convert("RGB")
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = CORNERS
    rect = img.transform((W, H), Image.QUAD, (x0, y0, x1, y1, x2, y2, x3, y3), Image.BICUBIC)
    out = Image.new("RGB", (W + PAD, H + 70), (16, 18, 22))
    out.paste(rect, (0, 70))
    d = ImageDraw.Draw(out)
    try:
        f = ImageFont.truetype("arial.ttf", 17)
        fb = ImageFont.truetype("arialbd.ttf", 19)
    except OSError:
        f = fb = ImageFont.load_default()
    d.text((10, 8), "X2 chest RealSense, infrared frame (same field of view as depth, 87 x 58 deg). "
                    "User's reference, 2026-11-11.", fill=(235, 235, 235), font=fb)
    d.text((10, 38), "Yellow = read from the user's tape on the floor.  Cyan = from the fitted camera pose "
                     "(0.90 m high, 53 deg down; estimates). Photo of a screen: good to a few % of height.", fill=(200, 200, 200), font=f)

    def line(fr, colour, label, dash=False):
        y = 70 + int(round(fr * H))
        y = min(max(y, 72), 70 + H - 2)
        if dash:
            for x in range(0, W, 24):
                d.line([(x, y), (x + 12, y)], fill=colour, width=2)
        else:
            d.line([(0, y), (W, y)], fill=colour, width=3)
        d.text((W + 10, y - 11), label, fill=colour, font=f)

    yel, cyan = (255, 214, 10), (64, 224, 255)
    # measured on the tape (zero at the toe line)
    line(0.985, yel, "0 m: toe tips (bottom edge)")
    line(0.54, yel, "about 0.55 m: rug inner border")
    # the two marks the user circled on the tape (2026-11-11): 3 ft and 110 cm. Rows placed between the
    # visible rug edge (0.95 m) and blue-rug edge (1.00 m) with the fitted pose.
    line(frac_from_top(0.914), yel, "0.91 m (3 ft mark): end of striped band")
    line(0.268, yel, "1.00 m: concrete strip / blue rug edge")
    line(frac_from_top(1.10), yel, "1.10 m (110 cm mark): on the blue rug")
    # from the fitted pose
    for dist in (0.25, 0.75, 1.3, 1.5, 1.75):
        fr = frac_from_top(dist)
        if 0.0 < fr < 1.0:
            line(fr, cyan, f"{dist:.2f} m (model)", dash=True)
    far = CAM_H / math.tan(math.radians(TILT - VFOV / 2)) - CAM_BACK
    line(0.004, cyan, f"top edge: about {far:.1f} m (model)", dash=True)
    out.save(dst, quality=90)
    for dist in (0, 0.55, 1.0, 1.5):
        print(dist, round(frac_from_top(dist), 3))
    print("far edge", round(far, 2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
