"""Align terrain-park's apparatus CAD export (the render dressing) with the eval's physics raster of the same apparatus.
The CAD's z = 0 is the venue floor; the eval raster's deck top is z = 0, so the drawn deck sat ~0.108 m above the deck the robot
stands on (2026-11-16: the robot and ACP's tiles looked sunk into the floor). Measures the median offset over deck points and
writes a copy shifted by it.

    python3 align_apparatus.py IN.xml OUT.xml        (IN with absolute asset paths)
"""
import os
import pathlib
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "eval"))
from ladder import load_course  # noqa: E402

src, dst = sys.argv[1], sys.argv[2]
app = load_course("ours")
m = mujoco.MjModel.from_xml_path(src)
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
gg, gid, off = np.ones(6, np.uint8), np.zeros(1, np.int32), []
for x in np.linspace(0.5, 10.5, 41):
    for y in np.linspace(0.3, 2.0, 9):
        dist = mujoco.mj_ray(m, d, np.array([x, y, 3.0]), np.array([0, 0, -1.0]), gg, 1, -1, gid)
        if dist >= 0:
            off.append(3.0 - dist - float(app.height(x, y)))
off = np.array(off)
dz = float(np.median(off[np.abs(off - np.median(off)) < 0.03]))        # the deck, not walls / headers
spec = mujoco.MjSpec.from_file(src)
for b in spec.worldbody.bodies:
    b.pos = [b.pos[0], b.pos[1], b.pos[2] - dz]
for g in spec.worldbody.geoms:
    g.pos = [g.pos[0], g.pos[1], g.pos[2] - dz]
for f in spec.worldbody.frames:
    f.pos = [f.pos[0], f.pos[1], f.pos[2] - dz]
pathlib.Path(dst).write_text(spec.to_xml())
print("apparatus CAD shifted by %+.3f m to the physics deck (%d deck points)" % (-dz, len(off)))
