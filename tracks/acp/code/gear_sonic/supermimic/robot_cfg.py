"""Which robot SuperMimic runs on. Unset: the X2, with every X2 value left where it was.

``SM_ROBOT_CFG`` names a JSON file describing another humanoid (keys in ``load``); it lives in that robot's own repo, so
this one holds no robot data. It is read once, at import, because the state and network sizes follow from it.
"""

from __future__ import annotations

import json
import os
import pathlib

_PATH = os.environ.get("SM_ROBOT_CFG", "")


def load() -> dict | None:
    """{mjcf, plant, num_joints, tracked_bodies, lowest_origin, torso, mass_bodies} with mjcf and plant relative to the
    JSON, tracked_bodies in motion.TRACKED_BODIES' order (pelvis, left hip, knee, ankle, right hip, knee, ankle, torso,
    left shoulder, elbow, wrist, right shoulder, elbow, wrist)."""
    if not _PATH:
        return None
    p = pathlib.Path(_PATH).resolve()
    c = json.loads(p.read_text(encoding="utf-8"))
    c["mjcf"], c["plant"] = p.parent / c["mjcf"], p.parent / c["plant"]
    return c


CFG = load()
