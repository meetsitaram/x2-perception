"""The rolling map must move the opposite way to the robot. Each check has a case that must fail."""
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from ssr.mapcache import HALF, N_CELL, RES, SCALE, MapCache


def marker_after(dx=0.0, dy=0.0, dyaw=0.0, dz=0.0, x=0.5, y=0.3):
    c = MapCache(1, "cpu", bias_v=0.0, bias_w=0.0, noise_v=0.0)
    c.write(torch.tensor([[x]]), torch.tensor([[y]]), torch.tensor([[0.3]]), torch.tensor([[True]]))
    c.shift(torch.tensor([dx]), torch.tensor([dy]), torch.tensor([dyaw]), torch.tensor([dz]), noisy=False)
    i, j = (c.known[0] > 0.5).nonzero()[0].tolist()
    return (i + 0.5) * RES - HALF, (j + 0.5) * RES - HALF, float(c.h[0, i, j]) * SCALE


def near(a, b, tol=RES * 1.01):
    return abs(a - b) <= tol


def run(sign=1.0):
    x, y, h = marker_after(dx=sign * 0.4)                  # walking 0.4 m forward: the point is 0.4 m closer
    assert near(x, 0.1) and near(y, 0.3), (x, y)
    x, y, h = marker_after(dy=sign * 0.2)                  # stepping 0.2 m left: the point is 0.2 m less to the left
    assert near(x, 0.5) and near(y, 0.1), (x, y)
    x, y, h = marker_after(dyaw=sign * math.pi / 2)        # turning left 90 deg: a point ahead and left ends up right and ahead
    assert near(x, 0.3) and near(y, -0.5), (x, y)
    x, y, h = marker_after(dz=sign * 0.1)                  # base rises 0.1 m: the ground is 0.1 m lower relative to it
    assert abs(h - 0.2) < 1e-3, h


if __name__ == "__main__":
    run()
    try:
        run(sign=-1.0)
    except AssertionError:
        print("bad case caught: motion applied the wrong way")
    else:
        raise SystemExit("BAD CASE NOT CAUGHT")
    print("map memory checks pass")
