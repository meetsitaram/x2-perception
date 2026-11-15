"""CPU unit test of v8/fair.py's pure functions (update_flags, promotion_masks, progress_ratio). Exit 0 = pass."""
import ast
import os
import sys

import torch

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fair.py")
tree = ast.parse(open(SRC).read())
ns = {"torch": torch}
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in ("update_flags", "promotion_masks", "progress_ratio"):
        exec(compile(ast.Module(body=[node], type_ignores=[]), SRC, "exec"), ns)
fail = []

# update_flags: OR-accumulates over resamples within an episode; untouched robots keep their flag
n = torch.zeros(6, dtype=torch.bool)
stand = torch.tensor([1, 0, 0, 0, 0, 0], dtype=torch.bool)
turn = torch.tensor([0, 1, 0, 0, 0, 0], dtype=torch.bool)
vcap = torch.tensor([float("inf"), float("inf"), 0.3, float("inf"), float("inf"), 0.2])
ns["update_flags"](n, torch.tensor([0, 1, 2, 3]), stand, turn, vcap)
if n.tolist() != [True, True, True, False, False, False]:
    fail.append("update_flags first pass %s" % n.tolist())
ns["update_flags"](n, torch.tensor([0, 3]), torch.zeros(6, dtype=torch.bool), torch.zeros(6, dtype=torch.bool), vcap)
if n.tolist() != [True, True, True, False, False, False]:
    fail.append("update_flags must not clear a flag %s" % n.tolist())

# promotion_masks: the paper's rule for unfrozen robots, no move for frozen ones
xy = torch.tensor([0.9, 0.1, 0.45, 0.9, 0.1])
yaw = torch.ones(5)
frozen = torch.tensor([0, 0, 0, 1, 1], dtype=torch.bool)
up, down = ns["promotion_masks"](xy, yaw, frozen)
if up.tolist() != [True, False, False, False, False] or down.tolist() != [False, True, False, False, False]:
    fail.append("promotion_masks up %s down %s" % (up.tolist(), down.tolist()))

# progress_ratio: speed along the command over the command, clipped to [0, 1]
v = torch.tensor([[0.25, 0.0], [0.0, 0.0], [-0.3, 0.0], [1.0, 0.0], [0.0, 0.2]])
c = torch.tensor([[0.5, 0.0, 0.0], [0.6, 0.0, 0.0], [0.5, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.2, 0.0]])
p = ns["progress_ratio"](v, c)
if [round(x, 3) for x in p.tolist()] != [0.5, 0.0, 0.0, 1.0, 1.0]:
    fail.append("progress_ratio %s" % p.tolist())
print("\n".join("FAIL " + f for f in fail) if fail else "TEST_FAIR PASS")
sys.exit(1 if fail else 0)
