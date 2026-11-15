"""CPU unit test of v9/v9.py's pure functions (column_type_names, set_stand_ranges, type_stats_update, scaled). Exit 0 = pass."""
import ast
import inspect
import os
import sys
import types

import numpy as np
import torch

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "v9.py")
tree = ast.parse(open(SRC).read())
ns = {"torch": torch, "np": np, "inspect": inspect, "os": os, "_st": {}}
for node in tree.body:
    if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "PAPER_TYPES" for t in node.targets):
        exec(compile(ast.Module(body=[node], type_ignores=[]), SRC, "exec"), ns)
    if isinstance(node, ast.FunctionDef) and node.name in ("column_type_names", "env_type_names", "set_stand_ranges", "type_stats_update", "scaled"):
        exec(compile(ast.Module(body=[node], type_ignores=[]), SRC, "exec"), ns)
fail = []

# column_type_names: proportion rule over 32 columns (20 paper of 32 as in v8: 0.625 paper share)
sub = {"perlin_rough": 0.05 * 0.625, "perlin_rough_stand": 0.05 * 0.625, "square_gaps": 0.1 * 0.625, "pyramid_stairs_inv": 0.75 * 0.625,
       "ext_a": 0.34375, "perlin_rough_velocity": 2 / 32}
s = sum(sub.values()); sub = {k: v / s for k, v in sub.items()}
gen = types.SimpleNamespace(num_cols=32, sub_terrains={k: types.SimpleNamespace(proportion=v) for k, v in sub.items()})
cols = ns["column_type_names"](gen)
if len(cols) != 32 or cols[0] != "perlin_rough" or cols[-1] != "perlin_rough_velocity" or "perlin_rough_stand" not in cols:
    fail.append("column_type_names %s" % cols)

# set_stand_ranges: only masked robots change, yaw untouched
x = torch.tensor([[0.0, 0.8]] * 4); y = torch.tensor([[-0.3, 0.3]] * 4)
m = torch.tensor([True, False, True, False])
ns["set_stand_ranges"](x, y, m)
if not (x[0].abs().sum() == 0 and y[2].abs().sum() == 0 and x[1, 1] == 0.8 and y[3, 0] == -0.3):
    fail.append("set_stand_ranges %s %s" % (x, y))

# type_stats_update: only eligible episodes enter score/up/down/early; non-paper types ignored
acc = {}
tps = ["square_gaps", "square_gaps", "ext_a", "pyramid_stairs_inv"]
score = torch.tensor([0.2, 0.9, 0.5, 0.7]); up = score > 0.6; down = score < 0.3
elig = torch.tensor([True, False, True, True]); term = torch.tensor([True, False, False, False])
ns["type_stats_update"](acc, tps, score, up, down, elig, term, a=1.0)
g = acc.get("square_gaps")
if "ext_a" in acc or g is None or abs(g["score"] - 0.2) > 1e-6 or g["down"] != 1.0 or g["early"] != 1.0 or abs(g["elig"] - 0.5) > 1e-6 or g["up"] != 0.0:
    fail.append("type_stats_update %s" % acc)
if acc.get("pyramid_stairs_inv", {}).get("up") != 1.0:
    fail.append("type_stats_update stairs %s" % acc)

# scaled: paper robots x scale, others x 1; signature kept
def term(env, sensor_cfg=None, asset_cfg=None):
    return torch.ones(3)
ns["_st"]["types"] = ["square_gaps", "ext_a", "boxes"]; ns["_st"]["paper_mask"] = torch.tensor([True, False, True])
f = ns["scaled"](term, 0.5, "paper")
r = f(None, sensor_cfg=1, asset_cfg=2)
if r.tolist() != [0.5, 1.0, 0.5] or list(inspect.signature(f).parameters) != ["env", "sensor_cfg", "asset_cfg"]:
    fail.append("scaled paper %s %s" % (r, inspect.signature(f)))
if ns["scaled"](term, 0.5, "all")(None).tolist() != [0.5, 0.5, 0.5]:
    fail.append("scaled all")

print("TEST_V9 " + ("PASS" if not fail else "FAIL " + " | ".join(fail)))
sys.exit(1 if fail else 0)
