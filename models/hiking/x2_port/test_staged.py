"""CPU unit test of staged.py's schedule and kernels (no Isaac): python test_staged.py <path to staged.py>"""
import os, sys, types, torch
src = open(sys.argv[1]).read()
a, b = src.index("STEPS_PER_ITERATION"), src.index("# ---- a: staged rewards")
ns = {"os": os, "torch": torch}
exec(src[a:b], ns)
c, d = src.index("def lin_tracking_kernel"), src.index("def track_lin_vel_xy_exp_staged")
exec(src[c:d], ns)
S = ns["schedule"]
assert S(0, 0.3, 500, 1500) == 0.3 and S(500, 0.3, 500, 1500) == 0.3 and S(1500, 0.3, 500, 1500) == 1.0 and S(9999, 0.3, 500, 1500) == 1.0
assert abs(S(1000, 0.3, 500, 1500) - 0.65) < 1e-9 and abs(S(1000, 4.0, 500, 1500) - 2.5) < 1e-9 and abs(S(1000, 0.25, 500, 1500, 0.5) - 0.375) < 1e-9
env = types.SimpleNamespace(common_step_counter=24 * 100)
os.environ["HIKING_ITER_OFFSET"] = "900"
assert ns["iteration"](env) == 1000.0
os.environ["HIKING_X2_STAGE"] = "500,1500"
assert abs(ns["_s"](env, 0.3) - 0.65) < 1e-9
os.environ["HIKING_X2_OPTS"] = "a, c"
assert ns["_opts"]() == {"a", "c"}
k = ns["lin_tracking_kernel"]
cmd = torch.tensor([[0.63, 0.0], [0.63, 0.0], [0.0, 0.0]]); vel = torch.tensor([[0.0, 0.0], [0.63, 0.0], [0.0, 0.0]])
wide, narrow = k(cmd, vel, 0.5), k(cmd, vel, 0.25)
assert abs(wide[0].item() - 0.2044) < 1e-3 and narrow[0].item() < 0.002 and wide[1] == 1 and narrow[1] == 1 and narrow[2] == 1
g = ns["progress_gate"]
cx = torch.tensor([0.6, 0.6, 0.6, 0.0, 0.6]); vx = torch.tensor([0.0, 0.15, 0.6, 0.0, -0.2])
full = g(cx, vx, 1.0); off = g(cx, vx, 0.0); half = g(cx, vx, 0.5)
assert torch.allclose(full, torch.tensor([0.0, 0.5, 1.0, 1.0, 0.0])) and torch.allclose(off, torch.ones(5)) and torch.allclose(half, torch.tensor([0.5, 0.75, 1.0, 1.0, 0.5]))
# must fail: a stander must not pass the full gate
assert not (full[0] > 0)
print("staged.py schedule and kernels: all checks passed; zero-speed pay for a 0.63 m/s command: std 0.5 -> %.3f, std 0.25 -> %.4f" % (wide[0], narrow[0]))
