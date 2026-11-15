"""CPU test of option t (staged.py) and the measured air time in monitor.py, with stub sensor and env (no Isaac).
python test_t.py <staged.py> <monitor.py>"""
import sys, types, torch

src = open(sys.argv[1]).read()
ns = {"torch": torch, "SceneEntityCfg": object}
exec(src[src.index("# ---- t: touchdown air time ----"):src.index("# ---- r: reference-state starts ----")], ns)
f = ns["feet_air_touchdown_positive"]
# 4 robots x 2 feet: robot 0 lands a 0.30 s swing (pays 0.20) and a 0.05 s swing (pays 0, never negative);
# robot 1 lands a 0.6 s swing (capped 0.30); robot 2 lands 0.3 s but stands commanded (0); robot 3 no touchdown
first = torch.tensor([[1., 1.], [1., 0.], [1., 0.], [0., 0.]])
air = torch.tensor([[0.30, 0.05], [0.60, 0.2], [0.30, 0.0], [0.5, 0.5]])
sensor = types.SimpleNamespace(compute_first_contact=lambda dt: first, data=types.SimpleNamespace(last_air_time=air))
cmd = torch.tensor([[0.7, 0., 0.], [0.7, 0., 0.], [0.0, 0., 0.], [0.7, 0., 0.]])
env = types.SimpleNamespace(scene=types.SimpleNamespace(sensors={"contact_forces": sensor}), step_dt=0.02,
                            command_manager=types.SimpleNamespace(get_command=lambda n: cmd))
cfg = types.SimpleNamespace(name="contact_forces", body_ids=[0, 1])
v = f(env, "base_velocity", cfg)
assert torch.allclose(v, torch.tensor([0.25, 0.30, 0.0, 0.0]), atol=1e-6), v
assert (v >= 0).all()
assert 'set("abcdefrst")' in src and "HIKING_X2_AIR_TD_WEIGHT" in src

# monitor: measured air per touchdown over moving-commanded robots
msrc = open(sys.argv[2]).read()
mns = {}
exec(msrc[msrc.index("import numpy"):], mns)
mns["_state"].update(feet=[0, 1], names=["flat"], col_type=torch.tensor([0]))
forces = torch.zeros(4, 2, 3); forces[:, :, 2] = 10.0
sensor.data.net_forces_w = forces
robot = types.SimpleNamespace(data=types.SimpleNamespace(root_lin_vel_b=torch.tensor([[0.5, 0, 0]] * 4)))
terrain = types.SimpleNamespace(terrain_types=torch.zeros(4, dtype=torch.long), terrain_levels=torch.zeros(4))
env2 = types.SimpleNamespace(scene={"robot": robot, "contact_forces": sensor, "terrain": terrain}, step_dt=0.02, device="cpu",
                             command_manager=env.command_manager)
env2.scene = type("S", (dict,), {"terrain": terrain})(env2.scene)
out = mns["walk_monitor"](env2, None)
assert abs(out["air_per_touchdown_s"] - (0.30 + 0.05 + 0.60) / 3) < 1e-6, out  # robot 2 (standing command) excluded
assert abs(out["touchdown_air_ge_0p2"] - 2 / 3) < 1e-6, out
print("option t: pays 0.25 / 0 / 0.30 (cap) / 0 (standing command) per touchdown, never negative; monitor: air per touchdown "
      "%.3f s, share >= 0.2 s %.2f (moving robots only). All checks passed." % (out["air_per_touchdown_s"], out["touchdown_air_ge_0p2"]))
