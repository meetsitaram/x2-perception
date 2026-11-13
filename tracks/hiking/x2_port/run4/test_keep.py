"""CPU test of keep.py (options k and y) with stubs (no Isaac). python test_keep.py <keep.py> <monitor.py>"""
import os, sys, types, tempfile
import numpy as np, torch

src = open(sys.argv[1]).read()
src = src.replace("from isaaclab.utils import configclass\n", "")
src = src.replace("from .staged import FarTargetPoseVelocityCommand, FarTargetPoseVelocityCommandCfg\n", "")
src = src.replace("from .monitor import _terrain_names", "_terrain_names = TN")
src = src.replace("from .staged import iteration", "iteration = lambda env: 1750.0")


class CommandTerm:  # Isaac Lab CommandTerm._resample, as in IsaacLab 37ddf62
    def _resample(self, env_ids):
        if len(env_ids) != 0:
            self.time_left[env_ids] = self.time_left[env_ids].uniform_(*self.cfg.resampling_time_range)
            self._resample_command(env_ids)


class FarTarget(CommandTerm):
    def __init__(self, cfg, env):
        self.cfg, self.num_envs, self.device = cfg, 8, "cpu"
        self.time_left = torch.zeros(8)
        self.random_velocity_indices = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0], dtype=torch.bool)
        self.vel_command_b = torch.zeros(8, 3)

    def _resample_command(self, env_ids):
        pass

    def _update_command(self):
        self.vel_command_b[:] = torch.tensor([0.7, 0.0, 0.9])


class Cfg:
    resampling_time_range = (8.0, 12.0)


mon = open(sys.argv[2]).read()
mns = {}
exec(mon[:mon.index("def walk_monitor")], mns)
ns = {"configclass": lambda c: c, "FarTargetPoseVelocityCommand": FarTarget, "FarTargetPoseVelocityCommandCfg": Cfg,
      "TN": mns["_terrain_names"], "__name__": "keep"}
exec(src, ns)

# ---- y: command ----
cfg = ns["TurnFarTargetPoseVelocityCommandCfg"]()
cfg.turn_share = 1.0
c = ns["TurnFarTargetPoseVelocityCommand"](cfg, None)
c._resample(torch.arange(8))
assert not c.is_turn_env[0] and c.is_turn_env[1:].all(), c.is_turn_env           # the stand column is never a turner
assert (c.time_left[1:] >= 4).all() and (c.time_left[1:] <= 6).all() and c.time_left[0] >= 8
c._update_command()
assert torch.allclose(c.vel_command_b[1], torch.tensor([0.0, 0.0, 0.9])) and torch.allclose(c.vel_command_b[0], torch.tensor([0.7, 0, 0.9]))
cfg.turn_share = 0.15
c._resample(torch.arange(8).repeat(1))
share = np.mean([(c._resample(torch.arange(8)) or c.is_turn_env[1:].float().mean().item()) for _ in range(2000)])
assert abs(share - 0.15) < 0.02, share


# ---- dont_turn_wait ----
class CM:
    def __init__(s, cmd, term=None): s.cmd, s.term = cmd, term
    def get_command(s, n): return s.cmd
    def get_term(s, n): return s.term


class Data: pass


robot = types.SimpleNamespace(data=Data())
env = types.SimpleNamespace(scene={"robot": robot})
env.command_manager = CM(torch.tensor([[0, 0, 0.8], [0, 0, 0.8], [0, 0, -0.8], [0.7, 0, 0.8], [0, 0, 0.3]]))
robot.data.root_ang_vel_b = torch.tensor([[0, 0, 0.05], [0, 0, 0.5], [0, 0, 0.5], [0, 0, 0.0], [0, 0, 0.0]])
v = ns["dont_turn_wait"](env, "base_velocity")
assert v.tolist() == [1, 0, 1, 0, 0], v  # slow turner pays; turning along pays nothing; turning the wrong way pays; walkers and small commands never

# ---- k: level keeper ----
names = ["perlin_rough", "perlin_rough_stand", "square_gaps", "pyramid_stairs", "pyramid_stairs_high", "pyramid_stairs_inv",
         "pyramid_stairs_inv_high", "boxes", "mesh_boxes", "hf_pyramid_slope_inv"]
props = [0.05, 0.05, 0.1, 0.15, 0.1, 0.15, 0.1, 0.1, 0.1, 0.1]
gen = types.SimpleNamespace(sub_terrains={n: types.SimpleNamespace(proportion=p) for n, p in zip(names, props)}, num_cols=20)
N = 8192
types_ = (torch.arange(N) * 20 // N).long()
origins = torch.arange(10 * 20 * 3, dtype=torch.float).reshape(10, 20, 3)
terrain = types.SimpleNamespace(cfg=types.SimpleNamespace(terrain_generator=gen), terrain_levels=torch.zeros(N, dtype=torch.long),
                                terrain_types=types_, terrain_origins=origins, env_origins=torch.zeros(N, 3), max_terrain_level=10)
cmdterm = types.SimpleNamespace(is_turn_env=torch.zeros(N, dtype=torch.bool))
env = types.SimpleNamespace(scene=types.SimpleNamespace(terrain=terrain), device="cpu", episode_length_buf=torch.zeros(N, dtype=torch.long),
                            max_episode_length=1000, command_manager=CM(torch.zeros(N, 3), cmdterm), common_step_counter=0)
Scene = type("Scene", (), {"__getitem__": lambda s, k: robot})
env.scene = Scene(); env.scene.terrain = terrain
tmp = tempfile.mkdtemp()
os.environ["HIKING_X2_LEVELS_MEANS"] = "pyramid_stairs:4.62,pyramid_stairs_high:7.07,square_gaps:0.61,perlin_rough_stand:6.5"
os.environ["HIKING_X2_LEVELS_IN"] = os.path.join(tmp, "missing.pt")
os.environ["HIKING_X2_LEVELS_OUT"] = os.path.join(tmp, "levels.pt")
os.environ["HIKING_X2_LEVELS_EVERY_S"] = "0"
os.environ["HIKING_X2_LEVELS_FORCE_RESET"] = "1"
out = ns["level_keeper"](env, None)
assert out["levels_restored"] == 1.0 and "turn/share" in out, out
col_type = mns["_terrain_names"](env)[1]
for n, m in [("pyramid_stairs", 4.62), ("pyramid_stairs_high", 7.07), ("square_gaps", 0.61), ("perlin_rough_stand", 6.5), ("boxes", 0.0)]:
    k = names.index(n)
    ids = torch.isin(types_, (col_type == k).nonzero().flatten())
    got = terrain.terrain_levels[ids].float().mean().item()
    assert abs(got - m) < 0.05, (n, got, m)
assert (env.episode_length_buf == 1000).all()
assert torch.equal(terrain.env_origins, origins[terrain.terrain_levels, types_])
assert os.path.exists(os.environ["HIKING_X2_LEVELS_OUT"])
# second process: restore from the file
saved = terrain.terrain_levels.clone()
ns["_k"].clear(); terrain.terrain_levels[:] = 0; env.episode_length_buf[:] = 0
os.environ["HIKING_X2_LEVELS_IN"] = os.environ["HIKING_X2_LEVELS_OUT"]
out = ns["level_keeper"](env, None)
assert out["levels_restored"] == 1.0 and torch.equal(terrain.terrain_levels, saved)
# without the forced reset: episode buffer untouched
ns["_k"].clear(); env.episode_length_buf[:] = 0; os.environ["HIKING_X2_LEVELS_FORCE_RESET"] = "0"
out = ns["level_keeper"](env, None)
assert out["levels_restored"] == 1.0 and (env.episode_length_buf == 0).all()
# pin: curriculum moves undone for resetting robots within the pin window, released after it
pinned = terrain.terrain_levels.clone()
terrain.terrain_levels[:10] = 0
out = ns["level_keeper"](env, torch.arange(5))
assert torch.equal(terrain.terrain_levels[:5], pinned[:5]) and (terrain.terrain_levels[5:10] == 0).all() and out.get("levels_pinned") == 1.0
env.common_step_counter = 2000
terrain.terrain_levels[:5] = 0
out = ns["level_keeper"](env, torch.arange(5))
assert (terrain.terrain_levels[:5] == 0).all() and out["levels_pinned"] == 0.0
env.common_step_counter = 0
print("pin: moves undone within 1,500 steps of a restore, released afterwards. Passed.")
# no variables: does nothing
ns["_k"].clear(); terrain.terrain_levels[:] = 0
for k in ("HIKING_X2_LEVELS_IN", "HIKING_X2_LEVELS_MEANS", "HIKING_X2_LEVELS_OUT"):
    del os.environ[k]
out = ns["level_keeper"](env, None)
assert out["levels_restored"] == 0.0 and terrain.terrain_levels.sum() == 0
print("keep.py: y command (stand column never turns, 4-6 s, forward zeroed, share %.3f), dont_turn_wait, k restore from means "
      "(per-type means kept) and from file, time-out after restore, no-op without variables. All checks passed." % share)

# ---- z: stop_motion ----
if "stop_motion" in ns:
    rb = types.SimpleNamespace(data=types.SimpleNamespace(root_lin_vel_b=torch.tensor([[0.6, 0, 0], [0.3, 0.4, 0], [0.6, 0, 0], [0.6, 0, 0], [0, 0, 0]])))
    e = types.SimpleNamespace(scene={"robot": rb}, command_manager=CM(torch.tensor([[0, 0, 0], [0.05, 0, 0.05], [0.6, 0, 0], [0, 0, 0.5], [0, 0, 0]])))
    v = ns["stop_motion"](e, "base_velocity")
    assert torch.allclose(v, torch.tensor([0.6, 0.5, 0.0, 0.0, 0.0])), v
    print("stop_motion: charges planar speed only under a stand command (0.6 / 0.5 / 0 walking / 0 turning / 0 still). Passed.")

# ---- o: overspeed, and what each side pays per control step (weight x dt, beside track_lin_vel_xy_exp +2.0, std 0.5) ----
if "overspeed" in ns:
    cases = [(0.6, 0.6), (0.84, 0.7), (0.30, 0.2), (0.45, 0.6), (0.3, 0.0), (0.5, 0.05)]
    rb = types.SimpleNamespace(data=types.SimpleNamespace(root_lin_vel_b=torch.tensor([[v, 0, 0] for v, c in cases])))
    e = types.SimpleNamespace(scene={"robot": rb}, command_manager=CM(torch.tensor([[c, 0, 0] for v, c in cases])))
    raw = ns["overspeed"](e, "base_velocity")
    assert torch.allclose(raw, torch.tensor([0.0, 0.14, 0.10, 0.0, 0.0, 0.0]), atol=1e-6), raw
    for (v, c), r in zip(cases[:4], raw.tolist()):
        track = 2.0 * 0.02 * float(np.exp(-((v - c) ** 2) / 0.25))
        print("  %.2f m/s under %.2f: overspeed %+.4f per step, speed reward %+.4f per step (perfect %+.4f)" % (v, c, -2.0 * 0.02 * r, track, 0.04))
    print("overspeed: 0 at or under the command, 0 for stand / tiny commands, linear above. Passed.")

# ---- the same log keys on every call, through the pin's release, with and without active turners, after an error ----
ns["_k"].clear()
os.environ["HIKING_X2_LEVELS_IN"] = os.path.join(tmp, "levels.pt"); os.environ["HIKING_X2_LEVELS_OUT"] = os.path.join(tmp, "levels.pt")
os.environ["HIKING_X2_LEVELS_EVERY_S"] = "0"; os.environ["HIKING_X2_LEVELS_FORCE_RESET"] = "0"
robot.data.root_ang_vel_b = torch.zeros(N, 3)
robot.data.root_lin_vel_b = torch.zeros(N, 3)
env.common_step_counter = 0
keys = None
for step in list(range(0, 3000, 100)):
    env.common_step_counter = step
    cmdterm.is_turn_env[:] = (step // 100) % 2 == 0          # turners on every other call
    env.command_manager.cmd[:, 2] = 0.8 if (step // 300) % 2 == 0 else 0.0
    out = ns["level_keeper"](env, torch.arange(0, N, 7))
    keys = keys or sorted(out)
    assert sorted(out) == keys, (step, sorted(out), keys)
    assert out["levels_pinned"] == (1.0 if step < 1500 else 0.0), (step, out["levels_pinned"])
env.scene.terrain = None                                       # force an error inside: the keys must survive
out = ns["level_keeper"](env, torch.arange(5)); env.scene.terrain = terrain
assert sorted(out) == keys, out
print("log keys: identical on all 31 calls across the pin release (1,500 steps), turners on/off, and after an internal error: %s. Passed." % keys)

# ---- z with HIKING_X2_STOP_ANY_YAW=1: turn-in-place commands are charged for planar drift too; walking never ----
os.environ["HIKING_X2_STOP_ANY_YAW"] = "1"
rb = types.SimpleNamespace(data=types.SimpleNamespace(root_lin_vel_b=torch.tensor([[0.3, 0.4, 0], [0.6, 0, 0], [0.1, 0, 0]])))
e = types.SimpleNamespace(scene={"robot": rb}, command_manager=CM(torch.tensor([[0, 0, 0.9], [0.6, 0, 0.5], [0, 0, 0]])))
v = ns["stop_motion"](e, "base_velocity")
assert torch.allclose(v, torch.tensor([0.5, 0.0, 0.1])), v
os.environ["HIKING_X2_STOP_ANY_YAW"] = "0"
assert torch.allclose(ns["stop_motion"](e, "base_velocity"), torch.tensor([0.0, 0.0, 0.1]))
print("stop_motion ANY_YAW: turner drifting 0.5 m/s charged 0.5, walker 0, stander 0.1; default unchanged. Passed.")
