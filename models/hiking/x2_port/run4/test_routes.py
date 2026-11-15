"""CPU test of routes.py (option w) without Isaac. python test_routes.py <routes.py> <extended_v1 dir>"""
import json, math, os, sys, types
import numpy as np, torch

src = open(sys.argv[1]).read()
src = src.replace("from isaaclab.utils import configclass\n", "")
src = src.replace("from .keep import TurnFarTargetPoseVelocityCommand, TurnFarTargetPoseVelocityCommandCfg\n", "")


class Base:  # stands in for TurnFarTargetPoseVelocityCommand: what the route class uses
    def __init__(self, cfg, env):
        pass

    def reset(self, env_ids=None):
        self._resample(env_ids)
        return {}

    def _resample(self, env_ids):
        pass

    def _update_command(self):
        pass


class BaseCfg:
    pass


ns = {"configclass": lambda c: c, "TurnFarTargetPoseVelocityCommand": Base, "TurnFarTargetPoseVelocityCommandCfg": BaseCfg, "__name__": "routes"}
exec(src, ns)
law, build = ns["waypoint_law"], ns["build_route_table"]

# ---- the waypoint law (eval harness form) ----
d = torch.tensor([[2.0, 0.0], [0.3, 0.0], [0.1, 0.0], [0.0, 1.0], [0.5, 0.5], [0.1, 0.0]])
vx, wz, arr = law(d, torch.zeros(6), torch.full((6,), 0.6), torch.tensor([False, False, False, False, False, True]))
exp_vx = [0.6, 0.3, 0.15, 0.0, min(0.6, math.hypot(0.5, 0.5)) * math.cos(math.pi / 4), 0.0]
exp_wz = [0.0, 0.0, 0.0, 1.0, min(1.0, 1.5 * math.pi / 4), 0.0]
assert torch.allclose(vx, torch.tensor(exp_vx), atol=1e-5), vx
assert torch.allclose(wz, torch.tensor(exp_wz), atol=1e-5), wz
assert arr.tolist() == [False, False, True, False, False, True]
print("law: 2 m ahead -> vmax 0.6; 0.3 m -> 0.30; 0.1 m (pass-through) -> 0.15 floor; 90 deg off -> turn only (1 rad/s); "
      "45 deg -> 0.5 m/s x cos and 1.0 rad/s cap; at the final station -> zero. Passed.")

# ---- the route table from the real set: generator order, coordinates, counts ----
root = sys.argv[2]
types_ = ["single_beams", "beam_fields", "blocks", "wedges", "tight_spots", "log_corridors", "homes", "street_front", "stairs"]
ncols = {t: (2 if t == "stairs" else 1) for t in types_}
meta = {t: json.load(open(os.path.join(root, t + ".json")))["tiles"] for t in types_}
num_rows, size = 10, 8.0
cols, c = {}, 20
for t in types_:
    cols[t] = list(range(c, c + ncols[t])); c += ncols[t]
calls, rng = {}, np.random.default_rng(0)
origins = np.zeros((num_rows, 32, 3))
for t in types_:
    for col in cols[t]:
        for row in range(num_rows):
            cand = [x for x in meta[t] if x["level"] == row and x["spawns"]] or [x for x in meta[t] if x["spawns"]]
            tile = cand[rng.integers(len(cand))]
            sp = max(tile["spawns"], key=lambda s: s.get("clearance_m", 0.0))
            calls.setdefault(t, []).append((tile["index"], [sp["x"], sp["y"], sp.get("z", 0.0)]))
            origins[row, col] = [(row + 0.5) * size + sp["x"] - 40, (col + 0.5) * size + sp["y"] - 128, sp.get("z", 0.0) + 0.0]
table = build(calls, meta, cols, num_rows, origins.tolist())
per_type = {t: sum(len(table.get((r, cc), [])) for cc in cols[t] for r in range(num_rows)) for t in types_}
cells = {t: sum(1 for cc in cols[t] for r in range(num_rows) if (r, cc) in table) for t in types_}
print("routes per type on this layout (cells with routes of %d):" % num_rows, {t: (per_type[t], cells[t]) for t in types_})
# a route start maps back to its tile: world - tile centre == route start in tile coordinates
t = "homes"; j = 3; i, sp = calls[t][j]; col, row = cols[t][j // num_rows], j % num_rows
tile = {x["index"]: x for x in meta[t]}[i]
start_w = table[(row, col)][0][0]
centre = [(row + 0.5) * size - 40, (col + 0.5) * size - 128]
assert abs(start_w[0] - centre[0] - tile["routes"][0]["start"][0]) < 1e-6 and abs(start_w[1] - centre[1] - tile["routes"][0]["start"][1]) < 1e-6
print("coordinates: a home route start lands at its tile position in world coordinates. Passed.")

# ---- following a route: pass-through inside 0.2 m, stop at the final station ----
R = ns["RouteFarTargetPoseVelocityCommand"]
cmd = R.__new__(R)
cmd.cfg = types.SimpleNamespace(pass_radius=0.2, route_share=1.0)
cmd.device, n = "cpu", 1
cmd.on_route = torch.tensor([True]); cmd.route_done = torch.tensor([False])
wps = [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0]]
cmd.route_wp = torch.zeros(1, ns["MAX_WP"], 3); cmd.route_wp[0, :3] = torch.tensor(wps); cmd.route_wp[0, 3:] = cmd.route_wp[0, 2]
cmd.route_n = torch.tensor([3]); cmd.route_i = torch.tensor([0])
cmd.vel_command_b = torch.zeros(1, 3); cmd.pos_command_w = torch.zeros(1, 3); cmd.max_command_b = torch.tensor([[0.6, 0, 0]])
data = types.SimpleNamespace(root_pos_w=torch.zeros(1, 3), heading_w=torch.zeros(1))
cmd.robot = types.SimpleNamespace(data=data)
pos, yaw, dt, hist = np.zeros(2), 0.0, 0.02, []
for step in range(2000):
    data.root_pos_w[0, :2] = torch.tensor(pos); data.heading_w[0] = yaw
    R._update_command(cmd)
    vx, wz = cmd.vel_command_b[0, 0].item(), cmd.vel_command_b[0, 2].item()
    yaw += wz * dt; pos += vx * dt * np.array([math.cos(yaw), math.sin(yaw)])
    hist.append((cmd.route_i[0].item(), vx))
    if cmd.route_done[0]:
        break
idx = [h[0] for h in hist]
assert idx[-1] == 2 and cmd.route_done[0] and np.linalg.norm(pos - np.array([2.0, 1.0])) < 0.2, (idx[-1], pos)
assert min(v for i_, v in hist if i_ == 0) > 0.1  # no stop at the pass-through waypoint before it advanced
R._update_command(cmd); assert cmd.vel_command_b[0].abs().sum() == 0  # holds the station
print("follow: a kinematic robot passes waypoints 1 and 2 without stopping, stops inside 0.2 m of waypoint 3 after %.1f s and holds. Passed." % (len(hist) * dt))

# ---- assignment at a reset: placement, origin, heading, flags; no assignment outside a reset ----
N = 64
cmd = R.__new__(R)
cmd.cfg = types.SimpleNamespace(pass_radius=0.2, route_share=1.0); cmd.device = "cpu"; cmd.table = table
cmd.on_route = torch.zeros(N, dtype=torch.bool); cmd.route_done = torch.zeros(N, dtype=torch.bool)
cmd.route_wp = torch.zeros(N, ns["MAX_WP"], 3); cmd.route_n = torch.ones(N, dtype=torch.long); cmd.route_i = torch.zeros(N, dtype=torch.long)
cmd.is_turn_env = torch.ones(N, dtype=torch.bool); cmd.time_left = torch.zeros(N); cmd.pos_command_w = torch.zeros(N, 3)
levels = torch.randint(0, 10, (N,)); tcols = torch.tensor([cols["homes"][0]] * 32 + [3] * 32)   # half on a home column, half on a paper column
cmd.terrain = types.SimpleNamespace(terrain_levels=levels, terrain_types=tcols, env_origins=torch.zeros(N, 3))
written = {}
cmd.robot = types.SimpleNamespace(data=types.SimpleNamespace(default_root_state=torch.tensor([[0, 0, 0.75, 1, 0, 0, 0]] * N, dtype=torch.float)),
                                  write_root_pose_to_sim=lambda pose, env_ids: written.update(pose=pose.clone(), ids=env_ids.clone()))
cmd._in_reset = False
R._resample(cmd, torch.arange(N)); assert not cmd.on_route.any() and not written
cmd._in_reset = True
R._resample(cmd, torch.arange(N))
assert cmd.on_route[:32].all() and not cmd.on_route[32:].any() and not cmd.is_turn_env[:32].any() and (cmd.time_left[:32] > 1e5).all()
ids = written["ids"]; pose = written["pose"]
assert torch.equal(ids, torch.arange(32))
for e in range(32):
    starts = [r[0] for r in table[(int(levels[e]), int(tcols[e]))]]
    o = cmd.terrain.env_origins[e].tolist()
    assert any(np.allclose(o, s_) for s_ in starts), (e, o)
    assert abs(pose[e, 2].item() - (o[2] + 0.75)) < 1e-6 and abs(torch.norm(pose[e, 3:]).item() - 1) < 1e-5
print("assign: at a reset every robot on a route cell gets a route start as origin, height + 0.75 m, a unit heading quaternion, "
      "no turner, no mid-episode resample; robots on paper columns untouched; nothing happens outside a reset. Passed.")
