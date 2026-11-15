"""CPU tests of v4j (no Isaac). python test_v4j.py <config/x2 dir of files_instinctlab> <extended_v4j> <extended_v1> <stairs_inset> <motion v3 npz> <motion v2 npz>"""
import json, math, sys
import numpy as np, torch

D, NEW, V1, INS, MV3, MV2 = sys.argv[1:7]
ok = []

# ---------- contact.py ----------
src = open(f"{D}/contact.py.v4j").read().replace("from isaaclab.managers import SceneEntityCfg\n", "SceneEntityCfg = object\n")
C = {"__name__": "contact"}
exec(src, C)
X2 = ["pelvis", "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link", "left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link",
      "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link", "right_knee_link", "right_ankle_pitch_link", "right_ankle_roll_link",
      "waist_yaw_link", "waist_pitch_link", "torso_link", "left_shoulder_pitch_link", "left_shoulder_roll_link", "left_shoulder_yaw_link",
      "left_elbow_link", "left_wrist_yaw_link", "left_wrist_pitch_link", "left_wrist_roll_link", "right_shoulder_pitch_link",
      "right_shoulder_roll_link", "right_shoulder_yaw_link", "right_elbow_link", "right_wrist_yaw_link", "right_wrist_pitch_link",
      "right_wrist_roll_link", "head_yaw_link", "head_pitch_link"]
g = C["body_groups"](X2)
assert [X2[i] for i in g["feet"]] == ["left_ankle_roll_link", "right_ankle_roll_link"]
assert all("elbow" in X2[i] or "wrist" in X2[i] for i in g["arms"]) and len(g["arms"]) == 8
assert "torso_link" in [X2[i] for i in g["trunk"]] and "left_shoulder_roll_link" in [X2[i] for i in g["trunk"]]
assert "left_knee_link" in [X2[i] for i in g["legs"]]
ok.append("body groups: feet 2, legs %d, trunk %d, arms %d" % (len(g["legs"]), len(g["trunk"]), len(g["arms"])))
# graded cost: body 0 strong, body 1 arm
is_arm = torch.tensor([False, True])
F = lambda a, b: torch.tensor([[[a, 0, 0], [b, 0, 0]]], dtype=torch.float)
gc = C["graded_cost"]
assert gc(F(0, 0), is_arm, torch.zeros(1, 2)).item() == 0
assert abs(gc(F(10, 0), is_arm, torch.zeros(1, 2)).item() - 1.1) < 1e-5        # strong touch 10 N -> 1.1
assert abs(gc(F(500, 0), is_arm, torch.zeros(1, 2)).item() - 4.0) < 1e-5       # capped at 4
assert gc(F(0, 20), is_arm, torch.zeros(1, 2)).item() == 0                     # arm brush 20 N, 0.1 s: free
assert abs(gc(F(0, 20), is_arm, torch.tensor([[0, 0.6]])).item() - 0.25 * 1.2) < 1e-5   # held 0.6 s: paid
assert abs(gc(F(0, 100), is_arm, torch.zeros(1, 2)).item() - 0.25 * 2.0) < 1e-5  # 100 N: paid
ok.append("graded cost: strong 1-4 per body; arm free under 30 N and 0.5 s, then 0.25 x")
rm = C["torso_rest_mask"](torch.tensor([[0, 0, 200.0], [200.0, 0, 50], [0, 0, 30.0], [100, 100, 150]]))
assert rm.tolist() == [True, False, False, True], rm
ok.append("torso termination: lying on something ends, wall push does not")
# boxes: a unit box at the origin rotated 90 deg about z, half (1, 0.2, 0.5)
q = torch.tensor([math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)])
R = C["quat_to_rot"](q)
cen = torch.tensor([[[0.0, 0.0, 0.5]]]); half = torch.tensor([[[1.0, 0.2, 0.5]]]); rot = R.view(1, 1, 3, 3); valid = torch.tensor([[True]])
pts = torch.tensor([[[0.0, 0.5, 0.5]], [[0.5, 0.0, 0.5]], [[0.0, 0.0, 0.99]]])   # inside (rotated long axis is y), outside, too shallow
ins = C["inside_depth"](pts, cen.expand(3, 1, 3), half.expand(3, 1, 3), rot.expand(3, 1, 3, 3), valid.expand(3, 1))
assert ins.tolist() == [True, False, False], ins
d = C["wall_distance_xy"](torch.tensor([[0.7, 0.0], [0.0, 1.5], [0.1, 0.0]]), torch.zeros(3), cen.expand(3, 1, 3), half.expand(3, 1, 3), rot.expand(3, 1, 3, 3), valid.expand(3, 1))
assert torch.allclose(d, torch.tensor([0.5, 0.5, -0.1]), atol=1e-4), d
ok.append("box tests: inside depth and planar distance with a rotated box")

# ---------- merged terrain set ----------
z1, zi, zn = np.load(f"{V1}/stairs.npz"), np.load(f"{INS}/stairs.npz"), np.load(f"{NEW}/stairs.npz")
jn = json.load(open(f"{NEW}/stairs.json"))["tiles"]; ji = {t["tile_id"]: t for t in json.load(open(f"{INS}/stairs.json"))["tiles"]}
assert len(jn) == 80 and list(zn["tile_id"]) == list(z1["tile_id"]) and len(zn["box_start"]) == 81
n_ins = 0
for r, t in enumerate(jn):
    assert t["index"] == r and t["tile_id"] == str(zn["tile_id"][r])
    src_z, i = (zi, ji[t["tile_id"]]["index"]) if t["tile_id"] in ji else (z1, r)
    n_ins += t["tile_id"] in ji
    a = zn["box_center"][zn["box_start"][r]:zn["box_start"][r + 1]]; b = src_z["box_center"][src_z["box_start"][i]:src_z["box_start"][i + 1]]
    assert a.shape == b.shape and np.allclose(a, b), t["tile_id"]
    assert np.array_equal(zn["height"][r], src_z["height"][i])
    m = zn["mesh_start"][r + 1] - zn["mesh_start"][r]; m2 = src_z["mesh_start"][i + 1] - src_z["mesh_start"][i]
    assert m == m2
assert n_ins == 60
kinds = {}
for t in jn:
    for rt in t["routes"]:
        kinds[rt.get("kind", "v1")] = kinds.get(rt.get("kind", "v1"), 0) + 1
        assert len(rt.get("z", [0] * (len(rt["waypoints"]) + 1))) == len(rt["waypoints"]) + 1
ok.append("stairs merged: 60 inset tiles + 20 amphitheatre at extended_v1 rows, boxes / heights / meshes match; routes by kind %s" % json.dumps(kinds))

# ---------- routes.py.v4j: table on the merged set, side law, lateral vy ----------
rsrc = open(f"{D}/routes.py.v4j").read().replace("from isaaclab.utils import configclass\n", "").replace(
    "from .keep import TurnFarTargetPoseVelocityCommand, TurnFarTargetPoseVelocityCommandCfg\n", "")
class Base:
    def __init__(self, cfg, env): pass
R_ = {"configclass": lambda c: c, "TurnFarTargetPoseVelocityCommand": Base, "TurnFarTargetPoseVelocityCommandCfg": object, "__name__": "routes"}
exec(rsrc, R_)
types_ = ["single_beams", "beam_fields", "blocks", "wedges", "tight_spots", "log_corridors", "homes", "street_front", "stairs"]
meta = {tt: json.load(open(f"{NEW}/{tt}.json"))["tiles"] for tt in types_}
rng = np.random.default_rng(0)
calls, cols = {}, {}
origins = [[[c * 8.0 + 4, r * 8.0 + 4, 0.0] for c in range(len(types_))] for r in range(10)]
for c, tt in enumerate(types_):
    cols[tt] = [c]
    lst = []
    for r in range(10):
        cand = [t for t in meta[tt] if t["level"] == r and t["spawns"]] or [t for t in meta[tt] if t["spawns"]]
        t = cand[rng.integers(len(cand))]
        sp = max(t["spawns"], key=lambda s: s.get("clearance_m", 0.0))
        lst.append((t["index"], [sp["x"], sp["y"], sp.get("z", 0.0)]))
    calls[tt] = lst
table = R_["build_route_table"](calls, meta, cols, 10, origins)
ok.append("route table on the merged set (one layout, 90 cells): %d routes on %d cells" % (sum(len(v) for v in table.values()), len(table)))
# boxes of the same layout, and start clearance on homes / tight spots in world coordinates (consistency of both transforms)
npz = {tt: np.load(f"{NEW}/{tt}.npz") for tt in types_}
idx, cen, half, rot, valid = C["build_cell_boxes"](calls, npz, cols, 10, origins)
for tt in ("homes", "tight_spots", "stairs"):
    c = cols[tt][0]; ds = []
    for r in range(10):
        for start, yaw0, pts in table.get((r, c), []):
            k = idx[r, c]
            ds.append(C["wall_distance_xy"](torch.tensor([start[:2]]), torch.tensor([start[2]]), cen[k:k + 1], half[k:k + 1], rot[k:k + 1], valid[k:k + 1], min_top=0.25).item())
    ds = np.array(ds)
    ok.append("%s: route starts in this layout %d, planar distance to props rising 0.25 m+ : p5 %.2f, share under 0.34 m %.3f" % (tt, len(ds), np.percentile(ds, 5), (ds < 0.34).mean()))
# side law: robot at origin facing +x, waypoint at +y 1 m, side +1 -> walks left, no turn
vx, vy, wz, arr = R_["side_law"](torch.tensor([[0.0, 1.0]]), torch.zeros(1), torch.ones(1), torch.tensor([False]), 0.2, 0.3)
assert abs(vy.item() - 0.3) < 1e-5 and abs(vx.item()) < 1e-5 and abs(wz.item()) < 1e-5
# waypoint ahead with side +1 -> turn right first (bearing must come to +90 deg), no translation
vx, vy, wz, arr = R_["side_law"](torch.tensor([[2.0, 0.0]]), torch.zeros(1), torch.ones(1), torch.tensor([False]), 0.2, 0.3)
assert wz.item() < -0.9 and abs(vx.item()) + abs(vy.item()) < 1e-6
# kinematic sideways walk through 3 waypoints
p, h = torch.zeros(2), torch.tensor(-math.pi / 2)   # facing -y; waypoints along +x are on the left
wps = [torch.tensor([1.0, 0.0]), torch.tensor([2.0, 0.5]), torch.tensor([3.0, 0.5])]
i, steps = 0, 0
while steps < 3000:
    d = (wps[i] - p).view(1, 2)
    last = i == len(wps) - 1
    vx, vy, wz, arr = R_["side_law"](d, h.view(1), torch.ones(1), torch.tensor([last]), 0.2, 0.3)
    if arr.item():
        if last: break
        i += 1; continue
    c, s = torch.cos(h), torch.sin(h)
    p = p + 0.02 * torch.stack([c * vx[0] - s * vy[0], s * vx[0] + c * vy[0]]); h = h + 0.02 * wz[0]; steps += 1
assert i == 2 and torch.norm(wps[2] - p) < 0.2, (i, p)
ok.append("side law: walks sideways through 3 waypoints in %.1f s (kinematic)" % (steps * 0.02))
vy = R_["lateral_vy"](torch.tensor([[0.5, 0.3], [0.5, -0.05], [3.0, 1.0], [-0.5, 0.3]]), torch.zeros(4), 0.2)
assert torch.allclose(vy, torch.tensor([0.2, -0.05, 0.0, 0.0])), vy
ok.append("lateral vy: clipped side offset within 1 m, none behind or far")

# ---------- motion v3 ----------
m3, m2 = np.load(MV3), np.load(MV2)
assert set(m3.files) == set(m2.files) and list(m3["joint_names"]) == list(m2["joint_names"])
n2 = len(m2["joint_pos"])
assert np.array_equal(m3["joint_pos"][:n2], m2["joint_pos"]) and np.allclose(m3["base_pos_w"][:n2], m2["base_pos_w"])
qn = np.linalg.norm(m3["base_quat_w"], axis=1)
assert np.abs(qn - 1).max() < 1e-4
v = np.linalg.norm(np.diff(m3["base_pos_w"][n2 - 1:, :2], axis=0), axis=1) * 50
assert v.max() < 2.0, v.max()
ok.append("motion v3: v2 unchanged in the first %d frames, %d frames added (%.0f s), quaternions unit, planar speed max %.2f m/s at seams" % (n2, len(m3["joint_pos"]) - n2, (len(m3["joint_pos"]) - n2) / 50, v.max()))
print("\n".join("PASS " + s for s in ok))
print("ALL PASS")
