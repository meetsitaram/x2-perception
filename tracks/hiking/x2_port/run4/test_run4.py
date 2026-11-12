"""CPU checks of run 4's fixed.py constants (no Isaac). python test_run4.py <fixed.py> <agibot_x2.py> <urdf> <reference npz>"""
import re, sys, xml.etree.ElementTree as ET
import numpy as np

fx, asset, urdf, npz = sys.argv[1:5]
src = open(fx).read()
ns = {}
exec(src[src.index("SIDES ="):src.index("def spawn_x2_run4")], ns)
a = open(asset).read()
ans = {"os": __import__("os")}
exec(a[a.index("# Joint order Isaac Lab"):a.index("X2_LINKS")], ans)
J = ans["X2_ISAAC_JOINTS"]
ACT = [j for j in J if not j.startswith("head_")]


def matches(table, name):
    return [k for k in table if re.fullmatch(k, name)]


# every action joint has exactly one scale; no joint matches two default-pose keys
for j in ACT:
    assert len(matches(ns["RUN4_ACTION_SCALE"], j)) == 1, j
for j in J:
    assert len(matches(ns["RUN4_DEFAULT_POSE"], j)) <= 1, j
scale = np.array([ns["RUN4_ACTION_SCALE"][matches(ns["RUN4_ACTION_SCALE"], j)[0]] for j in ACT])
default = np.array([ns["RUN4_DEFAULT_POSE"][matches(ns["RUN4_DEFAULT_POSE"], j)[0]] if matches(ns["RUN4_DEFAULT_POSE"], j) else 0.0 for j in J])
assert set(ns["LEG_JOINTS"]) <= set(J) and len(ns["LEG_JOINTS"]) == 12

# filtered pairs: links exist and are not parent-child (those PhysX skips anyway)
root = ET.parse(urdf).getroot()
links = {l.get("name") for l in root.findall("link")}
parent = {j.find("child").get("link"): j.find("parent").get("link") for j in root.findall("joint")}
for p, q in ns["RUN4_EXTRA_FILTERED_PAIRS"]:
    assert p in links and q in links, (p, q)
    assert parent.get(p) != q and parent.get(q) != p, ("parent-child", p, q)
print("filtered pairs:", len(ns["RUN4_EXTRA_FILTERED_PAIRS"]), "all links exist, none parent-child")

# default pose against the reference mean (legs, symmetrised), waist 0
z = np.load(npz, allow_pickle=True)
names = [str(n) for n in z["joint_names"]]
jp = np.asarray(z["joint_pos"], float)
R = np.stack([jp[:, names.index(j)] if j in names else np.zeros(len(jp)) for j in J], 1)
mean = R.mean(0)
for k in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"):
    sgn = -1 if ("roll" in k or "yaw" in k) else 1
    l, r = J.index(f"left_{k}_joint"), J.index(f"right_{k}_joint")
    want = (mean[l] + sgn * mean[r]) / 2
    assert abs(default[l] - want) < 0.002 and abs(default[r] - sgn * want) < 0.002, (k, default[l], want)
for k in ("waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint"):
    assert default[J.index(k)] == 0.0

# action-rate cost of the reference gait (sum over the 29 action joints of (delta action)^2 per 50 Hz step), old vs new scale
fr = float(z["framerate"]) if "framerate" in z.files else 50.0
step = max(1, int(round(fr / 50.0)))
dq = np.diff(R[::step], axis=0)
dq = dq[np.abs(dq).max(1) < 0.5]  # drop clip seams
idx = [J.index(j) for j in ACT]
exec(a[a.index("ARMATURE_PF52"):a.index("def _stiffness")], ans)
W = ans["NATURAL_FREQ"]


def plant(j):
    return next(v for k, v in ans["_PLANT"].items() if re.fullmatch(k, j))


old = np.array([0.25 * plant(j)[1] / (plant(j)[0] * W ** 2) for j in ACT])
cost_old = ((dq[:, idx] / old) ** 2).sum(1).mean()
cost_new = ((dq[:, idx] / scale) ** 2).sum(1).mean()
legs = [ACT.index(j) for j in ns["LEG_JOINTS"]]
lo = ((dq[:, idx][:, legs] / old[legs]) ** 2).sum(1).mean(); ln = ((dq[:, idx][:, legs] / scale[legs]) ** 2).sum(1).mean()
print("reference gait at %.0f fps -> 50 Hz steps: action-rate raw sum per step, all 29 joints: old scale %.3f, new %.3f (x%.2f); legs only: old %.3f, new %.3f (x%.2f)"
      % (fr, cost_old, cost_new, cost_new / cost_old, lo, ln, ln / lo))
print("old scale hip pitch %.3f knee %.3f ankle pitch %.3f -> new %.3f %.3f %.3f" % (
    old[ACT.index("left_hip_pitch_joint")], old[ACT.index("left_knee_joint")], old[ACT.index("left_ankle_pitch_joint")],
    scale[ACT.index("left_hip_pitch_joint")], scale[ACT.index("left_knee_joint")], scale[ACT.index("left_ankle_pitch_joint")]))
print("run-4 constants: all checks passed")
