"""Step 2 (prepared 2026-11-12, coordinator decision 05:00): fresh run with the default leg pose = reference mean. Default off.

--default_from_ref : x2.build replaces the 12 leg default positions by config/default_ref_v1.json (reference mean of the flat
                     moving windows of amp_v2, made left/right symmetric). Every action offset, dof_dev, stand_still and the
                     reset pose follow; the motion prior and the walking starts (stored as q - old default) are shifted by
                     (old default - new default) before mirroring, so they describe the same absolute poses.
                     Read by x2.build from X2_DEFAULT_FROM_REF=1. A from-scratch change: checkpoints are not interchangeable.
Backups: ../_bak_20261112/*.pre_step2
"""
import json
import pathlib
import shutil

C = pathlib.Path(__file__).resolve().parents[1]
BAK = C.parent / "_bak_20261112"

# reference means (eval/port_audit.json, flat, moving windows); symmetric = mean of right and mirrored left (S_LEG 1,-1,-1,1,1,-1)
REF = {"hip_pitch": -0.081, "hip_roll": 0.0325, "hip_yaw": -0.2095, "knee": 0.3195, "ankle_pitch": -0.060, "ankle_roll": -0.0525}
SIGN = {"hip_pitch": 1, "hip_roll": -1, "hip_yaw": -1, "knee": 1, "ankle_pitch": 1, "ankle_roll": -1}
vals = {}
for j, v in REF.items():
    vals[f"right_{j}_joint"] = v
    vals[f"left_{j}_joint"] = round(SIGN[j] * v, 4)
(C / "config/default_ref_v1.json").write_text(json.dumps(dict(source="eval/port_audit.json reference_vs_default.flat mean, L/R symmetrised", default_q=vals), indent=1))


def edit(rel, pairs):
    p = C / rel
    s = p.read_text()
    shutil.copy(p, BAK / (p.name + ".pre_step2"))
    for old, new in pairs:
        assert s.count(old) == 1, (rel, old)
        s = s.replace(old, new)
    p.write_text(s)


edit("ssr/x2.py", [
    ("    qd_max: np.ndarray\n",
     "    qd_max: np.ndarray\n    default_q_plant: np.ndarray = None     # the plant's default pose (differs from default_q under X2_DEFAULT_FROM_REF)\n"),
    ("    return X2(m, names, np.array([names.index(n) for n in NAMES21]), kp, kd, eff,\n"
     "              np.array([plant.action_scale_for(n) for n in names]), np.array([plant.default_pos_for(n) for n in names]),\n"
     "              rng[:, 0].copy(), rng[:, 1].copy(), qd_max)",
     "    dq_plant = np.array([plant.default_pos_for(n) for n in names])\n"
     "    dq = dq_plant.copy()\n"
     "    if _os.environ.get(\"X2_DEFAULT_FROM_REF\") == \"1\":                     # deviation (step 2, 2026-11-12): default leg pose = reference mean\n"
     "        _ref = json.load(open(ROOT / \"config/default_ref_v1.json\"))[\"default_q\"]\n"
     "        for _n, _v in _ref.items():\n"
     "            dq[names.index(_n)] = _v\n"
     "    return X2(m, names, np.array([names.index(n) for n in NAMES21]), kp, kd, eff,\n"
     "              np.array([plant.action_scale_for(n) for n in names]), dq,\n"
     "              rng[:, 0].copy(), rng[:, 1].copy(), qd_max, dq_plant)"),
])

edit("ssr/train.py", [
    ("    def __init__(self, device, min_speed=0.0, file=\"data/amp.npz\", balance=False):\n        z = np.load(ROOT / file)\n",
     "    def __init__(self, device, min_speed=0.0, file=\"data/amp.npz\", balance=False, q_shift=None):\n        z = np.load(ROOT / file)\n"),
    ("            f = torch.as_tensor(z[name], device=device, dtype=torch.float32)\n            f = torch.cat((f, M.amp(f)))",
     "            f = torch.as_tensor(z[name], device=device, dtype=torch.float32)\n"
     "            if q_shift is not None:                         # step 2: frames store q - plant default; re-express against the new default\n"
     "                f[:, 9:30] += torch.as_tensor(q_shift, device=device, dtype=torch.float32)\n"
     "            f = torch.cat((f, M.amp(f)))"),
    ("    if a.excl_pelvis_hiproll:\n        os.environ[\"X2_EXCL_PELVIS_HIPROLL\"] = \"1\"\n",
     "    if a.excl_pelvis_hiproll:\n        os.environ[\"X2_EXCL_PELVIS_HIPROLL\"] = \"1\"\n"
     "    if a.default_from_ref:\n        os.environ[\"X2_DEFAULT_FROM_REF\"] = \"1\"\n"),
    ("    amp = AMPData(dev, a.prior_min_speed, a.amp_file, a.prior_balance)\n",
     "    _shift = None\n"
     "    if a.default_from_ref:\n"
     "        _x = env.x2\n"
     "        _shift = (_x.default_q_plant - _x.default_q)[_x.idx21]\n"
     "        print(\"default leg pose from the reference mean; prior shifted by\", [round(float(v_), 3) for v_ in _shift[:12]], flush=True)\n"
     "    amp = AMPData(dev, a.prior_min_speed, a.amp_file, a.prior_balance, q_shift=_shift)\n"),
    ("    ap.add_argument(\"--excl_pelvis_hiproll\",",
     "    ap.add_argument(\"--default_from_ref\", action=\"store_true\", help=\"deviation (step 2, fresh runs only): default leg pose = reference mean\")\n"
     "    ap.add_argument(\"--excl_pelvis_hiproll\","),
])
print("patched")
