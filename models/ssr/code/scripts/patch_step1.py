"""Step 1 after the r6b 2,000 gate (coordinator, 2026-11-12 05:00): two listed deviations, both default off.

--excl_pelvis_hiproll : MuJoCo body exclusions pelvis / left_hip_roll_link and pelvis / right_hip_roll_link (port audit point 4:
                        their hulls overlap in 89 % of reference stepping frames, up to 11 mm). Read by x2.build from X2_EXCL_PELVIS_HIPROLL.
--dof_dev_ref         : the dof_dev penalty measures |q - q_ref| with q_ref = reference (amp_v2 flat, moving windows) mean for
                        the 12 leg joints, the default pose for waist and arms (port audit point 2). Action offsets unchanged.
Backups: ../_bak_20261112/*.pre_step1
"""
import pathlib
import shutil

C = pathlib.Path(__file__).resolve().parents[1]
BAK = C.parent / "_bak_20261112"


def edit(rel, pairs):
    p = C / rel
    s = p.read_text()
    shutil.copy(p, BAK / (p.name + ".pre_step1"))
    for old, new in pairs:
        assert s.count(old) == 1, (rel, old)
        s = s.replace(old, new)
    p.write_text(s)


edit("ssr/x2.py", [(
    "    m = spec.compile()\n    jid = ",
    "    import os as _os\n"
    "    if _os.environ.get(\"X2_EXCL_PELVIS_HIPROLL\") == \"1\":                   # deviation (step 1, 2026-11-12): hull overlap in reference poses\n"
    "        for _s in (\"left\", \"right\"):\n"
    "            spec.add_exclude(bodyname1=\"pelvis\", bodyname2=f\"{_s}_hip_roll_link\")\n"
    "    m = spec.compile()\n    jid = ")])

edit("ssr/env.py", [
    ("        self.cnt = {}                           # raw counts for the log",
     "        self.q_dev_ref = None                   # step 1 (not in the paper): dof_dev measured from this pose (21) instead of the default\n"
     "        self.cnt = {}                           # raw counts for the log"),
    ("            dof_dev=-1.8 * (1.0 - 0.75 * self.boot) * q_rel.abs().sum(-1) / N21f,",
     "            dof_dev=-1.8 * (1.0 - 0.75 * self.boot) * (q_rel if self.q_dev_ref is None else s[\"q\"] - self.q_dev_ref).abs().sum(-1) / N21f,"),
])

edit("ssr/train.py", [
    ("    env = SSREnv(a.num_envs, dev, seed=a.seed, camera=a.camera, map_memory=a.map_memory)\n",
     "    if a.excl_pelvis_hiproll:\n"
     "        os.environ[\"X2_EXCL_PELVIS_HIPROLL\"] = \"1\"\n"
     "    env = SSREnv(a.num_envs, dev, seed=a.seed, camera=a.camera, map_memory=a.map_memory)\n"
     "    if a.dof_dev_ref:\n"
     "        _pa = json.load(open(ROOT.parent / \"eval/port_audit.json\"))[\"reference_vs_default\"][\"flat\"]\n"
     "        _ref = env.q0.clone()\n"
     "        for _i, _n in enumerate(__import__(\"ssr.x2\", fromlist=[\"NAMES21\"]).NAMES21[:12]):\n"
     "            _ref[_i] = float(_pa[_n][\"mean\"])\n"
     "        env.q_dev_ref = _ref\n"
     "        print(\"dof_dev measured from the reference mean leg pose:\", [round(float(v_), 3) for v_ in _ref[:12]], flush=True)\n"
     "    if a.excl_pelvis_hiproll:\n"
     "        print(\"pelvis / hip-roll contacts excluded: nexclude =\", env.x2.model.nexclude, flush=True)\n"),
    ("    ap.add_argument(\"--disc_mask_arms\",",
     "    ap.add_argument(\"--excl_pelvis_hiproll\", action=\"store_true\", help=\"deviation (step 1): no pelvis / hip-roll link contacts\")\n"
     "    ap.add_argument(\"--dof_dev_ref\", action=\"store_true\", help=\"deviation (step 1): dof_dev from the reference mean leg pose\")\n"
     "    ap.add_argument(\"--disc_mask_arms\","),
])
print("patched")
