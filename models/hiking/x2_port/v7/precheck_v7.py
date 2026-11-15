"""CPU pre-flight of v7 (2026-11-13; = precheck_v6.py with the motion check for the paper reference v4). Run by continue_v7.sh BEFORE v5b is stopped; exit 1 leaves v5b training.
v6 = v5b + (1) option t with a speed-dependent cap (HIKING_X2_AIR_TD_CAP=ref, variants/staged.py.v6; weight 140 kept) and
(2) the paper reference v4 (torsoBase form read correctly, data/motion_v7). Slow commands unchanged (user, 2026-11-13).
 1. launcher (ENVV of continue_v6.sh): options allowed by staged.py.v6; AIR_TD_CAP=ref, AIR_TD_WEIGHT=140; every other
    HIKING_ variable equal to v5b's launcher (start_x2_parkour_v5b.sh) except the motion dir / yaml;
 2. staged.py.v6 compiles, differs from the live staged.py only by the v6 patch, and its refcap term passes plain-tensor tests;
 3. motion: data/motion_v7 copy = the v4 source (md5), the yaml lists it, the motion reader's steps (as precheck_v5b) pass,
    no waist_pitch frame within 0.005 rad of its lower limit (-0.314), 18982 frames, joint order = v2, legs within 0.05 rad mean of v2;
 4. one code source: every file of files_instinctlab equals its installed copy in InstinctLab, and the installed x2 config dir
    has no file the source lacks.
usage: precheck_v6.py <continue_v6.sh>"""
import ast, hashlib, os, py_compile, re, shutil, subprocess, sys, tempfile, types
import numpy as np, torch

SH = sys.argv[1]
T = "/mnt/filesystem-g4/tracks/hiking"
X = f"{T}/code/x2_port"
F = f"{X}/files_instinctlab"
I = f"{T}/code/hiking-in-the-wild/InstinctLab/source/instinctlab/instinctlab"
D = f"{F}/tasks/parkour/config/x2"
V6 = f"{X}/variants/staged.py.v6"
MD = f"{T}/data/motion_v7"
SRC = f"{MD}/parkour_motion_without_run_x2_v4_retargetted.npz"
V4 = "/mnt/filesystem-g4/data/motions/hiking_release_x2/parkour_motion_without_run_x2_v4.npz"
V2 = "/mnt/filesystem-g4/data/motions/hiking_release_x2/parkour_motion_without_run_x2_v2.npz"
ISAAC = "/mnt/filesystem-g4/data/motions/hiking_x2/x2_parkour_motion_without_run_v2_retargetted.npz"
fail, info = [], []


def envof(text):
    return dict(re.findall(r"(HIKING_[A-Z0-9_]+)=(\S+?)[\"\s]", text.replace("$T", T).replace("$S", f"{T}/runs/x2_parkour_v4/_supervisor")))


# 1
m = re.search(r'^ENVV="([^"]*)"', open(SH).read(), re.M)
env = envof(m.group(1) + " ") if m else {}
old = envof(open(f"{X}/start_x2_parkour_v5b.sh").read())
staged = open(V6).read()
mm = re.search(r'unknown = opts - set\("([a-z]+)"\)', staged)
allowed = set(mm.group(1)) if mm else set()
bad = set(env.get("HIKING_X2_OPTS", "").split(",")) - allowed
if not env or bad:
    fail.append("launcher: no ENVV or options %s not allowed" % sorted(bad))
if env.get("HIKING_X2_AIR_TD_CAP") != "ref" or env.get("HIKING_X2_AIR_TD_WEIGHT") != "140":
    fail.append("launcher: AIR_TD_CAP %s weight %s (want ref, 140)" % (env.get("HIKING_X2_AIR_TD_CAP"), env.get("HIKING_X2_AIR_TD_WEIGHT")))
skip = {"HIKING_X2_AIR_TD_CAP", "HIKING_X2_MOTION_DIR", "HIKING_X2_MOTION_YAML"}
diff = sorted(k for k in set(env) | set(old) if k not in skip and env.get(k) != old.get(k))
if diff:
    fail.append("launcher differs from v5b in %s" % diff)
if env.get("HIKING_X2_MOTION_DIR") != MD or env.get("HIKING_X2_MOTION_YAML") != "parkour_motion_without_run_x2_v4.yaml":
    fail.append("launcher motion %s/%s" % (env.get("HIKING_X2_MOTION_DIR"), env.get("HIKING_X2_MOTION_YAML")))
info.append("launcher: same as v5b except AIR_TD_CAP=ref and motion v4 (slow cmds: FWD_SHARE %s SLOW_VX %s VX_MIN %s)" % (
    env.get("HIKING_X2_SLOW_FWD_SHARE"), env.get("HIKING_X2_SLOW_VX"), env.get("HIKING_X2_VX_MIN")))
# 2
tmp = tempfile.mkdtemp()
try:
    shutil.copy(V6, f"{tmp}/x.py"); py_compile.compile(f"{tmp}/x.py", doraise=True)
    shutil.copy(f"{X}/v5/supervise_x2_v5.py", f"{tmp}/y.py"); py_compile.compile(f"{tmp}/y.py", doraise=True)
except Exception as e:
    fail.append("compile: %s" % e)
r = subprocess.run([sys.executable, f"{X}/v6/mk_v6.py", f"{tmp}/regen.py"], capture_output=True, text=True)
if r.returncode or open(f"{tmp}/regen.py").read() != staged:
    fail.append("staged.py.v6 is not the live staged.py + the v6 patch (mk_v6.py rc %d)" % r.returncode)
try:
    tree = ast.parse(staged)
    code = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in ("_ref_cadence", "feet_air_touchdown_refcap"))
            or (isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") in ("_REF_V", "_REF_F"))]
    g = {"torch": torch, "SceneEntityCfg": object}
    exec(compile(ast.Module(code, []), "staged_v6", "exec"), g)
    f = g["_ref_cadence"](torch.tensor([0.0, 0.3, 0.4, 0.5, 0.7, 1.0]))
    assert torch.allclose(f, torch.tensor([1.4, 1.4, 1.65, 1.9, 2.0, 2.0])), f

    class Sen:
        data = types.SimpleNamespace(last_air_time=torch.tensor([[0.2, 0.6], [0.6, 0.6], [0.03, 0.4]]))

        def compute_first_contact(self, dt):
            return torch.tensor([[1., 1.], [1., 0.], [1., 1.]])
    e = types.SimpleNamespace(scene=types.SimpleNamespace(sensors={"c": Sen()}), step_dt=0.02,
                              command_manager=types.SimpleNamespace(get_command=lambda n: torch.tensor([[0.7, 0, 0], [0.3, 0, 0], [0.0, 0, 0.05]])))
    p = g["feet_air_touchdown_refcap"](e, "b", types.SimpleNamespace(name="c", body_ids=[0, 1]))
    assert torch.allclose(p, torch.tensor([0.60, 0.55, 0.0])), p
    info.append("refcap term: f_ref 1.4/1.65/1.9/2.0 at 0.3/0.4/0.5/0.7 m/s; caps 0.664/0.476/0.450 s; tests pass")
except Exception as ex:
    fail.append("refcap tests: %r" % (ex,))
# 3
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
if not os.path.exists(SRC) or md5(SRC) != md5(V4):
    fail.append("motion: %s missing or not the v4 source" % SRC)
elif "parkour_motion_without_run_x2_v4_retargetted.npz" not in open(f"{MD}/parkour_motion_without_run_x2_v4.yaml").read():
    fail.append("motion: yaml does not list the v4 file")
else:
    try:
        isaac = list(np.load(ISAAC)["joint_names"])
        raw = np.load(SRC, mmap_mode="r", allow_pickle=True)
        need = {"framerate", "joint_names", "joint_pos", "base_pos_w", "base_quat_w"}
        assert need <= set(raw.keys()), sorted(raw.keys())
        fr = raw["framerate"].item()
        jn = raw["joint_names"] if isinstance(raw["joint_names"], list) else raw["joint_names"].tolist()
        jp, bp, bq = np.asarray(raw["joint_pos"]), np.asarray(raw["base_pos_w"]), np.asarray(raw["base_quat_w"])
        miss = [j for j in isaac if j not in jn]
        assert not miss, "joint_names lacks %s" % miss[:3]
        ids = [jn.index(j) for j in isaac]
        assert jp.shape[1] == len(jn) and len(jp) == len(bp) == len(bq) and bp.shape[1] == 3 and bq.shape[1] == 4
        assert np.isfinite(jp[:, ids]).all() and np.isfinite(bp).all() and np.isfinite(bq).all(), "non-finite"
        assert float(np.abs(np.linalg.norm(bq, axis=1) - 1).max()) < 1e-3, "quaternion norm"
        assert abs(fr - 50) < 1e-6 and len(jp) == 18982, (fr, len(jp))
        v2 = np.load(V2)
        assert list(v2["joint_names"]) == jn, "joint order differs from v2"
        w = jp[:, jn.index("waist_pitch_joint")]
        nlo, nhi = int((w < -0.314 + 0.005).sum()), int((w > 0.314 - 0.005).sum())
        assert nlo == 0, "%d waist frames at the -0.314 (backward) limit" % nlo
        legs = [jn.index(j) for j in jn if any(k in j for k in ("hip", "knee", "ankle"))]
        dl = float(np.abs(np.asarray(v2["joint_pos"])[:, legs] - jp[:, legs]).mean())
        assert dl < 0.05, "leg joints moved %.3f rad on average vs v2" % dl
        info.append("motion v4: %d frames, joint order = v2, waist at -limit 0 (v2: %d), at +18 deg limit %d (accepted by the user), legs vs v2 mean |d| %.4f rad, waist pitch mean |d| %.3f rad" % (
            len(jp), int((np.asarray(v2["joint_pos"])[:, jn.index("waist_pitch_joint")] < -0.309).sum()), nhi, dl,
            float(np.abs(np.asarray(v2["joint_pos"])[:, jn.index("waist_pitch_joint")] - w).mean())))
    except Exception as ex:
        fail.append("motion %s: %r" % (SRC, ex))
# 4
nf = 0
for root, _, files in os.walk(F):
    if "__pycache__" in root:
        continue
    for fn in files:
        rel = os.path.relpath(os.path.join(root, fn), F); nf += 1
        dst = os.path.join(I, rel)
        if not os.path.exists(dst) or md5(dst) != md5(os.path.join(F, rel)):
            fail.append("code tree: installed %s differs from files_instinctlab" % rel)
extra = [fn for fn in os.listdir(f"{I}/tasks/parkour/config/x2") if fn != "__pycache__" and not os.path.exists(f"{D}/{fn}")]
if extra:
    fail.append("code tree: installed x2 config has files not in the source: %s" % extra)
info.append("code tree: %d source files = installed copies, no extra files" % nf)
for l in info:
    print(l)
print("\n".join("FAIL " + f for f in fail) if fail else "PRECHECK PASS")
sys.exit(1 if fail else 0)
