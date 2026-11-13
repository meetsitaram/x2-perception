"""CPU pre-flight of v4k (2026-11-12 22:55). Run by continue_v4k.sh BEFORE v4j is stopped; exit 1 leaves v4j training.
 1. option letters of the launcher's HIKING_X2_OPTS lines are in staged.py's allowed set; staged.py wires option v;
 2. staged.py and the .v4k files compile;
 3. data: terrain dir (with the prop routes: single_beams >= 200 routes), motion yaml + npz (v4: 41,371 frames), and the
    motion reader's joint lookup (every robot joint in joint_names), shapes, finite values, unit quaternions;
 4. slow.py unit tests with stubs (caps, turn_effort, turn_drift);
 5. test_v4j.py on the v4j data (contact.py.v4k is identical to contact.py.v4j; checked).
usage: precheck_v4k.py <continue_v4k.sh> <config/x2 dir of files_instinctlab>"""
import filecmp, json, os, py_compile, re, shutil, subprocess, sys, tempfile, types
import numpy as np, torch

SH, D = sys.argv[1:3]
T = "/mnt/filesystem-g4/tracks/hiking"
fail = []
src = open(SH).read().replace("$T", T)
staged = open(f"{D}/staged.py").read()
m = re.search(r'unknown = opts - set\("([a-z]+)"\)', staged)
allowed = set(m.group(1)) if m else set()
for opts in re.findall(r"HIKING_X2_OPTS=([a-z,]+)", src):
    bad = {o for o in opts.split(",") if o} - allowed
    if bad:
        fail.append("HIKING_X2_OPTS %s: %s not allowed by staged.py (%s)" % (opts, sorted(bad), "".join(sorted(allowed))))
if 'if "v" in opts:' not in staged or "from .slow import" not in staged:
    fail.append("staged.py does not wire option v")
tmp = tempfile.mkdtemp()
for f in ["staged.py"] + [f"{b}.py.v4k" for b in ["extended", "routes", "keep", "contact", "slow"]]:
    try:
        shutil.copy(f"{D}/{f}", f"{tmp}/x.py"); py_compile.compile(f"{tmp}/x.py", doraise=True)
    except Exception as e:
        fail.append("compile %s: %s" % (f, e))
if not filecmp.cmp(f"{D}/contact.py.v4k", f"{D}/contact.py.v4j", shallow=False):
    fail.append("contact.py.v4k differs from contact.py.v4j (test_v4j covers only the latter)")
env = dict(re.findall(r"(HIKING_[A-Z0-9_]+)=(\S+?)[\"\s]", src))
ext, mdir, myaml = env.get("HIKING_EXTENDED_TERRAINS"), env.get("HIKING_X2_MOTION_DIR"), env.get("HIKING_X2_MOTION_YAML")
try:
    nb = sum(len(t.get("routes", [])) for t in json.load(open(f"{ext}/single_beams.json"))["tiles"])
    if nb < 200:
        fail.append("terrain %s: single_beams has %d routes (prop routes missing?)" % (ext, nb))
except Exception as e:
    fail.append("terrain %s: %s" % (ext, e)); nb = -1
yp = f"{mdir}/{myaml}"
nf = -1
try:
    sel = [l.strip()[2:].strip() for l in open(yp) if l.strip().startswith("- ")]
    z = np.load(f"{mdir}/{sel[0]}"); nf = len(z["joint_pos"])
    if nf != 41371:
        fail.append("motion %s: %d frames, expected 41371 (v4)" % (sel[0], nf))
    # 2026-11-12 23:35: the 23:11 check died in AmassMotion._read_retargetted_motion_file
    # (`'left_hip_pitch_joint' is not in list`: joint_names held clip names). Same lines as the reader, on CPU, for every file;
    # the robot's joint names are those of the v2 file that runs 1-4 loaded (= MJCF hinge order).
    isaac = list(np.load("/mnt/filesystem-g4/data/motions/hiking_x2/x2_parkour_motion_without_run_v2_retargetted.npz")["joint_names"])
    for fn in sel:
        raw = np.load(f"{mdir}/{fn}", mmap_mode="r", allow_pickle=True)
        framerate = raw["framerate"].item()
        jn = raw["joint_names"] if isinstance(raw["joint_names"], list) else raw["joint_names"].tolist()
        jp = np.asarray(raw["joint_pos"]); bp = np.asarray(raw["base_pos_w"]); bq = np.asarray(raw["base_quat_w"])
        miss = [j for j in isaac if j not in jn]
        if miss:
            fail.append("motion %s: joint_names lacks %d robot joints (%s ...; first names %s)" % (fn, len(miss), miss[:2], jn[:2]))
            continue
        ids = [jn.index(j) for j in isaac]
        if jp.shape[1] != len(jn) or not (len(jp) == len(bp) == len(bq)) or bp.shape[1] != 3 or bq.shape[1] != 4:
            fail.append("motion %s: shapes joint_pos %s names %d base_pos %s base_quat %s" % (fn, jp.shape, len(jn), bp.shape, bq.shape))
        if not (np.isfinite(jp[:, ids]).all() and np.isfinite(bp).all() and np.isfinite(bq).all()):
            fail.append("motion %s: non-finite values" % fn)
        qn = np.linalg.norm(bq, axis=1)
        if np.abs(qn - 1).max() > 1e-3 or not (10 <= framerate <= 1000):
            fail.append("motion %s: quaternion norm off by %.4f or framerate %s" % (fn, np.abs(qn - 1).max(), framerate))
except Exception as e:
    fail.append("motion %s: %s" % (yp, e))
for k, want in [("HIKING_X2_VX_BACK", "0"), ("HIKING_X2_SLOW_FWD_SHARE", "0.25"), ("HIKING_X2_SLOW_TURN_SHARE", "0.4")]:
    if env.get(k) != want:
        fail.append("launcher %s=%s, expected %s" % (k, env.get(k), want))
# slow.py with stubs
try:
    s = open(f"{D}/slow.py.v4k").read()
    s = s.replace("from isaaclab.utils import configclass\n", "configclass = lambda c: c\n")
    s = s.replace("from .routes import RouteFarTargetPoseVelocityCommand, RouteFarTargetPoseVelocityCommandCfg\n",
                  "class RouteFarTargetPoseVelocityCommand: pass\nclass RouteFarTargetPoseVelocityCommandCfg: pass\n")
    G = {"__name__": "slow"}; exec(s, G)
    inf = float("inf")
    v = torch.tensor([[0.7, 0.1, 0.9], [0.7, 0.0, -0.9], [-0.3, 0.0, 0.2], [0.0, 0.0, 0.0]])
    out = G["cap_commands"](v, torch.tensor([0.2, inf, 0.2, 0.2]), torch.tensor([0.3, 0.15, inf, 0.1]))
    assert torch.allclose(out, torch.tensor([[0.2, 0.1, 0.3], [0.7, 0.0, -0.15], [-0.3, 0.0, 0.2], [0.0, 0.0, 0.0]])), out
    cmd = torch.tensor([[0.0, 0.0, 0.2], [0.0, 0.0, 0.2], [0.5, 0.0, 0.2], [0.0, 0.0, 0.05]])
    rob = types.SimpleNamespace(data=types.SimpleNamespace(root_ang_vel_b=torch.tensor([[0, 0, 0.05], [0, 0, 0.15], [0, 0, 0.0], [0, 0, 0.0]]),
                                                         root_lin_vel_b=torch.tensor([[0.1, 0, 0], [0.0, 0.0, 0], [0.5, 0, 0], [0.2, 0, 0]])))
    envs = types.SimpleNamespace(command_manager=types.SimpleNamespace(get_command=lambda n: cmd), scene={"robot": rob})
    assert G["turn_effort"](envs, "c").tolist() == [1.0, 0.0, 0.0, 0.0]
    assert torch.allclose(G["turn_drift"](envs, "c"), torch.tensor([0.1, 0.0, 0.0, 0.0]))
except Exception as e:
    fail.append("slow.py tests: %r" % (e,))
r = subprocess.run([sys.executable, f"{T}/code/x2_port/v4j/test_v4j.py", D, f"{T}/data/extended_v4j", "/mnt/filesystem-g4/data/terrains/extended_v1",
                    "/mnt/filesystem-g4/data/terrains/extended_v1_2_stairs_inset", f"{T}/data/motion_v3/x2_parkour_motion_without_run_v3_retargetted.npz",
                    "/mnt/filesystem-g4/data/motions/hiking_x2/x2_parkour_motion_without_run_v2_retargetted.npz"], capture_output=True, text=True)
if "ALL PASS" not in r.stdout:
    o = (r.stdout + r.stderr).strip().splitlines()
    fail.append("test_v4j.py: " + (o[-1][:300] if o else "no output"))
print("opts allowed:", "".join(sorted(allowed)), "| terrain:", ext, "single_beams routes", nb, "| motion:", yp, nf, "frames")
print("\n".join("FAIL " + f for f in fail) if fail else "PRECHECK PASS")
sys.exit(1 if fail else 0)
