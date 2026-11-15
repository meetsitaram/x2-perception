"""CPU pre-flight of v5 (2026-11-13). Run by continue_v5.sh BEFORE v4j is stopped; exit 1 leaves v4j training.
 1. v4k content: precheck_v4k.py on continue_v4k.sh must PASS (options, slow.py tests, test_v4j, terrain routes);
 2. option letters of continue_v5.sh are allowed by staged.py.v5; staged.py.v5, fixed.py.v5, style_weights.py, simfault.py and
    supervise_x2_v5.py compile; the launcher has j, the paper-X2 motion dir and yaml;
 3. motion: the retarget file exists, and the motion reader's own steps work on it (joint lookup of every robot joint, shapes,
    finite values, unit quaternions, 50 fps, >= 300 s); the copy's name ends in _retargetted.npz;
 4. style_weights.py: weights per joint, the weighted term on stubs, apply_to_cfg on stub groups (funcs swapped, 23 joints in
    order), and the discriminator drop on the real checkpoint (330 inputs -> 550 dropped; same shapes kept);
 5. simfault.py on stub envs (NaN, overspeed, speed jump, normal) and the failure-term mask.
usage: precheck_v5.py <continue_v5.sh> <config/x2 dir of files_instinctlab> [checkpoint]"""
import glob, json, os, py_compile, re, shutil, subprocess, sys, tempfile, types
import numpy as np, torch

SH, D = sys.argv[1:3]
T = "/mnt/filesystem-g4/tracks/hiking"
X = f"{T}/code/x2_port"
SRC = os.environ.get("V5_SRC_TEST", "/mnt/filesystem-g4/data/motions/hiking_release_x2/parkour_motion_without_run_x2.npz")
V2 = "/mnt/filesystem-g4/data/motions/hiking_x2/x2_parkour_motion_without_run_v2_retargetted.npz"
fail, info = [], []
src = open(SH).read().replace("$T", T)
env = dict(re.findall(r"(HIKING_[A-Z0-9_]+)=(\S+?)[\"\s]", src))

# 1
r = subprocess.run([sys.executable, f"{X}/v4j/precheck_v4k.py", f"{X}/continue_v4k.sh", D], capture_output=True, text=True)
if "PRECHECK PASS" not in r.stdout:
    fail.append("precheck_v4k.py: " + " | ".join(l for l in r.stdout.splitlines() if l.startswith("FAIL"))[:300])
# 2
staged = open(f"{D}/staged.py.v5").read()
m = re.search(r'unknown = opts - set\("([a-z]+)"\)', staged)
allowed = set(m.group(1)) if m else set()
for opts in re.findall(r"HIKING_X2_OPTS=([a-z,]+)", src):
    bad = {o for o in opts.split(",") if o} - allowed
    if bad:
        fail.append("HIKING_X2_OPTS %s: %s not allowed by staged.py.v5" % (opts, sorted(bad)))
    if "j" not in opts.split(","):
        fail.append("HIKING_X2_OPTS %s lacks j" % opts)
tmp = tempfile.mkdtemp()
for f in [f"{D}/staged.py.v5", f"{D}/fixed.py.v5", f"{D}/style_weights.py", f"{D}/simfault.py", f"{X}/v5/supervise_x2_v5.py"]:
    try:
        shutil.copy(f, f"{tmp}/x.py"); py_compile.compile(f"{tmp}/x.py", doraise=True)
    except Exception as e:
        fail.append("compile %s: %s" % (f, e))
if env.get("HIKING_X2_MOTION_DIR") != f"{T}/data/motion_paper_x2" or env.get("HIKING_X2_MOTION_YAML") != "parkour_motion_without_run_x2.yaml":
    fail.append("launcher motion %s/%s" % (env.get("HIKING_X2_MOTION_DIR"), env.get("HIKING_X2_MOTION_YAML")))
# 3
isaac = list(np.load(V2)["joint_names"])
sys.path.insert(0, D)
import style_weights as SW  # noqa: E402
if not os.path.exists(SRC):
    fail.append("motion: %s does not exist yet (retarget not delivered)" % SRC)
else:
    try:
        raw = np.load(SRC, mmap_mode="r", allow_pickle=True)
        keys = set(raw.keys())
        need = {"framerate", "joint_names", "joint_pos", "base_pos_w", "base_quat_w"}
        if not need <= keys:
            fail.append("motion: keys %s lack %s" % (sorted(keys), sorted(need - keys)))
        else:
            framerate = raw["framerate"].item()
            jn = raw["joint_names"] if isinstance(raw["joint_names"], list) else raw["joint_names"].tolist()
            jp, bp, bq = np.asarray(raw["joint_pos"]), np.asarray(raw["base_pos_w"]), np.asarray(raw["base_quat_w"])
            miss = [j for j in isaac if j not in jn]
            if miss:
                fail.append("motion: joint_names lacks %d robot joints (%s; first names %s)" % (len(miss), miss[:2], jn[:2]))
            else:
                ids = [jn.index(j) for j in isaac]
                if jp.shape[1] != len(jn) or not (len(jp) == len(bp) == len(bq)) or bp.shape[1] != 3 or bq.shape[1] != 4:
                    fail.append("motion: shapes joint_pos %s names %d base_pos %s base_quat %s" % (jp.shape, len(jn), bp.shape, bq.shape))
                elif not (np.isfinite(jp[:, ids]).all() and np.isfinite(bp).all() and np.isfinite(bq).all()):
                    fail.append("motion: non-finite values")
                else:
                    qe = float(np.abs(np.linalg.norm(bq, axis=1) - 1).max())
                    if qe > 1e-3:
                        fail.append("motion: quaternion norm off by %.4f" % qe)
                    if abs(framerate - 50) > 1e-6:
                        fail.append("motion: framerate %s, expected 50" % framerate)
                    if len(jp) / framerate < 300:
                        fail.append("motion: %.0f s, expected about 380 s" % (len(jp) / framerate))
                    sj = [jn.index(j) for j in SW.STYLE_JOINTS]
                    info.append("motion %d frames %.0f s, pelvis z median %.3f, style joints std %.3f" % (
                        len(jp), len(jp) / framerate, float(np.median(bp[:, 2])), float(jp[:, sj].std())))
    except Exception as e:
        fail.append("motion %s: %r" % (SRC, e))
# 4
try:
    assert len(SW.STYLE_JOINTS) == 23 and all(j in isaac for j in SW.STYLE_JOINTS)
    w = [SW.joint_weight(j) for j in isaac]
    assert [SW.joint_weight(j) for j in ("left_knee_joint", "waist_pitch_joint", "right_elbow_joint", "left_shoulder_roll_joint",
                                         "left_wrist_yaw_joint", "head_pitch_joint")] == [1.0, 1.0, 0.3, 0.3, 0.0, 0.0]
    cfgo = types.SimpleNamespace(name="robot", joint_names=list(SW.STYLE_JOINTS))
    out = SW._apply(torch.ones(4, 23), cfgo)
    assert torch.allclose(out[0], torch.tensor([1.0] * 15 + [0.3] * 8)), out[0]

    class SEC:
        def __init__(self, name, joint_names=None, preserve_order=False):
            self.name, self.joint_names, self.preserve_order = name, joint_names, preserve_order
    def mk(fname):
        f = lambda env, asset_cfg: None
        f.__name__ = fname
        return f
    term = lambda fname, name: types.SimpleNamespace(func=mk(fname), params={"asset_cfg": SEC(name)})
    obs = types.SimpleNamespace(
        amp_policy=types.SimpleNamespace(joint_pos_rel=term("joint_pos_rel", "robot"), joint_vel=term("joint_vel_rel", "robot")),
        amp_reference=types.SimpleNamespace(joint_pos_rel=term("joint_pos_rel_reference_as_state", "motion_reference"),
                                            joint_vel=term("joint_vel_rel_reference_as_state", "motion_reference")))
    SW.apply_to_cfg(types.SimpleNamespace(observations=obs), SEC)
    assert obs.amp_policy.joint_pos_rel.func is SW.joint_pos_rel_w and obs.amp_reference.joint_vel.func is SW.joint_vel_rel_reference_w
    assert obs.amp_reference.joint_pos_rel.params["asset_cfg"].joint_names == SW.STYLE_JOINTS
    assert obs.amp_reference.joint_pos_rel.params["asset_cfg"].preserve_order and obs.amp_reference.joint_pos_rel.params["asset_cfg"].name == "motion_reference"
    ck = sys.argv[3] if len(sys.argv) > 3 else sorted(glob.glob(f"{T}/runs/x2_parkour_v4/*/model_*.pt"), key=os.path.getmtime)[-1]
    sd = torch.load(ck, weights_only=True, map_location="cpu")
    old_in = sd["discriminator"]["model.model.0.weight"].shape[1]
    new_in = 10 * (3 + 23 + 23 + 3 + 3)
    assert old_in == 330, old_in
    own = {k: v.clone() for k, v in sd["discriminator"].items()}
    own["model.model.0.weight"] = torch.zeros(own["model.model.0.weight"].shape[0], new_in)
    d1 = dict(sd); bad = SW.drop_mismatched_discriminator(own, d1)
    assert bad and "discriminator" not in d1 and "discriminator_optimizer" not in d1 and "model_state_dict" in d1
    d2 = dict(sd); assert SW.drop_mismatched_discriminator(sd["discriminator"], d2) == [] and "discriminator" in d2
    info.append("discriminator %d -> %d inputs: dropped on load (%s)" % (old_in, new_in, os.path.basename(ck)))
except Exception as e:
    fail.append("style_weights tests: %r" % (e,))
# 5
try:
    import simfault as SF
    def envs(v, root=None, epl=5):
        n = v.shape[0]
        data = types.SimpleNamespace(body_lin_vel_w=v, root_state_w=torch.zeros(n, 13) if root is None else root, joint_pos=torch.zeros(n, 31))
        return types.SimpleNamespace(common_step_counter=0, scene={"robot": types.SimpleNamespace(data=data)},
                                     episode_length_buf=torch.full((n,), epl))
    v = torch.zeros(4, 3, 3); v[1, 0, 0] = 60.0
    root = torch.zeros(4, 13); root[2, 0] = float("nan")
    e = envs(v, root)
    assert SF.sim_fault(e).tolist() == [False, True, True, False]
    e.common_step_counter = 1; v2 = v.clone(); v2[3, 2, 1] = 25.0; v2[1, 0, 0] = 0.0
    e.scene["robot"].data.body_lin_vel_w = v2; e.scene["robot"].data.root_state_w = torch.zeros(4, 13)
    assert SF.sim_fault(e).tolist() == [False, True, False, True]  # 1: 60 -> 0 is a jump; 3: 0 -> 25 is a jump
    w = SF._unless_sim_fault(lambda env: torch.tensor([True, True, False, True]))
    assert w(e).tolist() == [True, False, False, False]
except Exception as e:
    fail.append("simfault tests: %r" % (e,))
for l in info:
    print(l)
print("\n".join("FAIL " + f for f in fail) if fail else "PRECHECK PASS")
sys.exit(1 if fail else 0)
