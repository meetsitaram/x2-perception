"""2026-11-12 23:35: fix the v4k motion file (joint_names held clip names) and harden the pre-flight and launcher."""
import os, re, shutil
import numpy as np

T = "/mnt/filesystem-g4/tracks/hiking"
X = f"{T}/code/x2_port"
M4 = f"{T}/data/motion_v4"
V2 = "/mnt/filesystem-g4/data/motions/hiking_x2/x2_parkour_motion_without_run_v2_retargetted.npz"
TR = f"{T}/_trash/2026-11-12"
os.makedirs(TR, exist_ok=True)

# 1. append_motion_v4.py: the joint-name list `names` was reused for clip keys (line 49) and saved as joint_names.
p = f"{M4}/append_motion_v4.py"
s = open(p).read()
shutil.copy(p, f"{TR}/append_motion_v4.py.bad_jointnames")
s = s.replace("names = [mujoco.mj_id2name(", "JNAMES = [mujoco.mj_id2name(", 1)
s = s.replace('assert list(v2["joint_names"]) == names,', 'assert list(v2["joint_names"]) == JNAMES,', 1)
s = s.replace('zero = [i for i, n in enumerate(names) if', 'zero = [i for i, n in enumerate(JNAMES) if', 1)
s = s.replace("joint_names=np.array(names)", "joint_names=np.array(JNAMES)", 1)
assert s.count("JNAMES") == 4, s.count("JNAMES")
open(p, "w").write(s)

# 2. the npz: same arrays, joint_names from the v2 file (append_motion_v4.py asserts v2 order == MJCF hinge order).
f = f"{M4}/x2_parkour_motion_without_run_v4_retargetted.npz"
z = dict(np.load(f, allow_pickle=False))
jn = np.load(V2)["joint_names"]
assert z["joint_pos"].shape[1] == len(jn) == 31
bad = f"{TR}/x2_parkour_motion_without_run_v4_retargetted.npz.bad_jointnames"
shutil.move(f, bad)
z["joint_names"] = jn
np.savez(f, **z)
chk = np.load(f)
assert list(chk["joint_names"]) == list(jn) and np.array_equal(chk["joint_pos"], np.load(bad)["joint_pos"])
print("npz fixed:", len(chk["joint_pos"]), "frames, joint_names", list(chk["joint_names"][:2]), "...")

# 3. pre-flight: emulate AmassMotion._read_retargetted_motion_file for every file in the yaml.
p = f"{X}/v4j/precheck_v4k.py"
s = open(p).read()
shutil.copy(p, f"{TR}/precheck_v4k.py.before_jointcheck")
old = """    z = np.load(f"{mdir}/{sel[0]}"); nf = len(z["joint_pos"])
    if nf != 41371:
        fail.append("motion %s: %d frames, expected 41371 (v4)" % (sel[0], nf))
"""
new = """    z = np.load(f"{mdir}/{sel[0]}"); nf = len(z["joint_pos"])
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
"""
assert old in s
s = s.replace(old, new, 1)
s = s.replace(" 3. data: terrain dir (with the prop routes: single_beams >= 200 routes), motion yaml + npz (v4: 41,371 frames);",
              " 3. data: terrain dir (with the prop routes: single_beams >= 200 routes), motion yaml + npz (v4: 41,371 frames), and the\n"
              "    motion reader's joint lookup (every robot joint in joint_names), shapes, finite values, unit quaternions;", 1)
open(p, "w").write(s)

# 4. launcher: put the last error line of the check into launch.txt.
p = f"{X}/continue_v4k.sh"
s = open(p).read()
shutil.copy(p, f"{TR}/continue_v4k.sh.before_errline")
old = """  echo "$(date +%T) v4k NOT started: construction check failed (exit $CKRC, routes ${NR:-none}, or wall_contact_graded not in the reward table); v4j resumed" >> $S/launch.txt"""
new = """  echo "$(date +%T) v4k NOT started: construction check failed (exit $CKRC, routes ${NR:-none}, or wall_contact_graded not in the reward table); v4j resumed; last error: $(grep -aE '^[A-Za-z]*(Error|Exception)' $CK/check.log | tail -1 | cut -c1-200)" >> $S/launch.txt"""
assert old in s
s = s.replace(old, new, 1)
open(p, "w").write(s)
print("patched append_motion_v4.py, precheck_v4k.py, continue_v4k.sh")
