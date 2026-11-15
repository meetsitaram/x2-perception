"""CPU pre-flight of v4j (2026-11-12 20:25, after the 19:22 construction check died on `unknown HIKING_X2_OPTS: ['g']`).
Run by continue_v4j.sh BEFORE v4i is stopped; exit 1 leaves v4i training untouched.
Checks what the Isaac construction check needs but test_v4j.py never touched:
 1. every option letter in the launcher's HIKING_X2_OPTS (check and training lines) is in staged.py's allowed set (the 19:22 failure);
 2. the .v4j files and staged.py compile;
 3. the data the env vars point at exists (terrain dir, motion dir, yaml, the npz it selects) and motion v3 has no excluded take;
 4. test_v4j.py passes on the current files.
usage: precheck_v4j.py <continue_v4j.sh> <config/x2 dir of files_instinctlab>"""
import os, py_compile, re, shutil, subprocess, sys, tempfile
import numpy as np

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
        fail.append("HIKING_X2_OPTS %s has letters %s not in staged.py's allowed set %s" % (opts, sorted(bad), "".join(sorted(allowed))))
tmp = tempfile.mkdtemp()
for f in ["staged.py", "extended.py.v4j", "routes.py.v4j", "keep.py.v4j", "contact.py.v4j"]:
    try:
        shutil.copy(f"{D}/{f}", f"{tmp}/x.py")
        py_compile.compile(f"{tmp}/x.py", doraise=True)
    except Exception as e:
        fail.append("compile %s: %s" % (f, e))
env = dict(re.findall(r"(HIKING_[A-Z0-9_]+)=(\S+?)[\"\s]", src))
ext, mdir, myaml = env.get("HIKING_EXTENDED_TERRAINS"), env.get("HIKING_X2_MOTION_DIR"), env.get("HIKING_X2_MOTION_YAML")
if not (ext and os.path.isdir(ext) and os.listdir(ext)):
    fail.append("terrain dir missing or empty: %s" % ext)
yp = f"{mdir}/{myaml}"
if not os.path.isfile(yp):
    fail.append("motion yaml missing: %s" % yp)
else:
    for line in open(yp):
        line = line.strip()
        if line.startswith("- "):
            npz = f"{mdir}/{line[2:].strip()}"
            if not os.path.isfile(npz):
                fail.append("motion file in yaml missing: %s" % npz)
            else:
                z = np.load(npz)
                n = len(z["joint_pos"])
                if n != 20001 + 6065:  # v2 + 9 sideways + 12 turning stair takes (113941 dropped 20:14)
                    fail.append("motion v3 has %d frames, expected 26066 (v3 without tp_stairs_113941)" % n)
r = subprocess.run([sys.executable, f"{T}/code/x2_port/v4j/test_v4j.py", D, ext or "", "/mnt/filesystem-g4/data/terrains/extended_v1",
                    "/mnt/filesystem-g4/data/terrains/extended_v1_2_stairs_inset", yp.replace(".yaml", "_retargetted.npz"),
                    "/mnt/filesystem-g4/data/motions/hiking_x2/x2_parkour_motion_without_run_v2_retargetted.npz"],
                   capture_output=True, text=True)
if "ALL PASS" not in r.stdout:
    fail.append("test_v4j.py: " + (r.stdout + r.stderr).strip().splitlines()[-1][:300] if (r.stdout + r.stderr).strip() else "no output")
print("opts allowed:", "".join(sorted(allowed)), "| terrain:", ext, "| motion:", yp)
print("\n".join("FAIL " + f for f in fail) if fail else "PRECHECK PASS")
sys.exit(1 if fail else 0)
