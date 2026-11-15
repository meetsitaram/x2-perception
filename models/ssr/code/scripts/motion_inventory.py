"""Inventory of every motion file the SSR priors and walking starts use: size, sha256, clips, seconds (CPU, read-only)."""
import hashlib, json, os, sys, re, glob
import numpy as np, joblib


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


out = {"sources": {}, "priors": {}}
for p in sorted(glob.glob("/mnt/filesystem-g4/data/motions/ssr_prior_sources/*/*.pkl")):
    d = joblib.load(p)
    keys = list(d.keys())
    secs = sum(len(d[k]["dof"]) / float(d[k]["fps"]) for k in keys)
    # classify by key text only (names are not printed anywhere shown to the user)
    pico = sum(1 for k in keys if re.search(r"pico|tp_|park|2x2|4x4", k, re.I))
    out["sources"][p] = dict(bytes=os.path.getsize(p), sha256=sha(p), clips=len(keys), seconds=round(secs, 1),
                             keys_with_pico_or_park_marks=pico, example_key_prefix=sorted(set(re.split(r"[_\-/ ]", k)[0] for k in keys))[:6])
p = "/mnt/filesystem-g4/data/motions/hiking_x2/x2_parkour_motion_without_run_v2_retargetted.npz"
out["sources"][p] = dict(bytes=os.path.getsize(p), sha256=sha(p), seconds=round(len(np.load(p)["joint_pos"]) / 50, 1))
for p in sorted(glob.glob(sys.argv[1] + "/amp*.npz")):
    z = np.load(p)
    cats = {}
    for c in ("flat", "ramp", "stairs", "beam", "platform"):
        a, st = z[c], z[c + "_starts"]
        sp = np.linalg.norm(a[:, 3:5], axis=1)
        cats[c] = dict(clips=len(st), seconds=round(len(a) / 50, 1), frames_ge_0p2_mps_seconds=round(float((sp >= 0.2).sum()) / 50, 1))
    out["priors"][os.path.realpath(p)] = dict(bytes=os.path.getsize(p), sha256=sha(p), sets=cats)
print(json.dumps(out, indent=1))
