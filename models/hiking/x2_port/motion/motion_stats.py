import sys, numpy as np
def stats(path, label):
    d = np.load(path, allow_pickle=True); fps = float(d["framerate"])
    p, q = d["base_pos_w"].astype(float), d["joint_pos"].astype(float)
    v = np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1) * fps
    seam = v > 3.0                      # clip seams in a concatenated file
    qd = np.abs(np.diff(q, axis=0)).max(1) * fps
    # smooth over 0.5 s so a foot-plant instant inside a walk is not counted as standing
    k = int(0.5 * fps); ker = np.ones(k) / k
    vs = np.convolve(np.where(seam, 0, v), ker, mode="same"); qs = np.convolve(np.where(seam, 0, qd), ker, mode="same")
    slow = vs < 0.1; still = slow & (qs < 0.5)
    n = len(v)
    print("%s: %d frames %.1f s, seams %d" % (label, n + 1, (n + 1) / fps, seam.sum()))
    print("  planar speed < 0.1 m/s (0.5 s mean): %.1f%%   and joints still (max joint speed < 0.5 rad/s): %.1f%%" % (100 * slow.mean(), 100 * still.mean()))
    print("  speed percentiles 5/25/50/75/95: %s m/s" % np.percentile(vs, [5, 25, 50, 75, 95]).round(2))
    print("  share in command range 0.45-1.0 m/s: %.1f%%, below 0.45: %.1f%%, above 1.0: %.1f%%" % (100 * ((vs >= 0.45) & (vs <= 1.0)).mean(), 100 * (vs < 0.45).mean(), 100 * (vs > 1.0).mean()))
    # runs of standing
    runs, c = [], 0
    for s in still:
        if s: c += 1
        elif c: runs.append(c); c = 0
    if c: runs.append(c)
    runs = np.array(runs) / fps if runs else np.array([0.0])
    print("  standing stretches: %d, longest %.1f s, total %.1f s" % (len(runs), runs.max(), runs.sum()))
    return vs, still, seam
for a, b in zip(sys.argv[1::2], sys.argv[2::2]):
    stats(a, b)
