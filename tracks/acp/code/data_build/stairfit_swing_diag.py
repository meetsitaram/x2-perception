import sys, json, re, numpy as np, joblib
sys.path.insert(0, 'C:/Users/PC/projects/duck-transformer/tools'); import memcap; memcap.guard(4, 12)
sys.path.insert(0, 'C:/Users/PC/projects/loco-manip-apparatus/sim')
from pipeline import stairfit as SF
X0 = 1.0
class Flight:
    def __init__(s, rise, tread, n, width): s.rise, s.tread, s.n, s.w = rise, tread, n, width
    def height_min(s, xy):
        xy = np.atleast_2d(np.asarray(xy, float)); k = np.clip(np.floor((xy[:, 0] - X0) / s.tread) + 1, 0, s.n)
        return np.where(np.abs(xy[:, 1]) <= s.w / 2, k * s.rise, 0.0)
kin = SF._Kin()
clips = joblib.load(sys.argv[1])
for tag, c in clips.items():
    n, r, t, w = map(int, re.match(r'gs2_dn(\d+)_r(\d+)_t(\d+)_w(\d+)', tag).groups())
    course = Flight(r / 1000, t / 1000, n, w / 100)
    q = np.c_[c['root_trans_offset'], c['root_rot'][:, [3, 0, 1, 2]], c['dof']][::-1]
    pos, rot, sole = kin.feet(q)
    for f in range(2):
        runs = SF.plants(sole[:, f].mean(1))
        for (a0, e0), (a1, _) in zip(runs, runs[1:]):
            S = sole[e0:a1, f]
            gap = S[..., 2] - course.height_min(S[..., :2].reshape(-1, 2)).reshape(S.shape[:2])
            i = np.unravel_index(gap.argmin(), gap.shape)
            if gap.min() < -0.03:
                print(tag[:30], 'foot', f, 'swing', e0, a1, 'min gap cm', round(100 * gap.min(), 1), 'at frame', e0 + i[0], 'sphere', i[1],
                      'xyz', np.round(S[i[0], i[1]], 3), 'ankle x', round(pos[e0 + i[0], f, 0], 3), 'yaw deg', round(np.degrees(np.arctan2(rot[e0+i[0], f, 1, 0], rot[e0+i[0], f, 0, 0])), 0))
