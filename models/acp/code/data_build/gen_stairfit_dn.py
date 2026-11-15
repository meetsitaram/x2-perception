"""ACP track (2026-11-12): stair DESCENTS re-fitted with the feet facing down the stairs.

Why: gwbc-v6 gen_stairfit2.py fits a descent time-reversed (as a climb) with pipeline.stairfit, whose tread_targets turns
every tread foot "straight up the stairs". In a reversed descent the body faces DOWN the stairs, so each foot was turned
about 180 degrees against the pelvis: hip yaw at its limit (+-3.43 rad), the 38 gs2dn_v1 clips. Here tread_targets is
replaced by a version that, for descents, turns the foot to face down the stairs (pelvis heading) and places it heel
HEEL_GAP short of the next (higher) riser, toe toward the nose. Climbs are not touched (descents only). Same random
flights rule as gen_stairfit2 (never the apparatus's riser/tread), same acceptance (sole on tread >= 0.55, deepest <= 2 cm,
IK miss <= 8 cm) plus a leg-flip screen (|hip yaw| <= 1.5 rad).

    python gen_stairfit_dn.py OUT_PREFIX [K flights per capture]     (desktop, Python312, CPU; memory-capped)
Writes OUT_PREFIX.pkl (50 Hz clips, gs2dn_v2_*), _props.json, _report.json.
"""
import glob, json, math, os, sys
sys.path.insert(0, 'C:/Users/PC/projects/duck-transformer/tools')
import memcap  # noqa: E402
memcap.guard(cap_gb=6, need_free_gb=16)
import psutil  # noqa: E402
psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS if os.name == 'nt' else 19)   # low priority: the desktop comes first
import joblib, numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation as Rot  # noqa: E402
sys.path.insert(0, 'C:/Users/PC/projects/loco-manip-apparatus/sim')
from pipeline import stairfit as SF, hmap as H, stepgen as SG  # noqa: E402
sys.path.insert(0, 'C:/Users/PC/projects/gwbc-v6/scripts/acp/data')
from gen_stairfit_lib import load_csv  # noqa: E402

out = sys.argv[1]; K = int(sys.argv[2]) if len(sys.argv) > 2 else 4
X0, LANDING_D, SLAB, WALL_H, WALL_T = 1.0, 1.5, 0.30, 1.0, 0.05
TREAD_BOX, WIDTHS, WALL_HALF = 0.30, (0.9, 1.2, 1.5), (2.3, 1.38)
APP = ((0.186, 0.222), (0.203, 0.244))
rng = np.random.default_rng(2)
YAW_MAX = 1.5
HEEL_GAP = float(os.environ.get('DN_GAP', 0.05))   # m: descent foot's heel this far short of the next riser (2 cm let the heel
#   cut the riser in the frames around touchdown: swing clearance -13 to -17 cm in the first trial)
CLEAR_MIN = -3.0                                    # cm: least swing clearance accepted (the gs2_v1 fits: worst -2.5)


def draw(L):
    while True:
        rise, tread = rng.uniform(0.15, 0.22), rng.uniform(0.23, 0.29)
        if not any(abs(rise - r) < 0.006 and abs(tread - t) < 0.006 for r, t in APP):
            return round(rise, 3), round(tread, 3), int(np.clip(rng.integers(L - 2, L + 1), 3, 8)), WIDTHS[rng.integers(len(WIDTHS))], ('none', 'left', 'right', 'both')[rng.integers(4)]


class Flight:
    def __init__(s, rise, tread, n, width): s.rise, s.tread, s.n, s.w = rise, tread, n, width

    def height_min(s, xy):
        xy = np.atleast_2d(np.asarray(xy, float)); k = np.clip(np.floor((xy[:, 0] - X0) / s.tread) + 1, 0, s.n)
        return np.where(np.abs(xy[:, 1]) <= s.w / 2, k * s.rise, 0.0)
    height = height_min

    def cls(s, xy): return np.zeros(len(np.atleast_2d(xy)), int) + (H.OBSTACLE + 1)


def boxes(rise, tread, n, width, walls):
    b = []
    for k in range(1, n + 1):
        xa = X0 + (k - 1) * tread; xb = xa + (TREAD_BOX if k < n else LANDING_D)
        b.append(dict(kind='step', center=[(xa + xb) / 2, 0.0, k * rise - SLAB / 2], half=[(xb - xa) / 2, width / 2, SLAB / 2], yaw=0.0))
    xa, xb = X0 - 1.0, X0 + (n - 1) * tread + LANDING_D
    for side in ({'left': (1,), 'right': (-1,), 'both': (1, -1)}.get(walls, ())):
        assert xb - xa <= 2 * WALL_HALF[0] and n * rise + WALL_H <= 2 * WALL_HALF[1]
        b.append(dict(kind='wall', center=[xa + WALL_HALF[0], side * (width / 2 + WALL_T / 2), WALL_HALF[1]], half=[WALL_HALF[0], WALL_T / 2, WALL_HALF[1]], yaw=0.0))
    return b


def tread_targets_down(q, runs, lvl, course, kin, pos=None, rot=None, sole=None):
    """As stairfit.tread_targets, for a time-reversed descent: the foot faces down the stairs (u decreasing)."""
    if pos is None:
        pos, rot, sole = kin.feet(q)
    centre = sole.mean(2)
    head = (0.0 if SF.DIR > 0 else math.pi) + math.pi                     # down the stairs
    riser = [SF.DIR * r for r in SF.RISERS]
    offs, yaws = [], []
    for f in range(2):
        o_f, y_f, last_du = [], [], 0.0
        for (a, e), k in zip(runs[f], lvl[f]):
            m_ = (a + e) // 2
            c = centre[m_, f]
            yaw = math.atan2(rot[m_, f, 1, 0], rot[m_, f, 0, 0])
            turn = (head - yaw + math.pi) % (2 * math.pi) - math.pi
            au = SF.DIR * pos[m_, f, 0]                                     # ankle u; foot facing -u: heel at au + HEEL
            if 1 <= k <= len(SF.TOPS):                                      # heel TOE_GAP short of the next (higher) riser
                du, top = riser[k] - HEEL_GAP - SF.HEEL - au, SF.TOPS[k - 1]
            elif k > len(SF.TOPS):                                          # landing: sole centre (au - SOLE_X) past the edge
                du, top = max(au + last_du, riser[-1] + SF.LANDING_IN + SF.SOLE_X) - au, SF.LANDING
            else:                                                           # floor (unturned foot facing -u): heel off the first riser
                du, top = min(0.0, (riser[0] - SF.FLOOR_GAP) - (SF.DIR * c[0] + SF.SOLE_X + SF.HEEL)), SF.FLOOR
            last_du = du
            dx = SF.DIR * du
            if k >= 1:
                s = sole[m_, f, :, :2] - pos[m_, f, :2]
                s = Rot.from_euler('z', turn).apply(np.c_[s, np.zeros(len(s))])[:, :2] + pos[m_, f, :2] + [dx, 0.0]
                top = float(np.median(course.height_min(s)))
            o_f.append(np.array([dx, 0.0, top - sole[a:e, f, :, 2].min(-1).mean()]))
            y_f.append(np.array([0.0, 0.0, turn if k >= 1 else 0.0]))
        offs.append(o_f)
        yaws.append(y_f)
    return offs, yaws


SF.tread_targets = tread_targets_down                                   # fit() looks it up in the module at call time
downs = sorted(sum((glob.glob(os.path.join(SF.SEED, f'stairs_climbing_{k}_*.csv')) for k in ('down_loop', 'down_stop', 'jog_down_loop', 'jog_down_stop')), []))
kin, ik = SF._Kin(), SG.LegIK(); clips, props, report = {}, {}, {}
LIM = int(os.environ.get('LIMIT', 0))
hipyaw = None
for path in (downs[:LIM] if LIM else downs):
    name = os.path.basename(path)[:-4]; src = load_csv(path)[::-1].copy()
    SF.STAIRS['gen'] = dict(risers=tuple(X0 + k * 0.25 for k in range(6)), tops=tuple(k * 0.18 for k in range(1, 6)), landing=1.08, y=0.0, dir=1, floor=0.0)
    SF.use('gen'); pl = SF.place(src.copy(), kin)
    if pl[0] is None: report[name] = dict(reason=pl[1]); print(name, 'skip:', pl[1], flush=True); continue
    L = int(max(l.max() for l in pl[2]))
    for j in range(K):
        rise, tread, n, width, walls = draw(L)
        SF.STAIRS['gen'] = dict(risers=tuple(X0 + k * tread for k in range(n)), tops=tuple(k * rise for k in range(1, n)), landing=n * rise, y=0.0, dir=1, floor=0.0)
        SF.use('gen'); course = Flight(rise, tread, n, width)
        tag = f'gs2_dn{n}_r{round(1000 * rise)}_t{round(1000 * tread)}_w{round(100 * width)}_{walls}_{name}'
        try:
            q, info = SF.fit(src.copy(), course, kin, ik)
        except Exception as e:
            report[tag] = dict(error=str(e)[:120]); continue
        if q is None: report[tag] = info; continue
        info.update(SF.check(q, course, kin))
        q = np.asarray(q, np.float64)
        if hipyaw is None:
            import mujoco
            jn = [kin.m.joint(i).name for i in range(kin.m.njnt)]
            hipyaw = [kin.m.jnt_qposadr[i] for i, nm in enumerate(jn) if 'hip_yaw' in nm]
        info['max_abs_hip_yaw'] = round(float(np.abs(q[:, hipyaw]).max()), 2)
        info['max_jump_rad'] = round(float(np.abs(np.diff(q[:, 7:], axis=0)).max()), 3)
        report[tag] = info
        if not (info['sole_on_tread_mean'] >= 0.55 and info['deepest_cm'] <= 2.0 and info['ik_miss_max_cm'] <= 8.0 and info['max_abs_hip_yaw'] <= YAW_MAX and info['swing_clear_min_cm'] >= CLEAR_MIN):
            continue
        q = q[::-1]
        clips[tag] = dict(root_trans_offset=q[:, :3].astype(np.float32), root_rot=q[:, [4, 5, 6, 3]].astype(np.float32), dof=q[:, 7:].astype(np.float32), fps=50)
        props[tag] = boxes(rise, tread, n, width, walls)
    print(f'{name}: levels {L}, {sum(k.endswith(name) for k in clips)} / {K} fit', flush=True)
joblib.dump(clips, out + '.pkl'); json.dump(props, open(out + '_props.json', 'w'), indent=1); json.dump(report, open(out + '_report.json', 'w'), indent=1)
print(len(clips), 'descent clips ->', out)
