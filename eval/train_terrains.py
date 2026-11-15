"""Courses that copy each track's OWN training terrain types, for the review renders of 2026-11-13 (eval monitor; user: "I just
want to see success on the terrain they are training on"). On the training terrain, never an apparatus or ladder result.

    python eval/ladder.py --adapter ... --step 20 --tests ssr_stairs_L10 hik_gap_L2 ... --trials 3 --out DIR

SSR (`ssr_<type>_L<r>`): the track's own generator, tracks/ssr/code/ssr/terrain.py (`BUILD[type](tile, r / 19, rng)`, 20 levels),
one 8 x 4 m tile walked +x as in training. Exact walkable raster; the vision-only side walls and the stations of training are left out.
Hiking (`hik_<type>_L<row>`): the paper's nine sub-terrain types at curriculum rows 2, 5, 8 of 0-9, rebuilt from the
generator's own code (InstinctLab parkour_env_cfg.py + hf_terrains.py; see the Hiking section). Exact geometry except: no
Perlin noise on treads, small boxes approximated, boxes and rough one sampled layout. (Until 2026-11-13 12:00 these were
free approximations with 10 risers from 0.5 m and a slope that never flattened: the policy stalled where training's terrain
ends; that set, levels 1-3 of 3, is retired: hik_*_L2 now means curriculum row 2 - see eval/out/renders_20261113 for the old.)
Extended-map tiles (`ext_*`): the user's extended_v1 tiles and routes, exact copies (SSR v13+, Hiking's extended task).
ACP trains on per-clip prop scenes from the captures (ramp raster, 4x4, 2x2, stairs, boxes): those scenes are not rebuilt; the
closest courses are the eval ladder's step-1 obstacles (`acp_<skill>_L<1-3>`, eval/ladder_synth.py) plus box steps (`acp_box_<cm>`).
TRAIN_SETS: what the automatic scorer's `train` job runs per track; META: type, level, exact / approximation, floor per test.
Floor: the solid floor (phys_floor.py) models flat cells and 10-20 deg planar ramp faces only; slope and rough courses (SSR ramp,
Hiking slope_up and rough) are run with PHYS_FLOOR=heightfield (feet on the height field), and their captions say so.
"""
from __future__ import annotations

import importlib.util
import math
import pathlib
import sys
import types

import numpy as np



def wp(*a, **k):
    import ladder
    return ladder.wp(*a, **k)

RES, N = 0.02, 400
XS = -3.99 + RES * np.arange(N)
GX, GY = np.meshgrid(XS, XS, indexing="ij")
REPO = pathlib.Path(__file__).resolve().parent.parent
SSR_LEVELS = (5, 10, 15, 19)
DESC = {}                     # test -> caption text (terrain type and level / parameter)
META = {}                     # test -> dict(type, level, exact, floor): filled when the course is built (see meta())


def _course(name, g, spawn, way, tile, zones=()):
    from ladder import Course
    extra = dict(beams=[], beam_lines=[], zones=list(zones), tile=tile, level=0)
    return Course("ladder:" + name, g, -3.99, -3.99, RES, [], spawn, way, name, extra)


# ---- SSR: the track's own generator --------------------------------------------------------------------------------------
_SSR = None


def _ssr():
    global _SSR
    if _SSR is None:
        class _Stub(types.ModuleType):
            def __getattr__(self, k):
                return _Stub(k) if k[0].isupper() else (lambda *a, **kw: (lambda f: f))
        had = sys.modules.get("torch")
        sys.modules["torch"] = _Stub("torch")
        try:
            sp = importlib.util.spec_from_file_location("ssr_terrain_copy", REPO / "tracks" / "ssr" / "code" / "ssr" / "terrain.py")
            _SSR = importlib.util.module_from_spec(sp)
            sp.loader.exec_module(_SSR)
        finally:
            if had is None:
                del sys.modules["torch"]
            else:
                sys.modules["torch"] = had
    return _SSR


def ssr(ttype, r):
    T = _ssr()
    c = T.TYPES.index(ttype)
    rng = np.random.default_rng(1000 + 37 * c + r)
    t = np.zeros((T._i(T.TILE_X), T._i(T.TILE_Y)), np.float32)
    T.BUILD[c](t, r / (T.NLEVEL - 1), rng)
    # tile x 0..8 -> course x -4..4; tile y 0..4 -> course y -2..2 (outside the lane: the tile's own edge rows continued)
    fx = np.clip((GX + 4.0) / T.RES, 0, t.shape[0] - 1)
    fy = np.clip((GY + 2.0) / T.RES, 0, t.shape[1] - 1)
    if ttype == "ramp":                          # smooth faces: bilinear (nearest-cell resampling 0.025 -> 0.02 m made 5 mm steps)
        from scipy import ndimage
        g = ndimage.map_coordinates(t.astype(np.float64), [fx, fy], order=1, mode="nearest")
    else:
        g = t[np.round(fx).astype(int), np.round(fy).astype(int)].astype(np.float64)
    s = r / (T.NLEVEL - 1)
    par = {"ramp": "slope %.1f deg" % (20 * s), "stairs": "rise %.0f cm, run %.0f cm" % (100 * (0.05 + 0.15 * s), 100 * (0.40 - 0.15 * s)),
           "beam": "beams %.0f cm high, %.0f cm wide" % (100 * (0.03 + 0.09 * s), 100 * (0.05 + 0.09 * s)),
           "platform": "platforms %.0f cm high" % (100 * (0.05 + 0.35 * s)), "flat": "flat"}[ttype]
    name = "ssr_%s_L%d" % (ttype, r)
    DESC[name] = "SSR training tile '%s', level %d of 0-19 (%s), the track's own generator" % (ttype, r, par)
    META[name] = dict(type=ttype, level=r, exact="exact walkable raster from the track's generator (side walls and stations left out)",
                      floor="heightfield" if ttype == "ramp" else "solid")
    return _course(name, g, [-3.4, 0.0, 0.0], [wp(0.0, 0.0, name="middle of the tile"), wp(3.4, 0.0, hold=1.0, name="end of the tile, stand 1 s")],
                   "SSR terrain.py %s level %d" % (ttype, r))


# ---- Hiking: the paper's sub-terrain types, from the generator's own code (2026-11-13 rebuild) ---------------------------
# Training config: InstinctLab tasks/parkour/config/parkour_env_cfg.py ROUGH_TERRAINS_CFG (8 x 8 m tiles, horizontal 0.05 m,
# vertical 0.005 m, 10 curriculum rows, difficulty = (row + U(0,1)) / 10), height fields from InstinctLab
# terrains/height_field/hf_terrains.py and IsaacLab's height_field_to_mesh border handling (border pixels = int(b / 0.05) + 1).
# Level k here = curriculum row k of 0-9 (the terrain level Hiking's monitor reports), difficulty (k + 0.5) / 10.
# The functions below repeat those generators line by line for the walkable raster. Left out: the 0.05 m Perlin noise that
# training adds on stairs, platforms and boxes (about +-2 cm, 0.2 m wavelength; the solid floor models flat cells only), and
# the random side walls (wall_prob 0.3, vision only). Small boxes (mesh_boxes, a trimesh generator) are an approximation.
HS, VS, HIK_N = 0.05, 0.005, 161                 # IsaacLab: int(8.0 / 0.05) + 1 vertices per side
HIK_LEVELS = (2, 5, 8)
HIK_TYPES = ("stairs_up", "stairs_down", "platform_up", "platform_down", "slope_up", "boxes", "small_boxes", "rough", "gap")
HIK_EXACT = {"small_boxes": "approximation (mesh generator not ported: 0.40 m blocks, 30 % of cells, mean height as training)",
             "rough": "same generator and parameters (InstinctLab Perlin code), one sampled field",
             "boxes": "same generator and parameters, one sampled layout chosen so the route crosses a box",
             "slope_up": "exact (generator geometry, no noise in training either)", "gap": "exact (generator geometry; depth 0.5 m, training 0.4-0.6)"}


def _perlin2d(shape, res, rs):
    """InstinctLab utils/perlin.py generate_perlin_noise_2d, with an explicit RandomState."""
    def f(t):
        return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    ang = 2 * np.pi * rs.rand(res[0] + 1, res[1] + 1)
    gr = np.dstack((np.cos(ang), np.sin(ang)))
    g00 = gr[0:-1, 0:-1].repeat(d[0], 0).repeat(d[1], 1); g10 = gr[1:, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gr[0:-1, 1:].repeat(d[0], 0).repeat(d[1], 1); g11 = gr[1:, 1:].repeat(d[0], 0).repeat(d[1], 1)
    n00 = np.sum(grid * g00, 2); n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2); n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)
    t = f(grid)
    n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10; n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
    return np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1) * 0.5 + 0.5


def _fractal(size, samples, z, rs, freq=20, octaves=2, lac=2.0, gain=0.25):
    """InstinctLab generate_fractal_noise_2d (centering=True), in metres."""
    xs = int(freq * size); amp = 1.0
    n_exp = int(xs * lac ** octaves)
    noise = np.zeros((n_exp, n_exp))
    for _ in range(octaves):
        noise += amp * _perlin2d(noise.shape, (xs, xs), rs) * z
        amp *= gain; xs = int(lac * xs)
    noise -= noise.mean()
    return noise[:samples, :samples]


def _hik_hf(ttype, d, rs):
    """The tile's height field in metres on the 161 x 161 vertex grid (IsaacLab border handling), centre = spawn."""
    border = {"boxes": 0.0, "small_boxes": 0.0, "rough": 0.0}.get(ttype, 1.0)
    bp = int(border / HS) + 1
    n = HIK_N - 2 * bp
    size = n * HS
    w = int(size / HS)
    full = np.zeros((HIK_N, HIK_N))
    if ttype in ("stairs_up", "stairs_down", "platform_up", "platform_down"):
        high = ttype.startswith("platform")
        lo_, hi_ = (0.05, 0.45) if high else (0.05, 0.23)
        sh = lo_ + d * (hi_ - lo_)
        if ttype.endswith("_up"):                    # inverted pyramid: the robot spawns in the pit and walks up
            sh = -sh
        step_w, plat = round((1.5 if high else 0.3) / HS), round((4.0 if high else 2.5) / HS)
        sh_px = round(sh / VS)
        hf = np.zeros((w, w)); cur = 0; a, b = 0, w
        while (b - a) > plat:
            a += step_w; b -= step_w; cur += sh_px
            hf[a:b, a:b] = cur
        hf = hf * VS
    elif ttype == "slope_up":                        # hf_pyramid_slope_inv, slope_range (0, 0.7) rise over run
        slope = -(0.0 + d * 0.7)
        hmax = int(slope * size / 2 / VS)
        c = int(w / 2)
        xx = ((c - np.abs(c - np.arange(w))) / c)
        hf = hmax * xx.reshape(w, 1) * xx.reshape(1, w)
        pw = int(1.5 / HS / 2)
        zpf = hf[w // 2 - pw, w // 2 - pw]
        hf = np.rint(np.clip(hf, min(0, zpf), max(0, zpf))) * VS
    elif ttype == "gap":                             # square_gaps: gap 0.1-0.7 m, depth U(0.4, 0.6) (0.5 here), platform 2.5 m
        g_px = round((0.1 + d * 0.6) / HS)
        pl = round(2.5 / HS)
        s0 = (w - pl) // 2; e0 = s0 + pl
        sx, ex = max(0, s0 - g_px), min(w, e0 + g_px)
        gv = -round(0.5 / VS)
        hf = np.zeros((w, w))
        hf[sx:ex, sx:sx + g_px] = gv; hf[sx:ex, max(sx, ex - g_px):ex] = gv
        hf[sx:sx + g_px, sx:ex] = gv; hf[max(sx, ex - g_px):ex, sx:ex] = gv
        hf = hf * VS
    elif ttype == "boxes":                           # discrete obstacles, fixed height 0.05-0.45, 20 boxes 0.8-1.5 m, platform 1.5 m
        oh = round((0.05 + d * 0.40) / VS)
        wr = np.arange(round(0.8 / HS), round(1.5 / HS), 4)
        xr = np.arange(1, w, 4)
        hf = np.zeros((w, w))
        for _ in range(20):
            bw, bl = round(rs.choice(wr)), round(rs.choice(wr))
            x0, y0 = int(rs.choice(xr)), int(rs.choice(xr))
            if x0 + bw > w:
                x0 = w - bw - 1
            if y0 + bl > w:
                y0 = w - bl - 1
            hf[x0:x0 + bw, y0:y0 + bl] = oh
        pw = round(1.5 / HS)
        hf[(w - pw) // 2:(w + pw) // 2, (w - pw) // 2:(w + pw) // 2] = 0
        hf = hf * VS
    elif ttype == "rough":                           # perlin_rough: noise_scale 0 -> 0.1 with difficulty, frequency 20
        hf = np.rint(_fractal(size, w, 0.1 * d, rs) / VS) * VS
    elif ttype == "small_boxes":                     # APPROXIMATION of mesh_boxes (box mean height 0.1 -> 0.4, +-0.05, 0.4 +-0.1 m)
        m = 0.1 + 0.3 * d
        hf = np.zeros((w, w))
        k = round(0.4 / HS)
        for i in range(0, w - k, k):
            for j in range(0, w - k, k):
                if rs.rand() < 0.3:
                    lx, ly = round(rs.uniform(0.3, 0.5) / HS), round(rs.uniform(0.3, 0.5) / HS)
                    hf[i:i + lx, j:j + ly] = np.rint(rs.uniform(m - 0.05, m + 0.05) / VS) * VS
        pw = round(1.5 / HS)
        hf[(w - pw) // 2:(w + pw) // 2, (w - pw) // 2:(w + pw) // 2] = 0
    else:
        raise KeyError(ttype)
    full[bp:bp + w, bp:bp + w] = hf
    return full - full[HIK_N // 2, HIK_N // 2]


def _hik_route_ok(full, y_half=0.3, x0=0.8, x1=3.4):
    """boxes: the route y = 0 from the platform to the end crosses a box, and the end stands on one level patch."""
    ix = lambda x: int(round((x + 4.0) / HS))
    band = full[ix(x0):ix(x1) + 1, ix(-y_half):ix(y_half) + 1]
    end = full[ix(3.5) - 4:ix(3.5) + 5, ix(0) - 4:ix(0) + 5]
    return band.max() > 0.04 and np.ptp(end) < 0.01


def hiking(ttype, k):
    d = (k + 0.5) / 10.0
    seed = 0
    while True:                                      # fixed seeds; boxes take the first layout whose route crosses a box
        rs = np.random.RandomState(1000 + 31 * HIK_TYPES.index(ttype) + seed)
        full = _hik_hf(ttype, d, rs)
        if ttype != "boxes" or _hik_route_ok(full) or seed > 200:
            break
        seed += 1
    ix = np.clip(np.rint((GX + 4.0) / HS).astype(int), 0, HIK_N - 1)
    iy = np.clip(np.rint((GY + 4.0) / HS).astype(int), 0, HIK_N - 1)
    g = full[ix, iy].astype(np.float64)
    spawn = [0.0, 0.0, 0.0]
    way = [wp(2.0, 0.0, name="2 m out"), wp(3.5, 0.0, hold=1.0, name="0.2 m short of training's target (x 3.7), stand 1 s")]
    stair = ttype.startswith(("stairs", "platform"))
    if stair:
        hi = ttype.startswith("platform")
        sh = 0.05 + d * ((0.45 if hi else 0.23) - 0.05)
        par = "%s, %s %.1f cm, %s" % ("one platform edge at 1.48 m" if hi else "6 risers from 1.18 to 2.68 m, run 0.30 m",
                                      "rise", 100 * round(sh / VS) * VS, "up out of the pit" if ttype.endswith("_up") else "down from the top")
    elif ttype == "slope_up":
        par = "slope %.2f rise over run (%.1f deg) on the axis from 1.31 m to 2.98 m, then flat" % (0.7 * d, math.degrees(math.atan(0.7 * d)))
    elif ttype == "gap":
        par = "square gap %.0f cm wide at 1.25 m, 0.5 m deep" % (100 * (0.1 + 0.6 * d))
    elif ttype == "boxes":
        par = "20 boxes 0.8-1.5 m, %.0f cm high (layout seed %d)" % (100 * (0.05 + 0.4 * d), seed)
    elif ttype == "rough":
        par = "Perlin noise scale %.3f m" % (0.1 * d)
    else:
        par = "0.4 m blocks, mean height %.0f cm +-5" % (100 * (0.1 + 0.3 * d))
    name = "hik_%s_L%d" % (ttype, k)
    exact = HIK_EXACT.get(ttype, "exact step geometry; training's +-2 cm Perlin noise on the treads left out")
    DESC[name] = "Hiking training terrain '%s', curriculum row %d of 0-9 (%s); %s" % (ttype, k, par, exact)
    META[name] = dict(type=ttype, level=k, exact=exact, floor="heightfield" if ttype in ("slope_up", "rough") else "solid")
    return _course(name, g, spawn, way, "Hiking paper type %s row %d, %s" % (ttype, k, par))


# ---- ACP: box steps (the other ACP props are the ladder's step-1 courses) --------------------------------------------------
def acp_box(cm):
    h = cm / 100.0
    g = np.where((GX > -0.8) & (GX < 0.8) & (np.abs(GY) < 0.8), h, 0.0)
    name = "acp_box_%d" % cm
    DESC[name] = "box step %d cm, 1.6 m square, walked onto and off (closest course to ACP's box props)" % cm
    META[name] = dict(type="box_step", level=cm, exact="approximation: closest course to ACP's box props (clip scenes not rebuilt)", floor="solid")
    return _course(name, g, [-2.4, 0.0, 0.0], [wp(0.0, 0.0, hold=1.0, name="on the box, stand 1 s"), wp(2.0, 0.0, hold=1.0, name="past the box, stand 1 s")],
                   "box %d cm" % cm)


# ---- extended-map tiles (the user's extended_v1 export, trained on by SSR from v13 and by Hiking's extended task) and ACP's
# closest courses (the eval ladder's step-1 obstacle courses). Both are built by ladder.py / ladder_synth.py.
EXT = {"ext_log_single": ("log_single", "single_beams_L4_v1"), "ext_wedges": ("wedges", "wedges_L3_v1"),
       "ext_stairs_up": ("stairs_up", "stairs_L3_v0 up"), "ext_stairs_down": ("stairs_down", "stairs_L3_v0 down"),
       "ext_log_corridor_walls": ("log_corridor_walls", "log_corridors_L3_v0")}
ACP_SKILL = {"log": "4x4 log (L3: from the 0.51 m slot)", "wedge": "ramp cells (L3: 15 deg pinwheel)", "stairs_up": "stairs up",
             "stairs_down": "stairs down", "corridor": "corridor with logs"}


def _ext(test):
    import ladder
    src, tile = EXT[test]
    c = ladder.STEP1_V0[src]()
    DESC[test] = "extended-map tile %s, the export's own route (exact copy of the training tile)" % tile
    META[test] = dict(type="ext_" + src.replace("log_corridor_walls", "corridor"), level=int(tile.split("_L")[1][0]),
                      exact="exact copy of the extended_v1 tile and route", floor="solid")
    return c


def _acp(test):
    import ladder_synth
    c = ladder_synth.TESTS[test]()
    sk, lv = test.rsplit("_L", 1)
    DESC["acp_" + test] = "eval ladder course %s level %s, closest to ACP's training prop '%s'" % (sk, lv, ACP_SKILL[sk])
    META["acp_" + test] = dict(type=sk, level=int(lv), exact="approximation: closest eval course to ACP's per-clip prop scenes", floor="solid")
    return c


TESTS = {}
for _t in ("ramp", "stairs", "beam", "platform"):
    for _r in SSR_LEVELS:
        TESTS["ssr_%s_L%d" % (_t, _r)] = (lambda t=_t, r=_r: ssr(t, r))
TESTS["ssr_flat_L0"] = lambda: ssr("flat", 0)
for _t in HIK_TYPES:
    for _k in HIK_LEVELS:
        TESTS["hik_%s_L%d" % (_t, _k)] = (lambda t=_t, k=_k: hiking(t, k))
for _c in (15, 25):
    TESTS["acp_box_%d" % _c] = (lambda c=_c: acp_box(c))
for _e in EXT:
    TESTS[_e] = (lambda e=_e: _ext(e))
for _s in ("log", "wedge", "stairs_up", "stairs_down", "corridor"):
    for _l in (1, 2, 3):
        TESTS["acp_%s_L%d" % (_s, _l)] = (lambda t="%s_L%d" % (_s, _l): _acp(t))

# what the automatic scorer's `train` job runs per track (user rule 2026-11-13: progress on the training terrain)
TRAIN_SETS = {
    "ssr": ["ssr_%s_L%d" % (t, r) for t in ("ramp", "stairs", "beam", "platform") for r in SSR_LEVELS] + ["ssr_flat_L0"] + list(EXT),
    "hiking": ["hik_%s_L%d" % (t, k) for t in HIK_TYPES for k in HIK_LEVELS] + list(EXT),
    "acp": ["acp_%s_L%d" % (s, l) for s in ("log", "wedge", "stairs_up", "stairs_down", "corridor") for l in (1, 2, 3)] + ["acp_box_15", "acp_box_25"],
}


def floor_of(test):
    """The floor a test is run on: the solid floor models flat cells and 10-20 deg planar faces only."""
    return "heightfield" if test.startswith(("ssr_ramp_", "hik_slope_up_", "hik_rough_")) else "solid"


def info(test):
    """dict(type, level, exact, floor) from the test's name alone (no course built; used by the automatic scorer)."""
    if test in EXT:
        src, tile = EXT[test]
        return dict(type="ext_" + src.replace("log_corridor_walls", "corridor"), level=int(tile.split("_L")[1][0]),
                    exact="exact copy of the extended_v1 tile and route", floor="solid")
    head, lv = test.rsplit("_L", 1) if "_L" in test else test.rsplit("_", 1)
    if test.startswith("ssr_"):
        return dict(type=head[4:], level=int(lv), exact="exact walkable raster from the track's generator (side walls and stations left out)", floor=floor_of(test))
    if test.startswith("hik_"):
        t = head[4:]
        return dict(type=t, level=int(lv), exact=HIK_EXACT.get(t, "exact step geometry; training's +-2 cm Perlin noise on the treads left out"), floor=floor_of(test))
    if test.startswith("acp_box_"):
        return dict(type="box_step", level=int(lv), exact="approximation: closest course to ACP's box props (clip scenes not rebuilt)", floor="solid")
    return dict(type=head[4:], level=int(lv), exact="approximation: closest eval course to ACP's per-clip prop scenes", floor="solid")


meta = info


if __name__ == "__main__":
    for k_, f_ in TESTS.items():
        c_ = f_()
        print("%-26s z %.2f..%.2f  %s" % (k_, c_.grid.min(), c_.grid.max(), DESC[k_]))
