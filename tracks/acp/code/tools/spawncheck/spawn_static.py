# Static spawn check of all goal tiles against the stage-3 raster and the tile's boxes (ACP, 2026-11-14).
import json, sys, collections, numpy as np
T = '/mnt/filesystem-g4/tracks/acp'
gt = json.load(open(f'{T}/data/goals/goal_tiles_v2.json'))
cfg = json.load(open(f'{T}/code/cfg/stage3v2h_sets.json'))
z = np.load(f"{cfg['dir']}/{cfg['hmap']}")
G = np.nan_to_num(np.maximum(z['grid'], 0.0)); x0, y0, res = float(z['x0']), float(z['y0']), float(z['res'])
print('raster', G.shape, x0, y0, res, 'x range', x0, x0 + (G.shape[0] - 1) * res, 'y range', y0, y0 + (G.shape[1] - 1) * res)

def disc(x, y, r):
    i0, i1 = int(np.floor((x - r - x0) / res)), int(np.ceil((x + r - x0) / res))
    j0, j1 = int(np.floor((y - r - y0) / res)), int(np.ceil((y + r - y0) / res))
    if i0 < 0 or j0 < 0 or i1 >= G.shape[0] or j1 >= G.shape[1]:
        return None
    ii, jj = np.meshgrid(np.arange(i0, i1 + 1), np.arange(j0, j1 + 1), indexing='ij')
    m = (x0 + ii * res - x) ** 2 + (y0 + jj * res - y) ** 2 <= r * r
    return G[ii[m], jj[m]]

def in_box(x, y, zz, b, pad=0.0):
    cx, cy, cz = b['center']; hx, hy, hz = b['half']; c, s = np.cos(-b['yaw']), np.sin(-b['yaw'])
    dx, dy = x - cx, y - cy; lx, ly = c * dx - s * dy, s * dx + c * dy
    return abs(lx) <= hx + pad and abs(ly) <= hy + pad and (cz - hz) <= zz <= (cz + hz)

R_FOOT = float(sys.argv[1]) if len(sys.argv) > 1 else 0.30
rows = []
for t in gt['tiles']:
    x, y, sz, yaw = t['spawn']
    h = disc(x, y, R_FOOT)
    r = dict(name=t['name'], fam=t['family'], lvl=t['level'], spawn=t['spawn'])
    if h is None:
        r.update(kind='off_raster')
    else:
        r.update(hmax=float(h.max()), hmin=float(h.min()), hmed=float(np.median(h)))
    # body column inside a box (prop tiles): any box overlapping the body column (feet 0..1.3 m above the spawn floor)
    nb = [i for i, b in enumerate(t['boxes']) if any(in_box(x, y, sz + dz, b, pad=0.15) for dz in (0.02, 0.3, 0.6, 0.9, 1.2))]
    r['boxes_hit'] = nb
    rows.append(r)
bad = collections.Counter(); tot = collections.Counter(r['fam'] for r in rows)
out = []
for r in rows:
    why = []
    if r.get('kind') == 'off_raster': why.append('off_raster')
    else:
        if r['hmax'] - r['spawn'][2] > 0.05: why.append('raster_above_spawn_%.2f' % (r['hmax'] - r['spawn'][2]))
        if r['spawn'][2] - r['hmax'] > 0.10 and not r['boxes_hit']: why.append('spawn_above_raster_%.2f' % (r['spawn'][2] - r['hmax']))
    if r['boxes_hit']: why.append('box_%s' % r['boxes_hit'])
    r['why'] = why
    if why: bad[r['fam']] += 1
    out.append(r)
for f in sorted(tot): print('%-20s tiles %4d static-bad %4d' % (f, tot[f], bad[f]))
json.dump(out, open(sys.argv[2] if len(sys.argv) > 2 else 'spawn_static.json', 'w'), indent=0)
for r in [r for r in out if r['why']][:25]: print(r['name'], r['spawn'], r.get('hmin'), r.get('hmax'), r['why'])
