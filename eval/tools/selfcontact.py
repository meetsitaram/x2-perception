"""Robot self-contacts in the eval model (eval monitor, 2026-11-12): run from eval/: python tools/selfcontact.py ADAPTER (env: handover dir, X2_EXCLUDE_PAIRS). Standing (0 m/s, hold) and walking (0.6 m/s / goal 2 m ahead) 6 s each, flat; per robot body pair: share of physics steps in contact, deepest penetration, mean normal force."""
import sys, os, collections; sys.path.insert(0, '.')
import numpy as np, mujoco, harness as Hn, ladder, x2 as X
ad = Hn.load_adapter(sys.argv[1])
c = ladder.load_course('ladder:flat_walk')
names = X.joint_names_of(X.robot_dir() / 'mjcf' / 'x2_ultra.xml')
x = X.build(c, ad.plant(names)); m = x.model
bn = lambda g: m.body(m.geom_bodyid[g]).name
def pairs(d):
    out = {}
    f6 = np.zeros(6)
    for i in range(d.ncon):
        con = d.contact[i]
        if m.geom_bodyid[con.geom1] > 0 and m.geom_bodyid[con.geom2] > 0:
            k = tuple(sorted((bn(con.geom1), bn(con.geom2))))
            mujoco.mj_contactForce(m, d, i, f6)
            o = out.setdefault(k, [0, 0.0, 0.0]); o[0] += 1; o[1] = min(o[1], con.dist); o[2] += abs(f6[0])
    return out
# which robot pairs CAN collide
cand = set()
for g1 in range(m.ngeom):
    for g2 in range(g1 + 1, m.ngeom):
        b1, b2 = m.geom_bodyid[g1], m.geom_bodyid[g2]
        if b1 == 0 or b2 == 0 or b1 == b2: continue
        if not ((m.geom_contype[g1] & m.geom_conaffinity[g2]) or (m.geom_contype[g2] & m.geom_conaffinity[g1])): continue
        if m.body_parentid[b1] == b2 or m.body_parentid[b2] == b1: continue
        cand.add(tuple(sorted((m.body(b1).name, m.body(b2).name))))
print('collidable robot body pairs:', len(cand), '| model exclude pairs:', os.environ.get('X2_EXCLUDE_PAIRS', 'none'))
tr = Hn.Trial(x, c, ad, m.body_mass.copy(), m.body_inertia.copy())
for v in (0.0, 0.6):
    if os.environ.get('ONLY_WALK') and v == 0: continue
    tr.reset(1000, Hn.NOMINAL)
    cmd = tr.command(); cmd.vel[:] = [v, 0, 0]; cmd.hold = v == 0
    agg = collections.defaultdict(lambda: [0, 0.0, 0.0]); n = 0
    for k in range(int(6 / (tr.sub * X.SIM_DT))):
        tr.d.ctrl[:] = ad.act(tr.view, cmd)
        for _ in range(tr.sub):
            mujoco.mj_step(tr.m, tr.d); n += 1
            for kk, (cnt, dmin, f) in pairs(tr.d).items():
                a = agg[kk]; a[0] += 1; a[1] = min(a[1], dmin); a[2] += f
    print('cmd %.1f m/s, %d physics steps, x moved %.2f m:' % (v, n, tr.d.qpos[0] + 2.0))
    for kk, (cnt, dmin, f) in sorted(agg.items(), key=lambda t: -t[1][0]):
        print('   %-45s in contact %5.1f %% of steps, deepest %.1f mm, mean normal force %.0f N' % (' - '.join(kk), 100 * cnt / n, -1000 * dmin, f / max(cnt, 1)))
