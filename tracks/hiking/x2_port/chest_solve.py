import sys, math, numpy as np, mujoco
sys.argv = [sys.argv[0], sys.argv[1]]
import cam_geom as C
m, d, setj = C.x2(); C.foot(m, d)
b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link"); tz = d.xpos[b][2]
P = []
for g in range(m.ngeom):
    if m.geom_bodyid[g] == b and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
        mid = m.geom_dataid[g]; v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        P.append(v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])
P = np.concatenate(P)
toe = max(d.geom_xpos[g][0] for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE)
print("toe tip x %.3f; torso_link origin z %.3f" % (toe, tz))
print("chest front profile at |y| < 0.03 (world z : front x):")
front = {}
for z in np.arange(0.80, 1.16, 0.02):
    s = P[(np.abs(P[:, 2] - z) < 0.01) & (np.abs(P[:, 1]) < 0.03)]
    if len(s): front[round(z, 2)] = s[:, 0].max()
print("  " + "  ".join("%.2f:%.3f" % kv for kv in front.items()))
def solve(xc, f1=0.27, d1=0.985, vfov=58.0, bottom_margin=0.0):
    th = math.tan(math.radians(vfov / 2)); a_bot = math.degrees(math.atan(th)); a1 = math.degrees(math.atan((1 - 2 * f1) * th))
    best = None
    for T in np.arange(40, 75, 0.05):
        for h in np.arange(0.7, 1.25, 0.005):
            # bottom edge: ground point at toe + margin (toe tips just inside the bottom edge)
            db = T + a_bot; xb = xc + (h / math.tan(math.radians(db)) if db < 90 else -h * math.tan(math.radians(db - 90)))
            x1 = xc + h / math.tan(math.radians(T - a1))
            e = (xb - (toe - bottom_margin)) ** 2 + (x1 - (toe + d1)) ** 2
            if best is None or e < best[0]: best = (e, T, h)
    e, T, h = best
    far = xc + h / math.tan(math.radians(T - a_bot)) - toe
    return T, h, far, math.sqrt(e)
for proud in (0.02, 0.03, 0.04):
    for it in range(3):
        h = 0.95 if it == 0 else h
        zkey = min(front, key=lambda z: abs(z - h)); xc = front[zkey] + proud
        T, h, far, err = solve(xc)
    print("lens %.0f cm proud of the shell: lens x %.3f (%.3f m %s the toe tips), height %.3f m, tilt %.1f deg below horizontal, far edge %.2f m ahead of the toes, fit error %.3f m" % (proud * 100, xc, abs(xc - toe), "ahead of" if xc > toe else "behind", h, T, far, err))
for back in (0.12,):
    T, h, far, err = solve(toe - back); print("coordinator's assumption, lens 0.12 m behind the toes: height %.3f, tilt %.1f, far %.2f, err %.3f" % (h, T, far, err))
