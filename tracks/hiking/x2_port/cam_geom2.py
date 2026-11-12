import sys, numpy as np, mujoco
sys.argv = [sys.argv[0], sys.argv[1]]
import cam_geom as C
m, d, setj = C.x2(); C.foot(m, d)
tz = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")][2]
# head and torso mesh extents in the world at the standing pose
for bn in ("head_pitch_link", "head_yaw_link", "torso_link"):
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, bn); P = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == b and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]; v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            P.append(v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])
    P = np.concatenate(P); print(bn, "x %.3f..%.3f  z %.3f..%.3f" % (P[:, 0].min(), P[:, 0].max(), P[:, 2].min(), P[:, 2].max()))
    if bn == "head_pitch_link":
        sl = P[np.abs(P[:, 2] - 1.24) < 0.01]; print("  head front surface at z 1.24: x max %.3f" % sl[:, 0].max()); sl = P[np.abs(P[:, 2] - 1.20) < 0.01]; print("  at z 1.20: x max %.3f" % sl[:, 0].max())
for x in (0.0488, 0.10):
    for tilt in (58, 60, 62):
        C.cast(m, d, "torso_link", (x, 0.01, 1.24 - tz), tilt, (0, 36), "standing x%.3f %d deg full rows" % (x, tilt))
# mid-stride: left leg forward, right leg back
setj("left_hip_pitch_joint", -0.75); setj("left_knee_joint", 0.55); setj("left_ankle_pitch_joint", -0.1)
setj("right_hip_pitch_joint", 0.05); setj("right_knee_joint", 0.45); setj("right_ankle_pitch_joint", -0.45)
C.foot(m, d)
for x in (0.0488, 0.10):
    for tilt in (58, 60, 62):
        r, out, sh = C.cast(m, d, "torso_link", (x, 0.01, 1.24 - tz), tilt, (0, 36), "mid-stride x%.3f %d deg full rows" % (x, tilt))
C.cast(m, d, "torso_link", (0.0488, 0.01, 1.24 - tz), 48, (18, 36), "mid-stride paper crop 48 deg")
