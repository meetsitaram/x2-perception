"""Depth-camera ground footprint by ray casting in MuJoCo (CPU). Rays: the task's pinhole pattern 64x36, 89.51 x 58.29 deg."""
import math, sys, re, numpy as np, mujoco
W, H = 64, 36
FX, FY = math.tan(math.radians(89.51) / 2), math.tan(math.radians(58.29) / 2)
def rays(tilt_deg):
    u = (np.arange(W) + 0.5) / W * 2 - 1; v = 1 - (np.arange(H) + 0.5) / H * 2      # v=+1 top row
    uu, vv = np.meshgrid(u, v)
    d = np.stack([np.ones_like(uu), -uu * FX, vv * FY], -1)                          # camera frame x fwd, y left, z up
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    t = math.radians(tilt_deg); c, s = math.cos(t), math.sin(t)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])                                 # pitch down by tilt about y
    return d @ R.T
def x2():
    m = mujoco.MjModel.from_xml_path(sys.argv[1]); d = mujoco.MjData(m)
    def setj(n, v): d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]] = v
    for s in ("left", "right"):
        setj(s + "_hip_pitch_joint", -0.312); setj(s + "_knee_joint", 0.669); setj(s + "_ankle_pitch_joint", -0.363); setj(s + "_elbow_joint", -0.6)
    setj("left_shoulder_roll_joint", 0.2); setj("right_shoulder_roll_joint", -0.2); setj("left_shoulder_pitch_joint", 0.2); setj("right_shoulder_pitch_joint", 0.2)
    return m, d, setj
def foot(m, d):
    mujoco.mj_forward(m, d)
    zmin = min(d.geom_xpos[g][2] - m.geom_size[g][0] for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE)
    d.qpos[2] -= zmin; mujoco.mj_forward(m, d)
def cast(m, d, body, pos, tilt, rows, label, quiet=False):
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
    Rb = d.xmat[b].reshape(3, 3); o = d.xpos[b] + Rb @ np.array(pos)
    yaw = math.atan2(Rb[1, 0], Rb[0, 0]); cy, sy = math.cos(yaw), math.sin(yaw)       # ray_alignment "yaw": only the body's yaw
    D = rays(tilt) @ np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]).T
    toe = max(d.geom_xpos[g][0] for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE)
    grp = np.array([0, 1, 1, 1, 0, 0], np.uint8)                                      # visual meshes (group 1), collision 2/3; floor handled analytically
    gid = np.zeros(1, np.int32)
    out = np.full((H, W), np.nan); self_hit = np.zeros((H, W), bool)
    for r in range(H):
        for c in range(16, W - 16):
            dist = mujoco.mj_ray(m, d, o, D[r, c], grp, 1, -1, gid)
            tg = -o[2] / D[r, c, 2] if D[r, c, 2] < 0 else np.inf
            if dist >= 0.1 and dist < tg: self_hit[r, c] = True
            elif np.isfinite(tg): out[r, c] = o[0] + tg * D[r, c, 0] - toe
    sel = slice(rows[0], rows[1]); g = out[sel, 16:48]; sh = self_hit[sel, 16:48]
    cen = out[sel, 31:33].mean(1)
    res = dict(cam_height=o[2], cam_x_from_toe=o[0] - toe, near=np.nanmin(g), far=np.nanmax(cen[np.isfinite(cen)]) if np.isfinite(cen).any() else np.nan,
               self_share=sh.mean(), sky=np.isnan(g).mean() - sh.mean())
    if not quiet:
        print("%-44s cam z %.3f m, %.3f m %s the toe tips | ground seen from %+.2f m to %+.2f m ahead of the toes (centre columns) | own body %.0f%% of image, no ground hit %.0f%%" % (
            label, o[2], abs(res["cam_x_from_toe"]), "behind" if res["cam_x_from_toe"] < 0 else "ahead of", res["near"], res["far"], 100 * res["self_share"], 100 * max(res["sky"], 0)))
    return res, out, self_hit
if __name__ == "__main__":
    m, d, setj = x2(); foot(m, d)
    b = lambda n: d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)]
    print("X2 standing: pelvis z %.3f torso_link z %.3f head_pitch z %.3f; hip axis x %.3f, torso x %.3f, toe tip x %.3f" % (b("pelvis")[2], b("torso_link")[2], b("head_pitch_link")[2], b("left_hip_pitch_link")[0], b("torso_link")[0], max(d.geom_xpos[g][0] for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE)))
    tz = b("torso_link")[2]
    cast(m, d, "head_pitch_link", (0.05761, -0.011183, -0.04837), 40, (18, 36), "X2 head camera 40 deg, paper crop (v1)")
    for zc in (1.24,):
        for xoff in (0.0488, 0.08, 0.10, 0.12):
            cast(m, d, "torso_link", (xoff, 0.01, zc - tz), 48, (18, 36), "X2 torso x%.3f z%.2f 48 deg, paper crop" % (xoff, zc))
            for tilt in (55, 58, 60):
                cast(m, d, "torso_link", (xoff, 0.01, zc - tz), tilt, (0, 36), "X2 torso x%.3f z%.2f %d deg, full rows" % (xoff, zc, tilt))
