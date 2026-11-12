"""Zero-action rest pose of the X2 in MuJoCo (stock MJCF, sphere soles, CPU) under both plants: pelvis and torso pitch, COM over the feet."""
import sys, math, json, numpy as np, mujoco
m = mujoco.MjModel.from_xml_path(sys.argv[1]); m.opt.timestep = 0.005
a = json.load(open(sys.argv[2]))["joints"]
names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt) if m.jnt_type[j] == 3]
qa = np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names]); da = np.array([m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names])
q0 = np.array([a[n]["default_pos"] for n in names]); eff = np.array([a[n]["effort_limit"] for n in names]); arm = np.array([a[n]["armature"] for n in names])
W = 2 * math.pi * 10
plants = {"mc_20261024": (np.array([a[n]["kp"] for n in names]), np.array([a[n]["kd"] for n in names])), "armature rule": (arm * W * W, 4 * arm * W)}
m.dof_armature[da] = arm
m.actuator_gainprm[:, 0] = 0; m.actuator_biasprm[:, :] = 0   # disable the MJCF's own actuators
def pitch(d, body):
    R = d.xmat[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)].reshape(3, 3); return math.degrees(math.atan2(-R[2, 0], R[0, 0]))
for name, (kp, kd) in plants.items():
    d = mujoco.MjData(m); d.qpos[:] = 0; d.qpos[3] = 1; d.qpos[qa] = q0; mujoco.mj_forward(m, d)
    zmin = min(d.geom_xpos[g][2] - m.geom_size[g][0] for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE)
    d.qpos[2] -= zmin - 0.002; mujoco.mj_forward(m, d)
    sx = [d.geom_xpos[g][0] for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE]
    com0 = d.subtree_com[1][0]
    if name == "mc_20261024": print("default pose, before any motion: COM x %.3f m; sole contact points span x %.3f..%.3f (centre %.3f); ankle x %.3f" % (com0, min(sx), max(sx), (min(sx) + max(sx)) / 2, d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")][0]))
    fell = None
    for i in range(int(6 / 0.005)):
        tau = np.clip(kp * (q0 - d.qpos[qa]) - kd * d.qvel[da], -eff, eff)
        d.qfrc_applied[:] = 0; d.qfrc_applied[da] = tau
        mujoco.mj_step(m, d)
        if d.qpos[2] < 0.4 and fell is None: fell = i * 0.005
    err = d.qpos[qa] - q0; big = sorted(zip(np.abs(err), names), reverse=True)[:4]
    print("%-14s after 6 s of zero actions: pelvis pitch %+.1f deg, torso pitch %+.1f deg (negative = leaning back), pelvis height %.3f m, moved x %+.3f m, %s; largest joint sag: %s" % (
        name, pitch(d, "pelvis"), pitch(d, "torso_link"), d.qpos[2], d.qpos[0], "fell at %.1f s" % fell if fell else "did not fall", ", ".join("%s %.3f rad" % (n.replace("_joint", ""), e) for e, n in big)))
