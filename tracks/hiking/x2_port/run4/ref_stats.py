"""Reference statistics for the run-4 fixes: per-joint mean, p1/p5/p95/p99 of the v2 X2 reference, and a 3,000-frame
subsample of joint positions (Isaac order) for the collision census. python ref_stats.py <npz> <out.npz>"""
import sys, numpy as np, re
z = np.load(sys.argv[1], allow_pickle=True)
print({k: getattr(z[k], "shape", None) for k in z.files})
names = [str(n) for n in z["joint_names"]]
jp = None
for k in ("joint_pos", "dof_pos", "joints", "qpos"):
    if k in z.files:
        jp = np.asarray(z[k], float); break
print("joint key shape", jp.shape, "names", len(names))
ISAAC = ["left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint", "left_hip_roll_joint", "right_hip_roll_joint", "waist_pitch_joint",
         "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_roll_joint", "left_knee_joint", "right_knee_joint", "head_yaw_joint",
         "left_shoulder_pitch_joint", "right_shoulder_pitch_joint", "left_ankle_pitch_joint", "right_ankle_pitch_joint", "head_pitch_joint",
         "left_shoulder_roll_joint", "right_shoulder_roll_joint", "left_ankle_roll_joint", "right_ankle_roll_joint", "left_shoulder_yaw_joint",
         "right_shoulder_yaw_joint", "left_elbow_joint", "right_elbow_joint", "left_wrist_yaw_joint", "right_wrist_yaw_joint",
         "left_wrist_pitch_joint", "right_wrist_pitch_joint", "left_wrist_roll_joint", "right_wrist_roll_joint"]
idx = [names.index(n) if n in names else None for n in ISAAC]
J = np.stack([jp[:, i] if i is not None else np.zeros(len(jp)) for i in idx], 1)
for n, c in zip(ISAAC, J.T):
    p = np.percentile(c, [1, 5, 50, 95, 99])
    print("%-28s mean %+.3f  p1 %+.3f p5 %+.3f p50 %+.3f p95 %+.3f p99 %+.3f" % (n, c.mean(), *p))
rng = np.random.default_rng(0)
sel = rng.choice(len(J), size=min(3000, len(J)), replace=False)
np.savez(sys.argv[2], joint_pos=J[sel].astype(np.float32), joint_names=np.array(ISAAC), mean=J.mean(0), p1=np.percentile(J, 1, 0),
         p99=np.percentile(J, 99, 0), p5=np.percentile(J, 5, 0), p95=np.percentile(J, 95, 0))
print("frames", len(J), "saved", len(sel))
