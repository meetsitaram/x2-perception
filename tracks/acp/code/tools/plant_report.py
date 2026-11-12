"""What the simulator applies per joint, read back from the live MuJoCo Warp model, beside both plant files; the gain
randomisation's coverage of the measured (mc_20261024) gains; per-joint tracking error per clip set from a rollout dump.
    CUDA_VISIBLE_DEVICES= python tools/plant_report.py ROLLOUTS.pt cfg/stage1_sets.json OUT.json"""
import json
import os
import sys

import mujoco
import torch
import warp as wp

sys.path.insert(0, "/mnt/filesystem-g4/tracks/acp/code")
sys.path.insert(0, "/mnt/filesystem-g4/tracks/acp/code/gear_sonic/envs/manager_env/robots")
from acp import env as E  # noqa: E402
from acp.data import Dataset  # noqa: E402
from plant_config import load_plant  # noqa: E402

roll, cfg, out = sys.argv[1:4]
dev = torch.device("cpu")
data = Dataset(cfg, dev, cache=os.path.join(os.path.dirname(cfg), "labels_" + os.path.basename(cfg) + ".pt"))
env = E.AcpEnv(data, 1, eval_mode=True)          # eval mode: nominal values, no randomisation
x2, m = env.x2, env.x2.model
names = x2.joint_names
gain = wp.to_torch(env.mm.actuator_gainprm)[0]
bias = wp.to_torch(env.mm.actuator_biasprm)[0]
arm = wp.to_torch(env.mm.dof_armature)[0, 6:]
fr = wp.to_torch(env.mm.actuator_forcerange)
fr = fr[0] if fr.dim() == 3 else fr
spec = mujoco.MjSpec.from_file(str(E.MJCF_PATH))
ctrl = {}
for a in spec.actuators:
    ctrl[a.target] = [float(a.ctrlrange[0]), float(a.ctrlrange[1])]
v, mc = load_plant("vendor_20260823"), load_plant("mc_20261024")
rows = []
for i, n in enumerate(names):
    kp_live, kd_live = float(gain[i, 0]), float(-bias[i, 2])
    rows.append(dict(joint=n, live_kp=round(kp_live, 2), live_kd=round(kd_live, 3), live_armature=round(float(arm[i]), 5),
                     live_effort=round(float(fr[i, 1]), 1), mjcf_motor_ctrlrange=ctrl.get(n), action_scale=round(float(x2.action_scale[i]), 4),
                     default_q=round(float(x2.default_q[i]), 3),
                     vendor_kp=round(v.kp_for(n), 2), vendor_kd=round(v.kd_for(n), 3), vendor_armature=round(v.armature_for(n), 5), vendor_effort=v.effort_for(n),
                     mc_kp=round(mc.kp_for(n), 2), mc_kd=round(mc.kd_for(n), 3), mc_armature=round(mc.armature_for(n), 5), mc_effort=mc.effort_for(n),
                     mc_action_scale=round(mc.action_scale_for(n), 4), mc_default_q=round(mc.default_pos_for(n), 3),
                     kp_ratio_mc_over_live=round(mc.kp_for(n) / kp_live, 2), kd_ratio_mc_over_live=round(mc.kd_for(n) / kd_live, 2),
                     kp_covered_by_dr=bool(0.9 <= mc.kp_for(n) / kp_live <= 1.1), kd_covered_by_dr=bool(0.9 <= mc.kd_for(n) / kd_live <= 1.1)))
rep = dict(model="live MuJoCo Warp model of acp/env.py (assets.build_x2, plant %s)" % x2.plant.name, timestep=float(m.opt.timestep),
           substeps=E.DECIMATION, control_dt=E.DT, integrator=int(m.opt.integrator),
           actuator="PD position servo: gainprm[0] = kp, biasprm = [0, -kp, -kd], forcerange = +-plant effort; the MJCF's own <motor> actuators "
                    "(and their ctrlrange) are deleted at build time, so the plant's effort is what is applied",
           randomisation="kp, kd and armature each x U(0.9, 1.1) per environment (Table S2)", joints=rows)
# per-joint tracking error per clip set from the rollout dump (newest history step holds q - q0)
d = torch.load(roll, map_location="cpu")
mo = data.motion
hist, alive, clip, set_id = d["hist"].float(), d["alive"][::2][: len(d["hist"])], d["clip"], d["set_id"]
K, N = alive.shape
PROP = env.PROP
dq = hist[:, :, 4 * PROP + 6: 4 * PROP + 6 + len(names)]
f = (mo.clip_start[clip][None, :] + 2 * torch.arange(K)[:, None]).clamp(max=mo.num_frames - 1)
f = torch.minimum(f, mo.clip_end[clip][None, :])
err = (dq + env.default_q - mo.q[f]).abs() * alive.unsqueeze(-1)
groups = {"hip": [i for i, n in enumerate(names) if "hip" in n], "knee": [i for i, n in enumerate(names) if "knee" in n],
          "ankle_pitch": [i for i, n in enumerate(names) if "ankle_pitch" in n], "ankle_roll": [i for i, n in enumerate(names) if "ankle_roll" in n],
          "waist_yaw": [i for i, n in enumerate(names) if n == "waist_yaw_joint"],
          "waist_pitch_roll": [i for i, n in enumerate(names) if n in ("waist_pitch_joint", "waist_roll_joint")],
          "arms_head": [i for i, n in enumerate(names) if any(k in n for k in ("shoulder", "elbow", "wrist", "head"))]}
tr = {}
for s, name in enumerate(d["set_names"]):
    k = set_id == s
    den = alive[:, k].float().sum().clamp(min=1)
    tr[name] = {g: round(float(err[:, k][:, :, idx].sum() / (den * len(idx))), 4) for g, idx in groups.items()}
rep["tracking_error_rad_mean_abs_by_set_and_joint_group"] = tr
rep["tracking_note"] = "paper-eval rollouts (noise and randomisation off, nominal gains), iteration %s; error = |q - q_ref| while the episode is alive" % d["iteration"]
json.dump(rep, open(out, "w"), indent=1)
print("timestep", rep["timestep"], "substeps", rep["substeps"], "plant", x2.plant.name)
print("joint                      live kp/kd/arm/effort | vendor kp/kd | mc kp/kd/arm | mc/live kp kd | scale live/mc | ctrlrange")
for r in rows:
    print(f"{r['joint']:26s} {r['live_kp']:7.2f} {r['live_kd']:6.3f} {r['live_armature']:.4f} {r['live_effort']:5.0f} | {r['vendor_kp']:7.2f} {r['vendor_kd']:6.3f} | "
          f"{r['mc_kp']:6.1f} {r['mc_kd']:5.2f} {r['mc_armature']:.4f} | {r['kp_ratio_mc_over_live']:5.2f} {r['kd_ratio_mc_over_live']:5.2f} | "
          f"{r['action_scale']:.3f}/{r['mc_action_scale']:.3f} | {r['mjcf_motor_ctrlrange']}")
for name, g in tr.items():
    print(f"{name:28s}", g)
