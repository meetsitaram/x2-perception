"""Check the Hiking adapter (eval monitor, 2026-11-12).

    python eval/tools/hiking_check.py HANDOVER_DIR PORT_DIR      (PORT_DIR: a COPY of the track's sim2sim folder; needs pyyaml)

1. ONNX: the adapter's policy call on the card's obs_check observations reproduces actions_onnx (that file is synthetic:
   it checks the export, not the observation).
2. Observation: the track's own MuJoCo port (X2Sim of sim2sim_mujoco_x2.py, run's env.yaml + card) and the adapter, on the
   same robot states (perturbed poses, velocities, commands, actions over 60 control steps, flat ground), must build the
   same 10,016 numbers: proprio terms term by term, depth stack frame by frame. Differences in the depth come from the
   two robot models (training URDF meshes in the port; the eval MJCF here) where a ray hits the robot's own body.
"""
import os
import pathlib
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
d, port = sys.argv[1], sys.argv[2]
sys.path.insert(0, port)
import sim2sim_mujoco_x2 as P  # noqa: E402
import x2 as X  # noqa: E402
from adapters.base import View  # noqa: E402
from adapters.hiking import HikingHandover  # noqa: E402

ad = HikingHandover(d)
z = np.load(os.path.join(d, "obs_check.npz"))
a = np.array([ad.policy(o) for o in z["obs"]])
print("onnx: max |adapter - actions_onnx| %.3g, vs actions_torch %.3g" % (np.abs(a - z["actions_onnx"]).max(), np.abs(a - z["actions_torch"]).max()))

ec = P.EnvConfig(os.path.join(d, "run_params"), os.path.join(d, "adapter_card.json"))
sim = P.X2Sim(ec, P.Terrain("flat"))
names = X.joint_names_of(X.robot_dir() / "mjcf" / "x2_ultra.xml")
x = X.build(None, ad.plant(names))
data = mujoco.MjData(x.model)
view = View(x, data, None)
ad.reset(view, None)
assert list(ec.joints) == ad.jo and list(ec.action_joints) == ad.ao
rng = np.random.default_rng(0)
sim.reset((0.0, 0.0, ec.init_z))
lay = ec.layout
worst = {k: 0.0 for k in list(lay)}
rh_port, rh_ad = [], []
for k in range(60):
    yaw = 0.3 * np.sin(0.1 * k)
    qt = np.array([np.cos(yaw / 2), 0.03 * rng.standard_normal(), 0.05 + 0.03 * rng.standard_normal(), np.sin(yaw / 2)])
    qt /= np.linalg.norm(qt)
    jp = ec.default_q + 0.15 * rng.standard_normal(31)
    jv = 1.0 * rng.standard_normal(31)
    w = 0.5 * rng.standard_normal(3)
    root = np.array([0.02 * k, 0.0, ec.init_z + 0.02 * rng.standard_normal()])
    sim.set_state(root, qt, joint_pos=jp, joint_vel=jv, root_ang_vel_b=w)
    data.qpos[:3], data.qpos[3:7] = root, qt
    data.qvel[:] = 0
    data.qvel[3:6] = w
    data.qpos[x.qadr[ad.i31]] = jp
    data.qvel[x.dadr[ad.i31]] = jv
    mujoco.mj_forward(x.model, data)
    cmd = np.array([0.7, 0.0, 0.3 * np.cos(0.2 * k)]) if k else np.zeros(3)
    o_p = sim.observe(cmd)
    o_a = ad.observe(view, cmd)
    rh_port.append(sim.cam.robot_hits); rh_ad.append(ad.robot_hits)
    for name, (s, n, _) in lay.items():
        worst[name] = max(worst[name], float(np.abs(o_p[s:s + n] - o_a[s:s + n]).max()))
    act = rng.uniform(-1, 1, 29)
    sim.last_action = act.copy(); ad.a_prev = act.copy()
for name, v in worst.items():
    print("obs %-18s max |port - adapter| %.3g" % (name, v))
print("rays on the robot's own body per frame: port median %d (max %d), adapter median %d (max %d)" % (
    np.median(rh_port), max(rh_port), np.median(rh_ad), max(rh_ad)))
ok = all(worst[n] < 1e-4 for n in worst if n != "depth_image") and worst["depth_image"] < 0.02
print("PASS" if ok else "CHECK", "(proprio < 1e-4, depth < 0.02 of the 3 m range)")
