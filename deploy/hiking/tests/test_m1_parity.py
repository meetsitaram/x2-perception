"""M1 offline parity (deploy/hiking/PLAN.md): the deploy core against the eval adapter on the same MuJoCo states.

  1. policy.onnx through core.Policy reproduces obs_check.npz (synthetic inputs; ONNX vs torch of the export).
  2. Closed loop on flat ground in the eval's MuJoCo world, driven by the eval adapter (eval/adapters/hiking.py): at
     every control step the core rebuilds the observation from what the ROBOT publishes - the torso IMU (orientation +
     torso-frame gyro), joint states by name, a raw 64 x 36 z-depth image - and must match the adapter's observation
     (built from the pelvis directly) and action. Commands: stand, walk 0.5 m/s, walk + turn, turn in place, 0.8 m/s.
     The waist moves throughout, so the torso -> pelvis conversion is exercised.

Done = max |obs error| < 1e-5, max |action error| < 1e-4 (clip off for the comparison; the clip is tested separately).
Run:  PA_ROBOT_DIR=<dir with mjcf/x2_ultra.xml> HIKING_MODEL_DIR=deploy/hiking/models/x2-hiking-in-the-wild \
      python deploy/hiking/tests/test_m1_parity.py
"""
from __future__ import annotations

import os
import sys
import pathlib

import mujoco
import numpy as np

HERE = pathlib.Path(__file__).resolve()
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "eval"))
sys.path.insert(0, str(HERE.parents[1]))

from hiking_deploy import core                       # noqa: E402
from hiking_deploy.sim_depth import ChestDepthCaster  # noqa: E402

MODEL_DIR = os.environ.get("HIKING_MODEL_DIR", str(HERE.parents[1] / "models" / "x2-hiking-in-the-wild"))
OBS_TOL, ACT_TOL = 1e-5, 1e-4


def test_onnx_obs_check():
    z = np.load(os.path.join(MODEL_DIR, "obs_check.npz"))
    pol = core.Policy(MODEL_DIR)
    err = max(float(np.abs(pol(o) - a).max()) for o, a in zip(z["obs"], z["actions_onnx"]))
    err_t = max(float(np.abs(pol(o) - a).max()) for o, a in zip(z["obs"], z["actions_torch"]))
    print("[obs_check] core.Policy vs recorded onnx max err %.2e, vs torch %.2e" % (err, err_t))
    assert err < 1e-5 and err_t < 1e-4


def command(t):
    if t < 1.0:
        return np.zeros(3)
    if t < 4.0:
        return np.array([0.5, 0.0, 0.0])
    if t < 7.0:
        return np.array([0.5, 0.1, 0.5])
    if t < 9.0:
        return np.array([0.0, 0.0, -0.8])
    return np.array([0.8, 0.0, 0.0])


def test_closed_loop(seconds=11.0):
    import harness as H
    import x2 as X
    from adapters.hiking import HikingHandover

    ad = HikingHandover(MODEL_DIR)
    names = X.joint_names_of(X.robot_dir() / "mjcf" / "x2_ultra.xml")
    x = X.build(None, ad.plant(names))
    tr = H.Trial(x, None, ad, x.model.body_mass.copy(), x.model.body_inertia.copy())
    tr.reset(0, dict(H.NOMINAL, start_xy=0.0, start_yaw_deg=0.0))
    m, d, view = x.model, tr.d, tr.view
    ad.reset(view, np.random.default_rng(0))

    ctl = core.HikingController(MODEL_DIR, clip=None)
    caster = ChestDepthCaster(m, ctl.card.depth, geom_groups=(1, 4, 5))
    torso = m.body("torso_link").id
    v6 = np.zeros(6)
    worst = dict(obs=0.0, act=0.0, gyro=0.0, grav=0.0, depth=0.0, waist=0.0)
    steps = int(seconds / ad.control_dt)
    for k in range(steps):
        cmd = command(k * ad.control_dt)
        # reference (eval adapter, pelvis read directly)
        obs_ref = ad.observe(view, cmd)
        a_ref = ad.policy(obs_ref)
        ad.a_prev = a_ref
        # deploy core, from what the robot publishes
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_XBODY, torso, v6, 1)
        q, qd = view.q(), view.qd()
        qn, qdn = dict(zip(names, q)), dict(zip(names, qd))
        frame = core.preprocess_depth(caster.render(d), ctl.card.depth)
        targets, info = ctl.step(d.xmat[torso].reshape(3, 3).copy(), v6[:3].copy(), qn, qdn, cmd, frame)
        _, wp_true = view.body_velocity("pelvis")
        worst["obs"] = max(worst["obs"], float(np.abs(info["obs"] - obs_ref).max()))
        worst["act"] = max(worst["act"], float(np.abs(info["a_raw"] - a_ref).max()))
        worst["gyro"] = max(worst["gyro"], float(np.abs(info["omega_pelvis"] - wp_true).max()))
        worst["grav"] = max(worst["grav"], float(np.abs(core.projected_gravity(info["R_pelvis"]) - view.projected_gravity()).max()))
        worst["depth"] = max(worst["depth"], float(np.abs(frame - ad.depth_frame(view)).max()))
        worst["waist"] = max(worst["waist"], float(np.abs(q[[names.index(n) for n in core.WAIST]]).max()))
        q_tgt = ad.q0_31.copy()
        q_tgt[ad.i29] += ad.scale * a_ref
        _, tg_ref = ctl.amap.apply(a_ref)  # same action through the deploy mapping
        tq = np.array([tg_ref[n] for n in names])
        assert np.abs(tq - q_tgt).max() < 1e-9, "target mapping differs"
        if k < 3 or worst["obs"] > OBS_TOL:
            d_obs = np.abs(info["obs"] - obs_ref)
            if d_obs.max() > OBS_TOL:
                print("step %d obs err %.2e at idx %s" % (k, d_obs.max(), np.flatnonzero(d_obs > OBS_TOL)[:12]))
                break
        d.ctrl[:] = q_tgt
        for _ in range(tr.sub):
            mujoco.mj_step(m, d)
        mujoco.mj_forward(m, d)  # mj_step leaves xmat/cvel one substep behind qpos; the bridge publishes consistent state
        if d.xpos[m.body("pelvis").id][2] < 0.4:
            raise AssertionError("fell at t=%.2f s" % d.time)
    dist = float(np.linalg.norm(d.qpos[:2]))
    print("[closed loop] %d steps, %.1f m travelled, max |waist q| %.3f rad" % (steps, dist, worst["waist"]))
    print("[closed loop] max err: obs %.2e  action %.2e  pelvis gyro %.2e  gravity %.2e  depth %.2e"
          % (worst["obs"], worst["act"], worst["gyro"], worst["grav"], worst["depth"]))
    assert worst["obs"] < OBS_TOL and worst["act"] < ACT_TOL, worst


def test_clip():
    card = core.Card.load(MODEL_DIR)
    am = core.ActionMap(card, clip=20.0, limits={"left_knee_joint": (0.0, 0.1)})
    a = np.zeros(29)
    a[card.action_order.index("waist_pitch_joint")] = 40.0
    a[card.action_order.index("left_hip_pitch_joint")] = -35.0
    a[card.action_order.index("left_knee_joint")] = 15.0
    ac, tg = am.apply(a)
    assert ac[card.action_order.index("waist_pitch_joint")] == 40.0
    assert ac[card.action_order.index("left_hip_pitch_joint")] == -20.0
    assert tg["left_knee_joint"] <= 0.1 and tg["head_yaw_joint"] == card.default_pose["head_yaw_joint"]
    try:
        am.apply(np.full(29, np.nan))
        raise AssertionError("nan accepted")
    except FloatingPointError:
        pass
    print("[clip] ok")


if __name__ == "__main__":
    test_onnx_obs_check()
    test_clip()
    test_closed_loop()
    print("M1 PARITY: PASS")
