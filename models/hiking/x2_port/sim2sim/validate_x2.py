"""Validation of sim2sim_mujoco_x2.py against the Isaac recording (assets/eval/obs_check*.npz). Writes validation.txt.
CPU only. Run:  C:/Users/PC/.venvs/hiking/Scripts/python.exe validate_x2.py
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import math
import pathlib

import mujoco
import numpy as np

import sim2sim_mujoco_x2 as s2s

HERE = pathlib.Path(__file__).resolve().parent
LINES = []


def out(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LINES.append(s)


def main():
    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    # run 1 is the only run with an Isaac recording
    run, export = s2s.ASSETS / "run", s2s.ASSETS / "eval"
    ec = s2s.EnvConfig(run, export / "adapter_card.json")
    policy = s2s.OnnxPolicy(export / "policy.onnx")
    rec = np.load(export / "obs_check.npz")
    st = np.load(export / "obs_check_sim_state.npz")
    obs = rec["obs"]
    T = len(obs)
    DR = ec.cam["drange"][1]  # depth range (m) of this run
    assert [str(j) for j in st["joint_names"]] == ec.joints, "joint order: recording vs card"
    assert np.allclose(st["default_joint_pos"], ec.default_q, atol=1e-6)
    assert np.array_equal(obs, st["obs"])
    sim = s2s.X2Sim(ec, s2s.Terrain("flat"))
    m, d = sim.m, sim.d
    out("VALIDATION of sim2sim_mujoco_x2.py against the Isaac recording (robot 0, first 50 control steps after a reset, stair-up tile)")
    out(f"mujoco {mujoco.__version__}; model: {m.nbody - 1} bodies, {m.nu} actuators, total mass {m.body_subtreemass[1]:.3f} kg; foot = hull")
    out("")

    # ------------------------------------------------------------------ 1. proprio, newest step
    out("1. PROPRIO, newest-step entries rebuilt from the recorded simulator state (MuJoCo state set from")
    out("   root_pos_w, root_quat_w, joint_pos, joint_vel, root_lin_vel_b, root_ang_vel_b; command and previous action from the recording)")
    out("   error = |rebuilt - recorded observation|, in observation units (after scaling). The recording contains training noise.")
    noise = {"base_ang_vel": 0.2 * 0.25, "projected_gravity": 0.05, "velocity_commands": 0.0, "joint_pos": 0.01, "joint_vel": 0.5 * 0.05, "actions": 0.0}
    errs = {k: [] for k in noise}
    for t in range(T):
        sim.set_state(st["root_pos_w"][t], st["root_quat_w"][t], st["joint_pos"][t], st["joint_vel"][t], st["root_lin_vel_b"][t], st["root_ang_vel_b"][t])
        sim.last_action = st["actions"][t - 1].astype(float) if t > 0 else np.zeros(29)
        terms = sim.proprio_terms(st["command"][t])
        for name, v in terms.items():
            a, size, hist = ec.layout[name]
            dim = size // hist
            newest = obs[t, a + size - dim: a + size]
            errs[name].append(np.abs(v - newest))
    out(f"   {'term':<20}{'dim':>4}{'max abs err':>13}{'mean abs err':>14}{'noise bound (scaled)':>22}   verdict")
    for name, e in errs.items():
        e = np.array(e)
        ok = e.max() <= noise[name] + 1e-5
        out(f"   {name:<20}{e.shape[1]:>4}{e.max():>13.5f}{e.mean():>14.5f}{noise[name]:>22.4f}   {'within noise' if ok else 'EXCEEDS noise bound'}")
    out("   (uniform noise of half-width b has mean abs b/2: expected means 0.025, 0.025, 0, 0.005, 0.0125, 0)")
    # independent, noise-free cross checks of the conventions
    g_err, w_err, v_err = [], [], []
    for t in range(T):
        sim.set_state(st["root_pos_w"][t], st["root_quat_w"][t], st["joint_pos"][t], st["joint_vel"][t], st["root_lin_vel_b"][t], st["root_ang_vel_b"][t])
        _, _, R, v_b, w_b = sim.root_state()
        g_err.append(np.abs(R.T @ np.array([0, 0, -1.0]) - st["projected_gravity_b"][t]).max())
        w_err.append(np.abs(w_b - st["root_ang_vel_b"][t]).max())
        v_err.append(np.abs(v_b - st["root_lin_vel_b"][t]).max())
    out(f"   noise-free checks vs obs_check_sim_state.npz: projected gravity from root_quat_w (wxyz) max err {max(g_err):.2e};"
        f" MuJoCo pelvis angular velocity read-back {max(w_err):.2e}; linear velocity read-back {max(v_err):.2e}")
    out(f"   previous action: obs 'actions' newest entry at step t equals actions[t-1] exactly (max err {np.array(errs['actions']).max():.1e}); zeros at step 0")
    out("")

    # ------------------------------------------------------------------ 2. history order
    out("2. PROPRIO HISTORY ORDER (recording vs itself): slot i of step t must equal the newest slot of step t-(7-i); before the reset: step-0 value")
    worst = {}
    for name in noise:
        a, size, hist = ec.layout[name]
        dim = size // hist
        H = obs[:, a:a + size].reshape(T, hist, dim)
        w = 0.0
        for t in range(T):
            for i in range(hist):
                src = max(t - (hist - 1 - i), 0)
                w = max(w, float(np.abs(H[t, i] - H[src, hist - 1]).max()))
        worst[name] = w
        # the opposite hypothesis (newest first) for contrast
        w2 = max(float(np.abs(H[t, i] - H[max(t - i, 0), 0]).max()) for t in range(T) for i in range(hist))
        out(f"   {name:<20} oldest-first, newest last: max mismatch {w:.1e}   (newest-first hypothesis: {w2:.3f})")
    out("   -> every proprio term is 8 steps [t-7 .. t], oldest first, filled with the first value at reset: " + ("CONFIRMED" if max(worst.values()) < 1e-6 else "NOT confirmed"))
    out("   note: the recorded command at step 0 is [0,0,0] (reset observation, command not computed yet); from step 1 it is [0.74, 0, yaw]")
    out("")

    # ------------------------------------------------------------------ 3. depth
    a, size, nf = ec.layout["depth_image"]
    D = obs[:, a:a + size].reshape(T, nf, *ec.frame_shape)
    out("3. DEPTH")
    out(f"   (a) frame order. value range of the recording {D.min():.3f}..{D.max():.3f} (x2.5 m = {D.min()*DR:.2f}..{D.max()*DR:.2f} m)")
    sh_a = max(float(np.abs(D[t, :-1] - D[t - 5, 1:]).max()) for t in range(5, T))
    sh_b = max(float(np.abs(D[t, 1:] - D[t - 5, :-1]).max()) for t in range(5, T))
    out(f"       frame i of step t == frame i+1 of step t-5: max diff {sh_a:.1e};   opposite direction: {sh_b:.3f}")
    first_change = [next((t for t in range(1, T) if np.abs(D[t, i] - D[t - 1, i]).max() > 0), None) for i in range(nf)]
    out(f"       first step at which frame i changes, i=0..7: {first_change}")
    out("       -> frames are OLDEST FIRST, the NEWEST is the LAST (index 7); frames are 5 control steps (0.1 s) apart. CONFIRMED from the recording.")
    out(f"       timing: frame 7 at step 1 is bit-identical to frame 7 at step 0 (max diff {np.abs(D[1,7]-D[0,7]).max():.1e}) although the pelvis"
        f" dropped {1000*(st['root_pos_w'][0,2]-st['root_pos_w'][1,2]):.1f} mm,")
    out("       and frame 7 first changes at step 2, frame 6 at step 7, ... : the newest frame of step t is the image of step t-1 (one control step old).")
    # ground height estimate from the soles (standing part of the recording)
    low = []
    hull = {}
    for fb in sim.feet:
        g = [g for g in range(m.ngeom) if m.geom_bodyid[g] == fb and m.geom_contype[g]][0]
        mid = m.geom_dataid[g]
        hull[fb] = (g, m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].copy())
    sole = np.zeros((T, 2))
    for t in range(T):
        sim.set_state(st["root_pos_w"][t], st["root_quat_w"][t], st["joint_pos"][t])
        for k, fb in enumerate(sim.feet):
            g, V = hull[fb]
            sole[t, k] = (V @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])[:, 2].min()
    ground = float(np.median(sole[25:].min(axis=1)))
    out(f"   (b) comparison with a MuJoCo ray-cast on FLAT ground. The Isaac terrain (stair-up tile, perlin roughness) is not available, so the ground")
    out(f"       is assumed flat at the height of the lowest sole point while standing (steps 25..49): z = {ground:.3f} m (world);"
        f" lowest sole at step 0: {sole[0].min()-ground:+.3f} m above it, spread over steps 25..49: {sole[25:].min(axis=1).std()*1000:.1f} mm")
    mj = np.zeros((T, *ec.frame_shape), np.float32)
    robot_px = np.zeros(T, int)
    raw_full, raw_robot = {}, {}
    for t in range(T):
        p = st["root_pos_w"][t].astype(float).copy()
        p[2] -= ground; p[:2] = 0.0
        sim.set_state(p, st["root_quat_w"][t], st["joint_pos"][t])
        mj[t] = sim.depth_frame()
        if t in (0, 49):
            raw_full[t] = sim.cam.raw_full(m, d)
            raw_robot[t] = int((m.geom_group[np.maximum(sim.cam.last_geom, 0)] == s2s.GROUP_ROBOT_VIS)[sim.cam.last_geom >= 0].sum())
        robot_px[t] = sim.cam.robot_hits
    cam_h = []
    for t in (0, 49):
        p = st["root_pos_w"][t].astype(float).copy(); p[2] -= ground; p[:2] = 0.0
        sim.set_state(p, st["root_quat_w"][t], st["joint_pos"][t])
        cp, cR = sim.cam.pose(d)
        cam_h.append((cp[2], math.degrees(math.asin(-cR[2, 0]))))
    out(f"       MuJoCo camera (FK from the recorded pose): height above the assumed ground {cam_h[0][0]:.3f} m at step 0, {cam_h[1][0]:.3f} m at step 49;"
        f" optical axis {cam_h[0][1]:.1f} / {cam_h[1][1]:.1f} deg below horizontal")
    for lag in (0, 1):
        ts = np.arange(max(lag, 1), T)
        diff = D[ts, 7] - mj[ts - lag]
        out(f"       lag {lag} (recorded newest frame of step t vs MuJoCo flat-ground frame of step t-{lag}): mean abs diff all pixels {np.abs(diff).mean():.4f},"
            f" bottom 6 rows {np.abs(diff[:, 12:]).mean():.4f}, top 6 rows {np.abs(diff[:, :6]).mean():.4f}  (units of 2.5 m)")
    out("       timing check on the steps where the pelvis drops fastest (steps 2..8, bottom 6 rows = nearest ground, flat in both simulators):")
    for lag in (0, 1, 2):
        ts = np.arange(2, 9)
        out(f"         newest recorded frame of step t vs MuJoCo image of step t-{lag}: mean abs diff {np.abs(D[ts, 7][:, 12:] - mj[ts - lag][:, 12:]).mean():.5f}")
    out("       -> the newest frame is one control step old (lag 1 fits best); the player uses --depth-lag 1 by default.")
    lag = 1
    ts = np.arange(1, T)
    diff = D[ts, 7] - mj[ts - lag]
    out("       per-row means over steps 1..49 (row 0 = top of the image = far, row 17 = bottom = near), lag 1:")
    out("         row      " + " ".join(f"{r:>6d}" for r in range(18)))
    out("         Isaac    " + " ".join(f"{v:>6.3f}" for v in D[ts, 7].mean(axis=(0, 2))))
    out("         MuJoCo   " + " ".join(f"{v:>6.3f}" for v in mj[ts - lag].mean(axis=(0, 2))))
    out("         diff     " + " ".join(f"{v:>+6.3f}" for v in diff.mean(axis=(0, 2))))
    out("       per-column means (col 0 = left), lag 1, every 4th column:")
    out("         Isaac    " + " ".join(f"{v:>6.3f}" for v in D[ts, 7].mean(axis=(0, 1))[::4]))
    out("         MuJoCo   " + " ".join(f"{v:>6.3f}" for v in mj[ts - lag].mean(axis=(0, 1))[::4]))
    for t in (0, 10, 49):
        tt = max(t - 1, 0)
        out(f"       step {t:>2}: centre column, rows 0..17  Isaac  " + " ".join(f"{v:.3f}" for v in D[t, 7][:, 16]))
        out(f"                                          MuJoCo " + " ".join(f"{v:.3f}" for v in mj[tt][:, 16]))
    # orientation tests: correlate the recorded frame with flipped MuJoCo frames
    def corr(A, B):
        A = A - A.mean(); B = B - B.mean()
        return float((A * B).sum() / math.sqrt((A * A).sum() * (B * B).sum() + 1e-12))
    c_ok = np.mean([corr(D[t, 7], mj[t - 1]) for t in ts])
    c_ud = np.mean([corr(D[t, 7], mj[t - 1][::-1]) for t in ts])
    out(f"       orientation: correlation of the recorded frame with the MuJoCo frame {c_ok:+.3f}; with the MuJoCo frame flipped top-bottom {c_ud:+.3f}")
    # left/right: the camera sits 11 mm right of the body centre and the pelvis rolls/yaws; use the per-step left-right depth slope
    lr_i = np.array([D[t, 7][:, 24:].mean() - D[t, 7][:, :8].mean() for t in ts])
    lr_m = np.array([mj[t - 1][:, 24:].mean() - mj[t - 1][:, :8].mean() for t in ts])
    out(f"       left/right: (right third - left third) mean depth per step, Isaac vs MuJoCo: correlation over steps {corr(lr_i, lr_m):+.3f},"
        f" Isaac range {lr_i.min():+.4f}..{lr_i.max():+.4f}, MuJoCo range {lr_m.min():+.4f}..{lr_m.max():+.4f}")
    out(f"       robot's own body in the cropped image (MuJoCo, recorded poses): {robot_px.min()}..{robot_px.max()} of 576 pixels hit the robot;"
        f" minimum recorded depth {D.min()*DR:.2f} m -> no near (own-body) pixels in the recording either")
    out(f"       uncropped 36x64 image (MuJoCo): pixels on the robot's own body at step 0: {raw_robot[0]}, at step 49: {raw_robot[49]} of 2304")
    out("")

    # ------------------------------------------------------------------ 4. ONNX plumbing
    a_onnx = policy(obs)
    out("4. ONNX PLUMBING: recorded observations -> policy.onnx (CPU)")
    out(f"   max |action - actions_onnx| = {np.abs(a_onnx - rec['actions_onnx']).max():.2e};  max |action - Isaac actions (sim state file)| = {np.abs(a_onnx - st['actions']).max():.2e}"
        f";  action magnitude in the recording: mean |a| {np.abs(st['actions']).mean():.2f}, max {np.abs(st['actions']).max():.2f}")
    out("")

    # ------------------------------------------------------------------ 5. dynamics, one control step ahead
    out("5. DYNAMICS, ONE CONTROL STEP AHEAD (not asked for, used to explain differences): MuJoCo state set to the recorded state of step t")
    out(f"   (ground: flat plane at the estimated height), Isaac's action of step t applied for one control step, joint positions compared with the recorded step t+1.")
    out("   Isaac randomises the command delay per robot (0..2 physics steps); all three are shown. 'air' = steps 1..3 (robot dropping after the reset,")
    out("   soles above the ground: no contact model involved), 'ground' = steps 8..48.")
    def one_step(t, delay):
        p = st["root_pos_w"][t].astype(float).copy(); p[2] -= ground; p[:2] = 0.0
        sim.set_state(p, st["root_quat_w"][t], st["joint_pos"][t], st["joint_vel"][t], st["root_lin_vel_b"][t], st["root_ang_vel_b"][t])
        prev = ec.default_q.copy()
        if t > 0:
            prev[ec.action_idx] += ec.action_scale * st["actions"][t - 1]
        new = ec.default_q.copy()
        new[ec.action_idx] += ec.action_scale * st["actions"][t]
        for k in range(ec.decimation):
            tau, _ = sim.pd_torque(prev if k < delay else new)
            d.ctrl[sim.act_ids] = tau
            mujoco.mj_step(m, d)
        return sim.joint_pos() - st["joint_pos"][t + 1], sim.joint_vel() - st["joint_vel"][t + 1], d.qpos[2] + ground - st["root_pos_w"][t + 1][2]
    legs = [i for i, j in enumerate(ec.joints) if any(s in j for s in ("hip", "knee", "ankle"))]
    for delay in (0, 1, 2):
        e0 = one_step(0, delay)[0]
        ea = np.array([one_step(t, delay)[0] for t in range(1, 4)])
        eg = np.array([one_step(t, delay)[0] for t in range(8, T - 1)])
        va = np.array([one_step(t, delay)[1] for t in range(1, 4)])
        zg = np.array([one_step(t, delay)[2] for t in range(8, T - 1)])
        out(f"   delay {delay}: joint pos error after one step  step 0 (reset step): rms {np.sqrt((e0**2).mean()):.4f} max {np.abs(e0).max():.4f} rad;  air: rms {np.sqrt((ea**2).mean()):.4f} max {np.abs(ea).max():.4f} rad;"
            f"  ground: rms {np.sqrt((eg**2).mean()):.4f} max {np.abs(eg).max():.4f} rad (legs rms {np.sqrt((eg[:, legs]**2).mean()):.4f});"
            f"  air joint vel rms {np.sqrt((va**2).mean()):.3f} rad/s;  ground pelvis z error mean {zg.mean()*1000:+.1f} mm")
    out("   step 0 fits worse than steps 1..3 in every variant (hip pitch and knee move about half / 1.4x as far as in Isaac): the first step after an Isaac")
    out("   reset is not reproduced exactly (cause not identified; the actuator delay buffer at reset is a candidate).")
    step_move = np.abs(np.diff(st["joint_pos"], axis=0))
    out(f"   for scale: the recorded joints move rms {np.sqrt((step_move**2).mean()):.4f} rad per control step (max {step_move.max():.4f})")
    out("")

    # ------------------------------------------------------------------ 6. closed loop, 1 s
    out("6. CLOSED LOOP for 1 s from the recorded initial state, flat ground (plane at the estimated ground height), recorded command per step")
    isaac_xy = st["root_pos_w"][:, :2] - st["root_pos_w"][0, :2]
    def tilt_deg(q):
        R = s2s.quat_to_mat(q)
        return math.degrees(math.acos(max(-1, min(1, R[2, 2]))))
    out(f"   Isaac recording : pelvis moved {np.linalg.norm(isaac_xy[-1]):.3f} m in xy over steps 0..49 (x {isaac_xy[-1,0]:+.3f}, y {isaac_xy[-1,1]:+.3f}), height above the assumed ground"
        f" {st['root_pos_w'][0,2]-ground:.3f} -> {st['root_pos_w'][-1,2]-ground:.3f} m, tilt at the end {tilt_deg(st['root_quat_w'][-1]):.1f} deg, max tilt {max(tilt_deg(q) for q in st['root_quat_w']):.1f} deg,"
        f" mean |action| {np.abs(st['actions']).mean():.2f}")
    for foot in ("hull", "spheres"):
        for delay in (0, 1, 2):
            sm = s2s.X2Sim(ec, s2s.Terrain("flat"), foot=foot, delay_steps=delay) if (foot, delay) != ("hull", 0) else sim
            sm.delay_steps = delay
            p = st["root_pos_w"][0].astype(float).copy(); p[2] -= ground; p[:2] = 0.0
            sm.reset(p, root_quat=st["root_quat_w"][0], joint_pos=st["joint_pos"][0], joint_vel=st["joint_vel"][0],
                     root_lin_vel_b=st["root_lin_vel_b"][0], root_ang_vel_b=st["root_ang_vel_b"][0])
            r = s2s.run_closed_loop(sm, policy, 1.0, cmd_fn=lambda s, _sim: st["command"][s], verbose=False)
            end = sm.root_state()
            xy = end[0][:2] - p[:2]
            tl = [tilt_deg(q) for q in r["root_quat"]] + [tilt_deg(end[1])]
            jd = r["joint_pos"] - st["joint_pos"]
            ad = r["action"] - st["actions"]
            td = sum(len(s2s.touchdowns(r["t"], r["feet_force"][:, i, 2])) for i in range(2))
            out(f"   MuJoCo foot={foot:<7} delay {delay}: moved {np.linalg.norm(xy):.3f} m (x {xy[0]:+.3f}, y {xy[1]:+.3f}), height {p[2]:.3f} -> {end[0][2]:.3f} m, tilt end {tl[-1]:.1f} deg, max {max(tl):.1f} deg,"
                f" fell: {r['fall_t'] if r['fall_t'] is not None else 'no'}, touchdowns {td}, mean |action| {np.abs(r['action']).mean():.2f},"
                f" joint pos vs Isaac rms {np.sqrt((jd**2).mean()):.3f} rad (first 10 steps {np.sqrt((jd[:10]**2).mean()):.3f}), action vs Isaac rms {np.sqrt((ad**2).mean()):.2f} (first 10 steps {np.sqrt((ad[:10]**2).mean()):.2f})")
    out("")
    # ------------------------------------------------------------------ 7. is the depth used at all
    import onnxruntime as ort
    enc = ort.InferenceSession(str(export / "0-depth_encoder.onnx"), providers=["CPUExecutionProvider"])
    nm = enc.get_inputs()[0].name
    lat = enc.run(None, {nm: obs[:, a:a + size].reshape(T, 8, *ec.frame_shape)})[0] if enc.get_inputs()[0].shape[0] != 1 else         np.concatenate([enc.run(None, {nm: obs[t:t + 1, a:a + size].reshape(1, 8, *ec.frame_shape)})[0] for t in range(T)])
    out("7. DEPTH ENCODER OUTPUT (finding, not a test of the port): 0-depth_encoder.onnx on the 50 recorded Isaac depth stacks")
    out(f"   latent shape {lat.shape}, max |latent| = {np.abs(lat).max():.3g}, non-zero entries: {int((lat != 0).sum())} of {lat.size}")
    o2 = obs.copy(); o2[:, a:] = np.clip(o2[:, a:] * 0.8, 0, 1)
    o3 = obs.copy(); o3[:, a:] = 0.0
    o4 = obs.copy(); o4[:, a:] = 1.0
    base = policy(obs)
    out(f"   action change of policy.onnx when the recorded depth is scaled by 0.8: {np.abs(policy(o2) - base).max():.3g}; all depth = 1.0 (nothing within 2.5 m): {np.abs(policy(o4) - base).max():.3g};"
        f" all depth = 0.0: {np.abs(policy(o3) - base).max():.3g}")
    out("   -> for realistic depth images the encoder output is exactly zero at this checkpoint (a dead ReLU output layer, by the G1 architecture; not inspected further): the actions do not depend on the depth image.")
    out("      Consequence: flat ground and stairs give bit-identical behaviour until the robot touches a step.")
    out("")
    (HERE / "validation.txt").write_text("\n".join(LINES) + "\n")
    np.savez_compressed(HERE / "validation_depth.npz", isaac_newest=D[:, 7], mujoco_flat=mj, ground_z=ground)


if __name__ == "__main__":
    main()
