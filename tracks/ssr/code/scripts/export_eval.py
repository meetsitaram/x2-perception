"""Package an SSR checkpoint for the common eval (docs/EVAL.md, state/SHARED_LESSONS.md "What the eval needs"). CPU only.

CUDA_VISIBLE_DEVICES= X2_PLANT=mc_20261024 ../env/venv/bin/python scripts/export_eval.py --ckpt ../runs/r6b/ckpt_001000.pt --out ../eval/export/r6b_001000

Writes into --out:
  policy.onnx       (hist 1x5x72, depth 1x36x36, gru 1x256) -> (action 1x21, gru_out 1x256); batch 1, fixed sizes, deterministic
                    (encoder latent z_p = mu), actor normaliser inside the graph, recurrent state as explicit input/output.
  obs_check.npz     50 consecutive control steps of one robot in the SSR training simulator (MuJoCo Warp on the CPU, train=False,
                    flat tile, 0.6 m/s forward): simulator state, command, camera rays and hits, depth image, proprio history,
                    GRU state in/out, actor input, torch action and ONNX action.
  adapter_card.json observation, action, plant, command and camera as trained (built from the live training objects, not retyped).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np
import torch

from ssr import networks as N
from ssr import terrain as T
from ssr import x2 as X
from ssr.env import DEPTH_MIN, DEPTH_N, H, PROP_DIM, SSREnv


class Policy(torch.nn.Module):
    def __init__(self, enc, aux, actor):
        super().__init__()
        self.enc, self.aux, self.actor = enc, aux, actor

    def forward(self, hist, depth, gru):
        e = self.enc.step(hist, depth, gru, sample=False)
        o = torch.cat((hist[:, -1], self.aux.est(hist.flatten(1)), e["z"]), -1)
        return self.actor(o), e["h"]


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--camera", default="chest_rs")
    ap.add_argument("--speed", type=float, default=0.6)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--warm", type=int, default=100, help="control steps walked before the recorded 50 (so the record is mid-gait)")
    a = ap.parse_args()
    import onnxruntime as ort
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    plant_name = os.environ.get("X2_PLANT", "")
    assert plant_name, "set X2_PLANT to the run's plant (args.json 'plant')"
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    run_args = {}
    ap_json = pathlib.Path(a.ckpt).parent / "args.json"
    if ap_json.exists():
        run_args = json.loads(ap_json.read_text())
        assert run_args.get("plant") == plant_name, (run_args.get("plant"), plant_name)
        assert run_args.get("camera") == a.camera, (run_args.get("camera"), a.camera)
        assert not run_args.get("map_memory"), "map memory export not supported here"
        if run_args.get("default_from_ref"):             # the model (default pose) must be built as in training
            os.environ["X2_DEFAULT_FROM_REF"] = "1"
        if run_args.get("action_scale_paper"):
            os.environ["X2_ACTION_SCALE_PAPER"] = "1"
        if run_args.get("excl_pelvis_hiproll"):
            os.environ["X2_EXCL_PELVIS_HIPROLL"] = "1"
    enc, aux, actor = N.Encoder(False), N.Aux(), N.MoEActor()
    enc.load_state_dict(ck["enc"]); aux.load_state_dict(ck["aux"]); actor.load_state_dict(ck["actor"])
    pol = Policy(enc, aux, actor).eval()

    # ---- ONNX, batch 1, fixed sizes
    onnx_path = out / "policy.onnx"
    x = (torch.zeros(1, H, PROP_DIM), torch.zeros(1, DEPTH_N, DEPTH_N), torch.zeros(1, 256))
    torch.onnx.export(pol, x, str(onnx_path), input_names=["hist", "depth", "gru"], output_names=["action", "gru_out"],
                      opset_version=17, dynamo=False)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    # ---- one robot in the training simulator (no training noise / delay / pushes: train=False)
    env = SSREnv(1, "cpu", train=False, tiles=True, camera=a.camera)
    env.flat_only = True
    ids = torch.arange(1)
    env.type[:], env.level[:] = T.FLAT, 0
    env.reset(ids)
    env.cam_pos.zero_(); env.cam_rpy.zero_(); env.cam_fov.zero_()           # nominal mount, gains and masses for the record
    env.gain[:, :, 0] = env.kp0; env.bias[:, :, 1] = -env.kp0; env.bias[:, :, 2] = -env.kd0
    env.mass[:] = env.mass0; env.inertia[:] = env.inertia0; env.ipos[:] = env.ipos0
    env.friction[:, env.robot_geoms, 0] = 1.0
    env.speed[:], env.heading[:] = a.speed, 0.0
    env.qpos[:, 3], env.qpos[:, 4:7] = 1.0, 0.0
    env.refresh(); env._update_cmd(); env._contacts(); env._update_obs(ids); env._update_cmd()
    h = enc.init_h(1, "cpu")
    rec = {k: [] for k in ("qpos", "qvel", "cmd", "hist", "depth", "ray_origin_world", "ray_dir_world", "ray_t", "cam_body_pos",
                           "cam_body_xmat", "gru_in", "gru_out", "vel_est", "z", "actor_input", "action_torch", "action_onnx",
                           "joint_target_31", "t")}
    max_err = 0.0
    for t in range(a.warm + a.steps):
        e = enc.step(env.hist, env.depth, h, sample=False)
        vh = aux.est(env.hist.flatten(1))
        o = torch.cat((env.prop, vh, e["z"]), -1)
        act = actor(o)
        if t >= a.warm:
            r = sess.run(None, {"hist": env.hist.numpy().astype(np.float32), "depth": env.depth.numpy().astype(np.float32),
                                "gru": h.numpy().astype(np.float32)})
            max_err = max(max_err, float(np.abs(r[0] - act.numpy()).max()), float(np.abs(r[1] - e["h"].numpy()).max()))
            tgt = env.q0_all.clone()
            tgt[env.idx] = (env.q0 + env.scale * act.clamp(-8, 8)[0]).clamp(env.q_lo, env.q_hi)
            for k, v in (("qpos", env.qpos[0]), ("qvel", env.qvel[0]), ("cmd", env.cmd[0]), ("hist", env.hist[0]), ("depth", env.depth[0]),
                         ("ray_origin_world", env._ray_o[0]), ("ray_dir_world", env._ray_d[0]), ("ray_t", env._ray_t[0]),
                         ("cam_body_pos", env.xpos[0, env.cam_body]), ("cam_body_xmat", env.xmat[0, env.cam_body].reshape(3, 3)),
                         ("gru_in", h[0]), ("gru_out", e["h"][0]), ("vel_est", vh[0]), ("z", e["z"][0]), ("actor_input", o[0]),
                         ("action_torch", act[0]), ("joint_target_31", tgt)):
                rec[k].append(v.numpy().copy())
            rec["action_onnx"].append(r[0][0])
            rec["t"].append(t * X.CONTROL_DT)
        h = e["h"]
        res = env.step(act.clamp(-8, 8))
        if bool(res["term"].any()) and t >= a.warm:
            print("warning: robot terminated at step", t)
    rec = {k: np.asarray(v, np.float32) for k, v in rec.items()}
    rec["camera_rays_cam_frame"] = env.rays.numpy().astype(np.float32)
    rec["joint_names_31"] = np.array(env.x2.joint_names)
    rec["joint_names_21"] = np.array(X.NAMES21)
    np.savez_compressed(out / "obs_check.npz", **rec)
    vx = float(np.mean(rec["qvel"][:, 0]))

    # ---- adapter card
    x2 = env.x2
    names, idx = list(x2.joint_names), [int(i) for i in x2.idx21]
    held = [n for n in names if n not in X.NAMES21]
    cam = X.CAMERAS[a.camera]
    ac = lambda v: [round(float(f), 6) for f in np.asarray(v).ravel()]
    m = x2.model
    card = dict(
        track="ssr", method="SSR (arXiv 2605.30770), RL from scratch, no tracker", checkpoint=str(pathlib.Path(a.ckpt).resolve()),
        checkpoint_sha256=sha(a.ckpt), iteration=int(ck["iter"]), stage="single stage (paper); run args in train_args",
        code_snapshot="/mnt/filesystem-g4/tracks/ssr/code (no git; change log in state/ssr.md)", train_args=run_args,
        policy_file="policy.onnx", policy_sha256=sha(onnx_path),
        onnx_io=dict(inputs=dict(hist=[1, H, PROP_DIM], depth=[1, DEPTH_N, DEPTH_N], gru=[1, 256]), outputs=dict(action=[1, 21], gru_out=[1, 256]),
                     note="feed gru_out back as gru next step; gru = zeros at reset; deterministic (encoder z_p = mu, action = actor mean)"),
        control_dt=X.CONTROL_DT, physics_dt=X.SIM_DT, decimation=X.DECIMATION,
        joint_names_model=names, joint_names=list(X.NAMES21), joint_index_in_model=idx,
        commanded_joints=list(X.NAMES21), held_joints=dict(names=held, held_at="default_q (position targets at default, same PD gains)"),
        default_q=dict(zip(names, ac(x2.default_q))), action_scale=dict(zip(names, ac(x2.action_scale))),
        action=dict(clip=[-8.0, 8.0], target="q_target = clip(default_q + action_scale * clip(action, -8, 8), q_lo, q_hi) for the 21 commanded joints",
                    actor_output_bound="8 tanh(x / 8)", q_lo=dict(zip(names, ac(x2.q_lo))), q_hi=dict(zip(names, ac(x2.q_hi))),
                    delay_in_training="0, 1 or 2 control steps (uniform per episode); none in this record"),
        plant=plant_name, kp=dict(zip(names, ac(x2.kp))), kd=dict(zip(names, ac(x2.kd))), effort_limit=dict(zip(names, ac(x2.effort))),
        armature=dict(zip(names, ac(m.dof_armature[6:]))),
        actuator="MuJoCo position actuators, tau = kp (q_target - q) - kd qd, clipped at effort_limit",
        training_randomisation="per episode: kp x U(0.9,1.1) x motor U(0.8,1.2), kd x U(0.9,1.1) x motor; friction U(0.2,1.25); link mass x U(0.9,1.1), "
                               "pelvis +U(-2,12.5) kg; CoM +-1 cm (pelvis +-10 cm); pushes 0.4 m/s every 12 s; observation noise (below)",
        observation=dict(
            hist=dict(shape=[H, PROP_DIM], order="oldest first (row -1 = current step)", at_reset="all 5 rows = the first frame",
                      frame_terms=[
                          dict(name="base_ang_vel", size=3, frame="pelvis body frame", units="rad/s", scale=1.0, train_noise="U(-0.2,0.2)"),
                          dict(name="projected_gravity", size=3, frame="pelvis body frame", units="unit vector (-R^T e_z... = -R[2,:])", scale=1.0, train_noise="U(-0.05,0.05)"),
                          dict(name="command", size=3, frame="pelvis heading frame", units="vx m/s, vy m/s, yaw rate rad/s"),
                          dict(name="joint_pos_rel", size=21, order="joint_names", units="rad", value="q - default_q", train_noise="U(-0.01,0.01)"),
                          dict(name="joint_vel", size=21, order="joint_names", units="rad/s", scale=1.0, train_noise="U(-0.5,0.5)"),
                          dict(name="last_action", size=21, order="joint_names", value="the actor output of the previous step, before clipping; 0 at reset"),
                      ], clip=[-40.0, 40.0], nan_to="0"),
            depth=dict(shape=[DEPTH_N, DEPTH_N], body=cam["body"], pos_in_body=list(cam["pos"]), pitch_down_rad=float(cam["pitch"]),
                       fov_vertical_rad=float(cam.get("fov", X.CAM_FOV)), fov_horizontal_rad=float(cam.get("fov_h", cam.get("fov", X.CAM_FOV))),
                       camera_frame="x forward, y left, z up; mount = body R_body @ Rz(yaw) @ Ry(pitch) @ Rx(roll); rays in obs_check camera_rays_cam_frame",
                       pixel_layout="row 0 = image top, column 0 = image left (robot's left), pinhole, pixel centres",
                       value="z = ray length x ray[0] (depth along the optical axis); (clip(z, %.2f, dmax) - %.2f) / (dmax - %.2f)" % (DEPTH_MIN, DEPTH_MIN, DEPTH_MIN),
                       near=DEPTH_MIN, far=float(cam.get("dmax", 2.0)), ray_march_range=float(cam.get("dmax", 2.0)) / 0.4,
                       geometry="ray march on the terrain height field including side obstacles (vision raster), plus self-occlusion by limb spheres (x2.OCCLUDERS); robot meshes are not ray cast",
                       train_noise="self-occluded pixels + N(0, 0.1 m); mount randomised per episode pos +-%s m, rpy +-%s deg, fov +-1 deg" % (list(cam.get("dr_pos", (0.01,) * 3)), list(cam.get("dr_rpy_deg", (1.0,) * 3))),
                       stacking="none (one frame per control step; memory is the GRU)", delay="none", filtering="none"),
            gru=dict(shape=[256], at_reset="zeros"),
            internal="the graph computes the velocity estimate from hist and the actor input [hist[-1], vel_est(3), z(48)] itself",
        ),
        command=dict(
            takes="(vx, vy, yaw_rate) in the pelvis heading frame, every step in hist rows",
            ranges=dict(vx=[-1.0, 1.2], vy=[-0.4, 0.4], yaw_rate=[-1.2, 1.2]), forward_speed_limit=1.2,
            training="tile robots: a world direction (heading) and speed drawn per episode (speed U(0,1.2), slopes U(-1,1)); each step "
                     "v = speed rotated into the heading frame and yaw_rate = clip(1.0 x heading error, +-1.2). Some robots get a fixed "
                     "omnidirectional (vx U(-1,1), vy U(-0.4,0.4), yaw U(-1.2,1.2)) command. No waypoints, no slowing near a goal, "
                     "no station holding in training; a zero command appears only as a drawn low speed (about 6 % of walkers under 0.15 m/s)",
            near_waypoint="not trained: the harness's standard velocity command is what the policy should get"),
        trained_model=dict(mjcf=str(X.MJCF), excluded_body_pairs=[[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(b)) for b in divmod(int(sig), 65536)[::-1]] for sig in m.exclude_signature],
                           pelvis_hiproll_excluded=os.environ.get("X2_EXCL_PELVIS_HIPROLL") == "1", default_pose_from_reference=os.environ.get("X2_DEFAULT_FROM_REF") == "1",
                           action_scale_paper=os.environ.get("X2_ACTION_SCALE_PAPER") == "1", hfield_chunk=int(os.environ.get("X2_HF_CHUNK", "400")), note="the eval must use this collision model (coordinator decision 2026-11-12: pelvis/hip-roll pairs excluded)"),
        privileged_not_in_actor="critics / foothold / discriminators use base velocity, contacts, height maps; the actor uses none of them",
        observes="proprioception + command + chest depth camera (no height scan, no map)",
        obs_check=dict(file="obs_check.npz", steps=a.steps, after_warm_steps=a.warm, speed_cmd=a.speed, mean_vx=vx,
                       max_abs_torch_minus_onnx=max_err),
    )
    (out / "adapter_card.json").write_text(json.dumps(card, indent=1))
    print(json.dumps(dict(out=str(out), iter=int(ck["iter"]), max_err=max_err, mean_vx=vx)))


if __name__ == "__main__":
    main()
