"""Hand a stage-1 tracker checkpoint to the eval monitor (state/SHARED_LESSONS.md, "What the eval needs"). Runs on the CPU
(one robot), so it does not touch the GPU that trains.

    CUDA_VISIBLE_DEVICES= python tools/handover.py CKPT OUT_DIR

Writes OUT_DIR/meta.json (for /mnt/filesystem-g4/eval/code/tools/export_actor.py), adapter_card.json, obs_check.npz.
"""
import json
import os
import sys

import mujoco_warp as mjw
import numpy as np
import torch

sys.path.insert(0, "/mnt/filesystem-g4/tracks/acp/code")
from acp import env as E  # noqa: E402
from acp.data import BODIES, LINKS, POINT, Dataset  # noqa: E402
from acp.train import ActorCritic  # noqa: E402


def cpu_physics(self, ctrl_new):          # the trainer's captured CUDA graph does not exist on the CPU
    k = self.delay
    for j in range(E.DECIMATION):
        self.ctrl[:] = torch.where(j >= k, ctrl_new, self.ctrl_prev)
        mjw.step(self.mm, self.dd)
    self.ctrl_prev.copy_(ctrl_new)


def main():
    ckpt, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    dev = torch.device("cpu")
    ck = torch.load(ckpt, map_location=dev)
    cfg = ck["data"]
    data = Dataset(cfg, dev, cache=os.path.join(os.path.dirname(cfg), "labels_" + os.path.basename(cfg) + ".pt"))
    E.AcpEnv._physics = cpu_physics
    env = E.AcpEnv(data, 1, eval_mode=True)
    ac = ActorCritic(ck["actor_dim"], ck["critic_dim"], ck["act_dim"])
    ac.load_state_dict(ck["model"]); ac.eval()
    x2 = env.x2
    meta = dict(joint_names=list(x2.joint_names), default_q=x2.default_q.tolist(), action_scale=x2.action_scale.tolist(),
                kp=x2.kp.tolist(), kd=x2.kd.tolist())
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"))
    # one robot on a stair-climb clip, 50 steps from the clip start
    m = data.motion
    si = data.set_names.index("gst_v1")
    clip = int((data.clip_set == si).nonzero()[0])
    saved = data.clip_w
    data.clip_w = torch.zeros_like(saved); data.clip_w[clip] = 1.0
    env.reset(torch.zeros(1, dtype=torch.long))
    data.clip_w = saved
    rec = {k: [] for k in ("qpos", "qvel", "actor_input", "action", "scan", "contact_targets", "goal", "ref_frame", "terrain_under_scan_world_z")}
    with torch.no_grad():
        for _ in range(50):
            ao, _ = env.observe()
            a = ac.actor(ac.an(ao))
            rec["qpos"].append(env.qpos[0].clone().numpy()); rec["qvel"].append(env.qvel[0].clone().numpy())
            rec["actor_input"].append(ao[0].numpy()); rec["action"].append(a[0].numpy())
            H = 5 * env.PROP
            rec["scan"].append(ao[0, H:H + 63].numpy()); rec["contact_targets"].append(ao[0, H + 63:H + 90].numpy())
            rec["goal"].append(ao[0, H + 90:].numpy()); rec["ref_frame"].append(int(env.f[0]) - int(m.clip_start[clip]))
            rec["terrain_under_scan_world_z"].append((ao[0, H:H + 63] + env.qpos[0, 2]).numpy())
            env.step(a)
    props = [dict(center=[float(x) for x in b[:3]], half=[float(x) for x in env.terrain.half[i]], yaw=float(b[3]))
             for i, b in enumerate(env.terrain.boxes[clip]) if float(b[4]) > 0]
    np.savez(os.path.join(out, "obs_check.npz"), **{k: np.array(v) for k, v in rec.items()},
             norm_mean=ac.an.mean.numpy(), norm_var=ac.an.var.numpy(), clip_name=m.names[clip], props_json=json.dumps(props),
             note="50 consecutive control steps of one robot in MuJoCo Warp (CPU), eval mode (no noise, no randomisation, no delay, "
                  "floor friction 1.0), deterministic action. qpos = root xyz, quat wxyz, 31 joints (MJCF order); qvel = root linear "
                  "velocity (world), root angular velocity (body frame), 31 joint velocities; recorded BEFORE the step that applies "
                  "the action. props_json = the clip's static boxes (center, half extents, yaw) on a flat floor at z = 0.")
    H = 5 * env.PROP
    card = dict(
        track="acp", checkpoint=os.path.abspath(ckpt), iteration=int(ck["it"]), stage=1,
        code_snapshot="/mnt/filesystem-g4/tracks/acp/runs/s1_v1/code_snapshot (acp/env.py, acp/data.py, acp/train.py)",
        policy_file="actor.npz (plain MLP 588-512-256-128-31, ELU; input normalised as clip((x - norm_mean) / (sqrt(norm_var) + 1e-5), -10, 10))",
        simulator="MuJoCo Warp 3.17.0", control_dt=0.02, physics_dt=0.005, decimation=4, integrator="implicitfast",
        joint_names=list(x2.joint_names), commanded_joints="all 31, MJCF hinge order",
        default_q=x2.default_q.tolist(), action_scale=x2.action_scale.tolist(), action_clip=[-20.0, 20.0],
        joint_target="default_q + action_scale * action (position target for the PD servo, held for the control step)",
        plant=x2.plant.name, plant_note="kp / kd below are the nominal values the simulator applies (read from the plant file via "
        "assets.build_x2); training randomises each x U(0.9, 1.1). ACP_PLANT selects the plant; unset = the active file (vendor_20260823)",
        kp=x2.kp.tolist(), kd=x2.kd.tolist(), effort=x2.effort.tolist(), armature=x2.armature.tolist(),
        actuator="in-simulator PD position servo, torque clipped to +-effort; training randomised kp, kd, armature x U(0.9, 1.1) and "
                 "an action delay of 0-20 ms; the policy expects no fixed delay",
        foot_collision="stock MJCF sole spheres plus a foot box that collides only with props and height fields (assets.build_x2 foot_terrain=True)",
        observation=[
            dict(name="proprio_history", size=H, layout="5 control steps, OLDEST first, newest last; each step 99 numbers in this order",
                 per_step=[dict(name="base_ang_vel", size=3, frame="body", units="rad/s"),
                           dict(name="projected_gravity", size=3, frame="body", note="R^T (0, 0, -1): (0, 0, -1) when upright"),
                           dict(name="joint_pos_minus_default", size=31, units="rad", order="joint_names"),
                           dict(name="joint_vel", size=31, units="rad/s"),
                           dict(name="previous_action", size=31, note="the raw policy output of the previous step (after the +-20 clip), 0 at reset")],
                 at_reset="all 5 steps filled with the first sample", scale=1.0, clip="none before normalisation"),
            dict(name="height_scan", size=63, grid="9 (x) by 7 (y), 0.1 m spacing; x from -0.2 to +0.6 m, y from -0.3 to +0.3 m; index = ix * 7 + iy",
                 frame="body-yaw (heading) frame centred on the root (pelvis) xy", value="terrain height under the point minus the root height (m)",
                 clip=[-2.0, 2.0], walls="in training the scan sees the tops of prop boxes and the raster, so a tall box reads as a tall column; "
                 "there were no walls or rails in stage-1 data"),
            dict(name="contact_targets", size=27, layout="nine bodies x (x, y, z)", bodies=list(BODIES), links=list(LINKS),
                 contact_point_in_link_frame=[list(p) for p in POINT],
                 definition="for each body, the position of its NEXT contact onset if one starts within 1.5 s: the point on the terrain "
                            "under the body's contact point once it has settled (lowest gap within 0.5 s of touchdown). A body that is "
                            "in contact now gets its next NEW contact, not the current one. Zero vector when no onset is coming within 1.5 s.",
                 frame="body-yaw (heading) frame, relative to the root position (x forward, y left, z up; z is target height minus root height)",
                 units="m", stage1_source="reference clips (oracle); only the two feet are ever non-zero in this round's data"),
            dict(name="goal_command", size=3, layout="(dx, dy, dpsi)",
                 definition="d = horizontal vector from the root to the goal position in the body-yaw frame; (dx, dy) = d / max(|d|, r); "
                            "dpsi = goal heading minus robot heading, wrapped to [-pi, pi)", r_training=0.3,
                 note="in stage 1 the goal is the reference root 0.5-1.5 s ahead (0.5-1.5 m at walking speed), so |(dx, dy)| is 1 while "
                      "moving and shrinks inside 0.3 m; the paper uses r = 0.5 m for goal-reaching (stage 3)")],
        command="target direction and relative heading (goal_command above); no velocity command. Reference speeds in training: "
                "walking clips, mostly 0.3-1.2 m/s",
        privileged_in_training_only="the critic's inputs (base linear velocity, 17 x 11 scan, reference kinematics, reference contact positions "
                                    "and flags); the actor never sees them. The eval must supply the contact targets from a planner or "
                                    "another contact source, not from a reference.",
        not_scoreable_alone="stage 1 needs a contact-target source on the route; see state/acp.md",
    )
    json.dump(card, open(os.path.join(out, "adapter_card.json"), "w"), indent=1)
    print("wrote", out, "clip", m.names[clip], "action abs mean", float(np.abs(np.array(rec["action"])).mean()))


if __name__ == "__main__":
    main()
