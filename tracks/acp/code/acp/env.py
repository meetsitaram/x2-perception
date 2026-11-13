"""ACP stage-1 environment for the X2 on MuJoCo Warp (arXiv 2610.31608: contact-conditioned tracker, oracle contacts).

Actor (reference-free): 5-step proprioceptive history, 9 x 7 height scan, nine contact targets (body-yaw frame, zero
when invalid), 3-d goal command. Critic V_m (privileged): noise-free proprioception, 17 x 11 scan, base linear
velocity, reference kinematics, reference contact-body positions, reference contact flags. Rewards: Table 3, stage 1.
Resets: Curriculum State Initialization (Table S4). Randomisation: Table S2 (what MuJoCo can express).
"""
from __future__ import annotations

import math
import re

import mujoco
import mujoco_warp as mjw
import torch
import warp as wp

from gear_sonic.supermimic.assets import DECIMATION, MJCF_PATH
from gear_sonic.supermimic.kinematics import mtv3, mv3, quat_wxyz_to_mat

from acp.data import Dataset

HIST = 5
DT = 0.02
# Table 3 (stage 1)
W = dict(track_base_pos=3.0, track_base_vel=1.0, track_joint_pos=2.5, track_joint_vel=0.5, track_contact_label=5.0,
         track_contact_pos_base=2.0, track_contact_pos_global=1.0, action_rate=-0.1, action_rate_2nd=-5e-2,
         joint_torque=-1e-5, torque_limits=-1e-3, dof_pos_limits=-10.0, dof_vel_limits=-1.0, contact_acc=-0.01)
SIG_P, SIG_R, SIG_V, SIG_W, SIG_Q, SIG_QD, SIG_C = 0.5, 1.0, 1.0, 3.0, 2.0, 10.0, 1.0
GOAL_R = 0.3                      # goal scaling radius, reference-following (Table S4)
LOOKAHEAD = (25, 75)              # 0.5-1.5 s
CSI_INIT, CSI_STEP, CSI_DONE, CSI_MAX = 0.9, 0.05, 0.99, 0.99
# paper-silent (our choice): an episode ends early when the base is this far from the reference
TERM_POS, TERM_ROT = 0.5, 1.0     # m, rad
SOFT_LIMIT = 1.0                  # the MJCF joint range (the paper's "configured soft limits" are not given; 0.95 penalised the references themselves)
FALL_H = 0.35                     # m: root this low over the lowest ground under root and feet = a fall (fall_rule)
FORCE_MIN = 1.0                   # N: measured contact


def yaw_mat(R):
    yaw = torch.atan2(R[..., 1, 0], R[..., 0, 0])
    c, s = torch.cos(yaw), torch.sin(yaw)
    z, o = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack((c, -s, z, s, c, z, z, z, o), -1).reshape(*R.shape[:-2], 3, 3)


def mat_to_quat(R):
    m00, m11, m22 = R[..., 0, 0], R[..., 1, 1], R[..., 2, 2]
    tr = m00 + m11 + m22
    cand = torch.stack((1 + tr, 1 + m00 - m11 - m22, 1 - m00 + m11 - m22, 1 - m00 - m11 + m22), -1)
    i = cand.argmax(-1, keepdim=True)
    q_w = torch.stack((1 + tr, R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]), -1)
    q_x = torch.stack((R[..., 2, 1] - R[..., 1, 2], 1 + m00 - m11 - m22, R[..., 0, 1] + R[..., 1, 0], R[..., 0, 2] + R[..., 2, 0]), -1)
    q_y = torch.stack((R[..., 0, 2] - R[..., 2, 0], R[..., 0, 1] + R[..., 1, 0], 1 - m00 + m11 - m22, R[..., 1, 2] + R[..., 2, 1]), -1)
    q_z = torch.stack((R[..., 1, 0] - R[..., 0, 1], R[..., 0, 2] + R[..., 2, 0], R[..., 1, 2] + R[..., 2, 1], 1 - m00 - m11 + m22), -1)
    q = torch.gather(torch.stack((q_w, q_x, q_y, q_z), -2), -2, i.unsqueeze(-1).expand(*i.shape[:-1], 1, 4)).squeeze(-2)
    return q / q.norm(dim=-1, keepdim=True)


def rot_angle(Ra, Rb):
    tr = (Ra * Rb).sum((-1, -2))
    return torch.acos(((tr - 1) / 2).clamp(-1 + 1e-6, 1 - 1e-6))


def _grid(nx, ny, x0, dev):
    xs = x0 + 0.1 * torch.arange(nx, device=dev, dtype=torch.float32)
    ys = 0.1 * (torch.arange(ny, device=dev, dtype=torch.float32) - (ny - 1) / 2)
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    return gx.flatten(), gy.flatten()


class AcpEnv:
    def __init__(self, data: Dataset, num_envs: int, seed: int = 0, njmax: int = 300, nconmax: int = 64, eval_mode: bool = False):
        self.data, self.N, self.eval_mode = data, num_envs, eval_mode
        self.motion, self.terrain = data.motion, data.terrain
        dev = self.dev = data.device
        N = num_envs
        self.gen = torch.Generator(device=dev).manual_seed(seed)
        x2 = self.x2 = data.x2
        m = x2.model
        self.NJ = NJ = len(x2.joint_names)
        f32 = dict(device=dev, dtype=torch.float32)
        self.default_q = torch.tensor(x2.default_q, **f32)
        self.alpha = torch.tensor(x2.action_scale, **f32)
        self.kp0, self.kd0 = torch.tensor(x2.kp, **f32), torch.tensor(x2.kd, **f32)
        self.effort = torch.tensor(x2.effort, **f32)
        q_lo, q_hi = torch.tensor(x2.q_lo, **f32), torch.tensor(x2.q_hi, **f32)
        self.q_lo, self.q_hi = q_lo, q_hi
        mid, half = (q_lo + q_hi) / 2, (q_hi - q_lo) / 2
        self.soft_lo, self.soft_hi = mid - SOFT_LIMIT * half, mid + SOFT_LIMIT * half
        urdf = open(str(MJCF_PATH).replace("mjcf/x2_ultra.xml", "urdf/x2_ultra/x2_ultra.urdf").replace("mjcf\\x2_ultra.xml", "urdf/x2_ultra/x2_ultra.urdf")).read()
        vmax = {}
        for jm in re.finditer(r'<joint name="([^"]+)"(.*?)</joint>', urdf, re.S):
            lim = re.search(r'velocity="([-0-9.e]+)"', jm.group(2))
            if lim:
                vmax[jm.group(1)] = float(lim.group(1))
        self.qd_max = torch.tensor([vmax.get(n, 20.0) for n in x2.joint_names], **f32)

        d = mujoco.MjData(m)
        d.qpos[1], d.qpos[2], d.qpos[3], d.qpos[7:] = -50.0, 0.7, 1.0, x2.default_q
        mujoco.mj_forward(m, d)
        batch = {k: N for k in ("geom_friction", "body_ipos", "actuator_gainprm", "actuator_biasprm", "body_mass",
                                "body_inertia", "dof_armature")}
        with wp.ScopedDevice(str(dev)):
            self.mm = mjw.put_model(m, batch_sizes=batch)
            self.dd = mjw.put_data(m, d, nworld=N, njmax=njmax, nconmax=nconmax)
        self.qpos, self.qvel, self.ctrl = wp.to_torch(self.dd.qpos), wp.to_torch(self.dd.qvel), wp.to_torch(self.dd.ctrl)
        self.friction = wp.to_torch(self.mm.geom_friction)
        self.body_ipos = wp.to_torch(self.mm.body_ipos)
        self.gainprm, self.biasprm = wp.to_torch(self.mm.actuator_gainprm), wp.to_torch(self.mm.actuator_biasprm)
        self.body_mass, self.body_inertia = wp.to_torch(self.mm.body_mass), wp.to_torch(self.mm.body_inertia)
        self.dof_arm = wp.to_torch(self.mm.dof_armature)
        # contacts (shared arrays over all worlds)
        c = self.dd.contact
        self.c_world, self.c_geom, self.c_frame = wp.to_torch(c.worldid), wp.to_torch(c.geom), wp.to_torch(c.frame)
        self.c_addr, self.efc_force, self.nacon = wp.to_torch(c.efc_address), wp.to_torch(self.dd.efc.force), wp.to_torch(self.dd.nacon)
        self.c_index = torch.arange(self.c_world.shape[0], device=dev)
        bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)  # noqa: E731
        geom_body = torch.tensor(m.geom_bodyid, device=dev, dtype=torch.long)
        is_env_body = torch.tensor((m.body_mocapid >= 0) | (torch.arange(m.nbody).numpy() == 0), device=dev)
        self.geom_is_env = is_env_body[geom_body]
        # each geom -> which of the nine contact bodies it belongs to (-1: none). hands = the three wrist links
        nine = {"left_ankle_roll_link": 0, "right_ankle_roll_link": 1, "left_knee_link": 2, "right_knee_link": 3,
                "left_wrist_roll_link": 4, "left_wrist_pitch_link": 4, "left_wrist_yaw_link": 4, "right_wrist_roll_link": 5,
                "right_wrist_pitch_link": 5, "right_wrist_yaw_link": 5, "left_elbow_link": 6, "right_elbow_link": 7, "pelvis": 8}
        b2n = torch.full((m.nbody,), -1, dtype=torch.long, device=dev)
        for name, k in nine.items():
            if bid(name) >= 0:
                b2n[bid(name)] = k
        self.geom_nine = b2n[geom_body]
        self.torso = bid("torso_link")
        robot_geom = torch.tensor((m.geom_bodyid > 0) & (m.body_mocapid[m.geom_bodyid] < 0), device=dev)
        self.robot_geom_ids = robot_geom.nonzero().squeeze(-1)
        self.friction[:, ~robot_geom, 0] = 1.0 if eval_mode else 0.3   # max-combined: the robot's value decides
        if self.terrain.num_slots:
            slot_ids = [bid(f"prop_slot_{i}") for i in range(self.terrain.num_slots)]
            self.slot_mocap = torch.tensor([int(m.body_mocapid[b]) for b in slot_ids], device=dev, dtype=torch.long)
            self.mocap_pos, self.mocap_quat = wp.to_torch(self.dd.mocap_pos), wp.to_torch(self.dd.mocap_quat)
        self._reset_mask = wp.zeros(N, dtype=bool, device=str(dev))
        self._reset_mask_t = wp.to_torch(self._reset_mask)
        self._graph = None

        # ---- domain randomisation, sampled once per environment (Table S2)
        u = self._u
        self.calib = torch.zeros(N, NJ, device=dev)
        self.delay = torch.zeros(N, 1, device=dev)
        if not eval_mode:
            self.friction[:, self.robot_geom_ids, 0] = u(N, 1, lo=0.3, hi=1.2)
            leg = [bid(n) for n in x2_leg_bodies(m)]
            ms = u(N, len(leg), lo=0.85, hi=1.15)
            self.body_mass[:, leg] = self.body_mass[:, leg] * ms
            self.body_inertia[:, leg] = self.body_inertia[:, leg] * ms.unsqueeze(-1)
            add = u(N, lo=-2.0, hi=4.0)
            m0 = self.body_mass[:, self.torso].clone()
            self.body_mass[:, self.torso] = m0 + add
            self.body_inertia[:, self.torso] = self.body_inertia[:, self.torso] * ((m0 + add) / m0).unsqueeze(-1)
            self.body_ipos[:, self.torso] += torch.stack((u(N) * 0.025, u(N) * 0.05, u(N) * 0.05), -1)
            ankle = torch.tensor(["ankle" in n for n in x2.joint_names], device=dev)
            self.calib = u(N, NJ) * torch.where(ankle, 0.03, 0.01)
            self.dof_arm[:, 6:] = self.dof_arm[:, 6:] * u(N, NJ, lo=0.9, hi=1.1)
            kp, kd = self.kp0 * u(N, NJ, lo=0.9, hi=1.1), self.kd0 * u(N, NJ, lo=0.9, hi=1.1)
            self.gainprm[:, :, 0] = kp
            self.biasprm[:, :, 1] = -kp
            self.biasprm[:, :, 2] = -kd
            self.delay = torch.randint(0, DECIMATION + 1, (N, 1), device=dev, generator=self.gen).float()  # 0..20 ms
        self.kp, self.kd = self.gainprm[:, :, 0].clone(), -self.biasprm[:, :, 2].clone()

        # ---- bookkeeping
        self.f = torch.zeros(N, dtype=torch.long, device=dev)
        self.goal_f = torch.zeros(N, dtype=torch.long, device=dev)
        self.ep_step = torch.zeros(N, dtype=torch.long, device=dev)
        self.a_prev = torch.zeros(N, NJ, device=dev)
        self.a_prev2 = torch.zeros(N, NJ, device=dev)
        self.ctrl_prev = torch.zeros(N, NJ, device=dev)
        self.v_prev = torch.zeros(N, 9, 3, device=dev)
        self.c_prev = torch.ones(N, 9, dtype=torch.bool, device=dev)     # measured contact at the previous step
        self.tgt_prev = torch.zeros(N, 9, 3, device=dev)                 # commanded contact target (world) at the previous step
        self.occ_prev = torch.zeros(N, 9, dtype=torch.bool, device=dev)
        self.scan_off = torch.zeros(N, 1, device=dev)
        self.phase = torch.full((N,), CSI_INIT, device=dev)       # Curriculum State Initialization, per environment
        self.PROP = 6 + 3 * NJ
        self.hist = torch.zeros(N, HIST, self.PROP, device=dev)
        self.gx_a, self.gy_a = _grid(9, 7, -0.2, dev)             # tracker scan 0.8 x 0.6 m
        self.gx_c, self.gy_c = _grid(17, 11, -0.8, dev)           # planner / critic scan 1.6 x 1.0 m
        self.actor_dim = HIST * self.PROP + 63 + 27 + 3
        self.critic_dim = self.PROP + 187 + 3 + (6 + 2 * NJ) + 27 + 9
        self.csi = True
        self.fall_rule = False
        # stage 2 (set by the trainer): contact planner in the loop, input / reward schedules, hindsight recorder
        self.planner, self.rec, self.eta_obs, self.eta_rew = None, None, 0.0, 0.0
        self.plan_w = torch.zeros(N, 9, 3, device=dev)                   # planner targets (world), noise-free
        self.plan_m = torch.zeros(N, 9, dtype=torch.bool, device=dev)    # m_b: occupancy-gated, non-zero planner target
        self.tgt_held = torch.zeros(N, 9, 3, device=dev)                 # the commanded target before its last change
        self.held_ok = torch.zeros(N, 9, dtype=torch.bool, device=dev)
        self.air = torch.zeros(N, 9, dtype=torch.long, device=dev)       # control steps since the body's last measured contact
        self.plan_stats = torch.zeros(4, device=dev)                     # sums: target-vs-reference error, its count, fed share, steps
        self.stats = {}

    # ------------------------------------------------------------ helpers
    def _u(self, *shape, lo=-1.0, hi=1.0):
        return torch.rand(*shape, device=self.dev, generator=self.gen) * (hi - lo) + lo

    def state(self):
        quat = self.qpos[:, 3:7]
        R = quat_wxyz_to_mat(quat / quat.norm(dim=-1, keepdim=True))
        return (self.qpos[:, :3].clone(), R, self.qvel[:, :3].clone(), mv3(R, self.qvel[:, 3:6]), self.qpos[:, 7:].clone(),
                self.qvel[:, 6:].clone())

    def _physics(self, ctrl_new):
        k = self.delay
        if self.dev.type == "cpu":                                       # no captured graph on the CPU (small checks only)
            for j in range(DECIMATION):
                self.ctrl[:] = torch.where(j >= k, ctrl_new, self.ctrl_prev)
                mjw.step(self.mm, self.dd)
            self.ctrl_prev.copy_(ctrl_new)
            return
        for j in range(DECIMATION):
            self.ctrl[:] = torch.where(j >= k, ctrl_new, self.ctrl_prev)
            if self._graph is None:
                with wp.ScopedDevice(str(self.dev)):
                    mjw.step(self.mm, self.dd)
                    with wp.ScopedCapture() as cap:
                        mjw.step(self.mm, self.dd)
                self._graph = cap.graph
                continue
            wp.capture_launch(self._graph)
        self.ctrl_prev.copy_(ctrl_new)

    def contact_forces(self):
        """(N, 9) vertical and horizontal contact force on each of the nine bodies from the terrain (floor, props,
        raster): normal force of every active robot-terrain contact (pyramidal cone: the sum of its rows)."""
        n = self.nacon[0]
        act = self.c_index < n
        g = self.c_geom.long()
        env0, env1 = self.geom_is_env[g[:, 0]], self.geom_is_env[g[:, 1]]
        nine = torch.where(env0, self.geom_nine[g[:, 1]], self.geom_nine[g[:, 0]])
        ok = act & (env0 ^ env1) & (nine >= 0)
        w = self.c_world.long().clamp(0, self.N - 1)
        addr = self.c_addr.long()
        fn = (self.efc_force[w.unsqueeze(-1), addr.clamp(min=0)] * (addr >= 0)).sum(-1)
        nrm = self.c_frame[:, 0, :]
        fz = (fn * nrm[:, 2]).abs() * ok
        fxy = (fn.unsqueeze(-1) * nrm[:, :2]).norm(dim=-1) * ok
        idx = w * 9 + nine.clamp(min=0)
        Fz = torch.zeros(self.N * 9, device=self.dev).index_add_(0, idx, fz).view(self.N, 9)
        Fxy = torch.zeros(self.N * 9, device=self.dev).index_add_(0, idx, fxy).view(self.N, 9)
        return Fz, Fxy

    def _scan(self, p, R, gx, gy):
        heading = torch.atan2(R[:, 1, 0], R[:, 0, 0])
        c, s = torch.cos(heading), torch.sin(heading)
        wx = p[:, :1] + c[:, None] * gx - s[:, None] * gy
        wy = p[:, 1:2] + s[:, None] * gx + c[:, None] * gy
        return (self.terrain.height(wx, wy, self.motion.clip_id[self.f]) - p[:, 2:3]).clamp(-2.0, 2.0)

    # ------------------------------------------------------------ resets
    def reset(self, ids):
        n, m = len(ids), self.motion
        if n == 0:
            return
        clip = torch.multinomial(self.data.clip_w, n, replacement=True, generator=self.gen)
        if self.eval_mode:
            ph = torch.zeros(n, device=self.dev)
        elif self.csi:
            ph = self.phase[ids]
        else:
            ph = self._u(n, lo=0.0, hi=0.98)
            if self.rec is not None:                                 # recorder environments start in the first 10 % (paper:
                ph = torch.where(ids < self.rec.n, ph * (0.1 / 0.98), ph)   # only such episodes enter the hindsight buffer)
        frames = m.clip_start[clip] + (ph * (m.clip_len[clip] - 1).float()).long()
        frames = torch.minimum(frames, m.clip_end[clip] - 1)
        p, R, v, w, q, qd = (x.clone() for x in m.ref_state(frames))
        if not self.eval_mode:
            q = q + self._u(n, self.NJ) * 0.2                      # Table S2: reset joint position
            p[:, 2] += self._u(n, lo=0.0, hi=0.05)                 # reset base height
        q = torch.maximum(torch.minimum(q, self.q_hi), self.q_lo)
        self._reset_mask_t.zero_()
        self._reset_mask_t[ids] = True
        mjw.reset_data(self.mm, self.dd, self._reset_mask)
        self.qpos[ids, :3], self.qpos[ids, 3:7], self.qpos[ids, 7:] = p, mat_to_quat(R), q
        self.qvel[ids, :3] = v
        self.qvel[ids, 3:6] = (R.transpose(-1, -2) @ w.unsqueeze(-1)).squeeze(-1)
        self.qvel[ids, 6:] = qd
        if self.terrain.num_slots:
            pos, quat = self.terrain.mocap(clip)
            self.mocap_pos[ids.unsqueeze(-1), self.slot_mocap] = pos
            self.mocap_quat[ids.unsqueeze(-1), self.slot_mocap] = quat
        self.ctrl[ids] = q + self.calib[ids]
        self.ctrl_prev[ids] = self.ctrl[ids]
        self.f[ids], self.ep_step[ids] = frames, 0
        look = torch.randint(LOOKAHEAD[0], LOOKAHEAD[1] + 1, (n,), device=self.dev, generator=self.gen)
        self.goal_f[ids] = look                                     # per-episode lookahead (frames)
        self.a_prev[ids], self.a_prev2[ids] = 0.0, 0.0
        self.scan_off[ids] = 0.0 if self.eval_mode else self._u(n, 1) * 0.1
        S = tuple(x[ids] for x in self.state())
        self.v_prev[ids] = self.data.body_points(S)[1]
        self.c_prev[ids] = True
        self.tgt_prev[ids], self.occ_prev[ids], self.held_ok[ids], self.plan_m[ids] = 0.0, False, False, False
        self.air[ids] = 0
        if self.rec is not None:
            self.rec.start(ids)
        self.hist[ids] = self._proprio(S, self.a_prev[ids], ids, noise=not self.eval_mode).unsqueeze(1).expand(-1, HIST, -1)

    def reset_all(self):
        self.reset(torch.arange(self.N, device=self.dev))

    # ------------------------------------------------------------ observations
    def _proprio(self, S, a_prev, ids=None, noise=True):
        p, R, v, w, q, qd = S
        calib = self.calib if ids is None else self.calib[ids]
        wb = mtv3(R, w)
        grav = -R[..., 2, :]                                         # R^T (0, 0, -1)
        dq = q - self.default_q + calib
        if noise:
            n = q.shape[0]
            wb = wb + self._u(n, 3) * 0.05
            grav = grav + self._u(n, 3) * 0.0125
            dq = dq + self._u(n, self.NJ) * 0.01
            qd = qd + self._u(n, self.NJ) * 1.5
        return torch.cat((wb, grav, dq, qd, a_prev), -1)

    def observe(self):
        """(actor obs, critic obs) for the current state."""
        S = self.state()
        p, R, v, w, q, qd = S
        f, m, d = self.f, self.motion, self.data
        H = yaw_mat(R)
        noise = not self.eval_mode
        # contact targets: reference-derived next contacts, body-yaw frame, zero when none
        occ = d.occ[f]
        ref_w = d.next[f]
        tgt = mtv3(H.unsqueeze(1), ref_w - p.unsqueeze(1))
        # goal command: bounded direction to the reference root 0.5-1.5 s ahead + relative heading
        g = torch.minimum(f + self.goal_f, m.clip_end[m.clip_id[f]])
        dxy = mtv3(H, m.root_p[g] - p)[:, :2]
        dxy = dxy / dxy.norm(dim=-1, keepdim=True).clamp(min=GOAL_R)
        Rg = m.root_R[g]
        dpsi = torch.atan2(Rg[:, 1, 0], Rg[:, 0, 0]) - torch.atan2(R[:, 1, 0], R[:, 0, 0])
        dpsi = torch.remainder(dpsi + math.pi, 2 * math.pi) - math.pi
        goal = torch.cat((dxy, dpsi.unsqueeze(-1)), -1)
        scan = self._scan(p, R, self.gx_a, self.gy_a)
        if noise:
            tgt = tgt + self._u(self.N, 9, 3) * 0.1
            goal = goal + self._u(self.N, 3) * 0.2
            # paper: per-step Gaussian noise and a per-episode elevation offset; Table S2 gives U(-0.1, 0.1) m for the
            # tracker scan: read here as the per-episode offset, with the planner's N(0, 0.03) as the per-step noise
            scan = scan + self.scan_off + torch.randn(scan.shape, device=self.dev, generator=self.gen) * 0.03
        tgt = tgt * occ.unsqueeze(-1)
        cmd_w, cmd_ok = ref_w, occ
        if self.planner is not None:                                 # stage 2: planner contacts with probability eta_obs
            scan17 = self._scan(p, R, self.gx_c, self.gy_c)
            if noise:
                scan17 = scan17 + torch.randn(scan17.shape, device=self.dev, generator=self.gen) * 0.03
            with torch.no_grad():
                ptgt, pocc = self.planner.decode(self.hist.flatten(1), goal, scan17)
            pm = pocc > 0.5
            self.plan_w, self.plan_m = p.unsqueeze(1) + mv3(H.unsqueeze(1), ptgt), pm
            use = torch.rand(self.N, device=self.dev, generator=self.gen) < self.eta_obs
            if getattr(self, "probe", None) is not None:              # probe envs: planner contacts always (2026-11-12)
                use = use | self.probe
            tgt = torch.where(use[:, None, None], ptgt, tgt)       # the complete command is switched, not interpolated
            cmd_w, cmd_ok = torch.where(use[:, None, None], self.plan_w, ref_w), torch.where(use[:, None], pm, occ)
            both = pm & occ
            self.plan_stats += torch.stack((((self.plan_w - ref_w).norm(dim=-1) * both).sum(), both.float().sum(), use.float().mean(),
                                            torch.ones((), device=self.dev)))
            if self.rec is not None:
                self.rec.observe(self, S, goal, scan17)
        # commanded target per body (world), and the one before its last change, for the contact placement error
        changed = self.occ_prev & ((cmd_w - self.tgt_prev).norm(dim=-1) > 0.05)
        self.tgt_held = torch.where(changed.unsqueeze(-1), self.tgt_prev, self.tgt_held)
        self.held_ok = self.held_ok | changed
        self.tgt_prev, self.occ_prev = cmd_w, cmd_ok
        actor = torch.cat((self.hist.flatten(1), scan, tgt.flatten(1), goal), -1)
        # critic
        rp, rR, rv, rw, rq, rqd = m.ref_state(f)
        ref_pts = d.body_points((rp, rR, rv, rw, rq, rqd))[0]
        ref_rel = mtv3(H.unsqueeze(1), ref_pts - p.unsqueeze(1))
        critic = torch.cat((self._proprio(S, self.a_prev, noise=False), self._scan(p, R, self.gx_c, self.gy_c), mtv3(R, v),
                            mtv3(H, rv), mtv3(H, rw), rq - self.default_q, rqd, ref_rel.flatten(1), d.contact[f].float()), -1)
        return actor, critic

    # ------------------------------------------------------------ step
    def step(self, a):
        N, m, d = self.N, self.motion, self.data
        a = a.clamp(-20.0, 20.0)
        q_tgt = self.default_q + self.calib + self.alpha * a
        self._physics(q_tgt)
        bad = ~(torch.isfinite(self.qpos).all(1) & torch.isfinite(self.qvel).all(1)) | (self.qvel.abs().max(1).values > 200.0)
        if bool(bad.any()):                                           # diverged worlds: park them, end the episode
            ids = bad.nonzero().squeeze(-1)
            self.qpos[ids] = 0.0
            self.qpos[ids, 2], self.qpos[ids, 3] = 1.0, 1.0
            self.qvel[ids] = 0.0
        S = self.state()
        p, R, v, w, q, qd = S
        self.f = f = self.f + 1
        self.ep_step += 1
        rp, rR, rv, rw, rq, rqd = m.ref_state(f)
        pts, vel, _, _ = d.body_points(S)
        rpts = d.body_points((rp, rR, rv, rw, rq, rqd))[0]
        H, rH = yaw_mat(R), yaw_mat(rR)
        mov = (rv.norm(dim=-1) > 0.1).float()
        e_pos = (p - rp).norm(dim=-1)
        th = rot_angle(R, rR)
        Fz, Fxy = self.contact_forces()
        c_meas = (Fz > FORCE_MIN).float()
        c_ref = d.contact[f].float()
        pb = mtv3(H.unsqueeze(1), pts - p.unsqueeze(1))
        rpb = mtv3(rH.unsqueeze(1), rpts - rp.unsqueeze(1))
        acc = ((vel - self.v_prev) / DT).norm(dim=-1).sum(-1)
        tau = self.kp * (q_tgt - q) - self.kd * qd
        tau_app = torch.maximum(torch.minimum(tau, self.effort), -self.effort)
        t = {
            "track_base_pos": torch.exp(-mov * e_pos**2 / SIG_P**2 - th**2 / SIG_R**2),
            "track_base_vel": torch.exp(-((v - rv) ** 2).sum(-1) / SIG_V**2 - ((w - rw) ** 2).sum(-1) / SIG_W**2),
            "track_joint_pos": torch.exp(-((q - rq) ** 2).sum(-1) / SIG_Q**2),
            "track_joint_vel": torch.exp(-((qd - rqd) ** 2).sum(-1) / SIG_QD**2),
            "track_contact_label": (c_meas * (2 * c_ref - 1)).mean(-1),
            "track_contact_pos_base": torch.exp(-((pb - rpb) ** 2).sum((-1, -2)) / SIG_C**2),
            "track_contact_pos_global": torch.exp(-mov * ((pts - rpts) ** 2).sum((-1, -2)) / SIG_C**2),
            "action_rate": ((a - self.a_prev) ** 2).sum(-1),
            "action_rate_2nd": ((a - 2 * self.a_prev + self.a_prev2) ** 2).sum(-1),
            "joint_torque": (tau_app**2).sum(-1),
            "torque_limits": (tau - tau_app).abs().sum(-1),
            "dof_pos_limits": (torch.relu(self.soft_lo - q) + torch.relu(q - self.soft_hi)).sum(-1).clamp(max=30.0),
            "dof_vel_limits": torch.relu(qd.abs() - 0.9 * self.qd_max).clamp(max=1.0).sum(-1),
            "contact_acc": acc.clamp(max=1e4),
        }
        mb = self.plan_m.float()                                      # Table 3 track_planned_contact (stage 2: weight 1.5 eta_rew)
        t["track_planned_contact"] = torch.exp(-(mb * ((pts - self.plan_w) ** 2).sum(-1)).sum(-1) / (SIG_C**2 * mb.sum(-1).clamp(min=1.0)))
        wts = self.weights()
        r = DT * sum(wts[k] * x for k, x in t.items())
        term_pos, term_rot = (e_pos > TERM_POS) & ~bad, (th > TERM_ROT) & ~bad
        if self.fall_rule:                                            # evaluation of planner-driven runs: a failure is a fall, not
            fx = torch.cat((p[:, None, :2], pts[:, :2, :2]), 1)       # a distance from the reference (the robot may be early or late)
            ground = self.terrain.height(fx[..., 0], fx[..., 1], m.clip_id[f]).amin(-1)   # lowest ground under root and feet
            term_pos = ((p[:, 2] - ground) < FALL_H) & ~bad
            term_rot = (R[:, 2, 2] < 0.5) & ~bad                      # tilted more than 60 degrees
        term = term_pos | term_rot | bad
        # contact placement error (paper, Results 0.3): commanded vs realised contact position at contact onset
        c_now = c_meas > 0.5
        # the commanded target switches to the following step around touchdown, so the realised contact is compared with the
        # nearer of the current command and the one before its last change
        big = torch.full_like(e_pos, 1e3).unsqueeze(-1)
        dmin = torch.minimum(torch.where(self.occ_prev, (pts - self.tgt_prev).norm(dim=-1), big),
                             torch.where(self.held_ok, (pts - self.tgt_held).norm(dim=-1), big))
        # a contact event = touchdown after at least 0.1 s without contact (re-contacts within a stance are not events)
        onset = c_now & ~self.c_prev & (self.air >= 5) & (self.occ_prev | self.held_ok) & (self.ep_step > 1).unsqueeze(-1)
        place = dmin * onset
        self.air = torch.where(c_now, torch.zeros_like(self.air), self.air + 1)
        self.c_prev = c_now
        clip = m.clip_id[f]
        timeout = (f >= m.clip_end[clip]) & ~term
        done = term | timeout
        ph = (f - m.clip_start[clip]).float() / (m.clip_len[clip] - 1).float().clamp(min=1)
        finished = done & ~term & (ph >= CSI_DONE)
        if self.csi and not self.eval_mode:                           # Table S4: -0.05 after reaching phase 0.99, +0.05 otherwise
            self.phase = torch.where(done, (self.phase + torch.where(finished, -CSI_STEP, CSI_STEP)).clamp(0.0, CSI_MAX), self.phase)
        self.a_prev2, self.a_prev = self.a_prev, a.clone()
        self.v_prev = vel
        self.hist = torch.cat((self.hist[:, 1:], self._proprio(S, self.a_prev, noise=not self.eval_mode).unsqueeze(1)), 1)
        info = {"terms": {k: x.mean() for k, x in t.items()}, "e_pos": e_pos.mean(), "e_rot": th.mean(),
                "e_joint": (q - rq).abs().mean(), "e_contact_pos": (pb - rpb).norm(dim=-1).mean(),
                "contact_match": ((c_meas > 0.5) == (c_ref > 0.5)).float().mean(),
                "term": term.float().sum(), "done": done.float().sum(), "finished": finished.float().sum(),
                "bad": bad.float().sum(), "done_mask": done, "term_mask": term, "finished_mask": finished,
                "term_pos": term_pos.float().sum(), "term_rot": term_rot.float().sum(), "timeout": timeout.float().sum(),
                "place_sum": place.sum(), "place_n": onset.float().sum(), "place": place, "onset": onset,
                "phase": ph, "e_pos_env": e_pos, "reward_max": DT * sum(v for v in wts.values() if v > 0),
                "c_ref_frac": c_ref[:, :2].mean(), "c_meas_frac": c_meas[:, :2].mean(), "speed": v[:, :2].norm(dim=-1).mean(),
                "ref_speed": rv[:, :2].norm(dim=-1).mean(),
                "set": self.data.clip_set[clip]}
        ids = done.nonzero().squeeze(-1)
        if self.rec is not None:                                      # hindsight buffer admission: reached 90 % of the clip
            self.rec.end(done, done & ~term & (ph >= 0.9))
        if not self.eval_mode:
            self.reset(ids)
        return r, term, timeout, info


def _weights(self):
    """Reward weights of the current stage (Table 3): stage 2 moves the three reference-contact terms to the planner term."""
    w = dict(W)
    e = float(self.eta_rew) if self.planner is not None else 0.0
    for k in ("track_contact_label", "track_contact_pos_base", "track_contact_pos_global"):
        w[k] = W[k] * (1.0 - e)
    w["track_planned_contact"] = 1.5 * e
    return w


AcpEnv.weights = _weights


def x2_leg_bodies(m):
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)]
    return [n for n in names if n and any(k in n for k in ("hip_", "knee_", "ankle_"))]
