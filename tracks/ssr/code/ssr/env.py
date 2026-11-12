"""SSR environment on MuJoCo Warp: X2, terrain tiles, depth, commands, rewards (paper App. A), per-world DR.

One step = 4 x 5 ms physics substeps, then a kinematics refresh so body poses match the new qpos. All observations
are tensors on the GPU. Joint vectors are in the paper's 21-joint order (ssr.x2.NAMES21); feet / hands are [R, L].
"""
from __future__ import annotations

import math

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp

from ssr import terrain as T
from ssr import x2 as X

H = 5                     # proprioception history (paper-silent)
N21 = 21
PROP_DIM = 72
HB_NX, HB_NY, HF_NX, HF_NY = 17, 10, 10, 5
PRIV_DIM = PROP_DIM + 3 + 6 + 2 + 6 + 6 + HB_NX * HB_NY + 2 * HF_NX * HF_NY
AMP_DIM, AMP_FRAMES = 63, 5
DEPTH_N = 36
DEPTH_MIN, DEPTH_MAX = 0.1, 2.0
EPISODE_S = 20.0
FOOT_FORCE_ON = 5.0
GUIDE_LEVEL = 2.0         # paper Sec. 3.2: imagined guidance switches on, for good, once the global mean terrain level exceeds 2
EPS_H = 0.03              # support height tolerance (paper-silent)
SIG_V, SIG_W, SIG_G, SIG_H, SIG_D, SIG_F = 0.25, 0.25, 0.01, 0.01, 0.03, 0.0625
H_TARGET, T_AIR, D_Y_MIN, EPS_CMD = 0.68, 0.4, 0.22, 0.15
PUSH_STEPS = int(12.0 / X.CONTROL_DT)
GATED_TERMS = ("orientation", "ang_vel", "base_height", "ang_vel_xy", "lateral")   # t_e: posture/alive-like terms a stander collects
CORE_TERMS = ("lin_vel", "ang_vel", "orientation", "ang_vel_xy", "base_height", "lateral", "dof_dev", "stand_still")   # tracking + posture: the set run d2 moved with
SAMPLE_K = torch.tensor([-1.0, 0.0, 1.0])  # imagined landing: Gaussian-weighted 3 x 3 stencil


def quat_to_mat(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)), -1).reshape(*q.shape[:-1], 3, 3)


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class SSREnv:
    def __init__(self, num_envs: int, device="cuda:0", raster: dict | None = None, vision: dict | None = None,
                 walls: list | None = None, train: bool = True, spawn=None, camera: str = "head", map_memory: bool = False, tiles: bool | None = None, njmax: int = 900, nconmax: int = 150, seed: int = 0):
        self.N, self.dev, self.train = num_envs, torch.device(device), train
        self.spawn = spawn
        self.camera, self.map_memory = camera, map_memory
        self.raster = raster or T.generate(seed)
        self.tiles = raster is None if tiles is None else tiles
        self.hmap = T.HeightMap(self.raster, device, vision or self.raster.get("vision"))
        self.obst = None
        if self.tiles and "vision" in self.raster:             # side obstacles: seen in depth and in the critic's map, touching one ends the episode
            self.obst = (self.hmap.vz - self.hmap.z) > 0.25
            self.lane = torch.as_tensor(self.raster["lane"], device=device)
        self.x2 = X.build(self.raster, walls)
        m = self.x2.model
        d = mujoco.MjData(m)
        d.qpos[:3] = [0.0, 0.0, 0.75]
        d.qpos[3], d.qpos[7:] = 1.0, self.x2.default_q
        mujoco.mj_forward(m, d)
        with wp.ScopedDevice(str(self.dev)):
            batch = {k: num_envs for k in ("geom_friction", "body_ipos", "body_mass", "body_inertia", "actuator_gainprm",
                                           "actuator_biasprm")}
            self.mm = mjw.put_model(m, batch_sizes=batch)
            self.dd = mjw.put_data(m, d, nworld=num_envs, njmax=njmax, nconmax=nconmax)
        dd, mm = self.dd, self.mm
        t = wp.to_torch
        self.qpos, self.qvel, self.ctrl = t(dd.qpos), t(dd.qvel), t(dd.ctrl)
        self.xpos, self.xmat, self.tau = t(dd.xpos), t(dd.xmat), t(dd.actuator_force)
        self.friction, self.ipos, self.mass, self.inertia = t(mm.geom_friction), t(mm.body_ipos), t(mm.body_mass), t(mm.body_inertia)
        self.gain, self.bias = t(mm.actuator_gainprm), t(mm.actuator_biasprm)
        self.c_geom, self.c_world, self.c_adr = t(dd.contact.geom), t(dd.contact.worldid), t(dd.contact.efc_address)
        self.c_frame, self.c_fric, self.efc_f = t(dd.contact.frame), t(dd.contact.friction), t(dd.efc.force)
        self.nacon = t(dd.nacon)

        dev = self.dev
        f32 = lambda a: torch.tensor(np.asarray(a), device=dev, dtype=torch.float32)
        body = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
        self.idx = torch.tensor(self.x2.idx21, device=dev)
        self.q0_all, self.scale_all = f32(self.x2.default_q), f32(self.x2.action_scale)
        self.q0, self.scale = self.q0_all[self.idx], self.scale_all[self.idx]
        self.q_lo, self.q_hi, self.tau_max, self.qd_max = (f32(a)[self.idx] for a in (self.x2.q_lo, self.x2.q_hi, self.x2.effort, self.x2.qd_max))
        self.kp0, self.kd0 = f32(self.x2.kp), f32(self.x2.kd)
        self.feet = torch.tensor([body(b) for b in X.FOOT_BODIES], device=dev)
        self.hands = torch.tensor([body(b) for b in X.HAND_BODIES], device=dev)
        self.head, self.pelvis = body("head_pitch_link"), body("pelvis")
        cam = X.CAMERAS[self.camera]
        self.cam_dr_pos = torch.tensor(cam.get("dr_pos", (0.01, 0.01, 0.01)), device=dev, dtype=torch.float32)          # Table 10: +-1 cm, +-1 deg unless the camera says otherwise
        self.cam_dr_rpy = torch.tensor(cam.get("dr_rpy_deg", (1.0, 1.0, 1.0)), device=dev, dtype=torch.float32) * math.pi / 180
        self.cam_fov0, self.depth_max, self.march = cam.get("fov", X.CAM_FOV), cam.get("dmax", DEPTH_MAX), cam.get("march", 26)
        self.cam_body, self.cam_off, self.cam_pitch = body(cam["body"]), torch.tensor(cam["pos"], device=dev, dtype=torch.float32), cam["pitch"]
        self.occ_body = torch.tensor([body(b) for b, _, _ in X.OCCLUDERS], device=dev)
        self.occ_pos = torch.tensor([p_ for _, p_, _ in X.OCCLUDERS], device=dev, dtype=torch.float32)
        self.occ_r = torch.tensor([r_ for _, _, r_ in X.OCCLUDERS], device=dev, dtype=torch.float32)
        self.clear_ids = torch.tensor([body(b) for b, _ in X.CLEARANCE], device=dev)
        pts = [f"{a}_{b}" for a in ("left", "right") for b in ("shoulder_roll_link", "elbow_link", "wrist_roll_link", "hip_yaw_link", "knee_link", "ankle_roll_link")]
        self.side_ids = torch.tensor([body(b) for b in pts], device=dev)
        self.clear_m = torch.tensor([r for _, r in X.CLEARANCE], device=dev)
        self.mass0, self.inertia0, self.ipos0 = self.mass[0].clone(), self.inertia[0].clone(), self.ipos[0].clone()
        side = torch.full((m.ngeom,), -1, device=dev, dtype=torch.long)
        term = torch.zeros(m.ngeom, device=dev, dtype=torch.bool)
        for g in range(m.ngeom):
            b = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g])
            if b in X.FOOT_BODIES:
                side[g] = X.FOOT_BODIES.index(b)
            if b in X.TERMINATE_BODIES:
                term[g] = True
        self.geom_side, self.geom_term = side, term
        self.robot_geoms = torch.tensor(np.nonzero(m.geom_bodyid > 0)[0], device=dev)
        self.friction[:, (m.geom_bodyid == 0), 0] = 0.2         # terrain low, so max-combination gives the robot's value

        n = num_envs
        self.gen = torch.Generator(device=dev).manual_seed(seed)
        z = lambda *s: torch.zeros(n, *s, device=dev)
        self.a, self.a1, self.a2 = z(N21), z(N21), z(N21)
        self.delay_buf = z(3, N21)
        self.delay = torch.zeros(n, dtype=torch.long, device=dev)
        self.cmd, self.speed, self.heading = z(3), z(), z()
        self.level = torch.zeros(n, dtype=torch.long, device=dev)
        self.type = torch.zeros(n, dtype=torch.long, device=dev)
        self.free = torch.zeros(n, dtype=torch.bool, device=dev)
        self.origin = z(2)
        self.stall_pos = z(2)
        self.ep = torch.zeros(n, dtype=torch.long, device=dev)
        self.push = torch.zeros(n, dtype=torch.long, device=dev)
        self.hist = z(H, PROP_DIM)
        self.amp_hist = z(AMP_FRAMES, AMP_DIM)
        self.foot_prev = z(2, 3)
        self.vf = z(2, 3)
        self.air = z(2)
        self.single_t = z()
        self.contact_prev = torch.zeros(n, 2, dtype=torch.bool, device=dev)
        self.F = z(2, 3)
        self.term_force = z()
        self.cam_pos, self.cam_rpy, self.cam_fov = z(3), z(3), z(2)
        self.motor = torch.ones(n, device=dev)
        rays = T.camera_rays(DEPTH_N, (cam.get("fov_h", self.cam_fov0), self.cam_fov0), dev)
        self.rays = rays
        self.half_fov = math.tan(self.cam_fov0 / 2)
        self.depth = z(DEPTH_N, DEPTH_N)
        self.map_img = None
        if self.map_memory:
            from ssr.mapcache import MapCache
            self.cache = MapCache(n, dev, bias_v=0.05 if train else 0.0, bias_w=0.02 if train else 0.0, noise_v=0.02 if train else 0.0)
            self.prev_pose = z(4)
        self._hb = self._grid(HB_NX, HB_NY, -0.6, 0.1, 0.1)
        sx, sy = X.SOLE_HALF
        self._hf = self._grid(HF_NX, HF_NY, -sx, 2 * sx / (HF_NX - 1), 2 * sy / (HF_NY - 1))
        self._stencil = torch.stack(torch.meshgrid(SAMPLE_K.to(dev), SAMPLE_K.to(dev), indexing="ij"), -1).reshape(-1, 2)
        w = torch.exp(-0.5 * (self._stencil ** 2).sum(-1))
        self._stencil_w = w / w.sum()
        self._mask = wp.zeros(n, dtype=bool, device=str(dev))
        self._mask_t = t(self._mask)
        self._g_step = self._g_ref = None
        self.log = {}
        self.rsi_frames = None                  # walking states to start episodes from (set_rsi)
        self.air_bonus = 0.0
        self.imit_w = 0.0
        self.boot = 0.0                         # 1 -> 0 over early training: velocity tracking up, joint-deviation penalty down
        self.stuck_term = False                 # not in the paper: end episodes that make no progress (off; --stuck_term)
        self.guide_on = False                   # latched by curriculum() when the mean level passes GUIDE_LEVEL
        self.gait = {}
        self.flat_only = False                  # diagnosis: every robot on the open flat tile
        self.only_terms = None                  # diagnosis: keep only these locomotion terms
        self.v_min, self.sig_v = 0.0, SIG_V      # command curriculum (not in the paper): minimum commanded speed and tracking kernel width, set per iteration by train
        self.prog_gate = 0.0                    # t_e (not in the paper): weight of the forward-progress gate on GATED_TERMS for robots with a moving command
        self.v_ema = torch.zeros(n, 2, device=dev)   # base planar velocity (body frame), 0.5 s moving average, for the progress gate
        self.prog = torch.ones(n, device=dev)
        self.late_scale = 1.0                   # staged schedule (not in the paper): factor on every term outside CORE_TERMS, 0 -> 1 early in training
        self.cnt = {}                           # raw counts for the log (termination causes, touchdown support)
        self.reset(torch.arange(n, device=dev))
        self.refresh()
        self._contacts()
        self._update_obs(torch.arange(n, device=dev))

    # ---------------------------------------------------------------- helpers
    def _grid(self, nx, ny, x0, dx, dy):
        xs = x0 + dx * torch.arange(nx, device=self.dev, dtype=torch.float32)
        ys = dy * (torch.arange(ny, device=self.dev, dtype=torch.float32) - (ny - 1) / 2)
        gx, gy = torch.meshgrid(xs, ys, indexing="ij")
        return torch.stack((gx.flatten(), gy.flatten()), -1)

    def _u(self, *s, lo=-1.0, hi=1.0):
        return torch.rand(*s, device=self.dev, generator=self.gen) * (hi - lo) + lo

    def _physics(self):
        def sub():
            for _ in range(X.DECIMATION):
                mjw.step(self.mm, self.dd)
        if self._g_step is None:
            with wp.ScopedDevice(str(self.dev)):
                sub()
                with wp.ScopedCapture() as cap:
                    sub()
            self._g_step = cap.graph
        else:
            wp.capture_launch(self._g_step)

    def refresh(self):
        if self._g_ref is None:
            with wp.ScopedDevice(str(self.dev)):
                mjw.kinematics(self.mm, self.dd)
                with wp.ScopedCapture() as cap:
                    mjw.kinematics(self.mm, self.dd)
            self._g_ref = cap.graph
        else:
            wp.capture_launch(self._g_ref)

    # ---------------------------------------------------------------- state
    @property
    def R(self):
        q = self.qpos[:, 3:7]
        return quat_to_mat(q / q.norm(dim=-1, keepdim=True))

    def _state(self):
        R = self.R
        Rt = R.transpose(-1, -2)
        s = dict(R=R, p=self.qpos[:, :3], w_b=self.qvel[:, 3:6], v_b=(Rt @ self.qvel[:, :3, None]).squeeze(-1))
        s["g"] = -R[:, 2, :]
        s["yaw"] = torch.atan2(R[:, 1, 0], R[:, 0, 0])
        s["q"] = self.qpos[:, 7:][:, self.idx]
        s["qd"] = self.qvel[:, 6:][:, self.idx]
        s["tau"] = self.tau[:, self.idx]
        return s

    def _heading_points(self, p, yaw, grid):
        c, sn = torch.cos(yaw), torch.sin(yaw)
        gx, gy = grid[..., 0], grid[..., 1]
        return p[..., 0:1] + c[..., None] * gx - sn[..., None] * gy, p[..., 1:2] + sn[..., None] * gx + c[..., None] * gy

    def _foot_state(self):
        fp = self.xpos[:, self.feet]
        m = self.xmat[:, self.feet]
        fyaw = torch.atan2(m[..., 1, 0], m[..., 0, 0])
        c, s = torch.cos(fyaw), torch.sin(fyaw)
        sole = torch.stack((fp[..., 0] + c * X.SOLE_OFFSET_X, fp[..., 1] + s * X.SOLE_OFFSET_X, fp[..., 2] - X.SOLE_Z), -1)
        return fp, fyaw, sole

    def _contacts(self):
        self.F[:] = 0.0
        self.term_force[:] = 0.0
        n = int(self.nacon[0].item()) if self.nacon.dim() else int(self.nacon.item())
        n = min(n, self.c_adr.shape[0])
        if n == 0:
            return
        adr = self.c_adr[:n, 0]
        keep = (adr >= 0) & (adr + 3 < self.efc_f.shape[1]) & (self.c_world[:n] >= 0)
        adr, w, geom = adr[keep], self.c_world[:n][keep], self.c_geom[:n][keep]
        f = torch.stack([self.efc_f[w, adr + k] for k in range(4)], -1)
        mu = self.c_fric[:n, 0][keep]
        fr = self.c_frame[:n][keep]
        if fr.dim() == 2:
            fr = fr.reshape(-1, 3, 3)
        fc = torch.stack((f.sum(1), mu * (f[:, 0] - f[:, 1]), mu * (f[:, 2] - f[:, 3])), -1)
        Fw = (fr.transpose(-1, -2) @ fc[..., None]).squeeze(-1)
        for col in (0, 1):
            s = self.geom_side[geom[:, col]]
            ok = s >= 0
            ix = w[ok] * 2 + s[ok]
            Fflat = self.F.view(-1, 3)
            for d in range(3):
                Fflat[:, d].index_add_(0, ix, Fw[ok, d].abs() if d == 2 else Fw[ok, d])
        t_ok = self.geom_term[geom[:, 0]] | self.geom_term[geom[:, 1]]
        self.term_force.index_add_(0, w[t_ok], fc[t_ok, 0])

    def set_rsi(self, frames, frac=0.5, min_speed=0.0):
        """Reference state initialization: `frac` of the walking episodes start from captured flat-walking frames (63-D AMP frames)."""
        self.rsi_frames, self.rsi_frac = frames, frac
        self.rsi_ok = (frames[:, 3] >= min_speed).nonzero().squeeze(-1) if min_speed > 0 else torch.arange(len(frames), device=self.dev)
        self.rsi_idx = torch.full((self.N,), -1, dtype=torch.long, device=self.dev)

    def _apply_rsi(self, ids):
        """Pose and velocities from a random walking frame, heading as commanded, lowest foot on the ground; commanded speed = the frame's."""
        n = len(ids)
        self.rsi_idx[ids] = -1
        pick = (self._u(n, lo=0.0, hi=1.0) < self.rsi_frac) & ~self.free[ids] & (self.speed[ids] >= 0)
        rid = ids[pick]
        k = len(rid)
        if k == 0:
            return
        idx = self.rsi_ok[torch.randint(0, len(self.rsi_ok), (k,), device=self.dev, generator=self.gen)]
        f = self.rsi_frames[idx]
        self.rsi_idx[rid] = idx
        g, v, w, q, qd, feet = f[:, :3], f[:, 3:6], f[:, 6:9], f[:, 9:30], f[:, 30:51], f[:, 57:63].reshape(k, 2, 3)
        pitch, roll = torch.asin(g[:, 0].clamp(-1, 1)), torch.atan2(-g[:, 1], -g[:, 2])
        yaw = self.heading[rid]
        cr, sr, cp, sp, cy, sy = torch.cos(roll / 2), torch.sin(roll / 2), torch.cos(pitch / 2), torch.sin(pitch / 2), torch.cos(yaw / 2), torch.sin(yaw / 2)
        quat = torch.stack((cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy), -1)
        Rm = quat_to_mat(quat)
        lowest = ((Rm[:, 2, None] * feet).sum(-1)).amin(-1)
        gz = self.hmap(self.qpos[rid, 0], self.qpos[rid, 1])
        self.qpos[rid, 2] = gz - lowest + 0.078
        self.qpos[rid, 3:7] = quat
        q_all = self.q0_all.expand(k, -1).clone()
        qd_all = torch.zeros(k, 31, device=self.dev)
        q_all[:, self.idx] = (self.q0 + q).clamp(self.q_lo, self.q_hi)
        qd_all[:, self.idx] = qd
        self.qpos[rid, 7:] = q_all
        self.qvel[rid, :3] = (Rm @ v[..., None]).squeeze(-1)
        self.qvel[rid, 3:6] = w
        self.qvel[rid, 6:] = qd_all
        self.ctrl[rid] = q_all
        self.speed[rid] = v[:, 0].clamp(0.0, 1.2)

    # ---------------------------------------------------------------- reset
    def _terrain_reset(self, ids):
        n = len(ids)
        r = self._u(n, lo=0.0, hi=1.0)
        self.free[ids] = (r < 0.1) & self.train
        typ = self.type[ids]
        lvl = self.level[ids]
        free = self.free[ids]
        c = torch.where(free, torch.full_like(typ, T.FLAT), typ).float()
        lv = torch.where(free, torch.zeros_like(lvl), lvl).float()
        self.origin[ids, 0] = c * T.TILE_X
        self.origin[ids, 1] = lv * T.TILE_Y
        x = self.origin[ids, 0] + torch.where(free, torch.full((n,), 4.0, device=self.dev), self._u(n, lo=0.35, hi=0.9))
        spread = torch.full((n,), 0.6, device=self.dev)
        if self.obst is not None:
            spread = torch.minimum(spread, (self.lane[c.long(), lv.long()] - 0.32).clamp(min=0.02))
        y = self.origin[ids, 1] + T.TILE_Y / 2 + self._u(n) * torch.where(free, torch.full((n,), 1.0, device=self.dev), spread)
        return x, y

    def _command_reset(self, ids):
        n, typ = len(ids), self.type[ids]
        vlo = torch.where(typ == T.SLOPE, -1.0 * torch.ones(n, device=self.dev), torch.zeros(n, device=self.dev))
        vhi = torch.where(typ == T.SLOPE, torch.ones(n, device=self.dev), 1.2 * torch.ones(n, device=self.dev))
        hmax = torch.tensor([80.0, 30.0, 30.0, 15.0, 30.0], device=self.dev)[typ] * math.pi / 180
        self.speed[ids] = vlo + (vhi - vlo) * self._u(n, lo=0.0, hi=1.0)
        if self.v_min > 0.0:                                         # early in training: no stand or slow commands
            sp_ = self.speed[ids]
            sg_ = torch.where(sp_ < 0, -torch.ones_like(sp_), torch.ones_like(sp_))
            self.speed[ids] = sg_ * (self.v_min + (1.0 - self.v_min / 1.2) * sp_.abs()).clamp(max=1.2)
        self.heading[ids] = self._u(n) * hmax
        f = self.free[ids]
        if bool(f.any()):
            fi = ids[f]
            k = len(fi)
            self.cmd[fi] = torch.stack((self._u(k), self._u(k) * 0.4, self._u(k) * 1.2), -1)

    def reset(self, ids):
        n = len(ids)
        if n == 0:
            return
        if self.tiles:
            if self.train:
                self.type[ids] = self.type[ids] if self.ep.sum() else torch.randint(0, 5, (n,), device=self.dev, generator=self.gen)
                if not self.ep.sum():
                    self.level[ids] = torch.randint(0, 4, (n,), device=self.dev, generator=self.gen)
            if self.flat_only:
                self.type[ids], self.level[ids] = T.FLAT, 0
            x, y = self._terrain_reset(ids)
            self._command_reset(ids)
            yaw = self.heading[ids]
            rev = (self.speed[ids] < 0) & ~self.free[ids]             # backward walks on slopes start at the far end
            x = torch.where(rev, 2 * self.origin[ids, 0] + T.TILE_X - x, x)
        else:
            x, y, yaw = self.spawn[:, 0][ids], self.spawn[:, 1][ids], self.spawn[:, 2][ids]
        self._mask_t.zero_()
        self._mask_t[ids] = True
        mjw.reset_data(self.mm, self.dd, self._mask)
        gz = self.hmap(x, y)
        self.qpos[ids, 0], self.qpos[ids, 1], self.qpos[ids, 2] = x, y, gz + 0.665
        self.qpos[ids, 3] = torch.cos(yaw / 2)
        self.qpos[ids, 4], self.qpos[ids, 5] = 0.0, 0.0
        self.qpos[ids, 6] = torch.sin(yaw / 2)
        noise = self._u(n, 31) * 0.05 if self.train else 0.0
        q = (self.q0_all + noise)
        self.qpos[ids, 7:] = q
        self.qvel[ids] = 0.0
        self.ctrl[ids] = self.q0_all
        if self.train and self.tiles and self.rsi_frames is not None:
            self._apply_rsi(ids)
        if self.train:
            self._domain_randomize(ids)
            self.delay[ids] = torch.randint(0, 3, (n,), device=self.dev, generator=self.gen)
        self.a[ids], self.a1[ids], self.a2[ids] = 0.0, 0.0, 0.0
        self.delay_buf[ids] = 0.0
        self.ep[ids] = 0
        self.push[ids] = PUSH_STEPS
        self.stall_pos[ids] = self.qpos[ids, :2]
        self.air[ids], self.single_t[ids] = 0.0, 0.0
        self.v_ema[ids] = 0.0
        self.contact_prev[ids] = False
        self.reset_ids = ids

    def _domain_randomize(self, ids):
        n, u = len(ids), self._u
        self.friction[ids[:, None], self.robot_geoms, 0] = u(n, lo=0.2 + 0.5 * self.boot, hi=1.25)[:, None]     # slippery floors and heavy loads ease in with training
        mass = self.mass0.expand(n, -1).clone() * u(n, self.mass0.shape[0], lo=0.9, hi=1.1)
        mass[:, self.pelvis] = self.mass0[self.pelvis] + u(n, lo=-2.0, hi=12.5) * (1.0 - self.boot)
        self.mass[ids] = mass
        self.inertia[ids] = self.inertia0[None] * (mass / self.mass0.clamp(min=1e-6))[..., None]
        ipos = self.ipos0[None] + u(n, *self.ipos0.shape) * 0.01
        ipos[:, self.pelvis] = self.ipos0[self.pelvis] + u(n, 3) * 0.1
        self.ipos[ids] = ipos
        m, g = u(n, lo=0.8, hi=1.2), u(n, 31, lo=0.9, hi=1.1)
        d = u(n, 31, lo=0.9, hi=1.1)
        self.motor[ids] = m
        kp, kd = self.kp0 * g * m[:, None], self.kd0 * d * m[:, None]
        self.gain[ids, :, 0] = kp
        self.bias[ids, :, 1], self.bias[ids, :, 2] = -kp, -kd
        self.cam_pos[ids] = u(n, 3) * self.cam_dr_pos
        self.cam_rpy[ids] = u(n, 3) * self.cam_dr_rpy
        self.cam_fov[ids] = u(n, 2) * math.radians(1.0)

    # ---------------------------------------------------------------- observations
    def _render(self):
        pos = self.xpos[:, self.cam_body]
        Rh = self.xmat[:, self.cam_body]
        if Rh.dim() == 2:
            Rh = Rh.reshape(-1, 3, 3)
        off = self.cam_off[None] + self.cam_pos
        o = pos + (Rh @ off[..., None]).squeeze(-1)
        pitch = self.cam_pitch + self.cam_rpy[:, 1]
        c, s = torch.cos(pitch), torch.sin(pitch)
        e, zr = torch.ones_like(c), torch.zeros_like(c)
        Ry = torch.stack((c, zr, s, zr, e, zr, -s, zr, c), -1).reshape(-1, 3, 3)
        cr, sr, cy, sy = torch.cos(self.cam_rpy[:, 0]), torch.sin(self.cam_rpy[:, 0]), torch.cos(self.cam_rpy[:, 2]), torch.sin(self.cam_rpy[:, 2])
        Rx = torch.stack((e, zr, zr, zr, cr, -sr, zr, sr, cr), -1).reshape(-1, 3, 3)
        Rz = torch.stack((cy, -sy, zr, sy, cy, zr, zr, zr, e), -1).reshape(-1, 3, 3)
        Rc = Rh @ Rz @ Ry @ Rx                                    # mount yaw, pitch (nominal + offset), roll
        sc = torch.stack((e, 1 + self.cam_fov[:, 0] / self.cam_fov0, 1 + self.cam_fov[:, 1] / self.cam_fov0), -1)
        rays = self.rays[None] * sc[:, None]
        rays = rays / rays.norm(dim=-1, keepdim=True)
        d = (Rc[:, None] @ rays[..., None]).squeeze(-1)
        tmax = self.depth_max / 0.4
        t = self.hmap.depth(o, d, tmax=tmax, iters=self.march)
        t_occ = self._occlusion(o, d, tmax)
        self_hit = t_occ < t
        t = torch.min(t, t_occ)
        z_depth = t * rays[..., 0]                                # along the optical axis, as a depth camera reports
        if self.train:                                            # Table 10: self-occluded pixels get N(0, (0.1 m)^2)
            z_depth = z_depth + 0.1 * torch.randn(z_depth.shape, device=self.dev, generator=self.gen) * self_hit.float()
        self.depth = ((z_depth.clamp(DEPTH_MIN, self.depth_max) - DEPTH_MIN) / (self.depth_max - DEPTH_MIN)).reshape(-1, DEPTH_N, DEPTH_N)
        self._ray_o, self._ray_d, self._ray_t = o, d, t

    def _occlusion(self, o, d, tmax):
        """First hit of the rays on the limb spheres (self-occlusion, as in the paper's depth rendering)."""
        Rb = self.xmat[:, self.occ_body]
        if Rb.dim() == 3:
            Rb = Rb.reshape(self.N, -1, 3, 3)
        cen = self.xpos[:, self.occ_body] + (Rb @ self.occ_pos[None, ..., None]).squeeze(-1)
        best = torch.full(d.shape[:2], tmax, device=d.device)
        for k in range(cen.shape[1]):
            oc = cen[:, k, None] - o[:, None]
            b = (oc * d).sum(-1)
            disc = b * b - (oc * oc).sum(-1) + self.occ_r[k] ** 2
            hit = (disc > 0) & (b > 0)
            tk = torch.where(hit, b - disc.clamp(min=0).sqrt(), torch.full_like(b, tmax)).clamp(min=0.0)
            best = torch.min(best, torch.where(hit, tk, best))
        return best

    def _touches_obstacle(self, radius=0.06):
        """A limb's centre within `radius` of side-obstacle cells (five probe points). Trains the robot to keep its arms and legs clear."""
        p = self.xpos[:, self.side_ids]
        off = torch.tensor([[0.0, 0.0], [radius, 0.0], [-radius, 0.0], [0.0, radius], [0.0, -radius]], device=self.dev)
        i = torch.round((p[..., 0, None] + off[:, 0] - self.hmap.x0) / self.hmap.res).long().clamp(0, self.hmap.nx - 1)
        j = torch.round((p[..., 1, None] + off[:, 1] - self.hmap.y0) / self.hmap.res).long().clamp(0, self.hmap.ny - 1)
        # only limbs at wall height: below the top of the terrain step under them they would be inside the ground anyway
        hit = self.obst[i, j].any(-1).any(-1)
        self.log["touch"] = self.log.get("touch", 0) + int(hit.sum())
        return hit

    def _update_map(self, ids):
        s = self.s
        px, py, pz, yaw = s["p"][:, 0], s["p"][:, 1], s["p"][:, 2], s["yaw"]
        if not hasattr(self, "_map_init"):
            self._map_init = True
            self.prev_pose = torch.stack((px, py, pz, yaw), -1)
        wx, wy = px - self.prev_pose[:, 0], py - self.prev_pose[:, 1]
        c0, s0 = torch.cos(self.prev_pose[:, 3]), torch.sin(self.prev_pose[:, 3])
        dx, dy = c0 * wx + s0 * wy, -s0 * wx + c0 * wy
        self.cache.shift(dx, dy, wrap(yaw - self.prev_pose[:, 3]), pz - self.prev_pose[:, 2], noisy=self.train)
        self.prev_pose = torch.stack((px, py, pz, yaw), -1)
        if len(ids):
            self.cache.reset(ids)
        # write the depth points, in the heading frame, relative to the base; rays that reached the far range carry no information
        pts = self._ray_o[:, None] + self._ray_t[..., None] * self._ray_d - s["p"][:, None]
        c, sn = torch.cos(yaw)[:, None], torch.sin(yaw)[:, None]
        hx, hy = c * pts[..., 0] + sn * pts[..., 1], -sn * pts[..., 0] + c * pts[..., 1]
        self.cache.write(hx, hy, pts[..., 2], self._ray_t < DEPTH_MAX * 1.2)
        self.map_img = self.cache.image()

    def _base_frame(self, s, pts):
        return ((pts - s["p"][:, None]) @ s["R"]).reshape(pts.shape[0], -1)   # (R^T v)^T = v^T R

    def _update_obs(self, ids):
        s = self._state()
        self.s = s
        fp, fyaw, sole = self._foot_state()
        self.fp, self.fyaw, self.sole = fp, fyaw, sole
        q_rel = s["q"] - self.q0
        n = torch.zeros_like(s["q"]) if not self.train else None
        w, g, q, qd = s["w_b"], s["g"], q_rel, s["qd"]
        if self.train:
            w, g = w + self._u(self.N, 3) * 0.2, g + self._u(self.N, 3) * 0.05
            q, qd = q + self._u(self.N, N21) * 0.01, qd + self._u(self.N, N21) * 0.5
        self.prop = torch.nan_to_num(torch.cat((w, g, self.cmd, q, qd, self.a), -1), 0.0, 0.0, 0.0).clamp(-40.0, 40.0)     # the encoder has no normalizer: keep a bad state from reaching it
        self.hist = torch.cat((self.hist[:, 1:], self.prop[:, None]), 1)
        if len(ids):
            self.hist[ids] = self.prop[ids, None].expand(-1, H, -1)
        self._render()
        if self.map_memory:
            self._update_map(ids)
        hx, hy = self._heading_points(s["p"], s["yaw"], self._hb[None])
        h_b = (s["p"][:, 2:3] - self.hmap.vision(hx, hy)).clamp(-1.5, 1.5)    # includes side obstacles
        sg = self._hf[None, None]
        c, sn = torch.cos(fyaw)[..., None], torch.sin(fyaw)[..., None]
        fx = sole[..., 0:1] + c * sg[..., 0] - sn * sg[..., 1]
        fy = sole[..., 1:2] + sn * sg[..., 0] + c * sg[..., 1]
        self.h_k = self.hmap(fx, fy)
        h_f = (sole[..., 2:3] - self.h_k).clamp(-1.0, 1.0)
        self.h_b, self.h_f = torch.nan_to_num(h_b), torch.nan_to_num(h_f)
        hp = self._base_frame(s, self.xpos[:, self.hands])
        fbase = self._base_frame(s, fp)
        vf_b = (self.vf @ s["R"]).reshape(self.N, -1)
        c_now = (self.F[..., 2].abs() > FOOT_FORCE_ON).float()
        self.priv = torch.nan_to_num(torch.cat((self.prop, s["v_b"], vf_b, c_now, hp, fbase, h_b, h_f.reshape(self.N, -1)), -1), 0.0, 0.0, 0.0).clamp(-40.0, 40.0)
        amp = torch.nan_to_num(torch.cat((s["g"], s["v_b"], s["w_b"], q_rel, s["qd"], hp, fbase), -1), 0.0, 0.0, 0.0).clamp(-40.0, 40.0)
        self.amp_hist = torch.cat((self.amp_hist[:, 1:], amp[:, None]), 1)
        if len(ids):
            self.amp_hist[ids] = amp[ids, None].expand(-1, AMP_FRAMES, -1)
        self.foot_prev = fp.clone()

    # ---------------------------------------------------------------- commands
    def _update_cmd(self):
        if not self.tiles:
            return
        s = self.s
        err = wrap(self.heading - s["yaw"])
        wz = (err * 1.0).clamp(-1.2, 1.2)
        c, sn = torch.cos(s["yaw"]), torch.sin(s["yaw"])
        vx, vy = c * self.speed, -sn * self.speed
        walk = torch.stack((vx.clamp(-1.0, 1.2), vy.clamp(-0.4, 0.4), wz), -1)
        self.cmd = torch.where(self.free[:, None], self.cmd, walk)

    # ---------------------------------------------------------------- foothold support (Sec. 3.2)
    def support_stance(self):
        """rho at the current contact (the sole's own height, Eq. 2), (N, 2)."""
        return 1.0 - ((self.sole[..., 2:3] - self.h_k) < EPS_H).float().mean(-1)

    def support_imagined(self, mu: torch.Tensor, sigma: torch.Tensor):
        """Expected deficiency under the imagined landing Gaussians: mu (N, 2, 2) in the heading frame, sigma (N, 2).
        Stencil 3x3 at -sigma, 0, +sigma."""
        s = self.s
        st = self._stencil[None, None].to(mu.dtype) * sigma[:, :, None, None]         # (N, 2, 9, 2)
        p = mu[:, :, None] + st
        cy, sy = torch.cos(s["yaw"])[:, None, None], torch.sin(s["yaw"])[:, None, None]
        hx = s["p"][:, 0, None, None] + cy * p[..., 0] - sy * p[..., 1]
        hy = s["p"][:, 1, None, None] + sy * p[..., 0] + cy * p[..., 1]
        fyaw = self.fyaw[:, :, None, None]
        sg = self._hf[None, None, None]
        c, sn = torch.cos(fyaw), torch.sin(fyaw)
        sx = hx[..., None] + X.SOLE_OFFSET_X * c + c * sg[..., 0] - sn * sg[..., 1]
        sy = hy[..., None] + X.SOLE_OFFSET_X * sn + sn * sg[..., 0] + c * sg[..., 1]
        h = self.hmap(sx, sy)
        rho = 1.0 - ((h.amax(-1, keepdim=True) - h) < EPS_H).float().mean(-1)         # (N, 2, 9)
        return (rho * self._stencil_w[None, None]).sum(-1)

    # ---------------------------------------------------------------- step
    def step(self, action: torch.Tensor, foot_mu=None, foot_sig=None) -> dict:
        self.a2, self.a1, self.a = self.a1, self.a, action.clone()
        self.delay_buf = torch.cat((action[:, None], self.delay_buf[:, :-1]), 1)
        delayed = self.delay_buf[torch.arange(self.N, device=self.dev), self.delay] if self.train else action
        target = (self.q0 + self.scale * delayed).clamp(self.q_lo, self.q_hi)
        ctrl = self.q0_all.expand(self.N, -1).clone()
        ctrl[:, self.idx] = target
        self.ctrl[:] = ctrl
        if self.train:
            self.push -= 1
            hit = self.push <= 0
            if bool(hit.any()):
                ids = hit.nonzero().squeeze(-1)
                self.qvel[ids, :2] += self._u(len(ids), 2) * 0.4
                self.push[ids] = PUSH_STEPS
        self._physics()
        bad = ~(torch.isfinite(self.qpos).all(-1) & torch.isfinite(self.qvel).all(-1)) | (self.qvel.abs().amax(-1) > 60.0)   # joint limits are 12-15 rad/s
        self.bad = bad
        if bool(bad.any()):                                          # a diverged world: neutral numbers until its reset
            self.log["nan"] = self.log.get("nan", 0) + int(bad.sum())
            self.qpos[bad] = torch.cat((self.origin[bad], torch.ones(int(bad.sum()), 1, device=self.dev), torch.tensor([1.0, 0, 0, 0], device=self.dev).expand(int(bad.sum()), -1), self.q0_all.expand(int(bad.sum()), -1)), -1)
            self.qvel[bad] = 0.0
        self.refresh()
        self._contacts()
        self.F[bad] = 0.0
        self.ep += 1
        fp, _, _ = self._foot_state()
        self.vf = (fp - self.foot_prev) / X.CONTROL_DT
        s = self._state()
        self.s = s
        self.fp, self.fyaw, self.sole = self._foot_state()
        contact = self.F[..., 2].abs() > FOOT_FORCE_ON
        rew = self._rewards(s, contact, foot_mu, foot_sig)
        if self.tiles:                                               # gait numbers for the log: walkers told to go 0.3 m/s or faster
            wk = (~self.free & (self.speed.abs() >= 0.3)).float()
            nw = wk.sum().clamp(min=1.0)
            g_ = self.gait
            for k_, v_ in (("vx_ratio", self.qvel[:, 0] / self.speed.where(self.speed.abs() >= 0.3, torch.ones_like(self.speed))),
                           ("single", (contact.sum(-1) == 1).float()), ("double", (contact.sum(-1) == 2).float()), ("flight", (contact.sum(-1) == 0).float()),
                           ("steps_per_s", (contact & ~self.contact_prev).float().sum(-1) / X.CONTROL_DT),
                           ("cmd_speed", self.speed.abs()), ("vx_mps", self.qvel[:, 0] * torch.sign(self.speed)),
                           ("lin_err", (s["v_b"][:, :2] - self.cmd[:, :2]).norm(dim=-1)), ("yaw_rate_err", (s["w_b"][:, 2] - self.cmd[:, 2]).abs()),
                           ("yaw_rate_cmd", self.cmd[:, 2].abs()), ("power_w", (s["tau"] * s["qd"]).abs().sum(-1)),
                           ("foot_force_max_n", self.F[..., 2].abs().amax(-1))):
                g_[k_] = g_.get(k_, 0.0) + float((torch.nan_to_num(v_) * wk).sum() / nw)
            g_["n"] = g_.get("n", 0) + 1
        touchdown = contact & ~self.contact_prev & (self.air > 0.0)
        mtd = touchdown & (self.cmd[:, :2].norm(dim=-1) > EPS_CMD)[:, None]      # air time of each touchdown under a moving command
        self.cnt["air_n"] = self.cnt.get("air_n", 0) + mtd.sum()
        self.cnt["air_sum"] = self.cnt.get("air_sum", 0) + (self.air * mtd.float()).sum()
        self.cnt["air_long"] = self.cnt.get("air_long", 0) + (mtd & (self.air > 0.2)).sum()
        if self.tiles:                                               # support ratio at touchdown (paper Sec. 4.2: MSR, SFR), off slopes
            td = touchdown & ~((self.type == T.SLOPE) | self.free)[:, None]
            sr = 1.0 - self.rho_c
            for k_, v_ in (("td_n", td.sum()), ("td_support", (sr * td).sum()), ("td_safe", ((sr > 0.75) & td).sum())):
                self.cnt[k_] = self.cnt.get(k_, 0) + v_
        pw_contact = self.sole.clone()
        self.air = torch.where(contact, torch.zeros_like(self.air), self.air + X.CONTROL_DT)
        self.contact_prev = contact.clone()
        term, timeout, success = self._done(s)
        done = term | timeout
        out = dict(r_l=rew[0], r_f=rew[1], term=term, timeout=timeout, done=done, contact=contact, touchdown=touchdown,
                   sole=pw_contact, base_xy=s["p"][:, :2].clone(), yaw=s["yaw"].clone(), type=self.type.clone(), free=self.free.clone())
        ids = done.nonzero().squeeze(-1)
        self.curriculum(ids, success)
        if len(ids) and self.train:
            self.reset(ids)
            self.refresh()
        self.vf[ids] = 0.0
        self._update_cmd()
        self._update_obs(ids)
        self._update_cmd()
        return out

    def _imitation(self, q_rel):
        """Optional warm-up: an episode started from a walking clip is rewarded for following that clip's joints as time goes on (--imit_w)."""
        if self.imit_w <= 0.0 or self.rsi_frames is None:
            return torch.zeros(self.N, device=self.dev)
        k = self.rsi_idx + self.ep
        ok = (self.rsi_idx >= 0) & (k < len(self.rsi_frames))
        ref = self.rsi_frames[k.clamp(0, len(self.rsi_frames) - 1), 9:30]
        return self.imit_w * self.boot * torch.exp(-((q_rel - ref) ** 2).mean(-1) / 0.4) * ok.float()

    def _rewards(self, s, contact, foot_mu, foot_sig):
        N21f = float(N21)
        cmd = self.cmd
        v, w = s["v_b"], s["w_b"]
        q_rel = s["q"] - self.q0
        ground = self.hmap(s["p"][:, 0], s["p"][:, 1])
        qd, tau = s["qd"] / self.qd_max, s["tau"] / self.tau_max
        still = cmd.norm(dim=-1) <= EPS_CMD
        single = contact.sum(-1) == 1
        self.single_t = torch.where(single, torch.zeros_like(self.single_t), self.single_t + X.CONTROL_DT)
        first = contact & ~self.contact_prev
        vf = (self.fp - self.foot_prev) / X.CONTROL_DT
        fb = ((self.fp - s["p"][:, None]) @ s["R"])
        dy = (fb[:, 1, 1] - fb[:, 0, 1]).abs()
        da = self.a - self.a1
        terms = dict(  # Table 6, name -> weighted value
            lin_vel=(1.0 + 2.0 * self.boot) * torch.exp(-((v[:, :2] - cmd[:, :2]) ** 2).sum(-1) / self.sig_v),
            ang_vel=0.8 * torch.exp(-(w[:, 2] - cmd[:, 2]) ** 2 / SIG_W),
            orientation=0.5 * torch.exp(-(s["g"][:, :2] ** 2).sum(-1) / SIG_G),   # squared norm: with the plain norm and 0.01 this term is ~0 at any real tilt
            ang_vel_xy=0.25 * torch.exp(-(w[:, :2] ** 2).sum(-1) / SIG_W),
            base_height=0.4 * torch.exp(-(s["p"][:, 2] - ground - H_TARGET) ** 2 / SIG_H),
            action_rate=-0.12 * (da ** 2).sum(-1) / N21f,
            smoothness=-0.06 * ((self.a - 2 * self.a1 + self.a2) ** 2).sum(-1) / N21f,
            dof_vel=-0.96 * (qd ** 2).sum(-1) / N21f,
            dof_torque=-0.6 * (tau ** 2).sum(-1) / N21f,
            dof_dev=-1.8 * (1.0 - 0.75 * self.boot) * q_rel.abs().sum(-1) / N21f,
            dof_pos_lim=-1.5 * ((s["q"] < self.q_lo) | (s["q"] > self.q_hi)).float().sum(-1) / N21f,
            dof_vel_lim=-6.0 * (qd.abs() > 1.0).float().sum(-1) / N21f,
            dof_tau_lim=-6.0 * (tau.abs() >= 1.0).float().sum(-1) / N21f,
            stand_still=-0.12 * q_rel.abs().sum(-1) / N21f * still.float(),
            single_support=0.2 * (still | (self.single_t <= 0.2)).float(),
            impact=-1.3 * ((vf[..., 2] ** 2) * contact.float()).sum(-1),
            slip=-0.2 * ((vf[..., :2] ** 2).sum(-1) * contact.float()).sum(-1),
            air_time=-2.0 * ((self.air - T_AIR).clamp(min=0.0) * first.float()).sum(-1),   # only air time beyond the target costs
            stumble=-2.0 * (self.F[..., :2].norm(dim=-1) > 3 * self.F[..., 2].abs()).float().sum(-1),
            lateral=0.08 * torch.exp((dy - D_Y_MIN).clamp(max=0.0) / SIG_D),
            air_bonus=self.air_bonus * (self.air.clamp(max=T_AIR) * first.float()).sum(-1) * (~still).float(),   # early training only: gets stepping started
            imitate=self._imitation(q_rel),
        )
        self.v_ema += 0.04 * (v[:, :2] - self.v_ema)
        cn = cmd[:, :2].norm(dim=-1)
        mov = cn > EPS_CMD
        prog = ((self.v_ema * cmd[:, :2]).sum(-1) / cn.clamp(min=1e-3) ** 2).clamp(0.0, 1.0)   # share of the commanded speed actually made good
        self.prog = torch.where(mov, prog, torch.ones_like(prog))
        self.cnt["prog_sum"] = self.cnt.get("prog_sum", 0) + (prog * mov).sum()
        self.cnt["prog_n"] = self.cnt.get("prog_n", 0) + mov.sum()
        if self.prog_gate > 0.0:
            gf = 1.0 - self.prog_gate * (1.0 - self.prog)
            terms.update({k_: terms[k_] * gf for k_ in GATED_TERMS})
        if self.late_scale < 1.0:
            terms = {k_: (v_ if k_ in CORE_TERMS else v_ * self.late_scale) for k_, v_ in terms.items()}
        if self.only_terms is not None:
            terms = {k_: (v_ if k_ in self.only_terms else v_ * 0.0) for k_, v_ in terms.items()}
        r = sum(terms.values())
        self.terms = terms
        # foothold reward (Eq. 3). Support is evaluated on the post-step height patches.
        c, sn = torch.cos(self.fyaw)[..., None], torch.sin(self.fyaw)[..., None]
        sg = self._hf[None, None]
        fx = self.sole[..., 0:1] + c * sg[..., 0] - sn * sg[..., 1]
        fy = self.sole[..., 1:2] + sn * sg[..., 0] + c * sg[..., 1]
        self.h_k = self.hmap(fx, fy)
        rho_c = self.support_stance()
        self.rho_c = rho_c
        rho = contact.float() * rho_c
        if foot_mu is not None:
            guide = ~self.free & self.guide_on
            swing_rho = self.support_imagined(foot_mu, foot_sig)
            self.rho_imag = swing_rho
            rho = rho + (~contact).float() * swing_rho * guide[:, None].float()
        rho = torch.where((self.type == T.SLOPE)[:, None] | self.free[:, None], torch.zeros_like(rho), rho)
        self.rho = rho
        r_f = torch.exp(-rho.sum(-1) ** 2 / SIG_F)
        return r, r_f

    def _done(self, s):
        base = s["p"]
        g_z = self.hmap(base[:, 0], base[:, 1])
        tilt = s["g"][:, 2] > -0.5
        low = base[:, 2] - g_z < 0.3
        hit = self.term_force > 80.0
        pts = self.xpos[:, self.clear_ids]
        scrape = (pts[..., 2] - self.hmap(pts[..., 0], pts[..., 1]) < self.clear_m).any(-1)
        term = tilt | low | hit | scrape | self.bad
        if self.train and self.tiles and self.stuck_term:      # told to walk but not getting anywhere: standing still must not pay
            ck = (self.ep % 100 == 0) & (self.ep > 0) & ~self.free & (self.speed.abs() >= 0.3)
            stuck = ck & ((base[:, :2] - self.stall_pos).norm(dim=-1) < 0.3 * self.speed.abs() * 2.0)
            self.stall_pos = torch.where(ck[:, None], base[:, :2], self.stall_pos)
            term = term | stuck
            self.log["stuck"] = self.log.get("stuck", 0) + int(stuck.sum())
        causes = dict(tilt=tilt, low=low, link_force=hit, clearance=scrape, diverged=self.bad)
        if self.obst is not None:
            causes["side_obstacle"] = self._touches_obstacle()
            term = term | causes["side_obstacle"]
        if self.tiles:
            xrel = base[:, 0] - self.origin[:, 0]
            yrel = base[:, 1] - self.origin[:, 1]
            out_y = (yrel < 0) | (yrel > T.TILE_Y)
            term = term | (out_y & ~self.free)
            fwd = self.speed >= 0
            end = ((xrel > T.TILE_X) & fwd) | ((xrel < 0) & ~fwd)
            back = ((xrel < 0) & fwd) | ((xrel > T.TILE_X) & ~fwd)
            free_out = self.free & (out_y | end | back)
            success = end & ~self.free
            over = self.ep >= int(EPISODE_S / X.CONTROL_DT)
            term = term | (back & ~self.free)
            causes.update(left_tile_side=out_y & ~self.free, left_tile_back=back & ~self.free, stuck=term & ~term if not self.stuck_term else stuck,
                          reached_end=success, omni_left_tile=free_out, time_limit=over & ~success & ~free_out)
            causes["any"] = term | success | free_out | over
            for k_, v_ in causes.items():
                self.cnt["done_" + k_] = self.cnt.get("done_" + k_, 0) + v_.sum()
            return term, success | free_out | over, success
        return term, torch.zeros_like(term), torch.zeros_like(term)

    def curriculum(self, ids, success):
        if len(ids) == 0 or not self.train or not self.tiles or self.flat_only:
            return
        walk = ~self.free[ids]
        wid = ids[walk]
        if len(wid) == 0:
            return
        xrel = self.qpos[wid, 0] - self.origin[wid, 0]
        travel = torch.where(self.speed[wid] >= 0, xrel, T.TILE_X - xrel)
        ep_s = self.ep[wid].float() * X.CONTROL_DT
        need = (self.speed[wid].abs() * ep_s * 0.5).clamp(min=0.5)
        up = success[wid]
        down = ~up & (travel < need) & (ep_s > 3.0)
        lvl = self.level[wid] + up.long() - down.long()
        over = lvl >= T.NLEVEL
        lvl = torch.where(over, torch.randint(0, T.NLEVEL, lvl.shape, device=self.dev, generator=self.gen), lvl).clamp(min=0)
        self.level[wid] = lvl
        self.log["level"] = self.level[~self.free].float().mean().item() if bool((~self.free).any()) else 0.0
        if self.log["level"] > GUIDE_LEVEL:
            self.guide_on = True
