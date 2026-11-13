"""simcheck.core: the three preflight checks, independent of the track (coordinator's helper, 2026-11-12).

Every check works on the track's own compiled training model (mujoco.MjModel from the track's builder, see adapters/)
and runs it in MuJoCo Warp on the CPU with the model's own options, so it sees what training sees.

  hfield_size(m)                 fail: any height field over HF_MAX x HF_MAX vertices (MuJoCo Warp mis-collides, B10)
  foot_drop(m, foot, raster)     the training foot's collision geoms on a 10 kg free body, copied with the terrain geoms
                                 (exact compiled parameters) into a small model, dropped at DROP_SPEEDS on flat, slope and
                                 edge cells, then pulled sideways with PULL_N. fail: deeper than DROP_DEPTH for more than
                                 DROP_HOLD s in total, or moved less than PULL_FREE_M by the pull (B8, ACP's drop test)
  rollouts(m, tiles, ...)        the whole robot with random PD targets on every tile type. fail: any contact deeper than
                                 ROLL_DEPTH, any body speed jump over ROLL_JUMP m/s between two physics steps, any NaN,
                                 diverged share over ROLL_DIVERGED (B10)
"""
from __future__ import annotations

import time

import mujoco
import numpy as np

HF_MAX = 400
DROP_SPEEDS = (1.0, 2.0, 3.0, 4.5)
DROP_SIDE_V = 1.0                     # m/s sideways at touchdown (as ACP's and SSR's drop tests)
DROP_MASS = 10.0                      # kg on the foot (as ACP's drop test)
DROP_DEPTH, DROP_HOLD = 0.01, 0.07    # m, s
PULL_N, PULL_T0, PULL_T1, PULL_FREE_M = 150.0, 0.10, 0.30, 0.02
DROP_T = 0.8
ROLL_DEPTH, ROLL_JUMP, ROLL_DIVERGED = 0.05, 20.0, 0.01

G = mujoco.mjtGeom


def hfield_size(m, limit: int = HF_MAX) -> dict:
    rows = [(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_HFIELD, i) or f"hf{i}", int(m.hfield_nrow[i]), int(m.hfield_ncol[i]))
            for i in range(m.nhfield)]
    over = [dict(name=n, nrow=r, ncol=c) for n, r, c in rows if r > limit or c > limit]
    return dict(check="hfield_size", passed=not over, limit=limit, n_fields=len(rows),
                max_nrow=max((r for _, r, _ in rows), default=0), max_ncol=max((c for _, _, c in rows), default=0),
                n_over=len(over), over=over[:10])


# ------------------------------------------------------------------ helpers
def _copy_opt(spec, m):
    o, so = m.opt, spec.option
    for k in ("timestep", "impratio", "tolerance", "ls_tolerance", "noslip_tolerance"):
        try:
            setattr(so, k, float(getattr(o, k)))
        except Exception:
            pass
    for k in ("iterations", "ls_iterations", "noslip_iterations", "disableflags", "enableflags"):
        try:
            setattr(so, k, int(getattr(o, k)))
        except Exception:
            pass
    so.integrator, so.cone, so.solver = int(o.integrator), int(o.cone), int(o.solver)
    so.gravity = o.gravity.tolist()


def _contact_params(dst, m, g):
    dst.contype, dst.conaffinity = int(m.geom_contype[g]), int(m.geom_conaffinity[g])
    dst.condim, dst.priority = int(m.geom_condim[g]), int(m.geom_priority[g])
    dst.solref, dst.solimp = m.geom_solref[g].tolist(), m.geom_solimp[g].tolist()
    dst.solmix, dst.friction = float(m.geom_solmix[g]), m.geom_friction[g].tolist()
    dst.margin, dst.gap = float(m.geom_margin[g]), float(m.geom_gap[g])


def _bottom(m, d, g) -> float:
    """Lowest world z of geom g."""
    R = d.geom_xmat[g].reshape(3, 3)
    s, t = m.geom_size[g], m.geom_type[g]
    if t == G.mjGEOM_SPHERE:
        ext = s[0]
    elif t == G.mjGEOM_BOX:
        ext = float(np.abs(R[2]) @ s)
    elif t in (G.mjGEOM_CAPSULE, G.mjGEOM_CYLINDER):
        ext = s[0] + s[1] * abs(R[2, 2])
    elif t == G.mjGEOM_ELLIPSOID:
        ext = float(np.sqrt(((R[2] * s) ** 2).sum()))
    else:
        ext = m.geom_rbound[g]
    return float(d.geom_xpos[g][2] - ext)


def _ground(m, d, xy, exclude_body=-1, top=50.0) -> float:
    """Height of the highest static or mocap surface under xy (ray down)."""
    gid = np.zeros(1, np.int32)
    dist = mujoco.mj_ray(m, d, np.array([xy[0], xy[1], top]), np.array([0.0, 0.0, -1.0]), None, 1, exclude_body, gid)
    return top - dist if dist >= 0 else 0.0


def _sites(raster: dict, n_per_kind: int, seed: int = 0) -> list:
    """(kind, x, y) drop sites on flat, slope and edge cells of the raster (grid[ix, iy], world x = x0 + ix * res).
    Same classes as ACP's drop test: over the foot footprint 0.26 x 0.12 m, flat = height range < 5 mm, slope = 3-15 cm with no
    jump > 2 cm between neighbours, edge = a cell with a > 8 cm jump."""
    grid, res = np.asarray(raster["grid"], np.float32), float(raster["res"])
    x0, y0 = float(raster["x0"]), float(raster["y0"])
    fx, fy = int(0.13 / res) + 1, int(0.06 / res) + 1
    size = (2 * fx + 1, 2 * fy + 1)
    jump = np.maximum(np.abs(np.diff(grid, axis=0, prepend=grid[:1])), np.abs(np.diff(grid, axis=1, prepend=grid[:, :1])))
    try:
        from scipy.ndimage import maximum_filter, minimum_filter
        hi, lo = maximum_filter(grid, size, mode="nearest"), minimum_filter(grid, size, mode="nearest")
        jmax = maximum_filter(jump, size, mode="constant")
    except ImportError:
        from numpy.lib.stride_tricks import sliding_window_view as swv
        pad = np.pad(grid, ((fx, fx), (fy, fy)), mode="edge")
        hi, lo = swv(pad, size).max(axis=(2, 3)), swv(pad, size).min(axis=(2, 3))
        jmax = swv(np.pad(jump, ((fx, fx), (fy, fy))), size).max(axis=(2, 3))
    border = np.zeros_like(grid, bool)
    border[fx * 2:-fx * 2, fy * 2:-fy * 2] = True
    kinds = {"flat": border & (hi - lo < 0.005), "slope": border & (hi - lo > 0.03) & (hi - lo < 0.15) & (jmax < 0.02),
             "edge": border & (jmax > 0.08) & (jump > 0.08)}
    rng, out = np.random.default_rng(seed), []
    nx, ny = grid.shape
    for k, msk in kinds.items():
        idx = np.argwhere(msk)
        for i in (rng.choice(len(idx), size=min(n_per_kind, len(idx)), replace=False) if len(idx) else []):
            ix, iy = idx[i]
            gx = float(grid[min(ix + fx, nx - 1), iy] - grid[max(ix - fx, 0), iy])
            gy = float(grid[ix, min(iy + fy, ny - 1)] - grid[ix, max(iy - fy, 0)])
            n = np.hypot(gx, gy)
            pull = (-gx / n, -gy / n) if n > 0.01 else (1.0, 0.0)    # downhill: away from a riser, off an edge
            out.append((k, x0 + ix * res, y0 + iy * res, pull))
    return out


# ------------------------------------------------------------------ foot drop
def foot_model(m, foot_body: str, mass: float = DROP_MASS):
    """A model with the training model's static world geoms (height fields, planes, boxes; exact compiled contact
    parameters) and one free body carrying the training foot's collision geoms."""
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    spec = mujoco.MjSpec()
    _copy_opt(spec, m)
    hf_done = {}
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != 0 or not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        t = m.geom_type[g]
        kw = {}
        if t == G.mjGEOM_HFIELD:
            h = int(m.geom_dataid[g])
            if h not in hf_done:
                nr, nc, adr = int(m.hfield_nrow[h]), int(m.hfield_ncol[h]), int(m.hfield_adr[h])
                hf = spec.add_hfield(name=f"h{h}", nrow=nr, ncol=nc, size=m.hfield_size[h].tolist())
                hf.userdata = m.hfield_data[adr:adr + nr * nc].astype(np.float32).tolist()
                hf_done[h] = f"h{h}"
            kw["hfieldname"] = hf_done[h]
        elif t not in (G.mjGEOM_PLANE, G.mjGEOM_BOX):
            continue
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, d.geom_xmat[g])
        s = spec.worldbody.add_geom(name=f"t{g}", type=G(int(t)), size=m.geom_size[g].tolist(), pos=d.geom_xpos[g].tolist(),
                                    quat=q.tolist(), **kw)
        _contact_params(s, m, g)
    fb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, foot_body)
    assert fb >= 0, foot_body
    geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == fb and (m.geom_contype[g] or m.geom_conaffinity[g])]
    b = spec.worldbody.add_body(name="foot", pos=[0, 0, 1000.0])
    b.add_freejoint()
    for i, g in enumerate(geoms):
        s = b.add_geom(name=f"f{i}", type=G(int(m.geom_type[g])), size=m.geom_size[g].tolist(), pos=m.geom_pos[g].tolist(),
                       quat=m.geom_quat[g].tolist())
        _contact_params(s, m, g)
        s.mass = mass / len(geoms)
    fm = spec.compile()
    return fm, len(geoms), [int(fm.geom_type[g]) for g in range(fm.ngeom) if fm.geom_bodyid[g] == 1]


def foot_drop(m, foot_body: str, raster: dict, n_per_kind: int = 3, speeds=DROP_SPEEDS, seed: int = 0, log=print) -> dict:
    import torch
    import warp as wp
    import mujoco_warp as mjw
    t0 = time.time()
    fm, nfoot, _ = foot_model(m, foot_body)
    fbody = mujoco.mj_name2id(fm, mujoco.mjtObj.mjOBJ_BODY, "foot")
    d = mujoco.MjData(fm)
    mujoco.mj_forward(fm, d)
    sites = [(k, x, y, _ground(fm, d, (x, y), fbody), p) for k, x, y, p in _sites(raster, n_per_kind, seed)]
    d.qpos[:3] = [0, 0, 0.3]
    mujoco.mj_forward(fm, d)
    foot_g = [g for g in range(fm.ngeom) if fm.geom_bodyid[g] == fbody]
    low = min(_bottom(fm, d, g) for g in foot_g) - 0.3
    cases = [(v, s) for s in sites for v in speeds]
    W, dt = len(cases), fm.opt.timestep
    log(f"[drop] {len(sites)} sites x {len(speeds)} speeds, {nfoot} foot geoms, {fm.nhfield} height fields")
    with wp.ScopedDevice("cpu"):
        mm = mjw.put_model(fm)
        dd = mjw.put_data(fm, d, nworld=W, nconmax=200, njmax=600)
        T = wp.to_torch
        qpos, qvel, xfrc = T(dd.qpos), T(dd.qvel), T(dd.xfrc_applied)
        pull = torch.tensor([[p[0], p[1], 0.0] for v, (k, x, y, h, p) in cases], dtype=xfrc.dtype)
        px = pull[:, :2].numpy()
        for w, (v, (k, x, y, h, p)) in enumerate(cases):
            qpos[w, :3] = torch.tensor([x, y, h - low + 0.002])
            qpos[w, 3:7] = torch.tensor([1.0, 0, 0, 0])
            qvel[w, :] = 0
            qvel[w, 0], qvel[w, 2] = DROP_SIDE_V, -v
        deep, t_over, hold, run = np.zeros(W), np.zeros(W), np.zeros(W), np.zeros(W)
        x_at, x_end, nanw = np.zeros((W, 2)), np.zeros((W, 2)), np.zeros(W, bool)
        fgeom = [None] * W
        for ks in range(int(round(DROP_T / dt))):
            t = ks * dt
            if abs(t - PULL_T0) < dt / 2:
                x_at = qpos[:, :2].numpy().copy()
                xfrc[:, fbody, :3] = PULL_N * pull
            if abs(t - PULL_T1) < dt / 2:
                x_end = qpos[:, :2].numpy().copy()
                xfrc[:, fbody, :3] = 0.0
            mjw.step(mm, dd)
            nanw |= ~np.isfinite(qpos.numpy()).all(1)
            n = min(int(T(dd.nacon)[0]), dd.contact.dist.shape[0])
            dmax = np.zeros(W)
            if n:
                wid, dep = T(dd.contact.worldid)[:n].numpy(), -T(dd.contact.dist)[:n].numpy()
                np.maximum.at(dmax, wid, dep)
                for c in np.nonzero(dep > deep[wid] + 1e-9)[0]:          # new deepest contact: remember the foot geom
                    if dep[c] >= dmax[wid[c]] - 1e-9:
                        gg = T(dd.contact.geom)[c].numpy()
                        g = int(gg[0]) if fm.geom_bodyid[int(gg[0])] == fbody else int(gg[1])
                        fgeom[wid[c]] = f"{G(int(fm.geom_type[g])).name[7:].lower()} r{fm.geom_size[g][0]:.3f}"
            deep = np.maximum(deep, dmax)
            over = dmax > DROP_DEPTH
            t_over += over * dt
            run = np.where(over, run + dt, 0.0)
            hold = np.maximum(hold, run)
    out = []
    for w, (v, (k, x, y, h, p)) in enumerate(cases):
        moved = float((x_end[w] - x_at[w]) @ px[w])           # along the pull
        ok = (t_over[w] <= DROP_HOLD) and (moved >= PULL_FREE_M) and not nanw[w]
        out.append(dict(kind=k, v=v, x=round(x, 2), y=round(y, 2), pull_dir=[round(p[0], 2), round(p[1], 2)], deep_mm=round(1000 * deep[w], 1),
                        t_over_1cm_s=round(float(t_over[w]), 3), longest_hold_s=round(float(hold[w]), 3),
                        pull_moved_m=round(moved, 3), deepest_foot_geom=fgeom[w], nan=bool(nanw[w]), passed=bool(ok)))
    summ = {}
    for k in ("flat", "slope", "edge"):
        r = [o for o in out if o["kind"] == k]
        if r:
            summ[k] = dict(n=len(r), failed=sum(not o["passed"] for o in r), deep_mm_max=max(o["deep_mm"] for o in r),
                           t_over_1cm_max=max(o["t_over_1cm_s"] for o in r), pull_moved_min=min(o["pull_moved_m"] for o in r))
    missing = [k for k in ("flat", "slope", "edge") if k not in summ]
    return dict(check="foot_drop", passed=all(o["passed"] for o in out) and bool(out), limits=dict(
        depth_m=DROP_DEPTH, hold_s=DROP_HOLD, pull_N=PULL_N, freed_m=PULL_FREE_M, mass_kg=DROP_MASS, speeds=list(speeds)),
        kinds_missing=missing, summary=summ, failed_cases=[o for o in out if not o["passed"]][:12], n_cases=len(out),
        seconds=round(time.time() - t0, 1))


# ------------------------------------------------------------------ random-action rollouts
def _gname(m, g):
    return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or         f"{mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or 'world'}#{g}"


def _robot_bodies(m):
    """Bodies of the tree under the first free joint (the robot)."""
    fj = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE][0]
    root = int(m.jnt_bodyid[fj])
    return np.array([b for b in range(m.nbody) if m.body_rootid[b] == root]), fj


def spawn_qpos(m, q_joints, xy, yaw: float = 0.0, mocap=None, lift: float = 0.02) -> np.ndarray:
    """qpos with the robot's lowest collision geom `lift` above the ground under xy (ray against the static and mocap
    geoms; mocap = (pos, quat) arrays of the tile's props)."""
    bodies, fj = _robot_bodies(m)
    a = int(m.jnt_qposadr[fj])
    d = mujoco.MjData(m)
    if mocap is not None:
        d.mocap_pos[:], d.mocap_quat[:] = mocap
    d.qpos[a:a + 3] = [xy[0], xy[1], 1000.0]
    d.qpos[a + 3:a + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    d.qpos[a + 7:a + 7 + len(q_joints)] = q_joints
    mujoco.mj_forward(m, d)
    rg = [g for g in range(m.ngeom) if np.isin(m.geom_bodyid[g], bodies) and (m.geom_contype[g] or m.geom_conaffinity[g])]
    low = min(_bottom(m, d, g) for g in rg) - 1000.0
    # ground: highest hit over a 0.3 m square around xy
    h = max(_ground(m, d, (xy[0] + dx, xy[1] + dy)) for dx in (-0.15, 0, 0.15) for dy in (-0.15, 0, 0.15))
    q = d.qpos.copy()
    q[a + 2] = h - low + lift
    return q


def rollouts(m, tiles: list, q_default, k: int = 4, seconds: float = 2.0, noise: float = 0.5, hold: float = 0.1,
             diverge_qvel: float = 200.0, nconmax: int = 64, njmax: int = 450, seed: int = 0, log=print) -> dict:
    """tiles: [dict(name, qpos (nq) or xy/yaw, mocap=(pos, quat) or None)]. k worlds per tile, PD targets = the tile's
    start pose + U(-noise, noise) rad, new every `hold` s, clipped to the joint ranges."""
    import torch
    import warp as wp
    import mujoco_warp as mjw
    t0 = time.time()
    bodies, fj = _robot_bodies(m)
    a0 = int(m.jnt_qposadr[fj])
    nj = m.nq - 7
    q_default = np.asarray(q_default, np.float64)
    act_q = np.array([int(m.jnt_qposadr[m.actuator_trnid[i, 0]]) for i in range(m.nu)])   # qpos index each actuator targets
    rng_lo = np.array([m.jnt_range[m.actuator_trnid[i, 0], 0] for i in range(m.nu)])
    rng_hi = np.array([m.jnt_range[m.actuator_trnid[i, 0], 1] for i in range(m.nu)])
    starts, mocaps = [], []
    for t in tiles:
        q = np.asarray(t["qpos"], np.float64) if t.get("qpos") is not None else \
            spawn_qpos(m, q_default, t["xy"], t.get("yaw", 0.0), t.get("mocap"))
        starts.append(q)
        mocaps.append(t.get("mocap"))
    W = len(tiles) * k
    tile_of = np.repeat(np.arange(len(tiles)), k)
    robot_body = np.zeros(m.nbody, bool)
    robot_body[bodies] = True
    geom_robot = torch.as_tensor(robot_body[m.geom_bodyid])
    rb = torch.as_tensor(bodies[bodies != 0])
    d = mujoco.MjData(m)
    d.qpos[:] = starts[0]
    mujoco.mj_forward(m, d)
    dt = m.opt.timestep
    nsteps, nhold = int(round(seconds / dt)), max(1, int(round(hold / dt)))
    gen = np.random.default_rng(seed)
    log(f"[roll] {len(tiles)} tiles x {k} worlds, {nsteps} steps, nbody {m.nbody}, ngeom {m.ngeom}, nhfield {m.nhfield}")
    with wp.ScopedDevice("cpu"):
        mm = mjw.put_model(m)
        dd = mjw.put_data(m, d, nworld=W, nconmax=nconmax, njmax=njmax)
        T = wp.to_torch
        qpos, qvel, ctrl = T(dd.qpos), T(dd.qvel), T(dd.ctrl)
        Q0 = torch.as_tensor(np.stack([starts[i] for i in tile_of]), dtype=qpos.dtype)
        qpos[:] = Q0
        qvel[:] = 0
        if m.nmocap:
            mp, mq = T(dd.mocap_pos), T(dd.mocap_quat)
            for w in range(W):
                mc = mocaps[tile_of[w]]
                if mc is not None:
                    mp[w] = torch.as_tensor(mc[0], dtype=mp.dtype)
                    mq[w] = torch.as_tensor(mc[1], dtype=mq.dtype)
        base_t = Q0[:, act_q].numpy()
        dep_t, dep_s = np.zeros(W), np.zeros(W)
        jump, vmax = np.zeros(W), np.zeros(W)
        nan_w, div_w = np.zeros(W, bool), np.zeros(W, bool)
        div_events = 0
        pair = [None] * W
        xprev = vprev = None
        overflow = 0
        for ks in range(nsteps):
            if ks % nhold == 0:
                tgt = np.clip(base_t + gen.uniform(-noise, noise, base_t.shape), rng_lo, rng_hi)
                ctrl[:] = torch.as_tensor(tgt, dtype=ctrl.dtype)
            mjw.step(mm, dd)
            fin = torch.isfinite(qpos).all(1) & torch.isfinite(qvel).all(1)
            bad = ~fin | (qvel.abs().amax(1) > diverge_qvel)
            nan_w |= (~fin).numpy()
            xp = T(dd.xpos)[:, rb].clone()
            if xprev is not None:
                v = (xp - xprev) / dt
                sp = v.norm(dim=-1).amax(1)
                vmax = np.maximum(vmax, torch.nan_to_num(sp, 1e9).numpy())
                if vprev is not None:
                    jmp = torch.nan_to_num((v - vprev).norm(dim=-1).amax(1), 1e9)
                    jump = np.maximum(jump, jmp.numpy())
                vprev = v
            xprev = xp
            n = int(T(dd.nacon)[0])
            cap = dd.contact.dist.shape[0]
            overflow = max(overflow, n - cap)
            n = min(n, cap)
            if n:
                wid = T(dd.contact.worldid)[:n].long().numpy()
                dep = torch.nan_to_num(-T(dd.contact.dist)[:n], 1e9).numpy()
                gg = T(dd.contact.geom)[:n].long()
                selfc = (geom_robot[gg[:, 0]] & geom_robot[gg[:, 1]]).numpy()
                old = np.maximum(dep_t, dep_s)
                np.maximum.at(dep_t, wid[~selfc], dep[~selfc])
                np.maximum.at(dep_s, wid[selfc], dep[selfc])
                for c in np.nonzero(dep > old[wid] + 1e-9)[0]:          # new deepest contact of its world: remember the pair
                    if dep[c] >= np.maximum(dep_t, dep_s)[wid[c]] - 1e-9:
                        g1, g2 = int(gg[c, 0]), int(gg[c, 1])
                        pair[wid[c]] = (f"{_gname(m, g1)}|{_gname(m, g2)}", round(float(ks * dt), 3))
            if bool(bad.any()):                        # as training does: count, put the world back to its start
                ids = bad.nonzero().squeeze(-1)
                div_events += len(ids)
                div_w[ids.numpy()] = True
                qpos[ids] = Q0[ids]
                qvel[ids] = 0
                xprev = vprev = None                  # the reset moves bodies; skip the next difference
    per = []
    for i, t in enumerate(tiles):
        s = tile_of == i
        r = dict(tile=t["name"], depth_terrain_mm=round(1000 * dep_t[s].max(), 1), depth_self_mm=round(1000 * dep_s[s].max(), 1),
                 speed_jump_ms=round(float(jump[s].max()), 2), body_speed_max_ms=round(float(vmax[s].max()), 2),
                 nan_worlds=int(nan_w[s].sum()), diverged_worlds=int(div_w[s].sum()))
        wd = np.nonzero(s)[0][int(np.argmax(np.maximum(dep_t, dep_s)[s]))]
        r["deepest_pair_t"] = pair[wd]
        r["passed"] = bool(max(dep_t[s].max(), dep_s[s].max()) <= ROLL_DEPTH and jump[s].max() <= ROLL_JUMP and not nan_w[s].any()
                       and div_w[s].mean() <= ROLL_DIVERGED)
        per.append(r)
    share = float(div_w.mean())
    passed = all(r["passed"] for r in per) and share <= ROLL_DIVERGED and not nan_w.any()
    return dict(check="rollouts", passed=bool(passed), limits=dict(depth_m=ROLL_DEPTH, speed_jump_ms=ROLL_JUMP,
                diverged_share=ROLL_DIVERGED, diverge_qvel=diverge_qvel, seconds=seconds, noise_rad=noise, worlds_per_tile=k),
                worlds=W, diverged_share=round(share, 4), diverge_events=int(div_events), nan_worlds=int(nan_w.sum()),
                depth_terrain_mm_max=round(1000 * dep_t.max(), 1), depth_self_mm_max=round(1000 * dep_s.max(), 1),
                speed_jump_ms_max=round(float(jump.max()), 2), contact_overflow=int(max(overflow, 0)),
                failed_tiles=[r for r in per if not r["passed"]], tiles=per, seconds=round(time.time() - t0, 1))
