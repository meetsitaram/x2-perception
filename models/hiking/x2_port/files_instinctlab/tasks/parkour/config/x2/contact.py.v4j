"""Option g (v4j, 2026-11-12): wall and prop contact graded by body part (user rule, docs/USER_RULES.md "Wall and prop contact in
training"), plus the v4j logs.

Reward `wall_contact_graded` (replaces the paper's `undesired_contacts`, which charged every non-foot body over 1 N the same -1):
  strong  torso, pelvis, shoulders, head, hips, thighs, knees / shins: (1 + min(|F| / 100 N, 3)) per body in contact (> 1 N),
          so 1 for a touch, up to 4 for a 300 N push.
  arms    elbows, forearms, wrists, hands: free while the force is under ARM_FREE_N (30 N) and the body has been in contact for
          under ARM_FREE_S (0.5 s); beyond either, 0.25 x (1 + min(|F| / 100, 3)).
  feet    not here: the paper's edge term and `feet_stumble` (-2) keep charging feet against walls and riser fronts.
  Weight -1.0 (the paper's weight for the term it replaces).
Termination `torso_fall_contact` (replaces `base_contact`, which ended the episode on any torso force over 1 N): the episode ends
  only when the torso is resting on something (force over 50 N, vertical component larger than horizontal). A torso against a
  wall is charged by the reward above, not ended. Falls stay covered by `root_height` and `bad_orientation`.
Logs (from keep.level_keeper, every call; per tile group as keep.AGROUPS):
  end/<cause>_<group>      share of ended episodes per cause (running count since the start of this process)
  end/reached_<group>      share of ended route episodes whose final station was reached
  c10/<bodygroup>_<group>  share of robots with a force over 10 x body weight on that body group (feet, legs, trunk, arms)
  c50/<bodygroup>_<group>  share of robots with a horizontal force over 50 N on that body group (wall pressing)
  tunnel/<group>           share of robots with a body origin more than 2 cm inside a prop box (every 10th call)
  wallspeed/near_<group>, wallspeed/far_<group>   route robots' planar speed / commanded speed within 0.5 m of a tall prop
                           (top above floor + 0.5 m) and farther than 1.0 m (every 10th call)
"""
from __future__ import annotations

import math
import os

import torch

from isaaclab.managers import SceneEntityCfg

ARM_FREE_N = float(os.environ.get("HIKING_X2_ARM_FREE_N", "30"))
ARM_FREE_S = float(os.environ.get("HIKING_X2_ARM_FREE_S", "0.5"))
ARM_SCALE = float(os.environ.get("HIKING_X2_ARM_SCALE", "0.25"))

ARM_RE = r".*(elbow|wrist|hand|forearm).*"
FEET = ("left_ankle_roll_link", "right_ankle_roll_link")


def body_groups(names):
    """names: contact-sensor body names -> dict group -> list of indices (feet, legs, trunk, arms)."""
    g = {"feet": [], "legs": [], "trunk": [], "arms": []}
    for i, n in enumerate(names):
        if n in FEET:
            g["feet"].append(i)
        elif any(s in n for s in ("elbow", "wrist", "hand", "forearm")):
            g["arms"].append(i)
        elif any(s in n for s in ("hip", "knee", "ankle", "thigh", "shin")):
            g["legs"].append(i)
        else:  # pelvis, waist, torso, shoulders, head
            g["trunk"].append(i)
    return g


def graded_cost(force_w, is_arm, arm_contact_time, free_n=ARM_FREE_N, free_s=ARM_FREE_S, arm_scale=ARM_SCALE):
    """force_w (N, B, 3) net contact forces of the non-foot bodies; is_arm (B,) bool; arm_contact_time (N, B) s.
    -> (N,) cost (positive; the reward weight is negative)."""
    f = torch.norm(force_w, dim=-1)
    touch = f > 1.0
    mag = 1.0 + torch.clamp(f / 100.0, max=3.0)
    arm_paid = touch & ((f > free_n) | (arm_contact_time > free_s))
    c = torch.where(is_arm.unsqueeze(0), torch.where(arm_paid, arm_scale * mag, torch.zeros_like(mag)),
                    torch.where(touch, mag, torch.zeros_like(mag)))
    return c.sum(dim=1)


_st = {}


def wall_contact_graded(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    cs = env.scene.sensors[sensor_cfg.name]
    ids = sensor_cfg.body_ids
    if "is_arm" not in _st:
        names = [cs.body_names[i] for i in ids]
        _st["is_arm"] = torch.tensor([any(s in n for s in ("elbow", "wrist", "hand", "forearm")) for n in names], device=env.device)
        _st["t"] = torch.zeros(env.num_envs, len(ids), device=env.device)
    fw = cs.data.net_forces_w[:, ids]
    touch = torch.norm(fw, dim=-1) > 1.0
    _st["t"] = torch.where(touch, _st["t"] + env.step_dt, torch.zeros_like(_st["t"]))
    _st["t"][env.episode_length_buf == 0] = 0.0
    return graded_cost(fw, _st["is_arm"], _st["t"])


def torso_rest_mask(force_w, min_n=50.0):
    """force_w (N, 3) torso net force -> (N,) bool: resting on something (vertical part dominant and over min_n)."""
    f = torch.norm(force_w, dim=-1)
    return (f > min_n) & (force_w[:, 2].abs() > torch.norm(force_w[:, :2], dim=-1))


def torso_fall_contact(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    cs = env.scene.sensors[sensor_cfg.name]
    hist = cs.data.net_forces_w_history[:, :, sensor_cfg.body_ids[0]]   # (N, H, 3)
    out = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for h in range(hist.shape[1]):
        out |= torso_rest_mask(hist[:, h])
    return out


def apply_v4j(cfg):
    """Called from X2ParkourExtendedFixedEnvCfg.__post_init__ when option g is on."""
    from isaaclab.managers import RewardTermCfg as RewTerm, TerminationTermCfg as DoneTerm
    r = cfg.rewards.rewards if hasattr(cfg.rewards, "rewards") else cfg.rewards
    r.undesired_contacts = None
    r.wall_contact_graded = RewTerm(func=wall_contact_graded, weight=-1.0,
                                    params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="(?!.*_ankle_roll_link).*")})
    cfg.terminations.base_contact = DoneTerm(func=torso_fall_contact,
                                             params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="torso_link")})


# ---------------- boxes of the tile cells (tunnelling and wall distance) ----------------

def quat_to_rot(q):
    """q (..., 4) wxyz -> (..., 3, 3)."""
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1).reshape(q.shape[:-1] + (3, 3))


def build_cell_boxes(calls, npz_of_type, columns_of_type, num_rows, terrain_origins, device="cpu"):
    """-> (cell_index (rows, cols) long, -1 = no tile; centre (C, B, 3); half (C, B, 3); rot (C, B, 3, 3); valid (C, B) bool).
    World coordinates as routes.build_route_table: origin + (tile-centred coordinate - spawn)."""
    cells, num_cols = [], len(terrain_origins[0])
    idx = torch.full((num_rows, num_cols), -1, dtype=torch.long)
    for ttype, lst in calls.items():
        z = npz_of_type[ttype]
        cols = columns_of_type.get(ttype, [])
        for j, (i, spawn) in enumerate(lst):
            col, row = cols[j // num_rows], j % num_rows
            o = terrain_origins[row][col]
            b0, b1 = int(z["box_start"][i]), int(z["box_start"][i + 1])
            c = torch.as_tensor(z["box_center"][b0:b1], dtype=torch.float32) + torch.tensor([o[0] - spawn[0], o[1] - spawn[1], o[2] - spawn[2]])
            idx[row, col] = len(cells)
            cells.append((c, torch.as_tensor(z["box_half"][b0:b1], dtype=torch.float32), torch.as_tensor(z["box_quat"][b0:b1], dtype=torch.float32)))
    B = max([len(c[0]) for c in cells] + [1])
    C = max(len(cells), 1)
    cen = torch.zeros(C, B, 3); half = torch.zeros(C, B, 3); quat = torch.zeros(C, B, 4); quat[..., 0] = 1.0
    valid = torch.zeros(C, B, dtype=torch.bool)
    for k, (c, h, q) in enumerate(cells):
        n = len(c)
        cen[k, :n], half[k, :n], quat[k, :n], valid[k, :n] = c, h, q, True
    return idx.to(device), cen.to(device), half.to(device), quat_to_rot(quat).to(device), valid.to(device)


def inside_depth(points, cen, half, rot, valid, margin=0.02):
    """points (N, P, 3), boxes of each robot's cell (N, B, ...) -> (N,) bool: any point deeper than margin inside any box."""
    local = torch.einsum("nbji,npbj->npbi", rot, points.unsqueeze(2) - cen.unsqueeze(1))   # R^T (p - c)
    ins = (local.abs() < (half.unsqueeze(1) - margin)).all(-1) & valid.unsqueeze(1)
    return ins.any(-1).any(-1)


def wall_distance_xy(p, floor_z, cen, half, rot, valid, min_top=0.5):
    """Planar distance from p (N, 2) to the nearest box whose top is more than min_top above floor_z (N,): tall props only.
    Uses each box's yaw (its x axis projected on the floor) and its xy half sizes. -> (N,)"""
    top = cen[..., 2] + (rot[..., 2, :].abs() * half).sum(-1)       # (N, B) highest point of the box
    tall = valid & (top > floor_z.unsqueeze(1) + min_top) & (cen[..., 2] - (rot[..., 2, :].abs() * half).sum(-1) < floor_z.unsqueeze(1) + 1.6)
    ax = rot[..., :2, 0]                                             # box x axis in xy
    ax = ax / torch.clamp(torch.norm(ax, dim=-1, keepdim=True), min=1e-6)
    ay = torch.stack([-ax[..., 1], ax[..., 0]], -1)
    hx = (rot[..., :2, :].transpose(-1, -2) @ ax.unsqueeze(-1)).squeeze(-1).abs().mul(half).sum(-1)   # extent along ax
    hy = (rot[..., :2, :].transpose(-1, -2) @ ay.unsqueeze(-1)).squeeze(-1).abs().mul(half).sum(-1)
    d = p.unsqueeze(1) - cen[..., :2]
    lx = (d * ax).sum(-1).abs() - hx
    ly = (d * ay).sum(-1).abs() - hy
    dist = torch.norm(torch.stack([lx.clamp(min=0), ly.clamp(min=0)], -1), dim=-1) + torch.minimum(torch.maximum(lx, ly), torch.zeros_like(lx))
    dist = torch.where(tall, dist, torch.full_like(dist, 1e3))
    return dist.min(dim=1)[0]


# ---------------- logging, called from keep.level_keeper ----------------

CAUSES = ("time_out", "base_contact", "bad_orientation", "root_height", "terrain_out_bound")
LOG_BODIES = ("pelvis", "torso_link", "left_knee_link", "right_knee_link", "left_elbow_link", "right_elbow_link",
              "left_wrist_roll_link", "right_wrist_roll_link", "left_ankle_roll_link", "right_ankle_roll_link")


def log_v4j(env, env_ids, out, k, groups, grp_of_robot):
    """groups: list of group names; grp_of_robot (N,) long index into groups. Writes into out (the curriculum dict)."""
    cs = env.scene.sensors["contact_forces"]
    robot = env.scene["robot"]
    n_g = len(groups)
    if "v4j" not in k:
        k["v4j"] = dict(ends=torch.zeros(n_g, len(CAUSES), device=env.device), rends=torch.zeros(n_g, 2, device=env.device),
                        bg=body_groups(cs.body_names), calls=0,
                        weight=float(robot.data.default_mass[0].sum()) * 9.81)
        names = [n for n in LOG_BODIES if n in robot.body_names]
        k["v4j"]["log_body_ids"] = robot.find_bodies(names, preserve_order=True)[0]
    s = k["v4j"]
    s["calls"] += 1
    # episode ends of the robots being reset now, by cause and tile group
    if env_ids is not None and len(env_ids) > 0:
        ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
        g = grp_of_robot[ids]
        tm = env.termination_manager
        for ci, c in enumerate(CAUSES):
            if c in tm.active_terms:
                hit = tm.get_term(c)[ids].float()
                s["ends"][:, ci] += torch.zeros(n_g, device=env.device).index_add_(0, g, hit)
        cmd = env.command_manager.get_term("base_velocity")
        if hasattr(cmd, "on_route"):
            o = cmd.on_route[ids]
            s["rends"][:, 0] += torch.zeros(n_g, device=env.device).index_add_(0, g, o.float())
            s["rends"][:, 1] += torch.zeros(n_g, device=env.device).index_add_(0, g, (o & cmd.route_done[ids]).float())
    tot = s["ends"].sum(1).clamp(min=1)
    for gi, gn in enumerate(groups):
        for ci, c in enumerate(CAUSES):
            out["end/%s_%s" % (c, gn)] = (s["ends"][gi, ci] / tot[gi]).item()
        out["end/reached_%s" % gn] = (s["rends"][gi, 1] / s["rends"][gi, 0].clamp(min=1)).item()
    # forces by body group
    fw = cs.data.net_forces_w
    f = torch.norm(fw, dim=-1)
    fh = torch.norm(fw[..., :2], dim=-1)
    for bn, bi in s["bg"].items():
        if not bi:
            continue
        f10 = (f[:, bi] > 10 * s["weight"]).any(1).float()
        h50 = (fh[:, bi] > 50.0).any(1).float()
        cnt = torch.zeros(n_g, device=env.device).index_add_(0, grp_of_robot, torch.ones_like(f10)).clamp(min=1)
        a = torch.zeros(n_g, device=env.device).index_add_(0, grp_of_robot, f10) / cnt
        b = torch.zeros(n_g, device=env.device).index_add_(0, grp_of_robot, h50) / cnt
        for gi, gn in enumerate(groups):
            out["c10/%s_%s" % (bn, gn)] = a[gi].item()
            out["c50/%s_%s" % (bn, gn)] = b[gi].item()
    # tunnelling and speed near walls (tile cells only), every 10th call
    if s["calls"] % 10 == 1:
        try:
            if "boxes" not in s:
                s["boxes"] = _cell_boxes(env)
            idx, cen, half, rot, valid = s["boxes"]
            t = env.scene.terrain
            cell = idx[t.terrain_levels, t.terrain_types]
            on = (cell >= 0).nonzero().flatten()
            res_t = torch.zeros(env.num_envs, device=env.device)
            near_v = torch.full((env.num_envs,), float("nan"), device=env.device)
            far_v = torch.full((env.num_envs,), float("nan"), device=env.device)
            for chunk in on.split(1024):
                c = cell[chunk]
                pts = robot.data.body_pos_w[chunk][:, s["log_body_ids"]]
                res_t[chunk] = inside_depth(pts, cen[c], half[c], rot[c], valid[c]).float()
                cmd = env.command_manager.get_term("base_velocity")
                if hasattr(cmd, "on_route"):
                    p = robot.data.root_pos_w[chunk]
                    d = wall_distance_xy(p[:, :2], t.env_origins[chunk, 2], cen[c], half[c], rot[c], valid[c])
                    cv = torch.norm(env.command_manager.get_command("base_velocity")[chunk, :2], dim=1)
                    v = torch.norm(robot.data.root_lin_vel_b[chunk, :2], dim=1)
                    ok = cmd.on_route[chunk] & (cv > 0.1)
                    ratio = v / cv.clamp(min=0.1)
                    near_v[chunk] = torch.where(ok & (d < 0.5), ratio, torch.full_like(ratio, float("nan")))
                    far_v[chunk] = torch.where(ok & (d > 1.0), ratio, torch.full_like(ratio, float("nan")))
            for gi, gn in enumerate(groups):
                sel = (grp_of_robot == gi) & (cell >= 0)
                if sel.any():
                    s["tun_" + gn] = res_t[sel].mean().item()
                    nv, fv = near_v[sel], far_v[sel]
                    if (~nv.isnan()).any():
                        s["near_" + gn] = nv[~nv.isnan()].mean().item()
                    if (~fv.isnan()).any():
                        s["far_" + gn] = fv[~fv.isnan()].mean().item()
        except Exception as e:
            if not s.get("warned"):
                s["warned"] = True
                print("[X2 v4j] tunnel / wall-speed log off: %r" % (e,), flush=True)
    for gn in groups:
        if gn.startswith("ext_"):
            out["tunnel/" + gn] = s.get("tun_" + gn, 0.0)
            out["wallspeed/near_" + gn] = s.get("near_" + gn, 0.0)
            out["wallspeed/far_" + gn] = s.get("far_" + gn, 0.0)


def _cell_boxes(env):
    from . import extended as ext
    from .monitor import _terrain_names
    names, col_type = _terrain_names(env)
    type_of_name = {nm: v[0] for nm, v in ext.EXTENDED_COLUMNS.items()}
    cols = {}
    for c, kk in enumerate(col_type.tolist()):
        tt = type_of_name.get(names[kk])
        if tt is not None:
            cols.setdefault(tt, []).append(c)
    npz = {tt: ext._load(tt, ext.EXTENDED_DIR)[0] for tt in ext._CALLS}
    t = env.scene.terrain
    gen = t.cfg.terrain_generator
    return build_cell_boxes(ext._CALLS, npz, cols, gen.num_rows, t.terrain_origins.tolist(), device=env.device)
