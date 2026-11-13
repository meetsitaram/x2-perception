"""Clip sets, terrain and reference-derived contact labels for ACP stage 1 (arXiv 2610.31608) on the X2.

Only kinematic clips are used; nothing here reads a SuperMimic checkpoint, rollout or label file. Contact labels are
derived from the reference kinematics and the clip's own terrain (the paper's oracle contacts): a body is in contact
when its contact point is within Z of the terrain under it and slower than V_CONTACT; runs shorter than MIN_RUN frames
are absorbed; the target of a frame is the next contact ONSET within HORIZON, at the terrain point under the body's
contact point once settled.
"""
from __future__ import annotations

import json
import os

import joblib
import numpy as np
import torch

from gear_sonic.supermimic.assets import build_x2
from gear_sonic.supermimic.kinematics import X2Kinematics, mv3
from gear_sonic.supermimic.motion import MotionLib
from gear_sonic.supermimic.terrain import Terrain

BODIES = ("left_foot", "right_foot", "left_knee", "right_knee", "left_hand", "right_hand", "left_elbow", "right_elbow", "pelvis")
LINKS = ("left_ankle_roll_link", "right_ankle_roll_link", "left_knee_link", "right_knee_link", "left_wrist_roll_link",
         "right_wrist_roll_link", "left_elbow_link", "right_elbow_link", "pelvis")
# contact point of each body in its link frame: sole centre for the feet, link origin otherwise
SOLE_Z = -0.073
POINT = ((0.038, 0.0, SOLE_Z), (0.038, 0.0, SOLE_Z)) + ((0.0, 0.0, 0.0),) * 7
SOLE_CORNERS = ((-0.072, -0.065, SOLE_Z), (-0.072, 0.065, SOLE_Z), (0.148, -0.065, SOLE_Z), (0.148, 0.065, SOLE_Z))
RADIUS = (0.0, 0.0, 0.06, 0.06, 0.05, 0.05, 0.05, 0.05, 0.10)  # m: body surface below the contact point (non-feet)
Z_FOOT, Z_TOUCH, V_CONTACT, MIN_RUN = 0.04, 0.02, 0.5, 4
HORIZON_STEPS, SETTLE = 75, 25                                  # 1.5 s at 50 Hz; settle window after touchdown
SEG_S, SEG_MAX_S = 10.0, 12.0                                   # clips longer than SEG_MAX_S are cut into ~SEG_S pieces


def load_sets(cfg_path: str):
    """cfg: {"dir": ..., "hmap": file or "", "sets": [{"pkl", "props" (or ""), "weight", "shift": [x, y] (optional)}]}.
    Returns clips {name: clip}, props {name: boxes}, set_of {name: set index}, cfg."""
    cfg = json.load(open(cfg_path))
    d = cfg["dir"]
    clips, props, set_of = {}, {}, {}
    for si, s in enumerate(cfg["sets"]):
        data = joblib.load(os.path.join(d, s["pkl"]))
        pr = json.load(open(os.path.join(d, s["props"]))) if s.get("props") else {}
        sh = np.array(list(s.get("shift", [0.0, 0.0])) + [0.0])
        dropped = 0
        for name, c in data.items():
            # style filter (data set 2e, 2026-11-12): drop clips whose name contains one of the words (case-insensitive)
            if s.get("exclude_name") and any(w in name.lower() for w in s["exclude_name"]):
                dropped += 1
                continue
            if name in s.get("exclude_clips", ()):
                dropped += 1
                continue
            # clip-quality filters (cfg, per set): a leg solved on the flipped branch (|hip yaw| > 2 rad; MJCF joints 2 and 8),
            # or a single step higher than exclude_step_over metres in the clip's props (floor -> box or box -> box)
            if s.get("exclude_leg_flip") and float(np.abs(np.asarray(c["dof"])[:, [2, 8]]).max()) > 2.0:
                dropped += 1
                continue
            if s.get("exclude_step_over") and name in pr:
                tops = sorted({0.0} | {round(b_["center"][2] + b_["half"][2], 3) for b_ in pr[name] if b_["center"][2] + b_["half"][2] < 1.5})
                if any(b2 - b1 > float(s["exclude_step_over"]) for b1, b2 in zip(tops[:-1], tops[1:])):
                    dropped += 1
                    continue
            fps = float(c["fps"])
            T = len(c["dof"])
            dur = (T - 1) / fps
            k = 1 if dur <= SEG_MAX_S else int(np.ceil(dur / SEG_S))
            edges = np.linspace(0, T - 1, k + 1).round().astype(int)
            for i in range(k):
                a, b = edges[i], edges[i + 1] + 1
                nm = f"s{si}/{name}" + (f"#{i}" if k > 1 else "")
                if nm in s.get("exclude_clips", ()):          # data set 2f: single segments, by their loader name
                    dropped += 1
                    continue
                clips[nm] = dict(root_trans_offset=np.asarray(c["root_trans_offset"])[a:b] + sh,
                                 root_rot=np.asarray(c["root_rot"])[a:b], dof=np.asarray(c["dof"])[a:b], fps=fps)
                if name in pr:
                    boxes = pr[name]
                    if sh.any():
                        boxes = [dict(b_, center=[b_["center"][0] + sh[0], b_["center"][1] + sh[1], b_["center"][2]]) for b_ in boxes]
                    props[nm] = boxes
                set_of[nm] = si
        if dropped:
            print(f"[data] {s['pkl']}: {dropped} of {len(data)} clips left out by the quality filters", flush=True)
    return clips, props, set_of, cfg


class Dataset:
    def __init__(self, cfg_path: str, device, cache: str = ""):
        clips, props, set_of, cfg = load_sets(cfg_path)
        self.cfg = cfg
        dev = self.device = torch.device(device)
        plant = os.environ.get("ACP_PLANT") or None      # None = the active plant (vendor_20260823); e.g. ACP_PLANT=mc_20261024
        base = build_x2(plant, foot_terrain=True)
        self.kin = X2Kinematics(base.model, dev)
        self.motion = m = MotionLib(clips, self.kin, dev)
        hmap = os.path.join(cfg["dir"], cfg["hmap"]) if cfg.get("hmap") else ""
        self.terrain = Terrain(m.names, props, dev, hmap)
        self.x2 = build_x2(plant, prop_slots=self.terrain.slot_half or None, raster=self.terrain.raster, foot_terrain=True)
        self.link_idx = self.kin.body_index(list(LINKS))
        self.point = torch.tensor(POINT, device=dev)
        self.corners = torch.tensor(SOLE_CORNERS, device=dev)
        self.radius = torch.tensor(RADIUS, device=dev)
        # clip sampling: each set gets its weight, shared between its clips by duration
        w = torch.zeros(len(m.names), device=dev)
        sid = torch.tensor([set_of[n] for n in m.names], device=dev)
        for si, s in enumerate(cfg["sets"]):
            mask = sid == si
            ln = m.clip_len.float() * mask
            w += float(s["weight"]) * ln / ln.sum().clamp(min=1)
        self.clip_w = w / w.sum()
        self.clip_set = sid
        self.set_names = [s["pkl"].replace(".pkl", "") for s in cfg["sets"]]
        self.grounded = self._ground([si for si, s in enumerate(cfg["sets"]) if s.get("ground_stance")])
        self.armfix = self._armfix(cfg, os.path.dirname(os.path.abspath(cfg_path)))
        if cache and os.path.exists(cache):
            lab = torch.load(cache, map_location=dev)
            assert lab["contact"].shape[0] == m.num_frames, "label cache does not match the clip sets"
        else:
            lab = self._label()
            if cache:
                torch.save(lab, cache)
        self.contact, self.occ, self.next = lab["contact"], lab["occ"], lab["next"]   # (F, 9) bool, (F, 9) bool, (F, 9, 3)

    # ---------------------------------------------------------------- contact points
    def body_points(self, S):
        """Contact points of the nine bodies for a state (p, R, v, w, q, qd): positions (..., 9, 3), velocities, and
        the link rotations (..., 9, 3, 3)."""
        P, Rb, V, W = self.kin.forward(*S)
        i = self.link_idx
        P, Rb, V, W = P[..., i, :], Rb[..., i, :, :], V[..., i, :], W[..., i, :]
        off = mv3(Rb, self.point)
        return P + off, V + torch.cross(W, off, dim=-1), Rb, P

    def _armfix(self, cfg, cfg_dir):
        """Arm clearance data fix (cfg key "armfix": file from tools/arm_clear.py): per-frame offsets of shoulder roll and
        elbow so each arm clears its thigh and the torso by >= 1 cm with the training collision geometry; joint velocities
        follow by central differences within each clip. ACP_NO_ARMFIX=1 skips it (used by arm_clear.py itself)."""
        if not cfg.get("armfix") or os.environ.get("ACP_NO_ARMFIX"):
            return None
        fx = torch.load(os.path.join(cfg_dir, cfg["armfix"]), map_location="cpu")
        m = self.motion
        assert fx["num_frames"] == m.num_frames and list(fx["names"]) == list(m.names), "armfix file does not match the clip sets"
        cols = [self.x2.joint_names.index(n) for n in fx["joints"]]
        dq = fx["dq"].to(self.device, m.q.dtype)
        m.q[:, cols] += dq
        vel = torch.zeros_like(dq)
        vel[1:-1] = (dq[2:] - dq[:-2]) / (2 * 0.02)
        s, e = m.clip_start, m.clip_end
        vel[s] = (dq[torch.minimum(s + 1, e)] - dq[s]) / 0.02
        vel[e] = (dq[e] - dq[torch.maximum(e - 1, s)]) / 0.02
        m.qd[:, cols] += vel
        print(f"[data] armfix: {cfg['armfix']} applied ({int((dq.abs().amax(-1) > 1e-4).sum())} of {m.num_frames} frames changed, "
              f"largest {float(dq.abs().max()):.2f} rad)", flush=True)
        return cfg["armfix"]

    def _ground(self, sets):
        """Data-build fix for clips whose stance soles sit inside their terrain (cfg flag "ground_stance" per set): if the
        median gap of the lowest sole corner of the feet in stance is below -1.5 cm, the whole clip is lifted by that amount.
        Returns (clips lifted, clips checked, largest lift in m)."""
        if not sets:
            return (0, 0, 0.0)
        m, T = self.motion, self.terrain
        link = self.kin.body_index(["left_ankle_roll_link", "right_ankle_roll_link"])
        lifted, checked, most = 0, 0, 0.0
        for c in torch.isin(self.clip_set, torch.tensor(sets, device=self.device)).nonzero().squeeze(-1).tolist():
            a, b = int(m.clip_start[c]), int(m.clip_end[c]) + 1
            f = torch.arange(a, b, device=self.device)
            P, Rb, V, W = self.kin.forward(*m.ref_state(f))
            P, Rb, V, W = P[:, link], Rb[:, link], V[:, link], W[:, link]
            cp = P[:, :, None, :] + mv3(Rb[:, :, None, :, :], self.corners)
            h = T.height(cp[..., 0].reshape(len(f), 8), cp[..., 1].reshape(len(f), 8), m.clip_id[f]).reshape(len(f), 2, 4)
            gap = (cp[..., 2] - h).amin(-1)
            off = mv3(Rb, self.point[:2])
            st = ((V + torch.cross(W, off, dim=-1)).norm(dim=-1) < V_CONTACT) & (gap.abs() < 0.08)
            checked += 1
            if int(st.sum()) < 10:
                continue
            md = float(gap[st].median())
            if md < -0.015:
                m.root_p[a:b, 2] -= md
                lifted += 1; most = max(most, -md)
        print(f"[data] ground_stance: {lifted} of {checked} clips lifted (largest {100 * most:.1f} cm)", flush=True)
        return (lifted, checked, most)

    def _label(self):
        m, T = self.motion, self.terrain
        F = m.num_frames
        gap = torch.zeros(F, 9, device=self.device)
        spd = torch.zeros(F, 9, device=self.device)
        pts = torch.zeros(F, 9, 3, device=self.device)
        gnd = torch.zeros(F, 9, device=self.device)
        for a in range(0, F, 20000):
            f = torch.arange(a, min(a + 20000, F), device=self.device)
            p, v, Rb, origin = self.body_points(m.ref_state(f))
            clip = m.clip_id[f]
            h = T.height(p[..., 0], p[..., 1], clip)
            g = p[..., 2] - h - self.radius
            # feet: the lowest sole corner over the terrain under it (a toe-down foot touches with its toe)
            c = origin[:, :2, None, :] + mv3(Rb[:, :2, None, :, :], self.corners)          # (n, 2, 4, 3)
            hc = T.height(c[..., 0].reshape(len(f), 8), c[..., 1].reshape(len(f), 8), clip).reshape(len(f), 2, 4)
            g[:, :2] = (c[..., 2] - hc).min(-1).values
            gap[f], spd[f], pts[f], gnd[f] = g, v.norm(dim=-1), p, h
        zmax = torch.tensor([Z_FOOT, Z_FOOT] + [Z_TOUCH] * 7, device=self.device)
        raw = ((gap < zmax) & (spd < V_CONTACT)).cpu().numpy()
        gap_n, pts_n, gnd_n = gap.cpu().numpy(), pts.cpu().numpy(), gnd.cpu().numpy()
        contact = np.zeros((F, 9), bool)
        occ = np.zeros((F, 9), bool)
        nxt = np.zeros((F, 9, 3), np.float32)
        starts, lens = m.clip_start.cpu().numpy(), m.clip_len.cpu().numpy()
        for a, n in zip(starts, lens):
            for b in range(9):
                c = raw[a:a + n, b].copy()
                # absorb short runs of either state
                edges = np.flatnonzero(np.diff(c.astype(np.int8))) + 1
                bounds = np.concatenate(([0], edges, [n]))
                for i in range(len(bounds) - 1):
                    s, e = bounds[i], bounds[i + 1]
                    if e - s < MIN_RUN and s > 0 and e < n:
                        c[s:e] = c[s - 1]
                contact[a:a + n, b] = c
                on = np.flatnonzero(c[1:] & ~c[:-1]) + 1                      # onsets
                if len(on) == 0:
                    continue
                ends = np.concatenate((np.flatnonzero(~c[1:] & c[:-1]) + 1, [n]))
                tgt = np.zeros((len(on), 3), np.float32)
                for k, o in enumerate(on):
                    e = ends[ends > o][0]
                    w = slice(a + o, a + min(o + SETTLE, e))
                    j = a + o + int(np.argmin(gap_n[w, b]))
                    tgt[k] = (pts_n[j, b, 0], pts_n[j, b, 1], gnd_n[j, b])
                t = np.arange(n)
                k = np.searchsorted(on, t, side="right")                    # first onset strictly after t
                ok = k < len(on)
                kk = np.minimum(k, len(on) - 1)
                ok &= (on[kk] - t) <= HORIZON_STEPS
                occ[a:a + n, b] = ok
                nxt[a:a + n, b] = np.where(ok[:, None], tgt[kk], 0.0)
        dev = self.device
        return dict(contact=torch.as_tensor(contact, device=dev), occ=torch.as_tensor(occ, device=dev),
                    next=torch.as_tensor(nxt, device=dev))
