"""Extended terrain map in SSR training (user decision 2026-11-12). Flags: --ext_frac F --ext_types a,b,c --ext_variants K.

Tiles of /mnt/filesystem-g4/data/terrains/extended_v1 (8 x 8 m, 2 cm) are resampled to the SSR raster (2.5 cm) and placed to
the right of the paper-style tiles: column = (type, variant), row = level (8 m rows; 10 levels = the 80 m of the existing grid).
Physics floor = the export's floor raster plus low lifted pieces (top within 0.30 m of the floor, not tall: a slanted 4x4 becomes
solid); vision = floor + 1.3 m on the export's obstacle_mask (tall walls, rails, posts; door headers are not in the mask, so doors
stay open in depth). Tall things are not MuJoCo bodies: as on the paper-style tiles, touching one (limb centre on an obstacle
cell) ends the episode (with --side_soft the 0.06 m band is a penalty).
A share F of tile robots (after the flat phase) are route robots: a route of their tile, start pose = the route start, the
harness's command law towards each waypoint (yaw 1.5 x error capped 1.0, no forward speed over 60 deg, forward
clip(distance, 0.15, vmax) x cos(error), vmax U(0.4, 1.0)), next waypoint inside 0.25 m, at the target zero command and success
after 1 s inside 0.20 m. Curriculum per robot over the 10 levels of its route type.
Backups ../_bak_20261112/*.pre_ext
"""
import pathlib
import shutil

C = pathlib.Path(__file__).resolve().parents[1]


def edit(rel, pairs):
    p = C / rel
    s = p.read_text()
    shutil.copy(p, C.parent / "_bak_20261112" / (p.name + ".pre_ext"))
    for old, new in pairs:
        assert s.count(old) == 1, (rel, old)
        s = s.replace(old, new)
    p.write_text(s)


TERRAIN_ADD = '''

EXT_DIR = "/mnt/filesystem-g4/data/terrains/extended_v1"


def add_extended(r: dict, types=("tight_spots", "log_corridors", "homes", "street_front", "stairs"), variants=2, gap=2.0) -> dict:
    """Append extended_v1 tiles to a raster made by generate(): see scripts/patch_ext.py. Returns r with r["ext"] metadata."""
    import json
    z, v = r["z"], r["vision"]["z"]
    x_base = len(TYPES) * TILE_X + gap                                # local x where the extended block starts (raster x0 = -BORDER)
    ncol = len(types) * variants
    nx_new = _i(BORDER) * 2 + _i(x_base + ncol * 8.0) + 1
    if nx_new > z.shape[0]:
        pad = nx_new - z.shape[0]
        z = np.concatenate((z, np.zeros((pad, z.shape[1]), np.float32)), 0)
        v = np.concatenate((v, np.zeros((pad, v.shape[1]), np.float32)), 0)
    tiles, routes = [], []
    for ti, t in enumerate(types):
        dat = np.load(f"{EXT_DIR}/{t}.npz")
        meta = json.load(open(f"{EXT_DIR}/{t}.json"))["tiles"]
        for lvl in range(10):
            cand = [m for m in meta if m["level"] == lvl and m["routes"]]
            for k in range(variants):
                if not cand:
                    continue
                m = cand[k % len(cand)]
                col, row = ti * variants + k, lvl
                cx, cy = x_base + col * 8.0 + 4.0, row * 8.0 + 4.0          # tile centre in local raster coordinates
                h, top, mask = dat["height"][m["index"]], dat["height_top"][m["index"]], dat["obstacle_mask"][m["index"]]
                floor = np.where(~mask & (top - h <= 0.30), top, h).astype(np.float32)
                vis = (floor + 1.3 * mask).astype(np.float32)
                n8 = _i(8.0)
                gx = (np.arange(n8) + 0.5) * RES - 4.0                      # SSR cell centres in tile coordinates
                ii = np.clip(np.round((gx + 3.99) / 0.02).astype(int), 0, 399)
                i0, j0 = _i(BORDER + cx - 4.0), _i(BORDER + cy - 4.0)
                z[i0:i0 + n8, j0:j0 + n8] = floor[np.ix_(ii, ii)]
                v[i0:i0 + n8, j0:j0 + n8] = vis[np.ix_(ii, ii)]
                gcx, gcy = cx - BORDER, cy - BORDER                         # tile centre in world coordinates (x0 = y0 = -BORDER)
                tid = len(tiles)
                tiles.append(dict(type=ti, name=t, level=lvl, tile_id=m["tile_id"], centre=(gcx, gcy)))
                for ro in m["routes"]:
                    wp = [(gcx + a, gcy + b) for a, b in ro["waypoints"]]
                    routes.append(dict(tile=tid, start=(gcx + ro["start"][0], gcy + ro["start"][1], ro["start"][2]), wp=wp, length=ro["length_m"]))
    r["z"], r["vision"]["z"] = z, v
    r["ext"] = dict(types=list(types), variants=variants, tiles=tiles, routes=routes)
    return r
'''

ENV_EXT_METHODS = '''    # ---------------------------------------------------------------- extended map (route robots)
    def _ext_setup(self):
        e = self.raster.get("ext")
        self.ext_on = e is not None
        n, dev = self.N, self.dev
        self.ext = torch.zeros(n, dtype=torch.bool, device=dev)
        self.ext_type = torch.zeros(n, dtype=torch.long, device=dev)
        self.ext_level = torch.zeros(n, dtype=torch.long, device=dev)
        self.route = torch.zeros(n, dtype=torch.long, device=dev)
        self.wpi = torch.zeros(n, dtype=torch.long, device=dev)
        self.hold_t = torch.zeros(n, device=dev)
        self.ext_vmax = torch.full((n,), 0.6, device=dev)
        self.ext_frac = 0.0
        if not self.ext_on:
            return
        R = e["routes"]
        W = max(len(r["wp"]) for r in R)
        wp = torch.zeros(len(R), W, 2)
        for i, r in enumerate(R):
            w = torch.tensor(r["wp"], dtype=torch.float32)
            wp[i, :len(w)] = w
            wp[i, len(w):] = w[-1]
        self.r_wp = wp.to(dev)
        self.r_n = torch.tensor([len(r["wp"]) for r in R], device=dev)
        self.r_start = torch.tensor([r["start"] for r in R], dtype=torch.float32, device=dev)
        self.r_tile = torch.tensor([r["tile"] for r in R], device=dev)
        self.r_len = torch.tensor([r["length"] for r in R], dtype=torch.float32, device=dev)
        tiles = e["tiles"]
        self.t_type = torch.tensor([t["type"] for t in tiles], device=dev)
        self.t_level = torch.tensor([t["level"] for t in tiles], device=dev)
        self.t_centre = torch.tensor([t["centre"] for t in tiles], dtype=torch.float32, device=dev)
        self.ext_ntypes = len(e["types"])
        self.ext_stairs = torch.tensor([nm == "stairs" for nm in e["types"]], device=dev)
        # routes of (type, level): random pick by masking
        self.r_type = self.t_type[self.r_tile]
        self.r_level = self.t_level[self.r_tile]

    def _ext_pick(self, ids):
        """Route robots among ids: a route of (ext_type, ext_level); returns start x, y, yaw."""
        k = len(ids)
        match = (self.r_type[None] == self.ext_type[ids, None]) & (self.r_level[None] == self.ext_level[ids, None])
        w = match.float() + 1e-9 * (self.r_type[None] == self.ext_type[ids, None]).float()
        rid = torch.multinomial(w, 1, generator=self.gen).squeeze(-1)
        self.route[ids] = rid
        self.wpi[ids] = 0
        self.hold_t[ids] = 0.0
        self.ext_vmax[ids] = self._u(k, lo=0.4, hi=1.0)
        st = self.r_start[rid]
        x = st[:, 0] + self._u(k) * 0.05
        y = st[:, 1] + self._u(k) * 0.05
        yaw = st[:, 2] + self._u(k) * math.radians(10.0)
        return x, y, yaw

    def _ext_cmd(self):
        """The harness's command law towards the current waypoint (heading frame)."""
        m = self.ext
        if not bool(m.any()):
            return
        s = self.s
        tgt = self.r_wp[self.route, self.wpi]
        vec = tgt - s["p"][:, :2]
        dist = vec.norm(dim=-1)
        err = wrap(torch.atan2(vec[:, 1], vec[:, 0]) - s["yaw"])
        wz = (1.5 * err).clamp(-1.0, 1.0)
        vx = torch.where(err.abs() > math.radians(60.0), torch.zeros_like(dist), dist.clamp(0.15, None).minimum(self.ext_vmax) * torch.cos(err))
        last = self.wpi >= self.r_n[self.route] - 1
        hold = last & (dist < 0.20)
        c = torch.stack((torch.where(hold, torch.zeros_like(vx), vx), torch.zeros_like(vx), torch.where(hold, torch.zeros_like(wz), wz)), -1)
        self.cmd = torch.where(m[:, None], c, self.cmd)

    def _ext_advance(self):
        m = self.ext
        if not bool(m.any()):
            return
        tgt = self.r_wp[self.route, self.wpi]
        dist = (tgt - self.s["p"][:, :2]).norm(dim=-1)
        last = self.wpi >= self.r_n[self.route] - 1
        adv = m & ~last & (dist < 0.25)
        self.wpi = self.wpi + adv.long()
        inside = m & last & (dist < 0.20)
        self.hold_t = torch.where(inside, self.hold_t + X.CONTROL_DT, torch.zeros_like(self.hold_t))

    def ext_stats(self):
        if not self.ext_on or not bool(self.ext.any()):
            return {}
        out = {}
        for t in range(self.ext_ntypes):
            mt = self.ext & (self.ext_type == t)
            if bool(mt.any()):
                out[self.raster["ext"]["types"][t]] = dict(n=int(mt.sum()), level=round(float(self.ext_level[mt].float().mean()), 2))
        out["done"] = {k_[4:]: float(v_) for k_, v_ in self.cnt.items() if k_.startswith("ext_")}
        return out

'''

edit("ssr/terrain.py", [
    ("    vision = dict(z=v, x0=-BORDER, y0=-BORDER, res=RES)\n    return dict(z=z, x0=-BORDER, y0=-BORDER, res=RES, base=0.1, vision=vision, lane=lane)\n",
     "    vision = dict(z=v, x0=-BORDER, y0=-BORDER, res=RES)\n    return dict(z=z, x0=-BORDER, y0=-BORDER, res=RES, base=0.1, vision=vision, lane=lane)\n" + TERRAIN_ADD),
])

edit("ssr/env.py", [
    # setup after the counters exist and before the first reset
    ("        self.cnt = {}                           # raw counts for the log",
     "        self._ext_setup()\n        self.cnt = {}                           # raw counts for the log"),
    # reset: route robots
    ("            if self.flat_only:\n                self.type[ids], self.level[ids] = T.FLAT, 0\n            x, y = self._terrain_reset(ids)\n            self._command_reset(ids)\n            yaw = self.heading[ids]\n            rev = (self.speed[ids] < 0) & ~self.free[ids]             # backward walks on slopes start at the far end\n            x = torch.where(rev, 2 * self.origin[ids, 0] + T.TILE_X - x, x)\n",
     "            if self.flat_only:\n                self.type[ids], self.level[ids] = T.FLAT, 0\n            x, y = self._terrain_reset(ids)\n            self._command_reset(ids)\n            yaw = self.heading[ids]\n            rev = (self.speed[ids] < 0) & ~self.free[ids]             # backward walks on slopes start at the far end\n            x = torch.where(rev, 2 * self.origin[ids, 0] + T.TILE_X - x, x)\n"
     "            if self.ext_on:\n"
     "                self.ext[ids] = False\n"
     "                if self.train and self.ext_frac > 0 and not self.flat_only and self.obst_target is None and self.ep.sum() > 0:\n"
     "                    ex = (self._u(n, lo=0.0, hi=1.0) < self.ext_frac) & ~self.free[ids]\n"
     "                    if bool(ex.any()):\n"
     "                        eid = ids[ex]\n"
     "                        self.ext[eid] = True\n"
     "                        ex_x, ex_y, ex_yaw = self._ext_pick(eid)\n"
     "                        self.speed[eid] = 0.0\n"
     "                        self.heading[eid] = ex_yaw\n"
     "                        self.type[eid] = torch.where(self.ext_stairs[self.ext_type[eid]], torch.full_like(self.type[eid], T.STAIRS), torch.full_like(self.type[eid], T.FLAT))\n"
     "                        x = torch.where(ex, torch.zeros_like(x), x); y = torch.where(ex, torch.zeros_like(y), y); yaw = torch.where(ex, torch.zeros_like(yaw), yaw)\n"
     "                        x[ex], y[ex], yaw[ex] = ex_x, ex_y, ex_yaw\n"),
    # the obstacle-share ramp must not convert route robots
    ("                fl = ids[self.type[ids] == T.FLAT]\n",
     "                fl = ids[(self.type[ids] == T.FLAT) & ~self.ext[ids]]\n"),
    # no walking starts for route robots
    ("        pick = (self._u(n, lo=0.0, hi=1.0) < self.rsi_frac) & ~self.free[ids] & (self.speed[ids] >= 0)\n",
     "        pick = (self._u(n, lo=0.0, hi=1.0) < self.rsi_frac) & ~self.free[ids] & ~self.ext[ids] & (self.speed[ids] >= 0)\n"),
    # no 8 s speed resample for route robots
    ("                tl = rs[~self.free[rs]]\n",
     "                tl = rs[~self.free[rs] & ~self.ext[rs]]\n"),
    # waypoint advance once per step, before the command
    ("        if self.edge_stop_p > 0 and self.train and self.tiles:\n",
     "        if self.ext_on:\n            self._ext_advance()\n        if self.edge_stop_p > 0 and self.train and self.tiles:\n"),
    # command law
    ("        self.cmd = torch.where(self.free[:, None], self.cmd, walk)\n",
     "        self.cmd = torch.where(self.free[:, None], self.cmd, walk)\n        if self.ext_on:\n            self._ext_cmd()\n"),
    # done: route robots
    ("            free_out = self.free & (out_y | end | back)\n            success = end & ~self.free\n            over = self.ep >= int(EPISODE_S / X.CONTROL_DT)\n            term = term | (back & ~self.free)\n",
     "            if self.ext_on:\n"
     "                tile_ctr = self.t_centre[self.r_tile[self.route]]\n"
     "                off = (base[:, :2] - tile_ctr).abs().amax(-1) > 3.95\n"
     "                ext_succ = self.ext & (self.hold_t >= 1.0)\n"
     "                term = term & ~(self.ext & ~(tilt | low | hit | scrape | self.bad | causes.get(\"side_obstacle\", torch.zeros_like(tilt))))  # route robots: only their own failures\n"
     "                term = term | (self.ext & off)\n"
     "                out_y, end, back = out_y & ~self.ext, end & ~self.ext, back & ~self.ext\n"
     "            free_out = self.free & (out_y | end | back)\n            success = end & ~self.free\n            over = self.ep >= int(EPISODE_S / X.CONTROL_DT)\n            term = term | (back & ~self.free)\n"
     "            if self.ext_on:\n"
     "                success = success | ext_succ\n"
     "                for t_ in range(self.ext_ntypes):\n"
     "                    mt = self.ext & (self.ext_type == t_)\n"
     "                    nm = self.raster[\"ext\"][\"types\"][t_]\n"
     "                    self.cnt[f\"ext_{nm}_succ\"] = self.cnt.get(f\"ext_{nm}_succ\", 0) + int((ext_succ & mt).sum())\n"
     "                    self.cnt[f\"ext_{nm}_fail\"] = self.cnt.get(f\"ext_{nm}_fail\", 0) + int(((term | (over & ~ext_succ)) & mt).sum())\n"),
    # curriculum: route robots by route success
    ("        walk = ~self.free[ids]\n        wid = ids[walk]\n",
     "        if self.ext_on:\n"
     "            eid = ids[self.ext[ids]]\n"
     "            if len(eid):\n"
     "                up = success[eid]\n"
     "                prog = self.wpi[eid].float() / (self.r_n[self.route[eid]] - 1).clamp(min=1).float()\n"
     "                down = ~up & (prog < 0.5)\n"
     "                lv = self.ext_level[eid] + up.long() - down.long()\n"
     "                lv = torch.where(lv >= 10, torch.randint(0, 10, lv.shape, device=self.dev, generator=self.gen), lv).clamp(min=0)\n"
     "                self.ext_level[eid] = lv\n"
     "        walk = ~self.free[ids] & ~self.ext[ids]\n        wid = ids[walk]\n"),
])

edit("ssr/train.py", [
    ("    env = SSREnv(a.num_envs, dev, seed=a.seed, camera=a.camera, map_memory=a.map_memory)\n",
     "    if a.ext_frac > 0:                               # extended terrain map (user decision 2026-11-12)\n"
     "        from ssr import terrain as T_\n"
     "        _r = T_.add_extended(T_.generate(a.seed), types=tuple(a.ext_types.split(\",\")), variants=a.ext_variants)\n"
     "        env = SSREnv(a.num_envs, dev, seed=a.seed, camera=a.camera, map_memory=a.map_memory, raster=_r, tiles=True)\n"
     "        env.ext_type[:] = torch.randint(0, len(_r[\"ext\"][\"types\"]), (a.num_envs,), device=dev)\n"
     "        print(\"extended map:\", len(_r[\"ext\"][\"tiles\"]), \"tiles,\", len(_r[\"ext\"][\"routes\"]), \"routes, raster\", _r[\"z\"].shape, flush=True)\n"
     "    else:\n"
     "        env = SSREnv(a.num_envs, dev, seed=a.seed, camera=a.camera, map_memory=a.map_memory)\n"),
    ("    env.side_soft = a.side_soft\n",
     "    env.side_soft = a.side_soft\n    env.ext_frac = a.ext_frac\n"),
    ("    ap.add_argument(\"--side_soft\",",
     "    ap.add_argument(\"--ext_frac\", type=float, default=0.0, help=\"extended map: share of tile robots on route tiles (user decision 2026-11-12); 0 = off\")\n"
     "    ap.add_argument(\"--ext_types\", default=\"tight_spots,log_corridors,homes,street_front,stairs\")\n"
     "    ap.add_argument(\"--ext_variants\", type=int, default=2)\n"
     "    ap.add_argument(\"--side_soft\","),
    ("gpu_util=gu, gpu_mem_mb=gm, edge_stop=",
     "gpu_util=gu, gpu_mem_mb=gm, ext=env.ext_stats(), edge_stop="),
])

# env methods: insert before _edge_stop
p = C / "ssr/env.py"
s = p.read_text()
anchor = "    def _edge_stop(self, done_ids):\n"
assert s.count(anchor) == 1
s = s.replace(anchor, ENV_EXT_METHODS + anchor)
p.write_text(s)
print("patched")
