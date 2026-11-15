"""SSR 2026-11-12 19:05 (user rule on wall contact, same day): extended-map walls and props as physical MuJoCo boxes, one set
per world (K slots, batched geom_pos/quat/size/aabb/rbound, set at reset from the robot's route), contact graded by body part:
  - forearm / hand (elbow, wrist bodies) on a wall: free up to 40 N and 0.5 s continuous, beyond that -0.5 per step; never ends
    the episode;
  - any other body (shoulders, torso, pelvis, hips, legs, feet): -2 x (1 + F / 100 N, at most 5) per step; ends the episode only
    above 400 N (hard hit) or by the usual falls;
  - route robots with physical walls: no limb-overlap termination and no graze band; only the base inside an obstacle cell ends
    the episode (a wall not in the slots, walked through).
Also: sidestep law on route segments with less than 0.39 m clearance (turn the shoulders 90 deg to the path, then walk
sideways at up to 0.3 m/s), and tight_spots / homes use the beam discriminator (the user's sideways Pico takes are there).
Flag: --ext_phys K (0 = off, as before)."""
import shutil
C = "/mnt/filesystem-g4/tracks/ssr/code/ssr/"
for f in ("terrain.py", "train.py", "env.py"):
    shutil.copy(C + f, C + f + ".pre_extphys")


def rep(s, a, b, n=1):
    assert s.count(a) == n, (a[:80], s.count(a))
    return s.replace(a, b)


# ---------------------------------------------------------------- terrain: physical boxes per tile
t = open(C + "terrain.py").read()
a = "                tiles.append(dict(type=ti, name=t, level=lvl, tile_id=m[\"tile_id\"], centre=(gcx, gcy)))\n"
t = rep(t, a, a + """                pb, i_ = [], m["index"]                                     # physical tall boxes (role 2) and tall meshes as boxes, world coords
                bs = dat["box_start"]
                for bi in range(int(bs[i_]), int(bs[i_ + 1])):
                    c_, h_ = dat["box_center"][bi], dat["box_half"][bi]
                    if int(dat["box_role"][bi]) == 2 and c_[2] - abs(h_).max() < 1.6:
                        pb.append([gcx + float(c_[0]), gcy + float(c_[1]), float(c_[2])] + [float(x) for x in h_] + [float(x) for x in dat["box_quat"][bi]])
                if "mesh_start" in dat.files:
                    ms = dat["mesh_start"]
                    for mi in range(int(ms[i_]), int(ms[i_ + 1])):
                        c_, h_ = dat["mesh_aabb_center"][mi], dat["mesh_aabb_half"][mi]
                        if int(dat["mesh_role"][mi]) == 2 and c_[2] - h_[2] < 1.6:
                            pb.append([gcx + float(c_[0]), gcy + float(c_[1]), float(c_[2])] + [float(x) for x in h_] + [1.0, 0.0, 0.0, 0.0])
                tiles[-1]["phys"] = pb
""")
open(C + "terrain.py", "w").write(t)

# ---------------------------------------------------------------- train: flag
r = open(C + "train.py").read()
r = rep(r, "raster=_r, tiles=True)", "raster=_r, tiles=True, ext_phys=a.ext_phys)")
a = '    ap.add_argument("--ext_variants", type=int, default=2)\n'
r = rep(r, a, a + '    ap.add_argument("--ext_phys", type=int, default=0, help="extended map: K physical wall boxes per world, contact graded by body part (user rule 2026-11-12); 0 = vision-only walls")\n')
open(C + "train.py", "w").write(r)

# ---------------------------------------------------------------- env
e = open(C + "env.py").read()
e = rep(e, "nconmax: int = 150, seed", "nconmax: int = 150, ext_phys: int = 0, seed")
e = rep(e, "        self.x2 = X.build(self.raster, walls)\n", """        self.ext_phys = int(ext_phys)
        self._wall0 = len(walls or [])
        if self.ext_phys:
            walls = (walls or []) + [dict(center=(0.0, 0.0, -20.0 - 0.5 * i), half=(0.05, 0.05, 0.05), quat=(1.0, 0.0, 0.0, 0.0)) for i in range(self.ext_phys)]
        self.x2 = X.build(self.raster, walls)
""")
e = rep(e, "            self.mm = mjw.put_model(m, batch_sizes=batch)\n", """            if self.ext_phys:
                batch.update({k: num_envs for k in ("geom_pos", "geom_quat", "geom_size", "geom_aabb", "geom_rbound")})
            self.mm = mjw.put_model(m, batch_sizes=batch)
""")
a = "        self.gain, self.bias = t(mm.actuator_gainprm), t(mm.actuator_biasprm)\n"
e = rep(e, a, a + """        if self.ext_phys:
            self.g_pos, self.g_quat, self.g_size, self.g_aabb, self.g_rb = (t(mm.geom_pos), t(mm.geom_quat), t(mm.geom_size), t(mm.geom_aabb), t(mm.geom_rbound))
""")
a = "        self.geom_side, self.geom_term = side, term\n"
e = rep(e, a, a + """        self.geom_wall = torch.zeros(m.ngeom, device=dev, dtype=torch.bool)
        self.geom_cls = torch.zeros(m.ngeom, device=dev, dtype=torch.long)     # 0 world, 1 forearm / hand, 2 any other robot body
        for g in range(m.ngeom):
            b = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ""
            if m.geom_bodyid[g] > 0:
                self.geom_cls[g] = 1 if ("elbow" in b or "wrist" in b or "hand" in b) else 2
        self.wall_ids = torch.tensor([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"wall_{self._wall0 + i}") for i in range(self.ext_phys)], device=dev, dtype=torch.long)
        if self.ext_phys:
            self.geom_wall[self.wall_ids] = True
""")
e = rep(e, "        self.term_force = z()\n", "        self.term_force = z()\n        self.wall_f, self.arm_t = z(3), z()\n")
# contacts
e = rep(e, "        self.term_force[:] = 0.0\n        n = int(", "        self.term_force[:] = 0.0\n        self.wall_f[:] = 0.0\n        n = int(")
e = rep(e, """        Fw = (fr.transpose(-1, -2) @ fc[..., None]).squeeze(-1)
        for col in (0, 1):
            s = self.geom_side[geom[:, col]]
            ok = s >= 0
""", """        Fw = (fr.transpose(-1, -2) @ fc[..., None]).squeeze(-1)
        w0, w1 = self.geom_wall[geom[:, 0]], self.geom_wall[geom[:, 1]]
        nw = ~(w0 | w1)
        if self.ext_phys:                                              # wall contacts graded by body part (user rule 2026-11-12)
            cls = self.geom_cls[torch.where(w0, geom[:, 1], geom[:, 0])]
            okw = (w0 | w1) & (cls > 0)
            self.wall_f.view(-1).index_add_(0, w[okw] * 3 + cls[okw], fc[okw, 0])
        for col in (0, 1):
            s = self.geom_side[geom[:, col]]
            ok = (s >= 0) & nw
""")
e = rep(e, "        t_ok = self.geom_term[geom[:, 0]] | self.geom_term[geom[:, 1]]\n",
        "        t_ok = (self.geom_term[geom[:, 0]] | self.geom_term[geom[:, 1]]) & nw\n")
# arm contact timer: after contacts each step
e = rep(e, "        self.term_force.index_add_(0, w[t_ok], fc[t_ok, 0])\n", """        self.term_force.index_add_(0, w[t_ok], fc[t_ok, 0])

    def _wall_timer(self):
        self.arm_t = torch.where(self.wall_f[:, 1] > 1.0, self.arm_t + X.CONTROL_DT, torch.zeros_like(self.arm_t))
""")
# discriminator type for tight_spots / homes -> beam
e = rep(e, "torch.where(self.ext_stairs[self.ext_type[eid]], torch.full_like(self.type[eid], T.STAIRS), torch.full_like(self.type[eid], T.FLAT))",
        "torch.where(self.ext_stairs[self.ext_type[eid]], torch.full_like(self.type[eid], T.STAIRS), torch.where(self.ext_beam[self.ext_type[eid]], torch.full_like(self.type[eid], T.BEAM), torch.full_like(self.type[eid], T.FLAT)))")
e = rep(e, "        self.ext_stairs = torch.tensor([nm == \"stairs\" for nm in e[\"types\"]], device=dev)\n",
        "        self.ext_stairs = torch.tensor([nm == \"stairs\" for nm in e[\"types\"]], device=dev)\n"
        "        self.ext_beam = torch.tensor([nm in (\"tight_spots\", \"homes\") for nm in e[\"types\"]], device=dev)   # sideways Pico takes are in the beam set\n")
# setup: per-route boxes and per-segment clearance
a = "        self.r_level = self.t_level[self.r_tile]\n"
e = rep(e, a, a + """        import numpy as _np
        from scipy import ndimage as _nd
        hz, vz = _np.asarray(self.raster["z"]), _np.asarray(self.raster["vision"]["z"])
        dist = _nd.distance_transform_edt(~((vz - hz) > 0.25)) * self.raster["res"]
        segc = _np.full((len(R), W), 9.9, _np.float32)
        for i, r in enumerate(R):
            pts = [r["start"][:2]] + list(r["wp"])
            for j in range(len(pts) - 1):
                (ax, ay), (bx, by) = pts[j], pts[j + 1]
                s_ = _np.linspace(0, 1, max(2, int(_np.hypot(bx - ax, by - ay) / 0.02)))
                ii = _np.clip(_np.round((ax + s_ * (bx - ax) - self.raster["x0"]) / self.raster["res"]).astype(int), 0, dist.shape[0] - 1)
                jj = _np.clip(_np.round((ay + s_ * (by - ay) - self.raster["y0"]) / self.raster["res"]).astype(int), 0, dist.shape[1] - 1)
                segc[i, j] = dist[ii, jj].min()
            segc[i, len(pts) - 1:] = segc[i, len(pts) - 2]
        self.r_segclr = torch.as_tensor(segc, device=dev)
        if self.ext_phys:
            K = self.ext_phys
            park = torch.tensor([0.0, 0.0, -20.0, 0.05, 0.05, 0.05, 1.0, 0.0, 0.0, 0.0])
            rb = park.repeat(len(R), K, 1)
            for i, r in enumerate(R):
                pb = tiles[r["tile"]].get("phys", [])
                if not pb:
                    continue
                B = torch.tensor(pb, dtype=torch.float32)
                pts = torch.tensor([r["start"][:2]] + list(r["wp"]), dtype=torch.float32)
                a0, b0 = pts[:-1], pts[1:]
                ab = b0 - a0
                ap_ = B[:, None, :2] - a0[None]
                tt = ((ap_ * ab[None]).sum(-1) / (ab * ab).sum(-1).clamp(min=1e-6)[None]).clamp(0, 1)
                dd_ = (ap_ - tt[..., None] * ab[None]).norm(dim=-1).min(-1).values - B[:, 3:5].abs().amax(-1)
                o = torch.argsort(dd_)[:K]
                rb[i, :len(o)] = B[o]
            self.r_box = rb.to(dev)
            self.park = park.to(dev)
            self.ext_box_overflow = int(sum(max(0, len(tiles[r["tile"]].get("phys", [])) - K) for r in R))
""")
# per-world box slots at reset
e = rep(e, "        self.reset_ids = ids\n", "        self.reset_ids = ids\n        self._ext_walls(ids)\n")
e = rep(e, "    def _ext_pick(self, ids):\n", """    def _ext_walls(self, ids):
        if not self.ext_phys or not self.ext_on:
            return
        B = self.park.expand(len(ids), self.ext_phys, 10).clone()
        ex = self.ext[ids]
        if bool(ex.any()):
            B[ex] = self.r_box[self.route[ids[ex]]]
        gi = self.wall_ids[None].expand(len(ids), -1)
        wi = ids[:, None].expand_as(gi)
        self.g_pos[wi, gi] = B[..., 0:3]
        self.g_size[wi, gi] = B[..., 3:6].abs()
        self.g_quat[wi, gi] = B[..., 6:10]
        self.g_aabb[wi, gi, 0] = 0.0
        self.g_aabb[wi, gi, 1] = B[..., 3:6].abs()
        self.g_rb[wi, gi] = B[..., 3:6].norm(dim=-1)
        self.arm_t[ids] = 0.0

    def _ext_pick(self, ids):
""")
# sidestep law on tight segments
e = rep(e, "        c = torch.stack((torch.where(hold, torch.zeros_like(vx), vx), torch.zeros_like(vx), torch.where(hold, torch.zeros_like(wz), wz)), -1)\n",
"""        vy = torch.zeros_like(vx)
        tight = (self.r_segclr[self.route, self.wpi] < 0.39) & (self.ext_phys > 0)                # sidestep (user rule 2026-11-12): shoulders across the path, walk sideways
        if bool((m & tight).any()):
            prev = torch.where((self.wpi > 0)[:, None], self.r_wp[self.route, (self.wpi - 1).clamp(min=0)], self.r_start[self.route, :2])
            seg = tgt - prev
            sh = torch.atan2(seg[:, 1], seg[:, 0])
            e1, e2 = wrap(sh + math.pi / 2 - s["yaw"]), wrap(sh - math.pi / 2 - s["yaw"])
            eh = torch.where(e1.abs() < e2.abs(), e1, e2)
            spd = dist.clamp(0.15, None).minimum(self.ext_vmax).clamp(max=0.3)
            cy_, sy_ = torch.cos(s["yaw"]), torch.sin(s["yaw"])
            u = vec / dist.clamp(min=1e-6)[:, None]
            bx, by = cy_ * u[:, 0] + sy_ * u[:, 1], -sy_ * u[:, 0] + cy_ * u[:, 1]
            go = eh.abs() < math.radians(30.0)
            vx_t = torch.where(go, (spd * bx).clamp(-0.3, 0.3), torch.zeros_like(vx))
            vy_t = torch.where(go, (spd * by).clamp(-0.3, 0.3), torch.zeros_like(vx))
            wz_t = (1.5 * eh).clamp(-1.0, 1.0)
            vx, vy, wz = torch.where(tight, vx_t, vx), torch.where(tight, vy_t, vy), torch.where(tight, wz_t, wz)
        c = torch.stack((torch.where(hold, torch.zeros_like(vx), vx), torch.where(hold, torch.zeros_like(vy), vy), torch.where(hold, torch.zeros_like(wz), wz)), -1)
""")
e = rep(e, "        self._contacts()\n", "        self._contacts()\n        self._wall_timer()\n", n=2)
# rewards
e = rep(e, "            side_graze=-self.side_soft * (", """            wall_body=-2.0 * (self.wall_f[:, 2] > 1.0).float() * (1.0 + self.wall_f[:, 2] / 100.0).clamp(max=5.0),
            wall_arm=-0.5 * ((self.wall_f[:, 1] > 40.0) | (self.arm_t > 0.5)).float(),
            side_graze=-self.side_soft * (1.0 - (self.ext & (self.ext_phys > 0)).float()) * (""")
# done
e = rep(e, "            causes[\"side_obstacle\"] = self._touches_obstacle(0.0 if self.side_soft > 0 else 0.06)\n", """            so = self._touches_obstacle(0.0 if self.side_soft > 0 else 0.06)
            if self.ext_phys:                                          # route robots with physical walls: base inside a wall cell, or a hard hit
                bi = torch.round((base[:, 0] - self.hmap.x0) / self.hmap.res).long().clamp(0, self.hmap.nx - 1)
                bj = torch.round((base[:, 1] - self.hmap.y0) / self.hmap.res).long().clamp(0, self.hmap.ny - 1)
                wall_hit = self.ext & (self.wall_f[:, 2] > 400.0)
                causes["wall_hit"] = wall_hit
                so = torch.where(self.ext, self.obst[bi, bj] | wall_hit, so)
            causes["side_obstacle"] = so
""")
open(C + "env.py", "w").write(e)
print("patched")
