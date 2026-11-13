"""Edge stops (2026-11-12): --edge_stop P. Backups ../_bak_20261112/*.pre_edge"""
import pathlib
import shutil

C = pathlib.Path(__file__).resolve().parents[1]
BAK = C.parent / "_bak_20261112"


def edit(rel, pairs):
    p = C / rel
    s = p.read_text()
    shutil.copy(p, BAK / (p.name + ".pre_edge"))
    for old, new in pairs:
        assert s.count(old) == 1, (rel, old)
        s = s.replace(old, new)
    p.write_text(s)


EDGE = '''    def _edge_stop(self, done_ids):
        """Deviation (2026-11-12, ACP's fix for the apparatus' last station): in a share of tile episodes, the first time the
        ground 0.45 m ahead (along the robot's heading) is 0.15 m or more below the ground under it, the command is set to zero
        for 2 s (the robot must stop and stand at a down edge), then the episode's speed comes back."""
        if len(done_ids):
            self.es_armed[done_ids] = (self._u(len(done_ids), lo=0.0, hi=1.0) < self.edge_stop_p) & ~self.free[done_ids]
            self.es_timer[done_ids] = 0
        act = self.es_timer > 0
        if bool(act.any()):
            self.es_timer[act] = self.es_timer[act] - 1
            end = act & (self.es_timer == 0)
            if bool(end.any()):
                self.speed[end] = self.es_speed[end]
                self.cnt["es_done"] = self.cnt.get("es_done", 0) + int(end.sum())
        cand = self.es_armed & (self.es_timer == 0) & ~self.free & (self.speed > EPS_CMD)
        if bool(cand.any()):
            p, yaw = self.s["p"], self.s["yaw"]
            here = self.hmap(p[:, 0], p[:, 1])
            ahead = self.hmap(p[:, 0] + 0.45 * torch.cos(yaw), p[:, 1] + 0.45 * torch.sin(yaw))
            hit = cand & ((here - ahead) >= 0.15)
            if bool(hit.any()):
                self.es_speed[hit] = self.speed[hit]
                self.speed[hit] = 0.0
                self.es_timer[hit] = int(round(2.0 / X.CONTROL_DT))
                self.es_armed[hit] = False
                self.cnt["es_start"] = self.cnt.get("es_start", 0) + int(hit.sum())

'''

edit("ssr/env.py", [
    ("        self.vf[ids] = 0.0\n        self._update_cmd()\n",
     "        self.vf[ids] = 0.0\n        if self.edge_stop_p > 0 and self.train and self.tiles:\n            self._edge_stop(ids)\n        self._update_cmd()\n"),
    ("    def _rel_scale(self, c, c_ref, c_min):\n", EDGE + "    def _rel_scale(self, c, c_ref, c_min):\n"),
    ("        self.rel_kernel = False",
     "        self.edge_stop_p = 0.0                  # share of tile episodes with a stop at a down edge (deviation; --edge_stop)\n"
     "        self.es_armed = torch.zeros(n, dtype=torch.bool, device=dev)\n"
     "        self.es_timer = torch.zeros(n, dtype=torch.long, device=dev)\n"
     "        self.es_speed = torch.zeros(n, device=dev)\n"
     "        self.rel_kernel = False"),
])
edit("ssr/train.py", [
    ("    env.rel_kernel = a.rel_kernel\n", "    env.rel_kernel = a.rel_kernel\n    env.edge_stop_p = a.edge_stop\n"),
    ("    ap.add_argument(\"--rel_kernel\",",
     "    ap.add_argument(\"--edge_stop\", type=float, default=0.0, help=\"deviation: share of tile episodes that stop for 2 s at the first down edge (>= 0.15 m drop 0.45 m ahead)\")\n"
     "    ap.add_argument(\"--rel_kernel\","),
    ("gpu_util=gu, gpu_mem_mb=gm)",
     "gpu_util=gu, gpu_mem_mb=gm, edge_stop=dict(start=cnt.get(\"es_start\", 0.0), done=cnt.get(\"es_done\", 0.0)))"),
])
print("patched")
