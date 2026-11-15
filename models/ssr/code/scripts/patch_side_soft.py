"""Side obstacles (2026-11-12): --side_soft W. Before: a limb centre within 0.06 m of a side-obstacle cell ends the episode.
With W > 0: only a limb centre inside an obstacle cell ends it; the 0.06 m band costs W per step (log term side_graze).
Backups ../_bak_20261112/*.pre_sidesoft"""
import pathlib
import shutil

C = pathlib.Path(__file__).resolve().parents[1]


def edit(rel, pairs):
    p = C / rel
    s = p.read_text()
    shutil.copy(p, C.parent / "_bak_20261112" / (p.name + ".pre_sidesoft"))
    for old, new in pairs:
        assert s.count(old) == 1, (rel, old)
        s = s.replace(old, new)
    p.write_text(s)


edit("ssr/env.py", [
    ("            causes[\"side_obstacle\"] = self._touches_obstacle()\n            term = term | causes[\"side_obstacle\"]\n",
     "            causes[\"side_obstacle\"] = self._touches_obstacle(0.0 if self.side_soft > 0 else 0.06)\n            term = term | causes[\"side_obstacle\"]\n"),
    ("            swing_h=self.swing_h_w *",
     "            side_graze=-self.side_soft * (self._touches_obstacle(0.06, log=False).float() if (self.side_soft > 0 and self.obst is not None) else torch.zeros(self.N, device=self.dev)),   # deviation (2026-11-12)\n"
     "            swing_h=self.swing_h_w *"),
    ("    def _touches_obstacle(self, radius=0.06):\n",
     "    def _touches_obstacle(self, radius=0.06, log=True):\n"),
    ("        self.log[\"touch\"] = self.log.get(\"touch\", 0) + int(hit.sum())\n",
     "        if log:\n            self.log[\"touch\"] = self.log.get(\"touch\", 0) + int(hit.sum())\n"),
    ("        self.edge_stop_p = 0.0 ",
     "        self.side_soft = 0.0                    # side obstacles: 0 = touch within 0.06 m ends the episode; W > 0 = only overlap ends it, the band costs W per step\n"
     "        self.edge_stop_p = 0.0 "),
])
edit("ssr/train.py", [
    ("    env.edge_stop_p = a.edge_stop\n", "    env.edge_stop_p = a.edge_stop\n    env.side_soft = a.side_soft\n"),
    ("    ap.add_argument(\"--prior_turn_min\",",
     "    ap.add_argument(\"--side_soft\", type=float, default=0.0, help=\"deviation: side-obstacle 0.06 m band costs this per step instead of ending the episode (only overlap ends it); 0 = off\")\n"
     "    ap.add_argument(\"--prior_turn_min\","),
])
print("patched")
