# ACP data-2h code edits (2026-11-13): paper Table S2 items, gated by ACP_PAPER_DR=1 (default 0 = previous behaviour).
import sys
P = sys.argv[1]; s = open(P).read()
def rep(a, b):
    global s
    assert s.count(a) == 1, ("not unique", a[:80]); s = s.replace(a, b)
rep("import math\n", "import math\nimport os\n")
rep("class ", "PAPER_DR = os.environ.get(\"ACP_PAPER_DR\", \"0\") == \"1\"   # data 2h: Table S2 items (state/acp.md 17d)\n\n\nclass ", ) if s.count("\nclass ") == 1 else None
# friction: paper = robot U(0.3,1.2), ground 1.0, PhysX average combine -> effective (r+1)/2; set on both sides so MuJoCo's
# max-combine and the height field's priority (ACP_HF_FIX) both give that value
rep("            self.friction[:, self.robot_geom_ids, 0] = u(N, 1, lo=0.3, hi=1.2)\n",
    "            self.friction[:, self.robot_geom_ids, 0] = u(N, 1, lo=0.3, hi=1.2)\n"
    "            if PAPER_DR:\n"
    "                mu = (self.friction[:, self.robot_geom_ids[:1], 0] + 1.0) * 0.5\n"
    "                self.friction[:, self.robot_geom_ids, 0] = mu\n"
    "                self.friction[:, ~robot_geom, 0] = mu\n")
# odometry drift (Table S2): per-episode XY offset in a 0.05 m disk, Z U(+-0.05), on the estimated base position used for the
# reference contact targets and the goal (scans are body-mounted, planner targets are body-frame: not affected). Delay not modelled.
rep("        self.scan_off = torch.zeros(N, 1, device=dev)\n",
    "        self.scan_off = torch.zeros(N, 1, device=dev)\n        self.odo_off = torch.zeros(N, 3, device=dev)\n")
rep("        self.scan_off[ids] = 0.0 if self.eval_mode else self._u(n, 1) * 0.1\n",
    "        self.scan_off[ids] = 0.0 if self.eval_mode else self._u(n, 1) * 0.1\n"
    "        if PAPER_DR and not self.eval_mode:\n"
    "            rr = 0.05 * ((self._u(n, 1) + 1.0) * 0.5).sqrt(); th = self._u(n, 1) * math.pi\n"
    "            self.odo_off[ids] = torch.cat((rr * th.cos(), rr * th.sin(), self._u(n, 1) * 0.05), -1)\n")
rep("        tgt = mtv3(H.unsqueeze(1), ref_w - p.unsqueeze(1))\n",
    "        p_o = p + self.odo_off if (noise and PAPER_DR) else p      # odometry estimate (Table S2)\n"
    "        tgt = mtv3(H.unsqueeze(1), ref_w - p_o.unsqueeze(1))\n")
rep("        dxy = mtv3(H, m.root_p[g] - p)[:, :2]\n", "        dxy = mtv3(H, m.root_p[g] - p_o)[:, :2]\n")
# tracker scan: Table S2 U(-0.1, 0.1) per step (was N(0, 0.03), the planner's value); per-episode offset kept
rep("            scan = scan + self.scan_off + torch.randn(scan.shape, device=self.dev, generator=self.gen) * 0.03\n",
    "            if PAPER_DR:\n"
    "                scan = scan + self.scan_off + self._u(*scan.shape) * 0.1\n"
    "            else:\n"
    "                scan = scan + self.scan_off + torch.randn(scan.shape, device=self.dev, generator=self.gen) * 0.03\n")
# contact-target noise +-0.1 m on planner targets too (Table S2: all actor observations); command / reward stay noise-free
rep("            tgt = torch.where(use[:, None, None], ptgt, tgt)       # the complete command is switched, not interpolated\n",
    "            ptgt_o = ptgt + self._u(self.N, 9, 3) * 0.1 * pm.unsqueeze(-1).float() if (noise and PAPER_DR) else ptgt\n"
    "            tgt = torch.where(use[:, None, None], ptgt_o, tgt)     # the complete command is switched, not interpolated\n")
open(P, "w").write(s); print("ok", P)
