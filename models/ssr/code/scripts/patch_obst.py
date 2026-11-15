"""Obstacle-tile share ramp (coordinator, 2026-11-12 02:53; part of the flat-first departure): when --flat_until switches the
curriculum on, only --obst0 of the robots get obstacle tiles (ramp, stairs, beam, platform, level 0); the rest stay on the flat
type. Over --obst_ramp iterations the target share rises to 0.8 (the paper's uniform mix over five types); robots are moved from
the flat type to a random obstacle type (level 0) only when their episode ends. After the ramp, types are fixed again.
Default --obst_ramp 0 = all at once (uniform). Apply after patch_flat_until.py."""
import pathlib
R = pathlib.Path(__file__).resolve().parents[1]


def sub(path, old, new, count=1):
    p = R / path
    s = p.read_text()
    assert s.count(old) == count, (path, old[:60], s.count(old))
    p.write_text(s.replace(old, new))


E = "ssr/env.py"
sub(E, "        self.late_scale = 1.0 ",
    "        self.obst_target = None                 # obstacle-tile share ramp after the flat phase (None = types fixed)\n"
    "        self.late_scale = 1.0 ")
sub(E, "            if self.flat_only:\n                self.type[ids], self.level[ids] = T.FLAT, 0\n",
    "            if self.obst_target is not None and not self.flat_only:     # move robots from the flat type to obstacle tiles as their episodes end\n"
    "                fl = ids[self.type[ids] == T.FLAT]\n"
    "                need = int(self.obst_target * self.N) - int((self.type != T.FLAT).sum())\n"
    "                if need > 0 and len(fl):\n"
    "                    cv = fl[:need]\n"
    "                    self.type[cv] = torch.randint(0, 4, (len(cv),), device=self.dev, generator=self.gen)\n"
    "                    self.level[cv] = 0\n"
    "            if self.flat_only:\n                self.type[ids], self.level[ids] = T.FLAT, 0\n")
Tr = "ssr/train.py"
sub(Tr, "            env.type[:] = torch.randint(0, 5, (N_,), device=dev)\n",
    "            env.type[:] = torch.randint(0, 5, (N_,), device=dev) if a.obst_ramp <= 0 else torch.where(torch.rand(N_, device=dev) < a.obst0, torch.randint(0, 4, (N_,), device=dev), torch.full((N_,), T.FLAT, device=dev))\n")
sub(Tr, "        env.boot = max(0.0, 1.0 - it / a.boot_iters)",
    "        env.obst_target = None if (a.obst_ramp <= 0 or a.flat_until <= 0 or env.flat_only or it >= a.flat_until + a.obst_ramp) else a.obst0 + (0.8 - a.obst0) * (it - a.flat_until) / a.obst_ramp\n"
    "        env.boot = max(0.0, 1.0 - it / a.boot_iters)")
sub(Tr, "prog_gate=round(env.prog_gate, 3), gate_start=gate_start,",
    "prog_gate=round(env.prog_gate, 3), gate_start=gate_start, flat_phase=bool(env.flat_only), obst_share=round(float((env.type != T.FLAT).float().mean()), 3), obst_target=env.obst_target,")
sub(Tr, "    ap.add_argument(\"--flat_until\",",
    "    ap.add_argument(\"--obst0\", type=float, default=0.25, help=\"with --flat_until and --obst_ramp: share of robots on obstacle tiles at the switch\")\n"
    "    ap.add_argument(\"--obst_ramp\", type=int, default=0, help=\"with --flat_until: iterations over which the obstacle share rises from --obst0 to 0.8; 0 = all at once\")\n"
    "    ap.add_argument(\"--flat_until\",")
print("patched")
