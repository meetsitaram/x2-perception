"""1,000 restart options (coordinator, 2026-11-12 03:20; deviations):
--style_from N: style weight by schedule, 0 before N, ramped linearly to --w_style over --gate_ramp iterations (overrides the latch).
--feet_air_w W --feet_air_min T: feet-air-time reward, at each touchdown under a moving command
  W * clamp(air time - T, 0, T_AIR - T) (T_AIR = 0.4 s, the paper's air-time target). Positive only: taps under T and standing earn
  nothing, so it cannot pay standing over stepping. Logged as term `feet_air`. Default off. Apply after patch_obst.py."""
import pathlib
R = pathlib.Path(__file__).resolve().parents[1]


def sub(path, old, new, count=1):
    p = R / path
    s = p.read_text()
    assert s.count(old) == count, (path, old[:60], s.count(old))
    p.write_text(s.replace(old, new))


E = "ssr/env.py"
sub(E, "        self.obst_target = None ",
    "        self.feet_air_w, self.feet_air_min = 0.0, 0.1   # feet-air-time reward (not in the paper; off unless --feet_air_w)\n"
    "        self.obst_target = None ")
sub(E, "            imitate=self._imitation(q_rel),\n",
    "            imitate=self._imitation(q_rel),\n"
    "            feet_air=self.feet_air_w * ((self.air.clamp(max=T_AIR) - self.feet_air_min).clamp(min=0.0) * first.float()).sum(-1) * (~still).float(),\n")
Tr = "ssr/train.py"
sub(Tr, "        if a.style_low >= 0:  ",
    "        if a.style_from > 0:                         # style by schedule from style_from (overrides the latch)\n"
    "            w[2] = a.w_style * ramp * min(1.0, max(0.0, (it - a.style_from) / max(a.gate_ramp, 1)))\n"
    "        elif a.style_low >= 0:  ")
sub(Tr, "    env.flat_only = a.flat_only or a.flat_until > 0\n",
    "    env.flat_only = a.flat_only or a.flat_until > 0\n"
    "    env.feet_air_w, env.feet_air_min = a.feet_air_w, a.feet_air_min\n")
sub(Tr, "    ap.add_argument(\"--obst0\",",
    "    ap.add_argument(\"--style_from\", type=int, default=0, help=\"deviation: style weight 0 before this iteration, ramped to --w_style over --gate_ramp; overrides --style_low; 0 = off\")\n"
    "    ap.add_argument(\"--feet_air_w\", type=float, default=0.0, help=\"deviation: feet-air-time reward weight (per touchdown, W * clamp(air - feet_air_min, 0, 0.4 - feet_air_min)); 0 = off\")\n"
    "    ap.add_argument(\"--feet_air_min\", type=float, default=0.1, help=\"air time below which a touchdown earns nothing\")\n"
    "    ap.add_argument(\"--obst0\",")
print("patched")
