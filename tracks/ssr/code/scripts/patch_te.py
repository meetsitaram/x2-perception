"""t_e patch (coordinator, 2026-11-12; not in the paper): forward-progress gate on the posture/alive terms for robots with a
moving command, ramped off once mean touchdown air time passes --gate_air. Also logs touchdown air time and mean progress.
Default off (--prog_gate 0), so earlier commands behave as before. Applied with: python scripts/patch_te.py (from tracks/ssr/code)."""
import pathlib
R = pathlib.Path(__file__).resolve().parents[1]


def sub(path, old, new, count=1):
    p = R / path
    s = p.read_text()
    assert s.count(old) == count, (path, old[:60], s.count(old))
    p.write_text(s.replace(old, new))


E = "ssr/env.py"
sub(E, 'CORE_TERMS = (',
    'GATED_TERMS = ("orientation", "ang_vel", "base_height", "ang_vel_xy", "lateral")   # t_e: posture/alive-like terms a stander collects\n'
    'CORE_TERMS = (')
sub(E, "        self.late_scale = 1.0 ",
    "        self.prog_gate = 0.0                    # t_e (not in the paper): weight of the forward-progress gate on GATED_TERMS for robots with a moving command\n"
    "        self.v_ema = torch.zeros(n, 2, device=dev)   # base planar velocity (body frame), 0.5 s moving average, for the progress gate\n"
    "        self.prog = torch.ones(n, device=dev)\n"
    "        self.late_scale = 1.0 ")
sub(E, "        self.air[ids], self.single_t[ids] = 0.0, 0.0\n        self.contact_prev[ids] = False",
    "        self.air[ids], self.single_t[ids] = 0.0, 0.0\n        self.v_ema[ids] = 0.0\n        self.contact_prev[ids] = False")
sub(E, "        if self.late_scale < 1.0:\n            terms = {k_: (v_ if k_ in CORE_TERMS",
    "        self.v_ema += 0.04 * (v[:, :2] - self.v_ema)\n"
    "        cn = cmd[:, :2].norm(dim=-1)\n"
    "        mov = cn > EPS_CMD\n"
    "        prog = ((self.v_ema * cmd[:, :2]).sum(-1) / cn.clamp(min=1e-3) ** 2).clamp(0.0, 1.0)   # share of the commanded speed actually made good\n"
    "        self.prog = torch.where(mov, prog, torch.ones_like(prog))\n"
    "        self.cnt[\"prog_sum\"] = self.cnt.get(\"prog_sum\", 0) + (prog * mov).sum()\n"
    "        self.cnt[\"prog_n\"] = self.cnt.get(\"prog_n\", 0) + mov.sum()\n"
    "        if self.prog_gate > 0.0:\n"
    "            gf = 1.0 - self.prog_gate * (1.0 - self.prog)\n"
    "            terms.update({k_: terms[k_] * gf for k_ in GATED_TERMS})\n"
    "        if self.late_scale < 1.0:\n            terms = {k_: (v_ if k_ in CORE_TERMS")
sub(E, "        touchdown = contact & ~self.contact_prev & (self.air > 0.0)\n",
    "        touchdown = contact & ~self.contact_prev & (self.air > 0.0)\n"
    "        mtd = touchdown & (self.cmd[:, :2].norm(dim=-1) > EPS_CMD)[:, None]      # air time of each touchdown under a moving command\n"
    "        self.cnt[\"air_n\"] = self.cnt.get(\"air_n\", 0) + mtd.sum()\n"
    "        self.cnt[\"air_sum\"] = self.cnt.get(\"air_sum\", 0) + (self.air * mtd.float()).sum()\n"
    "        self.cnt[\"air_long\"] = self.cnt.get(\"air_long\", 0) + (mtd & (self.air > 0.2)).sum()\n")

Tr = "ssr/train.py"
sub(Tr, "    t_start = time.time()\n    done_stats = {\"ret\": [], \"len\": []}\n",
    "    t_start = time.time()\n    done_stats = {\"ret\": [], \"len\": []}\n"
    "    gate_start, gate_hits, air_td = (ck.get(\"gate_start\") if a.resume else None), 0, 0.0   # t_e progress gate: iteration its ramp-off began\n")
sub(Tr, "        env.stuck_term = a.stuck_term and (a.stuck_until <= 0 or it < a.stuck_until)\n",
    "        env.stuck_term = a.stuck_term and (a.stuck_until <= 0 or it < a.stuck_until)\n"
    "        env.prog_gate = 0.0 if a.prog_gate <= 0 else a.prog_gate * (1.0 if gate_start is None else max(0.0, 1.0 - (it - gate_start) / max(a.gate_ramp, 1)))\n")
sub(Tr, "        if it % a.log_every == 0:\n            lv = ",
    "        cnt_ = env.cnt\n"
    "        air_td = float(cnt_.get(\"air_sum\", 0.0)) / max(float(cnt_.get(\"air_n\", 0.0)), 1.0)\n"
    "        gate_hits = gate_hits + 1 if air_td > a.gate_air else 0\n"
    "        if a.prog_gate > 0 and gate_start is None and gate_hits >= 5:\n"
    "            gate_start = it + 1\n"
    "            print(\"progress gate: touchdown air time\", round(air_td, 3), \"for 5 iterations; ramping the gate off from\", gate_start, flush=True)\n"
    "        if it % a.log_every == 0:\n            lv = ")
sub(Tr, "samples_per_s=round(N_ * Tn / max(time.time() - t0, 1e-6)), num_envs=N_,",
    "samples_per_s=round(N_ * Tn / max(time.time() - t0, 1e-6)), num_envs=N_, air_td=round(air_td, 3), "
    "air_long_share=round(float(cnt.get(\"air_long\", 0.0)) / max(float(cnt.get(\"air_n\", 0.0)), 1.0), 3), "
    "prog_mean=round(float(cnt.get(\"prog_sum\", 0.0)) / max(float(cnt.get(\"prog_n\", 0.0)), 1.0), 3), prog_gate=round(env.prog_gate, 3), gate_start=gate_start,")
sub(Tr, "    ap.add_argument(\"--stuck_term\",",
    "    ap.add_argument(\"--prog_gate\", type=float, default=0.0, help=\"t_e (deviation): multiply orientation, yaw, base height, roll/pitch rate and lateral terms by 1 - g(1 - progress) for robots with a moving command; 0 = off\")\n"
    "    ap.add_argument(\"--gate_air\", type=float, default=0.2, help=\"with --prog_gate: start ramping the gate off once mean touchdown air time exceeds this for 5 iterations\")\n"
    "    ap.add_argument(\"--gate_ramp\", type=int, default=500, help=\"with --prog_gate: iterations over which the gate ramps off\")\n"
    "    ap.add_argument(\"--stuck_term\",")
print("patched")
sub(Tr, "                      guide_on=env.guide_on, args=vars(a))", "                      guide_on=env.guide_on, gate_start=gate_start, args=vars(a))")
print("patched ckpt")
