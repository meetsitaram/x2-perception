"""Transparent logs + checkpoint retention. Run in /mnt/filesystem-g4/tracks/ssr/code."""
import pathlib


def sub(p, pairs):
    f = pathlib.Path(p)
    s = f.read_text()
    for a, b in pairs:
        assert s.count(a) == 1, (p, a[:70], s.count(a))
        s = s.replace(a, b)
    f.write_text(s)


sub("ssr/env.py", [
 ('        self.gait = {}\n', '        self.gait = {}\n        self.cnt = {}                           # raw counts for the log (termination causes, touchdown support)\n'),
 # more per-step statistics for walkers
 ('                           ("steps_per_s", (contact & ~self.contact_prev).float().sum(-1) / X.CONTROL_DT)):\n',
  '                           ("steps_per_s", (contact & ~self.contact_prev).float().sum(-1) / X.CONTROL_DT),\n'
  '                           ("cmd_speed", self.speed.abs()), ("vx_mps", self.qvel[:, 0] * torch.sign(self.speed)),\n'
  '                           ("lin_err", (s["v_b"][:, :2] - self.cmd[:, :2]).norm(dim=-1)), ("yaw_rate_err", (s["w_b"][:, 2] - self.cmd[:, 2]).abs()),\n'
  '                           ("yaw_rate_cmd", self.cmd[:, 2].abs()), ("power_w", (s["tau"] * s["qd"]).abs().sum(-1)),\n'
  '                           ("foot_force_max_n", self.F[..., 2].abs().amax(-1))):\n'),
 ('        touchdown = contact & ~self.contact_prev & (self.air > 0.0)\n',
  '        touchdown = contact & ~self.contact_prev & (self.air > 0.0)\n'
  '        if self.tiles:                                               # support ratio at touchdown (paper Sec. 4.2: MSR, SFR), off slopes\n'
  '            td = touchdown & ~((self.type == T.SLOPE) | self.free)[:, None]\n'
  '            sr = 1.0 - self.rho_c\n'
  '            for k_, v_ in (("td_n", td.sum()), ("td_support", (sr * td).sum()), ("td_safe", ((sr > 0.75) & td).sum())):\n'
  '                self.cnt[k_] = self.cnt.get(k_, 0) + v_\n'),
 ('        rho_c = self.support_stance()\n', '        rho_c = self.support_stance()\n        self.rho_c = rho_c\n'),
 ('        if self.obst is not None:\n            term = term | self._touches_obstacle()\n',
  '        causes = dict(tilt=tilt, low=low, link_force=hit, clearance=scrape, diverged=self.bad)\n'
  '        if self.obst is not None:\n            causes["side_obstacle"] = self._touches_obstacle()\n            term = term | causes["side_obstacle"]\n'),
 ('            term = term | (back & ~self.free)\n            return term, success | free_out | over, success\n',
  '            term = term | (back & ~self.free)\n'
  '            causes.update(left_tile_side=out_y & ~self.free, left_tile_back=back & ~self.free, stuck=term & ~term if not self.stuck_term else stuck,\n'
  '                          reached_end=success, omni_left_tile=free_out, time_limit=over & ~success & ~free_out)\n'
  '            causes["any"] = term | success | free_out | over\n'
  '            for k_, v_ in causes.items():\n                self.cnt["done_" + k_] = self.cnt.get("done_" + k_, 0) + v_.sum()\n'
  '            return term, success | free_out | over, success\n'),
])

sub("ssr/train.py", [
 ('import argparse\nimport json\nimport math\nimport os\nimport pathlib\n', 'import argparse\nimport json\nimport math\nimport os\nimport pathlib\nimport subprocess\n'),
 ('WEIGHTS = torch.tensor([1.0, 0.25, 0.2])       # w_l, w_f, w_s\n',
  'WEIGHTS = torch.tensor([1.0, 0.25, 0.2])       # w_l, w_f, w_s\n'
  '# Table 6 weights, for the log: raw term = weighted term / weight\n'
  'TERM_W = dict(lin_vel=1.0, ang_vel=0.8, orientation=0.5, ang_vel_xy=0.25, base_height=0.4, action_rate=-0.12, smoothness=-0.06, dof_vel=-0.96, dof_torque=-0.6,\n'
  '              dof_dev=-1.8, dof_pos_lim=-1.5, dof_vel_lim=-6.0, dof_tau_lim=-6.0, stand_still=-0.12, single_support=0.2, impact=-1.3, slip=-0.2, air_time=-2.0,\n'
  '              stumble=-2.0, lateral=0.08)\n\n\n'
  'def gpu_now():\n'
  '    try:\n'
  '        o = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout\n'
  '        u, m = o.strip().split(",")\n        return int(u), int(m)\n'
  '    except Exception:\n        return -1, -1\n'),
 ('        env.air_bonus = 8.0 * env.boot\n        env.gait = {}\n',
  '        env.air_bonus = 8.0 * env.boot\n        env.gait = {}\n        env.cnt = {}\n'
  '        sty_sum, sty_n = torch.zeros(5, device=dev), torch.zeros(5, device=dev)\n'),
 ('                B["amp"][t], B["idx"][t] = env.amp_hist.flatten(1), idx\n',
  '                B["amp"][t], B["idx"][t] = env.amp_hist.flatten(1), idx\n'
  '                sty_sum.index_add_(0, idx, r_s)\n                sty_n.index_add_(0, idx, torch.ones_like(r_s))\n'),
 ('        d_real = d_fake = 0.0\n', '        d_real = d_fake = 0.0\n        disc_log = {}\n'),
 ('            d_real += dr.mean().item() / 5\n',
  '            disc_log[T.TYPES[i]] = dict(loss=round(loss.item(), 4), d_data=round(dr.mean().item(), 3), d_policy=round(df.mean().item(), 3), grad_pen=round(gp.item(), 5),\n'
  '                                        style_reward=round((sty_sum[i] / sty_n[i].clamp(min=1)).item(), 4), share=round((sty_n[i] / sty_n.sum()).item(), 3))\n'
  '            d_real += dr.mean().item() / 5\n'),
 ('            lv = [round(env.level[(env.type == c) & ~env.free].float().mean().item(), 2) for c in range(5)]\n',
  '            lv = [round(env.level[(env.type == c) & ~env.free].float().mean().item(), 2) for c in range(5)]\n'
  '            cnt = {k_: float(v_) for k_, v_ in env.cnt.items()}\n'
  '            n_done = max(cnt.get("done_any", 0.0), 1.0)\n'
  '            gu, gm = gpu_now()\n'
  '            extra = dict(terms_raw={k_: round(v_ / TERM_W[k_], 4) for k_, v_ in term_sum.items() if k_ in TERM_W},\n'
  '                         group_weighted=dict(locomotion=round(rs[0] * float(w[0]), 4), foothold=round(rs[1] * float(w[1]), 4), style=round(rs[2] * float(w[2]), 4)),\n'
  '                         disc=disc_log, foot_nll=round(float(foot_loss.detach()), 4), entropy_std=[round(x, 3) for x in actor.std().tolist()],\n'
  '                         done_share={k_[5:]: round(v_ / n_done, 4) for k_, v_ in cnt.items() if k_.startswith("done_") and k_ != "done_any"}, dones_per_iter=cnt.get("done_any", 0.0),\n'
  '                         foothold=dict(touchdowns=cnt.get("td_n", 0.0), msr=round(cnt.get("td_support", 0.0) / max(cnt.get("td_n", 0.0), 1.0), 4),\n'
  '                                       sfr=round(cnt.get("td_safe", 0.0) / max(cnt.get("td_n", 0.0), 1.0), 4)),\n'
  '                         levels_by_type=dict(zip(T.TYPES, lv)), level_max_share=round((env.level[~env.free] >= T.NLEVEL - 1).float().mean().item(), 4),\n'
  '                         samples_per_s=round(N_ * Tn / max(time.time() - t0, 1e-6)), num_envs=N_, gpu_util=gu, gpu_mem_mb=gm)\n'),
 ('gate_acc=round(gate_hit / max(gate_n, 1), 3), skipped=guard.get("skipped", 0),',
  'gate_acc=round(gate_hit / max(gate_n, 1), 3), skipped=guard.get("skipped", 0), **extra,'),
 ('            torch.save(ck, out / "last.tmp")\n            os.replace(out / "last.tmp", out / "last.pt")     # never a half-written last.pt\n',
  '            torch.save(ck, out / "last.tmp")\n'
  '            if (out / "last.pt").exists():\n                os.replace(out / "last.pt", out / "prev.pt")      # the newest two are always kept\n'
  '            os.replace(out / "last.tmp", out / "last.pt")     # never a half-written last.pt\n'),
 ('    ap.add_argument("--ckpt_every", type=int, default=500)\n', '    ap.add_argument("--ckpt_every", type=int, default=1000)\n'),
])
print("patched2")
