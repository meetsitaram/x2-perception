# Build the rollout guard (FINDINGS H7b, hiking track 2026-11-15) into copies of instinct_rl ppo.py and wasabi.py.
# usage: make_guard2.py <ppo.py in> <wasabi.py in> <out dir>
import sys, os
ppo_in, was_in, out = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)
p = open(ppo_in).read()
w = open(was_in).read()
assert "_guard_rollout_bad" not in p, "already patched"

def rep(s, old, new, n=1):
    assert s.count(old) == n, (old[:80], s.count(old))
    return s.replace(old, new)

# 1. update(): rollout check first; skip the whole update when out of range
p = rep(p, """    def update(self, current_learning_iteration):
        self.current_learning_iteration = current_learning_iteration
""", """    def update(self, current_learning_iteration):
        self.current_learning_iteration = current_learning_iteration
        # Rollout guard (hiking track 2026-11-15, FINDINGS H7b): one runaway robot (its actions, and their history in the
        # observations, growing ~1.6x per step to 1e8) dominates the batch-wide advantage normalisation, so updates whose
        # value loss stays below guard_value_loss_max still train on it (v9 collapse at 22,113-22,115). If any rollout tensor
        # is non-finite or beyond its bound, the whole update (PPO and, in Wasabi, the discriminator) is skipped. Read-only
        # when every value is in range.
        self._guard_rollout_skip = self._guard_rollout_bad(current_learning_iteration)
        if self._guard_rollout_skip:
            return self._guard_skip_update(current_learning_iteration)
""")
# 2. normal path: per-iteration rollout maxima as stats, remember the loss keys
p = rep(p, """            average_stats[k] = average_stats[k] / num_updates
        self.storage.clear()
""", """            average_stats[k] = average_stats[k] / num_updates
        for k, v in self._guard_rollout_stats.items():
            average_stats["guard_rollout_%s_absmax_max" % k] = torch.tensor(v, device=self.device)
        average_stats["guard_rollout_skipped_max"] = torch.zeros((), device=self.device)
        self._guard_loss_keys = list(mean_losses.keys())
        self.storage.clear()
""")
# 3. settings and helpers
p = rep(p, """    _guard_streak = 0

    def _guard_bad(""", """    _guard_streak = 0
    # Rollout guard settings (FINDINGS H7b). Normal values: |obs| and |critic obs| are scaled states, depth in metres and
    # policy actions (O(10)); per-step rewards O(1). The runaway reached 1e8 / 5e11.
    guard_rollout_obs_abs_max = 1.0e4
    guard_rollout_action_abs_max = 1.0e3
    guard_rollout_reward_abs_max = 1.0e4
    _guard_rollout_skip = False
    _guard_rollout_stats = {}
    _guard_loss_keys = None

    def _guard_rollout_bad(self, it):
        s = self.storage
        groups = (("obs", getattr(s, "observations", None), self.guard_rollout_obs_abs_max),
                  ("critic_obs", getattr(s, "critic_observations", None), self.guard_rollout_obs_abs_max),
                  ("action", getattr(s, "actions", None), self.guard_rollout_action_abs_max),
                  ("reward", getattr(s, "rewards", None), self.guard_rollout_reward_abs_max))
        stats, bad = {}, []
        with torch.no_grad():
            for name, x, bound in groups:
                if not torch.is_tensor(x) or x.numel() == 0:
                    continue
                lo, hi = torch.aminmax(x)
                m = float(torch.maximum(hi, -lo))  # NaN propagates
                stats[name] = m
                if not m <= bound:
                    bad.append((name, x, bound))
            self._guard_rollout_stats = stats
            if bad:
                name, x, bound = bad[0]
                a = torch.nan_to_num(x.detach().float(), nan=3.0e38).abs()
                a = a.reshape(a.shape[0], a.shape[1], -1)
                per_env = a.amax(dim=(0, 2))
                top = torch.topk(per_env, k=min(5, per_env.numel()))
                e0 = int(top.indices[0])
                over_t = (a[:, e0, :].amax(dim=1) > bound).nonzero().flatten()
                cols = torch.topk(a.amax(dim=(0, 1)), k=min(5, a.shape[2]))
                print("[GUARD] iteration %s: rollout out of range in %s; maxima %s; %d robots beyond %.0g, top robots %s;"
                      " first step out of range for robot %d: %s of %d; top columns %s; whole update skipped (PPO and discriminator)"
                      % (it, [b[0] for b in bad], {k: float("%.4g" % v) for k, v in stats.items()},
                         int((per_env > bound).sum()), bound,
                         [(int(i), float("%.4g" % v)) for v, i in zip(top.values, top.indices)], e0,
                         int(over_t[0]) if over_t.numel() else -1, a.shape[0],
                         [(int(i), float("%.4g" % v)) for v, i in zip(cols.values, cols.indices)]), flush=True)
        return bool(bad)

    def _guard_skip_update(self, it):
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_losses = defaultdict(float)
        for k in (self._guard_loss_keys or ["surrogate_loss", "value_loss", "entropy", "total_loss"]):
            mean_losses[k] = torch.zeros((), device=self.device)
        average_stats = defaultdict(float)
        average_stats["guard_skipped_minibatches_max"] = torch.tensor(float(num_updates), device=self.device)
        for k, v in self._guard_rollout_stats.items():
            average_stats["guard_rollout_%s_absmax_max" % k] = torch.tensor(min(v, 3.0e38), device=self.device)
        average_stats["guard_rollout_skipped_max"] = torch.ones((), device=self.device)
        self.storage.clear()
        self._guard_streak = self._guard_streak + 1
        if self._guard_streak >= self.guard_max_full_skips:
            raise RuntimeError("[GUARD] update skipped in %d iterations in a row (rollout out of range)" % self._guard_streak)
        return mean_losses, average_stats

    def _guard_bad(""")
open(os.path.join(out, "ppo.py"), "w").write(p)

# Wasabi: no discriminator step on a skipped rollout; remember the AMP loss keys
assert "_guard_rollout_skip" not in w
w = rep(w, """    def update(self, *args, **kwargs):
        mean_losses, average_stats = super().update(*args, **kwargs)
""", """    def update(self, *args, **kwargs):
        mean_losses, average_stats = super().update(*args, **kwargs)
        if getattr(self, "_guard_rollout_skip", False):  # rollout guard (FINDINGS H7b): no discriminator step either
            for k in (self._guard_amp_keys or ["discriminator_loss", "discriminator_gradient_penalty",
                                                "discriminator_weight_decay", "logit_weight_decay", "amp_total_loss"]):
                mean_losses[k] = torch.zeros((), device=self.device)
            self.amp_storage.clear()
            return mean_losses, average_stats
        amp_keys = None
""")
w = rep(w, """            self.wasabi_gradient_step(loss, average_stats)

        if self.discriminator.normalizer is not None:""", """            self.wasabi_gradient_step(loss, average_stats)
            amp_keys = list(losses.keys())

        if amp_keys:
            self._guard_amp_keys = amp_keys + ["amp_total_loss"]
        if self.discriminator.normalizer is not None:""")
w = rep(w, """    def state_dict(self):""", """    _guard_amp_keys = None

    def state_dict(self):""")
open(os.path.join(out, "wasabi.py"), "w").write(w)
print("built", out)
