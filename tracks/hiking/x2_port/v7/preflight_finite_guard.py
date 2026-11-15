# Pre-flight of the PPO finite guard on CPU: original vs patched ppo.py.
import sys, copy, importlib.util, types, torch, torch.nn as nn, io, contextlib
orig_path, new_path = sys.argv[1], sys.argv[2]
def load(path, name):
    spec = importlib.util.spec_from_file_location("instinct_rl.algorithms." + name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m; spec.loader.exec_module(m); return m
O = load(orig_path, "ppo_pf_orig").PPO; N = load(new_path, "ppo_pf_new").PPO

class AC(nn.Module):
    is_recurrent = False
    def __init__(s):
        super().__init__(); s.a = nn.Linear(6, 3); s.c = nn.Linear(8, 1); s.std = nn.Parameter(torch.ones(3))
    def clip_std(s, min=None): s.std.data.clamp_(min=min)
def stub(cls):
    class T(cls):
        def compute_losses(self, mb):
            mu = self.actor_critic.a(mb.obs); v = self.actor_critic.c(mb.critic_obs)
            sur = ((mu - mb.actions) ** 2).mean() * self.actor_critic.std.mean()
            vl = (v - mb.returns).pow(2).reshape(-1, 1).mean(dim=0)
            return dict(surrogate_loss=sur, value_loss=vl.mean()), {}, dict(ratio_max=sur.detach())
    return T
class St:
    def __init__(s, mbs): s.mbs = mbs; s.returns = torch.randn(4, 5, 1); s.rewards = torch.randn(4, 5, 1); s.values = torch.randn(4, 5, 1)
    def mini_batch_generator(s, nm, ne):
        for m in s.mbs: yield m
    def clear(s): pass
    def compute_returns(s, *a): pass
def make(cls, mbs, seed=0):
    torch.manual_seed(seed); p = stub(cls).__new__(stub(cls))
    p.actor_critic = AC(); p.optimizer = torch.optim.Adam(p.actor_critic.parameters(), lr=1e-3)
    p.storage = St(mbs); p.num_mini_batches = len(mbs); p.num_learning_epochs = 1
    p.max_grad_value = None; p.max_grad_norm = 1.0; p.gamma = 0.99; p.lam = 0.95; p.device = torch.device("cpu"); p.clip_min_std = 1e-12
    return p
def mb(seed, bad=False):
    g = torch.Generator().manual_seed(seed)
    m = types.SimpleNamespace(obs=torch.randn(32, 6, generator=g), critic_obs=torch.randn(32, 8, generator=g),
                              actions=torch.randn(32, 3, generator=g), returns=torch.randn(32, 1, generator=g), values=torch.randn(32, 1, generator=g))
    if bad == "nan": m.returns[3, 0] = float("nan")
    if bad == "big": m.critic_obs[5, 2] = 1e7
    return m
def params(p): return [x.detach().clone() for x in p.actor_critic.parameters()]
def same(a, b): return all(torch.equal(x, y) for x, y in zip(a, b))
res = {}
# 1. normal minibatches: bit-identical params and losses over 3 updates
mbs = [mb(i) for i in range(4)]
po, pn = make(O, mbs), make(N, mbs)
lo = [po.update(i) for i in range(3)]; ln = [pn.update(i) for i in range(3)]
res["normal_params_bit_identical"] = same(params(po), params(pn))
res["normal_losses_identical"] = all(torch.equal(a[0][k], b[0][k]) for a, b in zip(lo, ln) for k in a[0])
res["normal_only_extra_stat"] = sorted(set(ln[0][1]) - set(lo[0][1]))
res["normal_guard_stat"] = float(ln[0][1]["guard_skipped_minibatches_max"])
# 2. forced non-finite value loss: patched skips it (= original without that minibatch); original gets NaN weights
for kind in ("nan", "big"):
    bad = [mb(0), mb(1, kind), mb(2), mb(3)]
    po_b, pn_b, ref = make(O, bad), make(N, bad), make(O, [mb(0), mb(2), mb(3)])
    po_b.update(0)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): ln_b = pn_b.update(0)
    ref.update(0)
    res[kind + "_patched_equals_orig_without_bad"] = same(params(pn_b), params(ref))
    res[kind + "_patched_finite"] = all(torch.isfinite(x).all().item() for x in params(pn_b))
    res[kind + "_orig_finite"] = all(torch.isfinite(x).all().item() for x in params(po_b))
    res[kind + "_skipped_stat"] = float(ln_b[1]["guard_skipped_minibatches_max"])
    res[kind + "_log"] = buf.getvalue().strip().splitlines()[0][:230]
# 3. every minibatch bad: losses defined, raises after 3 full skips
pa = make(N, [mb(1, "nan"), mb(2, "nan")]); out = []
with contextlib.redirect_stdout(io.StringIO()):
    for i in range(3):
        try: out.append(sorted(pa.update(i)[0].keys()))
        except RuntimeError as e: out.append("raised: " + str(e))
res["all_bad"] = out
# 4. returns clamp: identity in range, clamps beyond the bound
p = make(N, mbs); p.actor_critic.evaluate = lambda x: torch.zeros(5, 1)
r0 = p.storage.returns.clone(); p.compute_returns({"policy": torch.zeros(5, 6)}); res["returns_in_range_identical"] = torch.equal(r0, p.storage.returns)
p.storage.returns[0, 0, 0] = 5e5; buf = io.StringIO()
with contextlib.redirect_stdout(buf): p.compute_returns({"policy": torch.zeros(5, 6)})
res["returns_clamped_to"] = float(p.storage.returns.max()); res["returns_log"] = buf.getvalue().strip()[:160]
for k, v in res.items(): print(k, v)
ok = (res["normal_params_bit_identical"] and res["normal_losses_identical"] and res["nan_patched_equals_orig_without_bad"]
      and res["big_patched_equals_orig_without_bad"] and not res["nan_orig_finite"] and res["returns_in_range_identical"]
      and res["returns_clamped_to"] == 1e4 and str(res["all_bad"][-1]).startswith("raised"))
print("PREFLIGHT", "PASS" if ok else "FAIL")
