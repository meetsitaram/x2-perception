# CPU pre-flight of the rollout guard (FINDINGS H7b): original vs patched instinct_rl ppo.py / wasabi.py.
# usage: preflight_guard2.py <orig dir with ppo.py, wasabi.py> <new dir>
import sys, importlib.util, types, io, contextlib, torch, torch.nn as nn
import instinct_rl.algorithms  # installed package (originals)
od, nd = sys.argv[1], sys.argv[2]

def load(path, name):
    spec = importlib.util.spec_from_file_location("instinct_rl.algorithms." + name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m; spec.loader.exec_module(m); return m

def load_pair(d, tag):
    ppo = load(d + "/ppo.py", "ppo_" + tag)
    saved = sys.modules.get("instinct_rl.algorithms.ppo")
    sys.modules["instinct_rl.algorithms.ppo"] = ppo  # wasabi's `from .ppo import PPO` gets this copy
    was = load(d + "/wasabi.py", "wasabi_" + tag)
    sys.modules["instinct_rl.algorithms.ppo"] = saved
    return ppo, was
PO, WO = load_pair(od, "orig"); PN, WN = load_pair(nd, "new")

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

class St:  # rollout storage stub: minibatches plus the full rollout tensors the guard reads
    def __init__(s, mbs, kind=None):
        g = torch.Generator().manual_seed(7)
        s.mbs = mbs; s.returns = torch.randn(4, 5, 1, generator=g); s.values = torch.randn(4, 5, 1, generator=g)
        s.observations = torch.randn(4, 5, 6, generator=g); s.critic_observations = torch.randn(4, 5, 8, generator=g)
        s.actions = torch.randn(4, 5, 3, generator=g); s.rewards = torch.randn(4, 5, 1, generator=g)
        if kind == "obs": s.critic_observations[3, 2, 5] = 1.1e8
        if kind == "action": s.actions[1, 4, 0] = -2e5
        if kind == "reward_nan": s.rewards[2, 1, 0] = float("nan")
        s.cleared = 0
    def mini_batch_generator(s, nm, ne):
        for m in s.mbs: yield m
    def clear(s): s.cleared += 1
    def compute_returns(s, *a): pass

def make(cls, mbs, kind=None, seed=0):
    torch.manual_seed(seed); T = stub(cls); p = T.__new__(T)
    p.actor_critic = AC(); p.optimizer = torch.optim.Adam(p.actor_critic.parameters(), lr=1e-3)
    p.storage = St(mbs, kind); p.num_mini_batches = len(mbs); p.num_learning_epochs = 1
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
quiet = lambda: contextlib.redirect_stdout(io.StringIO())

# 1. normal rollouts: bit-identical parameters and losses over 3 updates; only logging keys added
mbs = [mb(i) for i in range(4)]
po, pn = make(PO.PPO, mbs), make(PN.PPO, mbs)
lo = [po.update(i) for i in range(3)]; ln = [pn.update(i) for i in range(3)]
res["normal_params_bit_identical"] = same(params(po), params(pn))
res["normal_losses_identical"] = all(set(a[0]) == set(b[0]) and all(torch.equal(a[0][k], b[0][k]) for k in a[0]) for a, b in zip(lo, ln))
res["normal_shared_stats_identical"] = all(torch.equal(torch.as_tensor(a[1][k]), torch.as_tensor(b[1][k])) for a, b in zip(lo, ln) for k in a[1])
res["normal_extra_stats"] = sorted(set(ln[0][1]) - set(lo[0][1]))
res["normal_rollout_skipped_stat"] = float(ln[0][1]["guard_rollout_skipped_max"])

# 2. out-of-range rollouts: patched leaves every parameter unchanged and logs; original trains on them
for kind in ("obs", "action", "reward_nan"):
    po_b, pn_b = make(PO.PPO, mbs, kind), make(PN.PPO, mbs, kind)
    before = params(pn_b); po_b.update(0); buf = io.StringIO()
    with contextlib.redirect_stdout(buf): out = pn_b.update(0)
    res[kind + "_patched_params_unchanged"] = same(before, params(pn_b))
    res[kind + "_patched_loss_keys"] = sorted(out[0].keys())
    res[kind + "_patched_skipped_stat"] = float(out[1]["guard_rollout_skipped_max"])
    res[kind + "_storage_cleared"] = pn_b.storage.cleared
    res[kind + "_log"] = buf.getvalue().strip()[:260]
# 3. three out-of-range rollouts in a row raise (-> supervisor resumes from the newest checkpoint); a normal one resets
pa = make(PN.PPO, mbs, "obs"); seq = []
with quiet():
    for i in range(3):
        try: pa.update(i); seq.append("skip")
        except RuntimeError as e: seq.append("raised: " + str(e))
res["three_in_a_row"] = seq
pr = make(PN.PPO, mbs, "obs"); seq = []
with quiet():
    for i, k in enumerate(["obs", "obs", None, "obs", "obs"]):
        pr.storage = St(mbs, k)
        try: pr.update(i); seq.append("ok" if k is None else "skip")
        except RuntimeError: seq.append("raised")
res["streak_reset_by_normal"] = seq
# 4. the minibatch guard (H7) still works: a NaN minibatch is skipped, others applied
bad = [mb(0), mb(1, "nan"), mb(2), mb(3)]
pn_m, ref = make(PN.PPO, bad), make(PO.PPO, [mb(0), mb(2), mb(3)])
with quiet(): pn_m.update(0)
ref.update(0)
res["minibatch_guard_equals_orig_without_bad"] = same(params(pn_m), params(ref))

# 5. Wasabi: normal path identical; skipped rollout -> no discriminator step, AMP storage cleared, keys present
def wstub(Wcls, Pmod, kind):
    base = stub(Pmod.PPO)
    class W(Wcls, base):
        pass
    # MRO: W -> Wasabi -> stub(PPO) ... -> PPO; Wasabi.update's super() reaches the stubbed compute_losses
    w = W.__new__(W); p = make(Pmod.PPO, mbs, kind)
    w.__dict__.update(p.__dict__)
    calls = {"amp": 0, "step": 0, "clear": 0}
    w.discriminator = types.SimpleNamespace(is_recurrent=False, normalizer=None)
    class AS:
        def mini_batch_generator(s, nm, ne):
            for i in range(nm * ne): yield i
        def clear(s): calls["clear"] += 1
    w.amp_storage = AS()
    def cal(mbi):
        calls["amp"] += 1
        return dict(discriminator_loss=torch.tensor(0.5 + mbi), discriminator_gradient_penalty=torch.tensor(0.1)), None, dict(disc_acc=torch.tensor(0.7))
    w.compute_amp_losses = cal
    def gs(loss, st): calls["step"] += 1
    w.wasabi_gradient_step = gs
    return w, calls
try:
    wo, co = wstub(WO.WasabiPPO if hasattr(WO, "WasabiPPO") else [v for v in vars(WO).values() if isinstance(v, type) and issubclass(v, PO.PPO) and v is not PO.PPO][0], PO, None)
    WNc = WN.WasabiPPO if hasattr(WN, "WasabiPPO") else [v for v in vars(WN).values() if isinstance(v, type) and issubclass(v, PN.PPO) and v is not PN.PPO][0]
    wn, cn = wstub(WNc, PN, None)
    a = wo.update(0); b = wn.update(0)
    res["wasabi_normal_losses_identical"] = set(a[0]) == set(b[0]) and all(torch.equal(torch.as_tensor(a[0][k]), torch.as_tensor(b[0][k])) for k in a[0])
    res["wasabi_normal_params_identical"] = same(params(wo), params(wn))
    res["wasabi_normal_calls"] = (co, cn)
    wb, cb = wstub(WNc, PN, "obs"); wb._guard_amp_keys = wn._guard_amp_keys
    with quiet(): out = wb.update(0)
    res["wasabi_skip_calls"] = cb
    res["wasabi_skip_keys"] = sorted(out[0].keys())
except Exception as e:
    import traceback; traceback.print_exc(); res["wasabi_error"] = repr(e)
for k, v in res.items(): print(k, v)
ok = (res["normal_params_bit_identical"] and res["normal_losses_identical"] and res["normal_shared_stats_identical"]
      and res["normal_rollout_skipped_stat"] == 0.0
      and all(res[k + "_patched_params_unchanged"] and res[k + "_patched_skipped_stat"] == 1.0 and res[k + "_storage_cleared"] == 1
              for k in ("obs", "action", "reward_nan"))
      and res["three_in_a_row"][:2] == ["skip", "skip"] and str(res["three_in_a_row"][2]).startswith("raised")
      and res["streak_reset_by_normal"] == ["skip", "skip", "ok", "skip", "skip"]
      and res["minibatch_guard_equals_orig_without_bad"]
      and "wasabi_error" not in res and res["wasabi_normal_losses_identical"] and res["wasabi_normal_params_identical"]
      and res["wasabi_normal_calls"][0] == res["wasabi_normal_calls"][1] and res["wasabi_skip_calls"]["amp"] == 0
      and res["wasabi_skip_calls"]["step"] == 0 and res["wasabi_skip_calls"]["clear"] == 1
      and set(res["wasabi_normal_losses_identical"] and b[0].keys()) <= set(res["wasabi_skip_keys"]))
print("PREFLIGHT", "PASS" if ok else "FAIL")
