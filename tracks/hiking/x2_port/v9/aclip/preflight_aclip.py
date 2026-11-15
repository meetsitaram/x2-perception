"""CPU pre-flight of the action clip in vecenv_wrapper.py (2026-11-15). isaaclab / instinct_rl stubbed; a fake env whose obs is
the last applied action (as last_action) and which records what it applies. Linear feedback a_t = 1.6 a_{t-1} + noise = runaway."""
import os, sys, types, importlib.util, torch
for m in ["isaaclab", "isaaclab.envs", "instinct_rl", "instinct_rl.env"]:
    sys.modules[m] = types.ModuleType(m)
class _E: pass
sys.modules["isaaclab.envs"].DirectRLEnv = _E; sys.modules["isaaclab.envs"].ManagerBasedRLEnv = _E
sys.modules["isaaclab.envs"].ManagerBasedRLEnvCfg = _E
sys.modules["instinct_rl.env"].VecEnv = object
def load(path, env):
    for k, v in env.items(): os.environ[k] = v
    for k in ["HIKING_X2_ACTION_CLIP", "HIKING_X2_ACTION_CLIP_JOINT", "HIKING_X2_ACTION_CLIP_LOG_STEPS"]:
        if k not in env: os.environ.pop(k, None)
    spec = importlib.util.spec_from_file_location("w%d" % len(env), path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
class FakeEnv(_E):
    def __init__(s, n, d): s.n, s.d, s.applied = n, d, []; s.unwrapped = s
    def step(s, a):
        s.applied.append(a.clone()); return {"policy": a.clone()}, torch.zeros(s.n), torch.zeros(s.n, dtype=torch.bool), torch.zeros(s.n, dtype=torch.bool), {}
def run(mod, acts):
    W = mod.InstinctRlVecEnvWrapper; w = W.__new__(W); w.env = FakeEnv(*acts[0].shape)
    w._flatten_all_obs_groups = lambda o: o
    w.unwrapped.cfg = types.SimpleNamespace(is_finite_horizon=True)
    outs = [w.step(a)[0] for a in acts]; return torch.stack(w.env.applied), torch.stack(outs)
orig, new = sys.argv[1], sys.argv[2]
g = torch.Generator().manual_seed(0)
normal = [torch.randn(4096, 29, generator=g) * 3.0 for _ in range(50)]
normal[7][3, 5] = -19.9
ok = True
for name, env in [("orig", {}), ("new_off", {}), ("new_clip20_neverbinds", {"HIKING_X2_ACTION_CLIP": "20", "HIKING_X2_ACTION_CLIP_LOG_STEPS": "25"}),
                  ("new_clip20_j5_50", {"HIKING_X2_ACTION_CLIP": "20", "HIKING_X2_ACTION_CLIP_JOINT": "5:50", "HIKING_X2_ACTION_CLIP_LOG_STEPS": "25"})]:
    m = load(orig if name == "orig" else new, env); ap, ob = run(m, normal)
    if name == "orig": ref = (ap, ob)
    same = torch.equal(ap, ref[0]) and torch.equal(ob, ref[1]); ok &= same
    print(name, "normal rollouts bit-identical to orig:", same, "max|a| %.2f" % ap.abs().max())
# forced runaway: robot 0 joint 5 grows x1.6 per step from 1.0 (policy fed its own last action)
for name, env, bound in [("clip20", {"HIKING_X2_ACTION_CLIP": "20", "HIKING_X2_ACTION_CLIP_LOG_STEPS": "10"}, 20.0),
                         ("clip20_j5_50", {"HIKING_X2_ACTION_CLIP": "20", "HIKING_X2_ACTION_CLIP_JOINT": "5:50", "HIKING_X2_ACTION_CLIP_LOG_STEPS": "10"}, 50.0)]:
    m = load(new, env); W = m.InstinctRlVecEnvWrapper; w = W.__new__(W); w.env = FakeEnv(8, 29); w._flatten_all_obs_groups = lambda o: o
    w.unwrapped.cfg = types.SimpleNamespace(is_finite_horizon=True)
    last = torch.zeros(8, 29); last[0, 5] = 1.0; sampled_max = 0.0
    for t in range(60):
        a = 1.6 * last + 0.0; sampled_max = max(sampled_max, a.abs().max().item())
        last = w.step(a)[0]
    applied = torch.stack(w.env.applied)
    b = applied.abs().max().item(); ok &= b <= bound and torch.isfinite(applied).all().item()
    print(name, "forced runaway: policy output would reach %.3g in 60 steps; applied / fed back max %.1f (bound %.0f); first clipped step %d"
          % (1.6 ** 60, b, bound, int((applied[:, 0, 5].abs() >= bound).nonzero()[0])))
print("PREFLIGHT", "PASS" if ok else "FAIL")
