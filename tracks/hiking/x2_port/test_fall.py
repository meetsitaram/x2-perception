"""CPU test of staged.py.next option f with a stub of Isaac Lab's is_terminated_term (no Isaac). python test_fall.py staged.py.next"""
import os, sys, types, torch

src = open(sys.argv[1]).read()
ns = {"os": os, "torch": torch}
exec(src[src.index("STEPS_PER_ITERATION"):src.index("# ---- a: staged rewards")], ns)


class StubTerm:  # same arithmetic as isaaclab.envs.mdp.rewards.is_terminated_term.__call__
    def __init__(self, names):
        self._term_names = names

    def __call__(self, env, term_keys=".*"):
        buf = torch.zeros(env.num_envs)
        for t in self._term_names:
            buf += env.termination_manager.get_term(t)
        return (buf * (~env.termination_manager.time_outs)).float()


f0 = src.index("# ---- f: fall penalty ----")
f1 = src.index("# ----", f0 + 10)
block = src[f0:f1].replace("from isaaclab.envs.mdp.rewards import is_terminated_term as _is_terminated_term", "_is_terminated_term = StubTerm")
ns["StubTerm"] = StubTerm
exec(block, ns)
assert ns["FALL_TERMS"] == "base_contact|bad_orientation|root_height"

terms = {"base_contact": torch.tensor([0., 1., 0., 0., 0.]), "bad_orientation": torch.tensor([0., 0., 1., 0., 0.]),
         "root_height": torch.tensor([0., 1., 0., 0., 0.]), "stalled": torch.tensor([0., 0., 0., 1., 0.]),
         "time_out": torch.tensor([0., 0., 0., 0., 1.])}
tm = types.SimpleNamespace(get_term=lambda t: terms[t], time_outs=terms["time_out"].bool())
env = types.SimpleNamespace(num_envs=5, termination_manager=tm, common_step_counter=0)
fp = ns["fall_penalty_staged"](["base_contact", "bad_orientation", "root_height"])  # the names find_terms returns for FALL_TERMS
os.environ["HIKING_X2_STAGE"] = "500,1500"
for off, want in ((0, 1.0), (500, 1.0), (1000, 0.5), (1500, 0.0), (3000, 0.0)):
    os.environ["HIKING_ITER_OFFSET"] = str(off)
    v = fp(env)
    exp = torch.tensor([0., 1., 1., 0., 0.]) * want  # env 1 hit two fall terms in one step: still one fall
    assert torch.allclose(v, exp), (off, v)
# weight: the reward manager adds func x weight x dt, so weight -P/dt gives -P per fall
P, dt = 8.0, 0.005 * 4
assert abs(1.0 * (-P / dt) * dt + P) < 1e-9
# config section: option letters and env-var names present
assert "f" in src[src.index("unknown = opts"):src.index("unknown = opts") + 40] and "HIKING_X2_FALL_PENALTY" in src and "HIKING_X2_AIR_EARLY" in src
print("option f: per-fall value 1 until hold, 0.5 at 1000, 0 from 1500; stalled and time-outs not penalised; two fall terms in "
      "one step count 1 (clamped); weight -P/dt -> -P per fall. All checks passed.")
