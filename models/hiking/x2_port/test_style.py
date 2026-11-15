"""CPU unit test of option s (staged.py's install_style_schedule) on a stand-in algorithm class:
    python test_style.py <path to staged.py>"""
import os, sys, torch
src = open(sys.argv[1]).read()
ns = {"os": os, "torch": torch}
exec(src[src.index("STEPS_PER_ITERATION"): src.index("# ---- a: staged rewards")], ns)
exec(src[src.index("# ---- s: style"): src.index("# ---- end s")], ns)


def make():
    class Algo:
        discriminator_reward_coef = 0.25

        def process_env_step(self, rewards, dones, infos, next_obs, next_extra_obs):
            v = torch.full((4,), 0.6)
            self.seen = self.discriminator_reward_coef
            infos["step"]["discriminator_reward"] = v
            self.added = self.discriminator_reward_coef * v
    return Algo


os.environ["HIKING_X2_STAGE"] = "500,1500"
for offset, early, want in [("0", 0.3, 0.075), ("1000", 0.3, 0.1625), ("1500", 0.3, 0.25), ("5000", 0.3, 0.25), ("0", 1.0, 0.25)]:
    os.environ["HIKING_ITER_OFFSET"] = offset
    A = make()
    ns["install_style_schedule"](A, early)
    ns["install_style_schedule"](A, 0.0)  # second install is ignored
    a = A()
    infos = {"step": {}}
    a.process_env_step(None, None, infos, None, None)
    assert abs(a.seen - want) < 1e-9, (offset, early, a.seen, want)
    assert abs(float(infos["step"]["style_coef"]) - want) < 1e-6
    assert torch.allclose(infos["step"]["style_reward_weighted"], torch.full((4,), 0.6 * want))
    assert torch.allclose(a.added, torch.full((4,), 0.6 * want))
# the schedule advances with the step count: 24 steps = 1 iteration; 600 iterations from offset 0 -> scale 0.3 + 0.7 * 0.1
os.environ["HIKING_ITER_OFFSET"] = "0"
A = make(); ns["install_style_schedule"](A, 0.3); a = A()
for _ in range(600 * 24 + 1):
    infos = {"step": {}}
    a.process_env_step(None, None, infos, None, None)
assert abs(a.seen - 0.25 * (0.3 + 0.7 * 0.1)) < 1e-9, a.seen
print("option s: all checks passed (coef 0.075 early, 0.1625 at 1000, 0.25 from 1500; 0.25 always without s; per-step log keys present)")
