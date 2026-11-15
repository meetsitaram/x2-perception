"""Builds aclip/new/vecenv_wrapper.py from aclip/orig/vecenv_wrapper.py (Hiking X2 action clip, 2026-11-15)."""
import sys

src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()

old = """        # record step information
        obs_pack, rew, terminated, truncated, extras = self.env.step(actions)"""
new = """        # Hiking X2 action clip (2026-11-15, user: "Cap at 20 like others"; FINDINGS H7c): the clipped action is what the env
        # applies AND what last_action feeds back as observation; the runner keeps the sampled action for the PPO log-prob.
        # Off unless HIKING_X2_ACTION_CLIP is set (then this code path is the only change).
        if _ACLIP > 0.0:
            actions = self._aclip(actions)
        # record step information
        obs_pack, rew, terminated, truncated, extras = self.env.step(actions)"""
assert s.count(old) == 1
s = s.replace(old, new)

old2 = "from instinct_rl.env import VecEnv\n"
new2 = old2 + '''
import os as _os

_ACLIP = float(_os.environ.get("HIKING_X2_ACTION_CLIP", "0") or 0.0)
_ACLIP_EVERY = int(_os.environ.get("HIKING_X2_ACTION_CLIP_LOG_STEPS", "600"))  # 25 iterations x 24 steps
# optional per-joint bounds "index:bound,..." in action order (e.g. "5:50" = waist_pitch), others +-_ACLIP
_ACLIP_JOINT = {int(k): float(v) for k, v in (t.split(":") for t in _os.environ.get("HIKING_X2_ACTION_CLIP_JOINT", "").split(",") if t)}
'''
assert s.count(old2) == 1
s = s.replace(old2, new2)

old3 = "    def reset(self) -> tuple[torch.Tensor, dict]:  # noqa: D102"
new3 = '''    def _aclip(self, actions: torch.Tensor) -> torch.Tensor:
        """Clamp to +-_ACLIP; every _ACLIP_EVERY steps print the share of clipped values per joint and the |action| maxima."""
        a = actions.detach()
        if not hasattr(self, "_aclip_n"):
            self._aclip_n, self._aclip_steps, self._aclip_tot_n = 0, 0, 0
            self._aclip_cnt = torch.zeros(a.shape[1], device=a.device, dtype=torch.float64)
            self._aclip_tot_cnt = torch.zeros(a.shape[1], device=a.device, dtype=torch.float64)
            self._aclip_max = torch.zeros(a.shape[1], device=a.device)
            self._aclip_b = torch.full((a.shape[1],), _ACLIP, device=a.device, dtype=a.dtype)
            for j, b in _ACLIP_JOINT.items():
                self._aclip_b[j] = b
            print("[ACLIP] armed: bounds per joint %s" % [round(v, 1) for v in self._aclip_b.tolist()], flush=True)
        bind = (a.abs() > self._aclip_b).sum(0).double()
        self._aclip_cnt += bind
        self._aclip_tot_cnt += bind
        self._aclip_max = torch.maximum(self._aclip_max, torch.nan_to_num(a.abs(), nan=3.0e38, posinf=3.0e38).max(0).values)
        self._aclip_n += a.shape[0]
        self._aclip_tot_n += a.shape[0]
        self._aclip_steps += 1
        if self._aclip_steps % _ACLIP_EVERY == 0:
            sh = (100.0 * self._aclip_cnt / max(self._aclip_n, 1)).tolist()
            tot = (100.0 * self._aclip_tot_cnt / max(self._aclip_tot_n, 1)).tolist()
            print("[ACLIP] steps %d clip %.1f window robot-steps %d: clipped %% per joint %s; total %% %s; max|a| per joint %s"
                  % (self._aclip_steps, _ACLIP, self._aclip_n, [round(v, 4) for v in sh], [round(v, 4) for v in tot],
                     [round(v, 1) for v in self._aclip_max.tolist()]), flush=True)
            self._aclip_n = 0
            self._aclip_cnt.zero_()
            self._aclip_max.zero_()
        return torch.maximum(torch.minimum(actions, self._aclip_b), -self._aclip_b)

''' + old3
assert s.count(old3) == 1
s = s.replace(old3, new3)
open(dst, "w").write(s)
print("written", dst)
