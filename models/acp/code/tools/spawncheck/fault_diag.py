# CPU diagnosis of the stage-3 simulator faults (ACP, 2026-11-14). Runs acp.train unchanged, wrapping two things only:
# AcpEnv.reset (goal envs get a uniform random curriculum level 0..9, so all levels are seen) and the FaultDetector.update
# of the env (records, per control step, which cause fired, in goal or reference envs, at which episode step, family, level,
# and the contact depth). Output: $DIAG_OUT (json) at exit.
import os, sys, json, atexit, collections
import torch
sys.path.insert(0, os.getcwd())
sys.argv = [sys.argv[0]] + sys.argv[1:]
import acp.env as E
import acp.train as T

OUT = os.environ.get("DIAG_OUT", "fault_diag.json")
stats = collections.defaultdict(float)
hist = collections.defaultdict(float)
_orig_reset = E.AcpEnv.reset
_orig_init = E.AcpEnv.__init__

def reset(self, ids):
    if len(ids) and hasattr(self, "goal_env") and bool(self.goal_env[ids].any()) and os.environ.get("DIAG_RANDOM_LEVEL", "1") == "1":
        g = ids[self.goal_env[ids]]
        self.g_level[g] = torch.randint(0, 10, (len(g),), device=self.g_level.device)
    _orig_reset(self, ids)

def init(self, *a, **k):
    _orig_init(self, *a, **k)
    det = self.fault_det
    _upd = det.update
    env = self
    def update(qpos, qvel, xpos=None, depth=None):
        fault, why = _upd(qpos, qvel, xpos, depth)
        gm = env.goal_env
        st = env.ep_step
        stats["steps_goal"] += float(gm.sum()); stats["steps_ref"] += float((~gm).sum())
        for kind, sel in (("goal", gm), ("ref", ~gm)):
            stats[f"fault_{kind}"] += float((fault & sel).sum())
            for c, w in why.items():
                stats[f"{c}_{kind}"] += float((w & sel).sum())
        if bool(fault.any()):
            ids = fault.nonzero().squeeze(-1)
            for i in ids.tolist():
                kind = "goal" if bool(gm[i]) else "ref"
                causes = "+".join(c for c, w in why.items() if bool(w[i]))
                b = int(st[i])
                hist[f"{kind}|epstep_bin|{min(b // 10, 30) * 10}"] += 1
                hist[f"{kind}|cause|{causes}"] += 1
                if depth is not None:
                    dv = float(depth[i]); hist[f"{kind}|depth_bin_cm|{min(int(dv * 100 // 5) * 5, 100)}"] += 1
                if kind == "goal":
                    fam = env.data.goal_families[int(env.g_fam[i])]
                    hist[f"goal|fam|{fam}"] += 1
                    hist[f"goal|level|{int(env.g_level[i])}"] += 1
        # goal steps per family/level for normalisation
        if bool(gm.any()):
            for fi in torch.unique(env.g_fam[gm]).tolist():
                hist[f"goalsteps|fam|{env.data.goal_families[fi]}"] += float((gm & (env.g_fam == fi)).sum())
            for lv in torch.unique(env.g_level[gm]).tolist():
                hist[f"goalsteps|level|{lv}"] += float((gm & (env.g_level == lv)).sum())
        if depth is not None:
            for kind, sel in (("goal", gm), ("ref", ~gm)):
                if bool(sel.any()):
                    stats[f"depth_max_{kind}"] = max(stats[f"depth_max_{kind}"], float(depth[sel].max()))
        return fault, why
    det.update = update

E.AcpEnv.reset = reset
E.AcpEnv.__init__ = init

def dump():
    json.dump({"stats": dict(stats), "hist": dict(sorted(hist.items()))}, open(OUT, "w"), indent=1)
atexit.register(dump)
T.main()
