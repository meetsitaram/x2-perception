"""Stage-3 goal tiles, closed loop on the CPU: a checkpoint's tracker + its online planner (planner contacts only) on the
goal environments, every family at the given levels, deterministic actions, no noise. Per family and level: success
(final waypoint reached and held 1 s), falls, waypoints reached, closest approach. A plumbing check of tiles, goal
command and waypoint logic, and the starting point of stage 3; not a result.

    CUDA_VISIBLE_DEVICES= python tools/goal_check.py CKPT DATA_CFG OUT.json [EPISODES_PER_FAMILY_LEVEL=4] [LEVELS=0,4,9]
"""
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.environ.get("ACP_CODE", "/mnt/filesystem-g4/tracks/acp/code"))
sys.path.insert(0, "/mnt/filesystem-g4/tracks/acp/code/tools")
from acp import env as E  # noqa: E402
from acp import planner as P  # noqa: E402
from acp.data import Dataset  # noqa: E402
from acp.train import ActorCritic  # noqa: E402
from handover import cpu_physics  # noqa: E402


def main():
    ckpt, cfg, out = sys.argv[1:4]
    per = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    levels = [int(x) for x in (sys.argv[5] if len(sys.argv) > 5 else "0,4,9").split(",")]
    torch.set_num_threads(6)
    dev = torch.device("cpu")
    data = Dataset(cfg, dev, cache=os.path.join(os.path.dirname(cfg), "labels_" + os.path.basename(cfg) + ".pt"))
    ck = torch.load(ckpt, map_location=dev)
    F = len(data.goal_families)
    N = F * len(levels) * per
    E.AcpEnv._physics = cpu_physics
    env = E.AcpEnv(data, N, eval_mode=True, njmax=450)
    env.setup_goals(1.0)
    i = torch.arange(N)
    env.g_fam = (i // per) % F
    lv = torch.tensor(levels)[(i // (per * F)) % len(levels)]
    env.g_level = lv.clone()
    ac = ActorCritic(ck["actor_dim"], ck["critic_dim"], ck["act_dim"]); ac.load_state_dict(ck["model"]); ac.eval()
    pl = P.build(ck["planner_arch"], 5 * env.PROP); pl.load_state_dict(ck["planner"]); pl.eval()
    env.planner, env.eta_obs = pl, 1.0
    env.gen = torch.Generator().manual_seed(3)
    env.reset_all()
    alive = torch.ones(N, dtype=torch.bool); fell = torch.zeros(N, dtype=torch.bool); ok = torch.zeros(N, dtype=torch.bool)
    wp_max = torch.zeros(N, dtype=torch.long); best = torch.full((N,), 1e3)
    names = [data.motion.names[int(c)] for c in data.motion.clip_id[env.f]]
    t0 = time.time()
    with torch.no_grad():
        for step in range(E.GOAL_T + 2):
            ao, _ = env.observe()
            a = ac.actor(ac.an(ao))
            r, term, timeout, info = env.step(a)
            wp_max = torch.where(alive, torch.maximum(wp_max, env.g_wp), wp_max)
            best = torch.where(alive, torch.minimum(best, env.g_best), best)
            ok |= alive & env.g_reached
            fell |= alive & term
            alive &= ~(term | timeout)
            if not bool(alive.any()):
                break
    rows = {}
    for f in range(F):
        for L in levels:
            sel = (env.g_fam == f) & (lv == L)
            k = int(sel.sum())
            rows[f"{data.goal_families[f]} L{L}"] = dict(episodes=k, success=int(ok[sel].sum()), fell=int(fell[sel].sum()),
                                                         waypoints_reached_mean=round(float(wp_max[sel].float().mean()), 2),
                                                         closest_to_goal_m_median=round(float(best[sel].median()), 2))
    rep = dict(checkpoint=ckpt, cfg=cfg, levels=levels, per=per, seconds=round(time.time() - t0, 1), rows=rows, tiles=names)
    json.dump(rep, open(out, "w"), indent=1)
    for k, v in rows.items():
        print(f"{k:18s} {v}", flush=True)


if __name__ == "__main__":
    main()
