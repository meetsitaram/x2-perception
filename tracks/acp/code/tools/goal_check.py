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
    env = E.AcpEnv(data, N, eval_mode=True, njmax=450, nconmax=int(os.environ.get("ACP_NCONMAX", 64)))
    env.setup_goals(1.0)
    i = torch.arange(N)
    env.g_fam = (i // per) % F
    lv = torch.tensor(levels)[(i // (per * F)) % len(levels)]
    env.g_level = lv.clone()
    ac = ActorCritic(ck["actor_dim"], ck["critic_dim"], ck["act_dim"]); ac.load_state_dict(ck["model"]); ac.eval()
    if os.environ.get("PLANNER"):                       # planner file from acp.planner train / export_pair (stage-1 trackers)
        pk = torch.load(os.environ["PLANNER"], map_location=dev, weights_only=False)
        pl = P.build(pk["arch"], 5 * env.PROP); pl.load_state_dict(pk["net"]); pl.eval()
    else:
        pl = P.build(ck["planner_arch"], 5 * env.PROP); pl.load_state_dict(ck["planner"]); pl.eval()
    env.planner, env.eta_obs = pl, 1.0
    env.gen = torch.Generator().manual_seed(3)
    env.reset_all()
    alive = torch.ones(N, dtype=torch.bool); fell = torch.zeros(N, dtype=torch.bool); ok = torch.zeros(N, dtype=torch.bool)
    wp_max = torch.zeros(N, dtype=torch.long); best = torch.full((N,), 1e3)
    names = [data.motion.names[int(c)] for c in data.motion.clip_id[env.f]]
    # hold-at-goal check (2026-11-12): from the step the final station is first held (1 s), the next 3 s: travel of the
    # pelvis from where it was, its peak horizontal speed, falls; pass = travel < 0.20 m, speed < 0.15 m/s, no fall
    HOLD = 150; r_step = torch.full((N,), -1, dtype=torch.long); p0 = torch.zeros(N, 2)
    h_trav = torch.zeros(N); h_spd = torch.zeros(N); h_fell = torch.zeros(N, dtype=torch.bool)
    t0 = time.time()
    with torch.no_grad():
        for step in range(E.GOAL_T + 152):
            ao, _ = env.observe()
            a = ac.actor(ac.an(ao))
            r, term, timeout, info = env.step(a)
            wp_max = torch.where(alive, torch.maximum(wp_max, env.g_wp), wp_max)
            best = torch.where(alive, torch.minimum(best, env.g_best), best)
            new = alive & env.g_reached & (r_step < 0)
            r_step = torch.where(new, step, r_step); p0[new] = env.qpos[new, :2].float()
            inw = (r_step >= 0) & (step - r_step <= HOLD) & alive
            h_trav = torch.where(inw, torch.maximum(h_trav, (env.qpos[:, :2].float() - p0).norm(dim=-1)), h_trav)
            h_spd = torch.where(inw & (step > r_step), torch.maximum(h_spd, env.qvel[:, :2].float().norm(dim=-1)), h_spd)
            h_fell |= inw & term
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
            hs = sel & (r_step >= 0) & (r_step + HOLD <= E.GOAL_T)
            hp = hs & (h_trav < 0.20) & (h_spd < 0.15) & ~h_fell
            rows[f"{data.goal_families[f]} L{L}"].update(hold_checked=int(hs.sum()), hold_pass=int(hp.sum()),
                hold_travel_m_median=round(float(h_trav[hs].median()), 3) if hs.any() else None,
                hold_peak_speed_median=round(float(h_spd[hs].median()), 3) if hs.any() else None, hold_fell=int((hs & h_fell).sum()),
                station_height_m_max=round(float(env.data.goal_wp[env.data.motion.clip_id[env.f[sel]], :, 2].max()), 2))
    rep = dict(checkpoint=ckpt, cfg=cfg, levels=levels, per=per, seconds=round(time.time() - t0, 1), rows=rows, tiles=names)
    json.dump(rep, open(out, "w"), indent=1)
    for k, v in rows.items():
        print(f"{k:18s} {v}", flush=True)


if __name__ == "__main__":
    main()
