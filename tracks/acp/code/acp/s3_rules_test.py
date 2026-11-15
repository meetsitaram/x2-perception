"""CPU tests of the stage-3 paper rules (ACP agent, 2026-11-13, state/acp.md 17k). Never touches the GPU.
    CUDA_VISIBLE_DEVICES= python -m acp.s3_rules_test CFG CKPT OUT.json
Part A (synthetic, no simulator): TileReplay - bookkeeping, per-tile quality eviction, drop of a worse episode at a full tile,
       types -> tiles -> steps sampling shares.
Part B (simulator, 64 envs on CPU, stage-3 env with paper_s3): phase 0 on every reset, goal-env reset yaw within +-0.4 rad of the
       tile's spawn yaw and spread over it, success latch only within 0.5 m / 0.1 m of the final waypoint, segment cut at the first
       step within 0.2 m, demotion distance only while the height is right, Table 3 stumbling / slippage values are counts over the
       feet, recorder -> TileReplay flush keeps exactly the cut length, positive work per metre is recorded."""
from __future__ import annotations
import json, math, sys
import numpy as np
import torch
from acp import planner as P

R = {}
def check(name, ok, detail=""):
    R[name] = dict(ok=bool(ok), detail=detail); print(("PASS " if ok else "FAIL ") + name + (" | " + str(detail) if detail else ""), flush=True)

# ---------------------------------------------------------------- A: TileReplay
def part_a():
    dev = torch.device("cpu"); hd = 8
    clip_type = torch.tensor([0, 0, 0, 1, 1, 2])                 # 6 tiles (clips), 3 types: type 0 has 3 tiles, type 1 two, type 2 one
    rp = P.TileReplay(1000, hd, dev, clip_type)
    def ep(tile, n, q, tag):
        h = torch.full((n, hd), float(tag)); return (h, torch.zeros(n, P.NP), torch.zeros(n, 3), torch.zeros(n, P.NB, 3),
                                                     torch.zeros(n, P.NB, dtype=torch.bool), torch.full((n,), tile), torch.zeros(n, dtype=torch.long), torch.tensor([q]))
    # fill: tile 0 gets 8 episodes of 100 (qualities 1..8), tiles 1-5 one of 40 each
    for j in range(8):
        rp.push_episodes(*ep(0, 100, float(j + 1), j + 1))
    for t in range(1, 6):
        rp.push_episodes(*ep(t, 40, 0.5, 100 + t))
    used = int(rp.used.sum())
    check("A1 bookkeeping after fill", used == rp.size == int(rp.tile_cnt.sum()) and used <= 1000 and rp.nfree == 1000 - used,
          dict(used=used, size=rp.size, tile_cnt=rp.tile_cnt.tolist(), evicted=rp.stats["evicted"]))
    # buffer full (800 + 200 = 1000): tile 0 has 800 > share 1000/6; a better tile-0 episode evicts tile 0's worst (q = 8)
    q_before = sorted(rp.eps[e][1] for e in rp.tile_eps[0])
    rp.push_episodes(*ep(0, 100, 0.1, 99))
    q_after = sorted(rp.eps[e][1] for e in rp.tile_eps[0])
    check("A2 better episode replaces the tile's worst", 8.0 in q_before and 8.0 not in q_after and any(abs(x - 0.1) < 1e-6 for x in q_after) and rp.size == int(rp.used.sum()),
          dict(before=q_before, after=q_after))
    # a worse tile-0 episode (q = 9) is dropped
    d0 = rp.stats["dropped"]; rp.push_episodes(*ep(0, 100, 9.0, 98))
    check("A3 worse episode at a full tile is dropped", rp.stats["dropped"] == d0 + 1 and 9.0 not in [rp.eps[e][1] for e in rp.tile_eps[0]])
    # an under-share tile evicts from the largest tile (tile 0), its worst episode
    w0 = max(rp.eps[e][1] for e in rp.tile_eps[0]); rp.push_episodes(*ep(5, 60, 3.0, 97))
    check("A4 tile under its share evicts the largest tile's worst", w0 not in [rp.eps[e][1] for e in rp.tile_eps[0]] and rp.tile_cnt[5] == 100,
          dict(tile_cnt=rp.tile_cnt.tolist()))
    # every stored sample still carries its episode's tag and its tile
    ok = True
    for e, (k, q, sl) in rp.eps.items():
        ok &= bool((rp.clip[torch.from_numpy(sl)] == k).all())
    check("A5 slots consistent with episodes", ok and int((rp.clip >= 0).sum()) == rp.size)
    # sampling: types 1/3 each, tiles evenly within a type
    idx = rp.indices(300000); c = rp.clip[idx]
    ts = torch.bincount(clip_type[c], minlength=3).double() / len(c); tl = torch.bincount(c, minlength=6).double() / len(c)
    exp_t = torch.tensor([1 / 3] * 3, dtype=torch.double); exp_l = torch.tensor([1 / 9] * 3 + [1 / 6] * 2 + [1 / 3], dtype=torch.double)
    check("A6 sampling types -> tiles -> steps", float((ts - exp_t).abs().max()) < 0.01 and float((tl - exp_l).abs().max()) < 0.01,
          dict(type_share=[round(float(x), 4) for x in ts], tile_share=[round(float(x), 4) for x in tl]))
    # uniform over steps inside a tile: two tile-1 episodes -> both sampled in proportion to their sizes
    check("A7 no free slot sampled", bool(rp.used[idx].all()))

# ---------------------------------------------------------------- B: env rules
def part_b(cfg, ckpt):
    from acp.data import Dataset
    from acp.env import AcpEnv, S3_YAW, DT
    from acp import env as E
    dev = torch.device("cpu")
    import os
    data = Dataset(cfg, dev, cache=os.path.join(os.path.dirname(cfg), "labels_" + os.path.basename(cfg) + ".pt"))
    N = 64
    env = AcpEnv(data, N, seed=3, njmax=450)
    env.csi = False
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    pl = P.build("attn", 5 * env.PROP); pl.load_state_dict(ck["planner"]); pl.eval()
    env.planner, env.rec = pl, P.Recorder(N, 5 * env.PROP, dev)
    env.eta_obs = env.eta_rew = 1.0
    env.paper_s3 = True
    env.setup_goals(0.5)
    gm = env.goal_env
    env.reset_all()
    m, d = env.motion, env.data
    # B1 phase 0: every env starts at its clip's first frame
    cid = m.clip_id[env.f]
    check("B1 phase 0 on reset (all envs)", bool((env.f == m.clip_start[cid]).all()), dict(goal_envs=int(gm.sum()), envs=N))
    # B2 yaw: goal envs rotated within +-0.4 of the tile spawn yaw; reference envs not rotated
    S = env.state(); yaw = torch.atan2(S[1][:, 1, 0], S[1][:, 0, 0])
    spawn = torch.atan2(m.root_R[env.f][:, 1, 0], m.root_R[env.f][:, 0, 0])
    dy = torch.remainder(yaw - spawn + math.pi, 2 * math.pi) - math.pi
    ys = []
    for _ in range(5):
        env.reset(torch.arange(N)); S = env.state(); yaw = torch.atan2(S[1][:, 1, 0], S[1][:, 0, 0])
        spawn = torch.atan2(m.root_R[env.f][:, 1, 0], m.root_R[env.f][:, 0, 0])
        ys.append((torch.remainder(yaw - spawn + math.pi, 2 * math.pi) - math.pi)[gm])
    ys = torch.cat(ys)
    check("B2 goal reset yaw U(+-0.4)", float(ys.abs().max()) <= S3_YAW + 1e-4 and float(ys.std()) > 0.15 and float(dy[~gm].abs().max()) < 1e-4,
          dict(max_abs=round(float(ys.abs().max()), 4), sd=round(float(ys.std()), 4), expected_sd=round(0.4 / math.sqrt(3), 4), ref_max=round(float(dy[~gm].abs().max()), 6)))
    # B3-B6: teleports onto the final waypoint
    env.reset(torch.arange(N))
    gi = gm.nonzero().squeeze(-1)[:4].tolist()
    a, b, c_, e_ = gi
    for i in gi:
        env.g_wp[i] = d.goal_n[m.clip_id[env.f[i]]] - 1
    def put(i, dx, dz):
        cl = m.clip_id[env.f[i]]; fin = d.goal_wp[cl, d.goal_n[cl] - 1]
        env.qpos[i, 0], env.qpos[i, 1], env.qpos[i, 2] = fin[0] + dx, fin[1], fin[2] + d.stand_root_h + dz
        env.qvel[i] = 0.0
    zero = torch.zeros(N, env.NJ)
    for _ in range(3):                                                 # a few recorded steps first
        env.observe(); env.step(zero)
    for i in gi:
        env.g_wp[i] = d.goal_n[m.clip_id[env.f[i]]] - 1
    put(a, 0.10, 0.0); put(b, 0.40, 0.0); put(c_, 0.70, 0.0); put(e_, 0.10, 0.35)
    env.fault_det.reset(torch.tensor(gi))                              # a teleport is not a simulator fault
    len_before = env.rec.len.clone(); gb0 = env.g_best.clone()
    env.observe(); _, _, _, info0 = env.step(zero)
    check("B0 teleport step without simulator fault on the teleported envs", not bool(info0["done_mask"][gi].any()), dict(done=info0["done_mask"][gi].tolist()))
    got = env.g_succ[[a, b, c_, e_]].tolist()
    check("B3 success latch 0.5 m horizontal / 0.1 m vertical", got == [True, True, False, False],
          dict(at_0_10m=got[0], at_0_40m=got[1], at_0_70m=got[2], at_0_10m_but_0_35m_high=got[3]))
    cuts = env.rec.cut[[a, b, c_, e_]].tolist()
    check("B4 segment cut at the first step within 0.2 m", cuts[0] == int(len_before[a]) + 1 and cuts[1] == -1 and cuts[2] == -1,
          dict(cut=cuts, len_at_teleport=int(len_before[a]) + 1))
    gb = env.g_best[[a, b, c_, e_]].tolist()
    check("B5 demotion distance only at the right height", gb[0] < 0.2 and gb[2] <= 0.75 and abs(gb[3] - float(gb0[e_])) < 1e-6,
          dict(g_best=[round(x, 3) for x in gb], high_env_before=round(float(gb0[e_]), 3)))
    bad = info_bad = None
    # B7 recorder -> TileReplay: env a ends successfully; exactly cut samples (every 2nd step) enter, with its tile
    ns0 = len(data.set_names)
    clip_type = torch.where(data.is_goal, ns0 + data.goal_family.clamp(min=0), data.clip_set)
    rp = P.TileReplay(100000, 5 * env.PROP, dev, clip_type)
    done = torch.zeros(N, dtype=torch.bool); done[a] = True
    w_a, d0_a, cut_a, tile_a = float(env.rec.work[a]), float(env.rec.d0[a]), int(env.rec.cut[a]), int(env.rec.clip[a])
    env.rec.end(done, done.clone())
    n_in = env.rec.flush(rp)
    q = [v[1] for v in rp.eps.values()]
    check("B7 flush keeps the cut length, quality = work / max(0.5, d0)", n_in == math.ceil(cut_a / 2) and rp.size == n_in and len(q) == 1
          and abs(q[0] - w_a / max(0.5, d0_a)) < 1e-4 * max(1.0, q[0]) and w_a > 0 and bool((rp.clip[rp.used] == tile_a).all()),
          dict(samples=n_in, cut=cut_a, work_J=round(w_a, 2), d0_m=round(d0_a, 3), quality=round(q[0], 3) if q else None))
    # B6 term ranges: stumbling / slippage are counts over two feet
    acc = {"st": [], "sl": []}
    for _ in range(20):
        env.observe(); r, term, to, info = env.step(torch.randn(N, env.NJ) * 0.3)
        acc["st"].append(float(info["terms"]["goal/stumbling"])); acc["sl"].append(float(info["terms"]["goal/slippage"]))
    check("B6 stumbling / slippage in [0, 2] (sum over feet)", all(0 <= x <= 2 for x in acc["st"] + acc["sl"]),
          dict(stumbling_mean=round(float(np.mean(acc["st"])), 4), slippage_mean=round(float(np.mean(acc["sl"])), 4)))

if __name__ == "__main__":
    torch.set_num_threads(8)
    part_a()
    if len(sys.argv) > 2:
        part_b(sys.argv[1], sys.argv[2])
    ok = all(v["ok"] for v in R.values())
    R["ALL"] = ok
    if len(sys.argv) > 3:
        json.dump(R, open(sys.argv[3], "w"), indent=1)
    print("ALL PASS" if ok else "SOME FAILED", flush=True)
