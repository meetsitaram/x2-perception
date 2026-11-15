"""The ACP paper's own simulation evaluation (arXiv 2610.31608, Results 0.3), on the X2, for a stage-1 checkpoint.

Paper protocol: reference-anchored terrains = the training reference configurations; 3,000 episodes per method and
category pooled over three seeds; success = reaching within 0.5 m of the active goal horizontally; contact placement
error = distance between commanded and realised contact position at contact onset, averaged over valid contact events;
Wilson 95 % intervals for success, episode-clustered 95 % intervals for the contact error; sensor noise, physics
randomisation and action delay disabled; episodes with invalid outputs count as failures.

Here: every clip set is evaluated with EPISODES episodes per seed (start of a segment, deterministic actions), the goal is
the reference root at the end of the segment, and an episode also fails if it ends early under the training failure
conditions. This is a track report, not the team's apparatus score.

    python tools/paper_eval.py CKPT OUT.json [EPISODES_PER_SET_PER_SEED=334]
"""
import json
import os
import math
import sys
import time

import torch

sys.path.insert(0, "/mnt/filesystem-g4/tracks/acp/code")
from acp.data import Dataset  # noqa: E402
from acp.env import AcpEnv  # noqa: E402
from acp.train import ActorCritic  # noqa: E402

MIN_FREE_GB = 10
PAPER = {"stage1_contacts_only_reference_anchored_success_min_over_categories": 0.985,
         "stage3_composite_courses_success_mean": 0.925, "mgmt_composite_courses_success_mean": 0.588,
         "stage3_vs_stage2_sparse_whole_body_gain_points": 54.8,
         "note": "the paper gives the other values only as bars in Figure 7, not as numbers in the text"}


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(c - h, 4), round(c + h, 4)]


def main():
    ckpt, out = sys.argv[1], sys.argv[2]
    per = int(sys.argv[3]) if len(sys.argv) > 3 else 334
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cpu":                       # CPU run (rollout dumps for planner labels without the training GPU)
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from handover import cpu_physics
        import acp.env as _E
        _E.AcpEnv._physics = cpu_physics
    ck = torch.load(ckpt, map_location=dev)
    if ck.get("x2_exclude_pairs") is not None:      # evaluate on the body the checkpoint was trained on
        os.environ["X2_EXCLUDE_PAIRS"] = ck["x2_exclude_pairs"]
    print("[model] X2_EXCLUDE_PAIRS=" + os.environ.get("X2_EXCLUDE_PAIRS", ""), flush=True)
    cfg = os.environ.get("ACP_EVAL_DATA") or ck["data"]      # ACP_EVAL_DATA: score on another data set (old-set numbers beside 2e)
    data = Dataset(cfg, dev, cache=os.path.join(os.path.dirname(cfg), "labels_" + os.path.basename(cfg) + ".pt"))
    m = data.motion
    nset = len(data.set_names)
    N = per * nset
    ac = ActorCritic(ck["actor_dim"], ck["critic_dim"], ck["act_dim"]).to(dev)
    ac.load_state_dict(ck["model"])
    ac.eval()
    env = AcpEnv(data, N, eval_mode=True, njmax=int(os.environ.get("ACP_NJMAX", 450)))   # as training (300 overflowed: nefc warnings in every eval before 2026-11-12 02:20)
    set_id = torch.arange(nset, device=dev).repeat_interleave(per)
    res = {s: dict(n=0, succ=0, term=0, reach=0, place=[], dist=[]) for s in data.set_names}
    t0 = time.time()
    for seed in range(3):
        g = torch.Generator(device=dev).manual_seed(1000 + seed)
        clip = torch.zeros(N, dtype=torch.long, device=dev)
        for s in range(nset):                                   # clips of each set, by duration
            w = data.clip_w * (data.clip_set == s)
            clip[set_id == s] = torch.multinomial(w / w.sum(), per, replacement=True, generator=g)
        # start every environment at the first frame of its clip
        saved = data.clip_w
        env.gen = g
        ids = torch.arange(N, device=dev)
        # reset() draws clips from data.clip_w: draw one environment at a time is too slow, so set the frames directly
        data.clip_w = torch.zeros_like(saved); data.clip_w[clip[0]] = 1.0
        env.reset(ids)
        data.clip_w = saved
        frames = m.clip_start[clip]
        p, R, v, w_, q, qd = (x.clone() for x in m.ref_state(frames))
        from acp.env import mat_to_quat
        env.qpos[:, :3], env.qpos[:, 3:7], env.qpos[:, 7:] = p, mat_to_quat(R), q
        env.qvel[:, :3] = v
        env.qvel[:, 3:6] = (R.transpose(-1, -2) @ w_.unsqueeze(-1)).squeeze(-1)
        env.qvel[:, 6:] = qd
        if env.terrain.num_slots:
            pos, quat = env.terrain.mocap(clip)
            env.mocap_pos[ids.unsqueeze(-1), env.slot_mocap] = pos
            env.mocap_quat[ids.unsqueeze(-1), env.slot_mocap] = quat
        env.ctrl[:] = q
        env.ctrl_prev[:] = q
        env.f[:] = frames
        env.ep_step[:] = 0
        env.a_prev.zero_(); env.a_prev2.zero_()
        S = env.state()
        env.v_prev[:] = data.body_points(S)[1]
        env.c_prev[:] = True
        env.tgt_prev[:], env.occ_prev[:] = data.next[frames], data.occ[frames]
        env.hist[:] = env._proprio(S, env.a_prev, noise=False).unsqueeze(1).expand(-1, 5, -1)
        alive = torch.ones(N, dtype=torch.bool, device=dev)
        failed = torch.zeros(N, dtype=torch.bool, device=dev)
        place_sum, place_n = torch.zeros(N, device=dev), torch.zeros(N, device=dev)
        goal = m.root_p[m.clip_end[clip]][:, :2]
        final_d = torch.full((N,), 1e9, device=dev)
        with torch.no_grad():
            dump = dict(hist=[], scan=[], goal=[], c=[], pts=[], root=[], alive=[]) if seed == 0 and not os.environ.get("ACP_EVAL_NO_DUMP") else None
            for step_i in range(int(m.clip_len.max()) + 2):
                ao, _co = env.observe()
                if dump is not None:      # rollout record for the stage-2 planner's hindsight labels (state before the step)
                    S0 = env.state()
                    dump["c"].append(env.c_prev.clone()); dump["pts"].append(data.body_points(S0)[0].clone())
                    dump["root"].append(torch.cat((S0[0], torch.atan2(S0[1][:, 1, 0], S0[1][:, 0, 0]).unsqueeze(-1)), -1))
                    dump["alive"].append(alive.clone())
                    if step_i % 2 == 0:
                        dump["hist"].append(env.hist.flatten(1).half()); dump["goal"].append(ao[:, -3:].clone())
                        dump["scan"].append(env._scan(S0[0], S0[1], env.gx_c, env.gy_c).half())
                a = ac.actor(ac.an(ao))
                bad_out = ~torch.isfinite(a).all(-1)
                a = torch.nan_to_num(a)
                r, term, timeout, info = env.step(a)
                if step_i % 50 == 0 and dev.type == "cuda" and torch.cuda.mem_get_info()[0] < MIN_FREE_GB * 2**30:
                    raise SystemExit(f"aborted: free GPU memory under {MIN_FREE_GB} GB")   # never starve a training run on the same card
                failed |= alive & (term | bad_out)
                place_sum += torch.where(alive, info["place"].sum(-1), torch.zeros_like(place_sum))
                place_n += torch.where(alive, info["onset"].float().sum(-1), torch.zeros_like(place_n))
                d = (env.qpos[:, :2] - goal).norm(dim=-1)
                ended = alive & (term | timeout | bad_out)
                final_d = torch.where(ended, d, final_d)
                alive &= ~ended
                env.f = torch.minimum(env.f, m.clip_end[clip] - 1)       # finished environments keep stepping harmlessly
                if not bool(alive.any()):
                    break
        succ = ~failed & (final_d < 0.5)
        try:      # small per-episode record (all seeds) for the per-clip failure analysis; never costs the evaluation
            torch.save(dict(clip=clip.cpu(), success=succ.cpu(), failed=failed.cpu(), set_id=set_id.cpu(), set_names=data.set_names,
                            iteration=ck["it"], seed=seed), out.replace(".json", f"_episodes_seed{seed}.pt"))
        except Exception as e:
            print("episode record failed:", repr(e))
        if dump is not None:
            try:
                torch.save(dict({k: torch.stack(v).cpu() for k, v in dump.items()}, clip=clip.cpu(), success=succ.cpu(), set_id=set_id.cpu(),
                                set_names=data.set_names, iteration=ck["it"], obs_every=2,
                                note="state before each control step; hist = actor proprio history (5 x 99, noise-free), scan = 17 x 11 "
                                     "heights minus root height (x -0.8..0.8, y -0.5..0.5, body-yaw frame), goal = actor goal command, "
                                     "c = measured contact of the nine bodies, pts = their contact points (world), root = x, y, z, yaw"),
                           out.replace(".json", "_rollouts.pt"))
            except Exception as e:  # the dump must never cost the evaluation
                print("rollout dump failed:", repr(e))
            del dump
        for s, name in enumerate(data.set_names):
            k = set_id == s
            r_ = res[name]
            r_["n"] += int(k.sum()); r_["succ"] += int(succ[k].sum()); r_["term"] += int(failed[k].sum())
            r_["reach"] += int((final_d[k] < 0.5).sum())
            ok = k & (place_n > 0)
            r_["place"] += (place_sum[ok] / place_n[ok]).tolist()
            r_["dist"] += final_d[k].clamp(max=99).tolist()
    rows = {}
    tot_n = tot_s = 0
    allp = []
    for name, r_ in res.items():
        pl = torch.tensor(r_["place"]) if r_["place"] else torch.zeros(1)
        se = float(pl.std() / math.sqrt(max(len(pl), 1))) if len(pl) > 1 else 0.0
        rows[name] = dict(episodes=r_["n"], success=round(r_["succ"] / r_["n"], 4), success_wilson95=wilson(r_["succ"], r_["n"]),
                          ended_early_frac=round(r_["term"] / r_["n"], 4), within_0p5m_of_goal_frac=round(r_["reach"] / r_["n"], 4),
                          final_goal_distance_m_median=round(float(torch.tensor(r_["dist"]).median()), 3),
                          contact_placement_error_m=round(float(pl.mean()), 4),
                          contact_placement_error_ci95=[round(float(pl.mean()) - 1.96 * se, 4), round(float(pl.mean()) + 1.96 * se, 4)])
        tot_n += r_["n"]; tot_s += r_["succ"]; allp += r_["place"]
    allp_t = torch.tensor(allp) if allp else torch.zeros(1)
    report = dict(checkpoint=ckpt, iteration=ck["it"], stage=ck.get("stage", 1), data=cfg, x2_exclude_pairs=os.environ.get("X2_EXCLUDE_PAIRS", ""),
                  trained_model="x2_ultra_hipexclude.xml" if "hip" in os.environ.get("X2_EXCLUDE_PAIRS", "").split(",") else "x2_ultra.xml", simulator="MuJoCo Warp", robot="AgiBot X2 Ultra",
                  protocol="paper Results 0.3: reference-anchored terrains, noise / randomisation / delay off, deterministic actions, "
                           "3 seeds, success = within 0.5 m of the goal (reference root at the segment end) and not ended early",
                  category_dense_foot_only=dict(episodes=tot_n, success_pooled=round(tot_s / tot_n, 4), success_wilson95=wilson(tot_s, tot_n),
                                                success_sets_weighted_equally=round(sum(r["success"] for r in rows.values()) / len(rows), 4),
                                                contact_placement_error_m=round(float(allp_t.mean()), 4)),
                  per_set=rows, paper_reported=PAPER,
                  not_reproduced=["sparse foot-only, dense whole-body and sparse whole-body categories (no X2 data: stepping stones, beams, box "
                                  "climbs, get-ups, ladders, vaults)", "goal-reaching and unseen terrain groups and the eight composite courses "
                                  "(stage 3)", "MGMT baseline and the MLP-planner and kinematics-conditioned ablations", "real-robot results"],
                  seconds=round(time.time() - t0, 1))
    json.dump(report, open(out, "w"), indent=1)
    print(json.dumps({k: report[k] for k in ("iteration", "category_dense_foot_only", "seconds")}))
    for name, r in rows.items():
        print(f"  {name:30s} n {r['episodes']:5d} success {r['success']:.3f} {r['success_wilson95']} early-end {r['ended_early_frac']:.3f} "
              f"placement {r['contact_placement_error_m']:.3f} m")


if __name__ == "__main__":
    main()
