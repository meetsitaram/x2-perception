"""Planner + tracker closed loop on the reference-anchored terrains, on the CPU (does not touch the training GPU):
the stage-1 tracker driven (a) by the oracle contacts and (b) by the contact planner, same clips and start states.
Also records 50 steps of robot 0 of the planner-driven run for the eval monitor's adapter check.

    CUDA_VISIBLE_DEVICES= python tools/pair_eval.py TRACKER_CKPT PLANNER.pt OUT_DIR [EPISODES_PER_SET=12]

Success as in tools/paper_eval.py (not ended early, within 0.5 m of the segment's end point). Small sample: a plumbing
check and a first signal, not a result.
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/mnt/filesystem-g4/tracks/acp/code")
sys.path.insert(0, "/mnt/filesystem-g4/tracks/acp/code/tools")
from acp import env as E  # noqa: E402
from acp import planner as P  # noqa: E402
from acp.data import Dataset  # noqa: E402
from acp.train import ActorCritic  # noqa: E402
from handover import cpu_physics  # noqa: E402
from paper_eval import wilson  # noqa: E402


def main():
    ckpt, pl_path, out = sys.argv[1:4]
    per = int(sys.argv[4]) if len(sys.argv) > 4 else 12
    os.makedirs(out, exist_ok=True)
    torch.set_num_threads(6)
    dev = torch.device("cpu")
    ck = torch.load(ckpt, map_location=dev)
    cfg = ck["data"]
    data = Dataset(cfg, dev, cache=os.path.join(os.path.dirname(cfg), "labels_" + os.path.basename(cfg) + ".pt"))
    m = data.motion
    nset = len(data.set_names)
    N = per * nset
    E.AcpEnv._physics = cpu_physics
    env = E.AcpEnv(data, N, eval_mode=True)
    ac = ActorCritic(ck["actor_dim"], ck["critic_dim"], ck["act_dim"]); ac.load_state_dict(ck["model"]); ac.eval()
    pk = torch.load(pl_path, map_location=dev)
    pl = P.build(pk["arch"], pk["hist_dim"]); pl.load_state_dict(pk["net"]); pl.eval()
    set_id = torch.arange(nset).repeat_interleave(per)
    g = torch.Generator().manual_seed(7)
    clip = torch.zeros(N, dtype=torch.long)
    for s in range(nset):
        w = data.clip_w * (data.clip_set == s)
        if float(w.sum()) <= 0:  # a set with training weight 0 (e.g. the replaced pico 2x2 set in data 2h): sample its clips uniformly
            w = (data.clip_set == s).float()
        clip[set_id == s] = torch.multinomial(w / w.sum(), per, replacement=True, generator=g)
    H = 5 * env.PROP
    report = dict(tracker=ckpt, tracker_iteration=int(ck["it"]), planner=pl_path, planner_arch=pk["arch"], planner_stats=pk["stats"],
                  episodes_per_set=per, simulator="MuJoCo Warp, CPU", note="small sample; plumbing check and first signal")
    for mode, fall in (("oracle_contacts", False), ("planner_contacts", False), ("oracle_contacts_fall_rule", True), ("planner_contacts_fall_rule", True)):
        env.fall_rule = fall     # fall_rule: fail only on a fall (root under 0.35 m over the ground, or tilt over 60 degrees); success still
        #                          = no failure and within 0.5 m of the segment end point when the clip time is up
        t0 = time.time()
        env.gen = torch.Generator().manual_seed(11)
        ids = torch.arange(N)
        saved = data.clip_w
        for i in range(N):                       # reset each robot onto its own clip (start of the segment)
            data.clip_w = torch.zeros_like(saved); data.clip_w[clip[i]] = 1.0
            env.reset(ids[i:i + 1])
        data.clip_w = saved
        if hasattr(env, "fault_det"):           # simcheck detector: manual resets must clear it, or the jump from the last mode is a fault
            env.fault_det.reset(ids)
        alive = torch.ones(N, dtype=torch.bool); failed = torch.zeros(N, dtype=torch.bool)
        final_d = torch.full((N,), 1e9)
        goal_xy = m.root_p[m.clip_end[clip]][:, :2]
        rec = {k: [] for k in ("qpos", "qvel", "planner_proprio_history", "planner_goal", "planner_scan_17x11", "planner_targets",
                               "planner_occupancy", "actor_input", "action")}
        with torch.no_grad():
            for step in range(int(m.clip_len[clip].max()) + 2):
                ao, _ = env.observe()
                if mode.startswith("planner_contacts"):
                    S = env.state()
                    scan17 = env._scan(S[0], S[1], env.gx_c, env.gy_c)
                    tgt, occ = pl.decode(ao[:, :H], ao[:, -3:], scan17)
                    ao = torch.cat((ao[:, :H + 63], tgt.flatten(1), ao[:, -3:]), -1)
                    if step < 50 and not fall:
                        rec["qpos"].append(env.qpos[0].clone().numpy()); rec["qvel"].append(env.qvel[0].clone().numpy())
                        rec["planner_proprio_history"].append(ao[0, :H].numpy()); rec["planner_goal"].append(ao[0, -3:].numpy())
                        rec["planner_scan_17x11"].append(scan17[0].numpy()); rec["planner_targets"].append(tgt[0].numpy())
                        rec["planner_occupancy"].append(occ[0].numpy()); rec["actor_input"].append(ao[0].numpy())
                a = ac.actor(ac.an(ao))
                if mode == "planner_contacts" and step < 50:
                    rec["action"].append(a[0].clamp(-20.0, 20.0).numpy())   # record the action as env.step applies it (clamp +-20, 17i)
                r, term, timeout, info = env.step(a)
                failed |= alive & term
                d = (env.qpos[:, :2] - goal_xy).norm(dim=-1)
                ended = alive & (term | timeout)
                final_d = torch.where(ended, d, final_d)
                alive &= ~ended
                env.f = torch.minimum(env.f, m.clip_end[clip] - 1)
                if not bool(alive.any()):
                    break
        succ = ~failed & (final_d < 0.5)
        rows = {name: dict(episodes=per, success=round(float(succ[set_id == s].float().mean()), 3), ended_early=int(failed[set_id == s].sum()))
                for s, name in enumerate(data.set_names)}
        report[mode] = dict(episodes=N, success=round(float(succ.float().mean()), 4), success_wilson95=wilson(int(succ.sum()), N),
                            ended_early_frac=round(float(failed.float().mean()), 4), per_set=rows, seconds=round(time.time() - t0, 1))
        print(mode, json.dumps({k: report[mode][k] for k in ("episodes", "success", "success_wilson95", "ended_early_frac", "seconds")}), flush=True)
        if mode == "planner_contacts":
            np.savez(os.path.join(out, "obs_check_pair.npz"), **{k: np.array(v) for k, v in rec.items()}, clip_name=m.names[int(clip[0])],
                     note="50 steps of one robot, planner-driven, eval mode, before each step: planner inputs (proprio history 495 = the "
                          "tracker's, goal 3, scan 187), planner outputs (targets 9 x 3 in the body-yaw frame relative to the root, zero when "
                          "occupancy <= 0.5; occupancy 9), the tracker's full input (588: history, 9 x 7 scan, the planner's targets, goal) "
                          "and action. Flat floor (first clip set).")
    json.dump(report, open(os.path.join(out, "pair_eval_cpu.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
