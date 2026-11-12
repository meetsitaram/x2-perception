"""Planner footholds on low risers (coordinator, 2026-11-12, after the eval ladder failed 10 cm stairs-up at 2,250):
closed loop on the CPU, stage-1 tracker + exported planner, on the fitted low-riser flights (gs2lo_v1, risers 10-15 cm)
and the normal fitted climbs (gst_v1, 16-22 cm) as reference. Per riser bin: success (fall rule and 0.5 m rule as
pair_eval), falls, and where the planner puts the FEET: share of foot targets within 3 cm (horizontally) of a height
step of more than 3 cm (on a riser edge / face), and the target's height above the surface under it.

    CUDA_VISIBLE_DEVICES= python tools/riser_check.py TRACKER_CKPT PLANNER.pt OUT.json [EPISODES_PER_SET=48]
"""
import json
import os
import re
import sys
import time

import torch

sys.path.insert(0, "/mnt/filesystem-g4/tracks/acp/code")
sys.path.insert(0, "/mnt/filesystem-g4/tracks/acp/code/tools")
from acp import env as E  # noqa: E402
from acp import planner as P  # noqa: E402
from acp.data import Dataset  # noqa: E402
from acp.train import ActorCritic  # noqa: E402
from handover import cpu_physics  # noqa: E402
from gear_sonic.supermimic.kinematics import mv3  # noqa: E402

CFG = "/mnt/filesystem-g4/tracks/acp/code/cfg/riser_check_sets.json"


def main():
    ckpt, pl_path, out = sys.argv[1:4]
    per = int(sys.argv[4]) if len(sys.argv) > 4 else 48
    torch.set_num_threads(6)
    dev = torch.device("cpu")
    json.dump({"dir": "/mnt/filesystem-g4/tracks/acp/data/clips", "hmap": "", "note": "riser check only, not training",
               "sets": [{"pkl": "gs2lo_v1.pkl", "props": "gs2lo_v1_props.json", "weight": 0.5},
                        {"pkl": "gst_v1.pkl", "props": "gst_v1_props.json", "weight": 0.5}]}, open(CFG, "w"))
    data = Dataset(CFG, dev, cache=os.path.join(os.path.dirname(CFG), "labels_riser_check_sets.json.pt"))
    m, T = data.motion, data.terrain
    ck = torch.load(ckpt, map_location=dev)
    E.AcpEnv._physics = cpu_physics
    ups = [c for c, n in enumerate(m.names) if "gs2lo_up" in n or data.set_names[int(data.clip_set[c])] == "gst_v1"]
    rise = torch.tensor([int(re.search(r"_r(\d+)_", n).group(1)) if re.search(r"_r(\d+)_", n) else 0 for n in m.names])
    # climbs only: gs2lo up-clips and gst_v1 (all climbs)
    lo = [c for c in ups if data.set_names[int(data.clip_set[c])] == "gs2lo_v1"]
    hi = [c for c in ups if data.set_names[int(data.clip_set[c])] == "gst_v1"]
    g = torch.Generator().manual_seed(7)
    pick = lambda cl: torch.tensor(cl)[torch.randint(len(cl), (per,), generator=g)]
    clip = torch.cat([pick(lo), pick(hi)])
    N = len(clip)
    env = E.AcpEnv(data, N, eval_mode=True)
    ac = ActorCritic(ck["actor_dim"], ck["critic_dim"], ck["act_dim"]); ac.load_state_dict(ck["model"]); ac.eval()
    pk = torch.load(pl_path, map_location=dev)
    pl = P.build(pk["arch"], pk["hist_dim"]); pl.load_state_dict(pk["net"]); pl.eval()
    H = 5 * env.PROP
    rep = dict(tracker=ckpt, planner=pl_path, episodes=N, rise_mm=rise[clip].tolist())
    for mode in ("oracle", "planner"):
        env.fall_rule = True
        env.gen = torch.Generator().manual_seed(11)
        saved = data.clip_w
        for i in range(N):
            data.clip_w = torch.zeros_like(saved); data.clip_w[clip[i]] = 1.0
            env.reset(torch.tensor([i]))
        data.clip_w = saved
        alive = torch.ones(N, dtype=torch.bool); failed = torch.zeros(N, dtype=torch.bool); final_d = torch.full((N,), 1e9)
        goal_xy = m.root_p[m.clip_end[clip]][:, :2]
        n_t = torch.zeros(N); n_edge = torch.zeros(N); dz = [[] for _ in range(N)]
        t0 = time.time()
        with torch.no_grad():
            for step in range(int(m.clip_len[clip].max()) + 2):
                ao, _ = env.observe()
                S = env.state()
                if mode == "planner":
                    scan17 = env._scan(S[0], S[1], env.gx_c, env.gy_c)
                    tgt, occ = pl.decode(ao[:, :H], ao[:, -3:], scan17)
                    ao = torch.cat((ao[:, :H + 63], tgt.flatten(1), ao[:, -3:]), -1)
                else:
                    tgt = ao[:, H + 63:H + 63 + 27].reshape(N, 9, 3); occ = (tgt.norm(dim=-1) > 0).float()
                # foot targets (bodies 0, 1) to the world: p + Hyaw @ t
                Hy = E.yaw_mat(S[1])
                w = S[0][:, None, :] + mv3(Hy[:, None], tgt[:, :2])
                cid = m.clip_id[env.f]
                use = alive[:, None] & (occ[:, :2] > 0.5)
                for dx, dy in ((0.0, 0.0),):
                    h0 = T.height(w[..., 0], w[..., 1], cid)
                hx = torch.stack([T.height(w[..., 0] + ox, w[..., 1] + oy, cid) for ox, oy in ((0.03, 0), (-0.03, 0), (0, 0.03), (0, -0.03))], -1)
                edge = (hx.amax(-1) - hx.amin(-1)).clamp(min=0) > 0.03
                edge = edge | ((hx - h0[..., None]).abs().amax(-1) > 0.03)
                n_t += use.sum(-1); n_edge += (use & edge).sum(-1)
                for i in use.any(-1).nonzero().squeeze(-1).tolist():
                    for f in range(2):
                        if use[i, f]:
                            dz[i].append(float(w[i, f, 2] - h0[i, f]))
                a = ac.actor(ac.an(ao))
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
        res = {}
        for name, sel in (("rise 10-12 cm (gs2lo)", (rise[clip] <= 120) & (torch.arange(N) < per)),
                          ("rise 12-15 cm (gs2lo)", (rise[clip] > 120) & (torch.arange(N) < per)),
                          ("rise 16-22 cm (gst_v1)", torch.arange(N) >= per)):
            k = int(sel.sum())
            if k == 0:
                continue
            dzs = sorted(sum((dz[i] for i in sel.nonzero().squeeze(-1).tolist()), []))
            res[name] = dict(episodes=k, success=round(float(succ[sel].float().mean()), 3), fell=int(failed[sel].sum()),
                             foot_targets=int(n_t[sel].sum()), on_edge_share=round(float(n_edge[sel].sum() / n_t[sel].sum().clamp(min=1)), 3),
                             target_height_above_surface_median_cm=round(100 * dzs[len(dzs) // 2], 1) if dzs else None)
        rep[mode] = dict(per_bin=res, seconds=round(time.time() - t0, 1))
        print(mode, json.dumps(res), flush=True)
    json.dump(rep, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
