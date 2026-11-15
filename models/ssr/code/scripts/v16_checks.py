"""v16 cheap checks (CPU). (a) gait statistics of the motion prior per group, amp_v4 vs amp_v5, and of policy rollouts;
(b) style reward of v14's walking vs v15's shuffle under near-fresh amp_v5 discriminators (r8 at 7,050) and later ones.
Run from code/ (X2_TERRAIN_PAPER=1): env CUDA_VISIBLE_DEVICES= X2_PLANT=... python scripts/v16_checks.py --out ../runs/v16_checks/checks.json
"""
import argparse, json, os, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from ssr import networks as N
from ssr import terrain as T
from ssr.env import AMP_FRAMES, SSREnv
import ssr.env as E

HZ = 50.0
TH = 0.3   # foot world speed (m/s) under which a foot counts as in stance (kinematic contact)


def foot_world_speed(F):
    """F (T, 63) one contiguous sequence -> (T, 2) foot speeds in the world (expressed in base frame)."""
    v, w = F[:, 3:6], F[:, 6:9]
    out = []
    for j in (57, 60):
        p = F[:, j:j + 3]
        dp = np.gradient(p, axis=0) * HZ if len(p) > 2 else np.zeros_like(p)
        out.append(np.linalg.norm(dp + v + np.cross(w, p), axis=1))
    return np.stack(out, 1)


def gait(seqs, contacts=None):
    """seqs: list of (T,63) arrays; contacts: optional list of (T,2) bool arrays (force contact)."""
    vx, sp, st, onsets, mov_t = [], [], [], 0, 0.0
    cdbl = []
    for k, F in enumerate(seqs):
        if len(F) < 10:
            continue
        s = foot_world_speed(F) < TH
        sp_k = np.linalg.norm(F[:, 3:5], axis=1)
        m = sp_k > 0.2
        vx.append(F[:, 3]); sp.append(sp_k); st.append(s)
        on = (s[1:] & ~s[:-1]) & m[1:, None]
        onsets += int(on.sum()); mov_t += m.sum() / HZ
        if contacts is not None:
            cdbl.append((contacts[k].sum(1) == 2)[m])
    vx, sp, st = np.concatenate(vx), np.concatenate(sp), np.concatenate(st)
    m = sp > 0.2
    dbl = st.sum(1) == 2
    r = dict(seconds=round(len(vx) / HZ, 1), standing=round(float((sp < 0.1).mean()), 3), forward_gt02=round(float((vx > 0.2).mean()), 3),
             backward=round(float((vx < -0.2).mean()), 3),
             vx_p10_p50_p90_moving=[round(float(q), 2) for q in np.quantile(vx[m], [0.1, 0.5, 0.9])] if m.any() else None,
             speed_mean_moving=round(float(sp[m].mean()), 2) if m.any() else None,
             double_kin_moving=round(float(dbl[m].mean()), 3) if m.any() else None,
             flight_kin_moving=round(float((st.sum(1) == 0)[m].mean()), 3) if m.any() else None,
             steps_per_s_moving=round(onsets / max(mov_t, 1e-6), 2))
    if cdbl:
        r["double_force_moving"] = round(float(np.concatenate(cdbl).mean()), 3)
    return r


def split(z, g):
    f, s = z[g], list(z[g + "_starts"]) + [len(z[g])]
    return [f[a:b] for a, b in zip(s[:-1], s[1:])]


def load_disc(path, dev, part_w):
    ck = torch.load(path, map_location=dev, weights_only=False)
    d = N.Discriminators().to(dev)
    d.load_state_dict(ck["disc"]); d.eval()
    if part_w:
        wl, wa, wh = part_w
        pw = torch.full((63,), wl); pw[22:30], pw[43:51] = wa, wa; pw[51:57] = wh
        d.part_w = pw.repeat(AMP_FRAMES).to(dev)
    return d, int(ck["iter"])


def windows(seqs):
    xs = []
    for F in seqs:
        for t in range(0, len(F) - AMP_FRAMES + 1):
            xs.append(F[t:t + AMP_FRAMES].reshape(-1))
    return np.asarray(xs, np.float32)


@torch.no_grad()
def rollout(path, speed, dev, n=32, steps=300, keep=60, map_memory=False, tile=None):
    pk = torch.load(path, map_location=dev, weights_only=False)
    enc, aux, actor = N.Encoder(map_memory).to(dev), N.Aux().to(dev), N.MoEActor().to(dev)
    enc.load_state_dict(pk["enc"]); aux.load_state_dict(pk["aux"]); actor.load_state_dict(pk["actor"])
    env = SSREnv(n, str(dev), train=False, tiles=True)
    ids = torch.arange(n, device=dev)
    if tile is None:
        env.flat_only = True
        env.type[:], env.level[:] = T.FLAT, 0
        env.reset(ids)
        env.qpos[:, 3], env.qpos[:, 4:7] = 1.0, 0.0
    else:
        env.type[:], env.level[:] = T.TYPES.index(tile[0]), tile[1]
        env.ep[:] = 1
        env.reset(ids)
    env.speed[:], env.heading[:] = speed, 0.0
    env.refresh(); env._update_cmd(); env._contacts(); env._update_obs(ids); env._update_cmd()
    h = enc.init_h(n, dev)
    fr, cc, vxs = [], [], []
    for t in range(steps):
        e = enc.step(env.hist, env.depth, h, sample=False); h = e["h"]
        o = torch.cat((env.prop, aux.est(env.hist.flatten(1)), e["z"]), -1)
        env.step(actor(o).clamp(-8, 8))
        env.speed[:] = speed
        if tile is None:
            env.heading[:] = 0.0
        if t >= keep:
            fr.append(env.amp_hist[:, -1].clone().cpu().numpy())
            cc.append((env.F[..., 2].abs() > E.FOOT_FORCE_ON).cpu().numpy())
            vxs.append(float(env.qvel[:, 0].mean()))
    fr, cc = np.stack(fr, 1), np.stack(cc, 1)   # (n, T, 63), (n, T, 2)
    del env
    return [fr[i] for i in range(n)], [cc[i] for i in range(n)], float(np.mean(vxs)), int(pk["iter"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--policies", nargs="*", default=["v14=../eval/export/ckpts/r7_v14_final.pt", "v15shuffle=../runs/r8/last.pt"])
    ap.add_argument("--discs", nargs="*", default=["r8_7050=../eval/export/ckpts/r8_v15_p100.pt", "r8_last=../runs/r8/last.pt"])
    ap.add_argument("--speeds", default="0.5,0.74,1.0")
    ap.add_argument("--tiles", default="", help="type:level,... rollouts on tiles instead of the open floor (gait only)")
    a = ap.parse_args()
    dev = torch.device("cpu")
    out = dict(threshold_foot_speed=TH, prior={}, policy={}, style={})
    if a.tiles:
        mm = bool(json.load(open(ROOT.parent / "runs/r8/args.json")).get("map_memory", False))
        for spec in a.policies:
            tag, path = spec.split("=")
            for tl in a.tiles.split(","):
                ty, lv = tl.split(":")
                seqs, cc, vx, it = rollout(path, 0.74, dev, map_memory=mm, tile=(ty, int(lv)), steps=250, keep=25)
                r = gait(seqs, cc); r.update(iter=it, tile=tl, mean_vx=round(vx, 3))
                out["policy"][f"{tag}@{tl}"] = r
                print("tile", tag, tl, json.dumps(r), flush=True)
        pathlib.Path(a.out).write_text(json.dumps(out, indent=1)); print("V16_CHECKS DONE", flush=True); return
    z4, z5 = np.load(ROOT / "data/amp_v4.npz"), np.load(ROOT / "data/amp_v5.npz")
    for tag, z in (("amp_v4", z4), ("amp_v5", z5)):
        out["prior"][tag] = {g: gait(split(z, g)) for g in z.files if not g.endswith("_starts")}
        print(tag, json.dumps(out["prior"][tag]), flush=True)
    mm = bool(json.load(open(ROOT.parent / "runs/r8/args.json")).get("map_memory", False))
    roll = {}
    for spec in a.policies:
        tag, path = spec.split("=")
        for s in (float(x) for x in a.speeds.split(",")):
            seqs, cc, vx, it = rollout(path, s, dev, map_memory=mm)
            roll[(tag, s)] = seqs
            r = gait(seqs, cc); r.update(iter=it, cmd=s, mean_vx=round(vx, 3))
            out["policy"][f"{tag}@{s}"] = r
            print("policy", tag, s, json.dumps(r), flush=True)
    for spec in a.discs:
        dtag, path = spec.split("=")
        d, it = load_disc(path, dev, (1.0, 0.3, 0.0))
        sty = lambda x, i: (1 - 0.25 * (d(torch.as_tensor(x), i).squeeze(-1) - 1) ** 2).clamp(min=0.0)
        res = dict(iter=it)
        for i, g in enumerate(T.TYPES):
            ref = windows(split(z5, g))
            spd = np.linalg.norm(ref[:, 3:5], axis=1)
            row = {}
            for lab, m in (("ref_moving", spd >= 0.4), ("ref_slow", (spd >= 0.2) & (spd < 0.4)), ("ref_stand", spd < 0.1)):
                if m.sum() > 16:
                    row[lab] = round(float(sty(ref[m][:: max(1, int(m.sum()) // 3000)], i).mean()), 3)
            for (pt, s), seqs in roll.items():
                row[f"{pt}@{s}"] = round(float(sty(windows(seqs)[::4], i).mean()), 3)
            res[g] = row
            print("style", dtag, g, json.dumps(row), flush=True)
        out["style"][dtag] = res
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1))
    print("V16_CHECKS DONE", flush=True)


if __name__ == "__main__":
    main()
