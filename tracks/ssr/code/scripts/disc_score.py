"""Does the style reward separate standing from walking? Scores one checkpoint's discriminators (CPU) on reference walking
windows, reference standing windows, and windows produced by given policies on flat ground.

PROBE_DEVICE=cpu X2_PLANT=vendor_20260823 python scripts/disc_score.py --disc ../runs/r3/last.pt --policies stand=../runs/r3/last.pt shuffle=../runs/d1/last.pt
"""
import argparse
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from ssr import mirror as M
from ssr import networks as N
from ssr import terrain as T
from ssr.env import AMP_FRAMES, SSREnv


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--disc", required=True)
    ap.add_argument("--policies", nargs="*", default=[])
    ap.add_argument("--amp_file", default="data/amp.npz")
    ap.add_argument("--speed", type=float, default=0.6)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    dev = torch.device(os.environ.get("PROBE_DEVICE", "cpu"))
    ck = torch.load(a.disc, map_location=dev, weights_only=False)
    disc = N.Discriminators().to(dev)
    disc.load_state_dict(ck["disc"])
    disc.eval()
    style = lambda x, i: (1 - 0.25 * (disc(x, i).squeeze(-1) - 1) ** 2).clamp(min=0.0)
    z = np.load(ROOT / a.amp_file)
    out = dict(disc=a.disc, iteration=int(ck["iter"]), rows=[])

    def row(name, x, i):
        d = disc(x, i).squeeze(-1)
        r = style(x, i)
        out["rows"].append(dict(windows=name, discriminator=T.TYPES[i], n=len(x), d_mean=round(float(d.mean()), 3), style_reward_mean=round(float(r.mean()), 3),
                                style_p10=round(float(r.quantile(0.1)), 3), style_p90=round(float(r.quantile(0.9)), 3)))
        print(out["rows"][-1], flush=True)

    for i, name in enumerate(T.TYPES):
        f = torch.as_tensor(z[name], dtype=torch.float32, device=dev)
        n = len(f)
        ok = np.ones(n, bool)
        for s in z[name + "_starts"]:
            ok[max(0, s - AMP_FRAMES + 1):s] = False
        ok[-AMP_FRAMES + 1:] = False
        sp = np.linalg.norm(z[name][:, 3:5], axis=1)
        win = np.convolve(sp, np.ones(AMP_FRAMES) / AMP_FRAMES)[AMP_FRAMES - 1:]
        win = np.concatenate((win, np.zeros(n - len(win))))[:n]
        for tag, m in (("reference, moving (0.4 m/s or more)", ok & (win >= 0.4)), ("reference, slow (0.2-0.4 m/s)", ok & (win >= 0.2) & (win < 0.4)), ("reference, standing (under 0.05 m/s)", ok & (win < 0.05))):
            idx = torch.as_tensor(np.nonzero(m)[0][:: max(1, int(m.sum()) // 4000)], device=dev)
            if len(idx) < 16:
                continue
            x = f[idx[:, None] + torch.arange(AMP_FRAMES, device=dev)].reshape(len(idx), -1)
            row(tag, x, i)
    for spec in a.policies:
        tag, path = spec.split("=")
        pk = torch.load(path, map_location=dev, weights_only=False)
        enc, aux, actor = N.Encoder(False).to(dev), N.Aux().to(dev), N.MoEActor().to(dev)
        enc.load_state_dict(pk["enc"]); aux.load_state_dict(pk["aux"]); actor.load_state_dict(pk["actor"])
        n = 16
        env = SSREnv(n, str(dev), train=False, tiles=True)
        env.flat_only = True
        ids = torch.arange(n, device=dev)
        env.type[:], env.level[:] = T.FLAT, 0
        env.reset(ids)
        env.speed[:], env.heading[:] = a.speed, 0.0
        env.qpos[:, 3], env.qpos[:, 4:7] = 1.0, 0.0
        env.refresh(); env._update_cmd(); env._contacts(); env._update_obs(ids); env._update_cmd()
        for noise in (False, True):
            h = enc.init_h(n, dev)
            xs, vx = [], []
            for t in range(200):
                e = enc.step(env.hist, env.depth, h, sample=False)
                h = e["h"]
                o = torch.cat((env.prop, aux.est(env.hist.flatten(1)), e["z"]), -1)
                act = actor(o)
                if noise:
                    act = act + actor.std() * torch.randn_like(act)
                env.step(act.clamp(-8, 8))
                if t >= 50:
                    xs.append(env.amp_hist.flatten(1).clone()); vx.append(float(env.qvel[:, 0].mean()))
            row(f"policy '{tag}' (it {int(pk['iter'])}), cmd {a.speed} m/s, mean speed {np.mean(vx):.2f} m/s, {'with' if noise else 'no'} action noise", torch.cat(xs), T.FLAT)
        del env
    if a.out:
        pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
