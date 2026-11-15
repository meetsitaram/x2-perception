"""Symmetry checks on a trained checkpoint (paper Sec. 3.3, Eq. 4-5), added for v15's pre-flight (coordinator, 2026-11-13).

- encoder: exact equivariance by construction (Eq. 4): mirror(z(o)) == z(mirror(o)), also for the GRU state;
- actor: symmetric by mirrored PPO samples (Eq. 5), so approximate: mirror(pi(o)) vs pi(mirror(o)) on observations drawn
  from the actor's own running statistics, relative error below ACTOR_TOL;
- mirror maps are involutions on actor, privileged, joint and style vectors.

python tests/test_equivariance.py [ckpt]     (CPU; default ckpt $SSR_CKPT or ../runs/r7/last.pt). Exit 0 = all pass.
"""
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from ssr import mirror as M
from ssr import networks as N
from ssr.env import AMP_DIM, AMP_FRAMES, DEPTH_N, H, PRIV_DIM, PROP_DIM

ACTOR_TOL = 0.10          # mean |mirror(pi(o)) - pi(mirror(o))| / mean |pi(o)|
ENC_TOL = 1e-4


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    enc, actor = N.Encoder(), N.MoEActor()
    enc.load_state_dict(ck["enc"])
    actor.load_state_dict(ck["actor"])
    return ck, enc.eval(), actor.eval()


def encoder_equivariance(enc, n=64):
    g = torch.Generator().manual_seed(0)
    hist, depth = torch.randn(n, H, PROP_DIM, generator=g), torch.rand(n, DEPTH_N, DEPTH_N, generator=g)
    h = torch.randn(n, 256, generator=g) * 0.2
    with torch.no_grad():
        a = enc.step(hist, depth, h, sample=False)
        b = enc.step(M.prop(hist), M.depth(depth), N.swap(h), sample=False)
    sc = max(a["z"].abs().max().item(), 1e-6)
    err = (M.latent(a["z"]) - b["z"]).abs().max().item() / sc
    err_h = (N.swap(a["h"]) - b["h"]).abs().max().item() / max(a["h"].abs().max().item(), 1e-6)
    return dict(z_rel=err, h_rel=err_h, ok=max(err, err_h) < ENC_TOL)


def actor_symmetry(actor, n=4096):
    g = torch.Generator().manual_seed(1)
    mean, sd = actor.norm.mean, (actor.norm.var + 1e-6).sqrt()
    o = mean + sd * torch.randn(n, len(mean), generator=g) * 0.5
    with torch.no_grad():
        a = actor(o)
        b = actor(M.actor_obs(o))
    rel = ((M.joints(a) - b).abs().mean() / a.abs().mean().clamp(min=1e-6)).item()
    return dict(rel=rel, ok=rel < ACTOR_TOL)


def involutions():
    g = torch.Generator().manual_seed(2)
    out = {}
    for name, f, d in (("actor_obs", M.actor_obs, N.ACTOR_DIM), ("priv", M.priv, PRIV_DIM), ("joints", M.joints, 21), ("amp", M.amp, AMP_DIM)):
        x = torch.randn(8, d, generator=g)
        out[name] = (f(f(x)) - x).abs().max().item()
    return dict(err=out, ok=max(out.values()) < 1e-5)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("SSR_CKPT", str(pathlib.Path(__file__).resolve().parents[2] / "runs/r7/last.pt"))
    ck, enc, actor = load(path)
    res = dict(ckpt=path, iter=int(ck["iter"]), encoder=encoder_equivariance(enc), actor=actor_symmetry(actor), involutions=involutions())
    res["passed"] = all(v["ok"] for k, v in res.items() if isinstance(v, dict))
    print(json.dumps(res))
    sys.exit(0 if res["passed"] else 1)


if __name__ == "__main__":
    main()
