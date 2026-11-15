"""Mirror transforms of App. B.1 (Table 12). Joint vectors use [R leg, L leg, waist, R arm, L arm]."""
from __future__ import annotations

import torch

from ssr.env import H, HB_NX, HB_NY, HF_NX, HF_NY, N21, PROP_DIM
from ssr import x2 as X

S_LEG = torch.tensor(X.S_LEG, dtype=torch.float32)
S_ARM = torch.tensor(X.S_ARM, dtype=torch.float32)
S_LIN = torch.tensor([1.0, -1.0, 1.0])
S_ANG = torch.tensor([-1.0, 1.0, -1.0])
S_CMD = torch.tensor([1.0, -1.0, -1.0])
PERM = torch.tensor(X.MIRROR_PERM)
SIGN = torch.tensor(X.MIRROR_SIGN, dtype=torch.float32)


def joints(x):
    return x[..., PERM.to(x.device)] * SIGN.to(x.device, x.dtype)


def prop(o):
    d = o.device
    return torch.cat((o[..., :3] * S_ANG.to(d), o[..., 3:6] * S_LIN.to(d), o[..., 6:9] * S_CMD.to(d),
                      joints(o[..., 9:30]), joints(o[..., 30:51]), joints(o[..., 51:72])), -1)


def latent(z):
    """z = [z_f, z_b, z_p], 16 each: swap each head's halves."""
    out = []
    for c in z.chunk(3, -1):
        a, b = c.chunk(2, -1)
        out += [b, a]
    return torch.cat(out, -1)


def actor_obs(o):
    return torch.cat((prop(o[..., :PROP_DIM]), o[..., PROP_DIM:PROP_DIM + 3] * S_LIN.to(o.device), latent(o[..., PROP_DIM + 3:])), -1)


def _pair(v, s):                  # [R xyz, L xyz] -> [L xyz*s, R xyz*s]
    return torch.cat((v[..., 3:6] * s, v[..., :3] * s), -1)


def priv(s):
    d = s.device
    sy = S_LIN.to(d)
    i = PROP_DIM
    p = prop(s[..., :i])
    v = s[..., i:i + 3] * sy
    vf = _pair(s[..., i + 3:i + 9], sy)
    c = s[..., i + 9:i + 11].flip(-1)
    hand = _pair(s[..., i + 11:i + 17], sy)
    foot = _pair(s[..., i + 17:i + 23], sy)
    j = i + 23
    hb = s[..., j:j + HB_NX * HB_NY].reshape(*s.shape[:-1], HB_NX, HB_NY).flip(-1).flatten(-2)
    j += HB_NX * HB_NY
    hf = s[..., j:].reshape(*s.shape[:-1], 2, HF_NX, HF_NY).flip(-1).flip(-3).flatten(-3)
    return torch.cat((p, v, vf, c, hand, foot, hb, hf), -1)


def amp(x):
    """AMP frames (.., 63): g, v_b, w_b, q, qd, hands [R,L], feet [R,L]; handles stacked frames too."""
    d = x.device
    sy, sw = S_LIN.to(d), S_ANG.to(d)
    return torch.cat((x[..., :3] * sy, x[..., 3:6] * sy, x[..., 6:9] * sw, joints(x[..., 9:30]), joints(x[..., 30:51]),
                      _pair(x[..., 51:57], sy), _pair(x[..., 57:63], sy)), -1)


def depth(im):
    return im.flip(-1)
