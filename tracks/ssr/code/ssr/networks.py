"""SSR networks (paper Table 8, App. B): equivariant encoder, estimator, MoE actor, multi-critic, foothold model, discriminators.

Equivariant blocks use the paper's paired-channel construction: a feature of width 2C is [left C | right C], and M_c swaps
the halves. Linear layer W = [[A, B], [B, A]], shared bias; lift convolution [k, flip(k)]; later convolutions
[[k_a, k_b], [flip(k_b), flip(k_a)]]. tests/test_equivariance.py checks them numerically.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ssr.env import AMP_DIM, AMP_FRAMES, DEPTH_N, H, HB_NX, HB_NY, HF_NX, HF_NY, PRIV_DIM, PROP_DIM

LATENT_HEAD = 16                   # each of z_f, z_b, z_p (8 left + 8 right)
ACTOR_DIM = PROP_DIM + 3 + 3 * LATENT_HEAD
NUM_ACT = 21


def swap(x):
    a, b = x.chunk(2, -1)
    return torch.cat((b, a), -1)


class EqLinear(nn.Module):
    def __init__(self, c_in: int, c_out: int):                        # c_* = channels per side
        super().__init__()
        self.c_in, self.c_out = c_in, c_out
        self.A = nn.Parameter(torch.empty(c_out, c_in))
        self.B = nn.Parameter(torch.empty(c_out, c_in))
        self.bias = nn.Parameter(torch.zeros(c_out))
        nn.init.orthogonal_(self.A, gain=2 ** 0.5 / 2)
        nn.init.orthogonal_(self.B, gain=2 ** 0.5 / 2)

    def forward(self, x):
        l, r = x.chunk(2, -1)
        return torch.cat((l @ self.A.T + r @ self.B.T + self.bias, l @ self.B.T + r @ self.A.T + self.bias), -1)


class EqMLP(nn.Module):
    def __init__(self, c_in, hidden, c_out):
        super().__init__()
        d, layers = c_in, []
        for h in hidden:
            layers += [EqLinear(d, h), nn.ELU()]
            d = h
        layers.append(EqLinear(d, c_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class EqGRUCell(nn.Module):
    """Eq. 14 with paired-channel affine maps."""

    def __init__(self, c_in, c_h):
        super().__init__()
        self.ir, self.hr = EqLinear(c_in, c_h), EqLinear(c_h, c_h)
        self.iz, self.hz = EqLinear(c_in, c_h), EqLinear(c_h, c_h)
        self.in_, self.hn = EqLinear(c_in, c_h), EqLinear(c_h, c_h)
        self.c_h = c_h

    def forward(self, x, h):
        r = torch.sigmoid(self.ir(x) + self.hr(h))
        z = torch.sigmoid(self.iz(x) + self.hz(h))
        n = torch.tanh(self.in_(x) + r * self.hn(h))
        return (1 - z) * n + z * h


class LiftConv(nn.Module):
    def __init__(self, c_in, c_out, k, s):
        super().__init__()
        self.k = nn.Parameter(torch.empty(c_out, c_in, k, k))
        self.bias = nn.Parameter(torch.zeros(c_out))
        nn.init.orthogonal_(self.k, gain=1.0)
        self.s = s

    def forward(self, x):
        kk = torch.cat((self.k, self.k.flip(-1)), 0)
        return F.conv2d(x, kk, stride=self.s) + torch.cat((self.bias, self.bias))[None, :, None, None]


class EqConv(nn.Module):
    def __init__(self, c_in, c_out, k, s):
        super().__init__()
        self.ka = nn.Parameter(torch.empty(c_out, c_in, k, k))
        self.kb = nn.Parameter(torch.empty(c_out, c_in, k, k))
        self.bias = nn.Parameter(torch.zeros(c_out))
        nn.init.orthogonal_(self.ka, gain=0.7)
        nn.init.orthogonal_(self.kb, gain=0.7)
        self.s = s

    def forward(self, x):
        top = torch.cat((self.ka, self.kb), 1)
        bot = torch.cat((self.kb.flip(-1), self.ka.flip(-1)), 1)
        return F.conv2d(x, torch.cat((top, bot), 0), stride=self.s) + torch.cat((self.bias, self.bias))[None, :, None, None]


class Encoder(nn.Module):
    """Depth CNN + proprio MLP -> GRU -> fusion -> three paired latent heads. Sizes in Table 8 are total channels
    (left + right), so each side gets half."""

    def __init__(self, map_memory: bool = False):
        super().__init__()
        self.map_memory = map_memory
        self.c1, self.c2, self.c3 = LiftConv(1, 16, 8, 4), EqConv(16, 32, 4, 2), EqConv(32, 64, 3, 2)
        if map_memory:                                  # 64 x 64 x 2 -> 15 -> 7 -> 3 -> 1; every stride divides (width - kernel), which the flip needs
            self.m1, self.m2, self.m3, self.m4 = LiftConv(2, 16, 8, 4), EqConv(16, 32, 3, 2), EqConv(32, 48, 3, 2), EqConv(48, 64, 3, 1)
        self.prop = EqMLP(42 * H, [256, 128, 64], 64)
        self.gru = EqGRUCell(192 if map_memory else 128, 128)
        self.fusion = EqMLP(128, [256, 128, 64], 24)
        self.head_f, self.head_b = EqLinear(24, 8), EqLinear(24, 8)
        self.head_mu, self.head_logv = EqLinear(24, 8), EqLinear(24, 8)

    @staticmethod
    def structure(hist):
        """Operator T (Table 13) on the proprio history (.., H, 72) in [R leg, L leg, waist, R arm, L arm] order."""
        from ssr.mirror import S_ANG, S_LIN, S_CMD, S_LEG, S_ARM
        o = hist.to(hist.dtype)
        dev, dtype = hist.device, hist.dtype
        sw, sg, su = (s.to(dev, dtype) for s in (S_ANG, S_LIN, S_CMD))
        sl, sa = S_LEG.to(dev, dtype), S_ARM.to(dev, dtype)
        L, R = [o[..., :3], o[..., 3:6], o[..., 6:9]], [o[..., :3] * sw, o[..., 3:6] * sg, o[..., 6:9] * su]
        for st in (9, 30, 51):
            v = o[..., st:st + 21]
            r_leg, l_leg, waist, r_arm, l_arm = v[..., :6], v[..., 6:12], v[..., 12:13], v[..., 13:17], v[..., 17:21]
            L += [l_leg, l_arm, waist]
            R += [r_leg * sl, r_arm * sa, -waist]
        l, r = torch.cat(L, -1), torch.cat(R, -1)
        return torch.cat((l.flatten(-2), r.flatten(-2)), -1)

    def init_h(self, n, device):
        return torch.zeros(n, 256, device=device)

    def step(self, hist, depth, h, sample=True, map_img=None):
        c = F.elu(self.c1(depth[:, None]))
        c = F.elu(self.c2(c))
        c = self.c3(c)                                 # (B, 128, 1, 1) = left 64 | right 64 channels
        c = c.flatten(1)
        p = self.prop(self.structure(hist))
        cl, cr = c.chunk(2, -1)
        pl, pr = p.chunk(2, -1)
        if self.map_memory:
            m = F.elu(self.m1(map_img))
            m = F.elu(self.m2(m))
            m = F.elu(self.m3(m))
            ml, mr = self.m4(m).flatten(1).chunk(2, -1)
            x = torch.cat((cl, pl, ml, cr, pr, mr), -1)
        else:
            x = torch.cat((cl, pl, cr, pr), -1)
        h1 = self.gru(x, h)
        f = F.elu(self.fusion(h1))
        mu, logv = self.head_mu(f), self.head_logv(f).clamp(-8.0, 4.0)
        z_p = mu + (0.5 * logv).exp() * torch.randn_like(mu) if sample else mu
        return dict(z=torch.cat((self.head_f(f), self.head_b(f), z_p), -1), z_f=self.head_f(f), z_b=self.head_b(f), z_p=z_p,
                    mu=mu, logv=logv, h=h1)


def mlp(inp, hidden, out, act=nn.ELU):
    layers, d = [], inp
    for h in hidden:
        layers += [nn.Linear(d, h), act()]
        d = h
    layers.append(nn.Linear(d, out))
    return nn.Sequential(*layers)


class RunningNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4))

    @torch.no_grad()
    def update(self, x):
        x = x.reshape(-1, x.shape[-1])
        n, m, v = x.shape[0], x.mean(0), x.var(0)
        tot = self.count + n
        delta = m - self.mean
        self.mean += delta * n / tot
        self.var = (self.var * self.count + v * n + delta ** 2 * self.count * n / tot) / tot
        self.count = tot

    def forward(self, x):
        return ((x - self.mean) / (self.var + 1e-6).sqrt()).clamp(-10, 10)


class MoEActor(nn.Module):
    def __init__(self, n_exp=5):
        super().__init__()
        self.norm = RunningNorm(ACTOR_DIM)
        self.experts = nn.ModuleList([mlp(ACTOR_DIM, [1024, 512, 128], NUM_ACT) for _ in range(n_exp)])
        self.gate = mlp(ACTOR_DIM, [128], n_exp)
        self.log_std = nn.Parameter(torch.full((NUM_ACT,), -0.7))

    def forward(self, o):
        x = self.norm(o)
        w = self.gate(x).softmax(-1)
        return 8.0 * torch.tanh((torch.stack([e(x) for e in self.experts], -1) * w[..., None, :]).sum(-1) / 8.0)     # bounded mean, near-identity at normal sizes

    def std(self):
        return self.log_std.exp().clamp(min=0.2)


class Critics(nn.Module):
    def __init__(self, n=3):
        super().__init__()
        self.norm = RunningNorm(PRIV_DIM)
        self.nets = nn.ModuleList([mlp(PRIV_DIM, [512, 256, 128], 1) for _ in range(n)])

    def forward(self, s):
        x = self.norm(s)
        return torch.cat([n(x) for n in self.nets], -1)


class Aux(nn.Module):
    """Decoders for the hybrid prediction loss, the velocity estimator and the foothold imagination model."""

    def __init__(self):
        super().__init__()
        self.dec_f = mlp(LATENT_HEAD, [256, 128], 2 * HF_NX * HF_NY)
        self.dec_b = mlp(LATENT_HEAD, [256, 128], HB_NX * HB_NY)
        self.dec_p = mlp(LATENT_HEAD, [256, 128], PROP_DIM)
        self.est = mlp(PROP_DIM * H, [512, 256, 128], 3)


class Foothold(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = mlp(PRIV_DIM + NUM_ACT, [256, 128], 2 * 3)
        self.norm = RunningNorm(PRIV_DIM)

    def forward(self, s, a):
        y = self.net(torch.cat((self.norm(s), a), -1)).reshape(-1, 2, 3)
        return y[..., :2], y[..., 2].clamp(-5.0, 0.0).exp()


class Discriminators(nn.Module):
    def __init__(self, n=5):
        super().__init__()
        self.nets = nn.ModuleList([mlp(AMP_DIM * AMP_FRAMES, [512, 256, 128], 1, act=nn.ReLU) for _ in range(n)])
        self.norm = RunningNorm(AMP_DIM * AMP_FRAMES)

    def forward(self, x, i):
        return self.nets[i](self.norm(x))
