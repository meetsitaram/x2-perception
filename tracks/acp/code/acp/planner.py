"""ACP contact planner (arXiv 2610.31608): hindsight labels from the tracker's own successful rollouts, the paper's
attention planner and its MLP baseline (Table S3), the planner loss (Eq. 1-2), training on the CPU, ONNX export.

Ported from the earlier X2 work (gwbc-v6 scripts/acp/stage2/planner_data.py and planner.py: same hindsight rule and the
same Gaussian-over-scan-points loss), changed to the paper's interface: nine bodies, the tracker's own 5-step
proprioceptive history + goal command + 17 x 11 scan as input (no privileged base velocity), scan centred on the base
([-0.8, 0.8] x [-0.5, 0.5] m), targets chosen among the scan points (top-k, k = 1). No SuperMimic rollout or label is read.

    python -m acp.planner label ROLLOUTS.pt DATA.pt
    python -m acp.planner train DATA.pt OUT.pt [--arch attn|mlp] [--epochs 30]
    python -m acp.planner export OUT.pt OUT.onnx
"""
from __future__ import annotations

import sys
import time

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

L, Wd, NP, NB = 17, 11, 187, 9
SIGMA_A, HORIZON, MIN_SEG, SETTLE = 0.1, 75, 10, 5       # Table S5: 0.1 m; 75 steps; 10 steps. SETTLE: ours (steps after onset)
DECODE_CHUNK = 4096
FOOT_X, FOOT_Y = 0.8, 0.5                                # target footprint in body-yaw coordinates (Table S5)


def grid():
    xs = -0.8 + 0.1 * torch.arange(L, dtype=torch.float32)
    ys = 0.1 * (torch.arange(Wd, dtype=torch.float32) - (Wd - 1) / 2)
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    return torch.stack((gx.flatten(), gy.flatten()), -1)      # (187, 2), index = ix * 11 + iy (as env._scan)


# ------------------------------------------------------------------ hindsight labels
def label(rollouts: str, out: str):
    d = torch.load(rollouts, map_location="cpu")
    c, pts, root, alive = d["c"], d["pts"], d["root"], d["alive"]            # (T, N, 9), (T, N, 9, 3), (T, N, 4), (T, N)
    T, N = alive.shape
    ok_env = d["success"]                                                    # only successful episodes (paper)
    cf = c.float()
    # onset at t: contact starts at t and lasts at least MIN_SEG steps (shorter segments are discarded)
    cs = torch.cat((torch.zeros(1, N, NB), cf.cumsum(0)), 0)
    run = torch.zeros(T, N, NB)
    run[: T - MIN_SEG + 1] = cs[MIN_SEG:] - cs[: T - MIN_SEG + 1]
    onset = torch.zeros(T, N, NB, dtype=torch.bool)
    onset[1:] = c[1:] & ~c[:-1] & (run[1:] >= MIN_SEG)
    every = int(d["obs_every"])
    nxt_pos = torch.zeros(N, NB, 3)
    nxt_t = torch.full((N, NB), 10**6, dtype=torch.long)
    lab_pos = torch.zeros(len(d["hist"]), N, NB, 3)
    lab_occ = torch.zeros(len(d["hist"]), N, NB, dtype=torch.bool)
    for t in range(T - 2, -1, -1):                                           # scan the trajectory backward (paper)
        on = onset[t + 1] & alive[t + 1].unsqueeze(-1)
        s = min(t + 1 + SETTLE, T - 1)
        nxt_pos = torch.where(on.unsqueeze(-1), pts[s], nxt_pos)
        nxt_t = torch.where(on, torch.full_like(nxt_t, t + 1), nxt_t)
        if t % every == 0 and t // every < len(lab_pos):
            yaw = root[t, :, 3]
            cy, sy = torch.cos(yaw), torch.sin(yaw)
            dp = nxt_pos - root[t, :, None, :3]
            rel = torch.stack((cy[:, None] * dp[..., 0] + sy[:, None] * dp[..., 1], -sy[:, None] * dp[..., 0] + cy[:, None] * dp[..., 1], dp[..., 2]), -1)
            ok = ((nxt_t - t) <= HORIZON) & (rel[..., 0].abs() <= FOOT_X) & (rel[..., 1].abs() <= FOOT_Y)
            lab_pos[t // every] = rel * ok.unsqueeze(-1)
            lab_occ[t // every] = ok
    keep = alive[::every][: len(lab_pos)] & ok_env.unsqueeze(0)
    idx = keep.nonzero()
    ti, ni = idx[:, 0], idx[:, 1]
    data = dict(hist=d["hist"][ti, ni], scan=d["scan"][ti, ni], goal=d["goal"][ti, ni], pos=lab_pos[ti, ni], occ=lab_occ[ti, ni],
                env=ni.clone(), set_id=d["set_id"][ni], set_names=d["set_names"], iteration=d["iteration"])
    torch.save(data, out)
    occ = data["occ"].float().mean(0)
    print(f"[label] {int(ok_env.sum())} successful episodes of {N}, {len(ti)} samples, occupancy per body {[round(float(x), 3) for x in occ]}", flush=True)


def label_core(c, pts, root, valid, every=2):
    """Hindsight labels for trajectories stored time-first: c (T, N, 9) bool, pts (T, N, 9, 3), root (T, N, 4: x, y, z, yaw),
    valid (T, N) bool. Returns per observation step (every-th control step): target (T/every, N, 9, 3) in the body-yaw
    frame relative to the root, and occupancy (T/every, N, 9)."""
    T, N = valid.shape
    dev = c.device
    cf = (c & valid.unsqueeze(-1)).float()
    cs = torch.cat((torch.zeros(1, N, NB, device=dev), cf.cumsum(0)), 0)
    run = torch.zeros(T, N, NB, device=dev)
    run[: T - MIN_SEG + 1] = cs[MIN_SEG:] - cs[: T - MIN_SEG + 1]
    onset = torch.zeros(T, N, NB, dtype=torch.bool, device=dev)
    onset[1:] = c[1:] & ~c[:-1] & (run[1:] >= MIN_SEG) & valid[1:].unsqueeze(-1)
    K = (T + every - 1) // every
    nxt_pos = torch.zeros(N, NB, 3, device=dev)
    nxt_t = torch.full((N, NB), 10**6, dtype=torch.long, device=dev)
    lab_pos = torch.zeros(K, N, NB, 3, device=dev)
    lab_occ = torch.zeros(K, N, NB, dtype=torch.bool, device=dev)
    for t in range(T - 2, -1, -1):
        on = onset[t + 1]
        s = min(t + 1 + SETTLE, T - 1)
        nxt_pos = torch.where(on.unsqueeze(-1), pts[s], nxt_pos)
        nxt_t = torch.where(on, torch.full_like(nxt_t, t + 1), nxt_t)
        if t % every == 0:
            yaw = root[t, :, 3]
            cy, sy = torch.cos(yaw), torch.sin(yaw)
            dp = nxt_pos - root[t, :, None, :3]
            rel = torch.stack((cy[:, None] * dp[..., 0] + sy[:, None] * dp[..., 1], -sy[:, None] * dp[..., 0] + cy[:, None] * dp[..., 1], dp[..., 2]), -1)
            ok = ((nxt_t - t) <= HORIZON) & (rel[..., 0].abs() <= FOOT_X) & (rel[..., 1].abs() <= FOOT_Y)
            lab_pos[t // every] = rel * ok.unsqueeze(-1)
            lab_occ[t // every] = ok
    return lab_pos, lab_occ


class Replay:
    """The planner's hindsight replay buffer (ring), on the training device."""

    def __init__(self, cap, hist_dim, dev):
        self.cap, self.size, self.ptr = cap, 0, 0
        self.hist = torch.zeros(cap, hist_dim, dtype=torch.half, device=dev)
        self.scan = torch.zeros(cap, NP, dtype=torch.half, device=dev)
        self.goal = torch.zeros(cap, 3, device=dev)
        self.pos = torch.zeros(cap, NB, 3, device=dev)
        self.occ = torch.zeros(cap, NB, dtype=torch.bool, device=dev)

    def push(self, hist, scan, goal, pos, occ):
        n = hist.shape[0]
        if n == 0:
            return
        if n > self.cap:
            hist, scan, goal, pos, occ = (x[-self.cap:] for x in (hist, scan, goal, pos, occ)); n = self.cap
        idx = (self.ptr + torch.arange(n, device=hist.device)) % self.cap
        self.hist[idx], self.scan[idx], self.goal[idx], self.pos[idx], self.occ[idx] = hist.half(), scan.half(), goal, pos, occ
        self.ptr = int((self.ptr + n) % self.cap); self.size = min(self.cap, self.size + n)

    def sample(self, n):
        i = torch.randint(0, self.size, (n,), device=self.hist.device)
        return self.hist[i].float(), self.goal[i], self.scan[i].float(), self.pos[i], self.occ[i]


class Recorder:
    """Records the first n environments' episodes during training (state before each control step, at most MAXT steps;
    Table S5: 500) and turns the successful ones into hindsight samples once per PPO iteration."""
    MAXT = 500

    def __init__(self, n, hist_dim, dev, every=2):
        self.n, self.every, self.dev = n, every, dev
        K = self.MAXT // every
        self.hist = torch.zeros(n, K, hist_dim, dtype=torch.half, device=dev)
        self.scan = torch.zeros(n, K, NP, dtype=torch.half, device=dev)
        self.goal = torch.zeros(n, K, 3, device=dev)
        self.c = torch.zeros(n, self.MAXT, NB, dtype=torch.bool, device=dev)
        self.pts = torch.zeros(n, self.MAXT, NB, 3, device=dev)
        self.root = torch.zeros(n, self.MAXT, 4, device=dev)
        self.len = torch.zeros(n, dtype=torch.long, device=dev)
        self.state = torch.zeros(n, dtype=torch.long, device=dev)          # 0 idle, 1 recording, 2 finished successfully (waiting)
        self.ar = torch.arange(n, device=dev)
        self.episodes, self.successes = 0, 0

    def start(self, ids):
        ids = ids[ids < self.n]
        ids = ids[self.state[ids] != 2]
        self.state[ids], self.len[ids] = 1, 0

    def observe(self, env, S, goal, scan17):
        n = self.n
        on = (self.state == 1) & (self.len < self.MAXT)
        i = self.ar[on]
        if len(i) == 0:
            return
        t = self.len[i]
        p, R = S[0][:n][i], S[1][:n][i]
        self.c[i, t] = env.c_prev[:n][i]
        self.pts[i, t] = env.data.body_points(tuple(x[:n][i] for x in S))[0]
        self.root[i, t] = torch.cat((p, torch.atan2(R[:, 1, 0], R[:, 0, 0]).unsqueeze(-1)), -1)
        ev = (t % self.every) == 0
        j, k = i[ev], t[ev] // self.every
        self.hist[j, k] = env.hist[:n][j].flatten(1).half()
        self.scan[j, k], self.goal[j, k] = scan17[:n][j].half(), goal[:n][j]
        self.len[i] = t + 1

    def end(self, done, success):
        d, s = done[: self.n] & (self.state == 1), success[: self.n]
        self.episodes += int(d.sum()); self.successes += int((d & s).sum())
        self.state = torch.where(d, torch.where(s, 2, 0), self.state)

    def flush(self, replay):
        """Label the finished successful episodes and push their samples. Returns the number of samples added."""
        i = self.ar[self.state == 2]
        if len(i) == 0:
            return 0
        ln = self.len[i]
        T = int(ln.max())
        tt = torch.arange(T, device=self.dev)
        valid = (tt[:, None] < ln[None, :])
        pos, occ = label_core(self.c[i, :T].transpose(0, 1), self.pts[i, :T].transpose(0, 1), self.root[i, :T].transpose(0, 1), valid, self.every)
        K = pos.shape[0]
        keep = valid[:: self.every][:K].transpose(0, 1)                       # (n_i, K)
        e, k = keep.nonzero(as_tuple=True)
        replay.push(self.hist[i][e, k].float(), self.scan[i][e, k].float(), self.goal[i][e, k], pos.transpose(0, 1)[e, k], occ.transpose(0, 1)[e, k])
        self.state[i] = 0
        return int(len(e))


# ------------------------------------------------------------------ networks (Table S3)
class Base(nn.Module):
    def __init__(self, hist_dim):
        super().__init__()
        self.hist_dim = hist_dim
        self.register_buffer("pts", grid())
        self.register_buffer("h_mu", torch.zeros(hist_dim)); self.register_buffer("h_sd", torch.ones(hist_dim))
        self.register_buffer("s_mu", torch.zeros(1)); self.register_buffer("s_sd", torch.ones(1))

    def points(self, scan):                                                  # (B, 187, 3) scan points in the body-yaw frame
        return torch.cat((self.pts.expand(scan.shape[0], -1, -1), scan.unsqueeze(-1)), -1)

    def decode(self, hist, goal, scan):
        """-> targets (B, 9, 3) in the body-yaw frame relative to the root (zero when occupancy <= 0.5), occupancy (B, 9).
        Top-k decode with k = 1: the target is the scan point with the largest affordance weight."""
        if scan.shape[0] > DECODE_CHUNK:          # many robots: run in chunks (same result; the attention weights of 36,000
            parts = [self.decode(hist[a:a + DECODE_CHUNK], goal[a:a + DECODE_CHUNK], scan[a:a + DECODE_CHUNK])   # robots at once took ~35 GB)
                     for a in range(0, scan.shape[0], DECODE_CHUNK)]
            return torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts])
        logp, occ = self(hist, goal, scan)
        k = int(os.environ.get("ACP_DECODE_K", "1"))              # 1 = top-1 (default); k > 1: affordance-weighted mean of the top-k points (CPU test)
        if k > 1:
            v, i = logp.topk(k, -1)                                          # (B, 9, k)
            w = torch.softmax(v, -1).unsqueeze(-1)
            P = self.points(scan).unsqueeze(1).expand(-1, i.shape[1], -1, -1)
            tgt = (torch.gather(P, 2, i.unsqueeze(-1).expand(-1, -1, -1, 3)) * w).sum(2)
        else:
            i = logp.argmax(-1)                                              # (B, 9)
            tgt = torch.gather(self.points(scan), 1, i.unsqueeze(-1).expand(-1, -1, 3))
        p = torch.sigmoid(occ)
        return tgt * (p > 0.5).unsqueeze(-1), p


class MlpPlanner(Base):
    """Paper baseline: shared backbone (in -> 285 -> 285, ELU), terrain-weight head (9 x 187, softmax), occupancy head."""

    def __init__(self, hist_dim):
        super().__init__(hist_dim)
        self.body = nn.Sequential(nn.Linear(hist_dim + 3 + NP, 285), nn.ELU(), nn.Linear(285, 285), nn.ELU())
        self.w, self.o = nn.Linear(285, NB * NP), nn.Linear(285, NB)

    def forward(self, hist, goal, scan):
        x = torch.cat(((hist - self.h_mu) / self.h_sd, goal, (scan - self.s_mu) / self.s_sd), -1)
        h = self.body(x)
        return self.w(h).view(-1, NB, NP).log_softmax(-1), self.o(h)


class AttnPlanner(Base):
    """Paper planner: terrain-point embedding 3 -> 16, height-map conv encoder 1 -> 16 -> 48 (5 x 5), projection 64 -> 96 (keys and
    values), pooled terrain context 96 -> 64 -> 64, proprioception-goal-context encoder -> 128 -> 128 -> 96, nine body embeddings
    with FiLM (96 -> 64 -> 192), one cross-attention layer (d = 96, 32 heads), occupancy head 160 -> 512 -> 512 -> 512 -> 1."""

    def __init__(self, hist_dim):
        super().__init__(hist_dim)
        self.pt = nn.Sequential(nn.Linear(3, 16), nn.ELU())
        self.conv = nn.Sequential(nn.Conv2d(1, 16, 5, padding=2), nn.ELU(), nn.Conv2d(16, 48, 5, padding=2), nn.ELU())
        self.proj = nn.Linear(64, 96)
        self.ctx = nn.Sequential(nn.Linear(96, 64), nn.ELU(), nn.Linear(64, 64))
        self.enc = nn.Sequential(nn.Linear(hist_dim + 3 + 64, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 96))
        self.emb = nn.Parameter(torch.randn(NB, 96) * 0.1)
        self.film = nn.Sequential(nn.Linear(96, 64), nn.ReLU(), nn.Linear(64, 192))
        self.heads, self.d = 32, 96
        self.q, self.k, self.v, self.out = nn.Linear(96, 96), nn.Linear(96, 96), nn.Linear(96, 96), nn.Linear(96, 96)
        self.occ = nn.Sequential(nn.Linear(160, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 1))

    def forward(self, hist, goal, scan):
        B = scan.shape[0]
        sn = (scan - self.s_mu) / self.s_sd
        pts = torch.cat((self.pts.expand(B, -1, -1), sn.unsqueeze(-1)), -1)
        cm = self.conv(sn.view(B, 1, L, Wd)).flatten(2).transpose(1, 2)        # (B, 187, 48), same point order as the scan
        kv = self.proj(torch.cat((self.pt(pts), cm), -1))                     # (B, 187, 96)
        pooled = self.ctx(kv).amax(1)                                         # (B, 64)
        shared = self.enc(torch.cat(((hist - self.h_mu) / self.h_sd, goal, pooled), -1))
        g = self.film(self.emb)                                               # (9, 192): scale and shift per body
        qy = shared.unsqueeze(1) * (1 + g[:, :96]) + g[:, 96:]                # (B, 9, 96)
        H, dh = self.heads, self.d // self.heads
        q = self.q(qy).view(B, NB, H, dh).transpose(1, 2)
        k = self.k(kv).view(B, NP, H, dh).transpose(1, 2)
        v = self.v(kv).view(B, NP, H, dh).transpose(1, 2)
        att = (q @ k.transpose(-1, -2) / dh**0.5).softmax(-1)                 # (B, H, 9, 187)
        o = self.out((att @ v).transpose(1, 2).reshape(B, NB, self.d))
        alpha = att.mean(1)                                                   # affordance weights over the scan points
        occ = self.occ(torch.cat((o, pooled.unsqueeze(1).expand(-1, NB, -1)), -1)).squeeze(-1)
        return (alpha + 1e-9).log(), occ


def planner_loss(net, hist, goal, scan, pos, occ):
    """Eq. 2: masked cross-entropy to a 3-D Gaussian (sigma_a) over the scan points + BCE on occupancy."""
    logp, o = net(hist, goal, scan)
    pts = net.points(scan)                                                   # (B, 187, 3)
    d2 = ((pts.unsqueeze(1) - pos.unsqueeze(2)) ** 2).sum(-1)                # (B, 9, 187)
    tgt = torch.softmax(-d2 / (2 * SIGMA_A**2), -1)
    of = occ.float()
    l_aff = -(of * (tgt * logp).sum(-1)).sum() / of.sum().clamp(min=1.0)
    l_occ = F.binary_cross_entropy_with_logits(o, of)
    return l_aff + l_occ, l_aff, l_occ


def build(arch, hist_dim):
    return (AttnPlanner if arch == "attn" else MlpPlanner)(hist_dim)


def train(data_path, out, arch="attn", epochs=30, lr=1e-4, batch=4096, threads=16):
    torch.set_num_threads(threads)
    torch.manual_seed(0)
    D = torch.load(data_path, map_location="cpu")
    hist, scan, goal, pos, occ = D["hist"].float(), D["scan"].float(), D["goal"].float(), D["pos"], D["occ"]
    envs = D["env"]
    val = (envs % 10) == 0                                                   # a tenth of the episodes held out
    tr, va = (~val).nonzero().squeeze(-1), val.nonzero().squeeze(-1)
    net = build(arch, hist.shape[1])
    if os.environ.get("PLANNER_INIT"):            # fine-tune from an existing planner (2026-11-12)
        net.load_state_dict(torch.load(os.environ["PLANNER_INIT"], map_location="cpu", weights_only=False)["net"])
    net.h_mu.copy_(hist[tr].mean(0)); net.h_sd.copy_(hist[tr].std(0).clamp(min=1e-3))
    net.s_mu.copy_(scan[tr].mean()); net.s_sd.copy_(scan[tr].std().clamp(min=1e-3))
    opt = torch.optim.Adam(net.parameters(), lr=lr)                          # Table S5: Adam 1e-4, 4,096 samples per update
    nparam = sum(p.numel() for p in net.parameters())
    print(f"[train] arch {arch} parameters {nparam} train {len(tr)} val {len(va)} samples", flush=True)
    t0 = time.time()
    for ep in range(epochs):
        perm = tr[torch.randperm(len(tr))]
        for b in range(0, len(perm), batch):
            i = perm[b:b + batch]
            loss, _, _ = planner_loss(net, hist[i], goal[i], scan[i], pos[i], occ[i])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if ep % 5 == 4 or ep == epochs - 1:
            with torch.no_grad():
                tot, la, lo, errs, acc, n = 0.0, 0.0, 0.0, [], 0.0, 0
                for b in range(0, len(va), 16384):
                    i = va[b:b + 16384]
                    l, a_, o_ = planner_loss(net, hist[i], goal[i], scan[i], pos[i], occ[i])
                    tgt, p = net.decode(hist[i], goal[i], scan[i])
                    logp, _ = net(hist[i], goal[i], scan[i])
                    best = torch.gather(net.points(scan[i]), 1, logp.argmax(-1).unsqueeze(-1).expand(-1, -1, 3))
                    errs.append((best - pos[i]).norm(dim=-1)[occ[i]])
                    acc += float(((p > 0.5) == occ[i])[:, :2].float().sum()); n += len(i) * 2
                    tot += float(l) * len(i); la += float(a_) * len(i); lo += float(o_) * len(i)
                e = torch.cat(errs)
                stats = dict(epoch=ep + 1, val_loss=round(tot / len(va), 4), val_affordance=round(la / len(va), 4), val_occupancy=round(lo / len(va), 4),
                             target_error_median_m=round(float(e.median()), 4), target_error_p90_m=round(float(e.quantile(0.9)), 4),
                             feet_occupancy_accuracy=round(acc / n, 4), minutes=round((time.time() - t0) / 60, 1))
                print("[train]", stats, flush=True)
    torch.save(dict(net=net.state_dict(), arch=arch, hist_dim=hist.shape[1], stats=stats, data=data_path, tracker_iteration=D["iteration"],
                    parameters=nparam), out)
    print("->", out, flush=True)


class Export(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, hist, goal, scan):
        return self.net.decode(hist, goal, scan)


def export(ckpt, out):
    ck = torch.load(ckpt, map_location="cpu")
    net = build(ck["arch"], ck["hist_dim"]); net.load_state_dict(ck["net"]); net.eval()
    ex = (torch.zeros(1, ck["hist_dim"]), torch.zeros(1, 3), torch.zeros(1, NP))
    torch.onnx.export(Export(net), ex, out, input_names=["proprio_history", "goal_command", "scan_17x11"],
                      output_names=["contact_targets", "occupancy"], opset_version=17, dynamo=False)
    print("->", out)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "label":
        label(sys.argv[2], sys.argv[3])
    elif cmd == "train":
        a = sys.argv[4:]
        train(sys.argv[2], sys.argv[3], arch=a[a.index("--arch") + 1] if "--arch" in a else "attn",
              epochs=int(a[a.index("--epochs") + 1]) if "--epochs" in a else 30)
    elif cmd == "export":
        export(sys.argv[2], sys.argv[3])
