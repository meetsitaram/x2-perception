"""Sanity checks of the stage-1 environment: contact forces vs weight, label statistics, reference replay rewards."""
import sys, torch, mujoco
sys.path.insert(0, "/mnt/filesystem-g4/tracks/acp/code")
from acp.data import Dataset
from acp.env import AcpEnv, W, DT
dev = torch.device("cuda")
data = Dataset(sys.argv[1], dev, cache=sys.argv[2] if len(sys.argv) > 2 else "")
m = data.motion
print("clips", len(m.names), "frames", m.num_frames, "slots", data.terrain.num_slots)
for i, n in enumerate(data.set_names):
    fr = (data.clip_set[m.clip_id] == i)
    print(f"  {n:32s} clips {int((data.clip_set == i).sum()):4d} min {float(fr.sum()) / 3000:6.1f} foot contact {data.contact[fr][:, :2].float().mean():.2f} "
          f"other-body contact {data.contact[fr][:, 2:].float().mean():.4f} foot occ {data.occ[fr][:, :2].float().mean():.2f} w {float(data.clip_w[data.clip_set == i].sum()):.2f}")
env = AcpEnv(data, 512, eval_mode=True)
mass = float(mujoco.mj_getTotalmass(env.x2.model)) if hasattr(mujoco, "mj_getTotalmass") else float(env.x2.model.body_mass.sum())
env.reset_all()
ao, co = env.observe()
print("actor", ao.shape, "critic", co.shape, "finite", bool(torch.isfinite(ao).all() and torch.isfinite(co).all()))
# hold the reference pose with the PD targets (a = (q_ref - q0) / alpha): the feet should carry the weight
tot = {k: 0.0 for k in W}; fz = 0.0; n = 0; alive = torch.ones(512, dtype=torch.bool, device=dev); match = 0.0
for s in range(100):
    f1 = torch.minimum(env.f + 1, m.clip_end[m.clip_id[env.f]])
    a = (m.q[f1] - env.default_q) / env.alpha
    r, term, timeout, info = env.step(a)
    alive &= ~(term | timeout) | timeout
    Fz, _ = env.contact_forces()
    fz += float(Fz.sum(-1).mean()); match += float(info["contact_match"])
    for k in W: tot[k] += float(info["terms"][k])
    n += 1
    done = (term | timeout).nonzero().squeeze(-1)
    env.reset(done)
print(f"robot mass {mass:.1f} kg -> weight {mass * 9.81:.0f} N; mean total vertical contact force on the nine bodies {fz / n:.0f} N")
print("open-loop reference replay (PD targets = reference joints), 2 s: contact label match", round(match / n, 3))
print("  mean reward terms:", {k: round(v / n, 4) for k, v in tot.items()})
print("  weighted per-step reward:", round(DT * sum(W[k] * v / n for k, v in tot.items()), 4), "max possible", DT * 15)
