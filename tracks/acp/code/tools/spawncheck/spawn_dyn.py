# Dynamic spawn check of goal tiles (ACP, 2026-11-14): every tile is reset R times (training resets: yaw U(+-0.4), joint
# noise, base height +0-5 cm) and held for STEPS control steps with zero action (default pose); a tile faults when the simcheck
# detector fires in any repeat. Physics as in training (same env vars, njmax 450). CPU only.
# usage: python spawn_dyn.py CFG OUT.json [--tiles names.json] [--repeats 4] [--envs 256] [--steps 10]
import os, sys, json, argparse, collections, time
sys.path.insert(0, os.getcwd())
import torch
from acp.data import Dataset
from acp.env import AcpEnv

ap = argparse.ArgumentParser()
ap.add_argument("cfg"); ap.add_argument("out")
ap.add_argument("--tiles", default=""); ap.add_argument("--repeats", type=int, default=4)
ap.add_argument("--envs", type=int, default=256); ap.add_argument("--steps", type=int, default=10)
ap.add_argument("--cache", default="")
ap.add_argument("--policy", default="", help="checkpoint: act with its actor (sampled) and planner (eta 1) instead of zero action")
a = ap.parse_args()
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
dev = torch.device("cpu")
data = Dataset(a.cfg, dev, cache=a.cache)
m = data.motion
gt = data.cfg_goal_tiles if hasattr(data, "cfg_goal_tiles") else None
name2clip = {}
for i, nm in enumerate(m.names):
    if "/" in nm:
        name2clip[nm.split("/", 1)[1]] = i
cfg = json.load(open(a.cfg))
tiles = json.load(open(cfg["goal_tiles"]["file"]))["tiles"]
sel = set(json.load(open(a.tiles))) if a.tiles else None
names = [t["name"] for t in tiles if (sel is None or t["name"] in sel)]
fam_of = {t["name"]: t["family"] for t in tiles}
jobs = [n for n in names for _ in range(a.repeats)]
N = a.envs
env = AcpEnv(data, N, seed=1, njmax=450, nconmax=64)
env.paper_s3 = True
env.setup_goals(1.0)
ac = None
if a.policy:
    import acp.train as TR, acp.planner as P
    ck = torch.load(a.policy, map_location="cpu", weights_only=False)
    ac = TR.ActorCritic(env.actor_dim, env.critic_dim, env.NJ); ac.load_state_dict(ck["model"]); ac.eval()
    env.csi = False
    pl = P.build("attn", 5 * env.PROP); pl.load_state_dict(ck["planner"]); pl.eval()
    env.planner = pl; env.eta_obs = 1.0; env.eta_rew = 1.0
cap = {}
_upd = env.fault_det.update
def upd(qpos, qvel, xpos=None, depth=None):
    f, why = _upd(qpos, qvel, xpos, depth)
    cap["f"] = cap.get("f", torch.zeros_like(f)) | f
    for k, w in why.items():
        cap[k] = cap.get(k, torch.zeros_like(w)) | w
    if depth is not None:
        cap["dmax"] = torch.maximum(cap.get("dmax", torch.zeros_like(depth)), depth)
    return f, why
env.fault_det.update = upd
res = collections.defaultdict(lambda: dict(n=0, fault=0, causes=collections.Counter(), depth_max=0.0, first_step=[]))
t0 = time.time()
for b in range(0, len(jobs), N):
    chunk = jobs[b:b + N]
    ids = torch.tensor([name2clip[n] for n in chunk] + [name2clip[chunk[0]]] * (N - len(chunk)))
    data.goal_pool = ids.view(N, 1, 1).expand(N, 10, 1).contiguous()
    data.goal_pool_n = torch.ones(N, 10, dtype=torch.long)
    env.g_fam = torch.arange(N); env.g_level = torch.zeros(N, dtype=torch.long)
    env.reset_all(); ao, co = env.observe()
    cap.clear(); first = torch.full((N,), -1)
    for s in range(a.steps):
        before = cap.get("f", torch.zeros(N, dtype=torch.bool)).clone()
        with torch.no_grad():
            act = ac.dist(ao).sample() if ac is not None else torch.zeros(N, env.NJ)
        env.step(act); ao, co = env.observe()
        first = torch.where((cap["f"] & ~before) & (first < 0), torch.full_like(first, s), first)
        # do not let auto-resets hide later faults: the env resets done envs itself; the latch keeps the first fault
    for k, n in enumerate(chunk):
        r = res[n]; r["n"] += 1
        if bool(cap["f"][k]):
            r["fault"] += 1; r["first_step"].append(int(first[k]))
            for c in ("nan", "qvel", "depth", "speed_jump"):
                if c in cap and bool(cap[c][k]): r["causes"][c] += 1
        r["depth_max"] = max(r["depth_max"], float(cap.get("dmax", torch.zeros(N))[k]))
    print(f"batch {b // N + 1}/{(len(jobs) + N - 1) // N} {time.time() - t0:.0f}s", flush=True)
out = {n: dict(family=fam_of[n], n=r["n"], fault=r["fault"], causes=dict(r["causes"]), depth_max=round(r["depth_max"], 3),
               first_step=r["first_step"]) for n, r in res.items()}
json.dump(out, open(a.out, "w"), indent=0)
fam = collections.defaultdict(lambda: [0, 0])
for n, r in out.items():
    fam[r["family"]][0] += 1; fam[r["family"]][1] += int(r["fault"] > 0)
for f in sorted(fam): print(f"{f:20s} tiles {fam[f][0]:4d} faulting {fam[f][1]:4d}")
print("DONE", time.time() - t0)
