"""CPU smoke of --ext_phys: all five extended types (no clearance filter), K wall slots per world, 3,550 policy deterministic.
Usage: ckpt n level types K"""
import sys, time, json, torch
sys.path.insert(0, "/mnt/filesystem-g4/tracks/ssr/code")
from ssr import networks as N, terrain as T
from ssr.env import SSREnv
torch.set_grad_enabled(False)
t0 = time.time()
r = T.add_extended(T.generate(0))
K = int(sys.argv[5]); n = int(sys.argv[2]); lvl = int(sys.argv[3])
nb = [len(t.get("phys", [])) for t in r["ext"]["tiles"]]
print("tiles", len(nb), "routes", len(r["ext"]["routes"]), "phys boxes per tile max", max(nb), "mean %.1f" % (sum(nb) / len(nb)), flush=True)
env = SSREnv(n, "cpu", raster=r, tiles=True, train=True, camera="chest_rs", seed=5, ext_phys=K)
print("env built %.1f s, box overflow (route-nearest kept) %d, tight route segments %d" % (time.time() - t0, env.ext_box_overflow, int((env.r_segclr < 0.39).sum())), flush=True)
env.ext_frac = 1.0; env.side_soft = 2.0
tl = [int(c) for c in sys.argv[4].split(",")]
env.ext_type[:] = torch.tensor(tl)[torch.arange(n) % len(tl)]; env.ext_level[:] = lvl
env.ep[:] = 1; ids = torch.arange(n); env.reset(ids); env.refresh(); env._contacts(); env._update_obs(ids); env._update_cmd()
print("ext robots", int(env.ext.sum()), "slot z of robot 0", [round(float(v), 2) for v in env.g_pos[0, env.wall_ids[:4], 2]], flush=True)
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
enc, aux, actor = N.Encoder(False), N.Aux(), N.MoEActor()
enc.load_state_dict(ck["enc"]); aux.load_state_dict(ck["aux"]); actor.load_state_dict(ck["actor"])
h = enc.init_h(n, "cpu"); env.cnt = {}
st = dict(steps=0, strong_steps=0, arm_steps=0, arm_free_steps=0, max_strong=0.0, max_arm=0.0, side_cmd_steps=0, wall_body=0.0, wall_arm=0.0)
t1 = time.time()
for t in range(1000):
    e = enc.step(env.hist, env.depth, h, sample=False); h = e["h"]
    out = env.step(actor(torch.cat((env.prop, aux.est(env.hist.flatten(1)), e["z"]), -1)).clamp(-8, 8))
    wf = env.wall_f; st["steps"] += n
    st["strong_steps"] += int((wf[:, 2] > 1).sum()); st["arm_steps"] += int((wf[:, 1] > 1).sum())
    st["arm_free_steps"] += int(((wf[:, 1] > 1) & (wf[:, 1] <= 40) & (env.arm_t <= 0.5)).sum())
    st["max_strong"] = max(st["max_strong"], float(wf[:, 2].max())); st["max_arm"] = max(st["max_arm"], float(wf[:, 1].max()))
    st["side_cmd_steps"] += int((env.ext & (env.cmd[:, 1].abs() > 0.05)).sum())
    st["wall_body"] += float(env.terms["wall_body"].sum()); st["wall_arm"] += float(env.terms["wall_arm"].sum())
    h = torch.where(out["done"][:, None], torch.zeros_like(h), h)
print("sim %.1f s for 1000 steps" % (time.time() - t1))
print("wall", json.dumps({k: round(v, 1) if isinstance(v, float) else v for k, v in st.items()}))
print("ext", json.dumps({k: (float(v) if not isinstance(v, dict) else v) for k, v in env.ext_stats().items()}))
print("done causes", {k[5:]: int(v) for k, v in env.cnt.items() if k.startswith("done_") and int(v)})
