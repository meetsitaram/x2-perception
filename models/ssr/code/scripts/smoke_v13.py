"""CPU smoke of v13 (stairs inset, prop routes, 9 ext types, wall stations). Usage: ckpt n level steps
Robots i % 9 = type; odd robots are forced onto a station route of their type if one exists at that level."""
import sys, time, json, torch
sys.path.insert(0, "/mnt/filesystem-g4/tracks/ssr/code")
from ssr import networks as N, terrain as T
from ssr.env import SSREnv
torch.set_grad_enabled(False)
TY = ("tight_spots", "log_corridors", "homes", "street_front", "stairs", "wedges", "single_beams", "beam_fields", "blocks")
t0 = time.time()
r = T.add_extended(T.generate(0), types=TY, stairs_inset=True, prop_routes=True, stations=1)
n, lvl, steps = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
env = SSREnv(n, "cpu", raster=r, tiles=True, train=True, camera="chest_rs", seed=5, ext_phys=24)
print("env %.0f s, routes %d, box overflow %d, tight segs %d" % (time.time() - t0, len(r["ext"]["routes"]), env.ext_box_overflow, int((env.r_segclr < 0.39).sum())), flush=True)
env.ext_frac = 1.0; env.side_soft = 2.0
env.ext_type[:] = torch.arange(n) % len(TY); env.ext_level[:] = lvl
env.ep[:] = 1; ids = torch.arange(n); env.reset(ids)
st = torch.tensor([ro.get("station", 0) for ro in r["ext"]["routes"]])
# force odd robots onto station routes (approach for i%4==1, departure for i%4==3) and re-place them
forced = []
for i in range(1, n, 2):
    want = 1 if i % 4 == 1 else 2
    c = ((env.r_type == env.ext_type[i]) & (st == want)).nonzero().flatten()
    if len(c):
        c = c[torch.randint(len(c), (1,))]
        forced.append((i, int(c), want))
print("forced", len(forced), flush=True)
def place():
    for i, rid, _ in forced:
        env.route[i] = rid; env.wpi[i] = 0; env.hold_t[i] = 0.0
        s0 = env.r_start[rid]
        env.qpos[i, 0], env.qpos[i, 1] = s0[0], s0[1]
place()
# re-run reset with the forced routes: monkeypatch _ext_pick to keep them
orig = env._ext_pick
def pick(ids_):
    x, y, yaw = orig(ids_)
    for k, i in enumerate(ids_.tolist()):
        for j, rid, _ in forced:
            if j == i:
                env.route[i] = rid; s0 = env.r_start[rid]; x[k], y[k], yaw[k] = s0[0], s0[1], s0[2]
    return x, y, yaw
env._ext_pick = pick
env.ep[:] = 1; env.reset(ids); env.refresh(); env._contacts(); env._update_obs(ids); env._update_cmd()
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
enc, aux, actor = N.Encoder(False), N.Aux(), N.MoEActor()
enc.load_state_dict(ck["enc"]); aux.load_state_dict(ck["aux"]); actor.load_state_dict(ck["actor"])
h = enc.init_h(n, "cpu"); env.cnt = {}
fid = {i: (rid, w) for i, rid, w in forced}
res = {i: dict(kind=w, held=False, done=None, wall_min=9.9) for i, (rid, w) in fid.items()}
for t in range(steps):
    e = enc.step(env.hist, env.depth, h, sample=False); h = e["h"]
    out = env.step(actor(torch.cat((env.prop, aux.est(env.hist.flatten(1)), e["z"]), -1)).clamp(-8, 8))
    for i in res:
        if res[i]["done"] is None:
            if float(env.hold_t[i]) > 0.5:
                res[i]["held"] = True
            if bool(out["done"][i]):
                res[i]["done"] = t
    h = torch.where(out["done"][:, None], torch.zeros_like(h), h)
print("sim %.0f s" % (time.time() - t0))
for w in (1, 2):
    rr = [v for v in res.values() if v["kind"] == w]
    print("station kind", w, "robots", len(rr), "held", sum(v["held"] for v in rr), "ended early", sum(v["done"] is not None and not v["held"] for v in rr))
print("ext", json.dumps({k: (float(v) if not isinstance(v, dict) else v) for k, v in env.ext_stats().items()})[:1500])
print("done causes", {k[5:]: int(v) for k, v in env.cnt.items() if k.startswith("done_") and int(v)})
