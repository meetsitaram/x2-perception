"""CPU smoke of v14 = v13 + slope turn stations (station 3) + back_scale + drift penalty. Usage: ckpt n level steps
Robots i % 9 = type; odd robots are forced onto a slope-turn route of their type (any level) if one exists."""
import sys, time, json, torch
sys.path.insert(0, "/mnt/filesystem-g4/tracks/ssr/code")
from ssr import networks as N, terrain as T
from ssr.env import SSREnv
torch.set_grad_enabled(False)
TY = ("tight_spots", "log_corridors", "homes", "street_front", "stairs", "wedges", "single_beams", "beam_fields", "blocks")
t0 = time.time()
r = T.add_extended(T.generate(0), types=TY, stairs_inset=True, prop_routes=True, stations=1, slope_turns=2)
R = r["ext"]["routes"]; tl = r["ext"]["tiles"]
cnt = {}
for ro in R:
    k = (tl[ro["tile"]]["name"], ro.get("station", 0)); cnt[k] = cnt.get(k, 0) + 1
print("routes", len(R), "slope turns by type", {k[0]: v for k, v in cnt.items() if k[1] == 3}, "max wp", max(len(ro["wp"]) for ro in R), flush=True)
n, lvl, steps = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
env = SSREnv(n, "cpu", raster=r, tiles=True, train=True, camera="chest_rs", seed=5, ext_phys=24)
env.ext_frac = 1.0; env.side_soft = 2.0; env.back_scale = 0.5; env.drift_w = 0.5; env.creep_w = 0.5; env.ext_hold_r = 0.10
env.ext_type[:] = torch.arange(n) % len(TY); env.ext_level[:] = lvl
st = torch.tensor([ro.get("station", 0) for ro in R])
forced = {}
for i in range(1, n, 2):
    c = ((env.r_type == env.ext_type[i]) & (st == 3)).nonzero().flatten()
    if len(c):
        forced[i] = int(c[torch.randint(len(c), (1,))])
orig = env._ext_pick
def pick(ids_):
    x, y, yaw = orig(ids_)
    for k, i in enumerate(ids_.tolist()):
        if i in forced:
            rid = forced[i]; env.route[i] = rid; s0 = env.r_start[rid]; x[k], y[k], yaw[k] = s0[0], s0[1], s0[2]
    return x, y, yaw
env._ext_pick = pick
ids = torch.arange(n); env.ep[:] = 1; env.reset(ids); env.refresh(); env._contacts(); env._update_obs(ids); env._update_cmd()
half = {i: (len(R[rid]["wp"]) + 1) // 2 for i, rid in forced.items()}      # waypoint index just after the turn point
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
enc, aux, actor = N.Encoder(False), N.Aux(), N.MoEActor()
enc.load_state_dict(ck["enc"]); aux.load_state_dict(ck["aux"]); actor.load_state_dict(ck["actor"])
h = enc.init_h(n, "cpu"); env.cnt = {}
res = {i: dict(turned=False, done=None, back_m=0.0) for i in forced}
drift_sum, nstep, creep_sum = 0.0, 0, 0.0
for t in range(steps):
    e = enc.step(env.hist, env.depth, h, sample=False); h = e["h"]
    out = env.step(actor(torch.cat((env.prop, aux.est(env.hist.flatten(1)), e["z"]), -1)).clamp(-8, 8))
    if "creep" in env.terms:
        creep_sum = creep_sum + float(env.terms["creep"].mean()) if t else float(env.terms["creep"].mean())
    if "drift" in env.terms:
        drift_sum += float(env.terms["drift"].mean()); nstep += 1
    vb = env.s["v_b"][:, 0]
    for i in res:
        if res[i]["done"] is None:
            if int(env.wpi[i]) >= half[i]:
                res[i]["turned"] = True
            res[i]["back_m"] += max(0.0, -float(vb[i])) * 0.02
            if bool(out["done"][i]):
                res[i]["done"] = t
    h = torch.where(out["done"][:, None], torch.zeros_like(h), h)
print("sim %.0f s" % (time.time() - t0))
print("slope-turn robots", len(res), "turned", sum(v["turned"] for v in res.values()), "ended before turning", sum(v["done"] is not None and not v["turned"] for v in res.values()),
      "median backward m", sorted(v["back_m"] for v in res.values())[len(res) // 2] if res else None)
print("drift term mean", round(drift_sum / max(nstep, 1), 4))
print("creep term mean", round(creep_sum / max(nstep, 1), 4))
print("done causes", {k[5:]: int(v) for k, v in env.cnt.items() if k.startswith("done_") and int(v)})
print("DONE")
