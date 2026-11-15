"""CPU replay: diverged / done per ext type for v13 environment variants. Usage: ckpt n level steps variant
variant: v12 | v13 | no_inset | no_props | no_stations | types5"""
import sys, time, json, torch
sys.path.insert(0, "/mnt/filesystem-g4/tracks/ssr/code")
from ssr import networks as N, terrain as T
from ssr.env import SSREnv
torch.set_grad_enabled(False); torch.manual_seed(0)
T5 = ("tight_spots", "log_corridors", "homes", "street_front", "stairs")
T9 = T5 + ("wedges", "single_beams", "beam_fields", "blocks")
ck, n, lvl, steps, var = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
cfg = dict(types=T9, stairs_inset=True, prop_routes=True, stations=1)
if var == "v12": cfg = dict(types=T5, stairs_inset=False, prop_routes=False, stations=0)
if var == "no_inset": cfg["stairs_inset"] = False
if var == "no_props": cfg["prop_routes"] = False
if var == "no_stations": cfg["stations"] = 0
if var == "types5": cfg["types"] = T5
t0 = time.time()
r = T.add_extended(T.generate(0), **cfg)
TY = cfg["types"]
env = SSREnv(n, "cpu", raster=r, tiles=True, train=True, camera="chest_rs", seed=5, ext_phys=24)
env.ext_frac = 1.0; env.side_soft = 2.0
env.ext_type[:] = torch.arange(n) % len(TY); env.ext_level[:] = lvl
env.ep[:] = 1; ids = torch.arange(n); env.reset(ids); env.refresh(); env._contacts(); env._update_obs(ids); env._update_cmd()
c = torch.load(ck, map_location="cpu", weights_only=False)
enc, aux, actor = N.Encoder(False), N.Aux(), N.MoEActor()
enc.load_state_dict(c["enc"]); aux.load_state_dict(c["aux"]); actor.load_state_dict(c["actor"])
h = enc.init_h(n, "cpu"); env.cnt = {}
K = len(TY); div = torch.zeros(K + 1); done = torch.zeros(K + 1); steps_t = torch.zeros(K + 1); nac = 0
for t in range(steps):
    ty = torch.where(env.ext, env.ext_type, torch.full_like(env.ext_type, K)).clone()
    e = enc.step(env.hist, env.depth, h, sample=False); h = e["h"]
    out = env.step(actor(torch.cat((env.prop, aux.est(env.hist.flatten(1)), e["z"]), -1)).clamp(-8, 8))
    nac = max(nac, int(env.nacon.max()))
    b = env.bad.float(); d = out["done"].float()
    div.index_add_(0, ty, b); done.index_add_(0, ty, d); steps_t.index_add_(0, ty, torch.ones(n))
    h = torch.where(out["done"][:, None], torch.zeros_like(h), h)
names = list(TY) + ["non_ext"]
print(json.dumps(dict(var=var, sim_s=round(time.time() - t0), max_nacon=nac, box_overflow=env.ext_box_overflow,
      per_type={names[k]: dict(div=int(div[k]), done=int(done[k]), robot_s=round(float(steps_t[k]) * 0.02, 1)) for k in range(K + 1) if steps_t[k] > 0},
      causes={k[5:]: int(v) for k, v in env.cnt.items() if k.startswith("done_") and int(v)})))
print("DONE")
