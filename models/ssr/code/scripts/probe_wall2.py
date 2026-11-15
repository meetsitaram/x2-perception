"""Wall refusal vs wall distance and limb clearance (CPU, deterministic), probe_wall geometry. Per checkpoint and wall distance d
(centre line to wall face): stalled share, mean speed last 4 s, share of robot-steps in the 0.06 m graze band and overlapping,
and the limbs' lateral reach (max over side_ids of y - base y, left side) standing and walking. Usage: ckpt d1,d2,... [n]"""
import json, math, sys, torch, numpy as np
sys.path.insert(0, "/mnt/filesystem-g4/tracks/ssr/code")
from ssr import networks as N
from ssr.env import SSREnv
torch.set_grad_enabled(False)
RES = 0.025
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
enc, aux, actor = N.Encoder(False), N.Aux(), N.MoEActor()
enc.load_state_dict(ck["enc"]); aux.load_state_dict(ck["aux"]); actor.load_state_dict(ck["actor"])
n = int(sys.argv[3]) if len(sys.argv) > 3 else 32; out = {"iteration": int(ck["iter"]), "robots": n}
g = torch.Generator().manual_seed(11)
dy = (torch.rand(n, generator=g) * 2 - 1) * 0.03; yaw = (torch.rand(n, generator=g) * 2 - 1) * math.radians(10)
for d in [float(x) for x in sys.argv[2].split(",")]:
    nx, ny = int(8 / RES) + 1, int(4 / RES) + 1
    z = np.zeros((nx, ny), np.float32); v = z.copy()
    if d > 0:
        v[int(0.5 / RES):int(5.0 / RES), int((2.0 + d) / RES):int((2.0 + d + 0.08) / RES)] = 1.3
    R = dict(z=z, x0=0.0, y0=0.0, res=RES, base=0.1, vision=dict(z=v, x0=0.0, y0=0.0, res=RES))
    sp = torch.stack((torch.full((n,), 1.0), 2.0 + dy, yaw), -1)
    env = SSREnv(n, "cpu", raster=R, train=False, tiles=False, spawn=sp, camera="chest_rs")
    env.obst = (env.hmap.vz - env.hmap.z) > 0.25
    env.cmd[:] = 0.0; h = enc.init_h(n, "cpu"); acc = torch.zeros(n); band = torch.zeros(n); over = torch.zeros(n)
    reach_st = torch.zeros(n); reach_w = torch.zeros(n)
    for t in range(450):
        if t == 50: env.cmd[:] = torch.tensor([0.37, 0.0, 0.0])
        e = enc.step(env.hist, env.depth, h, sample=False); h = e["h"]
        env.step(actor(torch.cat((env.prop, aux.est(env.hist.flatten(1)), e["z"]), -1)).clamp(-8, 8))
        ry = (env.xpos[:, env.side_ids, 1] - env.s["p"][:, None, 1]).max(1).values
        if t < 50: reach_st = torch.maximum(reach_st, ry)
        else: reach_w = torch.maximum(reach_w, ry)
        if t >= 50:
            band += env._touches_obstacle(0.06, log=False).float() / 400; over += env._touches_obstacle(0.0, log=False).float() / 400
        if t >= 250: acc += env.s["v_b"][:, 0] / 200
    st = acc < 0.1
    out[str(d)] = dict(stalled=round(float(st.float().mean()), 3), v_last4=round(float(acc.mean()), 3),
                       band_share=round(float(band.mean()), 3), overlap_share=round(float(over.mean()), 3),
                       band_share_moving=round(float(band[~st].mean()), 3) if (~st).any() else None,
                       reach_stand=round(float(reach_st.mean()), 3), reach_walk=round(float(reach_w.mean()), 3))
    print(json.dumps(out), flush=True)
