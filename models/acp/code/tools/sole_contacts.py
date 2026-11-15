"""Sole contact physics in the training model (ACP agent, 2026-11-12): run the tracker (reference contacts, deterministic)
on ramp-raster clips and flat clips on the CPU (mujoco_warp CPU, same model as training: corrected body, data 2g), and per
physics step record every contact of a sole sphere / terrain foot box: partner (floor plane, height field, prop box), penetration,
normal direction and normal force. Reports per set: contacts per sphere in contact, depth p50/p99/max, share with a near-horizontal
normal (|n_z| < 0.7: sideways in a cell / against a step edge) and their force, normal force p50/p99/max."""
import sys, numpy as np, torch, mujoco
sys.path.insert(0, "/mnt/filesystem-g4/tracks/acp/code"); sys.path.insert(0, "/mnt/filesystem-g4/tracks/acp/code/tools")
import acp.env as E, mujoco_warp as mjw
from acp.data import Dataset
from acp.train import ActorCritic
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
data = Dataset("cfg/stage2g_sets.json", torch.device("cpu"), cache="cfg/labels_stage2g_sets.json.pt")
m = data.x2.model
names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(m.ngeom)]
ank = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{s}_ankle_roll_link") for s in ("left", "right")}
box_g = np.array(["foot_terrain" in n for n in names], dtype=bool)
sole = np.array([(m.geom_bodyid[g] in ank and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE) or box_g[g] for g in range(m.ngeom)], dtype=bool)
kind = np.array(["hfield" if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_HFIELD else "plane" if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE else "box" for g in range(m.ngeom)])
print("sole spheres", int(sole.sum() - box_g.sum()), "foot boxes", int(box_g.sum()), "solref of a sole geom", m.geom_solref[np.where(sole)[0][0]], "timestep", m.opt.timestep)
REC = []
def phys(self, ctrl_new):
    k = self.delay
    for j in range(E.DECIMATION):
        self.ctrl[:] = torch.where(j >= k, ctrl_new, self.ctrl_prev)
        mjw.step(self.mm, self.dd)
        n = int(self.dd.nacon.numpy()[0])
        if n == 0: continue
        g = self.dd.contact.geom.numpy()[:n]; dist = self.dd.contact.dist.numpy()[:n]; fr = self.dd.contact.frame.numpy()[:n]
        w = self.dd.contact.worldid.numpy()[:n]; ea = self.dd.contact.efc_address.numpy()[:n]; dim = self.dd.contact.dim.numpy()[:n]
        f = self.dd.efc.force.numpy()
        f = f[0] if f.ndim == 2 else f
        for i in range(n):
            a, b = g[i]
            s = a if sole[a] else b if sole[b] else -1
            if s < 0: continue
            other = b if s == a else a
            adr = ea[i][0] if np.ndim(ea[i]) else ea[i]
            fn = float(np.sum(f[adr:adr + max(2 * (dim[i] - 1), 1)])) if adr >= 0 else 0.0
            REC.append((int(w[i]), int(s), kind[other], float(dist[i]), float(abs(fr[i][0][2])), fn, bool(box_g[s])))
    self.ctrl_prev.copy_(ctrl_new)
E.AcpEnv._physics = phys
out = {}
for setname in sys.argv[2].split(","):
    si = data.set_names.index(setname); clips = (data.clip_set == si).nonzero().flatten()[:int(sys.argv[3])]
    N = len(clips)
    env = E.AcpEnv(data, N, eval_mode=True, njmax=450)
    ac = ActorCritic(ck["actor_dim"], ck["critic_dim"], ck["act_dim"]); ac.load_state_dict(ck["model"]); ac.eval()
    saved = data.clip_w.clone()
    REC.clear()
    for e in range(N):
        data.clip_w = torch.zeros_like(saved); data.clip_w[clips[e]] = 1.0
        env.reset(torch.tensor([e]))
    data.clip_w = saved
    with torch.no_grad():
        for st in range(int(sys.argv[4])):
            ao, _ = env.observe(); r, term, to, info = env.step(ac.actor(ac.an(ao)))
    R = np.array([(x[3], x[4], x[5]) for x in REC]); kd = np.array([x[2] for x in REC]); isbox = np.array([x[6] for x in REC], dtype=bool)
    sph = ~isbox
    def q(v): return [round(float(np.percentile(v, p)), 4) for p in (50, 99)] + [round(float(v.max()), 4)] if len(v) else None
    for part in ("plane", "hfield", "box"):
        k = sph & (kd == part)
        if not k.any(): continue
        dep = -R[k, 0]; side = R[k, 1] < 0.7
        print(f"{setname} spheres on {part}: {int(k.sum())} contacts; depth m p50/p99/max {q(dep)}; normal force N p50/p99/max {q(R[k, 2])}; "
              f"near-horizontal normals {100 * side.mean():.1f} % (their force p50/p99/max {q(R[k, 2][side])})")
    k = isbox
    if k.any(): print(f"{setname} terrain foot boxes: {int(k.sum())} contacts; depth p50/p99/max {q(-R[k, 0])}; force p50/p99/max {q(R[k, 2])}")
