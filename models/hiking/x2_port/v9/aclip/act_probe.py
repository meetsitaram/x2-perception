"""CPU probe (2026-11-15): per-joint |action| of Hiking v9 model_26000 in the MuJoCo eval, mean actions and training-like sampled
actions (mean + N(0, std) with the checkpoint's std), fed back as last action like training. Writes act_probe_<mode>.npz + log."""
import sys, json, pathlib, numpy as np, torch
sys.path.insert(0, '/mnt/filesystem-g4/eval/code')
import mujoco
_mr = mujoco.mj_multiRay
def _mr2(*a):
    try: return _mr(*a)
    except TypeError: return _mr(*(a[:9] + a[10:]))
mujoco.mj_multiRay = _mr2
import harness as H, x2 as X
from apparatus import Apparatus
mode, app_name, trials, secs = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
OUT = pathlib.Path('/mnt/filesystem-g4/tracks/hiking/code/x2_port/v9/aclip/probe'); OUT.mkdir(parents=True, exist_ok=True)
ck = '/mnt/filesystem-g4/tracks/hiking/runs/x2_parkour_v4/20261115_115130_from20261115_010523/model_26000.pt'
std = torch.load(ck, map_location='cpu', weights_only=False)['model_state_dict']['std'].numpy().astype(float)
ad = H.load_adapter("hiking:HikingHandover")
rng = np.random.default_rng(7); rec = []
pol = ad.policy
def policy(obs):
    m = pol(obs); a = m + std * rng.standard_normal(m.shape) if mode == 'sampled' else m
    rec.append(np.concatenate([m, a])); return a
ad.policy = policy
app = None if app_name == 'flat' else Apparatus(app_name)
x = X.build(app, ad.plant(X.joint_names_of(X.robot_dir() / 'mjcf' / 'x2_ultra.xml')))
tr = H.Trial(x, app, ad, x.model.body_mass.copy(), x.model.body_inertia.copy())
res = []
for i in range(trials):
    r = tr.run(H.SEED0 + i, H.RANDOM, seconds=secs, record=False); res.append({k: r.get(k) for k in ('success', 'reason', 'time') if k in r})
    print(i, res[-1], len(rec), flush=True)
A = np.array(rec); np.savez(OUT / f'act_probe_{mode}_{app_name}.npz', mean=A[:, :29], act=A[:, 29:], std=std)
for nm, M in (('mean', A[:, :29]), ('act', A[:, 29:])):
    a = np.abs(M); print(nm, 'steps', len(a), 'p99.9', np.round(np.percentile(a, 99.9, 0), 1).tolist(), "max", np.round(a.max(0), 1).tolist(), "share>20 %", np.round(100*(a>20).mean(0), 2).tolist(), "mean_signed", np.round(M.mean(0), 1).tolist(), flush=True)
open(OUT / f'DONE_{mode}_{app_name}', 'w').write(json.dumps(res))
