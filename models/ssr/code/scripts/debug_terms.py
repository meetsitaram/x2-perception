import sys; sys.path.insert(0,'.')
import torch, numpy as np
from ssr.env import *
e=SSREnv(256)
print('spawn z-g', (e.qpos[:,2]-e.hmap(e.qpos[:,0],e.qpos[:,1])).mean().item(), 'foot z', e.xpos[0,e.feet].cpu().numpy().round(3).tolist(), 'sole', (e.sole[...,2]-e.hmap(e.sole[...,0],e.sole[...,1])).mean().item())
a=torch.zeros(256,21,device=e.dev); mu=torch.zeros(256,2,2,device=e.dev); sg=torch.full((256,2),0.05,device=e.dev)
orig=e._done
def wrapped(s):
    base=s['p']; g=e.hmap(base[:,0],base[:,1]); pts=e.xpos[:,e.clear_ids]
    r=dict(tilt=(s['g'][:,2]>-0.5), low=(base[:,2]-g<0.3), hit=(e.term_force>80), bad=e.bad)
    sc=(pts[...,2]-e.hmap(pts[...,0],pts[...,1])<e.clear_m); r['scrape']=sc.any(-1)
    print({k:int(v.sum()) for k,v in r.items()}, 'scrape by', sc.sum(0).tolist(), 'zmin', round(float(pts[...,2].min()),2),'maxF',round(float(e.term_force.max()),0), 'Fz', round(float(e.F[...,2].abs().mean())), 'maxqd', round(float(e.qvel.abs().max()),1))
    return orig(s)
e._done=wrapped
for i in range(45):
    if i%4==0 or i>38: e.step(a,mu,sg)
    else:
        e._done=orig; e.step(a,mu,sg); e._done=wrapped
m=e.x2.model; import mujoco
print('mass total',m.body_mass.sum(), 'e.mass sum', e.mass[0].sum().item(), e.mass[1].sum().item())
print('elbow contacts geom pairs', )
n=int(e.nacon[0]); g=e.c_geom[:n].cpu().numpy(); w=e.c_world[:n].cpu().numpy()
from collections import Counter
c=Counter()
for (a1,b1) in g:
    if e.geom_term[a1] or e.geom_term[b1]:
        c[(mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,m.geom_bodyid[a1]),mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,m.geom_bodyid[b1]))]+=1
print(c.most_common(6))
