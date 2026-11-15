import sys; sys.path.insert(0,'.')
import torch, numpy as np, mujoco
from ssr import env as E, terrain as T
mode=sys.argv[1]
r=T.generate()
if mode=='flat': r['z']=r['z']*0
class V(E.SSREnv):
    def _domain_randomize(self, ids): pass
e=V(256, raster=None) if mode=='tiles' else None
if mode!='tiles':
    T_gen=T.generate; T.generate=lambda s=0: r
    e=V(256)
a=torch.zeros(256,21,device=e.dev); mu=torch.zeros(256,2,2,device=e.dev); sg=torch.full((256,2),.05,device=e.dev)
hist=[]
for i in range(30):
    o=e.step(a,mu,sg)
    if int(e.bad.sum()):
        ids=e.bad.nonzero().squeeze(-1)[:4].tolist()
    hist.append(int(e.bad.sum()))
print(mode,'bad per step',hist, 'by type', torch.bincount(e.type,minlength=5).tolist())
print('condim of foot', [int(e.x2.model.geom_condim[g]) for g in range(e.x2.model.ngeom) if e.geom_side[g]>=0][:4], 'nv',e.x2.model.nv)
