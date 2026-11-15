import sys; sys.path.insert(0,'.')
import torch, numpy as np, mujoco
from ssr import env as E, terrain as T
r=T.generate(); r['z']=r['z']*0
T.generate=lambda s=0: r
class V(E.SSREnv):
    def _domain_randomize(self, ids): pass
e=V(256)
a=torch.zeros(256,21,device=e.dev); mu=torch.zeros(256,2,2,device=e.dev); sg=torch.full((256,2),.05,device=e.dev)
m=e.x2.model
for i in range(8):
    # one substep at a time view
    prev=e.qvel.clone()
    o=e.step(a,mu,sg)
    if int(e.bad.sum()):
        w=e.bad.nonzero()[0,0].item()
        print('step',i,'bad world',w,'prev qvel max',prev[w].abs().max().item(), 'arg',prev[w].abs().argmax().item())
        n=int(e.nacon[0]); mask=e.c_world[:n]==w
        print(' contacts in world', int(mask.sum()), 'dist min', e.dd.contact.dist.numpy()[:n][mask.cpu().numpy()].min() if int(mask.sum()) else None)
        break
