import sys,time; sys.path.insert(0,'.')
import torch
from ssr.env import SSREnv, PRIV_DIM
N=int(sys.argv[1]) if len(sys.argv)>1 else 512
t=time.time(); e=SSREnv(N); print('init',round(time.time()-t,1), 'priv',e.priv.shape, PRIV_DIM, 'hist',e.hist.shape,'depth',e.depth.shape, flush=True)
a=torch.zeros(N,21,device=e.dev); mu=torch.zeros(N,2,2,device=e.dev); sg=torch.full((N,2),0.05,device=e.dev)
for i in range(3): e.step(a,mu,sg)
t=time.time(); n=60; fall=0
for i in range(n):
    o=e.step(a,mu,sg); fall+=int(o['term'].sum())
dt=time.time()-t; print(f'{N} envs {n/dt:.1f} steps/s = {N*n/dt:.0f} env-steps/s; term {fall}; nacon {e.nacon} depth mean {e.depth.mean():.3f}; r_l {o["r_l"].mean():.2f} r_f {o["r_f"].mean():.2f}; contact {o["contact"].float().mean():.2f}; F {e.F[...,2].abs().mean():.0f}', flush=True)
t=time.time(); e._render(); print('render ms',1000*(time.time()-t))
print('mem',torch.cuda.max_memory_allocated()/1e9,'GB torch')
