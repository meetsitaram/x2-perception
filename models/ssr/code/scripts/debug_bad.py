import sys; sys.path.insert(0,'.')
import torch, numpy as np, mujoco
from ssr import env as E
mode=sys.argv[1]
class V(E.SSREnv):
    def _domain_randomize(self, ids):
        if mode=='nodr': return
        n,u=len(ids),self._u
        if mode=='fric':
            self.friction[ids[:,None],self.robot_geoms,0]=u(n,lo=0.2,hi=1.25)[:,None]; return
        if mode=='mass':
            mass=self.mass0.expand(n,-1).clone()*u(n,self.mass0.shape[0],lo=.9,hi=1.1); mass[:,self.pelvis]=self.mass0[self.pelvis]+u(n,lo=-2.,hi=12.5); self.mass[ids]=mass; self.inertia[ids]=self.inertia0[None]*(mass/self.mass0.clamp(min=1e-6))[...,None]; return
        if mode=='ipos':
            ipos=self.ipos0[None]+u(n,*self.ipos0.shape)*.01; ipos[:,self.pelvis]=self.ipos0[self.pelvis]+u(n,3)*.1; self.ipos[ids]=ipos; return
        if mode=='gain':
            m,g=u(n,lo=.8,hi=1.2),u(n,31,lo=.9,hi=1.1); d=u(n,31,lo=.9,hi=1.1); kp,kd=self.kp0*g*m[:,None],self.kd0*d*m[:,None]; self.gain[ids,:,0]=kp; self.bias[ids,:,1],self.bias[ids,:,2]=-kp,-kd; return
        super()._domain_randomize(ids)
e=V(256)
a=torch.zeros(256,21,device=e.dev); mu=torch.zeros(256,2,2,device=e.dev); sg=torch.full((256,2),.05,device=e.dev)
tot=0; hit=0
for i in range(30):
    o=e.step(a,mu,sg); tot+=int(e.bad.sum())
print(mode,'bad worlds in 30 steps',tot, 'nan log', e.log.get('nan',0), 'Fz', round(float(e.F[...,2].abs().mean())))
