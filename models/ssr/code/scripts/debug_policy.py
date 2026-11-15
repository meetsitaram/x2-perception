import sys; sys.path.insert(0,'.')
import torch, numpy as np
from ssr import networks as N
from ssr.env import *
dev='cuda:0'; ck=torch.load(sys.argv[1],map_location=dev,weights_only=False)
enc,aux,actor,foot=N.Encoder().to(dev),N.Aux().to(dev),N.MoEActor().to(dev),N.Foothold().to(dev)
enc.load_state_dict(ck['enc']);aux.load_state_dict(ck['aux']);actor.load_state_dict(ck['actor']);foot.load_state_dict(ck['foot'])
e=SSREnv(512); e.level[:]=1
h=enc.init_h(512,dev); orig=e._done; C={k:0 for k in ('tilt','low','hit','scrape','bad','out_y','back','success','over')}
def w(s):
    b=s['p']; g=e.hmap(b[:,0],b[:,1]); pts=e.xpos[:,e.clear_ids]; sc=(pts[...,2]-e.hmap(pts[...,0],pts[...,1])<e.clear_m)
    t,to,su=orig(s); xr=b[:,0]-e.origin[:,0]; yr=b[:,1]-e.origin[:,1]
    r=dict(tilt=s['g'][:,2]>-.5,low=b[:,2]-g<.3,hit=e.term_force>80,scrape=sc.any(-1),bad=e.bad,out_y=((yr<0)|(yr>4))&~e.free,back=(xr<0)&~e.free,success=su,over=e.ep>=1000)
    for k,v in r.items(): C[k]+=int(v.sum())
    C['sc_each']=C.get('sc_each',0)+sc.sum(0).cpu().numpy()
    return t,to,su
e._done=w; T=0
with torch.no_grad():
  for t in range(250):
    z=enc.step(e.hist,e.depth,h,sample=False); h=z['h']; o=torch.cat((e.prop,aux.est(e.hist.flatten(1)),z['z']),-1); a=actor(o)
    fm,fs=foot(e.priv,a); r=e.step(a,fm,fs); h[r['done']]=0; T+=int(r['done'].sum())
print('iter',ck['iter'],'episodes',T,C); print('speed cmd mean',e.speed.abs().mean().item(),'base z-g',(e.qpos[:,2]-e.hmap(e.qpos[:,0],e.qpos[:,1])).mean().item(),'max a',a.abs().max().item())
