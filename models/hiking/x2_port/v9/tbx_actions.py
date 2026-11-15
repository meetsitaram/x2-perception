import glob,sys,numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
R='/mnt/filesystem-g4/tracks/hiking/runs/x2_parkour_v4/'
vals={}
for d in ['20261115_010523_from20261115_003650','20261115_115130_from20261115_010523']:
    for f in glob.glob(R+d+'/events.out.*'):
        ea=EventAccumulator(f,size_guidance={'scalars':0}); ea.Reload()
        tags=[t for t in ea.Tags()['scalars'] if 'guard_rollout' in t]
        for t in tags:
            for e in ea.Scalars(t): vals.setdefault(t,{})[e.step]=e.value
for t,d in vals.items():
    a=np.array([d[k] for k in sorted(d)])
    print(t,len(a),'min',a.min(),'p50',np.median(a),'p99',np.percentile(a,99),'p99.5',np.percentile(a,99.5),'n>40',(a>40).sum(),'n>60',(a>60).sum(),'n>100',(a>100).sum(),'n>1e3',(a>1e3).sum())
t=[t for t in vals if 'action' in t][0]; d=vals[t]
ks=sorted(d); a=np.array([d[k] for k in ks])
print('top30', sorted(zip(a,ks))[-30:])
