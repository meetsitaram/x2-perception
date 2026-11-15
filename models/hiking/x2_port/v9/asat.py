import json, xml.etree.ElementTree as ET
d=json.load(open('/mnt/filesystem-g4/tracks/hiking/eval/x2_parkour_v4_26000/adapter_card.json'))
u=ET.parse('/mnt/filesystem-g4/tracks/hiking/code/x2_port/robot/x2_ultra_pelvis_root.urdf').getroot()
lim={j.get('name'):(float(j.find('limit').get('lower')),float(j.find('limit').get('upper'))) for j in u.iter('joint') if j.find('limit') is not None}
def g(k,n):
    v=d[k]; return v[n] if isinstance(v,dict) else v[d['joint_order'].index(n)] if len(v)==len(d['joint_order']) else v[d['action_joint_order'].index(n)]
print('keys types', {k:type(d[k]).__name__ for k in ('default_pose','action_scale','kp','kd','effort_limit','velocity_limit')})
for i,n in enumerate(d['action_joint_order']):
    lo,hi=lim[n]; q0=g('default_pose',n); s=g('action_scale',n); kp=g('kp',n); kd=g('kd',n); tau=g('effort_limit',n); vl=g('velocity_limit',n)
    # target offset beyond which torque is saturated for any q in range and |qd|<=vl
    m=(tau+kd*vl)/kp
    asat=max(hi-q0, q0-lo)+m
    print(f"{i:2d} {n:28s} lim[{lo:+.2f},{hi:+.2f}] q0 {q0:+.2f} s {s:.3f} kp {kp:6.1f} kd {kd:5.2f} tau {tau:5.1f} a_sat {asat/s:6.1f} a_range {max(hi-q0,q0-lo)/s:5.1f}")
