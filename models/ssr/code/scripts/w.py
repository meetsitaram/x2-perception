import json,sys
run=sys.argv[1]; n=int(sys.argv[2]) if len(sys.argv)>2 else 6
L=[json.loads(l) for l in open('/mnt/filesystem-g4/tracks/ssr/runs/%s/log.jsonl'%run) if l.strip()]
for r in L[-n:]:
    g=r.get('gait',{}); t=r.get('terms',{})
    print(r['it'],'vxr %.3f vx %.3f sgl %.2f dbl %.2f td/s %.1f airtd %s long %s prog %s gate %s ramp %s r_l %.2f len %.1f tilt %.2f std %.3f'%(g.get('vx_ratio',0),g.get('vx_mps',0),g.get('single',0),g.get('double',0),g.get('steps_per_s',0),r.get('air_td'),r.get('air_long_share'),r.get('prog_mean'),r.get('prog_gate'),r.get('stage_ramp'),r['r_l'],r['ep_len_s'],r['done_share'].get('tilt',0),r['std']))
