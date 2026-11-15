#!/bin/bash
# gates_v4b.sh: gates of the option-t continuation (record only, no action). Lines to runs/x2_parkour_v4/_supervisor/gates.txt.
# Pass at 1,500: measured air per touchdown >= 0.07 s AND speed ratio >= 0.50. At 2,000: air >= 0.10 s and speed >= 0.6.
# Alert any time: trainer log silent > 15 min.
T=/mnt/filesystem-g4/tracks/hiking; X=$T/code/x2_port; RUN=$T/runs/x2_parkour_v4; G=$RUN/_supervisor/gates.txt
last() { tail -1 $RUN/metrics.jsonl | python3 -c "import json,sys;print(json.loads(sys.stdin.read())['iteration'])"; }
air() { python3 -c "
import json
v=[json.loads(l) for l in open('$RUN/metrics.jsonl')]
v=[d.get('Curriculum/walk_monitor/air_per_touchdown_s') for d in v if $1-10<d['iteration']<=$1]
v=[x for x in v if x is not None]; print('%.4f'%(sum(v)/len(v)) if v else 'nan')"; }
for it in 1250 1500 1750 2000; do
  while [ "$(last)" -lt $it ]; do
    a=$(( $(date +%s) - $(stat -c %Y $RUN/_supervisor/train.log 2>/dev/null || date +%s) ))
    [ $a -gt 900 ] && echo "$(date -u '+%F %T') ALERT train.log silent ${a}s" >> $G && sleep 900
    sleep 60
  done
  L=$(cd $X && ~/hiking-env/bin/python report_line.py $RUN $it 10 2>&1); A=$(air $it)
  echo "$(date -u '+%F %T') GATE $it (t on): measured air per touchdown $A s | $L" >> $G
  python3 - "$L" "$A" $it >> $G <<'PY'
import re,sys
s,a,it=sys.argv[1],float(sys.argv[2]),int(sys.argv[3]); sp=float(re.search(r"speed ratio ([0-9.]+)",s).group(1))
need={1500:(0.07,0.5),2000:(0.10,0.6)}.get(it)
print("  %d: air %.3f s, speed %.3f -> %s"%(it,a,sp,("PASS" if a>=need[0] and sp>=need[1] else "FAIL (resume the hiking agent)") if need else "line only"))
PY
done
