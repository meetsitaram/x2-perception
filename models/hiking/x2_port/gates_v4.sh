#!/bin/bash
# gates_v4.sh: run 4 gates, hands-off. Writes report lines to runs/x2_parkour_v4/_supervisor/gates.txt.
# 750: drift line (stalled resets > 40% with speed ratio < 0.2 -> "DRIFT"). 1,000: if air per touchdown (est.) < 0.05 s,
# start continue_v4.sh (option t, weight 140, threshold 0.05 s); otherwise leave run 4 running. Never starts a second GPU job:
# continue_v4.sh stops run 4 through its supervisor before it starts the continuation.
T=/mnt/filesystem-g4/tracks/hiking; X=$T/code/x2_port; RUN=$T/runs/x2_parkour_v4; G=$RUN/_supervisor/gates.txt
last() { tail -1 $RUN/metrics.jsonl | python3 -c "import json,sys;print(json.loads(sys.stdin.read())['iteration'])"; }
line() { cd $X && ~/hiking-env/bin/python report_line.py $RUN $1 10 2>&1; }
while [ "$(last)" -lt 750 ]; do sleep 60; done
L=$(line 750); echo "$(date -u '+%F %T') GATE 750: $L" >> $G
python3 - "$L" >> $G <<'PY'
import re,sys
s=sys.argv[1]; sp=float(re.search(r"speed ratio ([0-9.]+)",s).group(1)); st=float(re.search(r"stalled resets ([0-9.]+)%",s).group(1))
print("  750 check: speed %.3f stalled %.0f%% -> %s"%(sp,st,"DRIFT (report at once)" if st>40 and sp<0.2 else "no drift"))
PY
while [ "$(last)" -lt 1000 ]; do sleep 60; done
L=$(line 1000); echo "$(date -u '+%F %T') GATE 1000: $L" >> $G
A=$(printf '%s' "$L" | sed -n 's/.*air per touchdown (est.) \([0-9.]*\) s.*/\1/p' | head -1)
if [ -n "$A" ] && python3 -c "import sys;sys.exit(0 if float('$A')<0.05 else 1)"; then
  echo "$(date -u '+%F %T')   air per touchdown $A s < 0.05: starting continue_v4.sh (option t)" >> $G
  tmux new -d -s cont4 "bash $X/continue_v4.sh"
else
  echo "$(date -u '+%F %T')   air per touchdown '$A' s: not below 0.05 (or unreadable), run 4 continues unchanged" >> $G
fi
