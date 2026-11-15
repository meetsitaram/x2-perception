#!/bin/bash
# precheck_v8b.sh <continue_v8b.sh> (2026-11-14) = precheck_v8.sh for v8b: v7/precheck_v7.py on the launcher. Accepted, and only
# these: (1) the launcher differs from v5b in exactly the v8 vars (AIR_TD_PROGRESS, FAIR) and the three v8b shares (SLOW_FWD,
# STAND, TURN); (2) "staged.py.v6 is not the live staged.py + the v6 patch" (since v7 the live staged.py IS staged.py.v6, cmp).
# Also: the shares are exactly stand 0.05, turn 0.10, slow fwd 0.12, slow yaw 0.4. Prints PRECHECK_V8B PASS; exit 0 = pass.
X=/mnt/filesystem-g4/tracks/hiking/code/x2_port; D=$X/files_instinctlab/tasks/parkour/config/x2
OK1="FAIL launcher differs from v5b in ['HIKING_X2_AIR_TD_PROGRESS', 'HIKING_X2_FAIR', 'HIKING_X2_SLOW_FWD_SHARE', 'HIKING_X2_STAND_SHARE', 'HIKING_X2_TURN_SHARE']"
$HOME/hiking-env/bin/python $X/v7/precheck_v7.py "$1" 2>&1 | sed "s/^FAIL/(v7 check, filtered below) FAIL/"
OUT=$($HOME/hiking-env/bin/python $X/v7/precheck_v7.py "$1" 2>&1 | grep -a "^FAIL" | grep -avxF "$OK1" \
  | grep -av "^FAIL staged.py.v6 is not the live staged.py + the v6 patch")
cmp -s $D/staged.py $X/variants/staged.py.v6 || OUT="$OUT FAIL live staged.py is not variants/staged.py.v6"
grep -q "^ENVV=\"HIKING_X2_FAIR=1 HIKING_X2_AIR_TD_PROGRESS=1 " "$1" || OUT="$OUT FAIL ENVV lacks the v8 vars"
for kv in HIKING_X2_STAND_SHARE=0.05 HIKING_X2_TURN_SHARE=0.10 HIKING_X2_SLOW_FWD_SHARE=0.12 HIKING_X2_SLOW_TURN_SHARE=0.4; do
  grep "^ENVV=" "$1" | grep -q " $kv " || OUT="$OUT FAIL ENVV lacks $kv"; done
[ -z "$OUT" ] && { echo "PRECHECK_V8B PASS"; exit 0; }
echo "$OUT" | sed 's/ FAIL/\nFAIL/g'; exit 1
