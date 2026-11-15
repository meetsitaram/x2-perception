#!/bin/bash
# precheck_v8.sh <continue_v8.sh>: v7/precheck_v7.py on the v8 launcher. Accepted, and only these: (1) the launcher differs from
# v5b in exactly HIKING_X2_AIR_TD_PROGRESS and HIKING_X2_FAIR (the v8 change); (2) "staged.py.v6 is not the live staged.py + the
# v6 patch": since v7 the live staged.py IS staged.py.v6, checked here with cmp instead. Prints PRECHECK_V8 PASS; exit 0 = pass.
X=/mnt/filesystem-g4/tracks/hiking/code/x2_port; D=$X/files_instinctlab/tasks/parkour/config/x2
$HOME/hiking-env/bin/python $X/v7/precheck_v7.py "$1" 2>&1 | sed "s/^FAIL/(v7 check, filtered below) FAIL/"
OUT=$($HOME/hiking-env/bin/python $X/v7/precheck_v7.py "$1" 2>&1 | grep -a "^FAIL" \
  | grep -av "^FAIL launcher differs from v5b in \['HIKING_X2_AIR_TD_PROGRESS', 'HIKING_X2_FAIR'\]$" \
  | grep -av "^FAIL staged.py.v6 is not the live staged.py + the v6 patch")
cmp -s $D/staged.py $X/variants/staged.py.v6 || OUT="$OUT FAIL live staged.py is not variants/staged.py.v6"
grep -q "^ENVV=\"HIKING_X2_FAIR=1 HIKING_X2_AIR_TD_PROGRESS=1 " "$1" || OUT="$OUT FAIL ENVV lacks the v8 vars"
[ -z "$OUT" ] && { echo "PRECHECK_V8 PASS"; exit 0; }
echo "$OUT" | sed 's/ FAIL/\nFAIL/g'; exit 1
