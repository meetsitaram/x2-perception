#!/bin/bash
# Write /mnt/filesystem-g4/status/eval.json: bash eval/tools/heartbeat.sh "STATE" "RUNNING (what, started, expected finish)" "NEWEST RESULTS"
# Call it at the start and end of every eval job. Login and address come from agentic/config/nodes.local.json via eval/node.sh.
cd "$(dirname "$0")/../.."
python - "$1" "$2" "$3" <<'PY' | timeout 60 bash eval/node.sh run acp 'cat > /mnt/filesystem-g4/status/eval.json.tmp && mv /mnt/filesystem-g4/status/eval.json.tmp /mnt/filesystem-g4/status/eval.json'
import json, sys, datetime
print(json.dumps(dict(agent="eval-monitor", updated=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"), state=sys.argv[1],
                      running=sys.argv[2], results_so_far=sys.argv[3], results="/mnt/filesystem-g4/eval", code="/mnt/filesystem-g4/eval/code",
                      simulator="MuJoCo 3.15 CPU on the desktop", gpu_jobs=0)))
PY
