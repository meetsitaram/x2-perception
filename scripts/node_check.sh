#!/bin/bash
# Reachability, GPU, disk and shared-filesystem check of the three nodes. Read-only.
# Addresses and login come from config/nodes.local.json (git-ignored, never committed).
cd "$(dirname "$0")/.."
python -c "
import json
n = json.load(open('config/nodes.local.json'))
for k, ip in n['nodes'].items():
    print(k, ip, n['ssh_user'], n['ssh_key'])
" | tr -d '\r' > /tmp/pa_nodes.txt
while read -r name ip user key; do
  key="${key/#\~/$HOME}"
  echo "== $name"
  timeout 25 ssh -n -i "$key" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=10 "$user@$ip" \
    'nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader; df -h | grep -E "filesystem|/$"; tmux ls 2>/dev/null; ls /mnt/filesystem-g4 | head' 2>&1 \
    | grep -v -E "WARNING|vulnerable|upgraded"
done < /tmp/pa_nodes.txt
