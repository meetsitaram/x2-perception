#!/bin/bash
# Run a read-only command on a node, or rsync to/from the shared filesystem, without an address in the repo.
#   bash eval/node.sh run acp 'ls /mnt/filesystem-g4/status'
#   bash eval/node.sh target acp          -> prints login@address (for rsync)
# The address and login come from agentic/config/nodes.local.json (git-ignored).
cd "$(dirname "$0")/.."
T=$(python -c "import json;n=json.load(open('agentic/config/nodes.local.json'));print(n['ssh_user']+'@'+n['nodes']['$2'])" | tr -d '\r')
SSH="ssh -i $HOME/.ssh/nebius_corpus -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15"
case "$1" in
  target) echo "$T" ;;
  run) $SSH "$T" "$3" 2> >(grep -v -E "WARNING|vulnerable|upgraded" >&2) ;;
esac
