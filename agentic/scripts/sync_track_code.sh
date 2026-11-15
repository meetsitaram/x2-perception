#!/bin/bash
# Pull each track's code from the shared filesystem into models/<track>/ of this repo, so it has history and
# an off-node copy. Run from the desktop (the nodes hold no GitHub credentials). Read-only on the nodes.
#   bash agentic/scripts/sync_track_code.sh            pull, scrub, check; then commit and push by hand or with --commit
#   bash agentic/scripts/sync_track_code.sh --commit   also commit and push if the secrets check passes
# Left out on purpose: checkpoints, data, environments, robot meshes (they live in data/robots on the shared
# filesystem), anything over 2 MB. The node login in paths is replaced by NODE_USER in the repo copy.
cd "$(dirname "$0")/../.." || exit 1  # repo root
read -r IP USER KEY < <(python -c "
import json
n = json.load(open('agentic/config/nodes.local.json'))
print(n['nodes']['hiking'], n['ssh_user'], n['ssh_key'])" | tr -d '\r')
KEY="${KEY/#\~/$HOME}"
SSH="ssh -i $KEY -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=10"
FS=/mnt/filesystem-g4/tracks
pull() {  # pull <remote dir under tracks/> <local dir>
  mkdir -p "$2"
  timeout 240 "$HOME/bin/rsync" -az --bwlimit=3000 --max-size=2m --no-links \
    --exclude .git --exclude '__pycache__' --exclude '*.pt' --exclude '*.npz' --exclude '*.pkl' --exclude '*.onnx' \
    --exclude '*.STL' --exclude '*.stl' --exclude '*.obj' --exclude '*.mp4' --exclude '*.png' --exclude '*.jpg' \
    --exclude 'env/' --exclude 'venv/' --exclude 'runs/' --exclude 'data/' --exclude '_trash/' --exclude 'logs/' \
    -e "$SSH" "$USER@$IP:$FS/$1" "$2" 2>&1 | grep -v -E "WARNING|vulnerable|upgraded"
  echo "pulled $1 -> $2 (rsync ${PIPESTATUS[0]})"
}
pull acp/code/ models/acp/code/
pull ssr/code/ models/ssr/code/
pull hiking/code/x2_port/ models/hiking/x2_port/
# scrub the node login from text files in the repo copy
grep -rIl "$USER" models 2>/dev/null | while read -r f; do sed -i "s/$USER/NODE_USER/g" "$f"; echo "scrubbed login in $f"; done
date -u +"%Y-%m-%dT%H:%MZ" > models/LAST_SYNC.txt
git add models
if ! bash agentic/scripts/check_no_secrets.sh; then echo "secrets check failed: not committing"; exit 1; fi
if [ "$1" = "--commit" ]; then
  if git diff --cached --quiet; then echo "no code changes"; exit 0; fi
  git -c user.name=meetsitaram -c user.email=6304083+meetsitaram@users.noreply.github.com commit -q -m "Track code sync $(cat models/LAST_SYNC.txt)

Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>" && timeout 90 git push -q origin main && echo "committed and pushed"
fi
