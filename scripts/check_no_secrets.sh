#!/bin/bash
# Fails if tracked files contain an IPv4 address, the node login, cloud ids or key material. Run before every push.
cd "$(dirname "$0")/.."
bad=$(git grep -n -I -E '([0-9]{1,3}\.){3}[0-9]{1,3}|computeinstance-|project-e05|BEGIN [A-Z ]*PRIVATE KEY' -- . ':!scripts/check_no_secrets.sh' | grep -v -E '127\.0\.0\.1|0\.0\.0\.0|[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+[a-zA-Z]|version|cu12|==' )
user=$(python -c "import json;print(json.load(open('config/nodes.local.json'))['ssh_user'])" 2>/dev/null)
[ -n "$user" ] && bad="$bad$(git grep -n -I "$user" -- . | head -5)"
if [ -n "$bad" ]; then echo "$bad" | head -20; echo "FAIL: remove these before pushing"; exit 1; fi
echo "ok: no addresses, logins, cloud ids or keys in tracked files"
