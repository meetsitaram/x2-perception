#!/bin/bash
# Dynamic search over spawn candidates for the tiles still faulting after the static fix (ACP, 2026-11-14). CPU only.
# For rank k = 1..12 (and clearance 0.35): candidate k for every remaining tile, 8 training resets x 10 steps each; a tile whose
# candidate has 0 faults takes it into tiles_best.json. Tiles left after rank 12 are dropped later.
D=/mnt/filesystem-g4/tracks/acp/runs/s3_v2_diag; C=/mnt/filesystem-g4/tracks/acp/code; PY=$HOME/acp_env/bin/python
cd $D; rm -f SEARCH_DONE; cp tiles_fix3.json tiles_best.json
$PY - <<'EOF'
import json; d = json.load(open("dyn_fix3.json")); json.dump([n for n, r in d.items() if r["fault"]], open("remaining.json", "w"))
EOF
for k in 1 2 3 4 5 6 7 8 9 10 11 12; do
  n=$($PY -c "import json; print(len(json.load(open('remaining.json'))))"); echo "rank $k remaining $n" >> search.log; [ "$n" -eq 0 ] && break
  $PY spawn_fix.py tiles_best.json dyn_all.json tiles_k.json --clear 0.35 --rank $k --only remaining.json >> search.log
  $PY - <<EOF
import json; c = json.load(open("cfg_fix3.json")); c["goal_tiles"] = {"file": "$D/tiles_k.json"}; json.dump(c, open("cfg_k.json", "w"))
EOF
  ( cd $C && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 X2_EXCLUDE_PAIRS=hip ACP_PLANT=mc_20261024 ACP_HF_FIX=1 ACP_PAPER_DR=1 nice -n 15 taskset -c 12-19 \
     $PY $D/spawn_dyn.py $D/cfg_k.json $D/dyn_k.json --tiles $D/remaining.json --repeats 8 --envs 256 --cache $D/labels_cfg_fix1.json.pt > $D/dyn_k.out 2>&1 )
  $PY - <<'EOF'
import json
d = json.load(open("dyn_k.json")); tk = {t["name"]: t for t in json.load(open("tiles_k.json"))["tiles"]}
lk = json.load(open("tiles_k.json.fixlog.json"))
best = json.load(open("tiles_best.json")); rem = json.load(open("remaining.json"))
ok = [n for n in rem if n in d and d[n]["fault"] == 0 and lk.get(n, {}).get("new")]
for t in best["tiles"]:
    if t["name"] in ok: t["spawn"] = tk[t["name"]]["spawn"]; t["info"] = tk[t["name"]]["info"]
json.dump(best, open("tiles_best.json", "w")); json.dump([n for n in rem if n not in ok], open("remaining.json", "w"))
print("accepted", ok, flush=True)
EOF
done >> search.log 2>&1
touch SEARCH_DONE
