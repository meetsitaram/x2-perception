#!/bin/bash
# r6 -> r6b switch (coordinator, 2026-11-12 02:46): r6 (terrain curriculum from iteration 0, levels drawn 0-3) stood at 5% of commanded
# speed at 420 while t_a (same schedule on flat) moved. Rule: give r6 until iteration 600; if still under 15% of commanded speed,
# restart flat-first. r6b = t_a's first 550 flat iterations (its checkpoint, iteration 549) continued with the r6 settings plus
# --flat_until 550: terrain types redrawn, all levels 0, curriculum on from 550. If r6 is at 15% or more at 600, nothing is changed.
R=/mnt/filesystem-g4/tracks/ssr/runs
cd /mnt/filesystem-g4/tracks/ssr/code
while true; do
  it=$(tail -1 $R/r6/log.jsonl | python3 -c "import json,sys; print(json.loads(sys.stdin.read())['it'])")
  [ "$it" -ge 600 ] && break
  sleep 30
done
v=$(tail -3 $R/r6/log.jsonl | python3 -c "import json,sys; L=[json.loads(l) for l in sys.stdin if l.strip()]; print(round(sum(d['gait']['vx_ratio'] for d in L)/len(L),3))")
echo "QUEUE7 $(date -u +%F' '%T) r6 at iteration $it: mean share of commanded speed over the last 3 logs $v"
if python3 -c "import sys; sys.exit(0 if $v < 0.15 else 1)"; then
  touch $R/STOP
  for i in $(seq 1 60); do pgrep -f "scripts/supervise_node.py --name r6 " > /dev/null || break; sleep 10; done
  pgrep -f "ssr.train" > /dev/null && { echo "QUEUE7 training process still alive; not starting r6b"; exit 1; }
  mkdir -p $R/r6b && cp $R/t_a/last.pt $R/r6b/last.pt
  echo "QUEUE7 $(date -u +%F' '%T) start r6b from t_a iteration 549"
  ../env/venv/bin/python scripts/supervise_node.py --name r6b --iters 20000 --num_envs 8192 -- --flat_until 550 --stage_a 150 --stage_b 400 --w_style 0.2 --style_low 0 --gate_air 0.15 --gate_ramp 500 --prior_min_speed 0.2 --rsi_min_speed 0.3 --plant mc_20261024 --camera chest_rs --amp_file data/amp_v2.npz --prior_balance
  echo "QUEUE7 $(date -u +%F' '%T) end r6b"
else
  echo "QUEUE7 r6 at or above 15%: r6 continues"
fi
