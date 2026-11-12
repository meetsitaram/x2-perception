#!/bin/bash
# r6b restart at iteration 1,000 (coordinator, 2026-11-12 03:20). Continues from r6b's iteration-1,000 checkpoint (same run name).
# If mean touchdown air time (last 3 logs) < 0.10 s: style by schedule from 1,000 (ramped to the paper's 0.2 over 500) and the
# feet-air-time reward (W 20 per touchdown x clamp(air - 0.10, 0, 0.30), moving commands only). Otherwise: style latch lowered to 0.10 s.
# Both branches: the style discriminators no longer see the arms (--disc_mask_arms; port audit 03:25).
R=/mnt/filesystem-g4/tracks/ssr/runs
cd /mnt/filesystem-g4/tracks/ssr/code
BASE="--flat_until 550 --obst0 0.25 --obst_ramp 500 --stage_a 150 --stage_b 400 --w_style 0.2 --style_low 0 --gate_ramp 500 --prior_min_speed 0.2 --rsi_min_speed 0.3 --plant mc_20261024 --camera chest_rs --amp_file data/amp_v2.npz --prior_balance --disc_mask_arms"
while true; do
  it=$(tail -1 $R/r6b/log.jsonl | python3 -c "import json,sys; print(json.loads(sys.stdin.read())['it'])")
  [ "$it" -ge 1005 ] && break
  sleep 20
done
read air vx <<< $(tail -4 $R/r6b/log.jsonl | head -3 | python3 -c "import json,sys; L=[json.loads(l) for l in sys.stdin if l.strip()]; print(round(sum(d['air_td'] for d in L)/len(L),3), round(sum(d['gait']['vx_ratio'] for d in L)/len(L),3))")
echo "QUEUE9 $(date -u +%F' '%T) r6b at $it: touchdown air $air s, share of commanded speed $vx"
touch $R/STOP
for i in $(seq 1 60); do pgrep -f "ssr.train" > /dev/null || break; sleep 10; done
pgrep -f "ssr.train" > /dev/null && { echo "QUEUE9 training process still alive; not restarting"; exit 1; }
sleep 15   # let the r6b supervisor (queue8) exit after its training process
if python3 -c "import sys; sys.exit(0 if $air < 0.10 else 1)"; then
  EXTRA="--style_from 1000 --feet_air_w 20 --feet_air_min 0.1 --gate_air 0.15"
else
  EXTRA="--gate_air 0.10"
fi
echo "QUEUE9 $(date -u +%F' '%T) restart r6b from last.pt with: $EXTRA"
../env/venv/bin/python scripts/supervise_node.py --name r6b --iters 20000 --num_envs 8192 -- $BASE $EXTRA
echo "QUEUE9 $(date -u +%F' '%T) end r6b"
