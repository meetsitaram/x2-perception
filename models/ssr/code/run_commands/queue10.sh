#!/bin/bash
# r6b gate at 2,000 (coordinator decision 2026-11-12 05:00). If mean touchdown air time (logs 1,980-2,000) < 0.2 s:
# stop r6b through its supervisor and start r6c = copy of r6b/ckpt_002000.pt continued with step 1
# (--excl_pelvis_hiproll --dof_dev_ref, plus velocity kernel width 0.1 instead of 0.25 for slow commands (--sig_v0 0.1 --cmd_cur_iters 20000), all listed deviations) for 500 iterations (to 2,500). Otherwise leave r6b alone.
R=/mnt/filesystem-g4/tracks/ssr/runs
cd /mnt/filesystem-g4/tracks/ssr/code
BASE="--flat_until 550 --obst0 0.25 --obst_ramp 500 --stage_a 150 --stage_b 400 --w_style 0.2 --style_low 0 --gate_ramp 500 --prior_min_speed 0.2 --rsi_min_speed 0.3 --plant mc_20261024 --camera chest_rs --amp_file data/amp_v2.npz --prior_balance --disc_mask_arms --style_from 1000 --feet_air_w 20 --feet_air_min 0.1 --gate_air 0.15"
while true; do
  it=$(tail -1 $R/r6b/log.jsonl | python3 -c "import json,sys; print(json.loads(sys.stdin.read())['it'])")
  [ "$it" -ge 2005 ] && [ -f $R/r6b/ckpt_002000.pt ] && break
  sleep 20
done
python3 - > $R/gate2000.json <<'PY'
import json
L = [json.loads(l) for l in open("/mnt/filesystem-g4/tracks/ssr/runs/r6b/log.jsonl") if l.strip()]
L = [d for d in L if 1980 <= d["it"] <= 2000]
m = lambda f: round(sum(f(d) for d in L) / len(L), 4)
print(json.dumps(dict(its=[d["it"] for d in L], air_td=m(lambda d: d["air_td"]), air_long_share=m(lambda d: d["air_long_share"]),
      vx_ratio=m(lambda d: d["gait"]["vx_ratio"]), steps_per_s=m(lambda d: d["gait"]["steps_per_s"]), single=m(lambda d: d["gait"]["single"]),
      lvl_mean=m(lambda d: d["lvl_mean"]), levels=L[-1]["levels_by_type"], done_share=L[-1].get("done_share"))))
PY
cat $R/gate2000.json
air=$(python3 -c "import json; print(json.load(open('$R/gate2000.json'))['air_td'])")
if python3 -c "import sys; sys.exit(0 if $air >= 0.2 else 1)"; then
  echo "QUEUE10 $(date -u +%F' '%T) gate passed (air $air); r6b left alone"; exit 0
fi
echo "QUEUE10 $(date -u +%F' '%T) gate failed (air $air); stopping r6b for step 1"
touch $R/STOP
for i in $(seq 1 60); do pgrep -f "ssr.train" > /dev/null || break; sleep 10; done
pgrep -f "ssr.train" > /dev/null && { echo "QUEUE10 training process still alive; not starting r6c"; exit 1; }
for i in $(seq 1 30); do pgrep -f "supervise_node.py --name r6b" > /dev/null || break; sleep 5; done
pgrep -f "supervise_node.py --name r6b" > /dev/null && { echo "QUEUE10 r6b supervisor still alive; not starting r6c"; exit 1; }
mkdir -p $R/r6c && cp $R/r6b/ckpt_002000.pt $R/r6c/last.pt
echo "QUEUE10 $(date -u +%F' '%T) start r6c from r6b/ckpt_002000.pt with step 1"
../env/venv/bin/python scripts/supervise_node.py --name r6c --iters 2500 --num_envs 8192 -- $BASE --excl_pelvis_hiproll --dof_dev_ref --sig_v0 0.1 --cmd_cur_iters 20000
echo "QUEUE10 $(date -u +%F' '%T) end r6c"
