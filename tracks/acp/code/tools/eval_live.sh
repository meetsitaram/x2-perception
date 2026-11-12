#!/usr/bin/env bash
# Run the paper evaluation on the training GPU WHILE training continues (user, 2026-11-11), with guard rails:
# separate process; starts only after the checkpoint file is complete; one eval at a time (lock); robot count capped so
# total GPU memory stays under 80 GB; niced and pinned to four CPU cores; wall-clock limit; the eval aborts itself if free
# GPU memory drops under 10 GB (tools/paper_eval.py). Measures training speed before / during / after from metrics.jsonl.
#   [EVAL_TIMEOUT=900] bash tools/eval_live.sh RUN_DIR ITERATION EPISODES_PER_SET_PER_SEED [PLANT]   (3 seeds, fixed in paper_eval.py)
set -u
RUN=$1; IT=$(printf %06d "$2"); PER=$3; PLANT=${4:-}; export ACP_EVAL_SEEDS=${5:-3}
T=/mnt/filesystem-g4/tracks/acp
PY=$HOME/acp_env/bin/python
CK=$RUN/ckpt_$IT.pt
OUT=$T/eval/paper_eval/live_$(basename "$RUN")_${IT}_per${PER}
exec 9> /tmp/acp_eval.lock
flock -n 9 || { echo "another eval is running; not starting"; exit 3; }
while [ ! -f "$CK" ]; do sleep 5; done
while [ $(( $(date +%s) - $(stat -c %Y "$CK") )) -lt 15 ]; do sleep 3; done          # the checkpoint is fully written
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
need=$(( PER * 9 * 5 / 1000 + 6 ))                                                    # GB: about 5 MB per robot + 6 GB overhead (measured below)
if [ $(( used / 1024 + need )) -gt 80 ]; then echo "would exceed 80 GB (used $((used / 1024)) GB + about $need GB); not starting"; exit 4; fi
t0=$(date +%s)
( while kill -0 $$ 2>/dev/null; do nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits; sleep 3; done > "$OUT.gpu.txt" ) &
mon=$!
cd "$T/code"
w0=$(grep -c "nefc overflow\|height field collision overflow\|nan" "$RUN/train.out")
ACP_EVAL_NO_DUMP=1 ACP_PLANT=$PLANT nice -n 10 taskset -c 0-3 timeout ${EVAL_TIMEOUT:-900} "$PY" tools/paper_eval.py "$CK" "$OUT.json" "$PER" > "$OUT.out" 2>&1
rc=$?
t1=$(date +%s)
kill $mon 2>/dev/null
w1=$(grep -c "nefc overflow\|height field collision overflow\|nan" "$RUN/train.out")
echo "trainer log warnings (nefc / height-field overflow / nan lines) during the eval: $((w1 - w0))" | tee "$OUT.warn.txt"
python3 - "$RUN/metrics.jsonl" "$t0" "$t1" "$OUT" "$rc" "$PER" <<'EOF'
import json, sys, time, calendar
path, t0, t1, out, rc, per = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], int(sys.argv[5]), int(sys.argv[6])
rows = [json.loads(l) for l in open(path)][-400:]
def ts(r): return calendar.timegm(time.strptime(r["time"], "%Y-%m-%dT%H:%M:%S")) if time.timezone == 0 else time.mktime(time.strptime(r["time"], "%Y-%m-%dT%H:%M:%S"))
before = [r for r in rows if t0 - 90 <= ts(r) < t0 - 5][-10:]
during = [r for r in rows if t0 + 8 <= ts(r) <= t1]
mean = lambda rs, k: round(sum(r[k] for r in rs) / max(len(rs), 1), 2)
gpu = [tuple(float(x) for x in l.split(",")) for l in open(out + ".gpu.txt") if "," in l]
res = dict(robots=per * 9, eval_seconds=t1 - t0, eval_rc=rc, peak_gpu_mem_gb=round(max(g[0] for g in gpu) / 1024, 1) if gpu else None,
           train_steps_per_s_before=mean(before, "env_steps_per_s"), train_steps_per_s_during=mean(during, "env_steps_per_s"),
           iter_s_before=mean(before, "iter_s"), iter_s_during=mean(during, "iter_s"), iterations_during=len(during),
           value_loss_before=mean(before, "ppo/value_loss"), value_loss_during=mean(during, "ppo/value_loss"))
if res["train_steps_per_s_before"]:
    res["training_slowdown_pct"] = round(100 * (1 - res["train_steps_per_s_during"] / res["train_steps_per_s_before"]), 1)
json.dump(res, open(out + ".live.json", "w"), indent=1)
print(json.dumps(res))
EOF
