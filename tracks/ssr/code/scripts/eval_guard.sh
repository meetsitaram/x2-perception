#!/bin/bash
# Run one evaluation/probe on the GPU beside training, under limits (user rule 2026-11-11). usage: bash scripts/eval_guard.sh <tag> <seconds limit> <command...>
# Refuses if another guarded eval runs, if free GPU memory is under 10 GB, or if used memory is over 70 GB (so the total stays under 80 of 96).
cd /mnt/filesystem-g4/tracks/ssr/code
R=/mnt/filesystem-g4/tracks/ssr/runs
tag=$1; lim=$2; shift 2
exec 9> /tmp/ssr_eval.lock; flock -n 9 || { echo "another eval is running"; exit 3; }
read used total <<< $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | tr -d ",")
[ $((total - used)) -lt 10000 ] && { echo "free GPU memory under 10 GB: not starting"; exit 4; }
[ $used -gt 70000 ] && { echo "GPU memory over 70 GB used: not starting"; exit 4; }
t0=$(date +%s); echo "eval $tag start $(date +%H:%M:%S) gpu_used_mb $used" >> $R/eval_guard.log
( while sleep 5; do read u t <<< $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | tr -d ","); echo $u >> /tmp/ssr_eval_mem.$$; [ $((t - u)) -lt 10000 ] && pkill -TERM -f "EVAL_TAG_$tag"; done ) 9>&- & W=$!
EVAL_TAG=EVAL_TAG_$tag nice -n 10 timeout $lim "$@" --evaltag EVAL_TAG_$tag; code=$?
kill $W 2>/dev/null
echo "eval $tag end $(date +%H:%M:%S) exit $code seconds $(( $(date +%s) - t0 )) peak_gpu_mb $(sort -n /tmp/ssr_eval_mem.$$ | tail -1)" >> $R/eval_guard.log; rm -f /tmp/ssr_eval_mem.$$
exit $code
