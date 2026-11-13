#!/bin/bash
# r7 restart without --default_from_ref (SSR agent decision 2026-11-12 08:15: e3 falls in the deterministic probe; its 0.311 training
# share counted falling motion). The first r7 (with the reference pose, 50 iterations) is kept as runs/r7_dfr.
# Deterministic per-command probes (CPU, exclusion on, chest_rs) every 250 iterations from 250 to 2,000: eval/speed_bins/r7_NNNNNN.json.
R=/mnt/filesystem-g4/tracks/ssr/runs; E=/mnt/filesystem-g4/tracks/ssr/eval; C=/mnt/filesystem-g4/tracks/ssr/code; PY=/mnt/filesystem-g4/tracks/ssr/env/venv/bin/python
say() { echo "QUEUE15 $(date -u +%F' '%T) $*"; }
for i in $(seq 1 60); do pgrep -f "ssr[.]train" > /dev/null || break; sleep 10; done; sleep 15
pgrep -f "ssr[.]train" > /dev/null && { say "trainer alive; nothing started"; exit 1; }
for i in $(seq 1 30); do pgrep -f "supervise_node.py --name r7" > /dev/null || break; sleep 5; done
[ -d $R/r7 ] && mv $R/r7 $R/r7_dfr
FLAGS=$(sed "s/ --default_from_ref//" $R/r7_flags.txt); echo "$FLAGS" > $R/r7_flags_v2.txt
cd $C
say "start r7: $FLAGS"
$PY scripts/supervise_node.py --name r7 --iters 20000 --num_envs 8192 -- $FLAGS &
SUP=$!
next=250
while kill -0 $SUP 2>/dev/null; do
  it=$(tail -1 $R/r7/log.jsonl 2>/dev/null | python3 -c "import json,sys; print(json.loads(sys.stdin.read())['it'])" 2>/dev/null || echo 0)
  if [ "$it" -ge $((next + 5)) ] && [ $next -le 2000 ]; then
    tag=$(printf "r7_%06d" $next); cp $R/r7/last.pt $E/export/ckpts/$tag.pt
    (cd $E && X2_EXCL_PELVIS_HIPROLL=1 PROBE_CAMERA=chest_rs bash speed_bins.sh ../eval/export/ckpts/$tag.pt $tag > speed_bins/$tag.run.log 2>&1; say "probe $tag (ckpt near $it): $(tail -1 speed_bins/$tag.run.log)") &
    next=$((next + 250))
  fi
  sleep 60
done
say "r7 supervisor ended"
