#!/bin/bash
# r7 resumed with the command mix (SSR agent decision 2026-11-12 10:20): 20 % omni robots (was 10 %), of them 30 % turn in place
# (|wz| 0.3-1.2, no forward), 30 % slow forward 0.1-0.4 m/s; every robot gets a new command every 8 s (from whatever it is doing).
# Probes every 250 its (CPU): speed per command (eval/speed_bins/r7_NNNNNN.json) and turn/start-from-stand (eval/turn/r7_NNNNNN.json).
R=/mnt/filesystem-g4/tracks/ssr/runs; E=/mnt/filesystem-g4/tracks/ssr/eval; C=/mnt/filesystem-g4/tracks/ssr/code; PY=/mnt/filesystem-g4/tracks/ssr/env/venv/bin/python
say() { echo "QUEUE16 $(date -u +%F' '%T) $*"; }
lastit() { tail -1 $R/r7/log.jsonl | python3 -c "import json,sys; print(json.loads(sys.stdin.read())['it'])"; }
touch $R/STOP
for i in $(seq 1 60); do pgrep -f "ssr[.]train" > /dev/null || break; sleep 10; done; sleep 15
pgrep -f "ssr[.]train" > /dev/null && { say "trainer alive; nothing started"; exit 1; }
for i in $(seq 1 30); do pgrep -f "supervise_node.py --name r7" > /dev/null || break; sleep 5; done
FLAGS="$(cat $R/r7_flags_v2.txt) --omni_frac 0.2 --omni_turn 0.3 --omni_slow 0.3 --resample_s 8"; echo "$FLAGS" > $R/r7_flags_v3.txt
cd $C
say "resume r7 from last.pt at $(lastit): $FLAGS"
$PY scripts/supervise_node.py --name r7 --iters 20000 --num_envs 8192 -- $FLAGS &
SUP=$!
mkdir -p $E/turn
next=$(( ($(lastit) / 250 + 1) * 250 ))
while kill -0 $SUP 2>/dev/null; do
  it=$(lastit 2>/dev/null || echo 0)
  if [ "$it" -ge $((next + 5)) ] && [ $next -le 4000 ]; then
    tag=$(printf "r7_%06d" $next); cp $R/r7/last.pt $E/export/ckpts/$tag.pt
    (cd $E && X2_EXCL_PELVIS_HIPROLL=1 PROBE_CAMERA=chest_rs bash speed_bins.sh ../eval/export/ckpts/$tag.pt $tag > speed_bins/$tag.run.log 2>&1
     cd $C && X2_EXCL_PELVIS_HIPROLL=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 X2_PLANT=mc_20261024 nice -n 10 $PY scripts/probe_turn.py $E/export/ckpts/$tag.pt chest_rs 2>/dev/null | tail -1 > $E/turn/$tag.json
     say "probe $tag (ckpt near $it): $(tail -1 $E/speed_bins/$tag.run.log) | turn/start: $(cat $E/turn/$tag.json)") &
    next=$((next + 250))
  fi
  sleep 60
done
say "r7 supervisor ended"
