#!/bin/bash
# r7: edge stops kept (two +100 probes without a real regression) + slow / turning windows in the motion prior
# (--prior_min_speed 0.05 --prior_turn_min 0.3). SSR agent decision 2026-11-12 15:10. Start: the edge run's 3,040 checkpoint
# (eval/export/ckpts/r7_edge2_p100.pt), which is also the base for every mark.
# Marks are relative to the base, probed in this job with the same seeds and robot counts (probe_start 64 robots, probe_turn
# 8 robots, probe_gait 16 robots), and only on properties the base passes:
#   start share (mean vx >= 0.5 v over the last 4 s, 64 robots, start yaw +-15 deg) at 0.35 / 0.5 / 0.6 >= base - 0.15;
#   yaw from a stand at +-0.8 >= 0.4 (both sides) if the base has it; clearance at 0.5 >= base - 0.015; survival >= base - 0.1;
#   tilt + diverged ends <= 0.15. Fall-back: resume from the base with runs/r7_flags_v7.txt (edge stops, old prior).
R=/mnt/filesystem-g4/tracks/ssr/runs; E=/mnt/filesystem-g4/tracks/ssr/eval; C=/mnt/filesystem-g4/tracks/ssr/code; PY=/mnt/filesystem-g4/tracks/ssr/env/venv/bin/python
ENVV="X2_EXCL_PELVIS_HIPROLL=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=12 X2_PLANT=mc_20261024"
say() { echo "QUEUE23 $(date -u +%F' '%T) $*"; }
lastit() { tail -1 $R/r7/log.jsonl | python3 -c "import json,sys; print(json.loads(sys.stdin.read())['it'])"; }
stopall() { touch $R/STOP; for i in $(seq 1 60); do pgrep -f "ssr[.]train" > /dev/null || break; sleep 10; done; sleep 15; for i in $(seq 1 30); do pgrep -f "supervise_node.py --name r7" > /dev/null || break; sleep 5; done; ! pgrep -f "ssr[.]train" > /dev/null; }
probe3() { # $1 tag (checkpoint eval/export/ckpts/$1.pt)
  cd $C
  env $ENVV nice -n 10 $PY scripts/probe_start.py $E/export/ckpts/$1.pt 64 0.35,0.5,0.6 2>/dev/null | tail -1 > $E/start/$1.json
  env $ENVV nice -n 10 $PY scripts/probe_turn.py $E/export/ckpts/$1.pt chest_rs 2>/dev/null | tail -1 > $E/turn/$1.json
  env $ENVV PROBE_DEVICE=cpu nice -n 10 $PY scripts/probe_gait.py --ckpt $E/export/ckpts/$1.pt --out $E/speed_bins/$1_v0.5 --speed 0.5 --num 16 --steps 300 2>/dev/null | tail -1 > $E/speed_bins/$1_g05.json
  say "probe $1: start $(cat $E/start/$1.json | cut -c1-400) | turn $(cat $E/turn/$1.json) | gait0.5 $(python3 -c "import json; d=json.load(open('$E/speed_bins/$1_g05.json')); print(d['speed_mps'], d['swing_clearance_m']['median_real_steps'], d['survived_share'])")"; }
verdict() { python3 -c "
import json
def L(t):
    s=json.load(open('$E/start/'+t+'.json')); u=json.load(open('$E/turn/'+t+'.json')); g=json.load(open('$E/speed_bins/'+t+'_g05.json'))
    y8=min(abs(u['(0, 0, 0.8)']['yaw_rate']), abs(u['(0, 0, -0.8)']['yaw_rate']))
    return dict(st={v: s[v]['started'] for v in ('0.35','0.5','0.6')}, sv=min(s[v]['survived'] for v in ('0.35','0.5','0.6')), y8=y8, c=g['swing_clearance_m']['median_real_steps'])
b, c = L('$1'), L('$2')
bad=[]
for v in ('0.35','0.5','0.6'):
    if c['st'][v] < b['st'][v]-0.15: bad.append('start%s %.2f<%.2f' % (v, c['st'][v], b['st'][v]))
if b['y8'] >= 0.4 and c['y8'] < 0.4: bad.append('yaw0.8 %.2f' % c['y8'])
if c['c'] < b['c']-0.015: bad.append('clear %.3f<%.3f' % (c['c'], b['c']))
if c['sv'] < b['sv']-0.1: bad.append('survival %.2f' % c['sv'])
print(int(not bad), ';'.join(bad) or 'ok', 'cand', json.dumps(c).replace(' ',''))"; }
mkdir -p $E/start $E/turn
stopall || { say "trainer alive; nothing changed"; exit 1; }
BASE=r7_edge2_p100; cp $E/export/ckpts/$BASE.pt $R/r7/last.pt; cp $E/export/ckpts/$BASE.pt $E/export/ckpts/r7_pre_prior.pt
S=$(CUDA_VISIBLE_DEVICES= $PY -c "import torch; print(int(torch.load('$E/export/ckpts/$BASE.pt', map_location='cpu', weights_only=False)['iter']))")
V7=$(cat $R/r7_flags_v7.txt); V8="$V7 --prior_min_speed 0.05 --prior_turn_min 0.3"; echo "$V8" > $R/r7_flags_v8.txt
cd $C
say "resume r7 from $BASE (about $S) with slow / turning prior windows: $V8"
$PY scripts/supervise_node.py --name r7 --iters 20000 --num_envs 8192 -- $V8 &
SUP=$!
probe3 $BASE
fallback() { say "FALL BACK at $(lastit): $*"; stopall || exit 1
  cp $E/export/ckpts/r7_pre_prior.pt $R/r7/last.pt; say "resume r7 from r7_pre_prior with r7_flags_v7"; cd $C
  exec $PY scripts/supervise_node.py --name r7 --iters 20000 --num_envs 8192 -- $V7; }
p100=0; p250=0
while kill -0 $SUP 2>/dev/null; do
  it=$(lastit 2>/dev/null || echo 0); n=$((it - S))
  if [ $n -ge 50 ]; then
    bad=$(tail -5 $R/r7/log.jsonl | python3 -c "
import json,sys
L=[json.loads(l) for l in sys.stdin if l.strip()]
print(round(sum(d['done_share'].get('tilt',0)+d['done_share'].get('diverged',0) for d in L)/len(L),3))")
    python3 -c "import sys; sys.exit(0 if $bad > 0.15 else 1)" && fallback "tilt+div $bad"
  fi
  for k in 100 250; do
    eval "done_k=\$p$k"
    if [ $n -ge $((k + 5)) ] && [ "$done_k" = 0 ]; then
      eval "p$k=1"; tag=r7_prior_p$k; cp $R/r7/last.pt $E/export/ckpts/$tag.pt; probe3 $tag
      read ok msg rest <<< "$(verdict $BASE $tag)"; say "+$k: $ok $msg $rest"
      [ "$ok" = 1 ] || fallback "+$k: $msg"
      [ $k = 250 ] && say "prior windows kept"
    fi
  done
  sleep 60
done
say "r7 supervisor ended"
