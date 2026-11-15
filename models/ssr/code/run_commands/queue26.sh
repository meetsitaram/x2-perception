#!/bin/bash
# r7 + extended map, second start (SSR agent decision 2026-11-12 17:25): the first ext run (16:28-17:25) placed every route 2 m
# off its tile (terrain.add_extended used cx - BORDER as the tile centre); fixed, and route starts now need 0.35 m clearance and
# level ground. Restart from r7_pre_ext (3,550, before any ext training), flags runs/r7_flags_v10.txt; marks relative to it.
R=/mnt/filesystem-g4/tracks/ssr/runs; E=/mnt/filesystem-g4/tracks/ssr/eval; C=/mnt/filesystem-g4/tracks/ssr/code; PY=/mnt/filesystem-g4/tracks/ssr/env/venv/bin/python
ENVV="X2_EXCL_PELVIS_HIPROLL=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=12 X2_PLANT=mc_20261024"
say() { echo "QUEUE26 $(date -u +%F' '%T) $*"; }
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
PREV=$R/r7_flags_v9.txt
SPS0=29994
stopall || { say "trainer alive; nothing changed"; exit 1; }
BASE=r7_pre_ext; cp $E/export/ckpts/$BASE.pt $R/r7/last.pt
S=$(CUDA_VISIBLE_DEVICES= $PY -c "import torch; print(int(torch.load('$E/export/ckpts/$BASE.pt', map_location='cpu', weights_only=False)['iter']))")
V7=$(cat $PREV); V8=$(cat $R/r7_flags_v10.txt)
cd $C
say "resume r7 at $S with the extended map: --ext_frac 0.3 (base samples/s $SPS0)"
$PY scripts/supervise_node.py --name r7 --iters 20000 --num_envs 8192 -- $V8 &
SUP=$!
probe3 $BASE
mkdir -p $E/post; (cd $C && env $ENVV nice -n 10 $PY scripts/probe_post.py $E/export/ckpts/$BASE.pt wall_left_0.25,stub_close,post_close 0.37 2>/dev/null | tail -1 > $E/post/$BASE.json); say "post probe $BASE: $(cat $E/post/$BASE.json)"
fallback() { say "FALL BACK at $(lastit): $*"; stopall || exit 1
  cp $E/export/ckpts/r7_pre_ext.pt $R/r7/last.pt; say "resume r7 from r7_pre_ext with $PREV"; cd $C
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
    sps=$(tail -5 $R/r7/log.jsonl | python3 -c "import json,sys; L=[json.loads(l) for l in sys.stdin if l.strip()]; print(int(sum(d['samples_per_s'] for d in L)/len(L)))")
    python3 -c "import sys; sys.exit(0 if $sps < 0.6 * $SPS0 else 1)" && fallback "samples/s $sps < 60 % of $SPS0"
  fi
  for k in 100 250; do
    eval "done_k=\$p$k"
    if [ $n -ge $((k + 5)) ] && [ "$done_k" = 0 ]; then
      eval "p$k=1"; tag=r7_ext2_p$k; cp $R/r7/last.pt $E/export/ckpts/$tag.pt; probe3 $tag
      read ok msg rest <<< "$(verdict $BASE $tag)"; say "+$k: $ok $msg $rest"
      [ "$ok" = 1 ] || fallback "+$k: $msg"
      (cd $C && env $ENVV nice -n 10 $PY scripts/probe_post.py $E/export/ckpts/$tag.pt wall_left_0.25,stub_close,post_close 0.37 2>/dev/null | tail -1 > $E/post/$tag.json); say "post probe $tag: $(cat $E/post/$tag.json)"
      say "ext stats: $(tail -1 $R/r7/log.jsonl | grep -o '"ext": {.*}, "edge_stop"' | cut -c1-600)"
      [ $k = 250 ] && say "extended map kept"
    fi
  done
  sleep 60
done
say "r7 supervisor ended"
