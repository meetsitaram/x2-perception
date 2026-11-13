#!/bin/bash
# r7 + stops at down edges (SSR agent decision 2026-11-12 14:20): --edge_stop 0.3 (30 % of tile episodes stop for 2 s at the first
# >= 0.15 m drop 0.45 m ahead). The relative kernel is dropped. Fall-backs on the deterministic probes and tilt + diverged only:
#   +100 and +250: speed at 0.5 >= 0.45, at 1.0 >= 0.75, yaw from a stand at +-0.8 >= 0.4 both sides, survival >= 0.9,
#   forward 0.2 from a stand >= 0.10, clearance at 0.5 >= 0.05 (no regression on what the policy already does).
# Fall-back: resume from eval/export/ckpts/r7_pre_edge.pt with runs/r7_flags_v5.txt.
R=/mnt/filesystem-g4/tracks/ssr/runs; E=/mnt/filesystem-g4/tracks/ssr/eval; C=/mnt/filesystem-g4/tracks/ssr/code; PY=/mnt/filesystem-g4/tracks/ssr/env/venv/bin/python
say() { echo "QUEUE21 $(date -u +%F' '%T) $*"; }
lastit() { tail -1 $R/r7/log.jsonl | python3 -c "import json,sys; print(json.loads(sys.stdin.read())['it'])"; }
stopall() { touch $R/STOP; for i in $(seq 1 60); do pgrep -f "ssr[.]train" > /dev/null || break; sleep 10; done; sleep 15; for i in $(seq 1 30); do pgrep -f "supervise_node.py --name r7" > /dev/null || break; sleep 5; done; ! pgrep -f "ssr[.]train" > /dev/null; }
probe() { tag=$1; cp $R/r7/last.pt $E/export/ckpts/$tag.pt
  (cd $E && X2_EXCL_PELVIS_HIPROLL=1 PROBE_CAMERA=chest_rs bash speed_bins.sh ../eval/export/ckpts/$tag.pt $tag > speed_bins/$tag.run.log 2>&1)
  (cd $C && X2_EXCL_PELVIS_HIPROLL=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 X2_PLANT=mc_20261024 nice -n 10 $PY scripts/probe_turn.py $E/export/ckpts/$tag.pt chest_rs 2>/dev/null | tail -1 > $E/turn/$tag.json)
  say "probe $tag: $(tail -1 $E/speed_bins/$tag.run.log) | turn: $(cat $E/turn/$tag.json)"; }
verdict() { python3 -c "
import json
r=json.load(open('$E/speed_bins/$1.json')); t=json.load(open('$E/turn/$1.json'))
v=lambda c: [x['vx'] for x in r if x['cmd']==c][0]
c5=json.loads(open('$E/speed_bins/$1_log_v0.5.txt').read().strip().splitlines()[-1])['swing_clearance_m']['median_real_steps']
y8=min(abs(t['(0, 0, 0.8)']['yaw_rate']), abs(t['(0, 0, -0.8)']['yaw_rate'])); y4=min(abs(t['(0, 0, 0.4)']['yaw_rate']), abs(t['(0, 0, -0.4)']['yaw_rate']))
f2=t['(0.2, 0, 0)']['vx']; s=min(x['survived'] for x in r)
base=v(0.5)>=0.45 and v(1.0)>=0.75 and y8>=0.4 and s>=0.9 and f2>=0.10 and c5>=0.05
full=base
print(int(base if '$2'=='base' else full), 'v0.5 %.2f v1.0 %.2f yaw0.8 %.2f yaw0.4 %.2f fwd0.2 %.2f clear %.3f surv %.2f' % (v(0.5), v(1.0), y8, y4, f2, c5, s))"; }
stopall || { say "trainer alive; nothing changed"; exit 1; }
S=$(lastit); [ -f $E/export/ckpts/r7_pre_edge.pt ] || cp $R/r7/last.pt $E/export/ckpts/r7_pre_edge.pt
V5=$(cat $R/r7_flags_v5.txt); V6="$V5 --edge_stop 0.3"; echo "$V6" > $R/r7_flags_v7.txt
cd $C
say "resume r7 at $S with edge stops"
$PY scripts/supervise_node.py --name r7 --iters 20000 --num_envs 8192 -- $V6 &
SUP=$!
fallback() { say "FALL BACK at $(lastit): $*"; kill -0 $SUP 2>/dev/null && { touch $R/STOP; wait $SUP; }; stopall || exit 1
  cp $E/export/ckpts/r7_pre_edge.pt $R/r7/last.pt; say "resume r7 from r7_pre_edge ($S) with r7_flags_v5"
  exec $PY scripts/supervise_node.py --name r7 --iters 20000 --num_envs 8192 -- $V5; }
p100=0; p250=0; next=$(( (S / 250 + 1) * 250 ))
while kill -0 $SUP 2>/dev/null; do
  it=$(lastit 2>/dev/null || echo 0); n=$((it - S))
  if [ $n -ge 50 ]; then
    bad=$(tail -5 $R/r7/log.jsonl | python3 -c "
import json,sys
L=[json.loads(l) for l in sys.stdin if l.strip()]
print(round(sum(d['done_share'].get('tilt',0)+d['done_share'].get('diverged',0) for d in L)/len(L),3))")
    python3 -c "import sys; sys.exit(0 if $bad > 0.15 else 1)" && fallback "tilt+div $bad"
  fi
  if [ $n -ge 105 ] && [ $p100 = 0 ]; then p100=1; probe r7_edge_p100; read ok msg <<< "$(verdict r7_edge_p100 base)"; say "+100: $msg"; [ "$ok" = 1 ] || fallback "+100 regression: $msg"; fi
  if [ $n -ge 255 ] && [ $p250 = 0 ]; then p250=1; probe r7_edge_p250; read ok msg <<< "$(verdict r7_edge_p250 full)"; say "+250: $msg"; [ "$ok" = 1 ] && say "edge stops kept; counters: $(tail -1 $R/r7/log.jsonl | grep -o '"edge_stop": {[^}]*}')" || fallback "+250 marks missed: $msg"; fi
  if [ "$it" -ge $((next + 5)) ] && [ $next -le 4000 ]; then probe $(printf "r7_%06d" $next) & next=$((next + 250)); fi
  sleep 60
done
say "r7 supervisor ended"
