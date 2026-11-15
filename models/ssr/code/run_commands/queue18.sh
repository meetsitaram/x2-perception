#!/bin/bash
# r7 + more turn-in-place commands (SSR agent decision 2026-11-12 12:10): --omni_frac 0.3 --omni_turn 0.5 (pure turns 15 % of robots, was 6 %).
# Snapshot before the change: eval/export/ckpts/r7_pre_turn.pt. Fall-back = resume from it with r7_flags_v4 (swing kept, old mix).
# +250 probe also fails if yaw from a stand at +-0.8 is under 0.4 rad/s on either side (turning is in every fall-back test from now on).
# Tripwire +50..+250 (mean of last 5 logs): speed share < 0.60 or tilt+diverged > 0.15 -> fall back.
# +250 probe (CPU): fall back if clearance (real steps, 0.5 m/s) < 0.035 or speed at 0.5 < 0.45 or survival < 0.9 at any command.
R=/mnt/filesystem-g4/tracks/ssr/runs; E=/mnt/filesystem-g4/tracks/ssr/eval; C=/mnt/filesystem-g4/tracks/ssr/code; PY=/mnt/filesystem-g4/tracks/ssr/env/venv/bin/python
say() { echo "QUEUE18 $(date -u +%F' '%T) $*"; }
lastit() { tail -1 $R/r7/log.jsonl | python3 -c "import json,sys; print(json.loads(sys.stdin.read())['it'])"; }
stopall() { touch $R/STOP; for i in $(seq 1 60); do pgrep -f "ssr[.]train" > /dev/null || break; sleep 10; done; sleep 15; for i in $(seq 1 30); do pgrep -f "supervise_node.py --name r7" > /dev/null || break; sleep 5; done; ! pgrep -f "ssr[.]train" > /dev/null; }
probe() { tag=$1; cp $R/r7/last.pt $E/export/ckpts/$tag.pt
  (cd $E && X2_EXCL_PELVIS_HIPROLL=1 PROBE_CAMERA=chest_rs bash speed_bins.sh ../eval/export/ckpts/$tag.pt $tag > speed_bins/$tag.run.log 2>&1)
  (cd $C && X2_EXCL_PELVIS_HIPROLL=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 X2_PLANT=mc_20261024 nice -n 10 $PY scripts/probe_turn.py $E/export/ckpts/$tag.pt chest_rs 2>/dev/null | tail -1 > $E/turn/$tag.json)
  say "probe $tag: $(tail -1 $E/speed_bins/$tag.run.log) | turn: $(cat $E/turn/$tag.json) | clearance@0.5: $(tail -1 $E/speed_bins/${tag}_log_v0.5.txt | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print(d['swing_clearance_m']['median_real_steps'], d['step_length_m']['median_all'])")"; }
stopall || { say "trainer alive; nothing changed"; exit 1; }
S=$(lastit); cp $R/r7/last.pt $E/export/ckpts/r7_pre_turn.pt
V3=$(cat $R/r7_flags_v4.txt); V4="$V3 --omni_frac 0.3 --omni_turn 0.5"; echo "$V4" > $R/r7_flags_v5.txt
cd $C
say "resume r7 at $S with the turn-heavy command mix"
$PY scripts/supervise_node.py --name r7 --iters 20000 --num_envs 8192 -- $V4 &
SUP=$!
fallback() { say "FALL BACK at $(lastit): $*"; kill -0 $SUP 2>/dev/null && { touch $R/STOP; wait $SUP; }; stopall || exit 1
  cp $E/export/ckpts/r7_pre_turn.pt $R/r7/last.pt; say "resume r7 from r7_pre_turn ($S) with r7_flags_v4"
  exec $PY scripts/supervise_node.py --name r7 --iters 20000 --num_envs 8192 -- $V3; }
p250=0; next=$(( (S / 250 + 1) * 250 ))
while kill -0 $SUP 2>/dev/null; do
  it=$(lastit 2>/dev/null || echo 0); n=$((it - S))
  if [ $n -ge 50 ] && [ $n -le 250 ]; then
    read vx bad sw <<< $(tail -5 $R/r7/log.jsonl | python3 -c "
import json,sys
L=[json.loads(l) for l in sys.stdin if l.strip()]
m=lambda f: sum(f(d) for d in L)/len(L)
print(round(m(lambda d: d['gait']['vx_ratio']),3), round(m(lambda d: d['done_share'].get('tilt',0)+d['done_share'].get('diverged',0)),3), round(m(lambda d: d['terms'].get('swing_h',0)),3))")
    python3 -c "import sys; sys.exit(0 if ($vx < 0.60 or $bad > 0.15) else 1)" && fallback "share $vx tilt+div $bad swing_h $sw"
  fi
  if [ $n -ge 255 ] && [ $p250 = 0 ]; then p250=1; probe r7_turn_p250
    c=$(tail -1 $E/speed_bins/r7_turn_p250_log_v0.5.txt | python3 -c "import json,sys; print(json.loads(sys.stdin.read())['swing_clearance_m']['median_real_steps'])")
    ok=$(python3 -c "
import json; r=json.load(open('$E/speed_bins/r7_turn_p250.json')); v5=[x['vx'] for x in r if x['cmd']==0.5][0]; s=min(x['survived'] for x in r)
t=json.load(open('$E/turn/r7_turn_p250.json')); y=min(abs(t['(0, 0, 0.8)']['yaw_rate']), abs(t['(0, 0, -0.8)']['yaw_rate']))
print(int($c >= 0.035 and v5 >= 0.45 and s >= 0.9 and y >= 0.4))")
    [ "$ok" = 1 ] && say "turn mix kept (clearance $c)" || fallback "+250 probe failed (clearance $c, see turn json)"
  fi
  if [ "$it" -ge $((next + 5)) ] && [ $next -le 4000 ]; then probe $(printf "r7_%06d" $next) & next=$((next + 250)); fi
  sleep 60
done
say "r7 supervisor ended"
