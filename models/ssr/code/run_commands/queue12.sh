#!/bin/bash
# r6d (SSR agent decision 2026-11-12 05:58): r6b from 2,100 + ONE change: velocity kernel 0.25 -> 0.1 ramped over 300 its (slow commands).
# Tripwires: catastrophic at +50 (tilt+diverged > 50 % of ends); trend from +150 (mean of last 15 logs: speed share < 45 % or tilt+div > 16.6 %).
# Probes at +250, +500. At +500 (2,600): continue r6d if speed at a 0.35 m/s command >= 0.20 m/s and share >= 54 %, else start r7 (fresh).
R=/mnt/filesystem-g4/tracks/ssr/runs; E=/mnt/filesystem-g4/tracks/ssr/eval; C=/mnt/filesystem-g4/tracks/ssr/code
R6B="--flat_until 550 --obst0 0.25 --obst_ramp 500 --stage_a 150 --stage_b 400 --w_style 0.2 --style_low 0 --gate_ramp 500 --prior_min_speed 0.2 --rsi_min_speed 0.3 --plant mc_20261024 --camera chest_rs --amp_file data/amp_v2.npz --prior_balance --disc_mask_arms --style_from 1000 --feet_air_w 20 --feet_air_min 0.1 --gate_air 0.15"
S=2100; END=2600
R6D="$R6B --sig_v_to 0.1 --sig_v_from $S --sig_v_ramp 300"
L=$R/r6d/log.jsonl
say() { echo "QUEUE12 $(date -u +%F' '%T) $*"; }
lastit() { tail -1 $1 | python3 -c "import json,sys; print(json.loads(sys.stdin.read())['it'])"; }
stats() { tail -n $1 $L | python3 -c "
import json,sys
L=[json.loads(l) for l in sys.stdin if l.strip()]
m=lambda f: sum(f(d) for d in L)/len(L)
print(L[-1]['it'], round(m(lambda d: d['air_td']),4), round(m(lambda d: d['gait']['vx_ratio']),4), round(m(lambda d: d['gait']['steps_per_s']),3), round(m(lambda d: d['done_share'].get('tilt',0)+d['done_share'].get('diverged',0)),4), round(m(lambda d: d['lvl_mean']),3), round(L[-1]['sig_v'],3))"; }
gone() { for i in $(seq 1 60); do pgrep -f "ssr[.]train" > /dev/null || break; sleep 10; done; sleep 15; ! pgrep -f "ssr[.]train" > /dev/null; }
probe() { cp $R/r6d/last.pt $E/export/ckpts/r6d_$1.pt; cp $R/r6d/args.json $E/export/ckpts/args.json; (cd $E && bash speed_bins.sh ../eval/export/ckpts/r6d_$1.pt r6d_$1 > speed_bins/r6d_$1.run.log 2>&1); say "probe r6d_$1: $(tail -1 $E/speed_bins/r6d_$1.run.log)"; }
# export r6b 2,000 for the eval (CPU)
(cd $C && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 X2_PLANT=mc_20261024 nice -n 10 ../env/venv/bin/python scripts/export_eval.py --ckpt ../eval/export/ckpts/r6b_002000.pt --out ../eval/export/r6b_002000 > ../eval/export/r6b_002000.log 2>&1; say "export r6b_002000: $(tail -1 ../eval/export/r6b_002000.log)") &
until [ "$(lastit $R/r6b/log.jsonl)" -ge $((S+5)) ]; do sleep 20; done
say "stopping r6b at $(lastit $R/r6b/log.jsonl) for r6d"
touch $R/STOP; gone || { say "trainer alive; nothing started"; exit 1; }
for i in $(seq 1 30); do pgrep -f "supervise_node.py --name r6b" > /dev/null || break; sleep 5; done
mkdir -p $R/r6d && cp $R/r6b/last.pt $R/r6d/last.pt
cd $C
../env/venv/bin/python scripts/supervise_node.py --name r6d --iters $END --num_envs 8192 -- $R6D &
SUP=$!
say "r6d started from r6b/last.pt (supervisor pid $SUP)"
until [ -s $L ]; do sleep 30; done
p1=0
while kill -0 $SUP 2>/dev/null; do
  read it air vx steps bad lvl sig <<< $(stats 3)
  n=$((it - S))
  harm=0
  [ $n -ge 50 ] && [ $n -lt 150 ] && harm=$(python3 -c "print(int($bad > 0.5))")
  if [ $n -ge 150 ]; then read it air vx steps bad lvl sig <<< $(stats 15); harm=$(python3 -c "print(int($vx < 0.45 or $bad > 0.166))"); fi
  if [ "$harm" = 1 ]; then
    say "harm at $it (+$n; air $air vx $vx steps $steps tilt+div $bad lvl $lvl sig_v $sig): stop r6d, start r7 (fresh)"
    touch $R/STOP; wait $SUP; gone || { say "trainer alive"; exit 1; }
    bash $C/run_commands/step2_r7.sh; exit 0
  fi
  [ $n -ge 250 ] && [ $p1 = 0 ] && { p1=1; say "+250 at $it: air $air vx $vx steps $steps tilt+div $bad lvl $lvl sig_v $sig"; probe $((S+250)) & }
  sleep 60
done
read it air vx steps bad lvl sig <<< $(stats 3)
say "r6d end $it: air $air vx $vx steps $steps tilt+div $bad lvl $lvl sig_v $sig"
gone || { say "trainer alive at end"; exit 1; }
wait; probe $END
s035=$(python3 -c "import json; print([r['vx'] for r in json.load(open('$E/speed_bins/r6d_$END.json')) if r['cmd']==0.35][0])")
dec=$(python3 -c "print('continue' if ($s035 >= 0.20 and $vx >= 0.54) else 'r7')")
say "decision at $END: $dec (speed at 0.35 cmd $s035 m/s, share $vx)"
if [ "$dec" = continue ]; then
  ../env/venv/bin/python scripts/supervise_node.py --name r6d --iters 20000 --num_envs 8192 -- $R6D
else
  bash run_commands/step2_r7.sh
fi
