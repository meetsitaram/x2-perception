#!/bin/bash
# r6c watcher (SSR agent decision 2026-11-12 05:45). Fallback to plain r6b on harm; per-bin probes at 2,250 and 2,500; decision at 2,500.
R=/mnt/filesystem-g4/tracks/ssr/runs; E=/mnt/filesystem-g4/tracks/ssr/eval; C=/mnt/filesystem-g4/tracks/ssr/code
L=$R/r6c/log.jsonl
R6B="--flat_until 550 --obst0 0.25 --obst_ramp 500 --stage_a 150 --stage_b 400 --w_style 0.2 --style_low 0 --gate_ramp 500 --prior_min_speed 0.2 --rsi_min_speed 0.3 --plant mc_20261024 --camera chest_rs --amp_file data/amp_v2.npz --prior_balance --disc_mask_arms --style_from 1000 --feet_air_w 20 --feet_air_min 0.1 --gate_air 0.15"
R6C="$R6B --excl_pelvis_hiproll --dof_dev_ref --sig_v0 0.1 --cmd_cur_iters 20000"
say() { echo "QUEUE11 $(date -u +%F' '%T) $*"; }
stats() { tail -n $1 $L | python3 -c "
import json,sys
L=[json.loads(l) for l in sys.stdin if l.strip()]
m=lambda f: sum(f(d) for d in L)/len(L)
print(L[-1]['it'], round(m(lambda d: d['air_td']),4), round(m(lambda d: d['gait']['vx_ratio']),4), round(m(lambda d: d['gait']['steps_per_s']),3), round(m(lambda d: d['done_share'].get('tilt',0)+d['done_share'].get('diverged',0)),4), round(m(lambda d: d['lvl_mean']),3), round(max(d['gait']['vx_ratio'] for d in L),4))"; }
wait_trainer_gone() { for i in $(seq 1 60); do pgrep -f "ssr[.]train" > /dev/null || break; sleep 10; done; sleep 15; ! pgrep -f "ssr[.]train" > /dev/null; }
probe() { cp $R/r6c/last.pt $E/export/ckpts/r6c_$1.pt; cp $R/r6c/args.json $E/export/ckpts/args.json; (cd $E && bash speed_bins.sh ../eval/export/ckpts/r6c_$1.pt r6c_$1 > speed_bins/r6c_$1.run.log 2>&1); say "probe r6c_$1: $(tail -1 $E/speed_bins/r6c_$1.run.log)"; }
until [ -s $L ]; do sleep 30; done
p250=0
while true; do
  read it air vx steps bad lvl vxmax <<< $(stats 3)
  if [ "$it" -ge 2050 ]; then
    harm=$(python3 -c "print(int($vxmax < 0.40 or $bad > 0.166 or $lvl < 4.45))")
    if [ "$harm" = 1 ]; then
      say "harm at $it (air $air vx $vx steps $steps tilt+div $bad lvl $lvl): stop r6c, plain r6b from ckpt_002000"
      touch $R/STOP; wait_trainer_gone || { say "trainer still alive; nothing started"; exit 1; }
      cp $R/r6b/ckpt_002000.pt $R/r6b/last.pt
      cd $C && ../env/venv/bin/python scripts/supervise_node.py --name r6b --iters 20000 --num_envs 8192 -- $R6B; exit 0
    fi
  fi
  if [ "$it" -ge 2250 ] && [ $p250 = 0 ]; then p250=1; say "at $it: air $air vx $vx steps $steps tilt+div $bad lvl $lvl"; probe 002250 & fi
  [ "$it" -ge 2500 ] && break
  pgrep -f "supervise_node.py --name r6c" > /dev/null || { say "r6c supervisor gone at $it before 2500"; exit 1; }
  sleep 60
done
read it air vx steps bad lvl vxmax <<< $(stats 3)
say "r6c end $it: air $air vx $vx steps $steps tilt+div $bad lvl $lvl"
wait_trainer_gone || { say "trainer still alive at end; nothing started"; exit 1; }
wait; probe 002500
s035=$(python3 -c "import json; print([r['vx'] for r in json.load(open('$E/speed_bins/r6c_002500.json')) if r['cmd']==0.35][0])")
dec=$(python3 -c "
air,vx,s=$air,$vx,$s035
ok_air=(air>=0.15 or air-0.1237>=0.04) and vx>=0.54
ok_slow=s>=0.20 and vx>=0.54
print('continue' if (ok_air or ok_slow) else 'step2')")
say "decision at 2500: $dec (air $air vs 0.1237, vx $vx vs 0.5907, speed at 0.35 cmd $s035 vs r6b_002000 in speed_bins/r6b_002000.json)"
cd $C
if [ "$dec" = continue ]; then
  ../env/venv/bin/python scripts/supervise_node.py --name r6c --iters 20000 --num_envs 8192 -- $R6C
else
  bash run_commands/step2_r7.sh
fi
