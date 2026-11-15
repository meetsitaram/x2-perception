#!/bin/bash
# After queue13 (e1, e2): e3 = e1 + --default_from_ref (one variable vs e1), 300 its flat from scratch; checks on the CPU;
# then r7 starts automatically (SSR agent decision 2026-11-12 07:10):
#   base = the paper's plain table (e2's flags) if e2 walked at 290 (share >= 0.40, single >= 0.5), else t_a's staged recipe (r6b's flags without feet-air);
#   + exclusion; + --default_from_ref if e3 beat e1 at 290 by >= 0.10 share; + slow commands: walking starts from 0.15 m/s, kernel ramp 0.25 -> 0.1 over 600-900.
R=/mnt/filesystem-g4/tracks/ssr/runs; E=/mnt/filesystem-g4/tracks/ssr/eval; C=/mnt/filesystem-g4/tracks/ssr/code; PY=/mnt/filesystem-g4/tracks/ssr/env/venv/bin/python
say() { echo "QUEUE14 $(date -u +%F' '%T) $*"; }
gone() { for i in $(seq 1 60); do pgrep -f "ssr[.]train" > /dev/null || break; sleep 10; done; sleep 15; ! pgrep -f "ssr[.]train" > /dev/null; }
D4="--rsi_frac 0.5 --init_std 1.0 --rew_scale 0.02 --adv_norm joint --stage_a 0 --stage_b 0 --prior_min_speed 0.2 --rsi_min_speed 0.3 --plant mc_20261024 --flat_only"
hip() { $PY -c "
import numpy as np; z=np.load('$E/export/$1_last/obs_check.npz'); a=z['action_torch']; n=list(z['joint_names_21']); q=z['qpos'][:,7:]; n31=list(z['joint_names_31'])
for s in ('right','left'):
    i=n.index(s+'_hip_roll_joint'); j=n31.index(s+'_hip_roll_joint')
    print(s, 'action mean %.2f at-bound share %.2f q mean %.3f' % (a[:,i].mean(), (abs(a[:,i])>7.5).mean(), q[:,j].mean()), end='; ')
"; }
at290() { python3 -c "
import json
L=[json.loads(l) for l in open('$R/$1/log.jsonl') if l.strip()]
d=[x for x in L if x['it']>=270][-3:]
print(round(sum(x['gait']['vx_ratio'] for x in d)/len(d),3), round(sum(x['gait']['single'] for x in d)/len(d),3))"; }
check() {
  cp $R/$1/last.pt $E/export/ckpts/$1_last.pt; cp $R/$1/args.json $E/export/ckpts/args.json
  (cd $C && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 X2_PLANT=mc_20261024 nice -n 10 $PY scripts/export_eval.py --ckpt ../eval/export/ckpts/$1_last.pt --out ../eval/export/$1_last --camera $2 > ../eval/export/$1_last.log 2>&1)
  say "$1 export: $(tail -1 $E/export/$1_last.log)"; say "$1 hip roll: $(hip $1)"
  (cd $E && X2_EXCL_PELVIS_HIPROLL=1 X2_DEFAULT_FROM_REF=$3 PROBE_CAMERA=$2 bash speed_bins.sh ../eval/export/ckpts/$1_last.pt $1 > speed_bins/$1.run.log 2>&1); say "$1 bins: $(tail -1 $E/speed_bins/$1.run.log)"
}
until grep -q "QUEUE13.*done" $R/queue13.log; do pgrep -f "queue13.sh" > /dev/null || break; sleep 30; done
say "e1 hip roll (rerun with the venv): $(hip e1)"
say "e2 hip roll: $(hip e2)"
gone || { say "trainer alive; nothing started"; exit 1; }
cd $C
say "start e3"
$PY scripts/supervise_node.py --name e3 --iters 300 --num_envs 8192 -- $D4 --w_style 0.0 --camera head --amp_file data/amp.npz --excl_pelvis_hiproll --default_from_ref
read e1v e1s <<< $(at290 e1); read e2v e2s <<< $(at290 e2); read e3v e3s <<< $(at290 e3)
say "at 270-290: e1 share $e1v single $e1s; e2 $e2v / $e2s; e3 $e3v / $e3s"
check e3 head 1 &
STAGED="--flat_until 550 --obst0 0.25 --obst_ramp 500 --stage_a 150 --stage_b 400 --w_style 0.2 --style_low 0 --gate_ramp 500 --style_from 1000 --prior_min_speed 0.2 --prior_balance --disc_mask_arms"
PLAIN="--flat_until 550 --obst0 0.25 --obst_ramp 500 --stage_a 0 --stage_b 0 --w_style 0.2 --prior_min_speed 0.2"
if python3 -c "import sys; sys.exit(0 if ($e2v >= 0.40 and $e2s >= 0.5) else 1)"; then BASE="$PLAIN"; why="e2 walked"; else BASE="$STAGED"; why="e2 did not walk"; fi
EXTRA=""; python3 -c "import sys; sys.exit(0 if $e3v - $e1v >= 0.10 else 1)" && EXTRA="--default_from_ref"
R7="$BASE --plant mc_20261024 --camera chest_rs --amp_file data/amp_v2.npz --excl_pelvis_hiproll --rsi_min_speed 0.15 --sig_v_to 0.1 --sig_v_from 600 --sig_v_ramp 300 $EXTRA"
echo "$R7" > $R/r7_flags.txt
gone || { say "trainer alive; r7 not started"; exit 1; }
say "start r7 ($why; default_from_ref: ${EXTRA:-no}): $R7"
$PY scripts/supervise_node.py --name r7 --iters 20000 --num_envs 8192 -- $R7
say "r7 supervisor ended"
