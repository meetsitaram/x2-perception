#!/bin/bash
# Audit test (a)1 (SSR agent decision 2026-11-12 06:35). r6d stopped (trained on the hull-stop body).
# e1 = d4 exactly (flat, all Table 6 terms, no staged schedule, no air reward, style 0, head camera) + pelvis/hip-roll exclusion: clean A/B vs d4 (stood: 21 % at 290).
# e2 = e1 with the paper's style weight 0.2, chest_rs camera, amp_v2 prior (paper's plain table incl. style), from scratch.
# After each: eval export (obs_check, hip-roll action check) and the per-bin speed probe, all on the CPU.
R=/mnt/filesystem-g4/tracks/ssr/runs; E=/mnt/filesystem-g4/tracks/ssr/eval; C=/mnt/filesystem-g4/tracks/ssr/code
say() { echo "QUEUE13 $(date -u +%F' '%T) $*"; }
gone() { for i in $(seq 1 60); do pgrep -f "ssr[.]train" > /dev/null || break; sleep 10; done; sleep 15; ! pgrep -f "ssr[.]train" > /dev/null; }
D4="--rsi_frac 0.5 --init_std 1.0 --rew_scale 0.02 --adv_norm joint --stage_a 0 --stage_b 0 --prior_min_speed 0.2 --rsi_min_speed 0.3 --plant mc_20261024 --flat_only"
check() {  # $1 run, $2 camera
  cp $R/$1/last.pt $E/export/ckpts/$1_last.pt; cp $R/$1/args.json $E/export/ckpts/args.json
  (cd $C && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 X2_PLANT=mc_20261024 nice -n 10 ../env/venv/bin/python scripts/export_eval.py --ckpt ../eval/export/ckpts/$1_last.pt --out ../eval/export/$1_last --camera $2 > ../eval/export/$1_last.log 2>&1)
  say "$1 export: $(tail -1 $E/export/$1_last.log)"
  say "$1 hip roll: $(python3 -c "
import numpy as np; z=np.load('$E/export/$1_last/obs_check.npz'); a=z['action_torch']; n=list(z['joint_names_21']); q=z['qpos'][:,7:]; n31=list(z['joint_names_31'])
for s in ('right','left'):
    i=n.index(s+'_hip_roll_joint'); j=n31.index(s+'_hip_roll_joint')
    print(s, 'action mean %.2f |a|>7.5 share %.2f q mean %.3f' % (a[:,i].mean(), (abs(a[:,i])>7.5).mean(), q[:,j].mean()), end='; ')
")"
  (cd $E && X2_EXCL_PELVIS_HIPROLL=1 PROBE_CAMERA=$2 bash speed_bins.sh ../eval/export/ckpts/$1_last.pt $1 > speed_bins/$1.run.log 2>&1); say "$1 bins: $(tail -1 $E/speed_bins/$1.run.log)"
}
gone || { say "trainer still alive; nothing started"; exit 1; }
cd $C
say "start e1"
../env/venv/bin/python scripts/supervise_node.py --name e1 --iters 300 --num_envs 8192 -- $D4 --w_style 0.0 --camera head --amp_file data/amp.npz --excl_pelvis_hiproll
say "e1 ended: $(python3 scripts/w.py e1 1 2>&1 | tail -1)"
check e1 head &
gone || { say "trainer alive; e2 not started"; exit 1; }
say "start e2"
../env/venv/bin/python scripts/supervise_node.py --name e2 --iters 300 --num_envs 8192 -- $D4 --w_style 0.2 --camera chest_rs --amp_file data/amp_v2.npz --excl_pelvis_hiproll
say "e2 ended: $(python3 scripts/w.py e2 1 2>&1 | tail -1)"
wait
check e2 chest_rs
say "done; GPU idle until the SSR agent decides r7"
