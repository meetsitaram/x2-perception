#!/bin/bash
# Recipe tests for getting off the stand (coordinator, 2026-11-11), one GPU job at a time, each followed by a gait probe of its last checkpoint.
# Start only when nothing else trains. Stop between jobs: touch runs/QUEUE4_STOP (and runs/STOP for the running one).
cd /mnt/filesystem-g4/tracks/ssr/code
R=/mnt/filesystem-g4/tracks/ssr/runs
PY=../env/venv/bin/python
COMMON="--flat_only --w_style 0 --prior_min_speed 0.2 --rsi_min_speed 0.3 --plant mc_20261024 --camera chest_rs --amp_file data/amp_v2.npz"
run() {  # name iters flags...
  name=$1; iters=$2; shift 2
  [ -f $R/QUEUE4_STOP ] && exit 0
  $PY scripts/supervise_node.py --name $name --iters $iters --num_envs 8192 -- $COMMON "$@"
  for f in "" "--noise"; do X2_PLANT=mc_20261024 timeout 300 $PY scripts/probe_gait.py --ckpt $R/$name/last.pt --out ../eval/$name --speed 0.6 $f > $R/${name}_probe.log 2>&1; done
}
run t_b 500 --cmd_cur_iters 200 --cmd_min0 0.5 --sig_v0 0.06
run t_a 550 --stage_a 150 --stage_b 400
run t_c 450 --stuck_term --stuck_until 300
run t_d 300 --ent 0.001 --init_std 0.5
