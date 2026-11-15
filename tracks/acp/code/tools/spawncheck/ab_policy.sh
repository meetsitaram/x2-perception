#!/bin/bash
D=/mnt/filesystem-g4/tracks/acp/runs/s3_v2_diag; C=/mnt/filesystem-g4/tracks/acp/code; PY=$HOME/acp_env/bin/python; cd $C; rm -f $D/AB_DONE
for v in h i; do
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 X2_EXCLUDE_PAIRS=hip ACP_PLANT=mc_20261024 ACP_HF_FIX=1 ACP_PAPER_DR=1 nice -n 15 taskset -c 12-19 \
    $PY $D/spawn_dyn.py cfg/stage3v2${v}_sets.json $D/dynpol_v2$v.json --repeats 4 --envs 256 --steps 30 --policy $D/init_pf.pt \
    --cache cfg/labels_stage3v2${v}_sets.json.pt > $D/dynpol_v2$v.out 2>&1
  echo "v2$v rc $?" >> $D/ab.log
done
touch $D/AB_DONE
