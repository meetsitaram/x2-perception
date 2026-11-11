#!/bin/bash
# The 2026-11-02 baseline policy pair under the current rule, MuJoCo CPU: one core, low priority, memory-capped (harness),
# in slices of at most 9 minutes so no command runs unbounded. Run again until it prints "DONE".
#   bash eval/run_baseline.sh ours|nist nominal|randomised [trials]
cd "$(dirname "$0")/.."
APP=$1; PROTO=$2; N=${3:-200}
OUT=eval/out/baseline_20261102/${APP}_${PROTO}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
timeout 590 python eval/harness.py --adapter acp_baseline:ACPBaseline --apparatus $APP --trials $N --protocol $PROTO --out $OUT \
  --wall-limit 520 --label "2 November policy pair under the current rule, MuJoCo CPU" >> $OUT.log 2>&1
rc=$?
tail -1 $OUT.log | cut -c1-300
[ $rc -eq 0 ] && echo DONE || echo "slice ended (code $rc)"
