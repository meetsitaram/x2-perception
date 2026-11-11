#!/bin/bash
# One group of ladder tests for one adapter, bounded to 9.5 minutes: bash eval/run_ladder.sh OUT STEP TRIALS ADAPTER [CHECKPOINT|-] test1 test2 ...
cd "$(dirname "$0")/.."
OUT=$1; STEP=$2; N=$3; AD=$4; CK=$5; shift 5
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p $OUT
CKARG=""; [ "$CK" != "-" ] && CKARG="--checkpoint $CK"
timeout 570 python eval/ladder.py --adapter $AD $CKARG --step $STEP --trials $N --out $OUT --tests "$@" --label "${LADDER_LABEL:-}" >> $OUT/step$STEP.log 2>&1
echo "group done (code $?)"; tail -$# $OUT/step$STEP.log | cut -c1-200
