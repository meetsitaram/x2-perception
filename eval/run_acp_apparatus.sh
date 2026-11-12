#!/bin/bash
# An ACP hand-over pair on an apparatus under the current rule, MuJoCo CPU: one core, low priority, memory-capped (harness),
# in slices of at most 9 minutes that continue where they stopped. Run again (or through a loop) until it prints "DONE".
#   bash eval/run_acp_apparatus.sh HANDOVER_DIR RUN_NAME ours|nist nominal|randomised [trials]
# Eval variants (adapter env, off by default): ACP_GOAL_R=0.2, ACP_LOOKAHEAD=0.3 (m past non-station waypoints); set RUN_SUFFIX with them.
# The goal is the adapter's own: direction to the waypoint, d / max(|d|, goal_r), so it shrinks inside goal_r (0.3 m, stage 1).
cd "$(dirname "$0")/.."
export ACP_HANDOVER_DIR=$1; NAME=$2; APP=$3; PROTO=$4; N=${5:-200}
OUT=eval/out/acp/$NAME/${APP}_${PROTO}${RUN_SUFFIX:-}      # RUN_SUFFIX: a variant gets its own folder
mkdir -p eval/out/acp/$NAME
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
timeout 590 python eval/harness.py --adapter acp:ACPHandover --apparatus $APP --trials $N --protocol $PROTO --out $OUT \
  --wall-limit 520 --label "${RUN_LABEL:-ACP hand-over pair under the current rule}" >> $OUT.log 2>&1
rc=$?
tail -1 $OUT.log | cut -c1-300
[ $rc -eq 0 ] && echo DONE || echo "slice ended (code $rc)"
