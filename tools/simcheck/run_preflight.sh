#!/bin/bash
# Run simcheck preflight for a track on its node, in the background-friendly way: CPU only (GPU hidden), 4 cores, nice.
# usage: run_preflight.sh <acp|ssr> <tag> [preflight args...]
# writes /mnt/filesystem-g4/tools/simcheck/reports/<track>_<tag>.json, .log, and a .DONE (exit 0/1) or .FAILED (exit 2) marker.
set -u
T=$1; TAG=$2; shift 2
R=/mnt/filesystem-g4/tools/simcheck/reports; mkdir -p "$R"
case $T in
  acp) PY=$HOME/acp_env/bin/python; CD=/mnt/filesystem-g4/tracks/acp/code ;;
  ssr) PY=/mnt/filesystem-g4/tracks/ssr/env/venv/bin/python; CD=/mnt/filesystem-g4/tracks/ssr/code ;;
  *) echo "track $T: no CPU preflight (hiking: see adapters/hiking.py)"; exit 2 ;;
esac
rm -f "$R/${T}_$TAG".{DONE,FAILED}
cd "$CD"
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=/mnt/filesystem-g4/tools \
  nice -n 15 taskset -c ${SIMCHECK_CORES:-20-23} "$PY" -m simcheck.preflight --track "$T" --out "$R/${T}_$TAG.json" "$@" \
  > "$R/${T}_$TAG.out" 2> "$R/${T}_$TAG.log"
rc=$?
if [ $rc -le 1 ]; then echo $rc > "$R/${T}_$TAG.DONE"; else echo $rc > "$R/${T}_$TAG.FAILED"; fi
exit $rc
