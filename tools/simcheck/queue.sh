#!/bin/bash
# Preflight queue for this node, one job at a time (4 cores). usage: queue.sh acp|ssr
P=/mnt/filesystem-g4/tools/simcheck/run_preflight.sh
case $1 in
  acp)
    $P acp s2g_fix0 --hf-fix 0
    $P acp s2g_fix1 --hf-fix 1
    $P acp s3v2b_fix1 --hf-fix 1 --cfg cfg/stage3v2b_sets.json
    ACP_HF_CHUNK=100000 $P acp s3v2b_fix1_onefield --hf-fix 1 --cfg cfg/stage3v2b_sets.json ;;
  ssr)
    $P ssr r7
    $P ssr r7_onefield --hf-chunk 0 ;;
esac
touch /mnt/filesystem-g4/tools/simcheck/reports/queue_$1.DONE
