#!/usr/bin/env bash
# CPU tests of the stage-3 build before it goes into code/ (ACP agent, 2026-11-13, state/acp.md 17k). CUDA hidden; never the GPU.
#   usage: s3_build_test.sh <candidate code dir>     (a copy whose acp/ holds the new files; cfg/tools linked to code/)
#   1 stage-2 regression: the same 3-iteration stage-2 smoke (s2_v5 flags, seed 0) with the live code/ and the candidate; every
#     metric except timings must be identical -> the running s2_v5 is unchanged by the install
#   2 stage-3 smoke with the candidate from s2_v5's init checkpoint with code/ops/stage3_flags.env (the auto switch's own check)
# Out: runs/s3_build/{regression.txt, s3_smoke/, s3_smoke_check.txt, DONE}
set -u
C=$1; T=/mnt/filesystem-g4/tracks/acp; O=$T/runs/s3_build; PY=$HOME/acp_env/bin/python; INIT=$T/runs/s2_v5/init_from_s2_v4_latest.pt
mkdir -p "$O"; rm -f "$O/DONE"
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 ACP_PLANT=mc_20261024 X2_EXCLUDE_PAIRS=hip ACP_HF_FIX=1 ACP_PAPER_DR=1
IT0=$("$PY" -c "import torch,sys; print(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['it'])" "$INIT" | tail -1)
S2="--stage 2 --init $INIT --planner-arch attn --u-obs 200 --u-rew 1700 --n-ann 6000 --eta-anchor-it 12604 --eta-obs0 0.781 --eta-rew0 0.281 --replay-balance set_clip --popt-reset --probe-envs 8 --gate-signal probe --ramp-gate 0"
for v in live cand; do
  [ $v = live ] && D=$T/code || D=$C
  rm -rf "$O/reg_$v"; mkdir -p "$O/reg_$v"
  ( cd "$D" && ACP_SMOKE_PREFILL=2000 PYTHONUNBUFFERED=1 timeout 3000 nice -n 15 taskset -c 12-19 "$PY" -m acp.train --data cfg/stage2h_sets.json \
      --run "$O/reg_$v" --envs 64 --iters $((IT0 + 3)) --njmax 450 --device cpu --heartbeat "$O/hb_$v.json" --rec-envs 64 --replay 100000 --seed 0 $S2 \
      > "$O/reg_$v.out" 2>&1 ); echo "regression $v rc=$?" > "$O/reg_$v.rc"
done
"$PY" - "$O/reg_live/metrics.jsonl" "$O/reg_cand/metrics.jsonl" > "$O/regression.txt" 2>&1 <<'EOF'
import json, sys
a = [json.loads(l) for l in open(sys.argv[1])]; b = [json.loads(l) for l in open(sys.argv[2])]
skip = ("time", "iter_s", "env_steps_per_s", "gpu_util_pct", "gpu_mem_gb")
diff = []
for x, y in zip(a, b):
    for k in set(x) | set(y):
        if k in skip: continue
        if x.get(k) != y.get(k): diff.append((x["it"], k, x.get(k), y.get(k)))
print("rows", len(a), len(b), "keys per row", len(a[0]) if a else 0, "differences", len(diff))
for d in diff[:20]: print("  ", d)
print("REGRESSION", "PASS" if len(a) == len(b) == 3 and not diff else "FAIL")
EOF
# 2 stage-3 smoke (same command as auto_stage3.sh's check)
. "$T/code/ops/stage3_flags.env"
EX=$(echo "$S3_EX" | sed "s/@ANCHOR@/$IT0/; s/--probe-envs [0-9]*/--probe-envs 8/")
rm -rf "$O/s3_smoke"; mkdir -p "$O/s3_smoke"
( cd "$C" && env $S3_ENVV ACP_SMOKE_PREFILL=2000 PYTHONUNBUFFERED=1 timeout 3600 nice -n 15 taskset -c 12-19 "$PY" -m acp.train --data "$S3_DATA" \
    --run "$O/s3_smoke" --envs 64 --iters $((IT0 + 3)) --njmax 450 --device cpu --heartbeat "$O/hb_s3.json" --rec-envs 64 --replay 100000 \
    --init "$INIT" $EX > "$O/s3_smoke.out" 2>&1 ); echo "s3 smoke rc=$?" > "$O/s3_smoke.rc"
python3 "$T/code/tools/s3_smoke_check.py" "$O/s3_smoke/metrics.jsonl" > "$O/s3_smoke_check.txt" 2>&1
date -u +%FT%TZ > "$O/DONE"
