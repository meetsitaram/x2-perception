#!/usr/bin/env bash
# Re-run the planner closed loop (obs_check_pair.npz) for s2_v4 pairs after the pair_eval zero-weight fix (ACP agent, 2026-11-13, 17g).
T=/mnt/filesystem-g4/tracks/acp; PY=$HOME/acp_env/bin/python; L=$T/eval/fix_pairs_s2_v4.log
export CUDA_VISIBLE_DEVICES= ACP_PLANT=mc_20261024 X2_EXCLUDE_PAIRS=hip
cd $T/code
for P in 011000 010750 010500; do
  D=$T/eval/pair_s2_v4_$P
  [ -f "$D/pair_eval_40/obs_check_pair.npz" ] && continue
  timeout 1500 nice -n 19 "$PY" tools/pair_eval.py "$D/tracker_ckpt_$P.pt" "$D/planner_attn.pt" "$D/pair_eval_4" 4 > "$D/pair_eval_4.out" 2>&1 \
    && mkdir -p "$D/pair_eval_40" && cp "$D/pair_eval_4/obs_check_pair.npz" "$D/pair_eval_40/obs_check_pair.npz" \
    && echo "$(date -u +%FT%TZ) $P ok" >> $L || echo "$(date -u +%FT%TZ) $P FAILED $(tail -1 $D/pair_eval_4.out | cut -c1-200)" >> $L
done
echo "$(date -u +%FT%TZ) DONE" >> $L
