#!/bin/bash
# auto_export_v4.sh: every 2 min, export each run-4 checkpoint that has no eval/x2_parkour_v4_<N>/policy.onnx yet (CPU only,
# niced; waits 60 s after the file stops growing). Log: eval/auto_export_v4.log. Never touches the GPU or the trainer.
T=/mnt/filesystem-g4/tracks/hiking; X=$T/code/x2_port; LOG=$T/eval/auto_export_v4.log
while true; do
  for f in $(ls $T/runs/x2_parkour_v4/*/model_*.pt 2>/dev/null); do
    n=$(basename $f .pt); n=${n#model_}; O=$T/eval/x2_parkour_v4_$n
    [ -f $O/policy.onnx ] && continue
    [ $(( $(date +%s) - $(stat -c %Y $f) )) -lt 60 ] && continue
    echo "$(date -u '+%F %T') export $f" >> $LOG
    (cd $X && CUDA_VISIBLE_DEVICES= nice -n 10 timeout 900 ~/hiking-env/bin/python export_for_eval.py $f $O 2>&1 | grep -E "EXPORT DONE|max \||Error|error" | tail -3) >> $LOG
  done
  sleep 120
done
