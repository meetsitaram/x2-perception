#!/usr/bin/env bash
# Export a pair for every numbered checkpoint of the ACP runs as it appears (CPU, cores 8-23, nice 19). Every attempt's
# result goes to /mnt/filesystem-g4/status/acp_export.json (read by the coordinator's refresh), and a failed hand-over also
# leaves eval/pair_<run>_<it>/EXPORT_FAILED with the reason (2026-11-12, after the 2g set-name failure went unseen for 25 min).
#   tmux new -d -s acp-export "bash /mnt/filesystem-g4/tracks/acp/code/ops/export_every.sh"
T=/mnt/filesystem-g4/tracks/acp; L=$T/eval/export_every.log; ST=/mnt/filesystem-g4/status/acp_export.json; FROM=${FROM:-3500}
status() { printf '{"track":"acp","what":"pair export","state":"%s","pair":"%s","detail":"%s","time":"%s","unix":%d}\n' "$1" "$2" "$3" "$(date -u +%FT%TZ)" "$(date +%s)" > "$ST"; }
while true; do
  for RUN in $T/runs/s2_v1 $T/runs/s3_v1 $T/runs/s1_v2 $T/runs/s2_v2 $T/runs/s2_v3; do
    case $(basename $RUN) in s1_*) export PLANNER_FROM=${S1_PLANNER:-$T/runs/s2_v1/ckpt_004000.pt};; *) unset PLANNER_FROM;; esac
    for CK in $(ls $RUN/ckpt_*.pt 2>/dev/null | sort); do
      IT=$((10#$(basename $CK .pt | cut -d_ -f2))); [ $IT -lt $FROM ] && continue
      D=$T/eval/pair_$(basename $RUN)_$(printf %06d $IT); [ -f $D/DONE ] && continue; [ -f $D/EXPORT_FAILED ] && continue
      [ $(( $(date +%s) - $(stat -c %Y $CK) )) -lt 60 ] && continue
      echo "$(date -u +%FT%TZ) start $RUN $IT" >> $L; status running "$(basename $D)" ""
      timeout 3600 taskset -c 8-23 nice -n 19 bash $T/code/ops/export_pair.sh $RUN $IT >> $L 2>&1; rc=$?
      echo "$(date -u +%FT%TZ) rc $rc $RUN $IT" >> $L
      if [ $rc -eq 0 ] && [ -f $D/DONE ]; then status done "$(basename $D)" "";
      else
        why=$(grep -h "Error\|failed" $D/handover.out $D/export_actor.out $L 2>/dev/null | tail -1 | tr -d '"\' | cut -c1-200)
        mkdir -p $D; echo "$(date -u +%FT%TZ) rc $rc: $why" > $D/EXPORT_FAILED; status FAILED "$(basename $D)" "rc $rc: $why"
      fi
    done
  done
  sleep 120
done
