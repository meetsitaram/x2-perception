#!/usr/bin/env bash
# Depth -> height-scan student, DAgger rounds, unattended (user, 2026-11-16: "both will need a student dagger").
#   v2 = supervised on data_v2 (full teacher re-collection); d1, d2 = DAgger rounds: ACP walks the training tiles on the
#   previous student's scans (ACPD_MODE=dagger), the true scans are recorded, the student retrains on everything so far
#   (warm start). Each student is scored on our apparatus, the same 50 seeds as the perfect-map baseline (46/50 here),
#   in 3 parallel processes. Log: models/acp_depth/runs/run_dagger.log; a line per stage, DONE at the end.
set -uo pipefail
cd "$(dirname "$0")/../.."
R=models/acp_depth/runs
PY_EVAL="$HOME/Projects/gwbc-x2/.venv/bin/python"
PY_GPU="$HOME/miniconda3/envs/g1_train/bin/python"
ACP=$(cd ../../../release/ckpt/acp 2>/dev/null && pwd || echo "$HOME/projects/x2-perception/release/ckpt/acp")
export EVAL_REGRESSION=0 ACP_HANDOVER_DIR=$ACP ACP_GOAL_R=0.2 ACP_LOOKAHEAD=0.3 X2_EXCLUDE_PAIRS=hip \
       PA_ROBOT_DIR=$HOME/Projects/gwbc-x2/gear_sonic/data/assets/robot_description
log() { echo "$(date +%T) $*"; }

train() {   # out init data...
  local out=$1 init=$2; shift 2
  mkdir -p $out
  systemd-run --user --scope -q -p MemoryMax=28G "$PY_GPU" models/acp_depth/train.py --data "$@" --out $out --epochs 40 \
    ${init:+--init $init} > $out.train.log 2>&1 || { log "TRAIN FAILED $out"; tail -3 $out.train.log; exit 1; }
  log "trained $out: $(tail -1 $out.train.log | cut -c1-150)"
  systemd-run --user --scope -q -p MemoryMax=16G "$PY_GPU" models/acp_depth/region_error.py --data "$@" --run $out > $out/region_error.txt 2>&1
}

score() {   # student_dir
  local s=$1
  for part in "2000 17" "2017 17" "2034 16"; do set -- $part
    ( ulimit -v 12000000; ACPD_MODE=student ACPD_STUDENT=$s/student.pt timeout 10800 "$PY_EVAL" eval/harness.py \
      --adapter acp_depth:ACPDepth --apparatus ours --trials $2 --seed0 $1 --out $s/apparatus_$1 > $s/apparatus_$1.log 2>&1 ) &
  done
  wait
  "$PY_EVAL" - "$s" <<'EOF'
import json, sys, glob
r = [t for f in sorted(glob.glob(sys.argv[1] + "/apparatus_*/results.json")) for t in json.load(open(f))]
ok = sum(bool(t["success"]) for t in r)
out = {}
for t in r:
    out[t["reason"]] = out.get(t["reason"], 0) + 1
line = "apparatus with the student: %d / %d (%.0f %%), outcomes %s, mean time of successes %.1f s (baseline 46/50, ~55 s)" % (
    ok, len(r), 100 * ok / max(len(r), 1), out, sum(t["sim_time"] for t in r if t["success"]) / max(ok, 1))
open(sys.argv[1] + "/apparatus_summary.txt", "w").write(line + "\n")
print(line)
EOF
}

collect_dagger() {   # out student seed
  ACPD_MODE=dagger ACPD_STUDENT=$(pwd)/$2/student.pt "$PY_EVAL" models/acp_depth/collect.py --out $1 --levels 0-7 \
    --per-type 35 --seeds 2 --workers 16 --seed $3 > $1.log 2>&1
  log "dagger data $1: $(grep '^DONE' $1.log)"
}

until grep -q "^DONE" $R/data_v2.log 2>/dev/null; do sleep 30; done
log "data_v2: $(grep '^DONE' $R/data_v2.log)"
train $R/student_v2 "" $R/data_v2/data $R/data_v1/data
log "v2 $(score $R/student_v2)"
collect_dagger $R/data_d1 $R/student_v2 11
train $R/student_d1 $R/student_v2/best_state.pt $R/data_v2/data $R/data_v1/data $R/data_d1/data
log "d1 $(score $R/student_d1)"
collect_dagger $R/data_d2 $R/student_d1 12
train $R/student_d2 $R/student_d1/best_state.pt $R/data_v2/data $R/data_v1/data $R/data_d1/data $R/data_d2/data
log "d2 $(score $R/student_d2)"
echo DONE
