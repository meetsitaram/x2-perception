#!/usr/bin/env bash
# Fine-map student, unattended: wait for data_v3 (teacher) + data_v3d (DAgger on d1 with D435 noise in the loop), train
# (fine_student.py / train_fine.py), score on our apparatus (seeds 2000-2049, clean and with noise; d1: 46/50 clean, 35/50
# noisy), render the standard showcase, measure the 4x4 (true 8.9 cm; d1 / e1 estimated 4-6 cm). Log: runs/run_fine.log.
set -uo pipefail
cd "$(dirname "$0")/../.."
R=models/acp_depth/runs; S=$R/fine_v1
PY_EVAL="$HOME/Projects/gwbc-x2/.venv/bin/python"; PY_GPU="$HOME/miniconda3/envs/g1_train/bin/python"
ACP=$(cd ../../../release/ckpt/acp 2>/dev/null && pwd || echo "$HOME/projects/x2-perception/release/ckpt/acp")
export EVAL_REGRESSION=0 ACP_HANDOVER_DIR=$ACP ACP_GOAL_R=0.2 ACP_LOOKAHEAD=0.3 X2_EXCLUDE_PAIRS=hip \
       PA_ROBOT_DIR=$HOME/Projects/gwbc-x2/gear_sonic/data/assets/robot_description
log() { echo "$(date +%T) $*"; }
until grep -q "^DONE" $R/data_v3.log 2>/dev/null && grep -q "^DONE" $R/data_v3d.log 2>/dev/null; do sleep 30; done
log "data: v3 $(grep '^DONE' $R/data_v3.log) | v3d $(grep '^DONE' $R/data_v3d.log)"
mkdir -p $S
systemd-run --user --scope -q -p MemoryMax=28G "$PY_GPU" models/acp_depth/train_fine.py --data $R/data_v3/data $R/data_v3d/data \
  --out $S --epochs 30 > $S.train.log 2>&1 || { log "TRAIN FAILED"; tail -5 $S.train.log; exit 1; }
log "trained: $(tail -1 $S.train.log | cut -c1-200)"
score() {   # tag extra_env
  local tag=$1; shift
  for part in "2000 17" "2017 17" "2034 16"; do set -- $part
    ( ulimit -v 12000000; env ACPD_MODE=student ACPD_STUDENT=$(pwd)/$S/student.pt "${EXTRA[@]}" timeout 14400 "$PY_EVAL" eval/harness.py \
      --adapter acp_depth:ACPDepth --apparatus ours --trials $2 --seed0 $1 --out $S/${tag}_$1 > $S/${tag}_$1.log 2>&1 ) &
  done
}
EXTRA=(); score clean
EXTRA=(ACPD_DEPTH_NOISE=1); score noisy
wait
for tag in clean noisy; do
  "$PY_EVAL" - "$S" "$tag" <<'EOF'
import json, sys, glob
r = [t for f in sorted(glob.glob("%s/%s_*/results.json" % (sys.argv[1], sys.argv[2]))) for t in json.load(open(f))]
ok = sum(bool(t["success"]) for t in r)
print("fine_v1 %s: %d / %d" % (sys.argv[2], ok, len(r)))
EOF
done | while read l; do log "$l"; done
models/acp_depth/render_showcase.sh $S/student.pt $R/vis_fine1 "fine-map student" > $R/vis_fine1.log 2>&1
log "renders: $(ls $R/vis_fine1/*.mp4 2>/dev/null | wc -l) videos in $R/vis_fine1"
( ulimit -v 6000000; "$PY_EVAL" - <<'EOF'
import numpy as np, sys
sys.path.insert(0, "eval")
from ladder import load_course
app = load_course("ladder:acp_log_L2")
e = np.load("models/acp_depth/runs/vis_fine1/extra_acp_log_L2/ep0001.npz")
for name, XY, Hs in (("planner", e["plan_xy"], e["plan_h"]), ("tracker", e["scan_xy"], e["scan_h"])):
    est = np.array(Hs); tru = np.array([app.height(xy[:, 0], xy[:, 1]) for xy in XY])
    floor = np.median(tru, 1, keepdims=True); on = (tru - floor) > 0.05
    print("4x4 %s: true %.3f m, fine student %.3f m (median on the beam cells), error on beam %.3f, elsewhere %.3f"
          % (name, np.median((tru - floor)[on]), np.median((est - floor)[on]), np.mean(np.abs(est - tru)[on]), np.mean(np.abs(est - tru)[~on])))
EOF
) | while read l; do log "$l"; done
echo DONE
