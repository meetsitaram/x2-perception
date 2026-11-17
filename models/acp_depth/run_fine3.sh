#!/usr/bin/env bash
# score fine_v2 (clean + noisy, seeds 2000-2049) while fine_v3 trains on data_v3 + v3d + v4d (from fine_v2); then score fine_v3,
# its 4x4 height and the standard showcase renders. Log: runs/run_fine3.log
set -uo pipefail
cd "$(dirname "$0")/../.."
R=models/acp_depth/runs
PY_EVAL="$HOME/Projects/gwbc-x2/.venv/bin/python"; PY_GPU="$HOME/miniconda3/envs/g1_train/bin/python"
ACP=$(cd ../../../release/ckpt/acp 2>/dev/null && pwd || echo "$HOME/projects/x2-perception/release/ckpt/acp")
export EVAL_REGRESSION=0 ACP_HANDOVER_DIR=$ACP ACP_GOAL_R=0.2 ACP_LOOKAHEAD=0.3 X2_EXCLUDE_PAIRS=hip \
       PA_ROBOT_DIR=$HOME/Projects/gwbc-x2/gear_sonic/data/assets/robot_description
log() { echo "$(date +%T) $*"; }
score() {   # student_dir
  local S=$1
  for noise in 0 1; do for part in "2000 17" "2017 17" "2034 16"; do set -- $part
    tag=$([ $noise = 1 ] && echo noisy || echo clean)
    ( ulimit -v 12000000; ACPD_MODE=student ACPD_DEPTH_NOISE=$noise ACPD_STUDENT=$(pwd)/$S/student.pt timeout 14400 "$PY_EVAL" eval/harness.py \
      --adapter acp_depth:ACPDepth --apparatus ours --trials $2 --seed0 $1 --out $S/${tag}_$1 > $S/${tag}_$1.log 2>&1 ) &
  done; done
  wait
  for tag in clean noisy; do
    "$PY_EVAL" -c "
import json, glob
r = [t for f in sorted(glob.glob('$S/${tag}_*/results.json')) for t in json.load(open(f))]
print('$(basename $S) $tag: %d / %d' % (sum(bool(t['success']) for t in r), len(r)))"
  done
}
( score $R/fine_v2 | while read l; do log "$l"; done ) &
mkdir -p $R/fine_v3
systemd-run --user --scope -q -p MemoryMax=28G "$PY_GPU" models/acp_depth/train_fine.py --data $R/data_v3/data $R/data_v3d/data $R/data_v4d/data \
  --out $R/fine_v3 --epochs 150 --init $R/fine_v2/best_state.pt > $R/fine_v3.train.log 2>&1 || log "TRAIN FAILED"
log "fine_v3 trained: $(tail -1 $R/fine_v3.train.log | cut -c1-180)"
wait
score $R/fine_v3 | while read l; do log "$l"; done
models/acp_depth/render_showcase.sh $R/fine_v3/student.pt $R/vis_fine3 "fine-map student v3" > $R/vis_fine3.log 2>&1
( ulimit -v 6000000; "$PY_EVAL" - <<'PYEOF'
import numpy as np, sys
sys.path.insert(0, "eval")
from ladder import load_course
app = load_course("ladder:acp_log_L2")
e = np.load("models/acp_depth/runs/vis_fine3/extra_acp_log_L2/ep0001.npz")
for name, XY, Hs in (("planner", e["plan_xy"], e["plan_h"]), ("tracker", e["scan_xy"], e["scan_h"])):
    est = np.array(Hs); tru = np.array([app.height(xy[:, 0], xy[:, 1]) for xy in XY])
    floor = np.median(tru, 1, keepdims=True); on = (tru - floor) > 0.05
    print("fine_v3 4x4 %s: true %.3f m, estimated %.3f m (median on the beam cells)" % (name, np.median((tru - floor)[on]), np.median((est - floor)[on])))
PYEOF
) | while read l; do log "$l"; done
log "renders: $(ls $R/vis_fine3/*_small.mp4 2>/dev/null | wc -l) clips in $R/vis_fine3"
echo DONE
