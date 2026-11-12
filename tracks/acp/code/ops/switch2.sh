#!/usr/bin/env bash
# At checkpoint IT: pause stage 1 (heartbeat says why), install the code staged in code_next (smoke-tested, rolled back on
# failure), run the paper eval on that checkpoint under EVAL_PLANT, resume training under RUN_PLANT, and (PLANNER=1) start
# the planner relabel + training + hand-over + planner-driven check on the CPU.
#   tmux new -d -s acp-switch "bash /mnt/filesystem-g4/tracks/acp/code_next/switch2.sh IT EVAL_PLANT RUN_PLANT PLANNER 'reason'"
#   plant "default" = the active plant file (vendor_20260823)
set -u
IT=$(printf %06d "$1"); EVAL_PLANT=$2; RUN_PLANT=$3; PLANNER=$4; WHY=$5
T=/mnt/filesystem-g4/tracks/acp
R=$T/runs/s1_v1
PY=$HOME/acp_env/bin/python
L=$R/switch.log
HB=/mnt/filesystem-g4/status/acp.json
plant_env() { [ "$1" = default ] && echo "" || echo "$1"; }
say() { echo "$(date -u +%FT%TZ) $*" >> "$L"; }
gpu_free() { ! nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; }
say "waiting for ckpt_$IT.pt (eval plant $EVAL_PLANT, then train with plant $RUN_PLANT)"
while [ ! -f "$R/ckpt_$IT.pt" ]; do sleep 5; done
sleep 3
touch "$R/STOP"
for i in $(seq 60); do gpu_free && ! tmux has-session -t acp 2>/dev/null && break; sleep 2; done
say "trainer stopped; GPU free: $(gpu_free && echo yes || echo no)"
printf '{"track":"acp","state":"paused at checkpoint %s (under 10 minutes): %s","run":"%s","newest_checkpoint":"%s","unix":%d,"time":"%s"}\n' "$IT" "$WHY" "$R" "$R/ckpt_$IT.pt" "$(date +%s)" "$(date +%FT%T%z)" > "$HB"
if [ -d "$T/code_next/acp" ] && ! diff -q "$T/code_next/acp/env.py" "$T/code/acp/env.py" >/dev/null 2>&1 || ! diff -q "$T/code_next/acp/data.py" "$T/code/acp/data.py" >/dev/null 2>&1; then
  rm -rf "$T/code_prev"; mkdir -p "$T/code_prev"; cp -r "$T/code/acp" "$T/code/tools" "$T/code/supervise.sh" "$T/code_prev/"
  cp "$T"/code_next/acp/*.py "$T/code/acp/"; cp "$T"/code_next/tools/*.py "$T/code/tools/"; cp "$T/code_next/supervise.sh" "$T/code/supervise.sh"
  cd "$T/code"; rm -rf /tmp/smoke_new
  if ACP_PLANT=$(plant_env "$RUN_PLANT") timeout 240 "$PY" -m acp.train --data cfg/stage1_sets.json --run /tmp/smoke_new --envs 512 --iters 2 --njmax 450 --heartbeat /tmp/hb.json > /tmp/smoke_new.out 2>&1 \
     && [ "$(wc -l < /tmp/smoke_new/metrics.jsonl)" -ge 2 ]; then
    say "new code installed; smoke test passed (plant $RUN_PLANT)"
  else
    say "smoke test FAILED: restoring the previous code (see /tmp/smoke_new.out, mirrored to backup/tmp_outputs)"
    cp "$T"/code_prev/acp/*.py "$T/code/acp/"; cp "$T"/code_prev/tools/*.py "$T/code/tools/"; cp "$T/code_prev/supervise.sh" "$T/code/supervise.sh"
  fi
fi
cd "$T/code"
mkdir -p "$T/eval/paper_eval"
TAG=s1_v1_${IT}_plant_$EVAL_PLANT
if ACP_PLANT=$(plant_env "$EVAL_PLANT") timeout 420 "$PY" tools/paper_eval.py "$R/ckpt_$IT.pt" "$T/eval/paper_eval/$TAG.json" > "$T/eval/paper_eval/$TAG.out" 2>&1; then
  say "paper eval done: $TAG"
else
  say "paper eval FAILED or timed out (rc $?), see eval/paper_eval/$TAG.out"
fi
for i in $(seq 30); do gpu_free && break; sleep 2; done
rm -f "$R/STOP"
echo "--- $(date -u +%FT%T) restart after the pause at checkpoint $IT, plant $RUN_PLANT" >> "$R/train.out"
tmux new -d -s acp "ACP_PLANT=$(plant_env "$RUN_PLANT") bash $T/code/supervise.sh $R 36000 12000"
say "supervisor restarted with plant $RUN_PLANT"
# CPU: read the gains back from the live model under the training plant
(export CUDA_VISIBLE_DEVICES=; ACP_PLANT=$(plant_env "$RUN_PLANT") timeout 300 "$PY" tools/plant_report.py "$T/eval/paper_eval/${TAG}_rollouts.pt" cfg/stage1_sets.json \
   "$T/eval/plant_report_${IT}_live_$RUN_PLANT.json" > "$T/eval/plant_report_${IT}_live_$RUN_PLANT.txt" 2>&1; say "plant read-back written (live plant $RUN_PLANT)") &
if [ "$PLANNER" = 1 ]; then
  P=$T/runs/planner_s1_${IT}_$EVAL_PLANT; O=$T/eval/handover_pair_s1_${IT}_${EVAL_PLANT}_planner; mkdir -p "$P" "$O"
  cat > "$P/run.sh" <<EOF
#!/usr/bin/env bash
cd $T/code; export CUDA_VISIBLE_DEVICES=; export ACP_PLANT=$(plant_env "$EVAL_PLANT"); PY=\$HOME/acp_env/bin/python
\$PY tools/weak_sets.py $T/eval/paper_eval/${TAG}_rollouts.pt cfg/stage1_sets.json 2>&1 | grep -v "^Warp\|^Module\|^   [A-Z\"/]\|^\$" > $T/eval/paper_eval/${TAG}_weak_sets.txt
\$PY -m acp.planner label $T/eval/paper_eval/${TAG}_rollouts.pt $P/data.pt
\$PY -m acp.planner train $P/data.pt $P/planner_mlp.pt --arch mlp --epochs 30
\$PY -m acp.planner export $P/planner_mlp.pt $P/planner_mlp.onnx
cp $R/ckpt_$IT.pt $O/tracker_ckpt_$IT.pt
\$PY tools/handover.py $O/tracker_ckpt_$IT.pt $O
\$PY /mnt/filesystem-g4/eval/code/tools/export_actor.py $O/tracker_ckpt_$IT.pt $O/actor.npz --meta $O/meta.json --goal-r 0.3
cp $P/planner_mlp.onnx $P/planner_mlp.pt $O/
\$PY tools/card_planner.py $O $O/planner_mlp.pt $O/planner_mlp.onnx
timeout 2400 \$PY tools/pair_eval.py $O/tracker_ckpt_$IT.pt $P/planner_mlp.pt $O/mlp 12
echo ALL_DONE
EOF
  tmux kill-session -t acp-planner2 2>/dev/null
  tmux new -d -s acp-planner2 "bash $P/run.sh > $P/run.out 2>&1"
  say "planner job started on the CPU: $P"
fi
wait
