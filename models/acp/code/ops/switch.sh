#!/usr/bin/env bash
# At checkpoint IT: pause stage 1 (heartbeat says so), install the code staged in code_next (smoke-tested first, rolled
# back on failure), run the paper eval on that checkpoint (also saves rollouts for the planner), resume training, then
# start the planner relabel + training + planner-driven check on the CPU.
#   tmux new -d -s acp-switch "bash /mnt/filesystem-g4/tracks/acp/code_next/switch.sh 2000"
set -u
IT=$(printf %06d "$1")
T=/mnt/filesystem-g4/tracks/acp
R=$T/runs/s1_v1
PY=$HOME/acp_env/bin/python
L=$R/switch.log
HB=/mnt/filesystem-g4/status/acp.json
say() { echo "$(date -u +%FT%TZ) $*" >> "$L"; }
gpu_free() { ! nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; }
say "waiting for ckpt_$IT.pt"
while [ ! -f "$R/ckpt_$IT.pt" ]; do sleep 5; done
sleep 3
touch "$R/STOP"
for i in $(seq 60); do gpu_free && ! tmux has-session -t acp 2>/dev/null && break; sleep 2; done
say "trainer stopped; GPU free: $(gpu_free && echo yes || echo no)"
printf '{"track":"acp","state":"paused at checkpoint %s for the paper eval and a code update (under 10 minutes)","run":"%s","newest_checkpoint":"%s","unix":%d,"time":"%s"}\n' "$IT" "$R" "$R/ckpt_$IT.pt" "$(date +%s)" "$(date +%FT%T%z)" > "$HB"
rm -rf "$T/code_prev"; mkdir -p "$T/code_prev"; cp -r "$T/code/acp" "$T/code/tools" "$T/code/supervise.sh" "$T/code_prev/"
cp "$T"/code_next/acp/*.py "$T/code/acp/"; cp "$T"/code_next/tools/*.py "$T/code/tools/"; cp "$T/code_next/supervise.sh" "$T/code/supervise.sh"
cd "$T/code"
rm -rf /tmp/smoke_new
if timeout 240 "$PY" -m acp.train --data cfg/stage1_sets.json --run /tmp/smoke_new --envs 512 --iters 2 --njmax 450 --heartbeat /tmp/hb.json > /tmp/smoke_new.out 2>&1 \
   && [ "$(wc -l < /tmp/smoke_new/metrics.jsonl)" -ge 2 ]; then
  say "smoke test of the new code passed"
else
  say "smoke test FAILED: restoring the previous code (see /tmp/smoke_new.out, mirrored to backup/tmp_outputs)"
  cp "$T"/code_prev/acp/*.py "$T/code/acp/"; cp "$T"/code_prev/tools/*.py "$T/code/tools/"; cp "$T/code_prev/supervise.sh" "$T/code/supervise.sh"
fi
mkdir -p "$T/eval/paper_eval"
if timeout 420 "$PY" tools/paper_eval.py "$R/ckpt_$IT.pt" "$T/eval/paper_eval/s1_v1_$IT.json" > "$T/eval/paper_eval/s1_v1_$IT.out" 2>&1; then
  say "paper eval done"
else
  say "paper eval FAILED or timed out (rc $?), see eval/paper_eval/s1_v1_$IT.out"
fi
for i in $(seq 30); do gpu_free && break; sleep 2; done
rm -f "$R/STOP"
echo "--- $(date -u +%FT%T) restart after the pause at checkpoint $IT" >> "$R/train.out"
tmux new -d -s acp "bash $T/code/supervise.sh $R 36000 12000"
say "supervisor restarted"
# CPU: planner from this checkpoint's rollouts, and the planner-driven check
P=$T/runs/planner_s1_$IT; O=$T/eval/handover_pair_s1_${IT}_planner; mkdir -p "$P" "$O"
cat > "$P/run.sh" <<EOF
#!/usr/bin/env bash
cd $T/code; export CUDA_VISIBLE_DEVICES=; PY=\$HOME/acp_env/bin/python
\$PY tools/weak_sets.py $T/eval/paper_eval/s1_v1_${IT}_rollouts.pt cfg/stage1_sets.json 2>&1 | grep -v "^Warp\|^Module\|^   [A-Z\"/]\|^\$" > $T/eval/paper_eval/s1_v1_${IT}_weak_sets.txt
\$PY -m acp.planner label $T/eval/paper_eval/s1_v1_${IT}_rollouts.pt $P/data.pt
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
