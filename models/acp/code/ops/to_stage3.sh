#!/usr/bin/env bash
# Stage 2 -> stage 3 switch (coordinator, 2026-11-12: "prepare its switch so it can start early if stage 2 reaches its
# criterion before 6,500"). Criterion (ours; the paper gives only wall-clock): not before the reward ramp is complete
# (eta_rew = 1 at iteration 5,450), then at the first checkpoint from 5,500 on where the stage-2 finish rate
# (end/finished_phase099, mean of the last 250 iterations) gained less than 1 point over the 250 before (plateau);
# at the latest at 6,500 (end of the stage-2 budget). The switch: stop stage 2, install the stage-3 code (code_s3 ->
# code, previous code to _trash), check it in a throw-away run from that checkpoint (4,096 robots, 3 iterations), start
# stage 3 as run runs/s3_v1 from that checkpoint. On a failed check: the previous code is restored and stage 2 resumes.
#   tmux new -d -s acp-s3switch "bash /mnt/filesystem-g4/tracks/acp/code/ops/to_stage3.sh"
#   FORCE_IT=<it>: switch at that checkpoint regardless of the criterion.
set -u
T=/mnt/filesystem-g4/tracks/acp; R2=$T/runs/s2_v1; R3=$T/runs/s3_v1; L=$R2/stage3_switch.log; HB=/mnt/filesystem-g4/status/acp.json
PY=$HOME/acp_env/bin/python
S3_ITERS=${S3_ITERS:-14000}
EXTRA2="--stage 2 --init $R2/init_from_s1_v1_002250.pt --planner-arch attn --u-obs 200 --u-rew 1700 --n-ann 1500"
DATA2=${DATA2:-cfg/stage2c_sets.json}
say() { echo "$(date -u +%FT%TZ) $*" >> "$L"; }
gpu_free() { ! nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; }
plateau() {   # $1 = iteration: 0 if the finish rate gained < 1 point over the last 250 vs the 250 before
  python3 - "$R2/metrics.jsonl" "$1" <<'PY'
import json, sys
it = int(sys.argv[2]); rows = {}
for l in open(sys.argv[1]):
    r = json.loads(l); rows[r["it"]] = r.get("end/finished_phase099")
a = [rows[i] for i in range(it - 249, it + 1) if rows.get(i) is not None]
b = [rows[i] for i in range(it - 499, it - 249) if rows.get(i) is not None]
if len(a) < 200 or len(b) < 200:
    sys.exit(1)
ga = sum(a) / len(a) - sum(b) / len(b)
print(f"finish rate {sum(b) / len(b):.4f} -> {sum(a) / len(a):.4f} (gain {100 * ga:+.2f} points)")
sys.exit(0 if ga < 0.01 else 1)
PY
}
say "armed (criterion: plateau from 5,500, at the latest 6,500; FORCE_IT=${FORCE_IT:-none})"
IT=""
for c in $(seq 5500 250 6500); do
  [ -n "${FORCE_IT:-}" ] && c=$FORCE_IT
  P=$(printf %06d "$c")
  while [ ! -f "$R2/ckpt_$P.pt" ]; do sleep 20; done
  if [ -n "${FORCE_IT:-}" ] || [ "$c" -ge 6500 ]; then IT=$c; say "checkpoint $c: switching (${FORCE_IT:+forced}${FORCE_IT:-budget end})"; break; fi
  msg=$(plateau "$c"); rc=$?
  say "checkpoint $c: $msg"
  if [ $rc -eq 0 ]; then IT=$c; say "plateau at $c: switching"; break; fi
done
P=$(printf %06d "$IT"); CK=$R2/ckpt_$P.pt
exec 8> /tmp/acp_eval.lock; flock -w 1800 8   # wait for a live eval to finish; none starts during the switch
sleep 4
[ -f "$R2/ckpt_006500.pt" ] || touch "$R2/STOP"
for i in $(seq 90); do gpu_free && ! tmux has-session -t acp 2>/dev/null && break; sleep 2; done
say "stage 2 stopped at checkpoint $IT; GPU free: $(gpu_free && echo yes || echo no)"
printf '{"track":"acp","state":"stage 2 -> stage 3 switch at checkpoint %s (under 10 minutes)","run":"%s","newest_checkpoint":"%s","unix":%d,"time":"%s"}\n' "$IT" "$R3" "$CK" "$(date +%s)" "$(date +%FT%T%z)" > "$HB"
TR=$T/_trash/$(date -u +%F); mkdir -p "$TR/stop_flags"
[ -f "$R2/STOP" ] && mv "$R2/STOP" "$TR/stop_flags/STOP_s2_v1_$P" && echo "stop_flags/STOP_s2_v1_$P: stop flag of the stage-3 switch" >> "$TR/NOTES.txt"
[ -d "$T/code_prev" ] && mv "$T/code_prev" "$TR/code_prev_$(date -u +%H%M%S)" && echo "code_prev_*: roll-back copy from before an install; superseded" >> "$TR/NOTES.txt"
mkdir -p "$T/code_prev"; cp -r "$T/code/acp" "$T/code/tools" "$T/code_prev/"
cp "$T"/code_s3/acp/*.py "$T/code/acp/"; cp "$T"/code_s3/tools/*.py "$T/code/tools/"
cd "$T/code"
rm -rf /tmp/s3_check; mkdir -p /tmp/s3_check
if ACP_PLANT=mc_20261024 timeout 300 "$PY" -m acp.train --data cfg/stage3_sets.json --run /tmp/s3_check --envs 4096 --iters $((10#$IT + 3)) --njmax 450 \
     --init "$CK" --stage 3 --planner-arch attn --u-obs 200 --u-rew 1700 --n-ann 1500 --goal-share 0.5 --heartbeat /tmp/hb_s3.json > /tmp/s3_check.out 2>&1 \
   && grep -q "^\[stage 3\]" /tmp/s3_check.out && [ "$(wc -l < /tmp/s3_check/metrics.jsonl)" -ge 3 ]; then
  say "stage-3 check passed: $(grep '^\[stage 3\]\|^\[data\] goal' /tmp/s3_check.out | tr '\n' ' ' | cut -c1-300) | $(tail -1 /tmp/s3_check/metrics.jsonl | cut -c1-200)"
  for i in $(seq 30); do gpu_free && break; sleep 2; done
  mkdir -p "$R3"; cp "$CK" "$R3/init_from_s2_v1_$P.pt"
  tmux new -d -s acp "ACP_PLANT=mc_20261024 DATA=cfg/stage3_sets.json EXTRA='--stage 3 --init $R3/init_from_s2_v1_$P.pt --planner-arch attn --u-obs 200 --u-rew 1700 --n-ann 1500 --goal-share 0.5' bash $T/code/supervise.sh $R3 36000 $S3_ITERS"
  say "stage 3 started: run $R3 from $CK, budget to $S3_ITERS"
else
  say "stage-3 check FAILED (see /tmp/s3_check.out): previous code restored, stage 2 resumes"
  cp "$T"/code_prev/acp/*.py "$T/code/acp/"; cp "$T"/code_prev/tools/* "$T/code/tools/" 2>/dev/null
  for i in $(seq 30); do gpu_free && break; sleep 2; done
  [ -f "$R2/ckpt_006500.pt" ] || tmux new -d -s acp "ACP_PLANT=mc_20261024 DATA=$DATA2 EXTRA='$EXTRA2' bash $T/code/supervise.sh $R2 36000 6500"
fi
