#!/usr/bin/env bash
# Automatic stage-2 -> stage-3 switch of s2_v5 (ACP agent, 2026-11-13, state/acp.md 17k; user schedule in docs/USER_RULES.md
# "ACP fixed stage lengths"). No human step. Started by arm_s2v5.sh at GO (tmux acp-s3auto); restartable.
#   - s2_v5 is supervised to exactly S3IT (= 2,300 iterations after eta_rew reaches 1, runs/s2_v5/schedule.json).
#   - From S3IT - PRECHECK iterations on: if code/STAGE3_READY exists, a 3-iteration CPU smoke of stage 3 from s2_v5/latest.pt with
#     the launch flags of code/ops/stage3_flags.env (CUDA hidden, nice, cores 12-19; never the GPU) -> runs/s2_v5/S3_CPU_CHECK_OK.
#   - At S3IT (ckpt present, trainer exited): with S3_CPU_CHECK_OK, stage 3 starts at once in runs/s3_v2 from ckpt_<S3IT> for exactly
#     15,000 iterations (eta_obs = eta_rew = 1 carried: both are 1 at S3IT, no jump). Without it the run HOLDS: s2_v5 continues
#     stage 2 in chunks of HOLD iterations (same flags) and the check is retried; stage 3 then starts at the end of a chunk and
#     still runs 15,000 iterations. The untested stage-3 code is never started.
#   - Stage-3 fault fall-back (faults only, user rule): more than 3 trainer starts in 2 h or no metrics row for 45 min -> stage 3
#     stops, runs/s3_v2/S3_FAULT, s2_v5 continues stage 2 from its own latest (hold) until the agent acts.
# Nothing here pauses training for evaluation.
set -u
T=${ACP_T:-/mnt/filesystem-g4/tracks/acp}; R=$T/runs/s2_v5; R3=$T/runs/s3_v2; L=$R/restart.log; PY=${ACP_PY:-$HOME/acp_env/bin/python}
TICK=${ACP_TICK:-60}   # ACP_T / ACP_PY / ACP_TICK: sandbox test only (ops/test_auto_stage3.sh)
PRECHECK=300; HOLD=500; S3LEN=15000; MAX_STARTS=3; CRASH_WIN_S=7200; STALE_S=2700
FLAGS=$T/code/ops/stage3_flags.env; READY=$T/code/STAGE3_READY; OK=$R/S3_CPU_CHECK_OK
say() { echo "$(date -u +%FT%TZ) [s3auto] $*" >> "$L"; }
gpu_free() { ! nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; }
last_row() { tail -1 "$1/metrics.jsonl" 2>/dev/null | python3 -c "import json,sys; print(json.loads(sys.stdin.read()).get('it', 0))" 2>/dev/null || echo 0; }
sched() { python3 -c "import json;print(json.load(open('$R/schedule.json'))['$1'])"; }
while [ ! -f "$R/schedule.json" ]; do sleep $TICK; done
S3IT=$(sched stage3_start); TARGET=$S3IT
[ -f "$R/hold_target" ] && TARGET=$(cat "$R/hold_target")
say "started: stage-2 end $S3IT (current target $TARGET), stage 3 $S3LEN iterations, CPU check from $((S3IT - PRECHECK))"
cpu_check() {  # 3-iteration stage-3 smoke on CPU from s2_v5/latest.pt with the launch flags
  [ -f "$READY" ] && [ -f "$FLAGS" ] || return 1
  . "$FLAGS"; D=$R/s3_cpu_check; rm -rf "$D"; mkdir -p "$D"; cp "$R/latest.pt" "$D/init.pt"
  IT=$(CUDA_VISIBLE_DEVICES= "$PY" -c "import torch,sys; print(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['it'])" "$D/init.pt" | tail -1)
  ( cd "$T/code" && env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 $S3_ENVV ACP_SMOKE_PREFILL=2000 timeout 3600 nice -n 15 taskset -c 12-19 "$PY" -m acp.train \
      --data "$S3_DATA" --run "$D/run" --envs 64 --iters $((IT + 3)) --njmax 450 --device cpu --heartbeat "$D/hb.json" --rec-envs 64 \
      --replay 100000 --init "$D/init.pt" $(echo "$S3_EX" | sed "s/@ANCHOR@/$IT/; s/--probe-envs [0-9]*/--probe-envs 8/") > "$D/smoke.out" 2>&1 )
  rc=$?
  if [ $rc -eq 0 ] && grep -q "^\[stage 3\]" "$D/smoke.out" && [ "$(wc -l < "$D/run/metrics.jsonl" 2>/dev/null || echo 0)" -ge 3 ] \
     && python3 "$T/code/tools/s3_smoke_check.py" "$D/run/metrics.jsonl" > "$D/check.txt" 2>&1; then
    cp "$D/check.txt" "$OK"; say "stage-3 CPU check PASS from it $IT: $(tail -1 "$D/check.txt" | cut -c1-300)"; return 0
  fi
  say "stage-3 CPU check FAILED rc=$rc (see $D/smoke.out, $D/check.txt)"; return 1
}
start_s3() {  # $1 = stage-2 checkpoint
  . "$FLAGS"; IT=$((10#$(basename "$1" .pt | cut -d_ -f2))); END=$((IT + S3LEN)); mkdir -p "$R3"
  cp "$1" "$R3/init_stage2.pt"; EX=$(echo "$S3_EX" | sed "s/@ANCHOR@/$IT/")
  EX="$EX --init $R3/init_stage2.pt"
  echo "$S3_ENVV DATA=$S3_DATA EXTRA='$EX'" > "$R3/launch_flags.txt"
  mkdir -p "$R3/code_snapshot"; cp -r "$T/code/acp" "$T/code/ops" "$T/code/supervise.sh" "$R3/code_snapshot/" 2>/dev/null
  python3 - "$R/schedule.json" "$IT" "$END" <<'EOF'
import json, sys, time
p = sys.argv[1]; s = json.load(open(p)); s["stage3_start_actual"] = int(sys.argv[2]); s["run_end_actual"] = int(sys.argv[3])
s["stage3_started_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()); json.dump(s, open(p, "w"))
EOF
  tmux new -d -s acp "$S3_ENVV DATA=$S3_DATA EXTRA='$EX' bash $T/code/supervise.sh $R3 36000 $END"
  say "STAGE 3 STARTED in $R3 from $1 (iteration $IT) to $END (flags in $R3/launch_flags.txt)"
}
hold_chunk() {  # continue stage 2 of s2_v5 for HOLD iterations with its own flags
  TARGET=$(( $(last_row "$R") + HOLD )); echo "$TARGET" > "$R/hold_target"
  tmux new -d -s acp "$(cat "$R/launch_flags.txt") bash $T/code/supervise.sh $R 36000 $TARGET"
  say "HOLD: stage 3 not ready/tested; s2_v5 continues stage 2 to $TARGET"
}
recent_starts() { python3 - "$1/supervisor.log" "$CRASH_WIN_S" <<'EOF'
import sys, time, datetime
n = 0
try:
    for l in open(sys.argv[1]):
        if "starting trainer" in l:
            t = datetime.datetime.strptime(l[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=datetime.timezone.utc).timestamp()
            n += time.time() - t < float(sys.argv[2])
except Exception:
    pass
print(n)
EOF
}
watch_s3() {  # faults only
  while true; do
    sleep $((2 * TICK))
    [ -f "$R3/ckpt_$(printf %06d "$END").pt" ] && { say "stage 3 reached its end $END; switch script ends"; exit 0; }
    n=$(recent_starts "$R3"); age=$(( $(date +%s) - $(stat -c %Y "$R3/metrics.jsonl" 2>/dev/null || date +%s) ))
    why=""; [ "$n" -gt "$MAX_STARTS" ] && why="crash loop: $n trainer starts in $CRASH_WIN_S s"
    [ -f "$R3/metrics.jsonl" ] && [ "$age" -gt "$STALE_S" ] && why="no metrics row for $age s"
    if [ -n "$why" ]; then
      say "STAGE-3 FAULT: $why; stage 3 stops, s2_v5 continues stage 2 (hold)"; echo "$(date -u +%FT%TZ) $why" > "$R3/S3_FAULT"
      touch "$R3/STOP"; for i in $(seq 180); do gpu_free && ! tmux has-session -t acp 2>/dev/null && break; sleep 2; done
      TR=$T/_trash/$(date -u +%F); mkdir -p "$TR/stop_flags"; mv "$R3/STOP" "$TR/stop_flags/STOP_s3_v2_fault" && echo "stop_flags/STOP_s3_v2_fault: s3_v2 stopped by auto_stage3.sh" >> "$TR/NOTES.txt"
      rm -f "$OK"; hold_chunk; exit 0
    fi
  done
}
[ -f "$R3/launch_flags.txt" ] && { END=$(python3 -c "import json;print(json.load(open('$R/schedule.json'))['run_end_actual'])"); say "stage 3 already started; watching faults to $END"; watch_s3; }
while true; do
  sleep $TICK
  [ -f "$R/FALLBACK" ] && { say "s2_v5 fell back (watch_s2v5); switch script ends"; exit 0; }
  CK=$R/ckpt_$(printf %06d "$TARGET").pt
  stamp=$(stat -c %Y "$READY" "$FLAGS" 2>/dev/null | tr -s '\n' ' ')
  if [ ! -f "$OK" ] && [ "$(last_row "$R")" -ge $((TARGET - PRECHECK)) ] && [ -f "$READY" ] && [ -f "$FLAGS" ] \
     && [ "$(cat "$R/s3_cpu_check_failed_for" 2>/dev/null)" != "$stamp" ]; then
    cpu_check || echo "$stamp" > "$R/s3_cpu_check_failed_for"   # retried only after READY or the flags change
  fi
  if [ -f "$CK" ] && ! tmux has-session -t acp 2>/dev/null && gpu_free; then
    if [ -f "$OK" ]; then start_s3 "$CK"; watch_s3; else hold_chunk; fi
  fi
done
