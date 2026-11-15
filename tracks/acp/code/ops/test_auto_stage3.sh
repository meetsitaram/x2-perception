#!/usr/bin/env bash
# Sandbox test of ops/auto_stage3.sh (ACP agent, 2026-11-13, state/acp.md 17k): runs the real script against a fake tree in /tmp with
# stub tmux / nvidia-smi and a fake python (no simulator, no GPU, nothing under runs/ touched). Three scenarios:
#   S1 ready: CPU check at S3IT-300, stage 3 starts from ckpt_<S3IT> to S3IT+15000 with eta anchored at S3IT, ends at its last ckpt
#   S2 not ready: at S3IT the run HOLDS (s2_v5 continues stage 2 +500 with its own flags); READY appears; check; stage 3 from the chunk end
#   S3 stage-3 crash loop: stage 3 stops, S3_FAULT written, s2_v5 continues stage 2 (hold)
# Out: runs/s3_build/test_auto_stage3.txt
set -u
REAL=/mnt/filesystem-g4/tracks/acp; OUT=$REAL/runs/s3_build/test_auto_stage3.txt; : > "$OUT"
say() { echo "$*" | tee -a "$OUT"; }
mk() {  # fresh sandbox $1
  SB=$1; rm -rf "$SB"; mkdir -p "$SB/runs/s2_v5" "$SB/code/ops" "$SB/bin"
  ln -s "$REAL/code/tools" "$SB/code/tools"; cp "$REAL/code/ops/stage3_flags.env" "$SB/code/ops/"
  echo "X2_EXCLUDE_PAIRS=hip DATA=cfg/stage2h_sets.json EXTRA='--stage 2 --n-ann 6000'" > "$SB/runs/s2_v5/launch_flags.txt"
  echo '{"stage3_start": 1000, "run_end": 16000}' > "$SB/runs/s2_v5/schedule.json"
  echo x > "$SB/runs/s2_v5/latest.pt"
  cat > "$SB/bin/tmux" <<EOF
#!/usr/bin/env bash
echo "tmux \$*" >> $SB/tmux.log
[ "\$1" = has-session ] && exit 1
exit 0
EOF
  printf '#!/usr/bin/env bash\nexit 0\n' > "$SB/bin/nvidia-smi"
  cat > "$SB/bin/fakepy" <<'EOF'
#!/usr/bin/env bash
if [ "$1" = -c ]; then echo 990; exit 0; fi
run=""; while [ $# -gt 0 ]; do [ "$1" = --run ] && run=$2; shift; done
mkdir -p "$run"; echo "[stage 3] fake"
for it in 991 992 993; do echo "{\"it\": $it, \"stage\": 3, \"stage2/eta_obs\": 1.0, \"stage2/eta_rew\": 1.0, \"stage3/warmup\": 1, \"planner/updates\": 10, \"ppo/value_loss\": 0.1, \"ppo/value_loss_goal\": 0.1, \"planner/loss_affordance\": 0.1, \"reward_per_step\": 0.1, \"planner/replay_added\": 3, \"goal/success\": 0.0, \"goal/level_mean\": 0.0, \"weight/goal/stumbling\": -10, \"weight/goal/slippage\": -10}" >> "$run/metrics.jsonl"; done
EOF
  chmod +x "$SB/bin/"*
}
row() { echo "{\"it\": $2}" >> "$1/metrics.jsonl"; }
run_auto() { PATH=$SB/bin:$PATH ACP_T=$SB ACP_PY=$SB/bin/fakepy ACP_TICK=1 timeout 120 bash "$REAL/code/ops/auto_stage3.sh" & AP=$!; }
waitfor() { for i in $(seq 60); do grep -q "$2" "$1" 2>/dev/null && return 0; sleep 1; done; return 1; }
ok_all=1; res() { if [ "$1" = 0 ]; then say "PASS $2"; else say "FAIL $2"; ok_all=0; fi; }

# S1
mk /tmp/acp_auto_s1; touch "$SB/code/STAGE3_READY"; row "$SB/runs/s2_v5" 750; run_auto
waitfor "$SB/runs/s2_v5/restart.log" "CPU check PASS"; r1=$?
touch "$SB/runs/s2_v5/ckpt_001000.pt"; row "$SB/runs/s2_v5" 1000
waitfor "$SB/runs/s2_v5/restart.log" "STAGE 3 STARTED"; r2=$?
grep -q "supervise.sh $SB/runs/s3_v2 36000 16000" "$SB/tmux.log" && grep -q -- "--eta-anchor-it 1000 --eta-obs0 1 --eta-rew0 1" "$SB/runs/s3_v2/launch_flags.txt" \
  && grep -q -- "--init $SB/runs/s3_v2/init_stage2.pt" "$SB/runs/s3_v2/launch_flags.txt" && grep -q '"run_end_actual": 16000' "$SB/runs/s2_v5/schedule.json"; r3=$?
row "$SB/runs/s3_v2" 1001; touch "$SB/runs/s3_v2/ckpt_016000.pt"; waitfor "$SB/runs/s2_v5/restart.log" "reached its end"; r4=$?
wait $AP 2>/dev/null
res $((r1 + r2 + r3 + r4)) "S1 ready: check at S3IT-300 ($r1), switch at S3IT ($r2), stage 3 to S3IT+15000 with eta anchored, init and schedule ($r3), ends at its last ckpt ($r4)"
say "  S1 log: $(grep -o '\[s3auto\].*' "$SB/runs/s2_v5/restart.log" | cut -c1-160 | tr '\n' '|')"

# S2
mk /tmp/acp_auto_s2; row "$SB/runs/s2_v5" 1000; touch "$SB/runs/s2_v5/ckpt_001000.pt"; run_auto
waitfor "$SB/runs/s2_v5/restart.log" "HOLD"; r1=$?
grep -q "X2_EXCLUDE_PAIRS=hip DATA=cfg/stage2h_sets.json EXTRA='--stage 2 --n-ann 6000' bash $SB/code/supervise.sh $SB/runs/s2_v5 36000 1500" "$SB/tmux.log"; r2=$?
! grep -q "s3_v2" "$SB/tmux.log"; r3=$?
touch "$SB/code/STAGE3_READY"; row "$SB/runs/s2_v5" 1250; waitfor "$SB/runs/s2_v5/restart.log" "CPU check PASS"; r4=$?
touch "$SB/runs/s2_v5/ckpt_001500.pt"; row "$SB/runs/s2_v5" 1500; waitfor "$SB/runs/s2_v5/restart.log" "STAGE 3 STARTED"; r5=$?
grep -q "supervise.sh $SB/runs/s3_v2 36000 16500" "$SB/tmux.log"; r6=$?
kill $AP 2>/dev/null; wait $AP 2>/dev/null
res $((r1 + r2 + r3 + r4 + r5 + r6)) "S2 not ready: HOLD at S3IT ($r1) with s2_v5's own flags to +500 ($r2), no stage-3 start untested ($r3), check after READY ($r4), switch at the chunk end ($r5) for 15000 ($r6)"

# S3
mk /tmp/acp_auto_s3; touch "$SB/code/STAGE3_READY"; row "$SB/runs/s2_v5" 1000; echo ok > "$SB/runs/s2_v5/S3_CPU_CHECK_OK"; touch "$SB/runs/s2_v5/ckpt_001000.pt"; run_auto
waitfor "$SB/runs/s2_v5/restart.log" "STAGE 3 STARTED"; r1=$?
row "$SB/runs/s3_v2" 1001; for i in 1 2 3 4 5; do echo "$(date -u +%FT%T) starting trainer (restart $i)" >> "$SB/runs/s3_v2/supervisor.log"; done
waitfor "$SB/runs/s2_v5/restart.log" "STAGE-3 FAULT"; r2=$?
waitfor "$SB/runs/s2_v5/restart.log" "HOLD"; r3=$?
[ -f "$SB/runs/s3_v2/S3_FAULT" ] && [ ! -f "$SB/runs/s2_v5/S3_CPU_CHECK_OK" ]; r4=$?
kill $AP 2>/dev/null; wait $AP 2>/dev/null
res $((r1 + r2 + r3 + r4)) "S3 stage-3 crash loop: started ($r1), fault seen ($r2), s2_v5 continues stage 2 ($r3), S3_FAULT and check reset ($r4)"
[ $ok_all = 1 ] && say "AUTO SWITCH TEST ALL PASS" || say "AUTO SWITCH TEST FAILED"
rm -rf /tmp/acp_auto_s1 /tmp/acp_auto_s2 /tmp/acp_auto_s3
