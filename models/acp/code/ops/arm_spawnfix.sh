#!/bin/bash
# ARMED switch of the running stage 3 (runs/s3_v2) to the spawn-fixed goal tiles (cfg/stage3v2i_sets.json = stage3v2h + goal
# tiles v2c), ACP agent 2026-11-14, state/acp.md 17m. Waits for runs/s3_v2/GO_spawnfix (the user's decision; never created by an
# agent). Needs runs/s3_v2_diag/PREFLIGHT_SPAWNFIX_OK. Then: wait for the next latest.pt save, STOP the supervisor, start it again
# in the same run with only DATA changed (resume from latest.pt: same iteration count, s3_start, curriculum levels, optimiser;
# the planner replay refills as after any resume). Same end 34,218. Logs to runs/s2_v5/restart.log ([spawnfix]).
T=${ACP_T:-/mnt/filesystem-g4/tracks/acp}; C=$T/code; R3=$T/runs/s3_v2; L=$T/runs/s2_v5/restart.log
GO=$R3/GO_spawnfix; OK=$T/runs/s3_v2_diag/PREFLIGHT_SPAWNFIX_OK; NEWDATA=cfg/stage3v2i_sets.json; END=34218
say() { echo "$(date -u +%FT%TZ) [spawnfix] $*" | tee -a "$L"; }
say "armed: waiting for $GO"
until [ -f "$GO" ]; do sleep 60; done
[ -f "$OK" ] || { say "GO found but no $OK; not switching"; exit 1; }
[ -f "$R3/S3_FAULT" ] && { say "S3_FAULT present; not switching"; exit 1; }
grep -q "DATA=cfg/stage3v2h_sets.json" "$R3/launch_flags.txt" || { say "launch flags not the expected v2h ones; not switching"; exit 1; }
m0=$(stat -c %Y "$R3/latest.pt"); say "GO found; waiting for the next latest.pt save (now $(date -u -d @$m0 +%T))"
for i in $(seq 1 1800); do [ "$(stat -c %Y "$R3/latest.pt")" -gt "$m0" ] && break; sleep 2; done
sleep 20                                                   # let the save finish
touch "$R3/STOP"; say "STOP written after the latest.pt save"
for i in $(seq 1 180); do ! tmux has-session -t acp 2>/dev/null && ! nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . && break; sleep 2; done
tmux has-session -t acp 2>/dev/null && { say "supervisor still running after 6 min; leaving STOP, not switching"; exit 1; }
TR=$T/_trash/$(date -u +%F); mkdir -p "$TR/stop_flags"; mv "$R3/STOP" "$TR/stop_flags/STOP_s3_v2_spawnfix"
echo "stop_flags/STOP_s3_v2_spawnfix: s3_v2 stopped by ops/arm_spawnfix.sh for the data switch" >> "$TR/NOTES.txt"
cp "$R3/launch_flags.txt" "$R3/launch_flags_v2h.txt"
NEW=$(sed "s#DATA=cfg/stage3v2h_sets.json#DATA=$NEWDATA#" "$R3/launch_flags_v2h.txt"); echo "$NEW" > "$R3/launch_flags.txt"
tmux new -d -s acp "$NEW bash $C/supervise.sh $R3 36000 $END"
say "s3_v2 restarted from latest.pt with DATA=$NEWDATA (flags in $R3/launch_flags.txt; old ones launch_flags_v2h.txt)"
