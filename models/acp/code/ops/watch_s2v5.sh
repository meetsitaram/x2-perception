#!/usr/bin/env bash
# Fault fall-back watcher of s2_v5 stage 2 (ACP agent, 2026-11-13, state/acp.md 17j/17k). usage: watch_s2v5.sh <anchor it> <stage-2 end it>
# Only FAULTS act (user rule 2026-11-13: nothing stops automatically except crash loop, throughput, simulator faults). Constants below.
#   continuous (every 2 min, until the stage-2 end checkpoint exists):
#       CRASH       more than MAX_STARTS trainer starts within CRASH_WIN_S, or no new metrics row for STALE_S seconds
#   gates at IT0 + GATES (100-row window vs s2_v4's last 150 rows, runs/s2_v5/base_s2v4.json):
#       THROUGHPUT  window env_steps_per_s >= FPS_FRAC x base
#       SIM FAULTS  heartbeat sim_fault_share <= SIMF_MAX
#       RECIPE      window planner/updates mean == 10 (the planner is co-training; a recipe/code fault otherwise)
#   INFORMATION only (not a gate since 2026-11-13 17k): per-set training finish vs base. Window noise is at most 0.004 sd / 0.011
#       worst drop over s2_v4's last 1,000 rows, but the schedule itself moves training finish toward the planner-contact probe
#       finish as eta_obs -> 1, which is > 0.05 below training finish on 6 sets in s2_v4 (stairs turn -0.13, blocks -0.09, flat
#       -0.08, stairs -0.08, turn_place -0.06, chains -0.05): a finish gate would stop a healthy run doing what Eq. 3 demands.
# Fail = fall back: s2_v5 stops; s2_v4 resumes from its own latest.pt with its rearm_2h.sh flags.
# Writes runs/s2_v5/gate_<n>.txt, GATE_<n>_PASS or FALLBACK; restartable (gates with a PASS file are skipped).
set -u
MAX_STARTS=3; CRASH_WIN_S=7200; STALE_S=2700; FPS_FRAC=0.70; SIMF_MAX=0.05; GATES="200 700 1400 2200 3000"
IT0=$1; S3IT=$2; T=/mnt/filesystem-g4/tracks/acp; R=$T/runs/s2_v5; R4=$T/runs/s2_v4; L=$R/restart.log; GS=$T/code/tools/gate_stats.py
HB=/mnt/filesystem-g4/status/acp.json; BASE=$R/base_s2v4.json; PF=$T/eval/pair_s2_v2_005750/tracker_ckpt_005750.pt
END=$R/ckpt_$(printf %06d "$S3IT").pt
say() { echo "$(date -u +%FT%TZ) [watch5] $*" >> "$L"; }
gpu_free() { ! nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; }
fallback() {
  say "FALLBACK: $*"; echo "$(date -u +%FT%TZ) $*" > "$R/FALLBACK"
  touch "$R/STOP"; for i in $(seq 180); do gpu_free && ! tmux has-session -t acp 2>/dev/null && break; sleep 2; done
  TR=$T/_trash/$(date -u +%F); mkdir -p "$TR/stop_flags"; mv "$R/STOP" "$TR/stop_flags/STOP_s2_v5_fallback" && echo "stop_flags/STOP_s2_v5_fallback: s2_v5 stopped by watch_s2v5.sh" >> "$TR/NOTES.txt"
  OLD="--stage 2 --init $R4/init_from_s2_v3_latest.pt --planner-arch attn --u-obs 200 --u-rew 1700 --n-ann 3000 --eta-obs-max 0.8 --ramp-gate 2 --probe-envs 720 --gate-signal probe --ramp-gate-sets tp_t4x4u_y20_hs,tp_ramp_sh2_hs,x2_loco_flat_train_hold,x2_loco_blocks_train_hold --freeze-planner $PF --gate-backoff 1 --gate-rebase"
  tmux new -d -s acp "X2_EXCLUDE_PAIRS=hip ACP_PLANT=mc_20261024 ACP_HF_FIX=1 ACP_PAPER_DR=1 DATA=cfg/stage2h_sets.json EXTRA='$OLD' bash $T/code/supervise.sh $R4 36000 16000"
  say "s2_v4 resumed from its own latest.pt (its rearm_2h.sh flags)"; exit 0
}
check() {  # $1 = gate offset
  python3 "$GS" "$R/metrics.jsonl" 100 "$((IT0 + 1))" "$R/win_$1.json" > /dev/null
  python3 - "$BASE" "$R/win_$1.json" "$HB" "$R/metrics.jsonl" "$FPS_FRAC" "$SIMF_MAX" <<'EOF' > "$R/gate_$1.txt"
import json, sys
b, w = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])); fpsf, simf = map(float, sys.argv[5:7])
try:
    hb = json.load(open(sys.argv[3]))
except Exception:
    hb = {}
rows = [json.loads(l) for l in open(sys.argv[4]).readlines()[-100:]]
upd = sum(r.get("planner/updates", 0) for r in rows) / max(len(rows), 1)
sets = [k[len("finish/"):] for k, v in b.items() if k.startswith("finish/") and (b.get("episodes/" + k[len("finish/"):], 1) or 0) > 0 and (v or 0) > 0]
fb, fw, sf = b.get("env_steps_per_s", 0), w.get("env_steps_per_s", 0), hb.get("sim_fault_share")
fails = []
if fw < fpsf * fb: fails.append(f"throughput {fw:.0f} < {fpsf} x {fb:.0f}")
if sf is not None and sf > simf: fails.append(f"sim_fault_share {sf} > {simf}")
if upd != 10: fails.append(f"planner updates per iteration {upd:.2f} != 10")
eta = {k: rows[-1].get(k) for k in ("stage2/eta_obs", "stage2/eta_rew", "planner/fed_to_tracker_share")}
per = {s: [round(w.get("finish/" + s, 0), 4), round(b["finish/" + s], 4)] for s in sets}
low = [s for s in sets if (w.get("finish/" + s) or 0) < b["finish/" + s] - 0.05]
mean_w = sum(w.get("finish/" + s, 0) for s in sets) / max(len(sets), 1); mean_b = sum(b["finish/" + s] for s in sets) / max(len(sets), 1)
print("FAIL" if fails else "PASS", f"rows {w['n']} it {w['it_first']}-{w['it_last']} fps {fw:.0f} (base {fb:.0f}) sim_fault_share {sf} planner_updates {upd:.2f}",
      json.dumps(eta), "; ".join(fails), f"| INFO mean set finish {mean_w:.4f} (base {mean_b:.4f}); sets > 0.05 below base: {low}",
      "| per set [window, base]", json.dumps(per))
EOF
  say "gate +$1: $(cut -c1-900 "$R/gate_$1.txt")"
  grep -q '^PASS' "$R/gate_$1.txt" || fallback "gate +$1 $(cut -c1-400 "$R/gate_$1.txt")"
  cp "$R/gate_$1.txt" "$R/GATE_$1_PASS"
}
last_row() { tail -1 "$R/metrics.jsonl" 2>/dev/null | python3 -c "import json,sys; print(json.loads(sys.stdin.read()).get('it', 0))" 2>/dev/null || echo 0; }
recent_starts() { python3 - "$R/supervisor.log" "$CRASH_WIN_S" <<'EOF'
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
faults() {
  [ -f "$R/FALLBACK" ] && exit 0
  [ -f "$END" ] && { say "stage-2 end checkpoint $(basename "$END") present; watcher ends (acp-s3auto takes over)"; exit 0; }
  n=$(recent_starts); [ "$n" -gt "$MAX_STARTS" ] && fallback "crash loop: $n trainer starts within $CRASH_WIN_S s"
  age=$(( $(date +%s) - $(stat -c %Y "$R/metrics.jsonl" 2>/dev/null || date +%s) )); [ -f "$R/metrics.jsonl" ] && [ "$age" -gt "$STALE_S" ] && fallback "no metrics row for $age s"
}
wait_for() {  # $1 = iteration
  while true; do sleep 120; faults; [ "$(last_row)" -ge "$1" ] && return; done
}
say "watching from iteration $IT0 to stage-2 end $S3IT; gates at +$GATES; faults: starts>$MAX_STARTS in ${CRASH_WIN_S}s, stale>${STALE_S}s, fps>=${FPS_FRAC}xbase, sim_fault<=$SIMF_MAX, planner updates=10; per-set finish = information"
for g in $GATES; do
  [ -f "$R/GATE_${g}_PASS" ] || { wait_for $((IT0 + g)); check "$g"; }
done
say "all gates passed; continuous crash/stale watch until $S3IT"
while true; do sleep 120; faults; done
