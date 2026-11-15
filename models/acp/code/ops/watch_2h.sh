#!/usr/bin/env bash
# Fall-back gates for the data-2h restart (ACP agent, 2026-11-13, state/acp.md 17d). usage: watch_2h.sh <restart iteration>
# Baseline = s2_v3's last 150 rows before the stop (written once to runs/s2_v4/base_2g.json).
# Gate A at +150 iterations, gate B at +400 (window: the last 100 rows):
#   crash      more than 3 trainer starts, or no new metrics row for 45 min
#   throughput env_steps_per_s >= 0.70 x baseline
#   sim faults heartbeat sim_fault_share <= 0.05 (WARN logged above 0.02)
#   finish     mean training finish over the 2g sets >= baseline - 0.05 (A) / - 0.03 (B);
#              mean probe finish of the 4 gate sets (4x4, ramp, flat, blocks) >= baseline - 0.08 (A) / - 0.05 (B)
# Fail = fall back: s2_v4 stops, s2_v3 resumes from its own latest (data 2g, no ACP_HF_FIX / ACP_PAPER_DR) with code/ as it
# is (one code folder since 2026-11-13, state/acp.md 17e: with the flags off it is the pre-2h code except the simcheck
# truncation, which docs/TRACK_RULES.md requires). Restartable: a gate with its GATE_x_PASS file is skipped. Writes
# runs/s2_v4/GATE_A_PASS|GATE_B_PASS|FALLBACK with the numbers. The eval gate (hold-out gate on the scorer, first two
# post-restart pairs) is read by the agent / coordinator, not here: the scorer runs off the node.
set -u
IT0=$1; T=/mnt/filesystem-g4/tracks/acp; R=$T/runs/s2_v4; R3=$T/runs/s2_v3; L=$R/restart.log; GS=$T/code/tools/gate_stats.py
PF=$T/eval/pair_s2_v2_005750/tracker_ckpt_005750.pt; HB=/mnt/filesystem-g4/status/acp.json
say() { echo "$(date -u +%FT%TZ) [watch] $*" >> "$L"; }
gpu_free() { ! nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; }
[ -f "$R/base_2g.json" ] || python3 "$GS" "$R3/metrics.jsonl" 150 0 "$R/base_2g.json" > /dev/null
OLDSETS="x2_loco_flat_train_hold x2_loco_blocks_train_hold x2_chains_far_hs tp_stairs_x10_sh2_hs gst_v1_hs gs2dn_v2_hs gs2lo_v1_hs tp_ramp_sh2_hs tp_t4x4u_y20_hs stand_edge_v1 turn_place_v1"
BASE=${BASE:-$R/base_2g.json}; PERSET=${PERSET:-9}
GATESETS="tp_t4x4u_y20_hs tp_ramp_sh2_hs x2_loco_flat_train_hold x2_loco_blocks_train_hold"
fallback() {
  say "FALLBACK: $*"; echo "$(date -u +%FT%TZ) $*" > "$R/FALLBACK"
  touch "$R/STOP"; for i in $(seq 120); do gpu_free && ! tmux has-session -t acp 2>/dev/null && break; sleep 2; done
  TR=$T/_trash/$(date -u +%F); mkdir -p "$TR/stop_flags"; mv "$R/STOP" "$TR/stop_flags/STOP_s2_v4_fallback" && echo "stop_flags/STOP_s2_v4_fallback: s2_v4 stopped by the 2h watcher" >> "$TR/NOTES.txt"
  mkdir -p "$TR/code_2h_$(date -u +%H%M%S)"; cp "$T"/code/acp/*.py "$TR/code_2h_$(date -u +%H%M%S)/" 2>/dev/null
  OLD="--stage 2 --init $R3/init_from_s2_v2_007000.pt --planner-arch attn --u-obs 200 --u-rew 1700 --n-ann 3000 --eta-obs-max 0.8 --ramp-gate 2 --probe-envs 720 --gate-signal probe --ramp-gate-sets tp_t4x4u_y20_hs,tp_ramp_sh2_hs,x2_loco_flat_train_hold,x2_loco_blocks_train_hold --freeze-planner $PF --gate-backoff 1 --gate-rebase"
  tmux new -d -s acp "X2_EXCLUDE_PAIRS=hip ACP_PLANT=mc_20261024 DATA=cfg/stage2g_sets.json EXTRA='$OLD' bash $T/code/supervise.sh $R3 36000 14000"
  say "s2_v3 resumed from its latest (data 2g, code/ unchanged, no 2h flags)"; exit 0
}
check() {  # $1 = gate name, $2 = train tolerance, $3 = probe tolerance
  python3 "$GS" "$R/metrics.jsonl" 100 "$((IT0 + 1))" "$R/win_$1.json" > /dev/null
  python3 - "$BASE" "$R/win_$1.json" "$2" "$3" "$HB" "$OLDSETS" "$GATESETS" "$PERSET" <<'EOF' > "$R/gate_$1.txt"
import json, sys
b, w = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])); tt, tp = float(sys.argv[3]), float(sys.argv[4])
try:
    hb = json.load(open(sys.argv[5]))
except Exception:
    hb = {}
old, gate = sys.argv[6].split(), sys.argv[7].split()
mean = lambda d, ks: sum(d.get(k, 0.0) or 0.0 for k in ks) / len(ks)
tb, tw = mean(b, ["finish/" + s for s in old]), mean(w, ["finish/" + s for s in old])
pb, pw = mean(b, ["probe/finish/" + s for s in gate]), mean(w, ["probe/finish/" + s for s in gate])
fb, fw = b.get("env_steps_per_s", 0), w.get("env_steps_per_s", 0); sf = hb.get("sim_fault_share")
fails = []
if fw < 0.70 * fb: fails.append(f"throughput {fw:.0f} < 0.70 x {fb:.0f}")
if sf is not None and sf > 0.05: fails.append(f"sim_fault_share {sf} > 0.05")
if tw < tb - tt: fails.append(f"train finish {tw:.4f} < {tb:.4f} - {tt}")
if pw < pb - tp: fails.append(f"probe finish {pw:.4f} < {pb:.4f} - {tp}")
ps = float(sys.argv[8]) if len(sys.argv) > 8 else 9.0
fails += [f"set {s} {w.get('finish/'+s,0):.4f} < {b.get('finish/'+s,0):.4f} - {ps}" for s in old if (w.get("finish/"+s,0) or 0) < (b.get("finish/"+s,0) or 0) - ps]
new = {k: v for k, v in w.items() if k.startswith(("finish/", "probe/finish/")) and k.split("/")[-1] not in old}
print("FAIL" if fails else "PASS", f"rows {w['n']} it {w['it_first']}-{w['it_last']} train {tw:.4f} (base {tb:.4f}) probe-gate {pw:.4f} (base {pb:.4f})",
      f"fps {fw:.0f} (base {fb:.0f}) sim_fault_share {sf}" + (" WARN>0.02" if sf and sf > 0.02 else ""), "; ".join(fails), "| new sets", json.dumps(new))
EOF
  say "gate $1: $(cut -c1-900 "$R/gate_$1.txt")"
  grep -q '^PASS' "$R/gate_$1.txt" || fallback "gate $1 $(cut -c1-400 "$R/gate_$1.txt")"
  cp "$R/gate_$1.txt" "$R/GATE_$1_PASS"
}
last_row() { tail -1 "$R/metrics.jsonl" 2>/dev/null | python3 -c "import json,sys; print(json.loads(sys.stdin.read()).get('it', 0))" 2>/dev/null || echo 0; }
wait_for() {  # $1 = iteration
  while true; do
    sleep 120
    [ -f "$R/FALLBACK" ] && exit 0
    n=$(grep -c "starting trainer" "$R/supervisor.log" 2>/dev/null || echo 0); [ "$n" -gt 3 ] && fallback "crash: $n trainer starts"
    age=$(( $(date +%s) - $(stat -c %Y "$R/metrics.jsonl" 2>/dev/null || date +%s) )); [ -f "$R/metrics.jsonl" ] && [ "$age" -gt 2700 ] && fallback "no metrics row for $age s"
    [ "$(last_row)" -ge "$1" ] && return
  done
}
say "watching from iteration $IT0 (gate A at $((IT0 + 150)), gate B at $((IT0 + 400)))"
if [ "${REARM:-0}" = 1 ]; then
  # Re-arm after the gate-B fall-back of 07:17 (state/acp.md 17f): base = s2_v4's own window 10660-10759 (same data 2h + paper DR),
  # gate C at +400: old-set mean >= base - 0.015, probe-gate mean >= base - 0.03, no old set > 0.04 below its base value.
  [ -f "$R/GATE_C_PASS" ] || { wait_for $((IT0 + 400)); check C 0.015 0.03; }
else
[ -f "$R/GATE_A_PASS" ] || { wait_for $((IT0 + 150)); check A 0.05 0.08; }
[ -f "$R/GATE_B_PASS" ] || { wait_for $((IT0 + 400)); check B 0.03 0.05; }
fi
say "both gates passed; watcher ends (eval gate: scorer on the first two post-restart pairs)"
