#!/usr/bin/env bash
# CPU pre-flight of s2_v5 (ACP agent, 2026-11-13, state/acp.md 17j). CUDA hidden, niced, cores 12-23; never touches the GPU.
#   1 simcheck preflight with the launch's build (cfg 2h, ACP_HF_FIX=1, plant mc_20261024, hip pairs excluded)
#   2 3-iteration training smoke from a copy of s2_v4/latest.pt with the s2_v5 flags (64 envs; replay prefilled with random
#     samples because 3 iterations of 24 steps finish no episode): planner updating, eta rising, balanced update shares
#   3 export dry-run of the smoke checkpoint (export_pair.sh into PAIR_OUT, pair_eval with clamped actions) and the eval's
#     hand-over check (acp_handover_check.py) on it
# Out: runs/s2_v5/preflight/ (logs, smoke run, pair), preflight.txt summary, DONE marker. PREFLIGHT_OK is written by the agent
# after reading the summary (simcheck compared with the running build's report acp_s2h_fix1.json).
set -u
T=/mnt/filesystem-g4/tracks/acp; R=$T/runs/s2_v5; D=$R/preflight; PY=$HOME/acp_env/bin/python; SUM=$D/preflight.txt
mkdir -p "$D"; rm -f "$D/DONE"; : > "$SUM"
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 ACP_PLANT=mc_20261024 X2_EXCLUDE_PAIRS=hip ACP_HF_FIX=1 ACP_PAPER_DR=1
cd "$T/code"
# 1 simcheck
SIMCHECK_CORES=20-23 bash /mnt/filesystem-g4/tools/simcheck/run_preflight.sh acp s2v5 --cfg cfg/stage2h_sets.json --hf-fix 1 > "$D/simcheck.out" 2>&1
echo "simcheck rc=$? report /mnt/filesystem-g4/tools/simcheck/reports/acp_s2v5.json" >> "$SUM"
# 2 smoke
SM=$D/smoke_s2_v5; rm -rf "$SM"; mkdir -p "$SM"; cp "$T/runs/s2_v4/latest.pt" "$D/init_s2_v4_latest.pt"
read IT0 O0 W0 < <("$PY" - "$D/init_s2_v4_latest.pt" <<'EOF'
import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
it, s2, h = int(ck["it"]), int(ck["s2_start"]), int(ck.get("eta_hold", 0)); u = it - s2 - h
print(it, round(min(0.8, max(0.0, (u - 200) / 3000)), 4), round(min(1.0, max(0.0, (u - 1700) / 3000)), 4))
EOF
)
echo "smoke anchor it $IT0 eta_obs0 $O0 eta_rew0 $W0" >> "$SUM"
ACP_SMOKE_PREFILL=2000 OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 timeout 3000 nice -n 15 taskset -c 12-19 "$PY" -m acp.train --data cfg/stage2h_sets.json \
  --run "$SM" --envs 64 --iters $((IT0 + 3)) --njmax 450 --device cpu --heartbeat "$D/hb.json" --rec-envs 64 --replay 100000 \
  --stage 2 --init "$D/init_s2_v4_latest.pt" --planner-arch attn --u-obs 200 --u-rew 1700 --n-ann 3000 --eta-anchor-it "$IT0" \
  --eta-obs0 "$O0" --eta-rew0 "$W0" --replay-balance set_clip --popt-reset --probe-envs 8 --gate-signal probe --ramp-gate 0 > "$D/smoke.out" 2>&1
echo "smoke rc=$?" >> "$SUM"
"$PY" - "$SM" "$D/init_s2_v4_latest.pt" "$T/eval/pair_s2_v2_005750/tracker_ckpt_005750.pt" $((IT0 + 3)) >> "$SUM" 2>&1 <<'EOF'
import json, sys, torch
sm, init, p5750, last = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
rows = [json.loads(l) for l in open(sm + "/metrics.jsonl")]
for r in rows:
    print("row it", r["it"], "eta_obs", r["stage2/eta_obs"], "eta_rew", r["stage2/eta_rew"], "eta_hold", r["stage2/eta_hold"],
          "planner/updates", r["planner/updates"], "loss_aff", r["planner/loss_affordance"], "loss_occ", r["planner/loss_occupancy"])
u = {k[len("planner/update_share/"):]: v for k, v in rows[-1].items() if k.startswith("planner/update_share/")}
print("update share per set (last row):", json.dumps(u))
a = torch.load(init, map_location="cpu", weights_only=False)["planner"]; b = torch.load(p5750, map_location="cpu", weights_only=False)["planner"]
c = torch.load(f"{sm}/ckpt_{last:06d}.pt", map_location="cpu", weights_only=False)["planner"]
d0 = max(float((a[k].float() - b[k].float()).abs().max()) for k in a); d1 = max(float((c[k].float() - a[k].float()).abs().max()) for k in a)
print(f"init planner vs 5,750 planner max |diff| {d0:.3g} (0 = s2_v4 carried the frozen 5,750 planner)")
print(f"planner after 3 iterations vs init max |diff| {d1:.3g} (> 0 = updated)")
ok = all(r["planner/updates"] == 10 for r in rows) and all(rows[i + 1]["stage2/eta_obs"] > rows[i]["stage2/eta_obs"] for i in range(len(rows) - 1)) \
     and d1 > 0 and len(rows) == 3 and len(u) > 0
print("SMOKE", "PASS" if ok else "FAIL")
EOF
# 3 export dry-run + hand-over check
P=$D/pair_dry; rm -rf "$P"
PAIR_OUT=$P bash "$T/code/ops/export_pair.sh" "$SM" $((IT0 + 3)) > "$D/export.out" 2>&1
echo "export rc=$? DONE=$([ -f "$P/DONE" ] && echo yes || echo no)" >> "$SUM"
PYTHONPATH=/mnt/filesystem-g4/eval/code nice -n 15 "$PY" /mnt/filesystem-g4/eval/code/tools/acp_handover_check.py "$P" > "$D/handover_check.out" 2>&1
echo "handover_check rc=$? last: $(tail -3 "$D/handover_check.out" | tr '\n' ' ' | cut -c1-400)" >> "$SUM"
date -u +%FT%TZ > "$D/DONE"
