#!/usr/bin/env bash
# Pre-flight of the user-schedule change to s2_v5 (17k): 3-iteration CPU smoke with N_ann 6,000 from s2_v4 latest; eta must rise by
# 1/6,000 per iteration from the anchor values; schedule arithmetic as in arm_s2v5.sh. CPU only, niced, cores 12-19.
set -u
T=/mnt/filesystem-g4/tracks/acp; R=$T/runs/s2_v5; D=$R/preflight_sched; PY=$HOME/acp_env/bin/python; SUM=$D/preflight.txt
mkdir -p "$D"; rm -f "$D/DONE"; : > "$SUM"
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 ACP_PLANT=mc_20261024 X2_EXCLUDE_PAIRS=hip ACP_HF_FIX=1 ACP_PAPER_DR=1
cd "$T/code"
SM=$D/smoke; rm -rf "$SM"; mkdir -p "$SM"; cp "$T/runs/s2_v4/latest.pt" "$D/init.pt"
read IT0 O0 W0 < <("$PY" - "$D/init.pt" <<'EOF'
import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
it, s2, h = int(ck["it"]), int(ck["s2_start"]), int(ck.get("eta_hold", 0)); u = it - s2 - h
print(it, round(min(0.8, max(0.0, (u - 200) / 3000)), 4), round(min(1.0, max(0.0, (u - 1700) / 3000)), 4))
EOF
)
read OBS1 REW1 < <(python3 -c "import math;print($IT0+math.ceil(round((1-$O0)*6000,6)), $IT0+math.ceil(round((1-$W0)*6000,6)))")
echo "anchor it $IT0 eta_obs0 $O0 eta_rew0 $W0 -> eta_obs=1 at $OBS1, eta_rew=1 at $REW1, stage-3 start $((REW1 + 2300)), run end $((REW1 + 2300 + 15000))" >> "$SUM"
ACP_SMOKE_PREFILL=2000 OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 timeout 3000 nice -n 15 taskset -c 12-19 "$PY" -m acp.train --data cfg/stage2h_sets.json \
  --run "$SM" --envs 64 --iters $((IT0 + 3)) --njmax 450 --device cpu --heartbeat "$D/hb.json" --rec-envs 64 --replay 100000 \
  --stage 2 --init "$D/init.pt" --planner-arch attn --u-obs 200 --u-rew 1700 --n-ann 6000 --eta-anchor-it "$IT0" \
  --eta-obs0 "$O0" --eta-rew0 "$W0" --replay-balance set_clip --popt-reset --probe-envs 8 --gate-signal probe --ramp-gate 0 > "$D/smoke.out" 2>&1
echo "smoke rc=$?" >> "$SUM"
grep "^\[schedule\]" "$D/smoke.out" | cut -c1-250 >> "$SUM"
"$PY" - "$SM" "$IT0" "$O0" "$W0" >> "$SUM" 2>&1 <<'EOF'
import json, sys
sm, it0, o0, w0 = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
rows = [json.loads(l) for l in open(sm + "/metrics.jsonl")]
ok = len(rows) == 3
for r in rows:
    k = r["it"] - 1 - it0   # the row of iteration n reports the eta used in loop index n-1
    eo, er = o0 + k / 6000, w0 + k / 6000
    good = abs(r["stage2/eta_obs"] - eo) < 2e-4 and abs(r["stage2/eta_rew"] - er) < 2e-4 and r["planner/updates"] == 10
    ok &= good
    print("row it", r["it"], "eta_obs", r["stage2/eta_obs"], "expected", round(eo, 5), "eta_rew", r["stage2/eta_rew"], "expected", round(er, 5),
          "planner/updates", r["planner/updates"], "ok" if good else "MISMATCH")
d = [rows[i + 1]["stage2/eta_obs"] - rows[i]["stage2/eta_obs"] for i in range(len(rows) - 1)]
print("eta_obs step per iteration", [round(x, 6) for x in d], "expected", round(1 / 6000, 6))
print("SMOKE", "PASS" if ok else "FAIL")
EOF
bash -n "$T/code/ops/arm_s2v5.sh" && bash -n "$T/code/ops/watch_s2v5.sh" && bash -n "$T/code/ops/auto_stage3.sh" && echo "scripts syntax OK" >> "$SUM"
date -u +%FT%TZ > "$D/DONE"
