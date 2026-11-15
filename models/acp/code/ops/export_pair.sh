#!/usr/bin/env bash
# Pair export for the eval monitor, CPU only (CUDA hidden), one numbered checkpoint of a stage-2 or stage-3 run:
# tracker actor.npz + the checkpoint's own online planner as ONNX + adapter_card.json + obs_check.npz (recorded from the
# training simulator, MuJoCo Warp on the CPU, by tools/handover.py). Out: eval/pair_<run>_<it>/ ; DONE marker when complete.
#   bash export_pair.sh RUN_DIR IT
set -u
T=/mnt/filesystem-g4/tracks/acp; PY=$HOME/acp_env/bin/python
RUN=$1; IT=$2; P=$(printf %06d "$IT"); NAME=$(basename "$RUN"); CK=$RUN/ckpt_$P.pt
D=${PAIR_OUT:-$T/eval/pair_${NAME}_${P}}; mkdir -p "$D"; [ -f "$D/DONE" ] && exit 0
export CUDA_VISIBLE_DEVICES= ACP_PLANT=mc_20261024
cd "${ACP_CODE:-$T/code}"
GR=0.3; case "$NAME" in s3_*) GR=0.2;; esac
cp "$CK" "$D/tracker_ckpt_$P.pt"
export X2_EXCLUDE_PAIRS=$("$PY" -c "import torch,sys; print(torch.load(sys.argv[1], map_location='cpu', weights_only=False).get('x2_exclude_pairs') or '')" "$D/tracker_ckpt_$P.pt" 2>/dev/null | tail -1)   # record obs_check on the body the tracker was trained on
echo "body: X2_EXCLUDE_PAIRS=$X2_EXCLUDE_PAIRS"
"$PY" - "$D/tracker_ckpt_$P.pt" "$D/planner_attn.pt" <<'PY' || exit 1
import sys, torch
from acp import planner as P
import os
ck = torch.load(os.environ.get("PLANNER_FROM") or sys.argv[1], map_location="cpu", weights_only=False)   # PLANNER_FROM: a stage-1 tracker paired with a kept planner
assert ck.get("planner") is not None, "checkpoint has no planner"
arch = ck.get("planner_arch") or "attn"; hd = 5 * 99
net = P.build(arch, hd); net.load_state_dict(ck["planner"])
torch.save(dict(net=net.state_dict(), arch=arch, hist_dim=hd, parameters=sum(p.numel() for p in net.parameters()),
                stats={"note": "online stage-%s planner, trained beside the tracker (hindsight buffer); no held-out split" % ck.get("stage")},
                data=ck.get("data"), tracker_iteration=ck["it"]), sys.argv[2])
print("planner", arch, "iteration", ck["it"], "stage", ck.get("stage"))
PY
"$PY" -m acp.planner export "$D/planner_attn.pt" "$D/planner_attn.onnx" || exit 1
timeout 1800 nice -n 19 "$PY" tools/handover.py "$D/tracker_ckpt_$P.pt" "$D" > "$D/handover.out" 2>&1 || { echo "handover failed"; exit 1; }
"$PY" /mnt/filesystem-g4/eval/code/tools/export_actor.py "$D/tracker_ckpt_$P.pt" "$D/actor.npz" --meta "$D/meta.json" --goal-r $GR > "$D/export_actor.out" 2>&1 || exit 1
"$PY" tools/card_planner.py "$D" "$D/planner_attn.pt" "$D/planner_attn.onnx" > /dev/null || exit 1
"$PY" - "$D/adapter_card.json" "$NAME" "$IT" "$GR" "$D/tracker_ckpt_$P.pt" <<'PY'
import json, sys
p, name, it, gr = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]); c = json.load(open(p))
st = "3" if name.startswith("s3_") else "2"
c["label"] = (f"ACP stage-{st} pair, run {name}, iteration {it}: tracker co-trained with its own online attention planner "
              f"(planner contacts ramped in during stage 2)" + ("; stage 3: goal tiles, 0.20 m arrival disc, 1 s stand at stations"
              if st == "3" else "; no goal environments (stage 2 is reference-only): goal fades inside 0.3 m, use the reference-lookahead adapter"))
c["planner"]["trained_on"] = f"online, beside tracker iteration {it} (hindsight buffer of successful rollouts on the training terrains); no SuperMimic data"
c["goal_radius_m"] = gr
import os
if os.environ.get("PLANNER_FROM"):
    _pi = torch.load(os.environ["PLANNER_FROM"], map_location="cpu", weights_only=False)["it"]
    c["planner"]["trained_on"] = f"online, beside the old-body stage-2 tracker up to iteration {_pi} (byte-identical to the planner in " + os.path.basename(os.environ["PLANNER_FROM"]) + "); NOT trained with this tracker"
    c["planner_iteration"] = _pi
    c["label"] = f"interim pair: stage-1 tracker {name} iteration {it} (corrected body if trained_model says so) + the online planner of " + os.environ["PLANNER_FROM"] + " (trained beside the old-body stage-2 tracker); the tracker was not co-trained with this planner"
import torch
xe = torch.load(sys.argv[5], map_location="cpu", weights_only=False).get("x2_exclude_pairs") or ""
c["trained_model"] = ("data/robots/x2/mjcf/x2_ultra_hipexclude.xml (pelvis / hip-roll pairs excluded)" if "hip" in xe.split(",") else "data/robots/x2/mjcf/x2_ultra.xml (standard; pelvis / hip-roll hull contact present)")
json.dump(c, open(p, "w"), indent=1)
PY
# planner-driven closed loop (4 per set, CPU) for obs_check_pair.npz: the eval monitor checks the planner adapter with it
timeout 1500 nice -n 19 "$PY" tools/pair_eval.py "$D/tracker_ckpt_$P.pt" "$D/planner_attn.pt" "$D/pair_eval_4" 4 > "$D/pair_eval_4.out" 2>&1 && mkdir -p "$D/pair_eval_40" && cp "$D/pair_eval_4/obs_check_pair.npz" "$D/pair_eval_40/obs_check_pair.npz" || { echo "pair_eval for obs_check_pair FAILED (no DONE): $(tail -1 $D/pair_eval_4.out | cut -c1-200)"; exit 3; }
date -u +%FT%TZ > "$D/DONE"; echo "exported $D"
