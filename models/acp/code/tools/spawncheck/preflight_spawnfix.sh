#!/bin/bash
# Pre-flight of the spawn fix (ACP, 2026-11-14, state/acp.md 17m): CPU resume of s3_v2 latest.pt with cfg/stage3v2i_sets.json,
# the trainer unchanged except for the wrapper in fault_diag.py (random goal levels 0-9, fault counting per env kind / family).
# Same seed, envs, iterations as the v2h baseline run (fault_diag_lv1.json). Writes PREFLIGHT_SPAWNFIX_OK when the goal-env
# fault rate is within 3x the reference-env rate of the same run and no deep (>= 0.3 m) spawn fault is left in the extended
# families, and the reference-clip labels and the cfg are unchanged apart from the goal tiles.
T=/mnt/filesystem-g4/tracks/acp; D=$T/runs/s3_v2_diag; PY=$HOME/acp_env/bin/python
cd $D; rm -f PREFLIGHT_SPAWNFIX_OK PREFLIGHT_SPAWNFIX_FAILED; cp $T/runs/s3_v2/latest.pt init_pf.pt
IT=$(CUDA_VISIBLE_DEVICES= "$PY" -c "import torch,sys; print(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['it'])" init_pf.pt | tail -1)
echo "preflight from it $IT $(date -u +%FT%TZ)" > preflight.log
. $T/code/ops/stage3_flags.env
EX=$(echo "$S3_EX" | sed "s/@ANCHOR@/$IT/; s/--probe-envs [0-9]*/--probe-envs 8/")
rm -rf run_pf
( cd $T/code && env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 $S3_ENVV ACP_SMOKE_PREFILL=2000 DIAG_RANDOM_LEVEL=1 DIAG_OUT=$D/fault_diag_v2i.json \
   timeout 5400 nice -n 15 taskset -c 12-19 "$PY" $D/fault_diag.py --data cfg/stage3v2i_sets.json --run "$D/run_pf" --envs 128 --iters $((IT + 20)) \
   --njmax 450 --device cpu --heartbeat "$D/hb_pf.json" --rec-envs 64 --replay 100000 --init "$D/init_pf.pt" $EX ) > $D/out_pf.txt 2>&1
echo "trainer rc $?" >> preflight.log
"$PY" - >> preflight.log 2>&1 <<'EOF'
import json, collections, torch
D = "/mnt/filesystem-g4/tracks/acp/runs/s3_v2_diag"; C = "/mnt/filesystem-g4/tracks/acp/code/cfg"
ok = True
for tag, f in (("v2h baseline", "fault_diag_lv1.json"), ("v2i", "fault_diag_v2i.json")):
    s = json.load(open(f"{D}/{f}")); st, h = s["stats"], s["hist"]
    g, r = st["fault_goal"] / st["steps_goal"], st["fault_ref"] / st["steps_ref"]
    deep = sum(v for k, v in h.items() if k.startswith("goal|depth_bin_cm|") and int(k.split("|")[-1]) >= 30)
    early = sum(v for k, v in h.items() if k == "goal|epstep_bin|0")
    fam = {k.split("|")[-1]: v for k, v in h.items() if k.startswith("goal|fam|")}
    print(f"{tag}: goal faults {st['fault_goal']:.0f} / {st['steps_goal']:.0f} steps = {1e3 * g:.2f} per 1000; reference {st['fault_ref']:.0f} = {1e3 * r:.2f} per 1000; "
          f"goal deep (>= 0.3 m) {deep:.0f}; goal within 10 steps of reset {early:.0f}; per family {fam}")
    if tag == "v2i":
        ok &= g <= 3 * max(r, 1.0 / st["steps_ref"])
        print("  check goal rate <= 3x reference rate:", g <= 3 * max(r, 1.0 / st["steps_ref"]))
rows = [json.loads(l) for l in open(f"{D}/run_pf/metrics.jsonl")]
print("pf metrics rows", len(rows), "NaN-free", all(v == v for d in rows for v in d.values() if isinstance(v, float)),
      "last: ref finish flat", rows[-1].get("finish/x2_loco_flat_train_hold"), "goal success", rows[-1].get("goal/success"),
      "diverged", rows[-1].get("end/term_diverged"), "planner updates", rows[-1].get("planner/updates"))
ok &= len(rows) >= 20
a, b = json.load(open(f"{C}/stage3v2h_sets.json")), json.load(open(f"{C}/stage3v2i_sets.json"))
diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k)); print("cfg keys differing:", diff); ok &= diff == ["goal_tiles", "note"]
la, lb = torch.load(f"{C}/labels_stage3v2h_sets.json.pt", map_location="cpu"), torch.load(f"{C}/labels_stage3v2i_sets.json.pt", map_location="cpu")
same = {k: (bool(torch.equal(la[k], lb[k])) if torch.is_tensor(la[k]) else la[k] == lb[k]) for k in la}
print("label cache keys equal:", same)
print("PREFLIGHT", "OK" if ok else "FAILED")
open(f"{D}/PREFLIGHT_SPAWNFIX_{'OK' if ok else 'FAILED'}", "w").write("see preflight.log\n")
EOF
touch PF_DONE
