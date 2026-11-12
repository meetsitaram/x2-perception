#!/bin/bash
# Recipe tests, second batch (coordinator, 2026-11-12 ~01:05 UTC): t_e first, then t_c, t_d. One GPU job at a time, each under the
# supervisor and followed by a gait probe of its last checkpoint. Waits for queue4.sh (t_a and its probe) to exit first.
# Stop between jobs: touch runs/QUEUE5_STOP (and runs/STOP for the running one).
cd /mnt/filesystem-g4/tracks/ssr/code
R=/mnt/filesystem-g4/tracks/ssr/runs
PY=../env/venv/bin/python
while pgrep -f "runs/queue4.sh" > /dev/null; do sleep 30; done
COMMON="--flat_only --prior_min_speed 0.2 --rsi_min_speed 0.3 --plant mc_20261024 --camera chest_rs --amp_file data/amp_v2.npz"
run() {  # name iters flags...
  name=$1; iters=$2; shift 2
  [ -f $R/QUEUE5_STOP ] && exit 0
  echo "QUEUE5 $(date -u +%F' '%T) start $name $iters $*"
  $PY scripts/supervise_node.py --name $name --iters $iters --num_envs 8192 -- $COMMON "$@"
  for f in "" "--noise"; do X2_PLANT=mc_20261024 timeout 300 $PY scripts/probe_gait.py --ckpt $R/$name/last.pt --out ../eval/$name --speed 0.6 $f > $R/${name}_probe.log 2>&1; done
  echo "QUEUE5 $(date -u +%F' '%T) end $name"
}
# t_e: posture/alive-like terms (orientation, yaw kernel, base height, roll/pitch rate, lateral) x forward-progress gate for robots with a
# moving command; gate ramps off over 300 iterations once mean touchdown air time > 0.2 s for 5 iterations (300 not 500 so the switch-back is seen inside the test).
run t_e 650 --w_style 0 --prog_gate 1.0 --gate_air 0.2 --gate_ramp 300
# t_f (coordinator, 01:15): t_e plus style weight held at 0 until the same air-time latch, then ramped to the paper's 0.2 over 300 iterations.
# Only if t_e reached the latch: before it, t_f is identical to t_e (and a style-0 full table alone is d4, which stood).
if grep -q '"gate_start": [0-9]' $R/t_e/log.jsonl 2>/dev/null; then
  run t_f 650 --prog_gate 1.0 --gate_air 0.2 --gate_ramp 300 --w_style 0.2 --style_low 0
else echo "QUEUE5 t_e never reached the air-time latch: t_f skipped"; fi
run t_c 450 --w_style 0 --stuck_term --stuck_until 300
run t_d 300 --w_style 0 --ent 0.001 --init_std 0.5
