#!/bin/bash
# r6b (coordinator, 2026-11-12 02:53): r6 stood with the terrain curriculum from iteration 0 (5% of commanded speed at 420; stopped at 490,
# files kept). r6b continues t_a's flat gait (t_a checkpoint, iteration 549; t_a = the same staged schedule on flat ground) with r6's
# settings, plus: flat phase until 550, then terrain types redrawn at level 0 with 25% of robots on obstacle tiles rising to the
# paper's uniform mix (80% obstacle) over 500 iterations; style held at 0 until touchdown air time > 0.15 s for 5 iterations, then
# ramped to the paper's 0.2 over 500. Gate: 40% of commanded speed by 1,000; air time over 0.2 s by 2,000. Stop: touch runs/STOP.
R=/mnt/filesystem-g4/tracks/ssr/runs
cd /mnt/filesystem-g4/tracks/ssr/code
[ -f $R/r6b/last.pt ] || { mkdir -p $R/r6b && cp $R/t_a/last.pt $R/r6b/last.pt; }
echo "QUEUE8 $(date -u +%F' '%T) start r6b"
../env/venv/bin/python scripts/supervise_node.py --name r6b --iters 20000 --num_envs 8192 -- --flat_until 550 --obst0 0.25 --obst_ramp 500 --stage_a 150 --stage_b 400 --w_style 0.2 --style_low 0 --gate_air 0.15 --gate_ramp 500 --prior_min_speed 0.2 --rsi_min_speed 0.3 --plant mc_20261024 --camera chest_rs --amp_file data/amp_v2.npz --prior_balance
echo "QUEUE8 $(date -u +%F' '%T) end r6b"
