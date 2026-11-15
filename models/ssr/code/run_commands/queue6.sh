#!/bin/bash
# r6 (coordinator, 2026-11-12 01:50): full SSR run with t_a's staged schedule (tracking + posture only until 150, the other Table 6
# terms ramped in by 400), style weight held at 0 until mean touchdown air time > 0.15 s for 5 iterations, then ramped to the paper's
# 0.2 over 500 iterations; terrain curriculum on; plant mc_20261024; camera chest_rs with mount randomisation; 8,192 robots.
# Gates: 40% of commanded speed by 1,000; air time over 0.2 s by 2,000. Departures from the paper: the staged schedule and the style latch.
# Waits for queue4 (t_a probe) and queue5 (stopped by QUEUE5_STOP) to exit. Stop: touch runs/STOP.
cd /mnt/filesystem-g4/tracks/ssr/code
# (no wait: started by hand when the GPU was free; a pgrep wait matched the tmux server command line and never ended)
echo "QUEUE6 $(date -u +%F' '%T) start r6"
../env/venv/bin/python scripts/supervise_node.py --name r6 --iters 20000 --num_envs 8192 -- --stage_a 150 --stage_b 400 --w_style 0.2 --style_low 0 --gate_air 0.15 --gate_ramp 500 --prior_min_speed 0.2 --rsi_min_speed 0.3 --plant mc_20261024 --camera chest_rs --amp_file data/amp_v2.npz --prior_balance
echo "QUEUE6 $(date -u +%F' '%T) end r6"
