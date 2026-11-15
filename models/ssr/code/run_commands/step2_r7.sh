#!/bin/bash
# Step 2 (prepared 2026-11-12 05:10, NOT started): fresh run r7, default leg pose = reference mean (--default_from_ref),
# (pelvis/hip-roll exclusion removed 05:58: the eval model keeps those contacts until checked.) Otherwise r6b final command from iteration 0 (flat phase to 550, obstacle ramp, style by
# schedule from 1,000, feet-air reward and arm-masked discriminators from 0, unlike r6b where they came at 1,000).
# Start only after r6c has ended or been stopped (one GPU job). Start: tmux new-session -d -s ssr7r "bash <this file>"
cd /mnt/filesystem-g4/tracks/ssr/code
pgrep -f "ssr.train" > /dev/null && { echo "a trainer is running; not starting r7"; exit 1; }
../env/venv/bin/python scripts/supervise_node.py --name r7 --iters 20000 --num_envs 8192 -- --flat_until 550 --obst0 0.25 --obst_ramp 500 --stage_a 150 --stage_b 400 --w_style 0.2 --style_low 0 --gate_ramp 500 --prior_min_speed 0.2 --rsi_min_speed 0.3 --plant mc_20261024 --camera chest_rs --amp_file data/amp_v2.npz --prior_balance --disc_mask_arms --style_from 1000 --feet_air_w 20 --feet_air_min 0.1 --gate_air 0.15 --default_from_ref --sig_v0 0.1 --cmd_cur_iters 20000 --rsi_min_speed 0.15
