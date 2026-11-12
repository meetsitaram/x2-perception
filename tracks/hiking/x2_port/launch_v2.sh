#!/bin/bash
# Stop run 1, check the run-2 task once in Isaac (gains read back, depth frames dumped), start run 2 from scratch.
# If the check fails, run 1 is restarted so the GPU is not left idle.
T=/mnt/filesystem-g4/tracks/hiking; PY=/home/NODE_USER/hiking-env/bin/python; OUT=$T/runs/_smoke_v2; mkdir -p $OUT
TASK=Instinct-Parkour-TorsoCam-Amp-X2-v0
bash $T/code/x2_port/apply.sh > /dev/null
T0=$(date +%s)
touch $T/runs/x2_parkour_v1/_supervisor/STOP
while tmux has -t hiking_x2 2>/dev/null; do sleep 3; done
sleep 3  # (the kill-every-GPU-process line was removed 2026-11-12: the supervisor stops its own training process)
echo "{\"track\": \"hiking\", \"t\": $T0, \"when\": \"$(date '+%Y-%m-%d %H:%M:%S')\", \"state\": \"run 1 stopped; checking the run-2 task (new gains, torso camera) before starting run 2\"}" > /mnt/filesystem-g4/status/hiking.json
cd $T/code/hiking-in-the-wild/InstinctLab
OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 timeout 420 $PY $T/code/x2_port/check_x2_env.py --headless --task $TASK --num_envs 64 --steps 110 --out $OUT > $OUT/check.log 2>&1
sleep 3  # (the kill-every-GPU-process line was removed 2026-11-12: the supervisor stops its own training process)
if grep -q "CHECK DONE" $OUT/check.log; then
  tmux new -d -s hiking_x2 "$PY $T/code/x2_port/supervise_x2.py --task $TASK --name x2_parkour_v2 --total 30000 --num_envs 8192 2>&1 | tee -a $T/runs/x2_parkour_v2_supervisor_tmux.log"
  echo "$(date +%T) run 2 started, GPU idle $(( $(date +%s) - T0 )) s" >> $OUT/launch.log
else
  bash $T/code/x2_port/start_x2_parkour_v1.sh
  echo "$(date +%T) CHECK FAILED, run 1 restarted, GPU idle $(( $(date +%s) - T0 )) s" >> $OUT/launch.log
fi
