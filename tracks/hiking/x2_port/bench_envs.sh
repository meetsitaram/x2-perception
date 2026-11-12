#!/bin/bash
# Throughput benchmark for the X2 task at several robot counts. Waits for the given checkpoint of the running job, stops the
# supervisor (nothing lost: it resumes from that checkpoint), then runs short trainings one at a time into runs/_bench.
#   bash bench_envs.sh <run name> <checkpoint iteration to wait for> <robot counts...>
T=/mnt/filesystem-g4/tracks/hiking
RUN=$T/runs/$1; WAIT_IT=$2; shift 2
OUT=$T/runs/_bench; mkdir -p $OUT
PY=/home/NODE_USER/hiking-env/bin/python
echo "$(date +%T) waiting for model_$WAIT_IT.pt" | tee -a $OUT/bench.log
until ls $RUN/*/model_$WAIT_IT.pt >/dev/null 2>&1; do sleep 10; done
sleep 5
touch $RUN/_supervisor/STOP
while tmux has -t hiking_x2 2>/dev/null; do sleep 5; done
while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 5; done
echo "$(date +%T) training stopped at model_$WAIT_IT" | tee -a $OUT/bench.log
cd $T/code/hiking-in-the-wild
for N in "$@"; do
  L=$OUT/envs_$N.log
  ( while true; do nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits; sleep 2; done ) > $OUT/gpu_$N.csv &
  MON=$!
  ( while true; do top -b -n 1 | awk 'NR>7 && $9>20 {printf "%s:%s ", $12, $9} END {print ""}'; sleep 5; done ) > $OUT/cpu_$N.txt &
  MON2=$!
  OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 timeout 1200 $PY run_train.py --headless --task=Instinct-Parkour-Target-Amp-X2-v0 \
    --num_envs $N --max_iterations 12 --logroot $OUT/envs_$N > $L 2>&1
  kill $MON $MON2
  echo "$(date +%T) envs $N: exit $? | steps/s (last 6): $(grep -a 'Computation' $L | tail -6 | awk '{printf "%s ", $2}') | collection/learning: $(grep -a 'Computation' $L | tail -1 | sed 's/.*(//;s/).*//') | iter time: $(grep -a 'Iteration time' $L | tail -1 | awk '{print $3}') | GPU mem peak MiB: $(cut -d, -f1 $OUT/gpu_$N.csv | sort -n | tail -1) | GPU util median: $(cut -d, -f2 $OUT/gpu_$N.csv | sort -n | awk '{a[NR]=$1} END {print a[int(NR*0.7)]}') | errors: $(grep -ac 'Traceback\|out of memory\|PhysX error' $L)" | tee -a $OUT/bench.log
  rm -rf $OUT/envs_$N
done
echo "$(date +%T) BENCH DONE" | tee -a $OUT/bench.log
