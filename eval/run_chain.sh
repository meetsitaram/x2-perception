#!/bin/bash
# A bounded chain of ladder slices: at most MAX slices of at most 9.5 minutes each, stopping as soon as a slice finishes its list.
#   bash eval/run_chain.sh MAX OUT STEP TRIALS ADAPTER CKPT|- test1 test2 ...      (run detached; progress in OUT/stepN.log)
cd "$(dirname "$0")/.."
MAX=$1; shift
for i in $(seq 1 $MAX); do
  out=$(bash eval/run_ladder.sh "$@" 2>&1)
  echo "$out" | grep -q "group done (code 0)" && { echo "CHAIN DONE after $i slices"; exit 0; }
  [ -f eval/STOP ] && { echo "STOP file found"; exit 0; }
done
echo "CHAIN ENDED at the slice limit ($MAX) without finishing"
