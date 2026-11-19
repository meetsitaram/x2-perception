#!/bin/bash
# PC2: end an x2-perception session: stop our tmux sessions (the node first: the robot goes unpowered, so it must be on the
# gantry or held), restore the vendor face, and with --start-mc restart MC (aima em start-app mc) for normal operation.
set -u
X2P=/home/run/x2-perception
LOG=$X2P/log/ritual_fired.log
for s in x2p_deploy x2p_face x2p_pad x2p_depth; do
  tmux has-session -t $s 2>/dev/null && { tmux send-keys -t $s C-c; sleep 1; tmux kill-session -t $s 2>/dev/null; echo "$(date +%F_%T) $s: STOPPED" >> $LOG; }
done
/home/run/gear-sonic/interact/x2_face.sh off >/dev/null 2>&1
if [ "${1:-}" = "--start-mc" ]; then
  aima em start-app mc >> $LOG 2>&1 && echo "$(date +%F_%T) MC started" >> $LOG
fi
echo "$(date +%F_%T) X2-PERCEPTION STOPPED" >> $LOG
