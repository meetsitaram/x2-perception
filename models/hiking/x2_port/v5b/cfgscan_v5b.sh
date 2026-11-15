#!/bin/bash
# CPU scan of v5b's final env cfg (cfgscan.py) on an overlay copy of instinctlab with the v5b file swaps of continue_v5b.sh
# (.v4k: extended routes keep contact slow; .v5: staged fixed). usage: cfgscan_v5b.sh <log>; exit code of cfgscan.py.
T=/mnt/filesystem-g4/tracks/hiking; X=$T/code/x2_port; LOG=${1:-/tmp/cfgscan_v5b.log}
PKG=$T/code/hiking-in-the-wild/InstinctLab/source/instinctlab/instinctlab
O=$(mktemp -d /tmp/cfgscan_XXXX); cp -r $PKG $O/instinctlab; C=$O/instinctlab/tasks/parkour/config/x2
for f in extended.py routes.py keep.py contact.py slow.py; do cp $C/$f.v4k $C/$f; done
for f in staged.py fixed.py; do cp $C/$f.v5 $C/$f; done
ENVV=$(grep -o '^ENVV="[^"]*"' $X/v5b/continue_v5b.sh | head -1 | sed "s/^ENVV=//; s/\"//g; s#\$T#$T#g")
cd $O; env $ENVV PYTHONPATH=$O timeout 600 $HOME/hiking-env/bin/python $X/v5b/cfgscan.py \
  instinctlab.tasks.parkour.config.x2.extended:X2ParkourExtendedFixedEnvCfg > $LOG 2>&1; RC=$?
grep -aq "^cfgscan:" $LOG || RC=1
rm -rf $O; exit $RC
