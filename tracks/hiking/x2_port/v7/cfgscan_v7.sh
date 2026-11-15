#!/bin/bash
# CPU scan of v7 (= cfgscan_v6.sh with continue_v7.sh; same staged.py.v6)'s final env cfg (v7/cfgscan.py = v5b/cfgscan.py + a reward-term listing) on an overlay copy of the installed
# instinctlab with staged.py replaced by variants/staged.py.v6, env = ENVV of continue_v6.sh. Also requires the touchdown term to
# be feet_air_touchdown_refcap with weight 140. usage: cfgscan_v7.sh <log>; exit 0 = pass.
T=/mnt/filesystem-g4/tracks/hiking; X=$T/code/x2_port; LOG=${1:-/tmp/cfgscan_v7.log}; S=$T/runs/x2_parkour_v4/_supervisor
PKG=$T/code/hiking-in-the-wild/InstinctLab/source/instinctlab/instinctlab
O=$(mktemp -d /tmp/cfgscan_XXXX); cp -r $PKG $O/instinctlab; C=$O/instinctlab/tasks/parkour/config/x2
cp $X/variants/staged.py.v6 $C/staged.py
ENVV=$(grep -o '^ENVV="[^"]*"' $X/v7/continue_v7.sh | head -1 | sed "s/^ENVV=//; s/\"//g; s#\$T#$T#g; s#\$S#$S#g")
cd $O; env $ENVV PYTHONPATH=$O timeout 600 $HOME/hiking-env/bin/python $X/v7/cfgscan.py \
  instinctlab.tasks.parkour.config.x2.extended:X2ParkourExtendedFixedEnvCfg > $LOG 2>&1; RC=$?
grep -aq "^cfgscan:" $LOG || RC=1
grep -aq "^reward feet_air_touchdown feet_air_touchdown_refcap 140.0" $LOG || { echo "FAIL touchdown term is not refcap x 140" >> $LOG; RC=1; }
rm -rf $O; exit $RC
