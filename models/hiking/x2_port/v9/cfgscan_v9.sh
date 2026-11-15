#!/bin/bash
# v9 (2026-11-14): CPU scan of v9's final env cfg; derived from cfgscan_v8b.sh. Original header: CPU scan of v8's final env cfg (v8/cfgscan.py = v7/cfgscan.py + curriculum / command listing) on an overlay copy of the installed
# instinctlab with v8/fair.py and v8/extended.py.v8 added, env = ENVV of v8/continue_v8.sh. Requires: touchdown term
# feet_air_touchdown_refcap_progress x 140, curriculum terrain_levels = tracking_exp_vel_paper_cmd, command class
# FairSlowRouteFarTargetPoseVelocityCommand. usage: cfgscan_v8.sh <log>; exit 0 = pass.
T=/mnt/filesystem-g4/tracks/hiking; X=$T/code/x2_port; LOG=${1:-/tmp/cfgscan_v9.log}; S=$T/runs/x2_parkour_v4/_supervisor
PKG=$T/code/hiking-in-the-wild/InstinctLab/source/instinctlab/instinctlab
O=$(mktemp -d /tmp/cfgscan_XXXX); cp -r $PKG $O/instinctlab; C=$O/instinctlab/tasks/parkour/config/x2
cp $X/variants/staged.py.v6 $C/staged.py; cp $X/v8/fair.py $C/fair.py; cp $X/v9/v9.py $C/v9.py; cp $X/v9/extended.py.v9 $C/extended.py
ENVV=$(grep -o '^ENVV="[^"]*"' $X/v9/continue_v9.sh | head -1 | sed "s/^ENVV=//; s/\"//g; s#\$T#$T#g; s#\$S#$S#g")
cd $O; env $ENVV PYTHONPATH=$O timeout 600 $HOME/hiking-env/bin/python $X/v8/cfgscan.py \
  instinctlab.tasks.parkour.config.x2.extended:X2ParkourExtendedFixedEnvCfg > $LOG 2>&1; RC=$?
grep -aq "^cfgscan:" $LOG || RC=1
grep -aq "^reward feet_air_touchdown" $LOG && { echo "FAIL touchdown term still present" >> $LOG; RC=1; }
grep -aq "^reward feet_stumble feet_stumble -2.0" $LOG || { echo "FAIL feet_stumble is not the v8 term x -2.0" >> $LOG; RC=1; }
grep -aq "^reward wall_contact_graded wall_contact_graded -1.0" $LOG || { echo "FAIL wall_contact_graded is not the v8 term x -1.0" >> $LOG; RC=1; }
grep -aq "^reward feet_air_time feet_air_time 0.5" $LOG || { echo "FAIL paper feet_air_time 0.5 missing" >> $LOG; RC=1; }
grep -aq "^curriculum terrain_levels tracking_exp_vel_paper_cmd_v9" $LOG || { echo "FAIL curriculum terrain_levels is not tracking_exp_vel_paper_cmd_v9" >> $LOG; RC=1; }
grep -aq "^curriculum by_type by_type" $LOG || { echo "FAIL by_type curriculum log missing" >> $LOG; RC=1; }
grep -aq "^command V9CommandCfg V9Command" $LOG || { echo "FAIL command class is not V9Command" >> $LOG; RC=1; }
rm -rf $O; exit $RC
