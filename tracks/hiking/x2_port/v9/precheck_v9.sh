#!/bin/bash
# precheck_v9.sh <continue_v9.sh>: see v9/precheck_v9.py. exit 0 = pass.
$HOME/hiking-env/bin/python /mnt/filesystem-g4/tracks/hiking/code/x2_port/v9/precheck_v9.py "$1"
