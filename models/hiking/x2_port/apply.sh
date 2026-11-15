#!/bin/bash
# Copies the X2 port (new files only; nothing upstream is edited) into the InstinctLab checkout.
set -e
H=$(cd "$(dirname "$0")" && pwd)
cp -rv $H/files_instinctlab/. $H/../hiking-in-the-wild/InstinctLab/source/instinctlab/instinctlab/
