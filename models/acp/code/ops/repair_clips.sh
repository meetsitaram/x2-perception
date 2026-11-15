#!/usr/bin/env bash
cd /mnt/filesystem-g4/tracks/acp/code; export CUDA_VISIBLE_DEVICES=; PY=$HOME/acp_env/bin/python
nice -n 5 $PY tools/repair_legs.py /mnt/filesystem-g4/tracks/acp/data/clips/gs2dn_v1.pkl /mnt/filesystem-g4/tracks/acp/data/clips/gs2dn_v2_resolved.pkl
nice -n 5 $PY tools/repair_legs.py /mnt/filesystem-g4/tracks/acp/data/clips/pico_2026-10-29_2x2_train.pkl /mnt/filesystem-g4/tracks/acp/data/clips/pico_2026-10-29_2x2_train_v2_resolved.pkl --only-flagged
nice -n 5 $PY tools/repair_legs.py /mnt/filesystem-g4/tracks/acp/data/clips/gst_v1.pkl /mnt/filesystem-g4/tracks/acp/data/clips/gst_v2_resolved.pkl --only-flagged
echo REPAIR_DONE
