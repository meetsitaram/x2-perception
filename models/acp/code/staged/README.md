# Staged code (plain-text mirror for the repo; the scripts run from the folders named here)

- `next/` = `/mnt/filesystem-g4/tracks/acp/code_next` (installed by `ops/restart_s2b.sh` at the next needed restart):
  stage 2 + the arm clearance fix loader (`acp/data.py` `_armfix`); data cfg `cfg/stage2d_sets.json`.
- `s3/` = `/mnt/filesystem-g4/tracks/acp/code_s3` (installed by `ops/to_stage3.sh` at the stage-2 -> stage-3 switch):
  stage-3 goal environments (`acp/goals.py` tiles, `acp/env.py` goal terms / waypoints / curriculum, `acp/train.py` V_fs,
  per-type advantages, warm-up), `tools/goal_check.py`; data cfg `cfg/stage3_sets.json` (stage2d + `data/goals/goal_tiles_v1.json`,
  rebuilt with `python -m acp.goals data/goals/goal_tiles_v1.json 12`).
Mirrored 2026-11-12 04:00 UTC; re-mirror after any change there.
