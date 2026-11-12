# X2 motion references for the Hiking track (the recipe; the data is not in the repo)

User rule ("Motion data and the repo"): no BONES-SEED-derived motion and no Hiking-release data in the repo. The repo keeps
this file and the build scripts. The data stays on the shared filesystem.

## Files (on the shared filesystem under `/mnt/filesystem-g4/data/motions/`)

| Reference | Path | Size | sha256 | Content |
|---|---|---|---|---|
| v1 (run 1 until iteration 500) | `hiking_x2/x2_parkour_motion_without_run_retargetted.npz` (+ `.yaml` selector) | 3,087,986 B | d7f7803439b22fd33e14ab24326db1e67a4f450e344446d5ef68478b21bc0cd0 | 405.7 s at 50 Hz: about 322 s BONES-SEED flat walking and turning (26 clips), about 83 s of the user's Pico stair takes (16 takes); standing not removed (21% of frames under 0.1 m/s) |
| v2 (run 1 from 500, runs 2 and 3) | `hiking_x2/x2_parkour_motion_without_run_v2_retargetted.npz` (+ `_v2.yaml`) | 3,044,666 B | 66e777d585cbc13336fbb1d94eaee30f7747ab4e8aa5b0a8d43dc7b6679f4f58 | about 400 s: BONES-SEED walking pieces at least 0.2 m/s (budget 300 s), Pico stair pieces (budget 80 s), 15 s of standing kept on purpose (3.3% of frames); median speed 0.53 m/s |
| G1 release, for reference only | `hiking_release_g1/data_model.zip` | 11,905,690 B | bbba90bdb77bbab19e74cbe9f556ea0b9330ea9d125ad8271d431aa0e37a6609 | the paper's released G1 motion file (379.6 s). Kept only here; never copied anywhere else, never in the repo. On the node it is unzipped into `~/Datasets` for the G1 task |

The per-source seconds of v2 are the build budgets and the state-file figures. The build's own printout for v2 was not kept;
re-running the build prints it.

## How they are built

- Script: `build_x2_motion.py` (v2; it does the standing cut itself, so there is no separate cleaning script). Statistics:
  `motion_stats.py`. Copies are in `tracks/hiking/code/x2_port/motion/`; the originals are next to the data in
  `data/motions/hiking_x2/`.
- Command (desktop, an environment with `joblib`, `mujoco`, `numpy`):
  `python build_x2_motion.py <x2_ultra.xml> <x2_loco_flat_train.pkl> <tp_stairs_x10_sh2.pkl> x2_parkour_motion_without_run_v2_retargetted.npz`
  - `<x2_ultra.xml>`: the X2 MJCF (`data/robots/x2_ultra/mjcf/x2_ultra.xml`, also `gwbc-v6/gear_sonic/data/assets/robot_description/mjcf/`)
  - `<x2_loco_flat_train.pkl>`: BONES-SEED X2 retargets, desktop `loco-manip-apparatus/build/train/x2_loco_flat_train.pkl`
  - `<tp_stairs_x10_sh2.pkl>`: the user's Pico stair takes retargeted to the X2, desktop `terrain-park/data/train_sets/tp_stairs_x10_sh2.pkl`
- Fixed seed (`default_rng(0)`). Clips are filtered by name (walk, turn, stride; injured, crouch, jump, run, back and similar
  are dropped). Stretches whose 0.5 s mean speed is under 0.2 m/s are cut out, and only pieces of at least 1.5 s are kept.
  Head and wrist joints are zeroed. Clips are chained in xy and yaw, and joints clipped to the MJCF limits.
- v1 came from the same script before the standing cut (no `MIN_SPEED` / `STAND_S` step).
- `motion_stats.py <npz>` prints the standing share, seams and speed percentiles.

## Licence notes

- BONES-SEED (flat walking and turning): Motion Data by Bones Studio (bones.studio). Credit on anything shown; no clip names
  on screen; derived files are not put in the repo.
- Pico stair takes: captured by the user. Kept on the shared filesystem.
- Hiking in the Wild release (G1): the paper authors' released data, under its release terms. Reference only, kept in
  `data/motions/hiking_release_g1/`, not redistributed.

## Open

- Backward and sideways clips are not in v1 or v2. They are needed for the extended run's backward and sideways commands.
  They stay on the list for now (user: "for now just keep on list").
