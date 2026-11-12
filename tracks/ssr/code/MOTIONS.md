# Motion files used by the SSR track (the recipe, not the data)

User rule ("Motion data and the repo"): no BONES-SEED-derived motion and no Hiking-release data in the repo. This file lists
what the SSR motion priors (AMP discriminators, paper App. A.8) and the walking starts (reference-state initialisation) are made
from and how to rebuild them. The data itself stays on the shared filesystem; `code/data/amp*.npz` are only symlinks to
`/mnt/filesystem-g4/tracks/ssr/data/priors/` (the sync to the repo excludes `data/`, `*.npz`, `*.pkl` and links).
Full numbers: `/mnt/filesystem-g4/tracks/ssr/eval/motion_inventory.json` (from `scripts/motion_inventory.py`, 2026-11-12).

## Built priors (one file = five per-terrain sets: flat, ramp, stairs, beam, platform; 63-d AMP frames at 50 Hz)

| File (in `tracks/ssr/data/priors/`) | Bytes | sha256 | flat | ramp | stairs | beam | platform | Used by |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `amp.npz` | 47,101,908 | `3dcff971779207c3...` | 41 clips, 405.8 s | 63, 563.7 s | 206, 1240.6 s | 88, 605.2 s | 117, 922.3 s | r1-r4, d1-d4 |
| `amp_v2.npz` | 54,576,504 | `edb7dc271d4e6d2a...` | 107, 999.0 s | as amp | as amp | as amp | as amp | r5, recipe tests t_a-t_f |
| `amp_v3.npz` | 45,280,284 | `ccfb8fa4b27af890...` | 83, 261.2 s | as amp | as amp | as amp | as amp | not used (head-to-head only) |

Seconds with planar speed >= 0.2 m/s (what the discriminator sees with `--prior_min_speed 0.2`): flat 230.5 / 661.2 / 245.3 s
(amp / v2 / v3); ramp 357.1, stairs 764.0, beam 389.5, platform 508.5 s (all three files).
The full sha256 of every file is in `motion_inventory.json`.

Walking starts: with `--rsi_frac` (default 0.5) half the episodes start from frames of the flat set of the `--amp_file` in use
whose planar speed is at least `--rsi_min_speed` (0.3 m/s in r5 and the tests; 45,968 of 99,902 frames for amp_v2).

## Sources (read-only, `/mnt/filesystem-g4/data/motions/`)

| Source file | Bytes | sha256 (first 16) | Clips | Seconds | Origin | Feeds |
| --- | --- | --- | --- | --- | --- | --- |
| `ssr_prior_sources/apparatus_build_train/x2_loco_flat_train.pkl` | 22,777,412 | `5c6e628009f141d4` | 351 | 2986.2 | BONES-SEED X2 retargets (flat locomotion), from the user's `loco-manip-apparatus/build/train` | flat (all files), and the flat share of ramp/stairs/beam/platform |
| `.../x2_loco_blocks_train.pkl` | 22,778,692 | `50b699939cd8431d` | 351 | 2986.2 | BONES-SEED X2 retargets placed on blocks, same build | beam |
| `.../gst_v1.pkl` | 4,319,024 | `8d3603dac56f6a5f` | 113 | 564.6 | stair fits by rise/tread from the same build (BONES-SEED-derived) | stairs |
| `.../gs2_v1.pkl` | 3,334,516 | `3b88259ee786e4b3` | 89 | 435.7 | stair up/down fits, same build (BONES-SEED-derived) | stairs, platform |
| `.../gs2dn_v1.pkl` | 1,418,260 | `ff583077a8c93bab` | 38 | 185.3 | stair-down fits, same build (BONES-SEED-derived) | stairs |
| `.../x2_chains.pkl` | 12,808,988 | `3ddf3414a13fd001` | 96 | 1682.7 | chained clips across props (box, stairs), same build (BONES-SEED-derived) | platform (box clips only) |
| `ssr_prior_sources/terrain_park_train_sets/acp_tp_ramp_sh2_x30.pkl` | 12,552,108 | `1c417c55b7f490a8` | 182 | 1646.5 | the user's Pico captures on the terrain-park ramp (shifted variants) | ramp |
| `.../tp_stairs_x10.pkl` | 340,832 | `e4015cf427a35d87` | 8 | 44.6 | the user's Pico stair captures | platform |
| `.../tp_stairs_x10_sh2.pkl` | 3,747,680 | `6428992c570692bc` | 88 | 490.6 | the user's Pico stair captures (shifted variants) | stairs, platform |
| `.../tp_stairs_chain_x10.pkl` | 2,224,188 | `10fb76492846fc01` | 16 | 292.2 | the user's Pico stair captures, chained | stairs |
| `.../tp_t4x4u_y20.pkl` | 1,663,884 | `59c398584686151d` | 56 | 217.4 | the user's Pico captures over a 4x4 log | beam |
| `.../pico_2026-10-29_2x2_train.pkl` | 375,688 | `218288f972b01d80` | 10 | 49.1 | the user's Pico captures over a 2x2 | beam |
| `hiking_x2/x2_parkour_motion_without_run_v2_retargetted.npz` | 3,044,666 | `66e777d585cbc133` | one concatenated sequence (228 seams found) | 400.0 | the Hiking track's cleaned X2 walking file: BONES-SEED flat walking/turning + Pico stair takes, standing cut out | flat set of amp_v3 only |

The origin of the `apparatus_build_train` stair and chain sets is read from their clip-name patterns and the build directory;
to confirm with the user if it matters (treated as BONES-SEED-derived here).

## How each prior is built (exact commands, from `tracks/ssr/code`)

- `amp.npz`: `scripts/build_amp.py` (2026-11-09, run on the desktop; its source paths are the desktop's
  `C:/Users/PC/projects/loco-manip-apparatus/build/train` and `C:/Users/PC/projects/terrain-park/data/train_sets`, the same
  files now under `ssr_prior_sources/`). Per category it takes clips from the list in `SETS` up to a seconds budget per source
  (flat: 400 s from `x2_loco_flat_train.pkl`; each terrain set: 150 s of flat walking plus its terrain sources with budgets 45-400 s, e.g. ramp 400 s of the Pico ramp captures, beam the 4x4 and 2x2 captures plus 150 s of block clips), flat clips filtered by
  name (keep walk|turn|jog|run|stride|back; drop injured|crouch|jump|zombie|drunk|hop|crawl|tiptoe|limp|bump|sneak), clips
  under 0.5 s dropped, resampled to 50 Hz, forward kinematics in the SSR X2 model, frames with non-finite values or joint speed
  over 25 rad/s dropped.
- `amp_v2.npz`: `CUDA_VISIBLE_DEVICES= ../env/venv/bin/python scripts/build_flat_v2.py` (2026-11-11): amp.npz with the flat set
  rebuilt from all 107 clips that pass the same name filter (999 s).
- `amp_v3.npz`: `CUDA_VISIBLE_DEVICES= ../env/venv/bin/python scripts/build_flat_v3.py` (2026-11-12): amp_v2 with the flat set
  replaced by the Hiking file cut at its seams (joint jump > 0.25 rad per frame, pelvis height jump > 2 cm, pelvis
  acceleration > 40 m/s^2), pieces under 1 s and pieces with a pelvis height range over 8 cm (stair takes) left out.
- At training time (`ssr/train.py`, `AMPData`): `--prior_min_speed 0.2` keeps only 5-frame windows whose mean planar speed is
  at least 0.2 m/s (moving frames only); `--prior_balance` samples with equal mass per 0.2 m/s speed bin (no window more than
  5x uniform). Speed tables: `state/ssr.md` and `eval/flat_prior_v3.json`.

## Licences (one line per source)

- BONES-SEED (Bones Studio): credit "Motion Data by Bones Studio (bones.studio)" on anything shown; no dataset clip names on
  screen; not redistributed in the repo.
- The user's Pico captures (terrain-park ramp, stairs, 4x4, 2x2): the user's own data; not in the repo.
- Hiking track's file: derived from BONES-SEED and the user's Pico captures (same terms as above); a copy from another track,
  read-only under `data/`.
