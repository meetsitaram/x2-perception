# What data the three tracks train on (inventory, no data)

Read this before asking the user where any data came from. Checked on the shared filesystem on 2026-11-12 05:00 UTC by the
coordinator (file listings, clip-name patterns inside the pickles, and the build scripts on the desktop). Per-file sizes,
sha256 values and build commands are in each track's manifest:
`tracks/acp/code/MOTIONS.md`, `tracks/ssr/code/MOTIONS.md`, `tracks/hiking/x2_port/MOTIONS.md`.

**The motions themselves are never in this repo** (user, 2026-11-12: "we dont include the actual motions in the repo"). The repo
holds this inventory, the manifests and the build scripts only.

## Where it is (`/mnt/filesystem-g4`)

| Folder | Size | What |
|---|---|---|
| `data/motions/ssr_prior_sources/apparatus_build_train/` | part of 102 MB | Six clip sets from the desktop `loco-manip-apparatus/build/train` (50 Hz X2 clips: root position, root rotation, joints) |
| `data/motions/ssr_prior_sources/terrain_park_train_sets/` | | Six clip sets from the desktop `terrain-park/data/train_sets` |
| `data/motions/hiking_x2/` | | The Hiking track's X2 reference (v1, v2) with its build and statistics scripts |
| `data/motions/hiking_release_g1/data_model.zip` | 12 MB | The paper's released G1 motion file; reference only, never copied on |
| `tracks/acp/data/clips/`, `tracks/acp/data/goals/` | 103 MB | ACP's clip sets (same sources, plus its own re-fitted stair sets) and stage-3 goal tiles |
| `tracks/ssr/data/priors/` | 141 MB | SSR's built motion priors (`amp.npz`, `amp_v2.npz`, `amp_v3.npz`) |
| `data/apparatus/ours/`, `data/apparatus/nist/` | 0.4 MB | Height rasters, wall boxes and eval routes of the two apparatuses; held out of training; the NIST files are not to be published |
| `data/terrains/extended_v1/` | 9.4 MB | The user's extended training map: 330 tiles of 8 x 8 m, 9 types, 10 levels, 650 checked routes |
| `data/terrains/extended_v1_1_stair_routes/` | 0.9 MB | Additive to extended_v1 (2026-11-12): 526 extra routes on the 60 straight, L and U stairs tiles in `stairs_routes_extra.json` (side approach with a turn at the foot, oblique, far straight, descend, side down, L, C, S, cross); tiles and rasters stay those of `extended_v1/stairs.npz`; README lists what the tiles cannot hold |
| `data/terrains/extended_v1_2_stairs_inset/` | 1.7 MB | Additive (2026-11-12): the 60 straight, L and U stairs tiles regenerated with the stair foot 2.5-3.0 m inside the tile (same ids, seeds and stair parameters; 17 straight tiles lose 1-6 steps per flight), in extended_v1's `stairs.json` / `stairs.npz` format, with 700 routes in 11 kinds, each at least 0.6 m from the tile edge and 0.40 m from obstacle cells |
| `data/terrains/extended_v1_1_prop_routes/` | 2.4 MB | Additive to extended_v1 (2026-11-12): 1,116 extra routes on the 170 single-beam, beam-field, log-corridor, wedge and block tiles in `<type>_routes_extra.json` (straight, oblique, side approach with a 70-90 degree turn within 0.6 m of the prop, L, C, S, along a beam, across a wedge), each at least 0.6 m from the tile edge and 0.40 m from obstacle cells; tiles and rasters stay those of extended_v1; README has the before / after shape tables and what the tiles cannot hold (28 of 40 corridors have no open side; long ramp and pinwheel sit at the tile edge) |
| `data/robots/x2/` | 223 MB | X2 MJCF, URDF, meshes and plant files (training plant `mc_20261024`) |

## The source of every motion set

| Set | Clips | Source | BONES-SEED-derived? | Used by |
|---|---|---|---|---|
| `x2_loco_flat_train.pkl` | 351 | BONES-SEED X2 retargets, flat walking | yes | ACP, SSR, Hiking (flat walking and turning) |
| `x2_loco_blocks_train.pkl` | 351 | the same clips with block props | yes | ACP, SSR |
| `gst_v1.pkl` | 113 | BONES-SEED stair climbs fitted to generated flights | yes | ACP, SSR |
| `gs2_v1.pkl` | 89 | BONES-SEED stair climbs and descents on generated flights | yes | SSR |
| `gs2dn_v1.pkl` | 38 | BONES-SEED stair descents | yes | ACP stage 1 only, SSR |
| `gs2dn_v2.pkl`, `gs2lo_v1.pkl` (ACP's own re-fits) | 57, 82 | the same BONES-SEED stair climbs and descents | yes | ACP from checkpoint 3,000 |
| `x2_chains.pkl` / `x2_chains_far.pkl` | 96 | chained walks and prop skills from the clip library: BONES-SEED walks and stair climbs, plus the user's Pico teleop clips | yes (mixed) | ACP, SSR (box clips only) |
| `tp_stairs_x10.pkl`, `tp_stairs_x10_sh2.pkl`, `tp_stairs_chain_x10.pkl` | 8, 88, 16 | the user's Pico stair captures on the terrain park (2026-10-30), with stride and height variants | no | ACP, SSR, Hiking (stair takes) |
| `tp_stairs_all_v1.pkl` + `_props.json`, `_meta.json` (same folder, built 2026-11-12) | 44 (307 s) | the user's Pico stair captures (2026-10-30), both stair places: 44 of the 53 takes, base takes only, kinematic (no tracker replay); includes 6 climbs with a 63-102 degree turn at the stair foot, 3 C or S descents and 4 descents with a turn; shape class and checks per clip in `_meta.json` | no | none yet (new) |
| `tp_ramp_sh2.pkl` / `acp_tp_ramp_sh2_x30.pkl` | 264 / 182 | the user's Pico ramp captures (2026-10-30), with variants | no | ACP, SSR |
| `tp_t4x4u_y20.pkl` | 56 | the user's Pico 4x4 step-over captures (2026-10-30) | no | ACP, SSR |
| `pico_2026-10-29_2x2_train.pkl` | 10 | the user's Pico 2x2-log captures (2026-10-29) | no | ACP, SSR |
| `tp_t4x4u_y20_mir.pkl` + `_props.json` (built 2026-11-12, mirror sets: `terrain-park/tools/mirror/mirror_sets.py`) | 56 (217 s) | left/right mirror of `tp_t4x4u_y20` about each beam's crossing axis; beam maps onto itself | no | to add: ACP, SSR, Hiking |
| `pico_2026-10-29_2x2_train_mir.pkl` + `_props.json` | 10 (49 s) | mirror of `pico_2026-10-29_2x2_train`; log maps onto itself | no | to add: ACP, SSR, Hiking |
| `tp_stairs_all_v1_mir.pkl` + `_props.json`, `_meta.json` | 44 (307 s) | mirror of `tp_stairs_all_v1` about the stair up-axis; place 1 deck is NOT symmetric (mirrored deck moves 0.57 m): train only with `_mir_props.json` | no | to add: ACP, SSR, Hiking |
| `tp_ramp_base_x30_mir.pkl` + `tp_ramp_hmap_x30_mir.npz` | 15 (142 s) | mirror of the 15 base takes of `acp_tp_ramp_sh2_x30` (no `_sh` variants) about the ramp's up-slope axis; ramp height map NOT symmetric (204 cells differ > 2 cm, up to 18.5 cm): train only on `tp_ramp_hmap_x30_mir.npz` | no | to add: ACP, SSR, Hiking |
| `tp_driveway_v1.pkl` / `tp_driveway_v1_mir.pkl` + `_props.json` | 6 / 6 (47 s) | the user's Pico driveway takes (2026-10-30, as in the catalogue, 50 fps; no set held them before) and their mirror; props = the fitted plane `z = a x + b y + c` (4.5 deg; symmetric under the mirror) | no | to add: ACP, SSR, Hiking |
| `tp_stairs_all_v1_1.pkl` + `_props.json`, `_meta.json`; `tp_stairs_all_v1_1_mir.pkl` + `_mir_props.json`, `_mir_meta.json` (terrain-park `data/train_sets/`, built 2026-11-12, tools `terrain-park/tools/fix_v1_1/`) | 44 (307 s) / 44 | **supersedes `tp_stairs_all_v1` and `_v1_mir`**. One clip changed: `tp_stairs_113941` (+`_mir`) re-retargeted with a fresh IK state. In v1 the warm start carried over from take 113923 pinned waist yaw/pitch/roll at their limits (2.38, 0.35, -0.52 rad) and hip yaws near +-1.7 for the first 3.4 s, so the X2 pitched 46-59 deg forward while the person stood upright (raw: headset 1.66 m, lean 13 deg). Now upright (lean matches the human within 2 deg), builder checks pass (0 contacts over 2 cm). Class changes from "side approach then up" to "straight up". Other 43 clips and all props bit-identical to v1 | no | to add: ACP, SSR, Hiking (replace v1) |
| `pico_2026-10-29_2x2_train_v1_1.pkl`, `_v1_1_mir.pkl` + `_v1_1_mir_props.json` (base props: `pico_2026-10-29_2x2_props.json`, unchanged) | 9 (45 s) / 9 | **supersedes `pico_2026-10-29_2x2_train` and `_train_mir`**: drops `pico_20261029_211326_0_2x2` (+`_mir`): left hip yaw pinned at its 3.43 rad limit and waist yaw over 1.2 rad for the whole take (retarget warm-start artifact; the 2x2 tool places the prop during retargeting, so it was not re-run). Other 9 clips bit-identical | no | to add: ACP, SSR, Hiking |
| `tp_driveway_v1_1.pkl` / `_v1_1_mir.pkl` + `_props.json` | 6 / 6 (47 s) | **supersedes `tp_driveway_v1` / `_v1_mir`**: all 6 takes re-retargeted with a fresh IK state (v1 had the right hip pinned at pitch 2.56 / roll -2.91 / yaw -3.43 rad, all at limits, through every take), then 50 Hz interpolation + 0.04 s Gaussian smoothing and the 8 untracked joints zeroed (stairs recipe; v1 was nearest-sample, root speed spikes 3.2-4.3 m/s). Plane refitted (4.5 deg, c +3 mm). Open: 42 of 97 contacts are more than 2 cm off the single plane (v1: 40), max 8 cm | no | to add: ACP, SSR, Hiking |
| `x2_mirror_map.json` (same folder) | 31 joints | X2 left/right joint mirror map (perm + sign) from `x2_ultra.xml`, FK-checked | n/a | any track mirroring clips |
| `x2_parkour_motion_without_run_v2_retargetted.npz` | 400 s | about 300 s BONES-SEED flat walking, about 80 s Pico stair takes, 15 s standing | yes (mixed) | Hiking; SSR's unused `amp_v3` |

**Answer to the question that was open until 2026-11-12:** the stair sets (`gst_v1`, `gs2_v1`, `gs2dn_*`, `gs2lo_v1`) and the
chain set from `loco-manip-apparatus` are BONES-SEED-derived. Evidence: the clip names inside the pickles carry the BONES-SEED
stair-clip names; `gwbc-v6/scripts/acp/data/gen_stairfit.py` and `gen_stairfit2.py` read the BONES-SEED stair files and say so
in their headers; `loco-manip-apparatus/sim/pipeline/clipdb.py` lists the chain library's sources as Pico teleop sessions plus
BONES-SEED walks and stair climbs. So they follow the BONES-SEED rule: never in a repo, Bones Studio credit on anything shown,
no clip names on screen.

The stair and chain fits use generated flights and boxes, not the apparatus's own layout (the chain set is moved 30 m off the
apparatus), in line with the held-out rule.

## Not found on the shared filesystem

- The builders of the terrain-park sets (`tp_*`) and the flat train/test split script: the ACP manifest lists them as "builder
  not found". They are on the desktop in `terrain-park` and `loco-manip-apparatus`.
- Backward and sideways walking clips: not in any set (user: "for now just keep on list").
- A licence file for the user's Pico captures: they are the user's own recordings and are not committed anywhere.

## What is inside the "flat" walking set (counted 2026-11-12)

`x2_loco_flat_train.pkl` is flat-ground motion, not plain walking. `loco-manip-apparatus/sim/pipeline/loco_dataset.py` keeps
every clip from the BONES-SEED subsets `x2_loco30k` and `x2_walk_2k` that has no prop support (no foot or knee on a box or
stair, no seat); it does not filter by style. It was built for SuperMimic's general tracking. By clip name (first matching word):

| Kind | Clips | Seconds |
|---|---|---|
| Bump into a small obstacle | 87 | 904 |
| Plain walk | 80 | 609 |
| Injured walk | 79 | 613 |
| Idle | 32 | 219 |
| Crouch | 22 | 116 |
| Jump | 20 | 99 |
| Turn | 15 | 339 |
| "back" in the name | 12 | 62 |
| Other | 4 | 24 |

The blocks set is the same 351 clips with props added. SSR and Hiking filter this file by name when they build their priors
(keep walk, turn, jog, stride, back; drop injured, crouch, jump, bump and similar); ACP trained on all of it until data set 2e.

## Where the user's Pico captures live (checked 2026-11-12, at the user's request "so we don't miss them next time")
- **Repo of record: `terrain-park`** (GitHub `meetsitaram/terrain-park`, `main` since 2026-11-12; local `C:\Users\PC\projects\terrain-park`). Tracked there: every raw take and its X2 retarget (`data/2026-10-29/{raw,x2,x2_3.8cm}` 38 each; `data/2026-10-30/{raw,x2}` 150 and 149, including all 53 stair takes of 11:36-11:47), the fitted props (`data/2026-10-30/props`, `station_*`), the ready sets in `data/train_sets/` and the fitting tools in `tools/`.
- `C:\Users\PC\projects\pico-captures` is a working copy (not a git repository) with the same sessions plus physics replays; its README lists every station with take counts and times. Read it before concluding that a prop has few takes.
- **Check to run whenever a prop's motion set looks thin:** count the takes in `terrain-park/data/<date>/x2` for that station's time window against the clips in the training set. On 2026-11-12 this showed 53 stair takes recorded and 8 in `tp_stairs_x10.pkl` (the prop had only been fitted at the second recording place).
- Built 2026-11-12, **committed and pushed to `terrain-park` (commit 3f872c4) and merged into `main` with the whole `add-prop-captures` branch (pull request 1, merge 35f7c4c; the repository is public)** at the user's request: `data/train_sets/tp_stairs_all_v1.pkl` (+ `_props.json`, `_meta.json`, `_topview.png`) and its builders `tools/stairs_all_v1/{build.py,final.py}`. Also on the shared filesystem (`terrain_park_train_sets/`).
- Nothing of the captures is committed in this repo (user's data; this file is the pointer).

## The Hiking paper's reference retargeted to X2 (2026-11-13): shared filesystem only
`/mnt/filesystem-g4/data/motions/hiking_release_x2/` (v1 and v2, props, checks, tools). Source: the paper's `data_model.zip`. Licence: the release's InstinctLab is CC BY-NC 4.0 (non-commercial); the zip carries no licence of its own. Not committed to terrain-park or any repo (user, 2026-11-13: only if the licence is Apache-style and allows use). Used in training (Hiking's reference, SSR's amp_v5) as non-commercial material.
