# ACP track: motion and clip files used in training (recipe only, no data)

User rule ("Motion data and the repo"): no BONES-SEED-derived motions or Hiking-release data in the repo. This file is the
recipe. The data stays on the shared filesystem under `/mnt/filesystem-g4/tracks/acp/data/clips/`, and nothing here is a
copy of it. Updated 2026-11-12 01:30 UTC by the ACP agent. "Clips" is the number in the pickle; the trainer cuts clips into
10 s segments, and the segment counts are the ones in `state/acp.md`.

## Clip sets (`data/clips/`, 50 Hz kinematic X2 clips + box props)

| File | Bytes | sha256 | Clips / min | Source | Built by | Used in |
|---|---|---|---|---|---|---|
| x2_loco_flat_train.pkl | 22,777,412 | 5c6e6280...9bdb6a2d | 351 / 49.8 | BONES-SEED X2 retargets (`bones-seed/x2_loco30k`, `x2_walk_2k`), flat-ground clips | `data_build/upstream/loco-manip-apparatus/sim/pipeline/loco_dataset.py` (`python -m pipeline.loco_dataset`, run from `sim/`); the flat train/test split script was not found | stage 1, stage 2 (0.20) |
| x2_loco_blocks_train.pkl + _props.json | 22,778,692 / 248,651 | 50b69993...df97cdfe / 1031c058...f1dd0ac2 | 351 / 49.8 | same clips as flat, with block / clutter props | same base (`loco_dataset.py`, distractor props); split script not found | stage 1, stage 2 (0.20) |
| x2_chains_far.pkl + _props.json | 12,808,988 / 75,769 | 72af1f23...19f85625 / 99c0e846...32b71bbc | 96 / 28.0 | motion-matched walks and prop skills (clip library of `clipdb.py`: BONES-SEED walks and stair climbs, the user's Pico teleop clips) on generated boxes; moved 30 m off the apparatus | `data_build/upstream/.../pipeline/chain_dataset.py` (`python -m pipeline.chain_dataset`); the 30 m move is not recorded | stage 1; stage 2 with `exclude_step_over: 0.30` (50 of 96 clips out) (0.15) |
| tp_stairs_x10_sh2.pkl + _props.json | 3,747,680 / 39,675 | 6428992c...f6d7419d / 60de368f...a4f797ac | 88 / 8.2 | the user's terrain-park Pico stair captures (2026-10-30), retargeted, with fitted stairs; `_sh*` = stride / height variants of the same takes | builder not found; the stair fit is `data_build/upstream/terrain-park/tools/fit_stairs_x2.py` | stage 1 (0.10); stage 2, 2b (0.15, `ground_stance` in 2b); 2c (0.08) |
| gst_v1.pkl + _props.json | 4,319,024 / 117,569 | 8d3603da...4b281abd / cfee9229...d5c02b23 | 113 / 9.4 | BONES-SEED stair climbs (`bones-seed/x2_props_all`, `stairs_climbing_up_start_*`, `_jog_up_start_*`) fitted to generated flights (never the apparatus's riser / tread) | `data_build/upstream/gwbc-v6/scripts/acp/data/gen_stairfit.py OUT_PREFIX` (desktop, Python312) | stage 1, stage 2, 2b (0.10); 2c (0.07) |
| gs2dn_v1.pkl + gs2dn_v1q_props.json | 1,418,260 / 26,058 | ff583077...6a39237c / d660cad3...4cea2f83 | 38 / 3.1 | BONES-SEED stair descents (`stairs_climbing_{down,jog_down}_{loop,stop}_*`) on generated flights | `.../gen_stairfit2.py OUT_PREFIX K`; how the descents were split out and the props re-quantised is not recorded | stage 1 only (0.05). **Left out from stage 2**: every clip has the leg turned about 180 degrees (hip yaw +-3.43 rad), because the fit turns each foot up the stairs while the body faces down |
| gs2dn_v2.pkl + _props.json + _report.json | 1,995,980 / 69,424 / 45,332 | dda03ec7...ff750269 / 23274f3f...e844bc46 / 2cc08475...c9647f3d | 57 / 4.3 | same BONES-SEED descents | **ours:** `data_build/gen_stairfit_dn.py` (desktop, CPU, 6 GB cap, idle priority): `DN_GAP=0.05 python gen_stairfit_dn.py <out>/gs2dn_v2 4` (imports the upstream `stairfit.py`, replaces its `tread_targets` so the foot faces down the stairs; acceptance as gen_stairfit2 plus \|hip yaw\| <= 1.5 rad and swing clearance >= -3 cm); 57 of 108 fits accepted | `cfg/stage2c_sets.json` (0.05), from checkpoint 3,000 |
| gs2lo_v1.pkl + _props.json + _report.json | 3,007,420 / 98,854 / 60,358 | 30db5030...2bb6e1c1 (pkl) | 82 (39 climbs, 43 descents) | same BONES-SEED stair climbs and descents, on generated LOW-riser flights (risers 10-15 cm, treads 25-32 cm) | **ours:** `data_build/gen_stairfit_low.py` (desktop, CPU, capped): `DN_GAP=0.05 python gen_stairfit_low.py <out>/gs2lo_v1 3`; added because the eval ladder failed 10 cm risers and no fitted flight had one | `cfg/stage2c_sets.json` (0.05), from checkpoint 3,000 |
| tp_ramp_sh2.pkl | 20,051,356 | 32955375...be15e034 | 264 / 43.8 | the user's terrain-park Pico ramp captures (2026-10-30) + stride / height variants; the trainer shifts them +30 m in x (`"shift": [30, 0]`) | builder not found; ramp fit `data_build/upstream/terrain-park/tools/fit_ramp_x2.py` | stage 1, stage 2 (0.10; `ground_stance` from 2b) |
| tp_ramp_hmap_x30_40mm.npz | 63,240 | eb97b9ff...42d1588c | (height map) | the ramp map moved 30 m and coarsened to 4 cm by the ACP track (from `tp_ramp_hmap_x30.npz`, afc51199...b91e0625; that from `tp_ramp_hmap.npz`, 7a43d4f2...360c3304) | upstream builder not found; the 4 cm coarsening was done on the node on 2026-11-11 (see `state/acp.md` section 4) | stage 1, stage 2 |
| tp_t4x4u_y20.pkl + _props.json | 1,663,884 / 13,030 | 59c39858...71cb0923 / e0c5a0de...ad2cb519 | 56 / 3.6 | the user's terrain-park Pico 4x4 step-over captures (2026-10-30) | builder not found | stage 1, stage 2 (0.07; `ground_stance` from 2b) |
| pico_2026-10-29_2x2_train.pkl + _props.json | 375,688 / 3,146 | 218288f9...c7bce496 / 8e80f519...4bb4af99 | 10 / 0.8 | the user's Pico captures (2026-10-29, 2x2-log crossings) | `data_build/upstream/loco-manip-apparatus/sim/capture/build_train.py <pico-captures>/2026-10-29` (every n-th take held out as test) | stage 1; stage 2 with `exclude_leg_flip` (1 clip out) (0.03) |

Not trained on (in the folder, kept): `x2_loco_flat_test.pkl`, `x2_loco_blocks_test.pkl`, `pico_2026-10-29_2x2_test.pkl`
(held-out splits); `gst_v2_resolved.pkl` (b0f3708c...6dbea2f4, legs re-solved with `tools/repair_legs.py`; nothing changed,
so it was not used); `tp_ramp_hmap.npz`, `tp_ramp_hmap_x30.npz` (earlier versions of the map).

Full sha256 values: `sha256sum /mnt/filesystem-g4/tracks/acp/data/clips/*`. The table lists only the first and last 8
characters.

## Derived files (built from the clip sets on the node; no new motion)

| File | Bytes | sha256 | What | Built by |
|---|---|---|---|---|
| code/cfg/labels_stage1_sets.json.pt | 74,338,993 | fec43f8d...3fd2a5cc | oracle contact labels (nine bodies, next contact, occupancy) for stage 1 | `acp/data.py` `Dataset._label` (written on first use of the cfg) |
| code/cfg/labels_stage2_sets.json.pt | 66,809,265 | 8d8fea0f...cc407e24 | same for stage 2 | same |
| code/cfg/labels_stage2b_sets.json.pt | 66,809,274 | abc80908...5f3abaed | same for 2b (after the `ground_stance` lift) | same |
| code/cfg/labels_stage2c_sets.json.pt | 68,451,450 | 56bb452b...29de26ef | same for 2c (with gs2dn_v2) | same (built by `tools/clip_quality.py cfg/stage2c_sets.json`) |
| code/cfg/armfix_stage2d_sets.pt | 13,596,733 | 25feb324...7aad411e | per-frame offsets of shoulder roll, elbow and shoulder yaw (both arms) so every arm clears its thigh and the torso by >= 1 cm with the training collision geometry; applied to the clip sets by `acp/data.py` when the cfg has `"armfix"` (no new motion, a correction of the references) | `tools/arm_clear.py cfg/stage2d_sets.json 8` (node CPU, about 1 min) |
| code/cfg/labels_stage2d_sets.json.pt | 70,926,458 | 63d7f19d...fe1cca0c | labels for stage 2d (after the ground lift and the arm fix) | `acp/data.py` (built by `tools/self_pen.py` through the staged loader) |
| eval/paper_eval/s1_v1_{001250,002000_plant_default,002250_plant_mc_20261024}_rollouts.pt | 1.47 GB each | 54c1d9d7...16dda927, a22ecc4e...750fbbe6, 4c6a9de2...357f011e | the tracker's own rollouts from the paper evaluation: input for the CPU planner jobs (hindsight labels), not reference motion | `tools/paper_eval.py` |

Stage 2 trains the planner online. Its replay buffer holds labels from the tracker's own rollouts, kept in memory and not
stored on disk.

## Build commands (what a rebuild runs)

- Desktop data sets: run each upstream script from its own repo (`loco-manip-apparatus/sim`, `gwbc-v6/scripts/acp/data`,
  `terrain-park/tools`). The copies under `code/data_build/upstream/` record which version was used (`README.txt`).
  Copy the outputs to `data/clips/` with `scp -l` (rate-capped).
- Stair descents v2: as in the table (desktop).
- Training data per stage: `cfg/stage1_sets.json`, `cfg/stage2_sets.json`, `cfg/stage2b_sets.json` (2,500-3,000),
  `cfg/stage2c_sets.json` (from 3,000), `cfg/stage2d_sets.json` (2c + arm fix; at the next needed restart). Each holds the weights, the filters (`exclude_step_over`, `exclude_leg_flip`,
  `ground_stance`) and a note saying why each set is in or out. The filters run in `acp/data.py` every time the data is built.
- Quality checks: `tools/clip_quality.py CFG OUT.json`, `tools/ground_check.py CFG`, `tools/pen_cause.py CFG OUT.json`,
  `tools/pen_shift.py CFG OUT.json` (CPU: `CUDA_VISIBLE_DEVICES=`); results in `eval/clip_quality/`.

## Licence per source

- BONES-SEED (x2_loco_*, gst_v1, gs2dn_v1/v2, part of x2_chains_far): bones-seed-license (bones.studio/info/seed-license),
  licence-gated and non-commercial or small-company use only. Never in the repo; Bones Studio credit on anything shown;
  no clip names on screen.
- terrain-park captures (tp_*, the ramp map): the terrain-park repo states Apache-2.0 for its code and data. The data is
  still kept out of this repo by the rule above.
- The user's Pico captures (pico_2026-10-29_2x2, Pico teleop clips inside the chains): the user's own recordings; no
  licence file found; kept out of the repo.
