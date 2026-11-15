# Automatic scorer (eval/autoscore)

Routine scoring without an agent (user, 2026-11-12). One Python process on the desktop, no LLM.

## What it does
Every `poll_minutes` (10) it lists each track's hand-over exports over ssh:
- ACP: `/mnt/filesystem-g4/tracks/acp/eval/pair_*` with `DONE`
- SSR: `/mnt/filesystem-g4/tracks/ssr/eval/export/<name>/` with `policy.onnx`, `adapter_card.json`, `obs_check.npz`
- Hiking: `/mnt/filesystem-g4/tracks/hiking/eval/x2_parkour_v4_*` with those three plus `0-depth_encoder.onnx`
(an export counts only when its newest file is over 2 minutes old). It picks, per track, the `priority` names first, then the
newest export by file time that is not in `skip` and not scored yet (`policy: "newest"`; `"all"` scores every export, newest first).
An export it has started is finished before the track's next one starts.

For each export, on the default floor (solid, trimmed), strict stand, end retry, `X2_EXCLUDE_PAIRS=hip`, the same commands as
the earlier queue scripts:
1. fetch (scp into `eval/data/handover/<acp_s2v3_7250 | ssr_r7_3550 | hiking_v4_4750>`; Hiking gets `run_params` from
   `hiking_run_params`) and the hand-over check (`eval/tools/acp_handover_check.py`, `ssr_check.py`, `hiking_check.py`).
   Check failed: nothing is run, the row says so.
2. **Training terrain first (`train`, user rule 2026-11-13: progress is judged on what a track trains on).** The track's courses
   in `eval/train_terrains.py` `TRAIN_SETS` (ladder step 20), `train_trials` (20) strict trials each, split into chunks of `chunk_trials` (below;
   `train_job_size` is no longer used); slope / ramp / rough courses run with `PHYS_FLOOR=heightfield`
   (the solid floor models flat cells and 10-20 deg planar faces only). SSR: its generator's ramp / stairs / beam / platform at
   levels 5, 10, 15, 19 of 0-19, flat, and five extended-map tiles (22 courses). Hiking: the paper's nine types at curriculum rows
   2, 5, 8 of 0-9 rebuilt from the InstinctLab generator code, and the five extended-map tiles (32). ACP: the ladder's step-1
   obstacle courses (log, wedge, stairs up / down, corridor at levels 1-3) and box steps 15 / 25 cm (17). Which are exact copies
   and which approximations: the module's docstring and the `exact` column.
   ACP only: the hold-out gate (seeds 1001 1005 1036 turn-around, 1019 1022 ramp trough; PASS = 3/3 turn-around and at least 4/5).
   **Information only since 2026-11-13; it blocks nothing.**
3. Flat gate: `flat_walk`, `flat_stop_stand`, `flat_turn_in_place`, `flat_turn_and_return`, 50 each (one process per test).
4. Station legs 1-5, 50 each (one process per leg).
5. Full route 200 (seeds 1000-1199, 4 x 50) only if every leg is at least `route_min_leg` (45) of 50 (user's rule).

**Chunks (2026-11-14, user: "break the work into smaller pieces").** Every training-terrain course and every station leg is
split into chunks of `chunk_trials` (5) fixed seeds (chunk k = seeds 1000+5k .. 1004+5k), one process per chunk
(`ladder.py --chunk k` / `legs.py --legs K --chunk k`, writing `<course or legK>/chunks/c<k>.json` after every trial and
`c<k>.done` at the end). When all chunks of a unit have ended, its merge job (the usual unchunked command) finds every seed on
disk, runs nothing new and writes `summary.json` / `results.json`; the chunk folder is then removed. Per-trial results are identical
to an unchunked run (checked 2026-11-14 on SSR ssr_platform_L10 and ACP leg 3, state/eval.md). Jobs: `train.<course>.c<k>`,
`train.<course>`, `leg<K>.c<k>`, `leg<K>`; a `regress` job runs the regression check once per export, single-threaded, before
any chunk (chunks run with `EVAL_REGRESSION=0`). Old `partial.json` trials are reused. Flat tests, the ACP gate and the route are
not chunked (route: already 4 parallel jobs of 50).
At most `max_cores` (8) processes at once, one per core (OMP threads 1). A chunk job has 2,700 s per attempt and 1,800 s wall limit,
and loses at most one trial when killed; a finished chunk, test, leg or slice is never run again, also after a restart.

Outputs: `eval/out/<track>/<checkpoint>/auto/` (`train/<course>/`, `gate/`, `flat/step1/`, `legs/` with `legs.txt`, `route/s1000..s1150/`, `logs/`).

## Results
**Final-station stop rule (user 2026-11-13, docs/EVAL.md):** from 16:43 UTC the main columns (flat tests, legs,
product, route, train `successes`) are the headline count = strict 0.2 m successes + "stopped short" (stable 1 s stand within
0.4 m of the final station). The strict 0.2 m counts are beside them: `results.csv` `flat_strict`, `legs_strict`,
`product_strict`, `route_strict`, `train_strict`, `stopped_short` (trials flagged in the row); `train_results.csv`
`strict_successes`, `stopped_short`. Rows from before have these blank and their main columns are strict counts. Route
eligibility (legs >= 45/50) and train passes use the headline count. The scorer runs everything with one maths thread.
- `results.csv`: one row per export: track, checkpoint, floor, the four flat tests, flat gate, ACP gate, legs 1-5 (strict
  successes / trials), product of the leg rates, full route (k/200, Wilson 95 % interval, sections entered/completed), flags
  (hand-over check failure, joint-limit / trapped-foot / limb-clamp trial counts over legs and route, failed jobs, legs not run),
  finish time (UTC), minutes, output folder.
- A full route also writes a row `auto-<track>-<name>-trim` into `dashboard/data/apparatus_results.json` and runs
  `python prepare.py` in `dashboard/` (retried on OSError). The upload to the dashboard artifact (README steps 3-6) stays with the
  coordinator. Legs-only results are not put on the dashboard: its bars mean full-route success. `tracks.json` is not touched.
- `train_results.csv`: one row per track, checkpoint, terrain type and level: strict successes, trials, rate, course, exact or
  approximation, floor (an export scored again replaces its rows). `results.csv` column `train_summary`: courses passed (>= 80 %)
  and per type the highest level passed with levels passed / run. Log line `TRAIN`.
- `dashboard/data/train_terrain.json`: the same rows for the review page (with highest level passed per type, Wilson interval),
  rebuilt by the scorer after every export, or by hand: `python eval/autoscore/autoscore.py --build-train-dataset`. It is not a
  registered dataset yet (`prepare.py` ignores it); registering and uploading it is the coordinator's (dashboard README steps).
- `status.json`: what is running now, active exports, the last rows. `state.json`: done / active exports (delete an entry
  under `done` to score that export again; its finished parts are still reused).
- `autoscore.log`: every start, end, row and error. Heartbeat: `/mnt/filesystem-g4/status/eval.json` every 10 minutes.

## Start, stop, configure
- Start (from Git Bash, repo root):
  `nohup python eval/autoscore/autoscore.py >> eval/autoscore/autoscore.log 2>&1 &`
  Check it is not already running first: `cat eval/autoscore/status.json` (pid, updated) and `ps -ef | grep autoscore`.
- Stop: `touch eval/autoscore/STOP`. It starts nothing new and exits when the running slices end (at most about 10 minutes).
  Delete `STOP` before starting again.
- Configure: edit `config.json` (re-read every 15 s): per track `enabled`, `policy`, `skip`, `priority`; `export_options`
  `"<track>:<name>": {"order": ["leg1", "leg2", "leg5", "flat", "leg3", "leg4"]}` changes the order of one export's jobs;
  `max_cores`, `trials`, `train_trials`, `chunk_trials`, `route_min_leg`, `poll_minutes`. `"train"` is put first in any
  custom order that lacks it; `"no_train": true` in an export's options leaves it out.
- Test mode: `python eval/autoscore/autoscore.py --test --only acp:pair_s2_v3_007250 --trials 5 --root autotest` writes
  `results_test.csv`, no dashboard, no route. `python eval/autoscore/compare.py NEW_DIR OLD_DIR` compares two runs seed by seed.

## Not done by the scorer
Renders, gait / speed / turn tables, level 1 and step 2 routes, a second floor, anything on a GPU node.
