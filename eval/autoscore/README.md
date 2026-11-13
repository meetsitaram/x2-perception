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
2. ACP only: the hold-out gate (seeds 1001 1005 1036 turn-around, 1019 1022 ramp trough; PASS = 3/3 turn-around and at least 4/5).
   Legs only if it passes.
3. Flat gate: `flat_walk`, `flat_stop_stand`, `flat_turn_in_place`, `flat_turn_and_return`, 50 each (one process per test).
4. Station legs 1-5, 50 each (one process per leg).
5. Full route 200 (seeds 1000-1199, 4 x 50) only if every leg is at least `route_min_leg` (45) of 50 (user's rule).

At most `max_cores` (3) processes at once, one per core (OMP threads 1). Each process is a 9.5-minute slice that resumes from
`partial.json`; a finished test, leg or slice (`summary.json`) is never run again, also after a restart. Track settings:
ACP goal 0.20 + lookahead 0.30; SSR standard command; Hiking card command.

Outputs: `eval/out/<track>/<checkpoint>/auto/` (`gate/`, `flat/step1/`, `legs/` with `legs.txt`, `route/s1000..s1150/`, `logs/`).

## Results
- `results.csv`: one row per export: track, checkpoint, floor, the four flat tests, flat gate, ACP gate, legs 1-5 (strict
  successes / trials), product of the leg rates, full route (k/200, Wilson 95 % interval, sections entered/completed), flags
  (hand-over check failure, joint-limit / trapped-foot / limb-clamp trial counts over legs and route, failed jobs, legs not run),
  finish time (UTC), minutes, output folder.
- A full route also writes a row `auto-<track>-<name>-trim` into `dashboard/data/apparatus_results.json` and runs
  `python prepare.py` in `dashboard/` (retried on OSError). The upload to the dashboard artifact (README steps 3-6) stays with the
  coordinator. Legs-only results are not put on the dashboard: its bars mean full-route success. `tracks.json` is not touched.
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
  `max_cores`, `trials`, `route_min_leg`, `poll_minutes`.
- Test mode: `python eval/autoscore/autoscore.py --test --only acp:pair_s2_v3_007250 --trials 5 --root autotest` writes
  `results_test.csv`, no dashboard, no route. `python eval/autoscore/compare.py NEW_DIR OLD_DIR` compares two runs seed by seed.

## Not done by the scorer
Renders, gait / speed / turn tables, level 1 and step 2 routes, a second floor, anything on a GPU node.
