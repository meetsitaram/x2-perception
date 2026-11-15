# Review dashboard

The user's review page (2026-11-13 on): https://claude.ai/artifact/KcRGuVx3sH1nV9FdsUdjGi (slug 96b67cc3-34e8-4fe5-a22e-30be43693b07), a self-contained snapshot built by the session scratchpad's `make_snapshot.py` into `dashboard/snapshot/` (git-ignored) and republished with publish_files (slug + base_version). The older Data-type page https://claude.ai/artifact/JaZDeLFXXN28HLMd1Sia3V (a claude.ai Artifact of the "Data" type, private to the user).
This folder is its source of truth. Nothing here is published as a file: the page and each dataset are documents in the artifact's
store, written with the `ArtifactData` tool, and each dataset points at a JSON file uploaded with the `Artifact` tool.

| Here | In the artifact |
|---|---|
| `index.html` | document `files/index.html` (`{text, edit}`): the whole page, dark by design, seven tabs |
| `data/<id>.json` | uploaded file, referenced by document `datasets/<id>` |
| `images/*.jpg` | uploaded images (at most 1600 px), referenced by `url` in `data/renders.json` |
| `store.json` | title, description and current upload id (`blob`) of every dataset |
| `prepare.py` | checks the data and writes the store documents into `build/` |
| `preview.html` | local preview with a stand-in for the page's `dash` API (not published) |

## Update the data (after every agent report)

1. Edit the rows in `data/*.json` (see "What each file drives").
2. `python prepare.py`. It stops with a list if a row is missing a field or uses a value the page does not know.
3. For each data file you changed, upload it: `Artifact` with `url` (above), `file_path` (the JSON file), `asset: true`. One file per call.
   Put the 32-character id it returns into `store.json` as that dataset's `blob`.
4. `python prepare.py` again (it writes `build/ds_<id>.json` with the new upload id and the time).
5. Read the current versions: `ArtifactData` `list`, collection `datasets`.
6. Write: `ArtifactData` `batch`, one entry per changed dataset:
   `{op: "set", collection: "datasets", doc_id: "<id>", file_path: ".../dashboard/build/ds_<id>.json", if_version: <version from step 5>}`.
7. Check: `ArtifactData` `list`, collection `data/users/me`. `check` should say `passed` with an `at` later than your write. A `runs`
   document appears only when something is wrong (a file that did not load, code that threw, a number on the page that differs
   from its data). These documents are written only while the user has the page open.

New picture: resize to at most 1600 px into `images/`, upload it (`Artifact`, `url`, `asset: true`, `file_paths` takes several
images), add a row to `data/renders.json` with the returned `/_blob/<id>` as `url`, then steps 2 to 7 for `renders`.

## Change the page

Edit `index.html`, `python prepare.py`, then `ArtifactData` `list` collection `files` for the version and
`ArtifactData` `set` collection `files`, doc `index.html`, `file_path` `build/file_index.json`, `if_version`. Then step 7.
Preview first: `python -m http.server 8765` in this folder, open `http://localhost:8765/preview.html`
(`?sample` adds made-up X2 runs and scores to exercise the charts; never publish those).

Rules of the page format (from the type's own guide): the page opens with `<meta name="dash-one-writer">`; fixed words are in the
HTML and scripts only fill elements marked `data-draw`; every figure carries `data-source`/`data-field`; styles only on the
`pa-` classes; values are set with `textContent`; a chart has a tooltip. The dark colours are set on `.pa` at the top of the
`<style>` block, so the page is dark in every viewer theme (the toolbar around it still follows the viewer's theme).

## What each file drives

- `tracks.json`: the four status cards and the headline sentence. `id` is `acp`, `ssr`, `hiking` or `eval`. Set `state` (the words on
  the pill) and `state_kind` (`setup`, `training`, `evaluating`, `blocked`, `paused`, `done`), `running`, `checkpoint`, `next`,
  `blocker` (`"None"` and `blocked: false` when there is none), `watch`, and `updated_at` (UTC, `2026-11-11T19:39:00Z`).
- `progress.json`: one row per recorded training point. `run` is `acp`, `ssr`, `hiking` (X2 runs, `robot: "X2"`) or `g1`. A run's
  panel on the Training tab turns from "No training run yet" into charts as soon as it has a row. `reward`, `terrain_level` and
  `episode_len` may be `null`.
- `apparatus_results.json`: one row per eval run: `id`, `date`, `track` (`ACP`, `SSR`, `Hiking`), `checkpoint`, `apparatus`
  (`ours` or `nist`), `simulator`, `trials`, `success_pct`, and optionally `door_pct`, `stepovers_pct`, `ramps_pct`, `stairs_pct`,
  `note`. The newest row per track and apparatus fills the bars on the cards and in the goal chart. It is empty now.
- `train_terrain.json` (2026-11-13, not registered in `prepare.py` / `store.json` yet, not on the page): success per track,
  checkpoint, terrain type and level on MuJoCo copies of each track's own training terrain (`eval/train_terrains.py`), with the
  highest level passed (>= 80 %). Written by the automatic scorer after every export from `eval/autoscore/train_results.csv`;
  at a refresh rebuild it with `python eval/autoscore/autoscore.py --build-train-dataset`. To show it, add it to `REQUIRED` in
  `prepare.py` and to `store.json`, then the upload steps above.
- `targets.json`: the 99% goal and the "about 25%" starting figure. Change the baseline row when the eval monitor re-measures it.
- `g1_results.json`: the G1 reproduction against the paper (chart and table on the Eval tab).
- `details.json`: what goes into each training. `origin_kind` is `paper` (left column), `papers` or `ours` (right column, "Added");
  `status_kind` is `done`, `partial`, `planned` or `risk`.
- `robot.json`: robot, retargeted motion, props and camera cards. `status_kind` is `ready`, `partial`, `todo` or `missing`;
  `amount` is an optional large figure such as "about 67 min".
- `renders.json`: the galleries. `group` is `map` (terrain previews), `robot` (robot in a simulator) or `motion` (retargeted X2
  motion to review, on the Robot tab; none yet). `featured: true` puts a picture on the Overview. `label` says where the picture
  comes from ("Preview from the generator scripts, not from a simulator", "MuJoCo, not the training simulator").
- `open_items.json`: `who_kind` `you` puts an item under "Waiting on you" and in the count on the tab.
- `worklog.json`: newest first.
- `eval_rules.json`: the common eval definition shown on the Eval tab; keep it in step with `docs/EVAL.md`.

## Standing rules for what goes on the page

- Robot and motion pictures carry `credit: "Motion data: Bones Studio"` (`prepare.py` refuses them without it). No dataset clip names.
- MuJoCo frames are labelled as MuJoCo, not the training simulator. Terrain pictures are labelled as previews, not simulator output.
- No street address, house number or home photos. No made-up numbers: leave a dataset empty and let the page say what will appear.
- Videos are not embedded; use stills.

## One run per chart (2026-11-12)

The Training tab draws every `progress.json` row of a `run` on one line, so rows from different runs of the same track (each
with its own reward scale) make the line jump. Keep only the current run's rows for `ssr` and `hiking`; when a track starts a
new run, move the old run's rows to `progress_archive_<track>_<runs>.json` in this folder (not a dataset, not shown). ACP's
stages share one iteration counter and stay together.
