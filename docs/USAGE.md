# Using the models

Download, evaluate and inspect the policies. Overview: [README](../README.md).

## Models on Hugging Face

Policy exports with model cards on Hugging Face (simulation-trained; not yet tested on the real robot):

| Model | Hugging Face | Checkpoint | Licence | In this repo |
|---|---|---|---|---|
| ACP (planner + tracker) | [tinkerbuggy/x2-acp-contact-planner-tracker](https://huggingface.co/tinkerbuggy/x2-acp-contact-planner-tracker) | `pair_s3_v2_023000` (99% route) | Apache 2.0 | [`models/acp/`](../models/acp/) |
| Hiking in the Wild (X2) | [tinkerbuggy/x2-hiking-in-the-wild](https://huggingface.co/tinkerbuggy/x2-hiking-in-the-wild) | iteration 24,500 (best) | CC BY-NC 4.0 | [`models/hiking/`](../models/hiking/) |
| SSR | [tinkerbuggy/x2-ssr](https://huggingface.co/tinkerbuggy/x2-ssr) | `r7_v20x_032000` (final) | CC BY-NC 4.0 | [`models/ssr/`](../models/ssr/) |

Each Hugging Face repo holds the ONNX / numpy weights, the full training checkpoint and configs under `checkpoint/` (for
fine-tuning), `adapter_card.json` (input and output specification) and `obs_check.npz` (recorded inputs and outputs). The
evaluation adapters in [`eval/adapters/`](../eval/adapters/) show how to run them in MuJoCo.

**Data credits.** Motion references include **BONES-SEED by Bones Studio** (retargeted to the X2) and the author's own Pico
captures ([meetsitaram/terrain-park](https://github.com/meetsitaram/terrain-park): raw takes in `data/2026-10-29`,
`data/2026-10-30`; X2 training sets in `data/train_sets`; style clips in `data/style`). Hiking and SSR also use the *Hiking in
the Wild* (InstinctLab) release, licensed CC BY-NC 4.0, so those two models are non-commercial. Methods: Adaptive Contact
Planning (arXiv 2610.31608), Hiking in the Wild (arXiv 2601.07718), SSR (arXiv 2605.30770). Data provenance:
[`docs/DATA.md`](DATA.md).

## Run a model in the MuJoCo eval

The eval runs on CPU with Python, `mujoco`, `numpy` and `onnxruntime`. It needs two inputs that are not in this repo: the X2
MJCF (`PA_ROBOT_DIR` = a folder holding `mjcf/x2_ultra.xml`) and the apparatus rasters and routes (`PA_APPARATUS_DIR`, or
`eval/data/apparatus/`); see [`docs/DATA.md`](DATA.md). The NIST apparatus files are not published.

1. Download a model from Hugging Face into a folder, e.g. `huggingface-cli download tinkerbuggy/x2-ssr --local-dir hf/x2-ssr`.
2. Point the model's adapter at that folder and run the harness (one attempt per seed at the full apparatus route) or the
   station-to-station legs:

```bash
# ACP: folder with actor.npz, planner_attn.onnx, adapter_card.json
ACP_HANDOVER_DIR=hf/x2-acp-contact-planner-tracker ACP_GOAL_R=0.2 ACP_LOOKAHEAD=0.3 X2_EXCLUDE_PAIRS=hip \
  python eval/harness.py --adapter acp:ACPHandover --apparatus ours --trials 10 --out eval/out/acp_route

# Hiking: folder with policy.onnx, 0-depth_encoder.onnx, adapter_card.json, obs_check.npz
HIKING_HANDOVER_DIR=hf/x2-hiking-in-the-wild HIKING_CMD=card X2_EXCLUDE_PAIRS=hip \
  python eval/legs.py --adapter hiking:HikingHandover --apparatus ours --trials 50 --out eval/out/hiking_legs

# SSR: folder with policy.onnx, adapter_card.json, obs_check.npz
SSR_HANDOVER_DIR=hf/x2-ssr SSR_CMD=standard X2_EXCLUDE_PAIRS=hip \
  python eval/legs.py --adapter ssr:SSRHandover --legs 5 --trials 20 --out eval/out/ssr_leg5
```

`eval/harness.py` options: `--adapter <module>:<Class>` (or `zero`, `stand`), `--apparatus ours|nist|flat`, `--trials N`,
`--seed0 S`, `--protocol randomised|nominal`, `--seconds T`, `--stand`, `--kinematic` (route check without physics),
`--keep-traj picked|all`, `--wall-limit S`, `--label TEXT`, `--out DIR` (required). `eval/legs.py` takes `--adapter`, `--out`,
`--apparatus`, `--legs 1 2 ...`, `--trials` (default 50), `--seed0`, `--wall-limit`, `--label`, `--chunk K --chunk-size N`.
Output: `results.json` (every trial), `summary.json` (the table row) and `traj/<seed>.npz`, which `eval/render.py` turns into
videos. The environment variables above are the settings the automatic scorer uses (`eval/autoscore/autoscore.py`); every
adapter's other options are in its module docstring. Training-terrain courses: `eval/ladder.py`.

## Repository layout

| Path | What |
|---|---|
| `models/acp/`, `models/hiking/`, `models/ssr/` | per model: `README.md` (model card), `RECIPE.md`, `LEDGER.md` and the training code; `models/LAST_SYNC.txt` is the time of the last code sync |
| `eval/` | the common MuJoCo eval: `harness.py` (full route), `legs.py`, `ladder.py` (training-terrain courses), `adapters/`, `render.py`, `autoscore/` (automatic scorer), `metrics/` (training curves) |
| `docs/` | `EVAL.md` (eval definition), `DATA.md`, `FINDINGS.md` (defects found and fixed), `findings/` (paper-vs-code audits, surveys), `camera/` (training camera reference), `hardware/chest_camera_mount/` (3D-printable RealSense chest brackets for Hiking and SSR), `TODO_depth_to_heightmap.md`, `site/` (static review page), `media/` |
| `tools/simcheck/` | simulator fault checks run before every training launch |
| `agentic/` | the agent team that built this: instructions, rules, state, dashboard, node scripts |
| `.claude/agents/` | the agent definitions (Claude Code reads them here) |
