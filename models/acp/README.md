# AgiBot X2 - Adaptive Contact Planning (planner + tracker)

Port of *Adaptive Contact Planning* (arXiv 2610.31608) to the AgiBot X2: an attention contact planner and a contact-conditioned
whole-body tracker, trained in three stages (tracking, planner co-training with the paper's annealing, goal terrains) in
MuJoCo Warp. Released checkpoint: `pair_s3_v2_023000` (stage 3, iteration 23,000).

Weights and full training state: [tinkerbuggy/x2-acp-contact-planner-tracker](https://huggingface.co/tinkerbuggy/x2-acp-contact-planner-tracker) (Apache 2.0).

## Results (MuJoCo, held out)

| Test | Result |
|---|---|
| Full apparatus route ("ours", 22 waypoints, 5 stations) | **198/200 (99%)**; 193/200 (96.5%) on 200 fresh seeds |
| NIST apparatus route (35 waypoints, 20-21 cm stairs) | 155/200 (77.5%); most failures are falls on the ramps |
| Hold-out gate (turn-arounds, ramp troughs) | 5/5 |

The eval is defined in [`docs/EVAL.md`](../../docs/EVAL.md). **Important:** the policy consumes terrain **height scans**
(planner 17x11, tracker 9x7) read from the simulator. On a real robot these must come from a mapping pipeline (the paper uses
a LiDAR neural map); a depth-camera height-map estimator is planned but not built
([`docs/TODO_depth_to_heightmap.md`](../../docs/TODO_depth_to_heightmap.md)). Trained in simulation only; not yet tested on
the real robot.

## Files on Hugging Face

- `actor.npz` - tracker weights (numpy); `planner_attn.onnx` - contact planner; `adapter_card.json` - input/output
  specification (joint order, scales, observation layout, plant gains); `meta.json`; `obs_check.npz` - recorded inputs and
  outputs to check a re-implementation against.
- `checkpoint/ckpt_023000.pt` - full stage-3 training state (tracker, planner, critics incl. the goal critic, optimisers,
  annealing state); `checkpoint/launch_flags.txt` - the exact trainer flags.

Run it in the MuJoCo eval with the `acp:ACPHandover` adapter ([`eval/adapters/acp.py`](../../eval/adapters/acp.py)); see the
top-level README, "Run a model in the MuJoCo eval".

## Fine-tune

- Code: [`code/`](code/) (MuJoCo Warp; package `acp`, clip-set definitions in `code/cfg/`, supervisor `code/supervise.sh`).
  Resume stage 3 with `acp.train --stage 3 ... --init checkpoint/ckpt_023000.pt` and the flags of `checkpoint/launch_flags.txt`.
- [`RECIPE.md`](RECIPE.md) - the best recipe written as one uninterrupted run from scratch, every setting with its value.
- [`LEDGER.md`](LEDGER.md) - every run segment of this round (starts, restarts, fall-backs) with what changed and why.
- Training data is not included: BONES-SEED (Bones Studio) retargeted to the X2, and the author's Pico motion captures in
  [meetsitaram/terrain-park](https://github.com/meetsitaram/terrain-park) (`data/2026-10-29`, `data/2026-10-30`: raw takes;
  `data/train_sets`: X2-retargeted training sets; `data/style`: style clips). Data provenance: [`docs/DATA.md`](../../docs/DATA.md).

## Credits

Method: Adaptive Contact Planning (arXiv 2610.31608). Reference motions: BONES-SEED by Bones Studio (retargeted to the X2) and
the author's Pico captures. Licence of these weights: Apache 2.0.
