# AgiBot X2 - Hiking in the Wild (perceptive parkour, depth camera)

Port of *Hiking in the Wild* (arXiv 2601.07718, InstinctLab) to the AgiBot X2 in Isaac Lab: a depth-camera policy (8-frame
depth history, 32x36 crop, chest camera) with an AMP style reward. Recipe v9: the paper's command mix, no extra air-time
reward, a fair terrain curriculum, action clip +-20 (waist pitch +-50). Released checkpoint: iteration 24,500 (best apparatus
checkpoint).

Weights and full training state: [tinkerbuggy/x2-hiking-in-the-wild](https://huggingface.co/tinkerbuggy/x2-hiking-in-the-wild) (CC BY-NC 4.0).

## Results (MuJoCo, held-out apparatus, 50 trials per leg)

| Leg (station to station) | 1 | 2 | 3 (ramps) | 4 (ramps) | 5 (stairs) |
|---|---|---|---|---|---|
| Successes / 50 | 48 | 49 | 42 | 37 | 41 |

Flat tests passed; 24 of 32 training-terrain courses (paper terrain types) passed; no stair refusals (FINDINGS H8-H9 in
[`docs/FINDINGS.md`](../../docs/FINDINGS.md)). Not yet qualified for the full route. The eval is defined in
[`docs/EVAL.md`](../../docs/EVAL.md). Trained in simulation only; not yet tested on the real robot.

## Files on Hugging Face

- `policy.onnx`, `0-depth_encoder.onnx`, `actor.onnx`; `adapter_card.json` - input/output specification; `obs_check.npz` -
  recorded inputs and outputs; `agent.yaml`, `env.yaml` - the exact agent and environment configuration of the run.
- `checkpoint/model_24500.pt` - full InstinctLab / instinct_rl training state (policy with depth encoder, critic, AMP
  discriminator, optimiser).

Run it in the MuJoCo eval with the `hiking:HikingHandover` adapter ([`eval/adapters/hiking.py`](../../eval/adapters/hiking.py));
see the top-level README, "Run a model in the MuJoCo eval".

## Fine-tune

- Code: [`x2_port/`](x2_port/) holds the X2 files (task files, supervisor, launch, export and evaluation scripts, the MuJoCo
  port in `x2_port/sim2sim/`). The base code is the `hiking-in-the-wild` repository (branch `g1-20k-and-extended-terrain`,
  commit `27ec8d3`) plus InstinctLab, instinct_rl and Isaac Lab; `x2_port/apply.sh` copies the X2 files into it.
- Task `Instinct-Parkour-ExtendedFixed-Amp-X2-v0`; resume with `run_train.py --task=... --num_envs 8192 --resume` pointing at
  `checkpoint/model_24500.pt`.
- [`RECIPE.md`](RECIPE.md) - recipe v9 as one uninterrupted run from scratch. [`LEDGER.md`](LEDGER.md) - every run segment.
- Data not included: the X2-retargeted *Hiking in the Wild* reference motions (CC BY-NC 4.0 release) and the user's extended
  terrain map ([`docs/DATA.md`](../../docs/DATA.md)).

## Credits

Built on InstinctLab / *Hiking in the Wild* (Zhu, Zhuang, Zhao, Lee, Zhao; arXiv 2601.07718), whose code and release are
CC BY-NC 4.0; the style reference is the paper's released motion data retargeted to the X2, plus BONES-SEED (Bones Studio)
clips used in earlier training segments. **Licence of these weights: CC BY-NC 4.0 (non-commercial).**
