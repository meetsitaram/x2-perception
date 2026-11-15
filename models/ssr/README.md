# AgiBot X2 - SSR (surefooted and symmetric humanoid traversal)

Port of *SSR* (arXiv 2605.30770) to the AgiBot X2 in MuJoCo Warp: depth camera + GRU policy with five style discriminators.
Released checkpoint: `r7_v20x_032000` (final; training was stopped at 32,750).

Weights and full training state: [tinkerbuggy/x2-ssr](https://huggingface.co/tinkerbuggy/x2-ssr) (CC BY-NC 4.0).

## Results (MuJoCo, held-out apparatus, 50 trials per leg)

| Leg | 1 | 2 | 3 | 4 | 5 (stairs) |
|---|---|---|---|---|---|
| Successes / 50 (`r7_v20x_032000`) | 49 | 50 | 47 | 49 | 0 |

Known limitation: the policy refuses the 18.6 cm apparatus stairs (it stops at the foot); cause analysed in
[`docs/FINDINGS.md`](../../docs/FINDINGS.md) (E8-E8c). The best earlier checkpoint, `r7_v20_final_26000`, scored legs
49/50/49/44/5. The camera in this port is chest-mounted; the paper mounts it at the waist. Trained in simulation only; not yet
tested on the real robot.

## Files on Hugging Face

- `policy.onnx`; `adapter_card.json` - input/output specification; `obs_check.npz` - recorded inputs and outputs.
- `checkpoint/r7_v20x_032750.pt` - full training state (policy and GRU, critic, five style discriminators, optimisers,
  curriculum levels); `checkpoint/args.json` and `r7_flags_v20.txt` - the exact trainer arguments.

Run it in the MuJoCo eval with the `ssr:SSRHandover` adapter ([`eval/adapters/ssr.py`](../../eval/adapters/ssr.py)); see the
top-level README, "Run a model in the MuJoCo eval".

## Fine-tune

- Code: [`code/`](code/) (MuJoCo Warp; package `ssr`, its own [`code/README.md`](code/README.md)). Resume with
  `scripts/supervise_node.py --name r7 --iters N --num_envs 8192 -- $(cat r7_flags_v20.txt)` with the checkpoint as
  `runs/r7/last.pt`. Fix the stair refusal (separate advantage normalisation, FINDINGS E8c) before fine-tuning.
- [`RECIPE.md`](RECIPE.md) - the recipe as one uninterrupted run from scratch. [`LEDGER.md`](LEDGER.md) - every run segment.
- Data not included: the AMP style set (`amp_v5`: the author's Pico captures in
  [meetsitaram/terrain-park](https://github.com/meetsitaram/terrain-park), plus the *Hiking in the Wild* reference, CC BY-NC
  4.0); see [`docs/DATA.md`](../../docs/DATA.md).

## Credits

Method: SSR (arXiv 2605.30770). Style data: the author's Pico motion captures and the *Hiking in the Wild* released motion
data retargeted to the X2 (CC BY-NC 4.0); BONES-SEED (Bones Studio) clips in earlier segments. **Licence of these weights:
CC BY-NC 4.0 (non-commercial).**
