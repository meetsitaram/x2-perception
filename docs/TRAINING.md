# Training setup and compute

How the three policies were trained. Overview: [README](../README.md).

Each model trained on its own Nebius node with **one NVIDIA RTX PRO 6000 Blackwell (96 GB)**; evaluation ran on a desktop CPU
(MuJoCo, chunked and parallel).

| | ACP | Hiking in the Wild (X2) | SSR |
|---|---|---|---|
| Simulator | MuJoCo Warp | Isaac Lab (PhysX), InstinctLab | MuJoCo Warp |
| Parallel robots | 36,000 | 8,192 | 8,192 |
| Steps per robot per iteration | 24 | 24 | 24 |
| Samples per iteration | 864,000 | 196,608 | 196,608 |
| PPO | 5 epochs, 8 minibatches, lr 1e-4 (KL-adaptive) | WasabiPPO + AMP discriminator, 5 epochs, 4 minibatches, lr 1e-3 (KL-adaptive, target 0.01) | 5 epochs, 4 minibatches, lr 5e-4 x 4 optimisers (KL-adaptive) |
| Throughput | about 7.4 s/iteration (stage 1), 11.5-14 s (stages 2-3); 60-117k env-steps/s | about 9.6-9.8 s/iteration | about 6.5-6.8 s/iteration, about 30k samples/s |
| Policy input (terrain) | height scans (planner 17x11, tracker 9x7) | chest depth camera, 8-frame history | chest depth camera + GRU |
| Iterations reached | about 26,600 (stage 1 -> 2 -> 3; best at 23,000) | 27,000 (best at 24,500) | 32,750 (stopped) |
| Wall-clock time | about 91.5 h (paused) | about 91.5 h (paused) | about 82 h (stopped) |

Total: about **265 GPU-hours** on three RTX PRO 6000 GPUs, including restarts, rollbacks and abandoned segments. Every
segment, its settings and why it changed are in `models/<model>/LEDGER.md`; the single-run recipes are in
`models/<model>/RECIPE.md`.

## Fine-tune

Each model folder has the training code, the recipe and the ledger, and its README says how to resume from the Hugging Face
`checkpoint/`:

- ACP: [`models/acp/`](../models/acp/) - MuJoCo Warp trainer in `code/`, stage 3 resumed from `checkpoint/ckpt_023000.pt`.
- Hiking in the Wild: [`models/hiking/`](../models/hiking/) - X2 port in `x2_port/`, applied onto the `hiking-in-the-wild` /
  InstinctLab base code, resumed from `checkpoint/model_24500.pt`.
- SSR: [`models/ssr/`](../models/ssr/) - MuJoCo Warp trainer in `code/`, resumed from `checkpoint/r7_v20x_032750.pt`.
