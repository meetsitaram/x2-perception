# ssr-x2

SSR ("Scaling Surefooted and Symmetric Humanoid Traversal to the Open World", Yu et al., arXiv 2605.30770) for the AgiBot X2 Ultra.
The authors released no code and no motion data, so this is a reimplementation from the paper, on the MuJoCo Warp stack
used for SuperMimic and ACP. It stands alone: everything it needs is in this repo.

| Path | What |
| --- | --- |
| `ssr/env.py` | Batched MuJoCo Warp environment: terrain tiles, depth, commands, Table 6 rewards, domain randomization |
| `ssr/terrain.py` | The five terrain types, 20 levels, and the GPU depth ray-march |
| `ssr/networks.py`, `ssr/mirror.py` | Equivariant encoder, MoE actor, three critics, foothold model, discriminators, mirror transforms |
| `ssr/train.py` | Single-stage training |
| `scripts/build_amp.py` | Per-terrain motion priors from BONES-SEED and the Pico captures (`terrain-park`) |
| `scripts/route_eval.py`, `scripts/render_route.py` | The apparatus route, and the video with what the policy uses |
| `deploy/` | ONNX export, the ROS 2 node for the real X2, offline deploy test |
| `tests/test_mirror.py` | Mirror and equivariance checks, each with a case that must fail |
| `docs/PAPER_NOTES.md` | Every paper setting, and every choice the paper leaves open |
| `ISSUES_LOG.md` | What went wrong and how it was fixed |

```bash
python scripts/build_amp.py
```
```bash
python -m ssr.train --num_envs 4096 --iters 20000 --name v1
```
```bash
python scripts/route_eval.py --ckpt runs/v1/last.pt --num 50 --out runs/v1/route
```
```bash
python scripts/render_route.py runs/v1/route/best.npz route.mp4 "SSR on the route"
```
```bash
python deploy/export_onnx.py --ckpt runs/v1/last.pt --out deploy/ssr_x2.onnx
```
```bash
python deploy/sim2sim.py --onnx deploy/ssr_x2.onnx --scene stairs
```

Use `C:\Users\PC\.venvs\sm` (mujoco_warp 3.17, torch, onnx). On the robot, `deploy/ssr_x2_node.py` runs as a dry run unless given `--arm`.

Motion priors use BONES-SEED: Motion Data by Bones Studio (bones.studio).
