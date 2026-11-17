# ACP on a depth camera: the fine-map student

ACP (contact planner + tracker, `models/acp/`) needs two height scans of the floor around the robot: the planner's 17 x 11
and the tracker's 9 x 7 (0.1 m, heading frame, height minus root height). In training and in the eval they come from the
simulator. On the real X2 the only usable terrain sensor is an added RealSense D435 on the chest bracket
(`docs/hardware/chest_camera_mount/`). This folder trains a **student that builds those scans from the chest depth camera and
the robot's proprioception**, under the unchanged ACP. Plan, history and every result: `docs/TODO_depth_to_heightmap.md`;
findings: `docs/FINDINGS.md` D-depth1, B-app1, B-acp1.

## Result (2026-11-17, our apparatus, seeds 2000-2049)

| | clean sim depth | D435-like noise in the loop |
| --- | --- | --- |
| ACP with perfect scans | 46/50 | |
| grid student d1 (10 cm scans) | 46/50 | 35/50 |
| **fine-map student fine_v3** | 41/50 | **46/50** |

4x4 (8.9 cm): fine_v3 estimates 9.2 cm in the tracker scan, 7.7 cm in the planner scan (d1: 4.1 / 5.6).

## How it works

`fine_student.py`: the depth pixels are projected through the camera pose (on the robot: IMU attitude + leg kinematics; no
terrain information) into a 52 x 44 bird's-eye grid of 2.5 cm cells around the robot (max / mean height, point count, seen);
the grids of 4, 8 and 12 camera frames before (0.5 s at 25 Hz) are its short memory (the 4x4 under the feet); a small U-Net
turns them, with proprioception, into a fine height map; ACP's 187 + 63 points are read off it. TorchScript, CPU.

Why fine: a 4x4 (8.9 cm wide) on ACP's 10 cm sample points flickers in the labels, so a student that predicts the 10 cm
scans directly hedges at half height. Projection makes the beam's place and height a matter of geometry (projected depth vs
the true map: 0.5 mm median).

## Files

| File | What |
| --- | --- |
| `../../eval/adapters/acp_depth.py` | The eval adapter: ACP unchanged; its two scans recorded (teacher data) or replaced by the student (`ACPD_MODE=record / student / dagger / student_plan / student_track / noback / track_err`); chest camera ray-cast with the robot's own body; D435-like noise in the loop (`ACPD_DEPTH_NOISE=1`); foot anchoring (`ACPD_ANCHOR=1`); keeps the arrival heading on heading-less holds; overlay data for renders (`ACPD_EXTRA`) |
| `collect.py` | Teacher / DAgger data: ACP walks terrain-park `extended_v4k` training tiles (the apparatus and eval courses stay held out), parallel workers |
| `fine_student.py`, `train_fine.py` | The fine-map student and its trainer (local GPU) |
| `student.py`, `train.py` | The first, grid student (10 cm scans; kept for comparison; `train.d435` is the noise model) |
| `region_error.py` | Grid-student error by region (behind / under the feet / ahead, flat / terrain) |
| `render_showcase.sh`, `render_vis.py`, `align_apparatus.py` | The standard renders: ACP's overlay from the maps it was given, the chest depth inset, our apparatus with its real textures, five clips |
| `run_v1.sh`, `run_dagger.sh`, `run_fine.sh`, `run_fine3.sh` | The unattended chains that produced the results above |

Runs, data and checkpoints go to `models/acp_depth/runs/` (git-ignored; ~8 GB). The ACP export is
`tinkerbuggy/x2-acp-contact-planner-tracker` on Hugging Face.

## Reproduce (this PC: RTX 5090, 32 cores)

```bash
python3 models/acp_depth/collect.py --out models/acp_depth/runs/data_v3 --per-type 35 \
  --extra single_beams:35,beam_fields:35,log_corridors:25 --seeds 2 --workers 14
ACPD_MODE=dagger ACPD_DEPTH_NOISE=1 ACPD_STUDENT=<previous student.pt> python3 models/acp_depth/collect.py --out ...
systemd-run --user --scope -p MemoryMax=28G python3 models/acp_depth/train_fine.py --data <dirs> --out models/acp_depth/runs/fine_vN --epochs 150
models/acp_depth/render_showcase.sh models/acp_depth/runs/fine_vN/student.pt models/acp_depth/runs/vis_fineN "label"
```

## Open

- The real D435 (captures from PC2), Orin CPU timing, a share of clean frames in training, the robot-side node.
- Which apparatus geometry is real (eval vs CAD, `docs/FINDINGS.md` B-app1).
