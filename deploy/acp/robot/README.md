# ACP on the real X2: what runs on PC2

Everything the robot runs comes from this folder, `deploy/acp/acp_deploy/` and `deploy/hiking/hiking_deploy/`. Why it is built
this way, and the bugs found on the way: `docs/findings/real_robot_instability_imu_frame.md`, `docs/FINDINGS.md` X4-X15.

## Layout on PC2 (`/home/run/x2-perception`, user `run`)

| Path on PC2 | Source |
|---|---|
| `deploy/acp/`, `deploy/hiking/` | this repo, same paths (copied with scp / rsync; nothing installed) |
| `models/acp/` `actor.npz`, `planner_attn.onnx`, `tracker.onnx`, `adapter_card.json` | [tinkerbuggy/x2-acp-contact-planner-tracker](https://huggingface.co/tinkerbuggy/x2-acp-contact-planner-tracker) (`tracker.onnx` is `models/acp_depth/export_tracker_onnx.py` of `actor.npz`, parity 1.8e-5) |
| `models/student.onnx` | the fine-map depth student `fine_v3`: `depth_student/student.onnx` on the same Hugging Face page (trained with `models/acp_depth/`) |
| `ws_robot/` | colcon workspace for the C++ command relay: `cmd_relay/` in `ws_robot/src`, then `colcon build --packages-select x2p_cmd_relay` |
| `log/` | launch log `ritual_fired.log`, black boxes, camera recordings (`video/run_*`) |

Python: SONIC's venv (`/home/run/gear-sonic/venv`) and the system ROS 2 Humble with AgiBot's `aimdk_msgs`.

## Start and stop

- `x2-pad-autopair.service` (SONIC's pad daemon) runs `ritual_start_x2perception.sh` on the gamepad ritual (L1+R1+L2+R2
  held 3 s, then Y). Its `ExecStart` line was switched from SONIC's `ritual_start_sonic.sh` on 2026-11-19; SONIC's line is kept
  commented out. Backups on PC2: `/home/run/gear-sonic/x2-pad-autopair.service.sonic.bak`, `.sonic_active.bak`,
  `/home/run/x2-perception/x2-pad-autopair.service.x2perception.bak`.
- The script starts, each in its own tmux session: depth publisher, pad bridge, gate, relay, MC stop (PC1's EM HTTP API),
  ACP node (`--imu-on-pelvis --relay-prefix /x2p/relay --accel cuda --blackbox`), face feed and camera recorder.
  Gamepad: two long buzzes = live, three short = refused (reason in `log/ritual_fired.log`).
- `x2perception_stop.sh` ends every session and can restart MC. The pad e-stop (A+X, then pump L2 and R2) makes the robot limp.

## Files

| File | Role |
|---|---|
| `ritual_start_x2perception.sh`, `x2perception_stop.sh` | launch and stop |
| `d435_pub.py` | chest D435 depth (V4L2 Z16) to `/x2_acp/chest_depth`, retries a hung capture |
| `gate.py` | refuses to start unless joints, IMU and depth are fresh (and the bus is silent after the MC stop) |
| `cmd_relay/` | C++ relay: the node's joint targets out at 500 Hz, limp after 0.2 s without fresh targets |
| `face_depth_capture.py`, `face_depth_feed.sh` | chest depth and the student's map on the face screen (PC3) |
| `cam_record.py` | head stereo and chest colour video with frame times on the black box's clock |
| `perception_view.py`, `render_perception.py` | what the deployed stack perceives, subscribe-only, rendered on the desktop |
| `bus_record.py`, `replay_blackbox.py` | bus recordings (e.g. SONIC baseline) and black-box replays |

Secrets stay off the repo: PC3's password in `~/.x2_pc3_password`, hosts in `~/.x2_pc3_host` / `~/.x2_pc1_host` on PC2.
