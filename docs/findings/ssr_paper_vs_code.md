# SSR paper vs the live SSR code (audit 2026-11-13)

Paper: "SSR: Scaling Surefooted and Symmetric Humanoid Traversal to the Open World", arXiv 2605.30770, HTML version, read in full
(Sec. 1-6, App. A-E, Tables 1-15, Eq. 1-9, Algorithm 1).
Code: the live copy `/mnt/filesystem-g4/tracks/ssr/code/` as of 2026-11-13 (env.py 1131 lines, train.py 564, networks.py 250,
x2.py 208, terrain.py 380, mirror.py 71, plant_config.py, scripts/build_amp.py, build_amp_v4.py), plant
`config/robot_plant/x2_ultra_mc_20261024.yaml`, motion prior `data/amp_v4.npz`.
Flags: the running job (r7, PID read from `/proc`, iteration 6310 at audit time) = `runs/r7_flags_v14.txt` + `--num_envs 8192
--iters 20000 --resume runs/r7/last.pt`, env `X2_HF_CHUNK=400`, launched by `runs/queue31.sh` through `scripts/supervise_node.py`.
Repeated flags: argparse keeps the last value, so the effective values are `--omni_frac 0.3 --omni_turn 0.5 --prior_min_speed 0.05
--ext_frac 0.4`. Effective defaults read from `runs/r7/args.json`: rsi_frac 0.5, init_std 1.0, rew_scale 0.02, adv_norm joint,
gate_prior 0, ent 0.005, map_memory off, boot_iters 0, action_scale_paper off, default_from_ref off, dof_dev_ref off.
Live state at it 6310: w_style 0.2 (full), guide_on True, obstacle share 0.771, sig_v 0.1, levels by type [14.2, 10.9, 14.1, 9.9, 13.8].

Status labels:
- **MATCH**
- **DEV-J**: deviation, justified (paper silent, or impossible / different on X2, MuJoCo Warp or the user's hardware, or a recorded user rule)
- **DEV-P**: deviation, paper specifies, not followed. The last column says what following it would take.
- **ADDED**: not in the paper
- **UNVERIFIED**: could not be checked (says why)

Prior audit `agentic/state/ssr_port_audit.md` (2026-11-12) was used as a checklist. Where it differs from this one: (1) it called the
orientation term a deviation, but the paper's Table 6 has the squared norm, exp(-norm(g_xy)^2/sigma_g), which is what the code
does (MATCH); (2) it accepted "data not released" for the motion prior, but the paper's Table 11 sizes and composition can be
followed with our own data and are not (section 12); (3) it did not list the missing gap and discrete terrains, the 17-row base
height map, the per-group advantage normalisation of the cited multi-critic method, 8192 envs, or the vy clamp on terrain commands.

---

## 1. Observations

| Item | Paper (ref) | Code (file:line or flag) | Status | To follow |
|---|---|---|---|---|
| Proprioception content | o^p in R^72 = omega, g, u, theta, theta_dot, a_{t-1} (Sec. 3.1) | `env.py:582` cat(w, g, cmd, q-q0, qd, a), PROP_DIM 72 `env.py:21` | MATCH | |
| Joint positions relative to default | paper silent (Table 13 uses theta) | q_rel = q - q0 `env.py:576` | DEV-J | |
| History length h | "short history", h not given (Sec. 3.1, Eq. 7) | H = 5 `env.py:19` | DEV-J | |
| Depth image size | 36 x 36 (Sec. 3.1, App. D) | DEPTH_N 36 `env.py:25` | MATCH | |
| Depth range / normalisation | d_min, d_max in Alg. 1, values not given | 0.1 to 3.0 m (chest_rs dmax), mapped to [0, 1] `env.py:517`, `x2.py:46` | DEV-J | |
| Actor input | o^a = [o^p (current), v_hat, z_hat] (Sec. 3.1, Table 12) | `train.py:276` cat(env.prop, vh, e["z"]) | MATCH | |
| Encoder input | proprio history + depth (Sec. 3.1) | `networks.py:136-157` hist (5 x 72) + depth | MATCH | |
| Critic privileged state content | o^p, v, v^f, c, p^h, p^f, H^f, H^b (Sec. 3.1, Table 12) | `env.py:602`, PRIV_DIM `env.py:23` | MATCH | |
| Base height map size | H^b is 18 x 10 (Table 12) | 17 x 10 (HB_NX 17, x from -0.6 to 1.0 m at 0.1 m) `env.py:22,174` | DEV-P | HB_NX = 18 (e.g. -0.7..1.0 m); changes critic, decoder and mirror sizes, so a fresh run (or re-init of critics / dec_b) |
| Foot height maps | 2 x 10 x 5, sole patch 22.5 x 10 cm every 2.5 cm (Table 12, Sec. 3.2) | HF 10 x 5, SOLE_HALF (0.1125, 0.05) `env.py:22,176`, `x2.py:55` | MATCH | |
| Height map values | paper silent (relative or absolute) | base height minus terrain (incl. side obstacles), foot sole height minus terrain, clipped `env.py:590,596` | DEV-J | |
| Hand / foot positions | p^h, p^f in base frame (Table 12) | wrist_roll_link, ankle_roll_link origins in base frame `env.py:598-599` | MATCH | |
| Contact state | c (Table 12) | foot normal force > 5 N `env.py:28,601` | DEV-J (threshold silent) | |
| Observation noise | paper silent | uniform noise on omega 0.2, g 0.05, q 0.01, qd 0.5 (train) `env.py:579-581`; also in the critic's copy of o^p | ADDED | |
| Observation normalisation | paper silent | running mean / std on actor, critic, foothold and disc inputs `networks.py:169-187`; none on the encoder (inputs clipped +-40) | ADDED | |
| Map memory input | not in paper | `--map_memory` off in r7 | ADDED (off) | |

## 2. Actions and control

| Item | Paper (ref) | Code | Status | To follow |
|---|---|---|---|---|
| Action = 21 joint position targets | a_t in R^21 (Sec. 3.1, App. D) | NAMES21 `x2.py:26-27`; target = q0 + scale x a `env.py:651` | MATCH | |
| Joint order and blocks | [R leg, L leg, waist, R arm, L arm], leg (hip p/r/y, knee, ankle p/r), arm (sh p/r/y, elbow) (App. B.1) | `x2.py:24-27` | MATCH | |
| Other 10 joints | not in paper (X2 has 29 joints, policy drives 21) | waist pitch/roll, wrists, head held at default by PD `x2.py:125-130` | DEV-J | |
| Control rate | 50 Hz (Sec. 4.1, App. D) | CONTROL_DT 0.02 `x2.py:20-21` | MATCH | |
| Target clipping | paper silent | clamp to joint range `env.py:651`; action clamp +-8 `train.py:42,290`; policy mean bounded 8 tanh(x/8) `networks.py:201` | ADDED | |
| Action delay | U(0, 2) control steps (Table 10) | randint(0, 3) per episode, ring buffer `env.py:457,649-650` | MATCH | |

## 3. Robot, PD gains, action scale (Table 14)

Code gains from the plant `mc_20261024` (substring match, first key wins, `plant_config.py:83-87`); action scale =
0.25 x effort / (armature x (2 pi 10 Hz)^2) `plant_config.py:117-131`. Recorded user rules: gains = mc_20261024 (USER_RULES l.84);
action scale parked on the review-later list (USER_RULES l.130-131).

| Joint | Paper Kp / Kd / scale (Table 14) | Code Kp / Kd / scale | Status | To follow |
|---|---|---|---|---|
| Hip pitch | 120 / 4 / 0.25 | 100 / 4 / 0.261 | DEV-J (user plant rule) | plant with Table 14 gains; scale table 0.25 (`--action_scale_paper`); fresh run; user's agreement |
| Hip roll | 100 / 2 / 0.25 | 100 / 3 / 0.261 | DEV-J (user plant rule) | as above |
| Hip yaw | 100 / 2 / 0.25 | 100 / 3 / 0.261 | DEV-J (user plant rule) | as above |
| Knee | 120 / 4 / 0.25 | 150 / 5 / 0.261 | DEV-J (user plant rule) | as above |
| Ankle pitch | 40 / 2 / 0.25 | 40 / 3 / 0.258 | DEV-J (user plant rule) | as above |
| Ankle roll | 20 / 1 / 0.25 | 30 / 2 / **0.485** | DEV-J (user plant rule; scale 1.9x paper) | as above |
| Waist yaw | 100 / 3 / 0.2 | 40 / 8 / 0.261 | DEV-J (user plant rule) | as above |
| Shoulder pitch | 30 / 1 / 0.2 | 30 / 1 / 0.258 | DEV-J (user plant rule) | as above |
| Shoulder roll | 30 / 1 / 0.2 | 20 / 1 / 0.258 | DEV-J (user plant rule) | as above |
| Shoulder yaw | 30 / 1 / 0.2 | 20 / 1 / **0.485** | DEV-J (user plant rule; 2.4x) | as above |
| Elbow | 30 / 1 / 0.2 | 50 / 1 / **0.485** | DEV-J (user plant rule; 2.4x) | as above |
| `--action_scale_paper` option | waist 0.2 (Table 14) | sets waist yaw to 0.25 (only arms get 0.2) `x2.py:197-198`; not used by r7 | DEV-P (bug in the opt-in path) | put waist_yaw in the 0.2 group before anyone uses the flag |
| Default joint pose | paper silent | plant defaults: hip pitch -0.312, knee 0.669, ankle pitch -0.363, shoulder pitch 0.2, roll +-0.2, elbow -0.6 (yaml) | DEV-J | |
| Effort limits | tau^max "maximum allowable" (Table 7), values silent | plant effort (hip/knee/waist 120, ankle pitch 36, ankle roll 24, ...) as actuator force range `x2.py:138-139` | DEV-J | |
| Joint velocity limits | theta_dot^max, values silent | URDF limits `config/joint_velocity_limits.json`, `x2.py:188-189` | DEV-J | |
| Armature | silent | plant armature `x2.py:132` | DEV-J | |
| Robot | AgiBot X2, 1.31 m, 39 kg (App. D) | X2 Ultra MJCF `x2.py:19` | MATCH | |

## 4. Rewards, term by term (Table 6, Table 7)

Sigmas `env.py:31`: 0.25, 0.25, 0.01, 0.01, 0.03, 0.0625 = Table 7. h_tar 0.68, t_air 0.4, d_y_min 0.22, eps 0.15 `env.py:32` = Table 7.
N = 21 (`N21f`). Group weights w_l 1.0, w_f 0.25, w_s 0.2 `train.py:26`.

| Term (weight) | Paper expression | Code `env.py` line | Status | To follow |
|---|---|---|---|---|
| Linear velocity tracking (1.0) | exp(-norm(v_xy - v_cmd)^2 / sigma_v), sigma_v 0.25 | l.973; **sigma_v ramped 0.25 -> 0.1** from it 600 over 300 (`--sig_v_to 0.1 --sig_v_from 600 --sig_v_ramp 300`, `train.py:252-253`); live value 0.1 | DEV-P | drop `--sig_v_to` (sigma_v 0.25) |
| Linear velocity, backward motion | not in paper | x `--back_scale 0.5` when v_x < -0.05 l.1001-1002 | DEV-J (user rule 2026-11-12, no backward travel) | |
| Angular velocity tracking (0.8) | exp(-(w_z - w_cmd)^2 / 0.25) | l.974 (rel_kernel off) | MATCH | |
| Orientation (0.5) | exp(-norm(g_xy)^2 / 0.01) | l.975 squared norm | MATCH (prior audit wrong) | |
| Angular velocity xy (0.25) | exp(-norm(w_xy)^2 / 0.25) | l.976 | MATCH | |
| Base height (0.4) | exp(-(h_b - 0.68)^2 / 0.01), h rel. ground | l.977, ground = raster height under the pelvis | MATCH | |
| Action rate (-0.12) | (1/N) norm(a_t - a_{t-1})^2 | l.978 | MATCH | |
| Smoothness (-0.06) | (1/N) norm(a_t - 2a_{t-1} + a_{t-2})^2 | l.979 | MATCH | |
| DoF velocity (-0.96) | (1/N) norm(qd / qd_max)^2 | l.980 | MATCH | |
| DoF torque (-0.6) | (1/N) norm(tau / tau_max)^2 | l.981 | MATCH | |
| DoF deviation (-1.8) | (1/N) norm1(theta - theta_def) | l.982 (boot 0, dof_dev_ref off) | MATCH | |
| DoF position limits (-1.5) | (1/N) count outside [min, max] | l.983, hard limits | MATCH | |
| DoF velocity limits (-6.0) | (1/N) count abs(qd) > qd_max | l.984 | MATCH | |
| DoF torque limits (-6.0) | (1/N) count abs(tau) > tau_max | l.985 (>=; actuator force is clipped at the limit, so this fires at saturation) | MATCH | |
| Stand still (-0.12) | (1/N) norm1(theta - def) x 1(norm(u) <= 0.15) | l.986 (norm over vx, vy, wz) | MATCH | |
| Single support (0.2) | 1[norm(u) <= eps or one foot alone in [t-0.2 s, t]] | l.966, 987 | MATCH | |
| Impact velocity (-1.3) | sum v_z^2 c | l.988 (foot body origin, finite difference) | MATCH | |
| Contact slippage (-0.2) | sum norm(v_xy)^2 c | l.989 | MATCH | |
| Feet air time (-2.0) | sum max(t_air - 0.4, 0) c_first | l.990 | MATCH | |
| Feet stumble (-2.0) | sum 1(norm(F_xy) > 3 abs(F_z)) | l.991 | MATCH | |
| Feet lateral distance (0.08) | exp(min(d_y - 0.22, 0) / 0.03) | l.992, d_y in base frame from ankle origins | MATCH | |
| Swing height (0.5) | not in paper | `--swing_h_w 0.5 --swing_h_target 0.10` l.998, 949-955 | ADDED | |
| Creep / slow-command progress (0.5) | not in paper | `--creep_w 0.5` l.1003-1005 | ADDED | |
| Drift under stand / turn (-0.5) | not in paper | `--drift_w 0.5` l.1006-1007 | ADDED | |
| Side-obstacle graze (-2.0 per step) | not in paper | `--side_soft 2.0` l.997 | ADDED | |
| Wall body / arm contact | not in paper | l.995-996 (route robots with `--ext_phys 24`) | ADDED (user rule 2026-11-12) | |
| air_bonus, imitate, feet_air | not in paper | weights 0 in r7 (boot_iters 0, imit_w 0, feet_air_w 0) l.993-999 | ADDED (off) | |
| Progress gate, rel. kernels | not in paper | off in r7 (prog_gate 0, rel_kernel off) | ADDED (off) | |
| Staged schedule of terms | single stage, all terms from the start (Sec. 3.1, 4.1) | `--stage_a 150 --stage_b 400`: only CORE_TERMS before 150, rest ramped to 400 l.1018-1019, `train.py:247-248`; finished at it 400 | DEV-P (inactive now) | omit on the next fresh run |
| Reward scale | paper silent | all rewards x 0.02 `train.py:299` (advantages are normalised, so the policy gradient is unchanged; value targets smaller) | ADDED | |

## 5. Reward grouping and advantage weighting

| Item | Paper (ref) | Code | Status | To follow |
|---|---|---|---|---|
| Three groups r^l, r^f, r^s, three critics | Sec. 3.1, Table 6 | `train.py:295`, Critics(3) `networks.py:207-215` | MATCH | |
| Group advantage weights | w_l 1.0, w_f 0.25, w_s 0.2 (Table 6) | `train.py:26,219-220,256-258` | MATCH (w_s scheduled, see 15) | |
| GAE per critic | implied (Sec. 3.1) | `train.py:317-325` | MATCH | |
| How weighted advantages combine | multi-critic of [12] HoST and [46] RobotKeyframing (Sec. 3.1): each group's advantage normalised separately, then weighted sum | `--adv_norm joint` (default): A = sum(w_i A_i), then one normalisation `train.py:329-331` | DEV-P (by reference; the paper text does not spell it out) | `--adv_norm separate` (already implemented, `train.py:327-328`); judge on a fresh run or a controlled resume |
| Time-limit bootstrap | paper silent | r += gamma V on timeouts `train.py:300-301` | ADDED | |
| Episode return logging | n/a | `train.py:306-313` | ADDED | |

## 6. Commands and curriculum per terrain (A.2, Tables 4-5)

| Item | Paper (ref) | Code | Status | To follow |
|---|---|---|---|---|
| Command form | u = [vx, vy, wz]; walk along world x, heading target; wz from heading error; vx, vy = world forward speed projected into base frame (A.2) | `env.py:610-619` | MATCH | |
| Yaw from heading error | gain silent | wz = clip(1.0 x err, +-1.2) `env.py:615` | DEV-J | |
| Projected vy clamp | not in paper (projection only) | vy clamped +-0.4 and vx to [-1, 1.2] `env.py:618`: on a slope at 80 deg heading and 1 m/s the projected vy (0.98) is cut to 0.4 | DEV-P | remove the clamp for tile robots (keep for omni) |
| Slope speed | U(-1, 1) (Table 4) | U(0, 1): back_scale < 1 removes backward commands on terrain `env.py:379-380` | DEV-J (user rule 2026-11-12: no backward travel) | |
| Slope heading | U(-80, 80) deg | `env.py:381` 80 | MATCH | |
| Stairs speed / heading | U(0, 1.2), +-30 deg | `env.py:380-381` | MATCH | |
| Discrete speed / heading | U(0, 1.2), +-30 deg | no discrete type; beam gets U(0, 1.2), +-30 | DEV-P (see 7) | |
| Gap speed / heading | U(0, 1.2), +-15 deg | no gap type | DEV-P (see 7) | |
| Platform speed / heading | U(0, 1.2), +-15 deg | `env.py:381` 15 | MATCH | |
| Flat tile robots | no flat terrain type in the paper | flat type: U(0, 1.2), +-30 deg | ADDED | |
| Omnidirectional stage: when | only after the robot has cleared the highest difficulty and exited the terrain region (A.2) | 30 % of all robots on the open flat tile from the start, every episode (`--omni_frac 0.3`) `env.py:362` | DEV-P | omni commands only for robots that finish the top level (they leave the tile and keep going on a free area); needs a free area after each tile row |
| Omnidirectional ranges | vx U(-1, 1), vy U(-0.4, 0.4), wz U(-1.2, 1.2) (Table 5) | same base draw `env.py:392`, then 50 % turn-in-place (wz 0.3-1.2, `--omni_turn 0.5`) and 30 % slow forward 0.1-0.4 m/s (`--omni_slow 0.3`) `env.py:393-400`: only 20 % get the Table 5 draw | DEV-P | `--omni_turn 0 --omni_slow 0` |
| Command resampling | one command per episode (A.2) | new speed every 8 s (`--resample_s 8`) `env.py:674-682` | ADDED | |
| Edge stop | not in paper | 30 % of tile episodes stop 2 s at the first down edge (`--edge_stop 0.3`) `env.py:916-941` | ADDED (ACP's fix) | |
| Route robots (extended map) | not in paper | 40 % of tile robots (`--ext_frac 0.4`) on extended-map routes with a waypoint command law, vmax 0.4-1.0, sidestep in tight spots, hold at the end `env.py:425-437,859-903` | ADDED (user decision 2026-11-12) | |
| Zero-command threshold | eps 0.15 on norm(u) (Table 7) | EPS_CMD 0.15 `env.py:32` | MATCH | |
| Curriculum method | as in [30] (Rudin et al.): up when distance > half the terrain length, down when distance < half the commanded distance, over the top -> random level (A.1) | up only on reaching the tile end (x > 8 m from a 0.35-0.9 m start) `env.py:1074-1075,1123`; down if travel < max(0.5 v t, 0.5 m) after 3 s `env.py:1122-1124`; random level over the top `env.py:1126-1127` | DEV-P | up when travel > 4 m (TILE_X / 2) at episode end; commands under about 0.4 m/s can never reach 8 m in 20 s |
| Starting levels | [30]: random up to a max initial level | randint(0, 4) at the first reset `env.py:410`; flat phase then all reset to level 0 `train.py:235` | DEV-J | |
| Number of levels | 20 per type (A.1) | NLEVEL 20 `terrain.py:15` | MATCH | |
| Terrain mix | 5 terrain types, all robots on terrain (A.1) | flat phase until it 550, then obstacle share ramped 0.25 -> 0.8 over 500 it; at least 20 % on the flat type for good (`--flat_until 550 --obst0 0.25 --obst_ramp 500`) `train.py:231-244`, `env.py:411-417` | DEV-P | no flat phase; uniform over the terrain types |
| Walking starts (RSI) | not in paper | 50 % of tile episodes start from flat prior frames with v >= 0.15 m/s; commanded speed set to the frame's `env.py:319-356`, `--rsi_min_speed 0.15` | ADDED | |

## 7. Terrain generation and levels (A.1, Table 3)

| Item | Paper (ref) | Code | Status | To follow |
|---|---|---|---|---|
| Terrain types | slope, stairs, discrete, gap, platform (A.1, Fig. 10) | ramp, stairs, beam, platform, flat `terrain.py:13-14,73` | DEV-P (gap and discrete missing; beam and flat added; chosen for the apparatus) | add `_gap` (width 0.1-0.8 m, depth 0.3-0.6 m) and `_discrete` (random blocks 0.05-0.2 m) builders; a 6- or 7-type layout needs more experts / discriminators or a mapping decision |
| Tile size | 8 x 4 m (A.1) | TILE_X 8, TILE_Y 4 `terrain.py:15` | MATCH | |
| Slope incline | [0, 20] deg | 20 x s deg, up-then-down ramp with a 1.25 m flat start `terrain.py:23-28` | MATCH (shape silent) | |
| Stairs step length | [0.25, 0.4] m | 0.40 - 0.15 s `terrain.py:32` | MATCH | |
| Stairs step height | [0.05, 0.2] m | 0.05 + 0.15 s, up then down or down then up, at most 6 steps `terrain.py:31-46` | MATCH | |
| Discrete obstacle height | [0.05, 0.2] m | not built; "beam" = bars across the lane 0.03-0.12 m high, 0.05-0.14 m wide, yaw +-25 deg `terrain.py:49-59` | DEV-P | see types row |
| Gap width / depth | [0.1, 0.8] m / [0.3, 0.6] m | not built | DEV-P | see types row |
| Platform height | [0.05, 0.4] m | 0.05 + 0.35 s, 1.3 m long, 1 m apart `terrain.py:66-70` | MATCH (layout silent) | |
| Side obstacles (walls, rails, posts, door frames) | not in paper | `terrain.py:80-107`, seen in depth and the critic map, touching ends the episode / costs | ADDED | |
| Extended terrain map | not in paper | `terrain.py:245-...` types tight_spots, log_corridors, homes, street_front, stairs, wedges, single_beams, beam_fields, blocks; 24 physical wall boxes per world | ADDED (user decision 2026-11-12) | |
| Height raster resolution | paper silent | 2.5 cm `terrain.py:15` | DEV-J | |
| Height field split | n/a | X2_HF_CHUNK=400 (fields of at most 400 x 400) `x2.py:147-169` | DEV-J (MuJoCo Warp large-hfield bug) | |

## 8. Terminations (A.7)

| Item | Paper (ref) | Code | Status | To follow |
|---|---|---|---|---|
| Max duration | yes (A.7); length not stated (eval uses 20 s, Sec. 4.1) | 20 s `env.py:27,1086` (timeout, bootstrapped) | MATCH | |
| Outside the terrain boundary | terminate (A.7) | sideways out: terminate `env.py:1072-1073`; backward out: terminate `env.py:1087`; forward out at the tile end: success, treated as timeout `env.py:1085,1100` | DEV-J (end-of-tile as success is legged_gym practice) | |
| Contact force on termination links | > threshold (A.7) | elbows only, > 80 N `x2.py:58`, `env.py:1048`; trunk, thighs, head end by clearance instead | DEV-J (MuJoCo Warp collision buffer: only spheres touch the heightfield) | |
| Severe orientation instability | yes (A.7), threshold silent | g_z > -0.5 (tilt > 60 deg) `env.py:1046` | MATCH | |
| Gap: foot below the ground plane | yes (A.7) | no gap terrain, no such check | DEV-P (follows from 7) | add with the gap terrain |
| Base too low | not in paper | base < 0.3 m above ground `env.py:1047` | ADDED | |
| Clearance scrape (pelvis, torso, head, hip yaw spheres) | not in paper | `x2.py:59`, `env.py:1049-1050` | ADDED (stands in for link contact) | |
| Side obstacle / wall contact | not in paper | `env.py:1059-1068` | ADDED | |
| Diverged physics | not in paper | `env.py:663-668` | ADDED | |
| No-progress termination | not in paper | `--stuck_term` off in r7 | ADDED (off) | |

## 9. Domain randomisation (A.6, Table 10)

| Item | Paper | Code `env.py` | Status | To follow |
|---|---|---|---|---|
| Base added mass | U(-2.0, 12.5) kg | pelvis l.473 (boot 0) | MATCH | |
| Base CoM offset | U(-0.1, 0.1) m | pelvis ipos l.477 | MATCH | |
| Link mass | U(0.9, 1.1) x | l.472 (inertia scaled with mass, added) | MATCH | |
| Link CoM offset | U(-0.01, 0.01) m | l.476 | MATCH | |
| Motor strength | U(0.8, 1.2) x | l.479-482, applied as a factor on Kp and Kd (not on the torque limit) | MATCH (implementation choice) | |
| PD gains | U(0.9, 1.1) x | Kp and Kd drawn separately l.479-484 | MATCH | |
| Ground friction | U(0.2, 1.25) | robot geoms U(0.2, 1.25), terrain 0.2, max combination l.137, 471 | MATCH | |
| Action delay | U(0, 2) steps | l.457 | MATCH | |
| Push interval / velocity | 12 s, U(-0.4, 0.4) m/s | l.33, 655-661 (xy) | MATCH | |
| Camera position offset | U(-0.01, 0.01) m | chest_rs: +-2 cm forward, +-1 cm lateral, +-5 cm height `x2.py:47`, l.485 | DEV-J (user, 2026-11-11: bracket not final) | |
| Camera orientation offset | U(-1, 1) deg | roll +-2, pitch +-4, yaw +-2 deg `x2.py:47`, l.486 | DEV-J (user, 2026-11-11) | |
| Horizontal / vertical FoV offset | U(-1, 1) deg each | l.487, 505 | MATCH | |
| Self-occluded pixel noise | N(0, 0.1^2) m | l.515-516 | MATCH | |
| Initial joint noise | not in paper | U(-0.05, 0.05) rad l.448 | ADDED | |
| Observation noise | not in paper | see section 1 | ADDED | |

## 10. Networks (A.4, Table 8; App. B)

Table 8 sizes are read as total channels; each paired layer has half per side (stated in `networks.py:101-102`).

| Item | Paper | Code `networks.py` | Status | To follow |
|---|---|---|---|---|
| MoE actor: 5 experts [1024, 512, 128], gate MLP 128, ELU | Table 8 | l.190-201, softmax gate | MATCH | |
| Gate supervision | none in paper | cross-entropy to terrain label, `--gate_prior 0` (off) `train.py:359-363` | ADDED (off) | |
| Policy std | state-independent, min 0.2 (Table 9) | log_std, clamp min 0.2 l.196, 204; init 1.0 `train.py:193` | MATCH (init silent) | |
| Bounded mean | not in paper | 8 tanh(x / 8) l.201 | ADDED | |
| Multi-critic 3 x [512, 256, 128], ELU | Table 8 | l.207-215 | MATCH | |
| Proprio encoder MLP [512, 256, 128], emb 128 | Table 8 | EqMLP(42H, [256, 128, 64], 64) per side = [512, 256, 128] -> 128 l.110 | MATCH | |
| Depth CNN channels [32, 64, 128], kernels [8, 4, 3], strides [4, 2, 2], ELU, emb 128 | Table 8 | LiftConv(1, 16, 8, 4), EqConv(16, 32, 4, 2), EqConv(32, 64, 3, 2) -> 128 l.107, 137-140 (no ELU after the last conv) | MATCH | |
| GRU hidden 256, 1 layer | Table 8 | EqGRUCell(128, 128) per side = 256 l.111 | MATCH | |
| Fusion MLP [512, 256, 128], emb 48 | Table 8 | EqMLP(128, [256, 128, 64], 24) per side l.112 | MATCH | |
| Latent heads z_f 16, z_b 16, z_p 16 | Table 8 | 8 + 8 each l.113-114 | MATCH | |
| z_p as VAE (mu, logvar, sample) | Sec. 3.1 | l.114, 154-155; sampled also in rollouts (actor sees a sampled z_p) | MATCH (sampling at rollout silent) | |
| Decoders for H^f, H^b, o^p_{t+1} | Sec. 3.1 (sizes silent) | MLP [256, 128] each l.223-225 | DEV-J | |
| Estimator MLP [512, 256, 128] -> v_hat 3, from temporal proprio | Table 8, Sec. 3.1 | l.226, input flattened 5 x 72 history | MATCH | |
| Foothold imagination MLP [256, 128], ELU, (s, a) -> (mu 2, sigma) per foot | Table 8, Sec. 3.2 | l.229-237; sigma = exp(clamp(., -5, 0)) | MATCH | |
| 5 discriminators [512, 256, 128], ReLU | Table 8 | l.240-250 | MATCH | |
| Equivariant linear / MLP / GRU / lift conv / conv | App. B.2 (Eq. 5, Cesa et al.) | l.25-97 (W = [[A, B], [B, A]], shared bias; lift [k, flip k]); `tests/test_equivariance.py` | MATCH (proofs not re-derived; tests not re-run here) | |
| Operator T (Table 13) | left: omega, g, u, [leg_L, arm_L, waist]; right: mirrored right slices | `Encoder.structure` l.116-131 | MATCH | |
| Encoder equivariance (Eq. 4) | required | by construction | MATCH (UNVERIFIED numerically in this audit) | |

## 11. Discriminators and AMP (Sec. 3.4, Tables 6, 8, 9)

| Item | Paper | Code | Status | To follow |
|---|---|---|---|---|
| One discriminator per terrain type | 5: slope, stairs, discrete, gap, platform (Sec. 3.4) | 5: ramp, stairs, beam, platform, flat `terrain.py:13`; omni robots use the flat one `train.py:95-96` | DEV-P (follows the terrain set) | one per paper terrain; flat walking inside each set, no flat discriminator (Table 11) |
| Frame content, 63-D | g, v, omega, theta, theta_dot, limb positions (Sec. 3.4) | `env.py:603`, `build_amp.py:68-75`: g 3, v_b 3, w_b 3, q 21, qd 21, hands 6, feet 6 | MATCH | |
| What the discriminator sees | all 63 dims | arm q, arm qd and wrist positions masked to 0 (`--disc_mask_arms`) `train.py:188-191` | DEV-P | drop `--disc_mask_arms` |
| Window | five frames (Sec. 3.4) | AMP_FRAMES 5 `env.py:24`, windows may not cross clip starts `train.py:57-59` | MATCH | |
| Loss | as AMP [27]: least squares, targets +1 / -1 | `train.py:435` | MATCH | |
| Gradient penalty | w_gp 20 (Table 9) | 20 x E norm(grad D_real)^2 `train.py:434-435` (AMP writes w_gp / 2; which one SSR uses is not stated) | MATCH (factor 1/2 ambiguous) | |
| Weight decay | 1e-4 (Table 9) | Adam weight_decay 1e-4 `train.py:197` | MATCH | |
| Optimiser | own Adam, lr 5e-4 (A.5) | `train.py:197` | MATCH | |
| Style reward | max[0, 1 - 0.25 (D - 1)^2] (Table 6) | `train.py:99-106` | MATCH | |
| Style weight | w_s 0.2 (Table 6) | 0.2, scheduled (section 15) | MATCH now | |
| Updates per iteration | silent | one step per discriminator per iteration, at most 4096 policy windows, grad clip 10 `train.py:425-436` | DEV-J | |
| Input normalisation | silent | running norm `networks.py:244,250` | ADDED | |
| Mirrored reference data | silent (symmetry in Sec. 3.3 is for PPO) | reference frames doubled with mirror copies `train.py:53` | ADDED | |
| Reference window filter | silent | windows with mean planar speed < 0.05 m/s dropped unless mean abs(yaw rate) >= 0.3 (`--prior_min_speed 0.05 --prior_turn_min 0.3`) `train.py:60-69` | ADDED | |
| Reference sampling weights | uniform implied | equal mass per 0.2 m/s speed bin, cap 5x (`--prior_balance`) `train.py:73-77` | ADDED | |

## 12. Motion prior data (A.8, Table 11)

Paper: 71 unique clips, 521 s, own mocap + an AMASS subset, retargeted with Xie et al. [43], reused across terrains. Per set:
slope 261.3 s = flat forward + backward + turn + slope traversal; stairs 295.8 s = forward + turn + stairs; discrete 136.0 s =
forward + turn + step; gap 174.5 s = forward + turn + gap; platform 181.6 s = forward + turn + step + platform. No flat set.
Code `data/amp_v4.npz` (built by `scripts/build_amp.py` then `build_amp_v4.py`), measured on the node 2026-11-13 (seconds before
the mirror doubling and the train-time filter):

| Set | Clips | Seconds | Forward vx > 0.2 | Backward vx < -0.2 | Turn abs(wz) > 0.5 | Lateral abs(vy) > 0.2 | Standing speed < 0.1 |
|---|---|---|---|---|---|---|---|
| ramp | 75 | 703.4 | 328.4 | 25.9 | 93.1 | 146.2 | 160.6 |
| stairs | 297 | 1833.6 | 1010.0 | 11.0 | 241.7 | 154.2 | 499.9 |
| beam | 169 | 858.0 | 493.4 | 28.5 | 252.9 | 143.6 | 159.9 |
| platform | 117 | 922.3 | 443.3 | 15.4 | 143.6 | 78.5 | 284.6 |
| flat | 107 | 999.0 | 486.6 | 96.3 | 290.6 | 179.9 | 222.2 |

| Item | Paper | Code | Status | To follow |
|---|---|---|---|---|
| Source data | own mocap + AMASS subset, Xie et al. retargeting | BONES-SEED X2 retargets + the user's Pico captures (+ mirrors) `MOTIONS.md`, `build_amp_v4.py:13-24` | DEV-J (paper data not released; the user's captures are the right substitute) | |
| Set sizes | 261.3 / 295.8 / 136.0 / 174.5 / 181.6 s, total 1049 s | 703 / 1834 / 858 / 922 / 999 s, total 5316 s (x2 with mirrors) | DEV-P | rebuild with per-set budgets equal to Table 11 (flat part + traversal part) from the user's captures; plan acknowledged by the user 2026-11-13 (USER_RULES l.197) |
| Composition: flat forward + turn in every set | Table 11 | flat share is a 150 s BONES-SEED draw filtered by clip name (walk, turn, jog, run, stride, back), not a forward + turn selection `build_amp_v4.py:15` | DEV-P | select flat forward and turn segments explicitly (turn in place and walking turns), same seconds split per set |
| Backward only in the slope set | Table 11 | backward frames in every set (11-96 s), also lateral 79-180 s | DEV-P | drop backward and lateral windows from all but slope (and from slope too under the no-backward rule, a user call) |
| Traversal types | slope, stairs, step, gap, platform | ramp, stairs, beam (4x4 / 2x2 logs + blocks), platform (stair takes + box chains) | DEV-P (follows the terrain set) | step and gap takes are needed for discrete and gap sets; the platform set should be step + platform takes |
| Separate flat set / discriminator | none | flat set 999 s, used for omni and flat-tile robots | DEV-P | fold flat walking into each set |
| Standing frames | not listed (walking motions) | 160-500 s per set; dropped at train time only if the window mean speed < 0.05 m/s and not turning | DEV-J | |
| Clip reuse across sets | yes (71 unique clips) | the same 150 s flat draw is reused in ramp, stairs, beam; platform uses amp_v2's set | MATCH (in spirit) | |
| Frame rate | silent (50 Hz policy) | resampled to 50 Hz `build_amp.py:24,40-51` | DEV-J | |

## 13. PPO and optimiser hyperparameters (A.5, Table 9)

| Item | Paper | Code | Status | To follow |
|---|---|---|---|---|
| Number of environments | 4096 (Table 9, Sec. 4.1) | 8192 (`--num_envs 8192`, queue31) | DEV-P | 4096 on a fresh run (each mini-batch is then half the size; with mirroring, 2 x 24 x 1024 samples) |
| Steps per iteration | 24 | STEPS 24 `train.py:41` | MATCH | |
| Learning epochs | 5 | EPOCHS 5 | MATCH | |
| Mini-batches | 4 | MINI 4 | MATCH | |
| gamma / lambda | 0.99 / 0.95 | `train.py:40` | MATCH | |
| Value loss coef | 1.0 | VF 1.0 | MATCH | |
| Entropy coef | 0.005 | 0.005 | MATCH | |
| Desired KL | 0.01, adaptive lr | KL_TARGET 0.01, rsl_rl schedule per mini-batch (x1.5 / /1.5, bounds 1e-5 to 1e-3), KL on the original half only `train.py:365-373` | MATCH | |
| Clip ratio | silent | 0.2, value clipping 0.2 `train.py:40,357` | DEV-J | |
| Gradient clipping | silent | 1.0 for PPO, encoder, foothold; 10 for discriminators | DEV-J | |
| Four Adam optimisers, lr 5e-4 | PPO, encoder + estimator, foothold, discriminators (A.5) | `train.py:192-197` | MATCH | |
| Encoder losses | H^b MSE 2.0, H^f MSE 1.0, next o^p MSE 5.0, KL 1.0, velocity MSE 2.0 (Table 9) | `train.py:395` | MATCH | |
| Encoder training | silent | GRU rolled over the stored 24 steps from the stored h0, 4 chunks, one step each `train.py:377-398` | DEV-J | |
| Encoder gradient from PPO | silent (hybrid loss only) | none (rollout in no_grad, stored obs) | DEV-J | |
| Foothold NLL (Eq. 1) | Gaussian NLL, first valid future contact, base frame | `train.py:400-419`; target in the yaw (heading) frame, not the full base frame; only contacts inside the same 24-step rollout count | DEV-P (frame) / DEV-J (window) | rotate targets with the full base rotation (R^T) and use the base-frame xy; minor |
| Foothold updates | silent | one full-batch step per iteration | DEV-J | |
| Non-finite guard | n/a | skipped steps, no save on NaN `train.py:109-125,470-471` | ADDED | |

## 14. Symmetry (Sec. 3.3, App. B.1, Tables 12-13)

| Item | Paper | Code | Status | To follow |
|---|---|---|---|---|
| s_leg, s_arm | [1, -1, -1, 1, 1, -1], [1, -1, -1, 1] (Eq. 6) | `x2.py:205-208` | MATCH | |
| s_y, s_omega, s_u | [1, -1, 1], [-1, 1, -1], [1, -1, -1] | `mirror.py:11-13` | MATCH | |
| M_p on o^p | Table 12 | `mirror.prop` l.22-25 | MATCH | |
| M_o: v_hat x s_y, latent half swap per head | Table 12 | `mirror.actor_obs`, `latent` l.28-38 | MATCH | |
| M_s: v, foot vel, contacts, hands, feet swapped; H^b flipped; H^f flipped and swapped | Table 12 | `mirror.priv` l.45-59 | MATCH | |
| M_a = F(a) | Table 12 | `mirror.joints` | MATCH | |
| Mirrored samples appended for PPO, encoder run once | Sec. 3.3, [24] | per mini-batch, same old log-prob, returns, advantages `train.py:344-347` | MATCH | |
| Mirror correctness vs kinematics | implied | `tests/test_mirror.py` (not re-run in this audit) | UNVERIFIED | |
| Asymmetric robot / camera | symmetric assumed | camera centred y = 0 (`x2.py:31,46`) | MATCH | |

## 15. Training stages and schedule

| Item | Paper | Code | Status | To follow |
|---|---|---|---|---|
| Single-stage training | single stage, about 20k iterations (Sec. 3.1, 4.1) | `--iters 20000`, but: flat only until it 550; obstacle share ramp 550-1050 to 0.8; core terms only before 150, ramp to 400; style 0 until 1000, ramp to full at 1500 (`--style_from 1000 --gate_ramp 500`) `train.py:231-260` | DEV-P (all phases over by it 1500; run now at 6310) | plain single stage on a fresh run |
| Foothold guidance activation | permanently on once the global mean terrain level exceeds 2 (Sec. 3.2) | latched when mean level of non-omni robots > 2 `env.py:1129-1131`; mean includes the flat-type robots at level 0 and route robots; carried in checkpoints | MATCH (definition of "global mean" close) | |
| Before activation | swing feet: paper implies only contact-time support | swing rho = 0 until on; stance rho always `env.py:1032-1037` | MATCH | |
| Foothold on slopes | rho = 0 on slopes (Table 6 note 1) | `env.py:1038` | MATCH | |
| Foothold on omni / flat robots | no such robots in the paper | rho = 0 for omni robots; flat-type tile robots get it | ADDED | |
| Iterations | about 20k | 20000 | MATCH | |
| Resumed runs with flag changes | single run | r7 resumed many times with new flags (v13, v14, HF split) `queue31.sh` | DEV-J (operational) | |

## 16. Sim settings

| Item | Paper | Code | Status | To follow |
|---|---|---|---|---|
| Simulator | Isaac Gym (Sec. 4.1) | MuJoCo Warp, implicitfast `env.py:1`, `x2.py:118` | DEV-J (track choice) | |
| Physics dt / decimation | silent | 5 ms x 4 `x2.py:20` | DEV-J | |
| Contact model | silent | only spheres touch the heightfield (12 sole spheres r 2 cm, toe / heel / side, shin, elbow, hand guards); meshes collide with each other and with wall boxes `x2.py:78-110` | DEV-J (MuJoCo Warp limits) | |
| Self-collision exclusions | silent | pelvis vs hip-roll excluded (`--excl_pelvis_hiproll`) `x2.py:181-183` | DEV-J | |
| GPU | one RTX 4090 (Sec. 4.1) | node GPU (not checked) | UNVERIFIED (irrelevant) | |

## 17. Episode length

| Item | Paper | Code | Status | To follow |
|---|---|---|---|---|
| Training episode | silent (eval trials are 20 s, Sec. 4.1) | 20 s `env.py:27` | DEV-J | |

## 18. Depth and perception (Sec. 4.1, App. C, App. D)

| Item | Paper | Code | Status | To follow |
|---|---|---|---|---|
| Camera mount | waist-mounted D435i, pitched 50 deg down (App. D) | chest_rs: torso_link (0.126, 0, 0.16), 59.5 deg down `x2.py:46` | DEV-J (user's real chest mount, USER_RULES l.88-90) | |
| FoV | 87 x 58 deg (App. D) | fov_h 87, fov_v 58 rendered to 36 x 36 `x2.py:46`, `env.py:165` | MATCH (paper crops a 640 x 360 image to 36 x 36, so its final FoV is likely narrower; the crop is not given) | UNVERIFIED (crop window not stated) |
| Renderer | Warp ray cast on terrain mesh + per-link meshes, rays moved into mesh frames (App. C, Alg. 1) | ray march on the vision height raster `terrain.py:347`, self-occlusion by 20 limb spheres `x2.py:51-54`, `env.py:520-534` | DEV-J (equivalent purpose; spheres are an approximation of link meshes) | |
| z-depth conversion | c = (1 + tx^2 + ty^2)^-1/2 (Alg. 1) | z = t x ray x-component `env.py:514` | MATCH | |
| Clipping | clamp(d_min, d_max) (Alg. 1) | 0.1 to 3.0 m `env.py:517` | MATCH (values silent) | |
| Depth rate / latency | 60 Hz camera, 50 Hz policy (App. D) | rendered every control step, no latency | DEV-J (paper silent on sim latency) | |
| Real-image preprocessing | RealSense filter, downsample, hole-fill, crop (App. D) | deploy side, not in training code | UNVERIFIED (deploy code not audited) | |
| Side obstacles in depth | n/a | vision raster has walls 1.3 m `terrain.py:76,124` | ADDED | |

## 19. Anything else

| Item | Paper | Code | Status | To follow |
|---|---|---|---|---|
| Evaluation metric | success = 20 s without termination, 1000 trials per terrain at 1.0 m/s (Sec. 4.1) | track eval is the team's apparatus eval (`docs/EVAL.md`), not the paper's protocol | DEV-J (team goal) | |
| SFR / MSR logging | support ratio > 75 %, mean (Sec. 4.2) | logged at touchdown off slopes `env.py:708-712` | MATCH | |
| AP / PFCF (Table 1) | style proxies | power and peak foot force logged in gait stats `env.py:699-700` | MATCH | |
| Cross-platform (App. E) | DR02 | n/a | n/a | |

---

## Counts

MATCH 108, DEV-J 51, DEV-P 26, ADDED 37 (5 of them switched off in r7), UNVERIFIED 3 (rows marked "MATCH (UNVERIFIED ...)"
counted as MATCH; counted by grep on the status column).

## Ranked: deviations most likely to matter for apparatus success and style

1. **Motion prior does not follow Table 11** (sizes 2.7-6x, BONES-SEED-heavy flat share chosen by clip name, backward and lateral
   frames in every set, a separate flat discriminator, standing frames). Style first; it also shapes stepping on stairs and logs.
2. **Advantages combined jointly, not normalised per group** (paper's cited HoST / RobotKeyframing method). Under joint
   normalisation the foothold and style groups pull far less than 0.25 and 0.2 of the policy gradient: weaker foot placement
   and weaker style. One flag (`--adv_norm separate`).
3. **Velocity kernel 0.1 instead of 0.25**: a tracking-dominated objective that competes with style and with careful stepping.
4. **Command distribution**: 30 % omni robots from the start (and 80 % of those turn in place or creep), 8 s resampling, edge
   stops, 40 % route robots, no backward on slopes, vy clamp. Shapes what the policy is good at; mostly user / apparatus driven,
   but the omni stage and the vy clamp are not.
5. **Added reward terms** (swing height 0.5, creep 0.5, drift 0.5, side graze 2.0, back scale): each tilts the gait away from
   the paper's balance; swing height directly fights the style prior.
6. **Arms masked in the discriminators**: arm motion has no style signal (arm swing, elbow posture).
7. **Action scales 1.3-2.4x the paper on ankle roll, shoulders, shoulder yaw, elbow, waist** and different PD gains: jittery
   ankles and arms are likely; parked by the user.
8. **Terrain set without gap and discrete**: the paper's precise-placement terrains are missing; beam and flat stand in.
   Matters less for the apparatus (logs, stairs, platforms) than for fidelity.
9. **Curriculum promotes only at the tile end**: slow commands (< about 0.4 m/s) never promote; levels skew to fast walkers.
10. **Staged schedule** (flat phase, style from 1000): over now, but the first 1500 iterations shaped the gait without style.
11. **8192 envs** (twice the paper): larger batches, different effective PPO step; minor for quality.
12. **Base height map 17 x 10** (paper 18 x 10), **foothold targets in the heading frame**: minor.
