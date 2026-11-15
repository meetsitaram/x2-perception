# simcheck: simulator faults caught by code

User goal (2026-11-12): "simulator faults must be caught by code, not discovered by agents". Two faults punished policies for
hours that day: feet trapped in height-field cells under soft contact (FINDINGS B8) and MuJoCo Warp mis-colliding on one
large height field (B10; 10.8 % of SSR's episode ends were such blow-ups, counted as terminations). Rules:
`agentic/docs/TRACK_RULES.md`, "Simulator checks".

Location: `/mnt/filesystem-g4/tools/simcheck/` (import as `simcheck` with `PYTHONPATH=/mnt/filesystem-g4/tools`); mirror in
the repo at `tools/simcheck/`. No GPU, no training code is touched; patches are files, not applied.

| file | what |
|---|---|
| `core.py` | the three checks: `hfield_size`, `foot_drop`, `rollouts` (limits as constants at the top) |
| `preflight.py` | CLI: build the track's training model through its adapter, run the checks, JSON report, exit 0 pass / 1 fail / 2 error |
| `adapters/acp.py`, `adapters/ssr.py` | the track's own builder, as its training launch calls it (ACP: `load_sets` -> `Terrain` -> `build_x2`; SSR: `generate` -> `add_extended` -> `x2.build`) |
| `adapters/hiking.py` | Isaac Lab on the GPU: preflight only during a training pause (procedure in the module doc) |
| `episode_end.py` | the rule and `FaultDetector` / `truncate_faults` for the training loop |
| `alarm.py` | `sim_fault_share` and `sim_fault_warn` for the heartbeat |
| `patches/{acp,ssr,hiking}.diff` | exact patches for each track, NOT applied |
| `run_preflight.sh`, `queue.sh`, `summarize.py`, `diag_drop_c.py` | running, batch, reading, MuJoCo C vs Warp comparison |
| `reports/` | preflight reports `<track>_<tag>.json` with `.DONE` (exit 0/1) or `.FAILED` (exit 2) markers |

## 1. Preflight (before every launch or restart)

    /mnt/filesystem-g4/tools/simcheck/run_preflight.sh acp s2h_fix1 --hf-fix 1 --cfg cfg/stage2h_sets.json
    /mnt/filesystem-g4/tools/simcheck/run_preflight.sh ssr r7 [--ssr '{"ext_types": "..."}'] [--hf-chunk 400]
    python3 /mnt/filesystem-g4/tools/simcheck/summarize.py /mnt/filesystem-g4/tools/simcheck/reports/acp_s2h_fix1.json

CPU only (GPU hidden), 4 cores (`SIMCHECK_CORES`, default 20-23), nice 15; 20-60 s per run. Pass the same cfg, environment
(`ACP_HF_FIX`, `ACP_HF_CHUNK`, `X2_HF_CHUNK`, ...) and terrain flags as the launch. `--quick` halves the work.

| check | what | fail when |
|---|---|---|
| hfield_size | every height field of the compiled model | any over 400 x 400 vertices |
| foot_drop | the training foot's collision geoms (exact compiled parameters) on a 10 kg free body, with the training model's static terrain geoms; 3 sites each on flat, slope and edge cells (ACP's site classes) x 1, 2, 3, 4.5 m/s down + 1 m/s sideways; 150 N pull 0.1-0.3 s downhill (away from a riser) | deeper than 1 cm for more than 0.07 s in total, moved less than 2 cm along the pull, or NaN |
| rollouts | the whole robot, every tile type (ACP: one clip per data set, its props as mocap boxes, robot in the clip's first frame; SSR: every base type at levels 0/10/19 and one route start per extended tile), 4 worlds each, 2 s, PD targets = start pose + U(-0.5, 0.5) rad every 0.1 s | any contact deeper than 5 cm, any body speed change over 20 m/s between two physics steps, any NaN, diverged share (track's own |qvel| bound) over 1 % |

Reports name the deepest contact's geom pair and time per tile and the foot geom of each failed drop, so a failure points
at its cause. `diag_drop_c.py` repeats failed drops in MuJoCo C to tell a Warp bug from a contact-model trap.

## 2. Episode ends: simulator fault vs policy failure

Rule: a step is a simulator fault when, after the physics, any qpos/qvel is not finite, |qvel| exceeds the track's bound
(ACP 200, SSR 60), any contact is deeper than 5 cm, or any robot body's speed changes by more than 20 m/s between two
control steps. A fault ends the episode as a truncation: value bootstrapped like a time limit, the step's reward 0, no
termination penalty, no curriculum move, never counted as a termination. Everything else stays the policy's.

    from simcheck.episode_end import FaultDetector, contact_depth_per_world, truncate_faults
    det = FaultDetector(N, device, qvel_max=200.0, control_dt=0.02)                       # once
    fault, why = det.update(qpos, qvel, xpos[:, robot_bodies], contact_depth_per_world(dd, N))   # after physics
    term, timeout, r = truncate_faults(term, timeout, r, fault)
    det.reset(reset_ids)                                                                  # after resets

Cost: one scatter over the contact buffer and one difference of body positions per control step, on the GPU.
Patches: `patches/acp.diff` (acp/env.py, acp/train.py), `patches/ssr.diff` (ssr/env.py, ssr/train.py,
scripts/supervise_node.py), `patches/hiking.diff` (InstinctLab x2/staged.py: a `sim_fault` DoneTerm with `time_out=True`,
every failure term masked by it, so `fall_penalty_staged` skips faults; x2_port/supervise_x2.py). Apply with
`cd <track code root> && patch -p1 --dry-run < patches/<track>.diff`, then without `--dry-run`, at a planned restart only.

## 3. Alarm in the heartbeat

`alarm.fields(counts, total)` returns `sim_fault_share`, `sim_fault_n`, `episode_ends_n`, `sim_fault_warn` (a WARN string over
1 %, else null); `alarm.fields_from_shares(shares, sim_keys)` does the same from shares. The one line per heartbeat writer
(all three are in the patches):

- ACP `acp/train.py`, the `heartbeat(...)` call: `**_alarm.fields({"sim_fault": ep["bad"]}, total=ep["done"])`
- SSR `scripts/supervise_node.py`, the training `heartbeat(...)` call: `**alarm.fields_from_shares(rec.get("done_share") or {}, sim_keys=("diverged",))`
- Hiking `x2_port/supervise_x2.py`, `tail_stats`: `out.update(alarm.fields_from_shares({k: v / tot ...}, sim_keys=("sim_fault",)))` from the `Episode_Termination/*` lines

The coordinator's dashboard refresh reads `sim_fault_warn` from `/mnt/filesystem-g4/status/*.json`.
