# Shared lessons: problems each agent hit and how it solved them

Every agent reads this file when it starts or resumes and before a long run, and appends an entry whenever it hits and solves (or
fails to solve) a problem another track could meet: simulator quirks, the X2 model, motion data, throughput, supervisors, the
shared filesystem, eval. One entry per problem: date, who, the symptom, the cause if known, what fixed it, and what is still open.
Short and specific; no node addresses or logins. If an entry concerns you, say so in your next report. The coordinator relays
urgent ones directly.

| Date | From | Problem | Cause | What fixed it / status |
|---|---|---|---|---|
| 2026-11-11 | ACP | MuJoCo Warp printed "nefc overflow" 102 times with 12,000 robots: constraint rows dropped in some worlds | constraint buffer (`njmax`) of 300 too small for the contacts on clip-aligned props | raised `njmax` to 450; none since. Check your own logs for the same message. |
| 2026-11-11 | ACP | MuJoCo Warp prints "height field collision overflow (>= 50)" on a ramp raster | not confirmed; guess: fallen robots lying on the raster | open; coarsening the raster to 4 cm did not remove it |
| 2026-11-11 | ACP | More robots did not turn free GPU memory into speed: 12,000 -> 36,000 gave +20% in the real run, 48,000 only +3% more | the card's compute is saturated by physics, not its memory | use the paper's total (36,000) and stop there |
| 2026-11-11 | Hiking | A robot-count benchmark kept training stopped for 30+ minutes | each Isaac Sim test restarts the simulator (7-10 min) and four sizes were tested in a row | never stop training more than about 10 minutes for a measurement; write "benchmarking" into the heartbeat |
| 2026-11-11 | Hiking | Isaac Lab on the node: one CPU core at 100-120% while GPU utilisation reads low; 4,096 robots give about 15,400 steps/s, 12,288 only +59% | looks CPU-bound on a single core (to be confirmed) | open |
| 2026-11-11 | Hiking | A folder named `instinctlab` in the working directory shadowed the installed package | Python import order | renamed the folder; do not run Python from a folder that contains a package-named folder |
| 2026-11-11 | Hiking | The shared filesystem takes about 17 ms per small file | network filesystem | keep Python environments on the node's local disk; keep code, data, checkpoints and logs on the shared filesystem |
| 2026-11-11 | SSR | From-scratch training settles into standing still within about 150 iterations (3-5 % of commanded speed, both feet down 86-88 % of the time, curriculum at level 0); three runs so far | partly established. (1) The motion prior contained standing: 15-24 % of its frames stand still and the flat set's median forward speed is 0.09 m/s, so the discriminator scores a stand as reference motion, and walking starts drawn from it were mostly slow or standing. (2) The policy brakes out of every walking start and then never steps: a stander gives up about 1.1 of 2.3 reward per step (velocity and yaw tracking), but a fall costs seconds of posture reward, and the exploration noise alone (action std 0.7) costs about 1.0 per step in smoothness and angular-rate terms | open. Did not fix it: rewards x dt, initial std 1.0, per-critic advantage normalisation (motion prior at 14 % of the gradient), dropping still windows from the prior (tested to iteration 300). Check your own prior before training: share of frames under 0.1 m/s and the median speed (one numpy line on the 63-D frames). Watch commanded vs achieved speed and double-support share from iteration 50 on. |
| 2026-11-11 | Eval | The X2 MJCF's 5 mm sole spheres gave tilted contact normals on a flat height field in MuJoCo (CPU): 27 % of standing contacts off vertical, down to |normal z| 0.70, 0.43 at an impact, read as false stubs | sphere much smaller than the 2 cm raster cells | the eval model uses 2 cm sole spheres with the same lowest point (15 % off vertical, never below 0.92), and a stub also needs a real face in the raster. Not checked in MuJoCo Warp or Isaac: tracks that train on height fields with the stock MJCF should look at their own foot-contact normals. ssr-x2 already enlarges them. |
| 2026-11-11 | Eval | The X2's foot (24 cm with the eval's sole spheres, 22.5 cm sole) is longer than our apparatus' stair tread (22 cm; risers 18.6 cm). NIST's treads are 22 to 26 cm, risers 20 to 21 cm | geometry | on ours a foot fully on a tread touches the next riser (a scored side hit if it presses with 20 N or 0.5 N s) or overhangs the heel (an edge landing). Stair treads as short as 22 cm occur on both apparatuses; with a 24 cm foot, heel overhang is the only way to avoid touching the next riser. |
| 2026-11-11 | Eval | NIST wall boxes built from MuJoCo's own box per mesh piece closed the doorway and put the landing deck over the floor; the open door leaf was missing from the physics | a triangle's box in its inertia frame is far larger than the panel; my box collection skipped jointed bodies | boxes fitted to the vertices; the leaf added as one box at its open pose (94.5 deg). Use the files in `data/apparatus/` (they are for the eval and for copying single parts, not whole layouts, into training); do not rebuild them. |
| 2026-11-11 | Coordinator | Node check script failed on Windows with "identity file not accessible" | Python printed CRLF line ends into a shell loop | strip `\r` (`tr -d '\r'`) from anything Python prints for bash on the desktop |
| 2026-11-11 | ACP | Correction to my "height field collision overflow (>= 50)" entry: more likely the foot collision box than fallen robots | the X2 foot box (22 x 13 cm) over 4 cm cells spans about 48 height-field prisms, over 50 once the foot is yawed; over 2 cm cells always | not confirmed by a test. Contacts beyond 50 per geom are dropped. Prefer boxes for steps, coarse rasters only for smooth ground; do not put a large flat collision geom on a fine raster. |
| 2026-11-11 | ACP | Eval's sole-sphere finding, checked against my setup | I train with the stock MJCF (5 mm sole spheres) but stairs, 4x4s and blocks are boxes, not a raster; only the ramp set is on a height field (4 cm cells) | not re-measured on the raster. On boxes the contact read-out is right: 389 N total on a 39 kg robot, 95.4% agreement with reference contact labels. Open: check ramp foot contacts for tilted normals before stage 3 uses rasters. |
| 2026-11-11 | ACP | Stair descents are the weak clip set (finish 0.10 when every other set is 0.67-0.96) | not established; consistent with Eval's note that the X2 foot (24 cm) is longer than the treads (23-29 cm in my descent set): a descending foot must overhang or toe the riser | open; to look at failures after iteration 2,000 |
| 2026-11-11 | ACP | Reading solver contact forces in MuJoCo Warp without touch sensors | no per-body contact force array is filled by default | sum `efc.force` over each contact's `efc_address` rows (pyramidal cone: the sum is the normal force), use `contact.frame[:, 0]` as the normal, mask entries beyond `nacon`, scatter by world and body. Verified against robot weight. |
| 2026-11-11 | ACP | `pkill -f <pattern>` over SSH killed my own remote shell | the pattern was also in the SSH command line | match by PID, or put the pattern in a variable the command line does not contain |
| 2026-11-11 | ACP | A reset curriculum that advances on "episode not terminated" raced ahead of tracking quality (start phase 0.9 -> 0.03 in 1,000 iterations while joint error stayed 0.17 rad) | my early-termination threshold is loose (0.5 m; the paper gives none) | open; watch error terms, not the curriculum, as the progress signal |
| 2026-11-11 | ACP | Per-clip prop boxes: every distinct box size becomes a slot in every simulated world | shared model across worlds | quantise box sizes before use; one clip set with 239 sizes was left out for this reason |
| 2026-11-11 | ACP | A reference set penalised by its own joint-limit term | soft limits at 95% of the joint range are violated by the retargeted clips themselves | soft limits set to the full MJCF range |
| 2026-11-11 | Eval | Our walk raster keeps the feet of diagonal braces and the bulb box as spikes up to 0.5 m in the floor (351 cells), next to the same parts as wall boxes | the raster only removes cells above 0.5 m | left as is (it is what earlier runs were scored on). A policy's height scan or depth sees these spikes near the 4x4s and the ramps: copy such clutter into training terrains if you copy those parts. |
| 2026-11-11 | Eval | Tight places on the eval routes | geometry | ours: a 0.51 m slot between the door wall and the front 4x4 (the robot turns in it), doorway 0.87 m, a 0.37 m gap between the two 4x4s that must NOT be used (both beams have to be stepped over, else the trial fails). NIST: doorway 0.85 m with an 8 cm threshold, V of 4x4s 10 cm tall met at 62 to 90 degrees, a sloping panel 1.32 m above the deck over the pocket behind the upper 4x4 (the X2 stands about 1.3 m), 0.8 m lane partitions. Turning on the spot and walking through 0.85 m openings are needed on both. |
| 2026-11-11 | Eval | What the eval commands | definition | a waypoint (0.20 m arrival disc), then the next; at 11 (NIST) or 5 (ours) stations a 1 s stand. The harness offers the waypoint position and heading, and a velocity command (forward up to the track's limit, yaw up to 1 rad/s, turn on the spot beyond 60 deg, zero while holding). A policy must be able to stop, stand 1 s, turn in place and start again; legs between waypoints are 0.2 to 1.9 m. |
| 2026-11-11 | Eval | Side hits are scored (eval only) | user request | a foot, shin or knee against the side of a 4x4, a wedge face or a stair riser with 20 N or 0.5 N s costs 0.1 to 1.0 and spoils "clean success". Details in `docs/EVAL.md`. |
| 2026-11-11 | SSR | MuJoCo Warp throughput does not grow with robot count: 63.7k, 69.2k, 63.6k env-steps/s at 4,096 / 8,192 / 16,384 robots (X2, 2.5 cm height field, 36 x 36 depth) | the environment step is GPU compute bound; 42 % of it is the depth ray march written in PyTorch (27 ms of 64 ms at 4,096) | only the update phase gains from more robots: full training loop 24.9k / 33.3k / 37.6k samples/s. Chose 8,192. Open: move the ray march into a Warp kernel. Same conclusion as ACP's entry. |
| 2026-11-11 | SSR | `pip install` of torch + mujoco-warp into a venv on the shared filesystem took about 25 minutes, and an SSH command with a client-side timeout was cut off while the remote install carried on | small-file latency of the shared filesystem (Hiking's entry) | let it finish and poll with `pgrep`; it works once installed (imports take about 20 s). A local-disk environment is faster; if you keep it on the shared filesystem, budget the time. |
| 2026-11-11 | SSR | `python3 -m venv` fails on the stock node image | `python3.12-venv` is not installed | `sudo apt-get install -y python3.12-venv` (passwordless sudo works) |
| 2026-11-11 | SSR | Mid-run change of robot count from a checkpoint | per-robot curriculum state (levels, terrain types) is saved with the checkpoint's robot count | on resume, tile the saved levels over the new count; optimiser state and networks do not depend on the count |
| 2026-11-11 | SSR | A checkpoint written every N iterations under two names (`last` except on multiples of 5N) left `last` one save behind the newest file | naming logic | always write `last` (to a temp name, then rename, keeping the previous one as `prev`), and copy to a numbered file on the long interval |

## What the eval needs from each track to score a checkpoint (prepare it now)

Put these in `/mnt/filesystem-g4/tracks/<track>/eval/` (your own tree) and say so in your state file. The eval runs on CPU with
numpy and onnxruntime; it has **no torch**, and it never imports track code.

1. **The policy as a file the eval can run:** a plain MLP as `.npz` (`/mnt/filesystem-g4/eval/code/tools/export_actor.py`, run in
   your environment) or anything else (CNN, GRU, several networks, estimator + actor, planner + tracker) as `.onnx` with fixed
   input sizes, batch 1, the deterministic action (the mean), observation normalisation either inside the graph or exported
   beside it. Recurrent state as explicit inputs and outputs.
2. **An adapter card, `adapter_card.json`,** with:
   - `checkpoint`, `iteration`, `stage`, and the training code commit or snapshot folder;
   - `control_dt` (s) and the physics step and decimation you trained with;
   - `joint_names`: the order your joint observations and actions use (the eval's order is the MJCF's 31 hinges);
     `commanded_joints` if you command fewer, and what the others are held at;
   - `default_q`, `action_scale` (per joint), action clipping, and whether targets are `default + scale * action`;
   - `kp`, `kd`, effort limits and armature per joint as trained (and any actuator delay the policy expects);
   - **every observation term in order**: name, size, frame (body, heading/yaw, world), units, scale, clipping, history
     length and order (oldest or newest first), and what it is at reset;
   - the **command**: what the policy takes (velocity vx, vy, yaw rate with their ranges; or a target direction and
     heading; goal radius), and its forward speed limit;
   - the **terrain sensor**: height scan (grid size, spacing, origin offset, frame, relative to what height, clipping,
     whether walls are in it) or depth camera (body, position, pitch, field of view, resolution, crop, near / far, units,
     normalisation, frame stacking and skip, delay, self-occlusion model, any filtering);
   - anything privileged the actor was given in training that the eval must not supply.
3. **One recorded step for checking:** `obs_check.npz` with, for 50 consecutive steps of one robot in your simulator, the
   simulator state (root pose and velocity, joint positions and velocities), the command, the raw sensor image or scan, the
   full actor input and the action. The eval adapter is accepted only when it reproduces your actor input and action from
   that state (this caught unit and ordering mistakes in earlier sim-to-sim work).

Status of the adapters (eval monitor, 2026-11-11, corrected the same day after the user's note): **ACP is a planner plus a
tracker, and that pair is what walks a route.** The 2 November baseline (13 of 50 on our apparatus) was exactly that: the
stage-2 foot planner chose the next footholds from the height scan every step and the tracker executed them; no reference
clip was involved. My earlier line "stage 1 cannot be scored on a route" was wrong as written: the stage-1 tracker can be
scored as soon as something supplies its contact targets each step. The ACP adapter now takes a pluggable contact source
(`eval/adapters/acp.py`, `ContactSource`): zeros (plumbing only), the earlier foot planner from exported weights (for the
baseline re-run; it was trained on SuperMimic rollouts, so not for this round's policies), the track's own planner as ONNX
when it exists, or an interim non-learned source if the ACP agent writes one to the same interface. The adapter card for
ACP therefore lists two things: the tracker (as above) and the contact source (inputs, outputs per body: target in the
body-yaw frame, valid, time to contact, in contact; decode options). The adapter runs end to end with random tracker
weights and the real baseline planner; nothing is checked against a real tracker checkpoint yet. SSR and Hiking: adapters
are written when the card and the ONNX exist.
