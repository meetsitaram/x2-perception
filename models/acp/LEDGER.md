# ACP run ledger (AgiBot X2, MuJoCo Warp, one GPU)

Every training segment since the track started this round (2026-11-11), oldest first. Times are node clock (UTC), from each
run's `supervisor.log`, `restart.log` and the first / last row of `metrics.jsonl` under `/mnt/filesystem-g4/tracks/acp/runs/`.
Iteration numbers are one global counter: a new run folder continues the count of the checkpoint it starts from.
Tags: [paper] follows the paper (He et al., arXiv 2610.31608), [fix] defect fixed, [user rule], [ours] our addition or choice.
Rows marked CPU are side jobs (planner training on the node's CPU) whose output entered a GPU segment.
Detail: `agentic/state/acp.md` (section numbers in the Notes), `agentic/state/STATUS.md` decisions log, `docs/FINDINGS.md`.

**Lineage of the running tracker:** s1_v1 0-2,250 -> s2_v1 2,250-4,000 -> s1_v2 4,000-5,500 -> s2_v2 5,500-7,000 ->
s2_v3 7,000-10,356 -> s2_v4 10,356-12,604 -> s2_v5 12,604-now (stage 3 from 19,218 in s3_v2). Discarded branches: s1_v1 2,250-2,268, s2_v1 4,000-4,018, s2_v2 7,000-7,910,
s2_v3 10,356-10,428 (fall-back), s1_v2 5,500-5,602.

| # | Run / version | Start (node UTC) | End (node UTC) | Started from | Iterations | What changed vs previous segment | Tag | Why (problem + evidence) | Ended how | Key result |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | s1_v1 (stage 1) | 2026-11-11 20:02:42 | 2026-11-11 20:40:13 | scratch | 0-750 | Stage 1 PPO, oracle contacts, CSI resets, Table 3 stage-1 rewards [paper]; X2 31 joints, MuJoCo Warp [ours]; 12,000 envs, njmax 300 [ours]; plant `vendor_20260823` [ours]; data `cfg/stage1_sets.json` 9 sets, 1,683 segments, 196.7 min [ours]; seed 0 | paper; ours | Start of track (Note 1) | kept (stopped for GPU benchmark) | 59k-103k env-steps/s; reset phase 0.73 -> 0.33 |
| 2 | s1_v1 | 2026-11-11 20:51:34 | 2026-11-11 21:54:37 | latest.pt 750 | 750-1,250 | 36,000 envs [ours]; njmax 450 [fix]; budget 30,000 -> 12,000 iterations [ours] | ours; fix | GPU use +20 % (user: "make sure the GPUs are well used"); 102 nefc overflow lines at njmax 300 (Note 2) | kept | 116k env-steps/s; reward 0.098 at 1,010 |
| 3 | s1_v1 | 2026-11-11 21:57:21 | 2026-11-11 23:37:25 | latest.pt 1,250 | 1,250-2,000 | Logging switch (`metrics.jsonl` per iteration) [ours] | ours | Monitoring; paper eval at 1,250 (pause) | kept | Paper eval 1,250: 86.6 % pooled (njmax-300 eval defect, Note 6) |
| P1 | planner_v1 (CPU) | 2026-11-11 ~22:30 | 2026-11-11 ~23:00 | rollouts of s1_v1 1,250 | - | Offline MLP planner from hindsight labels [ours] | ours | Score a planner + tracker pair early (coordinator 22:10) | eval hand-over only | Planner-driven closed loop 46.3 % (oracle ~93 %) |
| 4 | s1_v1 | 2026-11-11 23:40:08 | 2026-11-12 00:13:39 | ckpt 2,000 | 2,000-2,250 | Plant `mc_20261024` (measured gains) [user rule]; stage-2-capable trainer installed [ours]; contact placement metric fix [fix] | user rule; fix | User: "Use the ones used for supermimic training last iteration"; Table S2 x U(0.9,1.1) covered none of the measured gains (Note 3) | kept | Paper eval 2,000: 91.0 % (old gains); 2,250: 91.2 % (new gains) |
| P2 | planner_v2 (CPU) | 2026-11-11 ~23:45 | 2026-11-11 23:51 | rollouts s1_v1 2,000 | - | MLP planner, old gains [ours] | ours | Last pair on old gains | superseded | Pair 44.4 % |
| 5 | s1_v1 | 2026-11-12 00:18:24 | 2026-11-12 00:21:24 | ckpt 2,250 | 2,250-2,268 | none (resume after the 2,250 pause) | - | Waiting for the stage-2 switch | superseded (discarded) | - |
| P3 | planner_s1_002250_mc (CPU) | 2026-11-12 ~00:25 | 2026-11-12 ~01:00 | rollouts s1_v1 2,250 | - | MLP planner, measured gains [ours] | ours | Pair for the eval | eval hand-over only | Pair 50.9 % (fall rule 58.3 %), oracle 92.6 % |
| 6 | s2_v1 (stage 2) | 2026-11-12 00:21:32 | 2026-11-12 01:13:04 | s1_v1 ckpt 2,250 | 2,250-2,500 | Stage 2 co-training, attention planner online from scratch, hindsight replay 1.5 M, KL 0.01 / entropy 0.005 [paper]; eta u_obs 200, u_rew 1,700, n_ann 1,500 [ours: paper gives no numbers]; data `stage2_sets` (stair descents gs2dn_v1 out, chains with step > 0.30 m out, 1 flipped Pico clip out) [fix] | paper; ours; fix | Switch rule: first checkpoint under new gains back at 2,000's paper-eval level (91.2 >= 91.0); descents had hip yaw at the limit (Note 4) | restarted (kept) | 71k env-steps/s, 81.9 GB GPU |
| 7 | s2_v1 | 2026-11-12 01:13:37 | 2026-11-12 03:00:11 | ckpt 2,500 | 2,500-3,000 | Planner decode in chunks of 4,096 [fix]; data 2b: soles > 1.5 cm in the ground lifted (`ground_stance`, 99 clips) [fix] | fix | GPU memory 82-84 GB over the 80 GB live-eval guard; ramp / stair soles up to 17 cm inside terrain | restarted (kept) | 51.6 GB, same speed; live paper eval 2,750: 93.9 % (8 sets) |
| 8 | s2_v1 | 2026-11-12 03:00:45 | 2026-11-12 06:49:52 | ckpt 3,000 | 3,000-4,018 | Data 2c: + stair descents v2 (0.05) [fix], + low risers 10-15 cm gs2lo_v1 (0.05) [ours]; stairs 0.08, gst 0.07 [ours]; 94 prop slots | fix; ours | Ladder 10 cm stairs 23/50; no fitted riser < 14 cm in data (Note 5) | superseded by s1_v2 (4,000-4,018 discarded) | eta_obs = 1 at 3,950; paper eval 4,000: 93.0 % (7 comparable sets 94.0 -> 91.6: worse); planner gap flat 42.5 vs 97.5 % |
| P4 | planner_s1v2_004500 (CPU) | 2026-11-12 ~08:20 | 2026-11-12 ~10:00 | rollouts s1_v2 4,500 (80/set) | - | Attention planner from corrected-body labels: scratch, and fine-tune of the s2_v1 4,000 planner (15 epochs) [ours] | ours | Planner trained on old body / old styles (Note 8) | fine-tune used in #10 | Closed loop: fine-tuned 83.0 %, kept 74.5 %, scratch 63.0 % (oracle 99 %) |
| 9 | s1_v2 (stage 1 again) | 2026-11-12 06:52:08 | 2026-11-12 10:12:44 | s2_v1 ckpt 4,000 (tracker + Adam) | 4,000-5,602 | Back to stage 1 [ours]; corrected body `X2_EXCLUDE_PAIRS=hip` [fix]; data 2e: style filter (no jump, crouch, injured, bump), gst jog-ups out, arm clearance 1 cm (2d) [fix; ours]; `--latest-minutes 20` [user rule]; tripwire to fresh s1_v3 [ours] | fix; ours; user rule | False pelvis / hip-roll hull contact, 753-799 N while walking (FINDINGS A1); stage 2 had made reference tracking worse; base gate set by user "get the base more stable first" (Note 7) | superseded by s2_v2 at 5,500 (5,500-5,602 discarded) | Recovered in ~60 iterations; paper eval 4,500: 97.7 % (flat 99.0, blocks 98.3, 4x4 89.8); interim pairs 4,250 17/50, 4,500 5/50 |
| 10 | s2_v2 (stage 2, corrected body) | 2026-11-12 10:13:19 | 2026-11-12 13:49:27 | s1_v2 ckpt 5,500 + planner P4 fine-tuned | 5,500-6,500 | New s2_start 5,500 [ours]; `--planner-init` fine-tuned planner [ours]; n_ann 3,000 [ours]; `--eta-obs-max 0.8` [ours]; `--ramp-gate 2` on flat / blocks training finish [ours]; data 2f: 7 defect clips out, 4x4 0.12, ramps 0.05, arm clearance 2 cm, 1.5 s end stands (`_hold`), stand-at-drop set 0.05 [fix; ours] | ours; fix | Planner did not fit the corrected-body tracker; arms still within 5 mm of real meshes on 8-9 % of frames; pair walked past the last station (Note 8) | restarted (kept) | Apparatus 5,750: 114/200 = 57.0 % (reference pair); 6,000 39.5 %; 6,250 36.0 %; paper eval 6,500: 99.46 % = base gate passed |
| 11 | s2_v2 | 2026-11-12 13:50:01 | 2026-11-12 14:38:33 | ckpt 6,500 | 6,500-6,698 | 720 planner-only probe envs; gate signal probe on 4x4, ramps, flat, blocks; back-off 3 per iteration (default) [ours] | ours | No training number measured planner-driven success; apparatus drop 5,750 -> 6,000 | restarted (kept) | Probe finish flat 82 %, 4x4 94 % |
| 12 | s2_v2 | 2026-11-12 14:39:08 | 2026-11-12 15:07:09 | latest.pt 6,698 | 6,698-6,750 | Data 2g: 1 s start stands (`_hs`), turn_place_v1 0.03, flat 0.12 [ours] | ours | Eval gap (2): step-down from a stand (Note 9) | restarted (kept) | - |
| 13 | s2_v2 | 2026-11-12 15:07:38 | 2026-11-12 18:54:16 | latest.pt 6,750 | 6,750-7,910 | `--freeze-planner` at the 5,750 planner [ours] | ours | Three falling apparatus scores; online planner forgot the turn-around (gap 1, 0.5 % of frames) (Note 10) | superseded: tracker restarted from 7,000 (7,000-7,910 discarded) | 7,000 = champion: 125/200 = 62.5 %, legs 49/39/46/50/43; 7,250 gate fail; 7,500 legs 45/36/41/49/43 |
| 14 | s2_v3 | 2026-11-12 18:54:48 | 2026-11-13 06:00:12 | s2_v2 tracker 7,000, planner frozen 5,750 | 7,000-10,356 | `--gate-backoff 1` (hold only) and `--gate-rebase` [fix]; budget 14,000 | fix | Probe gate backed eta_obs 0.35 -> 0.15 on noise (base = max of a noisy EMA) (Note 11) | superseded by s2_v4 | 8,000: gate 4/5, legs 49/43/45/50/44 = 231/250 (best yet); 9,500 and 10,000 gate fail |
| 15 | s2_v4 | 2026-11-13 06:00:46 | 2026-11-13 07:17:49 | s2_v3 latest 10,356 | 10,356-10,759 | Data 2h: corrected Pico sets v1_1 + mirrors, tp_stairs_all_v1_1 + mirror + turn subset, 4x4 and ramp mirrors on one raster [user rule; ours]; `ACP_HF_FIX=1` [fix]; simcheck fault truncation [user rule]; `ACP_PAPER_DR=1` (target noise, scan U(+-0.1), friction, odometry drift) [paper]; budget 16,000 | user rule; fix; paper | User: "ACP back to the paper" stage-2 items; user's new captures; height-field foot jam 9-19 kN (B8), Warp large-hfield defect (B10) (Note 12) | fell back (automatic, gate B) | Gate A pass; gate B fail: finish 0.9565 < 0.957 (base measured without DR) |
| 16 | s2_v3 (fall-back) | 2026-11-13 07:17:56 | 2026-11-13 07:31:56 | s2_v3 latest 10,356 | 10,356-10,428 | Back to data 2g, no 2h flags (automatic fall-back) [ours] | ours | Gate B fail | superseded (discarded) | - |
| 17 | s2_v4 (re-armed) | 2026-11-13 07:32:03 | running (12,447 at 12:58) | s2_v4 latest 10,750 | 10,750-16,000 (budget) | Watcher gate C rebased on s2_v4's own 2h+DR window [fix]; tool fixes (export: zero-weight set, fault-detector reset, clamped action record) do not touch training [fix] | fix | Gate B was the gate's fault (Note 13) | running | Gate C pass 08:50; 11,750 and 12,000 gate 3/5; 12,250 legs 48/38/49/50/44 = 229/250; finish 0.964, sim_fault_share 0.027 |
| 18 | s2_v5 | 2026-11-13 13:39:27 (GO 13:38:40, ACP agent after the user's yes via the coordinator) | running | s2_v4 latest.pt = 12,604 | 12,604-19,218 (stage 2), then s3_v2 19,218-34,218 (stage 3) | Deterministic Eq. 3 on the USER'S FIXED SCHEDULE (USER_RULES 'ACP fixed stage lengths'): eta_obs, eta_rew linear at 1/6,000 per iteration (N_ann 6,000) from their checkpoint values 0.781 / 0.281 -> eta_obs 1 at 13,918, eta_rew 1 at 16,918; stage 2 ends 2,300 later (19,218); stage 3 exactly 15,000, switched automatically (ops/auto_stage3.sh) [user rule]; planner unfrozen, co-trained, fresh Adam [paper]; balanced replay sets -> clips -> frames [paper principle, applied by us]; data 2h, ACP_HF_FIX, ACP_PAPER_DR, hip pairs excluded; fault-only fall-backs (crash loop, stale, throughput, sim faults, planner updates); per-set finish gate demoted to information (it would fire on the schedule's own effect) [ours] | paper; user rule; ours | Frozen planner and capped / held ramp were judgement-based deviations (user); fixed stage lengths (user) | running | Pre-flight: agentic/state/acp.md 17j, 17k |
| 19 | s3_build (CPU only) | 2026-11-13 13:40-14:00 | - | - | - | Stage 3 to the paper in code/ behind --s3-paper 1 (default off): success 0.5 m / 0.1 m, 0.2 m segment cut, TileReplay (per tile, work-per-metre eviction, types -> tiles -> steps), yaw +-0.4, phase 0, Table 3 stumbling / slippage, qualifying demotion distance, V_fs clipping off in warm-up; cfg stage3v2h (2h reference sets + goal tiles v2b on one raster); flags ops/stage3_flags.env | paper; user rule (terrain) | Stage 3 had to be built and tested before the automatic switch at 19,218 | CPU tests ALL PASS (agentic/state/acp.md 17k) | runs/s3_build/ |
| 20 | s3_v2 data switch (prepared, ARMED, not run) | 2026-11-14 15:29 armed (ACP agent, coordinator request) | waits for runs/s3_v2/GO_spawnfix | s3_v2 latest.pt at GO | same run, same end 34,218 | fix: goal tiles v2 -> v2c (cfg stage3v2i = stage3v2h + goal_tiles_v2c.json). CPU spawn check of all 1,604 tiles found 80 faulting (simulator faults within 10 steps of a training reset; spawns against tall raster walls of the extended map); 62 spawns moved to a clear point on the same tile, 20 tiles take a clean sibling of the same family and level (2 of them from the second check), 2 unchanged (clean on re-check) (tile count and clip ids unchanged). Frozen-policy A/B, all tiles x 4 resets x 30 steps: 203 fault events (114 deep) -> 49 (8). Only DATA changes; resume from latest.pt. Outcome: pending the decision. agentic/state/acp.md 17m. |

## Notes

1. Stage-1 setup (`agentic/state/acp.md` 3, 4): paper values where given (PPO Table S6: 24 steps, 5 epochs, 8 minibatches, lr 1e-4
   KL-adaptive target 0.02, clip 0.2, entropy 0.006); paper-silent choices are ours: termination at 0.5 m base position /
   1.0 rad orientation error, action std 1.0, running obs normalisation, 10 s clip segments, scan noise read as per-episode
   offset U(+-0.1) + per-step N(0, 0.03). No odometry delay / drift. MuJoCo Warp (no restitution). Data: Bones-SEED X2 retargets,
   Pico captures, generated stairs / chains (no ladders, vaults, get-ups).
2. GPU benchmark `agentic/state/acp.md` 4 "GPU use": 36,000 envs = paper's 3 x 12,000; 48,000 adds 3 %. Batch 864,000 transitions per
   iteration, minibatch 108,000.
3. `agentic/state/acp.md` 11-12. The armature-rule gains were x0.4 to x6 off the measured ones on most joints. Without `ACP_PLANT` the
   old gains come back.
4. `agentic/state/acp.md` 13: gs2dn_v1 had the pelvis facing up the stairs while feet walk down (hip yaw 3.43 rad); 39-41 cm step-ups
   failed every try. Clip quality table there. Rule from then: nothing deleted on the shared filesystem, moved to `_trash/`.
5. `agentic/state/acp.md` 14: stair descents v2 re-fitted (57 accepted); low-riser flights 39 climbs + 43 descents; riser check:
   10-12 cm 35 % with planner contacts.
6. All paper evals before 2026-11-12 02:20 UTC ran with njmax 300 (training 450) and carry nefc overflows; 2,750 figure
   corrected from 92.7 to 93.9 % (`agentic/state/acp.md` 16a). "Paper eval" = `category_dense_foot_only.success_pooled` of the
   `*_per334.json` file.
7. `agentic/state/acp.md` 16f-16i; FINDINGS A1, `docs/findings/hip_false_contact.md`. Style filter evidence: kept styles 99.3 % vs
   92.5 % (flat). The planner gap at 3,500: flat 42.5 % planner vs 97.5 % oracle.
8. `agentic/state/acp.md` 16k-16o. Decode top-4 did not help (65.5 vs 67.5 %); arm fix 2 cm: arms within 5 mm of real meshes 0.2 %.
   End stands: `tools/hold_clips.py`; stand at the drop: `tools/stand_edge.py` (78 clips, 1.1 m slab).
9. `agentic/state/acp.md` 16q: turn-around goals 0.5 % of frames, step-downs from a stand 45 segments, turn onto a flight 0.
10. `agentic/state/acp.md` 16r-16s: swap test showed the stall follows the tracker; freezing stops the hindsight buffer overwriting rare
    skills. The 5,750 checkpoint survives only as `eval/pair_s2_v2_005750/tracker_ckpt_005750.pt` (retention pruned the run copy).
11. `agentic/state/acp.md` 17a-17b: selection rule (champion = best leg product passing the gate) and noise rule (a pair difference
    counts at >= 10 failures over five legs). The hold-out gate is information only since 2026-11-13 (USER_RULES).
12. `agentic/state/acp.md` 17c-17d; FINDINGS B8, B10; `docs/findings/acp_paper_vs_code.md` (25 DN items, ranked list). Preflight
    `reports/acp_s2h_fix1.json`: max terrain depth 45.5 mm (was 231.4). With HF fix the field friction (0.3) applied: fixed by the
    paper-DR friction item.
13. `agentic/state/acp.md` 17e-17i. Code folders folded into one `code/` at 06:31 (equivalence: 174 metrics and all tensors bit-identical
    over 3 iterations). Old-set finish under DR flat at 0.956-0.957 since 10,407; stairs sets dropped most (gst 0.978 -> 0.882).

## Gaps (could not establish)

- Exact start / end minutes of the CPU planner jobs P1-P4 (no timestamps read; times are from `agentic/state/acp.md` text, approximate).
- Whether the RNG state is restored on resume: `--seed 0` is passed implicitly on every start, so every restart may replay the
  same random stream; not checked in `acp/train.py`.
- The s1_v1 segment-1 run used njmax 300 by default; the supervisor default became 450 only at segment 2 (inferred from the
  overflow count, not from a saved command line). No per-segment command line is stored for s1_v1 / s2_v1 other than the
  resume commands written in `agentic/state/acp.md`.
- The fine-tuned planner P4 itself was initialised from the s2_v1 4,000 planner, which was trained online on the old body and
  old data: its full history cannot be reproduced exactly from scratch.
- Gains 4,000 -> 4,018 (s2_v1) and 5,500 -> 5,602 (s1_v2) are discarded; whether their checkpoints still exist was not checked.
- The 2 November SuperMimic-based ACP run (planner + tracker, about 25 % on the apparatus) is a previous round and not in this ledger.
- Scores for s2_v4 pairs 10,500-11,500 after the 11:22 re-release were not yet in `eval/autoscore/results.csv` when this was written.
