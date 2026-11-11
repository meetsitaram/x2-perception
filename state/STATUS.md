# Status (the coordinator keeps this current)

Updated 2026-11-11.

| Track | Node | State | Newest checkpoint | Last common-eval result | Blocker |
|---|---|---|---|---|---|
| ACP | acp <acp-node> | not started | none reusable (earlier runs used SuperMimic) | none yet | SSH access |
| SSR | ssr <ssr-node> | not started | none reusable (earlier runs used a SuperMimic initialisation) | none yet | SSH access |
| Hiking | hiking <hiking-node> | not started | G1 only (`model_20000`); X2 port needed | none yet | SSH access |
| Eval | all | not started | | baseline "about 25%" to re-measure | SSH access; open items in `docs/EVAL.md` |

- Local G1 reproduction (`hiking-in-the-wild`): paused at the user's request at `model_20000` of 30,000 on 2026-11-11. Do not
  resume or change it without the user.
- Nodes: SSH was refused for users `ubuntu` and `root` with the desktop's `~/.ssh/nebius_corpus` key (2026-11-11). Needed from
  the user: the user name and key, and the shared filesystem's mount path.

## Decisions made without the user (date, decision, reason)

- 2026-11-11: a separate repo, `perception-agents`, holds the team (roles, nodes, eval, status) so it survives a lost session.
  Track code stays in its own repos and trees.
- 2026-11-11: the eval monitor, not the tracks, scores checkpoints, with one definition (`docs/EVAL.md`), so the three numbers
  are comparable.
- 2026-11-11: the Hiking track ports to the X2 from the code, not from the G1 checkpoints (different robot).
- 2026-11-11 (user): ACP and SSR follow their papers' own RL methods this round; no SuperMimic tracker or checkpoints on top.
  Earlier ACP and SSR checkpoints that depend on SuperMimic are not reused.
- 2026-11-11 (user): a station in the eval is a 1 s stand in the arrival disc; button presses do not count toward success ("station is just a 1 s stand - that's good"). Time limit 180 s per trial, confirmed by the user the same day.
- 2026-11-11 (user): the eval's side-hit / stumble penalty score is for the eval only; it does not have to go into any track's training rewards ("they don't have to go into training - the penalties"). Tracks are not required to add it. Whether the Hiking track's separately planned stumble/edge/clearance training rewards (extended run) stay is the user's call; asked.
- 2026-11-11 (user): the Hiking track's extended run includes the planned training penalties (stumble, foothold, foot clearance): "INCLUDE THEM". The eval's stumble score itself stays eval-only.
- 2026-11-11 21:35 UTC (Hiking agent, accepted by coordinator): X2 run resumed from model_250 with 8,192 robots (22,800 samples/s, +43% over 4,096). 12,288 gave 24,500/s but PhysX dropped contacts (collision stack overflow), so it was not used; enlarging the PhysX buffers can be tried at a later checkpoint within the 10-minute stop limit. Steps per robot, minibatch count, epochs and total iterations unchanged, so each iteration uses twice the paper's samples; iteration numbers are not one-to-one with the G1 run; 30,000 iterations is about 72 h.
- 2026-11-11 22:10 UTC (coordinator): eval monitor told to re-run the 2 November ACP policy pair (planner `planner_v2_hist1` + tracker `acp3_v2` ckpt 1000) under the current eval rule on the desktop CPU, to replace "about 25%" with a like-for-like starting number. Eval only; those SuperMimic-based weights are not used in any training. Reason: the goal is measured against this baseline and the rule has become stricter since.
- 2026-11-11 22:10 UTC (coordinator): SSR told to stop r3 at iteration 500 if still under 20% of commanded speed and bisect on flat ground (tracking terms only, then add terms back).
- 2026-11-11 22:10 UTC (coordinator): ACP told to bring the planner forward beside stage 1 from the existing stage 2/3 code in gwbc-v6, so a planner-plus-tracker pair can be scored early.
- 2026-11-11 22:30 UTC (coordinator, ACP): stage 2 co-training port starts now beside stage 1; stage 1 switches to stage 2 on a criterion (paper eval pooled >= ~95% with the two weak sets above 85%, or a plateau), not at a fixed 12,000 iterations. No geometric foothold picker and no old SuperMimic planner paired with the new tracker; the old pair is scored only as the baseline. First ACP pair (tracker 1,250 + MLP planner v1): 46.3% on the training reference terrains with planner contacts (108 episodes) against 93.5% with oracle contacts; paper eval of the tracker alone 86.6% over 9,018 episodes (paper: at least 98.5% at the end of stage 1).
