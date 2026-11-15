# For the user: open decisions

Kept by the coordinator (user, 2026-11-12: "Go with your recommendation if I don't respond and keep it as todo list for me").
Each row is a decision or a fact the user was asked for. Until the user answers, the team goes with the coordinator's
recommendation in the third column, and that is recorded as a decision made without the user in `state/STATUS.md`. The user can
overturn any of them; the last column says what overturning costs. Answered rows move to `docs/USER_RULES.md` and are removed
here. Not covered by this rule: git commits and pushes (held until the user says so), anything that deletes data, starts a
second GPU job, touches the local G1 run, or changes a held-out apparatus course.

| # | Asked | Question | Going with (coordinator's recommendation) | To overturn |
|---|---|---|---|---|
| 1 | 2026-11-12 | The leaning board at the door of our apparatus is modelled as a solid wedge up to 0.48 m; the real board has open space under it. Correct the course model? | Wait for the labelled test (board as a tilted box; SSR's stalled trials, ACP and Hiking for side effects). The held-out course is not changed without the user's yes, whatever the test shows; the result will be put here. | Nothing yet. |
| 4 | 2026-11-12 | Should a track's own eval command rule be its headline number (SSR's 0.40 m/s speed floor, ACP's turn rule), as Hiking's already is? | No: the standard command stays the headline for SSR and ACP; the variants are run and reported beside it, labelled. Hiking's card stays its standard because it is its training command. | Re-label existing results; no re-runs needed. |
| 5 | 2026-11-12 | Hiking's chest camera (61.6 deg down, 58 deg vertical field, as the real RealSense) sees from the toe tips to 1.66 m ahead when upright; the user asked for about 10 cm behind the toes to at least 2 m. The policy bends its waist back to the joint limit (-18 deg) in every checkpoint, which reaches 3 m and loses the feet; nothing in the reward resists it. 2 m cannot be met by pitch alone with a 58 deg field (it needs about 67 deg). Options: (a) pitch the camera up about 5 deg (56-57 deg down): toe tips to 2.0 m; (b) a wider field, which the real camera does not have; (c) leave the mount and penalise the lean. Is the real camera's pitch adjustable, and to what? | Leave the mount and the lean as they are for now (watch item; torso pitch is logged per terrain from v4h on). Option (a) is the recommendation if the real mount can be pitched up 5 deg: a fine-tune, not a restart, as its own change after the extended-map trial. Not done without the user, because training must match the real mount. | Revert the pitch at a restart; a short dip while the depth encoder relearns. |
| 6 | 2026-11-12 | The full route is run only after a checkpoint passes all five station legs (the user's rule). What counts as passing a leg? | 45 of 50 on each leg, the same mark as the obstacle levels. | Change one constant; re-read existing leg scores against the new mark. |

## Standing items (no decision needed, for the user's attention)

- Git: nothing has been committed or pushed since 2026-11-12 about 04:58 UTC, by the user's instruction. The working tree holds
  all of today's changes.
- The claude.ai dashboard cannot be reached from this session; only the local page (`http://localhost:8765/preview.html`) is
  current.
- Review later: `docs/REVIEW_LATER.md` (SSR's action scale).

Added after the audit (same columns as the table above):

| 7 | 2026-11-12 | Eval floor: replace the height-field physics floor with planes and boxes (audit, 2026-11-12); all apparatus scores get re-run and may move a few points | Do it (acting on this) | done 19:00 UTC: solid floor is the default; ACP 7,000 scored 27/50 on it against 26/50 control |
| 8 | 2026-11-12 | Stair tiles on your extended map put the stair foot 0.6-1.0 m from the tile edge, so there is no room for side approaches or a walk-in. Regenerate the straight / L / U stair tiles with the foot 2.5-3.0 m inside the tile, as a separate added set (your original map untouched)? | Do it (acting on this) | open |
| 9 | 2026-11-13 | The new simulator pre-flight fails every current build (real defects: feet trapped at edges, hands and wrists sinking into terrain in the GPU simulator). Allow a launch while they are being fixed if it is no worse than the running build and each failure has an owner and a fix date? And: depth limit 5 cm for feet and sustained contacts, 10 cm for brief non-foot impacts? | Yes to both (acting on this) | open |
| 10 | 2026-11-13 | ACP back to the paper on the 25 paper-specified items | Decided by the user: yes, except the goal terrain (keep our families + the extended map), env counts and data choices | done |
| 11 | 2026-11-13 | Double the simulated robots on SSR and Hiking at their next planned restarts | Decided by the user: go with the recommendation (yes) | done |
- (2026-11-13) ACP depth student (teacher-student DAgger, depth -> height map under the ACP policy): revisit once ACP performs well with the perfect height map (user). Needs a GPU node (SSR's frees at 20,000) and a teacher checkpoint.
- (2026-11-14, user) Stall penalty: penalise staying in one place for a long time while a forward command is active (Hiking first; SSR has a similar candidate). Not implemented yet; owner Hiking agent, after v9's first judgement.
- (2026-11-15, user, after waking) Depth camera -> height map for ACP: run the literature review (last 12 months, depth-camera terrain mapping for legged/humanoid robots), then decide the approach. Paused 2026-11-15 ~06:45 UTC; plan in docs/USER_RULES.md "Depth camera -> height map for ACP".
