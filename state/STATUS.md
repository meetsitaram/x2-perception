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
