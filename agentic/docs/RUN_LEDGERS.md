# Run ledgers and recipes

Why (user, 2026-11-13): "Make note of all these iterations and changes we are doing somewhere. Because once we find the results
are good, we might run another bigger training run on more GPUs without interruptions as a single run. And we won't be there to
evaluate at every checkpoint to make decisions whether to proceed or stop like we are deciding for ACP."

## Files

| File | What it holds |
|---|---|
| `models/acp/LEDGER.md`, `models/ssr/LEDGER.md`, `models/hiking/LEDGER.md` | One chronological table per track (oldest first) of every run segment: every start, restart, resume and fall-back. Columns: run or version, node-clock start and end (UTC), started from (checkpoint or scratch), iteration range, what changed against the previous segment, tag, why (the problem and its evidence), how it ended (kept, fell back, crashed, superseded, running) and the key result. Notes and a Gaps list follow the table. |
| `models/acp/RECIPE.md`, `models/ssr/RECIPE.md`, `models/hiking/RECIPE.md` | The current best recipe written as one uninterrupted run from scratch. It covers every setting with its value, the data sets with versions and paths, and the schedule as iteration numbers. Every automatic rule that replaces a human decision is a code-checkable condition with a threshold, the measure it reads and the file that computes it. Also seeds, environment counts, GPU settings, open items that would change the recipe, and where today's run differs from the recipe because of its history. |

Tags on every change and setting: **paper** (follows the paper), **fix** (a defect fixed), **user rule** (decided by the user;
the words are in `agentic/docs/USER_RULES.md`), **ours** (our own addition).

## The rule (agents and coordinator)

1. **Every restart or change adds one ledger row the same day.** This covers a resume, a fall-back, a flag, reward, data,
   terrain, command, physics or code change, and a stage switch. Whoever makes the change (the track agent, or the coordinator
   when it decides) writes the row. A row needs the node-clock time, the checkpoint it started from, the tag and the evidence.
   Fill in the end time and the outcome when the segment ends.
2. **Update the recipe when a change is kept.** It is kept when it passes its gate or the track adopts it. A change that fell
   back stays in the ledger only. In the recipe, write the change as the from-scratch schedule (the iteration at which it turns
   on) and turn any human decision it needed into an automatic rule with a threshold.
3. Keep "Where today's run deviates" and "Open items" in each recipe current. A big run starts only from a recipe with no open
   item that would change it, and only after the user approves.
4. Sources of detail stay where they are: `agentic/state/<track>.md` (dated sections and HOW TO RESUME), `agentic/state/STATUS.md`
   (decisions log), `docs/FINDINGS.md`. The ledger points to them and does not copy them.
5. Same limits as the rest of the repo: no node addresses, logins, cloud ids or home paths, and no BONES-SEED or paper data.

First written 2026-11-13 from the state files, the decisions log, the findings, the audits and the run folders on the shared
filesystem.
