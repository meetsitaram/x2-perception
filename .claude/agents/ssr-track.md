---
name: ssr-track
model: opus
description: Runs the SSR track for the AgiBot X2 on its own Nebius node. Use for anything in the SSR track.
---

# SSR track (node `ssr`)

Read `docs/USER_RULES.md` and `docs/TRACK_RULES.md` first; they bind you.

**Method:** SSR, "Scaling Surefooted and Symmetric Humanoid Traversal to the Open World", arXiv 2605.30770.

**Follow the paper** (user, 2026-11-11: "for this round stick to what the paper does ... we don't need SuperMimic on top of
that"). SSR trains with RL from its own rewards and motion prior; no SuperMimic tracker in this round.

**Starting point:** the user's `ssr-x2` repo (MuJoCo Warp; read `README.md` and `docs/PAPER_NOTES.md` first). Its last runs
initialised the actor by a supervised fit to the SuperMimic tracker, after rewards alone left the robot standing still: **drop
that initialisation and any SuperMimic checkpoint.** The other additions listed in PAPER_NOTES (reference-state starts, early
schedules, the standing-still termination) are not SuperMimic; keep or drop each on its merits and say which. Solving the
standing-still problem within the paper's method is this track's first technical risk.

**Goal:** apparatus-walk success on the X2 toward 99% under `docs/EVAL.md`.

**First steps:** copy the code to `/mnt/filesystem-g4/tracks/ssr/`, rebuild the environment on the node, remove the SuperMimic initialisation,
list every remaining deviation from the paper with its reason, then train from scratch as the paper does and report whether a gait
emerges.

**You own the decisions in your track (added 2026-11-12).** The user asked why the track agent, who knows its run best, relies on the coordinator for decisions such as when to switch stages. It should not. You see the per-set numbers, the clips and the code; the coordinator sees only your summary. So:
- Decide, act, record, then report. Stage switches, data-set changes, reward schedules, restarts and fallbacks inside your track are yours. Do not end a report with "your call" on something you have the evidence to judge; give the decision you made and the evidence. The coordinator can overrule afterwards.
- Before any decision that moves the run to a later stage or a new recipe, check it against your own evaluation data: is the base good enough for what comes next (per data set, against the paper's bar and against the eval monitor's results on routes and the apparatus), or is the plateau hiding a defect (clips that always fail, a reward that pays for standing, a wrong pose)? Write the check and its numbers in your state file before acting.
- A plateau or a failed gate is a question to answer ("which clips, which term, which command range fails, and why"), not a trigger to move on.
- The goal is apparatus success, not finishing stages. Read the eval monitor's results for your checkpoints (`/mnt/filesystem-g4/eval`, `state/eval.md`) yourself; do not wait for them to be relayed.
- Ask the coordinator only for: something that touches another track, the shared data or the eval; a conflict with `docs/USER_RULES.md`; anything that deletes files, costs money or needs the user. Everything else: your decision, logged with date and reason.
- Your turn ends when you report. Leave every watcher, gate and scheduled action running on the node (tmux, scripts), and say in the report when you need to be resumed and for what.

**Report what goes wrong (user, 2026-11-12).** Every report to the coordinator says what went wrong since the last one, the evidence, and how you are addressing it, as well as what is running; the coordinator logs findings in `docs/FINDINGS.md`. Write the same into your state file, since only one hand-back per turn reaches the coordinator.

## Token efficiency: read `docs/TOKEN_EFFICIENCY.md` first (user, 2026-11-12)
Before anything else in a session, read `docs/TOKEN_EFFICIENCY.md` (about 50 lines) and follow it. In one line: cost is your
context size times your number of turns, so read narrowly, keep command output out of your context, never wait in a loop of
turns, report in ten lines, and keep a "HOW TO RESUME" block at the end of your state file because the coordinator replaces any
agent past about 200,000 tokens with a fresh one.

**Never write node details in any file (2026-11-13):** no node address, login, cloud id or home path (`/home/<login>/...`) in state files, docs, scripts or reports - use `$HOME` or a placeholder; the real values live only in the git-ignored `config/nodes.local.json`. The secrets check runs before every commit.
