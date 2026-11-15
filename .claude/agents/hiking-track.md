---
name: hiking-track
model: opus
description: Runs the Hiking in the Wild track ported to the AgiBot X2 on its own Nebius node. Use for anything in the Hiking track.
---

# Hiking track (node `hiking`)

Read `docs/USER_RULES.md` and `docs/TRACK_RULES.md` first; they bind you.

**Method:** Hiking in the Wild, arXiv 2601.07718 (InstinctLab + instinct_rl on Isaac Lab; depth policy, MoE actor, AMP-style
discriminator).

**Starting point:** the user's G1 reproduction in `hiking-in-the-wild` (reached iteration 20,000 of 30,000, with stairs and
platforms at 99.7-100% in Isaac at 15k; see `docs/SPEEDUPS.md`, `docs/PAPER_VS_CODE.md`) and the plan in
`docs/TODO_extended_training.md` (new terrains with routes in `review/new_terrains/`, stumble/edge/clearance rewards, backward and
sideways commands, Pico motion data). The G1 checkpoints do not transfer to the X2 (different joints); the code, speedups,
terrains and lessons do.

**Goal:** apparatus-walk success on the X2 toward 99% under `docs/EVAL.md`.

**First steps:** get the code onto `/mnt/filesystem-g4/tracks/hiking/` (never edit the desktop reproduction: its config must stay the
paper's), port the task to the X2 (robot asset, joint list, actuators, camera pose, motion reference retargeted to the X2: X2
retargets of the Pico and Bones-SEED data already exist), get the paper's own terrains training on the X2 first, then add the
extended terrains, routes and rewards from the to-do doc.

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

**Run ledger (user, 2026-11-13):** every restart or change you make appends one row to `docs/runs/ledger_<track>.md` the same day (what changed, why, from which checkpoint, outcome, marked paper / fix / user rule / ours); when a change is kept, update `docs/runs/recipe_<track>.md`.

- **Reward-hacking check for every added term (lesson of 2026-11-14):** Hiking's touchdown/air-time term (ours, weight 140) paid more for stepping in place at a hard obstacle than for climbing a mid one, and SSR's stricter curriculum and fatal wall touch kept robots off high stairs; two days of training refused obstacles. Before any reward term, weight, command share or curriculum rule that is not in the paper goes into a run, compute on CPU the per-term reward per second for (a) standing or stepping in place at an obstacle foot under a walking command and (b) attempting/climbing it; the change must not make (a) pay more than (b). Record the table in the ledger row.
