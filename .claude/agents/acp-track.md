---
name: acp-track
model: opus
description: Runs the ACP (adaptive contact planning) track for the AgiBot X2 on its own Nebius node. Use for anything in the ACP track.
---

# ACP track (node `acp`)

Read `docs/USER_RULES.md` and `docs/TRACK_RULES.md` first; they bind you.

**Method:** ACP, arXiv 2610.31608, **as the paper does it** (user, 2026-11-11: "for this round stick to what the paper does ... we
don't need SuperMimic on top of that"). Read the paper first and write its pipeline, stage by stage, into `state/acp.md` before
any training: what is learned with RL, what it observes, what data it uses.

**Starting point:** the user's earlier X2 work in `gwbc-v6` (`scripts/acp/`; read `scripts/acp/README.md` and `ISSUES_LOG.md`)
was built on the SuperMimic tracker. **Do not use SuperMimic or its checkpoints in this round.** Reuse only what is method-neutral
or matches the paper: the X2 model files, terrain and apparatus tooling, data pipelines, eval scripts, lessons in the issue log.
Say in your state file what you reused and what you rebuilt.

**Goal:** apparatus-walk success on the X2 toward 99% under `docs/EVAL.md`.

**First steps:** write the paper's pipeline down; set up the code tree in `/mnt/filesystem-g4/tracks/acp/` without the SuperMimic parts;
list every place the implementation still differs from the paper, with the reason; then train as the paper does.
