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
