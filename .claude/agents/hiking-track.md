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
