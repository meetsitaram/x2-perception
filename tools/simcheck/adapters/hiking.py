"""Hiking adapter: not runnable on the CPU. Hiking trains in Isaac Lab (PhysX on the GPU, InstinctLab parkour task), which has
no CPU path and no MuJoCo model, and a second GPU job beside training is not allowed. Its preflight is therefore run only
while training is paused (x2_port/pause_and_eval.sh pauses it), with Isaac's own physics:

  1. Height-field size: not applicable (PhysX triangle meshes); instead check every tile mesh is a closed union of boxes
     without overlapping faces at seams (ghost contacts, FINDINGS B8 "Isaac").
  2. Foot drop: spawn a free foot rigid body with the training foot collider (convex hull) on flat, slope and edge cells of
     the training terrain at 1, 2, 3, 4.5 m/s (+1 m/s sideways), 150 N sideways pull at 0.1-0.3 s; same limits as
     simcheck.core (deeper than 1 cm for more than 0.07 s; moved less than 2 cm by the pull).
  3. Random-action rollouts: 4 envs per terrain type, 2 s, PD targets default + U(-0.5, 0.5) rad every 0.1 s; fail on any NaN
     in the root state, body speed jump over 20 m/s per physics step, root below the terrain by more than 5 cm, or a
     diverged share over 1 %.
The episode-end rule (simcheck.episode_end) and the heartbeat alarm (simcheck.alarm) do apply to Hiking: see
patches/hiking.diff and the README."""


def build(opts: dict) -> dict:
    raise SystemExit("hiking: preflight needs Isaac Sim on the GPU; run it during a training pause (see this module's doc)")
