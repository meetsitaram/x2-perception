# Track code: three isolated folders

The user (2026-11-12): "Let's have the code as three separate folders in the perception agents repo."

| Folder | What it is | Working copy on the shared filesystem |
|---|---|---|
| `acp/code/` | The ACP track's code for this round (PPO tracker, contact planner, stage 1-3 trainers, tools, evaluation) | `/mnt/filesystem-g4/tracks/acp/code/` |
| `ssr/code/` | The SSR track's code (from `ssr-x2`, with this round's changes; no SuperMimic script) | `/mnt/filesystem-g4/tracks/ssr/code/` |
| `hiking/x2_port/` | The Hiking track's X2 port: task files, supervisor, launch, export and evaluation scripts, the MuJoCo port | `/mnt/filesystem-g4/tracks/hiking/code/x2_port/` |

The Hiking track's base code is the separate `hiking-in-the-wild` repository (branch `g1-20k-and-extended-terrain`, commit
`27ec8d3`; a git bundle of it is at `/mnt/filesystem-g4/tracks/hiking/hiking-in-the-wild.bundle`) plus InstinctLab, instinct_rl
and Isaac Lab at the commits named in `state/hiking.md`; `x2_port/apply.sh` copies the X2 files into it.

## Rules
- **Isolation:** a track agent edits only its own folder and its own tree on the shared filesystem. No code is shared between
  tracks; common things (robot model, plant files, motions, terrains) are data under `/mnt/filesystem-g4/data/`.
- **Where work happens:** the agents work in the shared-filesystem copy, because that is what the nodes run. This repo is the
  record. `bash scripts/sync_track_code.sh --commit` (from the desktop) pulls the three folders, checks them and pushes; the
  coordinator runs it at every 30-minute refresh and after every agent report that changed code. `LAST_SYNC.txt` has the time.
- **Not in the repo:** checkpoints, data, environments, robot meshes, renders, files over 2 MB.
- **No addresses or logins:** the sync replaces the node login in paths with `NODE_USER`; agents should write `$HOME` or
  `Path.home()` in the first place. `scripts/check_no_secrets.sh` must pass before any push.
- **Restoring a node from the repo:** copy the folder back to the shared filesystem path above, replace `NODE_USER` with the
  login from `config/nodes.local.json` if any file still has it, then follow the track's `state/<track>.md`.
