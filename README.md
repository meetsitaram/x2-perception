# perception-agents

A standing team of Claude agents that trains perceptive locomotion for the **AgiBot X2** on the IEEE loco-manipulation apparatus,
with three methods run side by side. The final score is the walk on both humanoid-challenge apparatuses, ours and NIST's,
which are held out of training (`docs/EVAL.md`). This repo is the team's memory: roles, node map, shared-filesystem layout, the eval
definition and the current status. A new session can pick the work up from here alone.

**Goal (set by the user, 2026-11-11):** take the apparatus walk from about 25% success (the user's figure; the first job of the eval
agent is to re-measure it under `docs/EVAL.md`) to close to 99%. X2 only for this round.

## The team

| Role | Agent definition | Where it works | Owns |
|---|---|---|---|
| Coordinator | `CLAUDE.md` (the main session, opened in this repo) | the user's desktop | plan, priorities, status, talking to the user |
| ACP track | `.claude/agents/acp-track.md` | node `acp` | adaptive contact planning on the X2 (`gwbc-v6`, `scripts/acp`) |
| SSR track | `.claude/agents/ssr-track.md` | node `ssr` | SSR on the X2 (`ssr-x2`) |
| Hiking track | `.claude/agents/hiking-track.md` | node `hiking` | Hiking in the Wild ported to the X2 (`hiking-in-the-wild`) |
| Eval monitor | `.claude/agents/eval-monitor.md` | reads all three; runs evals on the track's own node when it is idle, or on the desktop CPU (MuJoCo) | the common apparatus eval, renders, reports to the coordinator |

Nodes are in `config/nodes.json`. Each track has one RTX 6000 node on Nebius and its own code checkout; all three mount one
shared filesystem (`docs/SHARED_FS.md`).

## Rules that keep the three tracks apart

- One track, one node, one code tree (`/mnt/filesystem-g4/tracks/<track>/code`), one Python environment, one run folder. A track never edits
  another track's tree, and never imports from it.
- Anything a track wants from another (a motion file, a terrain, a checkpoint) is **copied** through `/mnt/filesystem-g4/exchange/` with a
  note in `state/STATUS.md`; never linked.
- Shared, read-only inputs live in `/mnt/filesystem-g4/data` (motions, apparatus CAD rasters, terrain generators). Changes to them go
  through the coordinator.
- Every track is scored by the **same** eval (`docs/EVAL.md`), run by the eval monitor, not by the track itself.

## Starting or resuming the team

**Lost the session? Follow `docs/RUNBOOK.md`.** The user's rules and decisions are in `docs/USER_RULES.md`; every agent reads that file first.

1. Open a Claude Code session in this folder; it becomes the coordinator (`CLAUDE.md`).
2. Read `state/STATUS.md` (status and the decisions log).
3. Check the nodes: `bash scripts/node_check.sh`.
4. The user's review dashboard is https://claude.ai/artifact/JaZDeLFXXN28HLMd1Sia3V; its data files and page are in `dashboard/` (update the JSON, upload, rewrite the store documents).
5. Start or resume each track agent with its definition; each one first reads its own `state/<track>.md`.

## Standing constraints (from the user)

- Nothing is bought; never search Amazon. Bones Studio credit on shown results; no dataset clip names on screen.
- Robot videos come from the simulator's own camera and are checked against simulator data; anything else is labelled.
- Do not commit or push unless the user asks. Large downloads are rate-capped and need the user's okay.
- Long runs go through a supervisor with stall detection and auto-resume; one GPU job per node; every shell command has a timeout.
- Do not wait on the user: take the recommended option, record it in `state/STATUS.md`, keep going.
