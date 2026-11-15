# How x2-perception was built: the agent team

The three models in `models/` were trained by a standing team of Claude agents, with three methods run side by side on the
AgiBot X2. This folder is the team's memory: roles, rules, the user's decisions, state files, the review dashboard and the
node scripts. A new Claude Code session opened in this repo picks the work up from here alone (the root `CLAUDE.md` imports
`agentic/CLAUDE.md`, the coordinator instructions).

**Goal (set by the user, 2026-11-11):** take the apparatus walk from about 25% success (the user's figure; the first job of the
eval agent was to re-measure it under `docs/EVAL.md`) to close to 99%. X2 only for this round. The final score is the walk on
both humanoid-challenge apparatuses, ours and NIST's, which are held out of training.

## Contents

| Path | What |
|---|---|
| `CLAUDE.md` | coordinator instructions (loaded through the root `CLAUDE.md`) |
| `../.claude/agents/` | the track and eval agent definitions (Claude Code needs them at the repo root) |
| `docs/` | how the team runs: `USER_RULES.md` (every user decision, with their words), `USER_TODO.md`, `TOKEN_EFFICIENCY.md`, `HANDOVER.md`, `RUNBOOK.md`, `TRACK_RULES.md`, `SHARED_FS.md`, `REVIEW_LATER.md`, `RUN_LEDGERS.md` (the ledger and recipe rules), `TRACK_CODE.md` (how track code is synced into `models/`) |
| `state/` | `STATUS.md` (status and the decisions log) and one state file per track and for the eval monitor |
| `dashboard/` | the user's review dashboard: data files, page, `prepare.py`, `progress_from_metrics.py` |
| `config/` | `nodes.json` (node roles; addresses and logins are in the git-ignored `nodes.local.json`) |
| `scripts/` | `node_check.sh`, `sync_track_code.sh`, `check_no_secrets.sh`, `serve_dashboard.*` |

## The team

| Role | Agent definition | Where it works | Owns |
|---|---|---|---|
| Coordinator | `agentic/CLAUDE.md` (the main session, opened in this repo) | the user's desktop | plan, priorities, status, talking to the user |
| ACP track | `.claude/agents/acp-track.md` | node `acp` | adaptive contact planning on the X2 (`gwbc-v6`, `scripts/acp`) |
| SSR track | `.claude/agents/ssr-track.md` | node `ssr` | SSR on the X2 (`ssr-x2`) |
| Hiking track | `.claude/agents/hiking-track.md` | node `hiking` | Hiking in the Wild ported to the X2 (`hiking-in-the-wild`) |
| Eval monitor | `.claude/agents/eval-monitor.md` | reads all three; runs evals on the track's own node when it is idle, or on the desktop CPU (MuJoCo) | the common apparatus eval, renders, reports to the coordinator |

Nodes are in `agentic/config/nodes.json`. Each track has one RTX 6000 node on Nebius and its own code checkout; all three
mount one shared filesystem (`agentic/docs/SHARED_FS.md`).

## Rules that keep the three tracks apart

- One track, one node, one code tree (`/mnt/filesystem-g4/tracks/<track>/code`), one Python environment, one run folder. A
  track never edits another track's tree, and never imports from it.
- Anything a track wants from another (a motion file, a terrain, a checkpoint) is **copied** through
  `/mnt/filesystem-g4/exchange/` with a note in `agentic/state/STATUS.md`; never linked.
- Shared, read-only inputs live in `/mnt/filesystem-g4/data` (motions, apparatus CAD rasters, terrain generators). Changes to
  them go through the coordinator.
- Every track is scored by the **same** eval (`docs/EVAL.md`), run by the eval monitor, not by the track itself.
- Track code is worked on in the shared-filesystem copy; `bash agentic/scripts/sync_track_code.sh --commit` pulls it into
  `models/<track>/` of this repo (`agentic/docs/TRACK_CODE.md`).

## Starting or resuming the team

**Lost the session? Follow `agentic/docs/RUNBOOK.md`.** The user's rules and decisions are in `agentic/docs/USER_RULES.md`;
every agent reads that file first.

1. Open a Claude Code session in the repo root; it becomes the coordinator (`CLAUDE.md` -> `agentic/CLAUDE.md`).
2. Read `agentic/state/STATUS.md` (status and the decisions log).
3. Check the nodes: `bash agentic/scripts/node_check.sh`.
4. The user's review dashboard is https://claude.ai/artifact/JaZDeLFXXN28HLMd1Sia3V; its data files and page are in
   `agentic/dashboard/` (update the JSON, upload, rewrite the store documents; `agentic/dashboard/README.md`).
5. Start or resume each track agent with its definition; each one first reads its own `agentic/state/<track>.md`.

## Standing constraints (from the user)

- Nothing is bought; never search Amazon. Bones Studio credit on shown results; no dataset clip names on screen.
- Robot videos come from the simulator's own camera and are checked against simulator data; anything else is labelled.
- Do not commit or push unless the user asks. Large downloads are rate-capped and need the user's okay.
- Long runs go through a supervisor with stall detection and auto-resume; one GPU job per node; every shell command has a
  timeout.
- Do not wait on the user: take the recommended option, record it in `agentic/state/STATUS.md`, keep going.
