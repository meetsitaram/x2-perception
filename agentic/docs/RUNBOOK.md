# Runbook: restart the team from scratch

Use this when the session that ran the agents is gone. Training itself keeps running on the nodes (tmux, supervisors); only the
agents that watch and steer it are lost.

## 1. Become the coordinator
1. Open a Claude Code session in `C:\Users\PC\projects\perception-agents` (it reads `CLAUDE.md`).
2. Read, in this order: `agentic/docs/HANDOVER.md` (latest state and open items), `agentic/docs/USER_RULES.md`, `agentic/state/SHARED_LESSONS.md`, `agentic/state/STATUS.md`, `docs/EVAL.md`, `agentic/docs/TRACK_RULES.md`, `agentic/config/nodes.json`.
3. `bash agentic/scripts/node_check.sh` (GPU use, disk, shared filesystem), then read the heartbeats:
   `ssh -i ~/.ssh/nebius_corpus -o IdentitiesOnly=yes <user>@<ip> 'cat /mnt/filesystem-g4/status/*.json; tmux ls'`.
4. Read each `agentic/state/<track>.md`: it has what is running, the newest checkpoint and the exact resume and stop commands.
5. Each track's code is in `models/<track>/` of this repo (`agentic/docs/TRACK_CODE.md` says how to put it back on a node).
6. A copy of this repo is on the shared filesystem at `/mnt/filesystem-g4/perception-agents` in case the desktop copy is lost.

## 2. Start the agents again
Start one agent per role (in Claude Code: the Agent tool, general-purpose, Opus 5.5 (`model: opus`; the user switched from Fable 5.2 on 2026-11-12 when its weekly allowance ran out), in the background; or
one session per role). Give each this prompt, with its own track filled in:

> You are the `<acp | ssr | hiking>` track agent of the perception-agents team (or: the eval monitor). Read, in this order, and
> follow: `agentic/docs/USER_RULES.md`, your brief `.claude/agents/<role>.md`, `agentic/docs/TRACK_RULES.md`, `agentic/docs/SHARED_FS.md`, `docs/EVAL.md`,
> `agentic/config/nodes.json`, `agentic/state/STATUS.md`, `agentic/state/SHARED_LESSONS.md`, and your own `agentic/state/<role>.md`. You are resuming, not starting: your state file says what
> is already running on your node and how to resume it; check the node and the heartbeat before changing anything, and do not
> restart a healthy run. Your node is `<ip>` only (`ssh -i ~/.ssh/nebius_corpus -o IdentitiesOnly=yes -o BatchMode=yes
> <user>@<ip>`); never touch the other nodes or another track's tree. Keep your state file current, do not wait for answers,
> and report to the coordinator in a few lines with numbers, what you verified and what you did not.

The eval monitor runs on the desktop CPU (MuJoCo), reads the nodes' `status/` and `runs/` folders only, and uses a node's GPU only
when that track's supervisor has paused for it.

## 3. Keep the user's view current
- Dashboard: https://claude.ai/artifact/JaZDeLFXXN28HLMd1Sia3V ; update steps in `agentic/dashboard/README.md`. Local page (in this repo: `agentic/dashboard/preview.html`, reading `agentic/dashboard/data/*.json` and `agentic/dashboard/images/`):
  double-click `scripts\serve_dashboard.bat` (or `bash agentic/scripts/serve_dashboard.sh`) and open `http://localhost:8765/preview.html`.
- After every agent report: update `agentic/state/STATUS.md`, the dashboard data, and `agentic/docs/USER_RULES.md` if the user decided something.
- Before any result goes to the user: the checks under "What is shown to the user" in `agentic/docs/USER_RULES.md`.

## 4. Scoring a checkpoint
`docs/EVAL.md` is the definition; the harness is in `eval/` (`agentic/state/eval.md` says what it can do and which adapters exist). A
number is a result only when the eval monitor produced it with the common eval, with its trial count.
