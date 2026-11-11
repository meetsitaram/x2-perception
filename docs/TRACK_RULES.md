# Rules for every track agent

- Work only in `/mnt/filesystem-g4/tracks/<your track>/` on your own node. Never edit, import from or run code in another track's tree. To
  use something from another track, ask the coordinator; it arrives as a copy in `/mnt/filesystem-g4/exchange/`.
- Shared inputs in `/mnt/filesystem-g4/data` are read-only. If you need a motion or asset from the desktop, ask the coordinator to copy it
  there.
- One GPU job at a time on your node, run through a supervisor with stall detection and auto-resume from the newest checkpoint,
  launched so it survives your session ending (tmux or systemd), with a heartbeat in `/mnt/filesystem-g4/status/<track>.json`. Every shell
  command has a timeout.
- You do not score yourself. Save checkpoints to `runs/`; the eval monitor runs `docs/EVAL.md` on them. Your own quick checks are
  fine but are not results.
- Keep `state/<track>.md` in the perception-agents repo current: what is running, the newest checkpoint, what you changed and
  why, blockers. A new agent must be able to continue from that file alone.
- Do not wait for answers: take your recommended option, write the decision and reason in your state file, keep going. Report to
  the coordinator in a few lines: what changed, numbers, what is next, what you did not verify.
- X2 only. No purchases. No commits or pushes unless the coordinator says the user asked. Bones Studio credit on anything shown;
  no dataset clip names in renders.
- Installs on the nodes: accept EULAs and install-time approval prompts without asking (user, 2026-11-11: "any EULA or other approvals - go ahead and say YES for those when installing"). Not purchases or paid subscriptions.
- Use the node's GPU well (user, 2026-11-11): one training job, sized by a benchmark of larger robot counts; never a second job on the node.
- The user's rules and decisions are in `docs/USER_RULES.md`; read it before this file. It wins over a brief.
- Never write node IP addresses, the login name, cloud project or instance ids into any file of this repo (state files, docs, scripts, dashboard data): the repo is on GitHub. Use the node names (acp, ssr, hiking); the addresses are in the git-ignored `config/nodes.local.json`. (user, 2026-11-11: "why would you include ip addresses in a repo. don't do that")
