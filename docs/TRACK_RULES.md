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
- Back up to the shared filesystem (user, 2026-11-11: "periodically save checkpoints and other motion data files and any intermediate files to the shared filesystem as backup"): checkpoints, training logs, generated motion and label files, terrain files, configs and code changes go to `/mnt/filesystem-g4/tracks/<track>/` at least every 30 minutes (rsync from the node's local disk if the run writes there), and the supervisor does it automatically, not by hand. Nothing a restart would need may live only on a node's local disk. Keep the newest two checkpoints plus one every 1,000 iterations; record in the state file where each kind of file is and how to restore it.
- **Paper's own evaluation** (user, 2026-11-11): besides the common apparatus eval, each track reports its paper's own success criteria and metrics, computed as the paper defines them (terrains, trial counts, success definition, tables), with the paper's figures beside ours, and says what could not be reproduced and why.
- **Transparent logs** (user, 2026-11-11): every training run logs, per iteration, every reward and penalty term separately (raw value and weighted contribution), every termination cause with its share, curriculum levels, command statistics, and every metric the paper reports or plots (tracking errors, success rates, gait statistics), plus throughput and GPU use. Logs go to the shared filesystem in a plain format (CSV or JSON lines) next to the checkpoints, with a one-page legend in the state file: each logged name, what it measures, the paper's symbol or table row, and its weight. A term that the paper has and the run does not log is a gap to state.
- **Knowledge sharing** (user, 2026-11-11): read `state/SHARED_LESSONS.md` when you start or resume and before a long run; append an entry for every problem you hit that another track could meet, with what fixed it. Ask the coordinator to put a question to another track when their entry is close to your problem; answers come back through the same file or the coordinator.

## Nothing is deleted on the shared filesystem (coordinator, 2026-11-12)
After a track removed two of its own intermediate clip files with `rm` and the session's safety check flagged it: agents do not
delete files under `/mnt/filesystem-g4`. A file that is no longer wanted is moved to `tracks/<track>/_trash/<date>/` with a
one-line note of what it was; clearing a trash folder is the user's. Checkpoint retention (newest two plus one per 1,000
iterations) is also done by moving, not deleting, until the user says otherwise. Scratch under `/tmp` on a node is the agent's.
