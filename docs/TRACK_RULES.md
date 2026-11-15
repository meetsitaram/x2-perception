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

## Simulator checks (user goal 2026-11-12: "simulator faults must be caught by code, not discovered by agents")
After two faults punished policies for hours (FINDINGS B8 feet trapped in height-field cells, B10 MuJoCo Warp on one large
height field). Package: `/mnt/filesystem-g4/tools/simcheck/` (README there; mirror `tools/simcheck/`).
1. **Preflight before every launch or restart.** `tools/simcheck/run_preflight.sh <track> <tag> [--hf-fix 1 --cfg ...]` with the
   exact terrain build and flags of the launch; it must exit 0 (report in `tools/simcheck/reports/<track>_<tag>.json`; CPU,
   4 cores, about 1-3 minutes). Checks: no height field over 400 x 400 vertices; foot drop at 1-4.5 m/s on flat, slope and edge
   cells (no contact deeper than 1 cm for over 0.07 s, foot freed by a 150 N pull); random-action rollouts on every tile type
   (no contact deeper than 5 cm, no body speed jump over 20 m/s, no NaN, diverged share at most 1 %). A failing launch needs
   the coordinator's written exception. Hiking (Isaac, GPU only) runs its preflight during a training pause.
2. **Simulator faults are truncations, never terminations.** NaN, the track's |qvel| bound, a contact deeper than 5 cm or a
   body speed jump over 20 m/s ends the episode time-out style: value bootstrapped, step reward 0, no termination penalty, no
   curriculum move (`simcheck.episode_end`; each track's patch in `tools/simcheck/patches/`).
3. **`sim_fault_share` in every heartbeat** (`simcheck.alarm`): share of episode ends that were simulator faults; over 1 %
   the heartbeat carries `sim_fault_warn` and the track stops trusting new checkpoints until the cause is found.

## One code folder (user, 2026-11-13)
One code folder per track; stages and variants by flags; per-run `code_snapshot` protects running jobs. No copied code trees. Where an installed copy must exist (Hiking's InstinctLab files), keep one source and an install step that verifies the installed copy is identical before every launch (part of the pre-flight).

- **Reward-hacking check for every added term (lesson of 2026-11-14):** Hiking's touchdown/air-time term (ours, weight 140) paid more for stepping in place at a hard obstacle than for climbing a mid one, and SSR's stricter curriculum and fatal wall touch kept robots off high stairs; two days of training refused obstacles. Before any reward term, weight, command share or curriculum rule that is not in the paper goes into a run, compute on CPU the per-term reward per second for (a) standing or stepping in place at an obstacle foot under a walking command and (b) attempting/climbing it; the change must not make (a) pay more than (b). Record the table in the ledger row.
- X2 waist pitch (FINDINGS X1): small range (about +-18 deg) and 24 N*m limit; every recipe checks its saturation share in pre-flight and treats it as a constrained joint (user, 2026-11-15).
- Every policy clips its actions (lesson of the Hiking action runaway, FINDINGS H7d, 2026-11-15): ACP and SuperMimic +-20, SSR +-8, Hiking +-20 (waist_pitch +-50). Unclipped actions fed back as observations can run away and poison a whole update.
