# Token efficiency

Read this at the start of every session, coordinator and agents alike (user, 2026-11-12: "let's have an efficiency token usage
that agents learn on start"). It is short on purpose.

## Why

On 2026-11-11/12 this team used 28% of the week's tokens in about a day. Measured from the transcripts, by share of cost: eval
monitor 40%, coordinator 19%, ACP 14%, SSR 11%, Hiking 7%, others 9%. The cause was the same everywhere: contexts of 500,000 to
1,000,000 tokens, re-read in full on every turn (818 turns for the eval monitor).

**Cost = context size x number of turns.** Every tool call is a turn. Keep both small.

## Rules for every agent

1. **Read narrowly.** Use grep, tail, head or a line range for the lines you need. Never read a whole log, a whole state file
   or a results table "to be safe". Never print file contents you are about to edit.
2. **Keep output out of your context.** Send long command output to a file and print a summary (counts, the last lines, the
   one number). Cap printed output (`| tail -20`, `| cut -c1-200`).
3. **Never wait in a loop of turns.** Start long work as a background script that logs to a file and writes a DONE or FAILED
   marker, then end your turn. Do not poll with sleep, repeated `ls` or repeated status checks.
4. **Batch.** Several independent commands go in one call. Several findings go in one report.
5. **Reports are ten lines or fewer.** The answer first, numbers, what you decided, what you need. Detail goes in your state
   file, with a pointer. Do not repeat what an earlier report said.
6. **Write state so anyone can take over.** Your state file (`agentic/state/<track>.md`, `agentic/state/eval.md`) ends with a "HOW TO RESUME"
   block, current at every report: what is running and where, the last results, the next check and its time, open items. Use
   node-clock times (`date -u`).
7. **Expect to be replaced.** The coordinator starts a fresh agent from your brief and state file once your context passes
   about 200,000 tokens. That is normal, not a failure; a good state file makes it free.
8. **Do the cheap check first.** A CPU replay, a grep through a log or a ten-trial run before a 200-trial run or a restart.
   Do not re-derive what a state file or `docs/FINDINGS.md` already records.
9. **One job, one owner.** Before starting a queue, list your own running processes; do not start work another agent has
   running (it happened: finished evals were re-run by a forgotten queue).

## Extra rules for the coordinator

Measured again on 2026-11-12 for the two hours around the compaction (2% of the week, the same rate as before): the old eval
monitor took 37% (214 turns at about 480,000 tokens each, because the coordinator resumed it three more times and it woke itself
to poll its queues), the coordinator 19% (163 turns), last turns of the other old agents 15%, new work the user asked for 15%,
fresh agents the rest. Lessons, now rules:

- **Never send a message to an agent over about 150,000 tokens, not even "one small thing".** Start a fresh one. Every message
  to a large agent costs its whole context several times over.
- An agent that reports "waiting on background work" must not wake itself to poll; its queue scripts write DONE markers and the
  coordinator reads the results at the next refresh.
- Refresh hourly, not every 30 minutes (recurring job; one gather script, one refresh script, about six tool calls).
- Model choice (user, 2026-11-12): helper agents for routine work (searches, data and route builds, plots, measurements) are
  started with `model: opus` or `sonnet`; Fable only where the problem is hard. Track and eval agents run on Opus.
- User requests that need more than about eight tool calls go to a fresh agent with a tight brief; the coordinator keeps the
  conclusion, not the work.

- Do not resume an agent over about 150,000 tokens; start a fresh one of the same type with a brief of what changed and a
  pointer to its state file.
- Relay results to a track once per refresh in one message, unless a result changes what is running now.
- Answer "agent is waiting" notifications in one line and do nothing else.
- Refresh scripts go in files (the Write tool), not long shell heredocs; reuse the last one's pattern.
- When this session passes about 300,000 tokens, bring `agentic/state/STATUS.md`, `docs/FINDINGS.md`, `agentic/docs/USER_TODO.md` and
  `agentic/docs/HANDOVER.md` up to date and ask the user to run `/compact`.
- Replies to the user are short when they are on Remote Control; tables only when they carry numbers.

## How to measure

Transcripts are JSONL files under `~/.claude/projects/<project>/<session>.jsonl` and `<session>/subagents/agent-<id>.jsonl`.
Sum `message.usage` over assistant messages (one per message id): `cache_read_input_tokens` is the context re-read each turn.
A cost-weighted total uses input 1, cache write 1.25, cache read 0.1, output 5. The coordinator measures this when the user
asks where tokens went, and after any day of heavy use.
