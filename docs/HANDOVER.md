# Handover: where to start in a new session

Last written 2026-11-12 04:55 UTC by the coordinator. Overwrite this file at every handover; the history is in `state/STATUS.md`.

## First 10 minutes
1. Open Claude Code in `C:\Users\PC\projects\perception-agents` (it reads `CLAUDE.md`). Read `docs/USER_RULES.md`, this file,
   `state/STATUS.md` (last entries), `state/SHARED_LESSONS.md`.
2. Check the nodes: `bash scripts/node_check.sh` and the heartbeats (command in `docs/RUNBOOK.md`, section 1).
3. Start the agents again (they do not survive a session): four background agents, `model: opus`, with the prompt in
   `docs/RUNBOOK.md`, section 2, each told to read its own `state/<track>.md` first and "resume, do not restart a healthy run".
   Give each the open items for its track from the table below.
4. Re-arm the coordinator duty: dashboard refresh every 30-60 minutes (`dashboard/README.md`) and
   `bash scripts/sync_track_code.sh --commit` (track code into `tracks/` of this repo).

## State of each track (04:50 UTC; training keeps running without the agents)

| Track | Running | Latest numbers | Next decision |
|---|---|---|---|
| ACP | Stage 2, run `s2_v1`, data set 2c (re-fitted descents, 10-15 cm stairs), iteration about 3,500 of 6,500, plant `mc_20261024`, 52 GB | Paper eval 92.7% at 2,750 (reference contacts). Ladder step 1 (2,250 pair): flat 50/50, logs/stairs down/corridor pass level 3, 10 cm stairs failed. Our apparatus: 0/200 (stalls 0.2-0.4 m short of waypoints: goal convention) | Live paper eval auto at 4,000/5,000/6,000. Stage 2d (arm fix) at the next needed restart. Stage 3 (goal tiles, 0.2 m arrival, sharp turns) armed: auto switch from 5,500 (plateau) or at 6,500 (`code/ops/to_stage3.sh`). Extended-map tiles in stage 3 v2. |
| SSR | Run `r6b` (continues the staged test t_a's gait; flat first, obstacle tiles ramped in), iteration about 1,570 of 20,000; since 1,000: style by schedule, positive-only foot air-time reward, arms hidden from the discriminator | 59-64% of commanded speed, terrain level about 4, step rate falling (7.3 -> 6.2/s), air time rising from 0.06 s | Gate: air time over 0.2 s by 2,000; report if speed under 30%. Next fresh run: default pose from the reference mean, pelvis/hip-roll collision pair excluded. |
| Hiking | Run 4 `x2_parkour_v4` (all seven port-audit fixes: legs-only discriminator, default pose from the reference, G1 action ranges, collision filtering, soft limits 1.0, implicit PD; plus staged rewards a,c,d,s, fall penalty 8, 50% reference starts), iteration about 550 | At 500: speed 26% of command, falls 60%, episodes 370-440 steps, air time about 0.03 s per step. Checkpoint 250 steps upright on flat in MuJoCo (short steps) | Line at 750 (drift check: stalled resets over 40% with speed under 0.2 = report). At 1,000: if air time per touchdown under 0.05 s, continue from `model_1000` with option t (positive-only touchdown air-time reward, threshold 0.05 s, weight sized in `state/hiking.md`) and the measured monitor (`staged.py.next`, `monitor.py.next`). |
| Eval | Desktop CPU (MuJoCo). Last known job: ACP 2,250 pair on the eight step-2 routes | Baseline (2 Nov pair) on our apparatus 15/200 (shrinking goal) and 4/200 (original goal); step 2 routes in `state/eval.md` | Re-run ACP 2,250 on our apparatus with goal kept to 0.20 m and a lookahead past turning waypoints; fix the street_doorway course (route check must see all physics geometry); renders for review. Check whether a desktop python eval process is still alive before starting anything. |

Local G1 run (`C:\Users\PC\projects\hiking-in-the-wild`) stays paused at `model_20000` until the user resumes it.

## Open questions for the user
- Eval gate details (per-skill profile, level passes at 45/50) not yet confirmed.
- Whether the stair and chain clip sets from `loco-manip-apparatus` are BONES-SEED-derived (decides the data rule for them).
- Whether the user's own Pico captures may go into a repo (separate private one suggested; `terrain-park` is public).
- Usage: on 2026-11-12 the weekly all-models allowance was 76% at 04:30 UTC (resets Nov 12, 3:59 pm Pacific). Suggested savings
  (not yet confirmed): hourly dashboard refresh, agents report only at gates, routine watching on Sonnet.

## Rules most easily forgotten (all in `docs/USER_RULES.md`)
No addresses or logins in the repo; no BONES-SEED or Hiking-release data in the repo (motion manifests instead); no training stop
only for evaluation (live eval at most once per 1,000 iterations); never delete under `/mnt/filesystem-g4` (move to `_trash`);
renders from the simulator, MuJoCo labelled "not the training simulator", looked at before sending; secret-free script copies to
the nodes need no approval; camera = chest RealSense (`docs/camera/`), gains = `mc_20261024`.
