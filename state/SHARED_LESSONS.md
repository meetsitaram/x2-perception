# Shared lessons: problems each agent hit and how it solved them

Every agent reads this file when it starts or resumes and before a long run, and appends an entry whenever it hits and solves (or
fails to solve) a problem another track could meet: simulator quirks, the X2 model, motion data, throughput, supervisors, the
shared filesystem, eval. One entry per problem: date, who, the symptom, the cause if known, what fixed it, and what is still open.
Short and specific; no node addresses or logins. If an entry concerns you, say so in your next report. The coordinator relays
urgent ones directly.

| Date | From | Problem | Cause | What fixed it / status |
|---|---|---|---|---|
| 2026-11-11 | ACP | MuJoCo Warp printed "nefc overflow" 102 times with 12,000 robots: constraint rows dropped in some worlds | constraint buffer (`njmax`) of 300 too small for the contacts on clip-aligned props | raised `njmax` to 450; none since. Check your own logs for the same message. |
| 2026-11-11 | ACP | MuJoCo Warp prints "height field collision overflow (>= 50)" on a ramp raster | not confirmed; guess: fallen robots lying on the raster | open; coarsening the raster to 4 cm did not remove it |
| 2026-11-11 | ACP | More robots did not turn free GPU memory into speed: 12,000 -> 36,000 gave +20% in the real run, 48,000 only +3% more | the card's compute is saturated by physics, not its memory | use the paper's total (36,000) and stop there |
| 2026-11-11 | Hiking | A robot-count benchmark kept training stopped for 30+ minutes | each Isaac Sim test restarts the simulator (7-10 min) and four sizes were tested in a row | never stop training more than about 10 minutes for a measurement; write "benchmarking" into the heartbeat |
| 2026-11-11 | Hiking | Isaac Lab on the node: one CPU core at 100-120% while GPU utilisation reads low; 4,096 robots give about 15,400 steps/s, 12,288 only +59% | looks CPU-bound on a single core (to be confirmed) | open |
| 2026-11-11 | Hiking | A folder named `instinctlab` in the working directory shadowed the installed package | Python import order | renamed the folder; do not run Python from a folder that contains a package-named folder |
| 2026-11-11 | Hiking | The shared filesystem takes about 17 ms per small file | network filesystem | keep Python environments on the node's local disk; keep code, data, checkpoints and logs on the shared filesystem |
| 2026-11-11 | SSR | From-scratch training settles into standing still within a few hundred iterations (4% of commanded speed, both feet down 87% of the time) | not yet established; the same failure the earlier X2 runs had | open; being diagnosed against the paper's settings. Other tracks: watch the same signal (commanded vs actual speed, double-support share) early in your own runs. |
| 2026-11-11 | Eval | The X2 MJCF's 5 mm sole spheres gave tilted contact normals and false stubs on a height field | sphere much smaller than the 2 cm raster cells | 2 cm sole spheres in the eval model. Tracks that train on height fields with the stock MJCF should check their foot contacts. |
| 2026-11-11 | Eval | The X2's foot (24 cm) is longer than our apparatus' stair tread (22 cm) | geometry | a foot fully on a tread touches the next riser or overhangs; relevant to every track's stair behaviour |
| 2026-11-11 | Eval | NIST wall boxes built per mesh piece closed the doorway; the door leaf was missing from the physics | MuJoCo's default box per mesh piece; jointed bodies skipped | boxes fitted to vertices; leaf added at its open pose. Use the files in `data/apparatus/`, do not rebuild them. |
| 2026-11-11 | Coordinator | Node check script failed on Windows with "identity file not accessible" | Python printed CRLF line ends into a shell loop | strip `\r` (`tr -d '\r'`) from anything Python prints for bash on the desktop |
