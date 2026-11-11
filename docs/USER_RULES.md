# The user's rules and decisions for this team

Everything the user has decided, in one place, with the date and their words where they gave them. Every agent reads this before
anything else. When the user decides something new, the coordinator adds it here the same day (and to `docs/EVAL.md` or
`docs/TRACK_RULES.md` if it changes them). If a rule here conflicts with an agent brief, this file wins.

## Goal and scope
- Train perceptive walking for the **AgiBot X2** only, with three methods side by side: ACP, SSR, Hiking in the Wild; one cloud GPU
  node each; one coordinator; one eval monitor that reports results and renders for the user's visual review. (2026-11-11)
- Target: apparatus-walk success from about 25% to close to 99%. The measured source of "25%" is 13 of 50 (26%) on our apparatus on
  2026-11-02 with the earlier SuperMimic-based ACP policy, under a looser success rule. (2026-11-11)
- Keep the three tracks' code isolated; shared data lives on the common filesystem; motions needed from the desktop are copied there.

## Methods
- "For this round stick to what the paper does ... we don't need SuperMimic on top of that." ACP and SSR follow their papers' own
  RL methods: no SuperMimic trainer, checkpoints, initialisation or SuperMimic-derived labels. Motion data that the paper's own
  method uses (ACP's reference clips for its tracker, SSR's and Hiking's motion priors) stays. (2026-11-11)
- Hiking in the Wild: port to the X2 and train on the paper's terrains and rewards first; the extended terrain map, routes,
  commands and rewards in `hiking-in-the-wild/docs/TODO_extended_training.md` come after, as a separate task.
- Reuse earlier work where it fits the rule above.

## The two apparatuses are held out
- "The final result will be how well they perform on the humanoid challenge apparatus - both ours and the NIST one, but they
  shouldn't directly go into training. You can copy parts of them to go there if needed as part of the terrain map." No whole
  layout, route or clip recorded on a full apparatus route in any training set; re-arranged single parts are allowed and each track
  lists the parts it copied. (2026-11-11)

## The eval (details in `docs/EVAL.md`)
- A station is a 1 s stand in the arrival disc; button presses do not count. ("station is just a 1 s stand - that's good")
- Time limit 180 s per trial. ("sure")
- The route "has to step over both 4x4 logs" on each apparatus; a trial that goes round one fails.
- "Also make sure the door is open": the door leaf is in the physics, swung clear; the route checker fails a narrow doorway.
- The NIST route visits the pocket behind the upper 4x4 and ends at the far end of the landing (the user's two marked-up maps).
- "Add negative score for hitting on the sides of the 4x4 as well as the side of the wedges and stairs which will cause stumbles":
  the eval reports success, clean success (no side hits) and a stumble penalty. **Eval only**: "they don't have to go into
  training - the penalties". The Hiking track's own planned training penalties for its extended run (stumble, foothold, foot clearance) stay in: "INCLUDE THEM" (2026-11-11).

## How agents work
- "Any EULA or other approvals - go ahead and say YES for those when installing." Install-time prompts on the nodes only; not
  purchases or paid subscriptions.
- "Make sure the GPUs are well used": one training job per node, sized to use the card (benchmark larger robot counts; restart
  from a checkpoint with the count that gives clearly more samples per second), never a second job on the same node.
- Do not wait on the user: take the recommended option, record it with the reason, keep going.
- Long runs go through a supervisor with stall detection, auto-resume and a heartbeat on the shared filesystem; every shell
  command has a timeout.
- Do not commit or push unless the user asks. Nothing is bought; never search Amazon.

## What is shown to the user
- "Please don't present fake renders": robot videos come from the simulator's own camera and are checked against simulator data
  before they are shown; anything else is labelled (MuJoCo frames as "MuJoCo, not the training simulator", terrain pictures as
  previews, kinematic playback as kinematic).
- Nothing in front of the robot: "I want a close up look at the legs"; a legs close-up with walls hidden (still in the physics).
- Targets and waypoints must not be placed inside or against obstacles, and must not look like footholds.
- Bones Studio credit on shown motion results; no dataset clip names on screen. No home photos, street address or house number.
- A count of "no contact" must include the feet. A success count comes with its trial count.
- The dashboard is dark-themed and must stay current after every agent report (`dashboard/README.md`).

## Nodes
- Log in as the login in `config/nodes.local.json` with the desktop key `~/.ssh/nebius_corpus` (`config/nodes.json`); shared filesystem `/mnt/filesystem-g4`.

## Repository hygiene
- "Why would you include ip addresses in a repo. Don't do that." No node addresses, login names, cloud project or instance ids, keys or tokens in the repo; they live in the git-ignored `config/nodes.local.json`. Run `bash scripts/check_no_secrets.sh` before every push. The repo is https://github.com/meetsitaram/perception-agents . (2026-11-11)
- Back up to the shared filesystem (user, 2026-11-11: "periodically save checkpoints and other motion data files and any intermediate files to the shared filesystem as backup"): checkpoints, training logs, generated motion and label files, terrain files, configs and code changes go to `/mnt/filesystem-g4/tracks/<track>/` at least every 30 minutes (rsync from the node's local disk if the run writes there), and the supervisor does it automatically, not by hand. Nothing a restart would need may live only on a node's local disk. Keep the newest two checkpoints plus one every 1,000 iterations; record in the state file where each kind of file is and how to restore it.

## Transparency and sharing (2026-11-11)
- "Ask the agents to also have paper-mentioned evaluation success criteria": each track reports its paper's own evaluation, as the paper defines it, beside the common apparatus eval.
- "Include enough logs to capture all the metrics, rewards, penalties etc included in the papers so we have more transparency on what is going on": every reward and penalty term, termination cause, curriculum level and paper metric is logged per iteration, on the shared filesystem, with a legend. The dashboard shows them.
- "Ask them to talk to each other to share common issues each encountered and addressed": `state/SHARED_LESSONS.md` is the shared notebook; every agent reads it and adds to it, and the coordinator relays questions between tracks.

## The extended training map (2026-11-11)
- The user asked "how are all the 3 training agents using our training map that we created recently?" It is meant for all three tracks, not only Hiking. The Hiking agent exports it to `/mnt/filesystem-g4/data/terrains/extended_v1/` in a simulator-neutral form; each track uses it in a later stage within its paper's method (Hiking: extended task after the paper-terrain run; ACP: stage 3; SSR: extra terrain categories once it walks) and writes its plan in its state file. Coordinator's reading of the question; confirm with the user.

## ACP on the apparatus (2026-11-11)
- The user corrected the claim that ACP's stage-1 tracker cannot be scored on a route: "it has a planner contact point estimator and the tracker or executor .. we already tried acp on our apparatus and that's where we got 25%". ACP is scored as planner plus tracker; do not wait for stage 3. The planner is brought forward, and the eval adapter takes contact targets from a planner.

## Eval ladder (2026-11-11)
- The user: "Eval shouldn't go straight to the apparatus for evals, and instead do simpler tests on our terrain map with some of the routes we reviewed earlier." Checkpoints are scored in steps: (1) simple tests on tiles of the extended terrain map (flat command following incl. turn in place, backward and sideways; one 4x4 log; slanted 4x4; wedges; stairs; a log corridor), (2) some of the routes on the map that the user reviewed (loops with turns, routes through the main walk area), (3) only then the two held-out apparatuses, which stay the final result. Each step is reported per section with trial counts, so it is visible what a policy can and cannot do before it meets the apparatus.

## Patience with long runs (2026-11-11)
- The user, when the X2 Hiking run was judged at iteration 500: "We trained that for almost two days to have decent results" (the G1 run: little air time until after iteration 2,000, decent only towards 20,000). Judge a run against the G1 curve at equal samples, not against a finished policy; check for defects on CPU without stopping it; restart only for a defect shown with numbers.
- Review renders must not stop training (user, 2026-11-11: "Use mujoco and don't interrupt training by running isaacsim"): render checkpoints in MuJoCo on CPU, labelled "MuJoCo, not the training simulator", after checking the MuJoCo observation build against recorded simulator observations. Isaac Sim renders only when training is paused for another reason or the user asks.

## X2 plant and gains (2026-11-11)
- The user, asked which X2 gains this round trains with: "Use the ones used for supermimic training last iteration". That is the plant `mc_20261024` (`gear_sonic/config/robot_plant/x2_ultra_mc_20261024.yaml`; shared filesystem `data/robots/x2/robot_plant/`): vendor datasheet armature, effort and default pose as in `vendor_20260823`, plus the vendor motion controller's measured stiffness and damping (hip 100, knee 150, ankle pitch 40, ankle roll 30, waist 40, ...). It was used by SuperMimic run 2 and run 3 / `dr=sonic2` (gwbc-v6 commits 69a1b0b, ae3617b, 76fb459, 2026-10-23). Action scale stays 0.25 x effort / (armature x w^2), so earlier checkpoints can be continued. All three tracks train with it; no hand-set gains. The later desktop perception and ACP runs (incl. the 2 November baseline) had fallen back to the trainer's default `vendor_20260823`; a policy is always evaluated under the plant it was trained with.
- Gains confirmed by the user the same day: "I recommend yes - that's fine too".
- Camera (user, 2026-11-11): "We already setup one for the hiking training which was able to see the feet properly and place in front"; "We used camera for hiking in wild training". The camera tracks use the Hiking in the Wild training camera: the paper's G1 setup (on the torso, 48 degrees below horizontal, 64 x 36 pinhole at 89.5 x 58.3 degrees, lower-half crop, depth only, 0-2.5 m), mounted on the X2 torso to reproduce the G1 view (same camera height and forward position; shown with numbers). Hiking restarts from scratch as run x2_parkour_v2 with this camera and the mc_20261024 gains; SSR uses the same mount with its own image preprocessing. The X2 head camera (40 degrees down) is only a fallback setting. (The coordinator first misread this as the ssr-x2 75-degree chest mount; corrected the same hour.)
- Camera footprint (user, 2026-11-11, after seeing that the paper's cropped G1 view reaches about 1.1 m): "1m is too low to make any meaningful decisions in time"; "How much like 2M ahead? And 10 cm behind or something?" The requirement is a footprint: on flat ground at the standing pose the depth view spans from about 10 cm behind the toe tips to at least 2 m ahead, and shows the robot's own feet. Mount point as the Hiking in the Wild camera (torso); tilt, crop and depth range are chosen to meet the footprint (coordinator's starting suggestion: about 58 degrees down, full image height, 3 m range), verified by each track with computed footprints and depth frames. This is a user-directed departure from the paper's camera crop.
