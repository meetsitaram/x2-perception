# Review later

Things the user has decided to leave as they are for now and revisit later (user, 2026-11-12: "add this to review-later list of
things to revisit sometime later. for now, let it go as is"). Kept by the coordinator. Each entry: what, why it is parked, what
would bring it back, and where the detail is. Nothing here is being worked on.

| Added | Item | Left as is because | Revisit when | Detail |
|---|---|---|---|---|
| 2026-11-12 | **SSR action scale against the paper.** Ours is 0.485 rad per action unit on ankle roll, shoulder yaw and elbow (plant rule 0.25 x effort / stiffness); the paper uses 0.25 legs, 0.2 arms. Other joints match (0.26). A test with the paper's values is ready behind `--action_scale_paper`. | SSR's run walks, passes the flat gate and survives 100% with our scale; its measured blockers are elsewhere. The test cannot continue from a checkpoint and would pause SSR's only GPU for about an hour. | SSR's sideways balance or arm motion looks jittery on the apparatus or on the real robot. | `docs/FINDINGS.md` A3; `state/ssr_port_audit.md` headline 2; `state/ssr.md` |
