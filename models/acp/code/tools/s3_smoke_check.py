"""Check of a 3-iteration stage-3 smoke (ACP agent, 2026-11-13, state/acp.md 17k). usage: s3_smoke_check.py METRICS.jsonl
PASS needs: 3 rows, stage 3, eta_obs = eta_rew = 1 (carried, no jump), V_fs warm-up on, goal metrics present and finite,
planner co-training (10 updates per iteration) on the TileReplay (replay stats logged), finite losses."""
import json, math, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
r = rows[-1]
fails = []
if len(rows) != 3: fails.append(f"rows {len(rows)} != 3")
for x in rows:
    if x.get("stage") != 3: fails.append(f"it {x['it']} stage {x.get('stage')}")
    if x.get("stage2/eta_obs") != 1.0 or x.get("stage2/eta_rew") != 1.0: fails.append(f"it {x['it']} eta {x.get('stage2/eta_obs')} / {x.get('stage2/eta_rew')}")
    if x.get("stage3/warmup") != 1: fails.append(f"it {x['it']} warm-up {x.get('stage3/warmup')}")
    if x.get("planner/updates") != 10: fails.append(f"it {x['it']} planner updates {x.get('planner/updates')}")
    for k in ("ppo/value_loss", "ppo/value_loss_goal", "planner/loss_affordance", "reward_per_step"):
        v = x.get(k)
        if v is None or not math.isfinite(v): fails.append(f"it {x['it']} {k} {v}")
for k in ("planner/replay_added", "goal/success", "goal/level_mean", "weight/goal/stumbling", "weight/goal/slippage"):
    if k not in r: fails.append(f"missing {k}")
print("eta", r.get("stage2/eta_obs"), r.get("stage2/eta_rew"), "goal/episodes", r.get("goal/episodes"), "goal/level_mean", r.get("goal/level_mean"),
      "replay", {k: r.get("planner/replay_" + k) for k in ("added", "dropped", "evicted")}, "replay_size", r.get("planner/replay_size"),
      "raw stumbling", r.get("raw/goal/stumbling"), "raw slippage", r.get("raw/goal/slippage"), "value_loss_goal", r.get("ppo/value_loss_goal"))
print("S3 SMOKE", "PASS" if not fails else "FAIL " + "; ".join(fails))
sys.exit(1 if fails else 0)
