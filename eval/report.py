"""The table row of one eval run, as text: python eval/report.py RUN_DIR [RUN_DIR ...]"""
import json
import pathlib
import sys

for run in sys.argv[1:]:
    s = json.load(open(pathlib.Path(run) / "summary.json"))
    n = s["trials"]
    pc = lambda k: "%.1f %%" % (100.0 * k / max(n, 1))  # noqa: E731
    print("== %s" % (s.get("label") or s["adapter"]))
    print("   plant (the gains the policy was trained with): %s" % s.get("plant", "not recorded in this run"))
    print("   %s | apparatus %s (%s) | protocol %s | seeds %d-%d | %s" % (s["checkpoint"], s["apparatus"], s["route"], s["protocol"], s["seeds"][0], s["seeds"][1], s["simulator"]))
    if s.get("strict_successes") is not None:
        print("   SUCCESS, strict stand (main number)  %d / %d = %s   (95 %% interval %.1f - %.1f %%); clean %d" % (
            s["strict_successes"], n, pc(s["strict_successes"]), 100 * s["strict_ci95"][0], 100 * s["strict_ci95"][1], s.get("strict_clean_successes") or 0))
        print("      rule: %s" % s["strict_rule"])
        if s.get("strict_fail_reasons"):
            print("      disc-only successes that fail the strict rule: %s" % ", ".join("%s %d" % kv for kv in sorted(s["strict_fail_reasons"].items(), key=lambda kv: -kv[1])))
    else:
        print("   strict stand: NOT SCORED (run made before the strict rule; disc-only below)")
    if s.get("stand_rule") == "strict":
        print("   (strict-rule run: stations advanced only on a strict stand; the disc-only number comes from a separate STAND_RULE=disc run)")
    else:
      print("   success, disc only (old rule; in this run stations advanced on 1 s in the disc, so the strict number above is a lower bound)  %d / %d = %s   (95 %% interval %.1f - %.1f %%)" % (s["successes"], n, pc(s["successes"]), 100 * s["ci95"][0], 100 * s["ci95"][1]))
    if s.get("stand_rule") != "strict":
      print("   clean success, disc only  %d / %d = %s   (95 %% interval %.1f - %.1f %%)" % (s["clean_successes"], n, pc(s["clean_successes"]), 100 * s["clean_ci95"][0], 100 * s["clean_ci95"][1]))
    print("   mean stumble penalty %.2f per trial; side hits %.2f per trial; mean waypoints %.1f of %s" % (
        s["mean_stumble_penalty"], s["side_hits"]["per_trial"], s["mean_waypoints"], json.load(open(pathlib.Path(run) / "results.json"))[0]["of"]))
    if s.get("time_success"):
        t = s["time_success"]
        print("   time of the successes: fastest %.0f s, median %.0f s, slowest %.0f s" % (t["min"], t["median"], t["max"]))
    print("   section        entered  completed   failures ending there   side hits   stumble penalty per trial")
    for k, v in s["sections"].items():
        h = s["side_hits"]["by_section"][k]
        print("   %-12s %8s %10s %12d %19d %17.2f" % (k, pc(v["entered"]), pc(v["completed"]), s["failures_by_section"].get(k, 0), h["side_hits"], h["mean_penalty_per_trial"]))
    print("   outcomes: %s" % ", ".join("%s %d" % kv for kv in sorted(s["outcomes"].items(), key=lambda kv: -kv[1])))
    print("   both 4x4s: trials whose pelvis path crossed each: %s" % s.get("beams_crossed"))
    bc = s["side_hits"]["by_class"]
    print("   side hits by class: " + "; ".join("%s: %d foot (%d in swing), %d shin/knee, %d scuffs, impulse %.0f N s" % (
        c, v["foot"], v["foot_in_swing"], v["shin_knee"], v["scuffs"], v["impulse"]) for c, v in bc.items()))
    c = s["contacts"]
    print("   landings %d, on an edge %d (%.1f %%); other contact events: shin %d, knee %d, thigh %d, arm %d, trunk %d, head %d; feet checked: %s" % (
        c["steps"], c["edge_landings"], 100 * c["edge_landing_rate"], c["shin"], c["knee"], c["thigh"], c["arm"], c["trunk"], c["head"], c["feet_checked"]))
    if s.get("station_travel_1s"):
        print("   travel after each station (pelvis path in the 1 s after it counted): " + "; ".join("%s: reached %d, median %.2f m, max %.2f m, %d over 0.2 m" % (
            k, v["reached"], v["median_m"], v["max_m"], v["over_0_2m"]) for k, v in s["station_travel_1s"].items()))
    print("   renders picked (seed): %s" % s.get("renders"))
