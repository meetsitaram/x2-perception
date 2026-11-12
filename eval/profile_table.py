"""Success per skill and level for every checkpoint and track scored (docs/EVAL.md, gates item 5).

    python eval/profile_table.py        -> eval/out/profile_table.csv and .txt

Reads every eval/out/**/ladder/step1.json that has a skill profile. Ladder step 1 is on held-out skill courses and the
training map: never an apparatus result.
"""
import csv
import json
import pathlib

OUT = pathlib.Path(__file__).resolve().parent / "out"
rows = []
for f in sorted(OUT.glob("**/ladder/step1.json")):
    j = json.load(open(f))
    if "profile" not in j:
        continue
    by = {r["test"]: r for r in j["rows"]}
    run = str(f.parent.parent.relative_to(OUT)).replace("\\", "/")
    for sk, p in j["profile"].items():
        for lv in (1, 2, 3):
            r = by.get("%s_L%d" % (sk, lv))
            rows.append(dict(run=run, plant=j.get("plant"), protocol=j.get("protocol"), simulator="MuJoCo CPU", skill=sk, level=lv,
                             trials=r["trials"] if r else "", success=r["success"] if r else "", clean=r["clean"] if r else "",
                             falls=sum(v for k, v in r["outcomes"].items() if k.startswith("fall")) if r else "",
                             stalled=r["outcomes"].get("stalled", 0) if r else "", time_limit=r["outcomes"].get("time limit", 0) if r else "",
                             highest_passed=p["highest_level_passed"], flat_gate=j.get("flat_gate_passed"), weakest=j.get("weakest_skill")))
if rows:
    with open(OUT / "profile_table.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lines = ["Ladder step 1 skill profile, every run scored (held-out skill courses; NOT an apparatus result). MuJoCo CPU, 50 trials per level.",
             "%-58s %-16s %-12s %-6s %-9s %-7s %-6s %-6s %s" % ("run", "plant", "skill", "level", "success", "clean", "falls", "stall", "time limit")]
    for r in rows:
        lines.append("%-58s %-16s %-12s %-6s %-9s %-7s %-6s %-6s %s" % (r["run"], r["plant"], r["skill"], r["level"],
                     "%s / %s" % (r["success"], r["trials"]) if r["trials"] != "" else "not run", r["clean"], r["falls"], r["stalled"], r["time_limit"]))
    open(OUT / "profile_table.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
