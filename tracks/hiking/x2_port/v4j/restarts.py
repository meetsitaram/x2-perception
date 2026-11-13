"""For every restart in train.log (iteration counter drops or log starts): mean over its first 32 iterations of key metrics."""
import re
LOG = "/mnt/filesystem-g4/tracks/hiking/runs/x2_parkour_v4/_supervisor/train.log"
K = {"Mean episode length": "ep", "Curriculum/level_keeper/turn/rate_along_cmd": "turn", "Curriculum/walk_monitor/actual_speed": "vx",
     "Curriculum/walk_monitor/cmd_speed": "cmd", "Curriculum/level_keeper/turn/planar_speed": "tplanar",
     "Curriculum/level_keeper/turn/cmd_abs": "tcmd", "Curriculum/level_keeper/end/time_out_paper": "tout",
     "Episode_Reward/rewards_is_alive/sum": "alive"}
it_pat = re.compile(rb"Learning iteration (\d+)/")
runs, cur, last = [], None, None
with open(LOG, "rb") as f:
    off = 0
    for line in f:
        m = it_pat.search(line)
        if m:
            it = int(m.group(1))
            if last is None or it < last or it > last + 1:
                cur = {"start_it": it, "off": off, "its": []}; runs.append(cur)
            last = it
            cur["its"].append({})
        elif cur is not None and cur["its"]:
            s = line.decode("utf8", "replace").strip()
            if s.startswith("Mean episode "):
                s2 = s[len("Mean episode "):] if not s.startswith("Mean episode length") else s
            else:
                s2 = s
            k, _, v = s2.rpartition(":")
            k = k.strip()
            if k in K:
                try: cur["its"][-1][K[k]] = float(v)
                except ValueError: pass
        off += len(line)
for r in runs[-12:]:
    its = r["its"][:32]
    def mean(k):
        v = [d[k] for d in its if k in d]; return sum(v) / len(v) if v else float("nan")
    def tailm(k):
        v = [d[k] for d in r["its"][-32:] if k in d]; return sum(v) / len(v) if v else float("nan")
    print("start %5d off %10d n %5d | first32: ep %6.1f turn %5.2f tcmd %4.2f tplan %4.2f vx %5.2f cmd %4.2f tout %4.2f | last32: ep %6.1f turn %5.2f vx %5.2f"
          % (r["start_it"], r["off"], len(r["its"]), mean("ep"), mean("turn"), mean("tcmd"), mean("tplanar"), mean("vx"), mean("cmd"), mean("tout"),
             tailm("ep"), tailm("turn"), tailm("vx")))
