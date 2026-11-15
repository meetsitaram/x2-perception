import json, sys, os
sys.path.insert(0, os.path.dirname(sys.argv[0])); from collapse_check import check
rows = []
for l in open('/mnt/filesystem-g4/tracks/hiking/runs/x2_parkour_v4/metrics.jsonl'):
    try: d = json.loads(l); rows.append((int(d["iteration"]), float(d["length"]), float(d["reward"])))
    except Exception: pass
print('rows', len(rows), 'avg line bytes', os.path.getsize('/mnt/filesystem-g4/tracks/hiking/runs/x2_parkour_v4/metrics.jsonl') // max(1, len(rows)))
fires = []
for e in range(400, len(rows) + 1):
    c, m = check(rows[max(0, e - 3000):e])
    if c == 2: fires.append((rows[e - 1][0], m))
print('fires', len(fires))
seen = set()
for it, m in fires:
    k = it // 200
    if k in seen: continue
    seen.add(k); print(it, m)
