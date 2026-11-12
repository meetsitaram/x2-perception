import json, sys
rows=[json.loads(l.replace("NaN","null")) for l in open(f"/mnt/filesystem-g4/tracks/ssr/runs/{sys.argv[1]}/log.jsonl")]
n=int(sys.argv[2]) if len(sys.argv)>2 else 3
st=int(sys.argv[3]) if len(sys.argv)>3 else 1
for r in rows[::st][-n:]:
    g=r["gait"]; d=r["done_share"]
    print(r["it"], "vx", g["vx_ratio"], "sgl", g["single"], "dbl", g["double"], "td/s", round(g["steps_per_s"],1), "r_l", r["r_l"], "r_s", r["r_s"], "len", r["ep_len_s"], "lvl", r["lvl_mean"], "std", r["std"], "lr", "%.0e" % r["lr"], "tilt", d["tilt"], "end", d["reached_end"], "tl", d["time_limit"])
