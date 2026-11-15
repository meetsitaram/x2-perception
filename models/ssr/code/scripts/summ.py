import json, sys
name = sys.argv[1]; every = int(sys.argv[2]) if len(sys.argv) > 2 else 100
rows = [json.loads(l) for l in open(f"/mnt/filesystem-g4/tracks/ssr/runs/{name}/log.jsonl") if l.strip()]
pick = [r for r in rows if r["it"] % every == 0] + rows[-1:]
for r in pick:
    g, t = r.get("gait", {}), r["terms"]
    print(f"it {r[it]:5d} len {r[ep_len_s]:5.2f}s r_l {r[r_l]:5.2f} r_s {r[r_s]:4.2f} lvl {r.get(lvl_mean,0):4.2f} vx {g.get(vx_ratio,0):5.2f} sgl {g.get(single,0):4.2f} dbl {g.get(double,0):4.2f} td/s {g.get(steps_per_s,0):4.1f} linv {t[lin_vel]:4.2f} angv {t[ang_vel]:4.2f} std {r[std]:4.2f} lr {r[lr]:.1e} d_f {r[d_f]:5.2f} sps {r.get(samples_per_s,0)} util {r.get(gpu_util,)}")
