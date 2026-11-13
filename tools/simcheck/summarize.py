"""Print one compact block per preflight report: python summarize.py reports/*.json"""
import json
import sys

for p in sys.argv[1:]:
    r = json.load(open(p))
    print(f"== {p.split('/')[-1]}: passed={r['passed']} {r.get('seconds')} s  {r.get('error', '') and r['error'][-300:]}")
    for c in r.get("checks", []):
        k = c["check"]
        if k == "hfield_size":
            print(f"  hfield: {c['passed']} n={c['n_fields']} max {c['max_nrow']}x{c['max_ncol']} over={c['n_over']}")
        elif k == "foot_drop":
            s = {kk: (v['failed'], v['n'], v['deep_mm_max'], v['t_over_1cm_max'], v['pull_moved_min']) for kk, v in c.get('summary', {}).items()}
            print(f"  drop: {c['passed']} (failed,n,deep_mm,t>1cm,pull_min) {s} missing={c.get('kinds_missing')}")
            for f in c.get("failed_cases", [])[:4]:
                print(f"    fail {f['kind']} v{f['v']} ({f['x']},{f['y']}) deep {f['deep_mm']} t {f['t_over_1cm_s']} pull {f['pull_moved_m']} {f.get('deepest_foot_geom')}")
        elif k == "rollouts":
            print(f"  roll: {c['passed']} worlds {c['worlds']} depth {c['depth_terrain_mm_max']}/{c['depth_self_mm_max']} mm jump {c['speed_jump_ms_max']} "
                  f"div {c['diverged_share']} nan {c['nan_worlds']} overflow {c['contact_overflow']}")
            for t in c.get("failed_tiles", [])[:4]:
                print(f"    fail {t['tile']} depth {t['depth_terrain_mm']}/{t['depth_self_mm']} jump {t['speed_jump_ms']} div {t['diverged_worlds']} nan {t['nan_worlds']} {t.get('deepest_pair_t')}")
