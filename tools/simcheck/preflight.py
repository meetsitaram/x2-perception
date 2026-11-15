"""simcheck preflight: run the simulator checks on a track's own terrain build, on the CPU (MuJoCo Warp, CPU device).

    PYTHONPATH=/mnt/filesystem-g4/tools <track venv python> -m simcheck.preflight --track acp|ssr [--hf-fix 0|1] [--cfg ...]
        [--out report.json] [--quick]

Exit code 0 = all checks passed, 1 = a check failed, 2 = the build or a check crashed. The JSON report has one entry per
check with its numbers and limits (simcheck.core). Must pass before every launch or restart (agentic/docs/TRACK_RULES.md,
"Simulator checks"). Use run_preflight.sh, which picks the track's venv, pins 4 CPU cores and hides the GPU."""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
import traceback


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True, choices=("acp", "ssr", "hiking"))
    ap.add_argument("--cfg", default=None, help="ACP: data cfg (default cfg/stage2g_sets.json)")
    ap.add_argument("--hf-fix", default=None, help="ACP: ACP_HF_FIX for the build (default: environment)")
    ap.add_argument("--hf-chunk", default=None, help="SSR: X2_HF_CHUNK for the build (default: environment / 400)")
    ap.add_argument("--ssr", default=None, help="SSR: JSON of train flags that shape the terrain (default: run r7 v14)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--quick", action="store_true", help="2 drop sites per kind, 2 worlds per tile, 1 s")
    ap.add_argument("--skip", default="", help="comma list of checks to skip: hfield_size,foot_drop,rollouts")
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        import torch
        torch.set_num_threads(int(os.environ.get("SIMCHECK_THREADS", "4")))
    except ImportError:
        pass
    from simcheck import core
    t0 = time.time()
    rep = dict(track=a.track, time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), checks=[], passed=False)
    log = lambda s: print(s, file=sys.stderr, flush=True)
    code = 0
    try:
        b = importlib.import_module(f"simcheck.adapters.{a.track}").build(
            dict(cfg=a.cfg, hf_fix=a.hf_fix, hf_chunk=a.hf_chunk, ssr=json.loads(a.ssr) if a.ssr else None))
        m = b["model"]
        rep["config"] = b.get("config", {})
        rep["model"] = dict(nbody=m.nbody, ngeom=m.ngeom, nhfield=m.nhfield, nmocap=m.nmocap, timestep=m.opt.timestep,
                            tiles=[t["name"] for t in b["tiles"]], build_s=round(time.time() - t0, 1))
        log(f"[build] {a.track}: {rep['model']}")
        skip = set(filter(None, a.skip.split(",")))
        if "hfield_size" not in skip:
            rep["checks"].append(core.hfield_size(m))
            log(f"[hfield] {rep['checks'][-1]}")
        if "foot_drop" not in skip:
            if b.get("raster") is None:
                rep["checks"].append(dict(check="foot_drop", passed=True, note="no height raster in this build: nothing to drop on"))
            else:
                rep["checks"].append(core.foot_drop(m, b["foot"], b["raster"], n_per_kind=2 if a.quick else 3, log=log))
            log(f"[drop] passed={rep['checks'][-1]['passed']} {rep['checks'][-1].get('summary')}")
        if "rollouts" not in skip:
            rep["checks"].append(core.rollouts(m, b["tiles"], b["q_default"], k=2 if a.quick else 4,
                                               seconds=1.0 if a.quick else 2.0, diverge_qvel=b.get("diverge_qvel", 200.0),
                                               nconmax=b.get("nconmax", 64), njmax=b.get("njmax", 450), log=log))
            r = rep["checks"][-1]
            log(f"[roll] passed={r['passed']} depth {r['depth_terrain_mm_max']}/{r['depth_self_mm_max']} mm jump {r['speed_jump_ms_max']} "
                f"diverged {r['diverged_share']}")
        rep["passed"] = all(c["passed"] for c in rep["checks"])
        code = 0 if rep["passed"] else 1
    except SystemExit as e:
        rep["error"] = str(e)
        code = 2
    except Exception:
        rep["error"] = traceback.format_exc()[-3000:]
        code = 2
    rep["seconds"] = round(time.time() - t0, 1)
    s = json.dumps(rep, indent=1, default=float)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        open(a.out, "w").write(s)
    print(s if not a.out else json.dumps(dict(passed=rep["passed"], out=a.out, seconds=rep["seconds"],
                                               checks={c["check"]: c["passed"] for c in rep["checks"]}, error=rep.get("error"))))
    return code


if __name__ == "__main__":
    sys.exit(main())
