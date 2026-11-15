"""Regression check of the eval (coordinator, 2026-11-12, after the 2 November baseline did not reproduce across code changes):
ten fixed short trials whose outcomes are stored; every harness and ladder run compares them at its start, so a code or data
change that alters results is seen at once.

    python eval/regression.py --record      # (re)store the reference after a deliberate, documented change
    python eval/regression.py               # compare; exit code 1 on a difference

The ten trials: our apparatus, nominal protocol, seeds 1000-1009, 12 s each, standard stand rule; the adapter / model of each
entry below. Stored per trial: waypoints reached, outcome, final pelvis position (1e-4 m) and a hash of the final qpos rounded
to 1e-5. Covers the harness (commands, contacts, progress), x2.py (model build), the apparatus data, and the ACP hand-over and
baseline adapters. It does not cover the Hiking / SSR adapters (their own obs checks do) or anything after 12 s (end stands).
"""
import argparse
import contextlib
import hashlib
import json
import os
import pathlib
import sys

# Single-threaded maths (2026-11-13): a multi-threaded run changed 3 of the 10 regression trials (baseline adapter); the
# eval sets one thread unless the caller chose otherwise. Must come before numpy / torch / onnxruntime are imported.
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REF = HERE / "config" / "regression_ref.json"
SECONDS = 12.0
CASES = [  # (name, adapter, env, seeds)
    ("acp_4250_corrected", "acp:ACPHandover", dict(X2_EXCLUDE_PAIRS="hip", ACP_HANDOVER_DIR=str(HERE / "data" / "handover" / "acp_s1v2_4250"),
                                                     ACP_GOAL_R="0.2", ACP_LOOKAHEAD="0.3"), [1000, 1001, 1002, 1003, 1004, 1005]),
    ("baseline_20261102_old_model", "acp_baseline:ACPBaseline", dict(X2_EXCLUDE_PAIRS="", ACP_GOAL="shrink"), [1000, 1001, 1002, 1003]),
]


@contextlib.contextmanager
def env(vals):
    old = {k: os.environ.get(k) for k in vals}
    for k, v in vals.items():
        if v:
            os.environ[k] = v
        else:
            os.environ.pop(k, None)
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def run_all():
    import harness as Hn
    import x2 as X
    from apparatus import Apparatus
    out = {}
    for name, adn, e, seeds in CASES:
        with env(dict(e, STAND_RULE="strict", PA_APPARATUS_DIR="", APP_NOCONTACT_WALLS="")):
            app = Apparatus("ours")                         # the real course (PA_APPARATUS_DIR cleared above)
            ad = Hn.load_adapter(adn, None)
            x = X.build(app, ad.plant(X.joint_names_of(X.robot_dir() / "mjcf" / "x2_ultra.xml")))
            tr = Hn.Trial(x, app, ad, x.model.body_mass.copy(), x.model.body_inertia.copy())
            for s in seeds:
                r = tr.run(s, Hn.NOMINAL, SECONDS, record=False)
                q = np.round(tr.d.qpos.astype(np.float64), 5)
                out["%s/%d" % (name, s)] = dict(waypoints=r["waypoints"], outcome=r["reason"], final=[round(float(v), 4) for v in tr.d.qpos[:3]],
                                                qpos_hash=hashlib.sha1(q.tobytes()).hexdigest()[:16])
    return out


def check(quiet=False):
    """Compare with the stored reference. Returns (ok, message)."""
    if not REF.exists():
        return False, "regression reference missing (%s): run python eval/regression.py --record" % REF
    ref = json.load(open(REF))
    now = run_all()
    bad = []
    for k, v in ref["trials"].items():
        n = now.get(k)
        if n is None or n["waypoints"] != v["waypoints"] or n["outcome"] != v["outcome"] or np.max(np.abs(np.subtract(n["final"], v["final"]))) > 2e-3:
            bad.append("%s: stored %s %s %s, now %s" % (k, v["waypoints"], v["outcome"], v["final"], n and (n["waypoints"], n["outcome"], n["final"])))
    exact = sum(now[k]["qpos_hash"] == v["qpos_hash"] for k, v in ref["trials"].items() if k in now)
    msg = "REGRESSION %s: %d of %d trials match (waypoints, outcome, final pelvis within 2 mm); %d bit-identical (qpos hash); reference of %s" % (
        "OK" if not bad else "CHANGED", len(ref["trials"]) - len(bad), len(ref["trials"]), exact, ref["recorded"])
    if bad:
        msg += "\n  " + "\n  ".join(bad)
    return not bad, msg


def gate():
    """Called at the start of a harness or ladder run (skipped with EVAL_REGRESSION=0). Prints the result; a change is reported
    and written into the run's log but does not stop the run (the run's results then say which code they came from)."""
    if os.environ.get("EVAL_REGRESSION", "1") == "0":
        return None
    ok, msg = check()
    print(msg, flush=True)
    return dict(ok=ok, message=msg.splitlines()[0])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    a = ap.parse_args()
    if a.record:
        import datetime
        REF.parent.mkdir(parents=True, exist_ok=True)
        json.dump(dict(recorded=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), seconds=SECONDS, trials=run_all()), open(REF, "w"), indent=1)
        print("recorded", REF)
    else:
        ok, msg = check()
        print(msg)
        sys.exit(0 if ok else 1)
