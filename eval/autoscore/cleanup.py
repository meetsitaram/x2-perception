# Local disk retention for fetched checkpoints and eval videos (user, 2026-11-13: "cleanup older local checkpoint
# periodically to not fill up disk space"). Run at every coordinator refresh:  python eval/autoscore/cleanup.py [--dry-run]
#
# Per track, the KEEP newest hand-over folders (by modification time) and the best-scored checkpoint keep everything.
# Older ones lose only re-fetchable or bulky files: model files (policy*.onnx, *.pt, obs_check.npz) in eval/data/handover/,
# and videos (*.mp4) in eval/out/<track>/. Scores, logs, CSV and JSON are never touched. The exports themselves stay on
# the shared filesystem, so any checkpoint can be fetched again.
import csv, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KEEP = 5
TRACKS = ("acp", "ssr", "hiking")
BULKY_HANDOVER = (".onnx", ".pt", ".pth")
BULKY_NAMES = ("obs_check.npz",)
DRY = "--dry-run" in sys.argv
# Never touched: the eval's fixed references (eval/regression.py CASES and baselines). 2026-11-13: the first cleanup run removed
# acp_s1v2_4250's planner and broke the regression check, so every training-terrain job failed until it was restored.
PROTECT = ("acp_s1v2_4250", "acp", "acp_2250", "baseline")


def _disabled_tracks():
    import json
    try:
        c = json.load(open(ROOT / "eval/autoscore/config.json"))
        return {t for t, v in c.get("tracks", {}).items() if not v.get("enabled", True)}
    except Exception:
        return set()


def protected(d):
    # 2026-11-15: never touch stopped tracks (SSR r7_v20_final_26000 lost its policy.onnx while still wanted for renders)
    if any(d.name.startswith(t) or ("/" + t + "/") in d.as_posix() for t in _disabled_tracks()):
        return True
    return d.name in PROTECT or d.name.startswith("baseline") or d.name in _regression_dirs()


def _regression_dirs():
    import re
    f = ROOT / "eval/regression.py"
    return set(re.findall(r'"handover" / "([A-Za-z0-9_]+)"', f.read_text(encoding="utf-8"))) if f.exists() else set()


def best_checkpoints():
    """Highest sum of station-leg successes per track in results.csv (kept regardless of age)."""
    best = {}
    f = ROOT / "eval/autoscore/results.csv"
    if not f.exists():
        return set()
    for r in csv.DictReader(open(f, encoding="utf-8")):
        s = 0
        for k in ("leg1", "leg2", "leg3", "leg4", "leg5"):
            v = (r.get(k) or "").split("/")[0]
            s += int(v) if v.isdigit() else 0
        t = r.get("track")
        if t and s > best.get(t, (-1, ""))[0]:
            best[t] = (s, r.get("checkpoint", ""))
    return {c for _, c in best.values() if c}


def track_of(name):
    for t in TRACKS:
        if name.startswith(t + "_") or name == t:
            return t
    return "acp" if name.startswith(("pair_", "handover_pair")) else None


def main():
    best = best_checkpoints()
    freed, n = 0, 0
    groups = {}
    for base in (ROOT / "eval/data/handover", ROOT / "eval/out/acp", ROOT / "eval/out/ssr", ROOT / "eval/out/hiking"):
        if not base.is_dir():
            continue
        for d in base.iterdir():
            if not d.is_dir():
                continue
            t = base.name if base.parent.name == "out" else track_of(d.name)
            if t:
                groups.setdefault((base, t), []).append(d)
    for (base, t), dirs in groups.items():
        dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        for d in dirs[KEEP:]:
            if any(b and b in d.name for b in best) or protected(d):
                continue
            for f in d.rglob("*"):
                if not f.is_file():
                    continue
                bulky = f.suffix == ".mp4" or (base.name == "handover" and (f.suffix in BULKY_HANDOVER or f.name in BULKY_NAMES))
                if bulky:
                    sz = f.stat().st_size
                    if not DRY:
                        f.unlink()
                    freed += sz; n += 1
    print("%s %d files, %.2f GB; kept newest %d per track plus best %s" % (
        "would remove" if DRY else "removed", n, freed / 1e9, KEEP, sorted(best)))


if __name__ == "__main__":
    main()
