# Rebuild the X2 runs' points in dashboard/data/progress.json from the training metrics cache of eval/metrics/plots.py
# (run `python eval/metrics/plots.py` first; the hourly refresh does both). Added 2026-11-13 after the ACP card was found
# frozen at 9,230 with no episode length: progress points had been appended by hand and stopped.
# Every 250 iterations per track: reward, episode length in seconds, terrain level (ACP has no terrain curriculum: None).
import json, math, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval" / "metrics"))
import plots  # noqa: E402

STEP = 250
TOTAL = {"acp": 34218, "ssr": 20000, "hiking": 30000}
LABEL = {"acp": "ACP on the X2 (stage 1, 2 and 3 lineage)", "ssr": "SSR, run r7", "hiking": "Hiking on the X2 (run x2_parkour_v4)"}


def num(v):
    return v if isinstance(v, (int, float)) and math.isfinite(v) else None


def point(track, it, r):
    if track == "acp":
        ep = num(r.get("episodes"))
        envs = num(r.get("envs")) or 36000
        return dict(reward=num(r.get("reward_per_step")), episode_len=round(envs * 24 / ep * 0.02, 2) if ep else None, terrain_level=None)
    if track == "ssr":
        return dict(reward=num(r.get("ep_ret")), episode_len=num(r.get("ep_len_s")), terrain_level=num(r.get("lvl_mean")))
    L = num(r.get("Train/mean_episode_length"))
    return dict(reward=num(r.get("Train/mean_reward_0")), episode_len=round(L * 0.02, 2) if L else None,
                terrain_level=num(r.get("Episode/Curriculum/terrain_levels")))


def main():
    p = ROOT / "dashboard" / "data" / "progress.json"
    old = json.load(open(p, encoding="utf-8"))
    keep = [x for x in old if x.get("run") not in TOTAL]
    new = []
    for track in TOTAL:
        rows, _ = plots.load(track)
        its = sorted(rows)
        if not its:
            continue
        picks = [i for i in its if i % STEP == 0] + [its[-1]]
        for it in sorted(set(picks)):
            win = [rows[j] for j in its if it - 20 <= j <= it]          # mean of up to 20 iterations ending here
            pts = [point(track, j, rr) for j, rr in zip([it] * len(win), win)]
            agg = {}
            for k in ("reward", "episode_len", "terrain_level"):
                v = [x[k] for x in pts if x[k] is not None]
                agg[k] = round(sum(v) / len(v), 3) if v else None
            new.append(dict(id="%s-%d" % (track, it), run=track, robot="X2", track=LABEL[track], iteration=it, total=TOTAL[track],
                            note="from the run's own metrics (mean of the last 20 iterations)", **agg))
    json.dump(keep + new, open(p, "w", encoding="utf-8", newline="\n"), ensure_ascii=False, indent=1)
    for t in TOTAL:
        last = [x for x in new if x["run"] == t][-1:]
        print(t, last[0] if last else None)


if __name__ == "__main__":
    main()
