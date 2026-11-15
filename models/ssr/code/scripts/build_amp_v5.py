"""amp_v5 (SSR v15, 2026-11-13): the motion prior per terrain group sized and composed as the paper's Table 11 (user rules
2026-11-13 in docs/USER_RULES.md: "SSR back to the paper", BONES-SEED dropped from style references, gap -> beams / logs).

Group budgets (s, unique data; training mirrors every frame on top): slope 261.3, stairs 295.8, discrete 136.0, beam (the gap
slot) 174.5, platform 181.6. Each group = flat forward 35 % + turn 15 % + its traversal 50 % (slope: forward 30 %, backward 5 %,
turn 15 %, ramp 50 %). Table 11 gives only the totals and the composition; the split is ours, the same for every group.

Sources (no BONES-SEED):
- the paper's Hiking reference retargeted to X2, v2 (soles on the floor): paper_ref_x2_v2.pkl. Segments 0-47 are the terrain
  clips (paper_ref_terrain_clips.json: step down off block, step onto block, stairs down, stairs up); 48-49 are the long flat
  takes, cut here into forward / turn / backward windows by smoothed base velocity.
- the user's Pico style set pico_style_v1.pkl (category = clip name prefix) and the user's Pico terrain-park takes beside it.

Rebuilt 2026-11-13 (SSR track) from the paper reference v4 (user-approved; torso / waist reading fixed): segment categories from
paper_ref_x2_v4_categories.json "terrain" time ranges on v4's own time line (checked against the table seconds and the v4 labels);
long takes 48-49 cut into forward / turn / side / backward windows. Layout follows ssr.terrain: paper layout (X2_TERRAIN_PAPER=1)
= ramp, stairs, beam (gap slot), platform, discrete as before; v14 layout (default) = ramp, stairs, beam, platform, flat, where
"flat" takes the discrete slot's Table 11 size (136.0 s) as flat forward 50 % + turn 35 % + side walk 15 % (open-floor robots and
the walking starts use it; user 2026-11-13: normal-pace flat walking and turning from the paper reference), and beam's traversal
is all Pico logs windows + the 2x2 and 4x4 Pico takes.

Writes data/amp_v5.npz (keys = ssr.terrain.TYPES + "_starts") and data/amp_v5_meta.json (seconds per group, source and
category; window counts; forward / backward / turn / standing seconds per group like the audit table). Exit 3 if the v2 paper
reference is not there yet. CPU only. Run from tracks/ssr/code (with --paper_terrain).
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import joblib
import mujoco
import numpy as np

import build_amp as B
from ssr import terrain as T
from ssr import x2 as X

HK = pathlib.Path("/mnt/filesystem-g4/data/motions/hiking_release_x2")
import os
PAPER = pathlib.Path(os.environ.get("AMP_V5_PAPER", HK / "paper_ref_x2_v4.pkl"))
CATS = pathlib.Path(os.environ.get("AMP_V5_CATS", HK / "paper_ref_x2_v4_categories.json"))
LABELS = pathlib.Path(os.environ.get("AMP_V5_LABELS", HK / "paper_ref_x2_v4_labels.json"))
OUT = os.environ.get("AMP_V5_OUT", "amp_v5")
PK = pathlib.Path("/mnt/filesystem-g4/data/motions/ssr_prior_sources/terrain_park_train_sets")
BUDGET = dict(ramp=261.3, stairs=295.8, discrete=136.0, beam=174.5, platform=181.6) if T.PAPER else dict(ramp=261.3, stairs=295.8, beam=174.5, platform=181.6, flat=136.0)
FLAT_SPLIT = dict(fwd=0.50, turn=0.35, side=0.15)
SPLIT = dict(fwd=0.35, turn=0.15, trav=0.50)
SLOPE_SPLIT = dict(fwd=0.30, back=0.05, turn=0.15, trav=0.50)
FPS = int(B.FPS)
WIN_MIN, WIN_MAX = 1.0, 4.0


def smooth(x, w=25):
    k = np.ones(w) / w
    return np.convolve(np.pad(x, (w // 2, w - 1 - w // 2), mode="edge"), k, mode="valid")


def runs(mask, fmin, fmax):
    """Contiguous True runs of at least fmin frames, split into pieces of at most fmax frames: list of (a, b)."""
    out, i, n = [], 0, len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        a = i
        while j - a >= fmin:
            b = min(a + fmax, j)
            if b - a < fmin:
                break
            out.append((a, b))
            a = b
        i = j
    return out


def main():
    if not PAPER.exists():
        print("waiting: %s not there yet" % PAPER, flush=True)
        sys.exit(3)
    x2 = X.build(foot_boxes=False)
    m = x2.model
    data = mujoco.MjData(m)
    ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in X.HAND_BODIES + X.FOOT_BODIES]
    rng = np.random.default_rng(15)
    fr = lambda v: B.frames(x2, data, ids, *B.resample(v))

    # ---- paper reference: terrain clips by segment, flat windows from the long takes
    pv2 = joblib.load(PAPER)
    keys = sorted(pv2)
    assert len(keys) == 50, len(keys)
    # segment categories from the time ranges of paper_ref_x2_categories.json "terrain" (the table quoted to the user: stairs up 24 s,
    # stairs down 16.8 s, step onto / off block 9.3 / 9.4 s), by each segment's midpoint on the v1 time line
    rngs = json.load(open(CATS))["terrain"]
    labs = json.load(open(LABELS))["segments"]
    cats, t_ = {}, 0.0
    for i, k in enumerate(keys):
        d_ = len(pv2[k]["dof"]) / float(pv2[k]["fps"])
        mid = t_ + d_ / 2
        for c_, v_ in rngs.items():
            if v_["t"][0] <= mid < v_["t"][1]:
                cats[i] = c_
        t_ += d_
    cat_chk = {}
    for c_, v_ in rngs.items():
        secs = sum(len(pv2[keys[i]]["dof"]) / float(pv2[keys[i]]["fps"]) for i in cats if cats[i] == c_)
        mains = sorted({labs[str(i)]["main"] for i in cats if cats[i] == c_})
        cat_chk[c_] = dict(clips=sum(1 for i in cats if cats[i] == c_), table_clips=v_["clips"], sec=round(secs, 1), table_sec=v_["sec"], label_main=mains)
        assert abs(secs - v_["sec"]) <= 1.0 and cat_chk[c_]["clips"] == v_["clips"], ("v4 category check", c_, cat_chk[c_])
    print("paper ref v4 terrain categories:", json.dumps(cat_chk), flush=True)
    pool, rej = {}, {}                                          # (source, category) -> list of frame arrays (windows / clips)

    def add(src, cat, f):
        why = "short" if len(f) < 2 * 5 else ("nonfinite" if not np.isfinite(f).all() else ("qd>25" if np.abs(f[:, 30:51]).max() > 25 else None))
        if why:
            rej["%s/%s %s" % (src, cat, why)] = rej.get("%s/%s %s" % (src, cat, why), 0) + 1
            return
        pool.setdefault((src, cat), []).append(f)
    for i, k in enumerate(keys):
        f = fr(pv2[k])
        if i in cats:
            add("paper_ref", cats[i], f)
            continue
        vx, vy, wz = smooth(f[:, 3]), smooth(f[:, 4]), smooth(f[:, 8])
        sp = np.hypot(vx, vy)
        fwd = (vx >= 0.15) & (np.abs(wz) < 0.5) & (np.abs(vy) < 0.6 * np.maximum(sp, 1e-6))   # the reference labeler's walking rule (speed >= 0.15, lateral share < 0.6), minus turns
        turn = np.abs(wz) >= 0.5                                                              # turning in place and walking turns
        back = vx <= -0.15
        side = (np.abs(vy) >= 0.15) & (np.abs(vy) >= 0.6 * np.maximum(sp, 1e-6)) & (np.abs(wz) < 0.5)      # side walk (v14 layout flat group)
        for lab, msk in (("flat forward", fwd), ("turn", turn), ("backward", back), ("side", side)):
            for a, b in runs(msk, int(WIN_MIN * FPS), int(WIN_MAX * FPS)):
                add("paper_ref", lab, f[a:b])

    # ---- Pico style set (curated windows) and the full terrain-park takes
    ps = joblib.load(PK / "pico_style_v1.pkl")
    for k in sorted(ps):
        cat = next((c for c in ("flat_walk", "ramp", "logs", "stairs", "turn_in_place", "side_walk") if k.startswith(c)), None)
        if cat is None:
            print("pico_style_v1: clip without a category prefix:", k, flush=True)
            continue
        add("pico_style_v1", cat, fr(ps[k]))
    full = {}
    for nm in ("acp_tp_ramp_sh2_x30", "tp_ramp_base_x30_mir", "tp_stairs_all_v1_1", "tp_t4x4u_y20", "pico_2026-10-29_2x2_train_v1_1", "tp_stairs_x10"):
        d = joblib.load(PK / (nm + ".pkl"))
        ks = sorted(d)
        rng.shuffle(ks)
        full[nm] = [f for f in (fr(d[k]) for k in ks) if len(f) >= FPS // 2 and np.isfinite(f).all() and np.abs(f[:, 30:51]).max() <= 25]
    for nm, L in full.items():
        pool[(nm, "take")] = L

    def shuffled(key):
        L = list(pool.get(key, []))
        o = rng.permutation(len(L))
        return [L[i] for i in o]

    # split the Pico logs windows and the corrected stair takes between two groups each, no window used twice
    logs = shuffled(("pico_style_v1", "logs"))
    stairs_all = shuffled(("tp_stairs_all_v1_1", "take"))

    def take(items, secs, used):
        """Frames from the list in order until secs; the last one cut to fit. Returns the list of arrays and the seconds taken."""
        out, got = [], 0.0
        while items and got < secs - 0.25:
            f = items.pop(0)
            need = int(round((secs - got) * FPS))
            f = f[:need] if len(f) > need else f
            if len(f) < 10:
                continue
            out.append(f)
            got += len(f) / FPS
        used.append(got)
        return out, got

    flat_fwd = lambda: shuffled(("paper_ref", "flat forward")) + shuffled(("pico_style_v1", "flat_walk"))
    turns = lambda: shuffled(("paper_ref", "turn")) + shuffled(("pico_style_v1", "turn_in_place"))
    # pools mixed, then reshuffled, so each group's flat share draws from both sources (reused across groups, as the paper's clips are)
    mix = lambda L: [L[i] for i in rng.permutation(len(L))]
    plan = {
        "ramp": [("fwd", lambda: mix(flat_fwd())), ("back", lambda: shuffled(("paper_ref", "backward"))), ("turn", lambda: mix(turns())),
                 ("trav", [("pico_style_v1/ramp", shuffled(("pico_style_v1", "ramp")), None), ("acp_tp_ramp_sh2_x30", shuffled(("acp_tp_ramp_sh2_x30", "take")), 0.5),
                           ("tp_ramp_base_x30_mir", shuffled(("tp_ramp_base_x30_mir", "take")), 1.0)])],
        "stairs": [("fwd", lambda: mix(flat_fwd())), ("turn", lambda: mix(turns())),
                   ("trav", [("paper_ref/stairs up", shuffled(("paper_ref", "stairs up")), None), ("paper_ref/stairs down", shuffled(("paper_ref", "stairs down")), None),
                             ("pico_style_v1/stairs", shuffled(("pico_style_v1", "stairs")), None), ("tp_stairs_all_v1_1", stairs_all, 1.0)])],
        "discrete": [("fwd", lambda: mix(flat_fwd())), ("turn", lambda: mix(turns())),
                     ("trav", [("pico_2026-10-29_2x2_train_v1_1", shuffled(("pico_2026-10-29_2x2_train_v1_1", "take")), None), ("pico_style_v1/logs", logs, 1.0)])],
        "beam": [("fwd", lambda: mix(flat_fwd())), ("turn", lambda: mix(turns())),
                 ("trav", [("pico_style_v1/logs", logs, None), ("tp_t4x4u_y20", shuffled(("tp_t4x4u_y20", "take")), 1.0)])] if T.PAPER else
                [("fwd", lambda: mix(flat_fwd())), ("turn", lambda: mix(turns())),
                 ("trav", [("pico_style_v1/logs", logs, None), ("pico_2026-10-29_2x2_train_v1_1", shuffled(("pico_2026-10-29_2x2_train_v1_1", "take")), 0.5),
                           ("tp_t4x4u_y20", shuffled(("tp_t4x4u_y20", "take")), 1.0)])],
        "flat": [("fwd", lambda: mix(flat_fwd())), ("turn", lambda: mix(turns())),
                 ("side", lambda: shuffled(("paper_ref", "side")) + shuffled(("pico_style_v1", "side_walk")))],
        "platform": [("fwd", lambda: mix(flat_fwd())), ("turn", lambda: mix(turns())),
                     ("trav", [("paper_ref/step onto block", shuffled(("paper_ref", "step onto block")), None), ("paper_ref/step down off block", shuffled(("paper_ref", "step down off block")), None),
                               ("tp_stairs_x10", shuffled(("tp_stairs_x10", "take")), None), ("tp_stairs_all_v1_1", stairs_all, 1.0)])],
    }
    # traversal lists: None = take all of it (up to what is left of the budget); a number = share of the remainder (1.0 = fill)
    out, meta = {}, dict(budget_s=BUDGET, split=SPLIT, slope_split=SLOPE_SPLIT, groups={})
    for g in (("ramp", "stairs", "discrete", "beam", "platform") if T.PAPER else ("ramp", "stairs", "beam", "platform", "flat")):   # discrete before beam: it takes its share of the Pico logs windows first
        tot = BUDGET[g]
        sp = SLOPE_SPLIT if g == "ramp" else (FLAT_SPLIT if g == "flat" else SPLIT)
        frames, rec = [], {}
        short = 0.0
        for part, src in plan[g]:
            want = tot * sp[part] + (short if part == "turn" else 0.0)   # a backward shortfall goes to turning
            if part != "trav":
                used = []
                fs, got = take(src(), want, used)
                frames += fs
                rec[part] = round(got, 1)
                if part == "back":
                    short = max(0.0, want - got)
                continue
            left, trav = want, {}
            for i_, (nm, items, share) in enumerate(src):
                rem = [s for s in src[i_ + 1:]]
                if share is None:
                    secs = left if not rem else min(left, sum(len(f) for f in items) / FPS)
                elif share >= 1.0:
                    secs = left
                else:
                    secs = left * share
                fs, got = take(items, secs, [])
                frames += fs
                trav[nm] = round(got, 1)
                left -= got
            rec["trav"] = trav
        n = np.cumsum([0] + [len(f) for f in frames])[:-1]
        F = np.concatenate(frames).astype(np.float32)
        out[g], out[g + "_starts"] = F, n.astype(np.int64)
        vx, vy, wz = F[:, 3], F[:, 4], F[:, 8]
        spd = np.hypot(vx, vy)
        rec.update(total_s=round(len(F) / FPS, 1), clips=len(frames), check=dict(forward_vx_gt_0p2=round(float((vx > 0.2).sum()) / FPS, 1),
                   backward_vx_lt_m0p2=round(float((vx < -0.2).sum()) / FPS, 1), turn_wz_gt_0p5=round(float((np.abs(wz) > 0.5).sum()) / FPS, 1),
                   lateral_vy_gt_0p2=round(float((np.abs(vy) > 0.2).sum()) / FPS, 1), standing_lt_0p1=round(float((spd < 0.1).sum()) / FPS, 1)))
        meta["groups"][g] = rec
        print(g, json.dumps(rec), flush=True)
    meta["rejected"] = rej
    meta["pool_s"] = {"%s/%s" % k: round(sum(len(f) for f in v) / FPS, 1) for k, v in pool.items()}
    meta["gaps"] = ("Pico gap list (user 2026-11-13): open-floor turning and sidestepping are thin (pico_style_v1 turn_in_place 12.6 s, "
                    "side_walk windows cross the 4x4); no side walk in any group (Table 11 composition), flat walking at normal pace and "
                    "ramp down also thin in Pico, covered by the paper reference's flat takes for flat walking only.")
    meta["paper_ref_file"] = str(PAPER)
    meta["paper_ref_categories"] = cat_chk
    meta["layout"] = list(T.TYPES)
    assert sorted(g for g in out if not g.endswith("_starts")) == sorted(T.TYPES), (sorted(out), T.TYPES)
    np.savez(ROOT / ("data/%s.npz" % OUT), **out)
    json.dump(meta, open(ROOT / ("data/%s_meta.json" % OUT), "w"), indent=1)
    print("wrote data/%s.npz" % OUT, {g: round(len(out[g]) / FPS, 1) for g in T.TYPES}, flush=True)




if __name__ == "__main__":
    main()
