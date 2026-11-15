"""Training plots for review (wandb-style, offline): sync the metrics the three runs already write, build one HTML page.

    python eval/metrics/plots.py            # incremental sync from the shared filesystem, then build the page
    python eval/metrics/plots.py --no-sync  # build from the local cache only

Read-only on the nodes: one ssh call pipes remote_extract.py (stdlib, CPU) into python3 on the first reachable node; it
returns only the bytes appended since the last sync (jsonl) or the scalars of new tfevents records. Nothing is written on
the node. Node addresses / logins come from /tmp/pa_nodes.txt (scripts/node_check.sh) and are never printed or stored.

Cache: eval/metrics/cache/ (git-ignored). Output: eval/metrics/out/training_plots.html (self-contained, opens offline).
Lineage: one global iteration axis per track; inside a run folder a restart from an earlier checkpoint overwrites the rows it
repeats (last written wins), and the ACP folders are cut at the lineage boundaries of docs/runs/ledger_acp.md. Vertical
markers = rows of docs/runs/ledger_<track>.md (solid = kept segment, dashed = tried and discarded / fell back).
"""
import argparse, base64, csv, gzip, io, json, math, os, re, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
CACHE = HERE / "cache"
OUT = HERE / "out" / "training_plots.html"
# Git Bash's /tmp is %TEMP% on Windows
NODES = Path(os.environ.get("PA_NODES") or ("/tmp/pa_nodes.txt" if os.name != "nt" else
                                             os.path.join(os.environ.get("TEMP", ""), "pa_nodes.txt")))
SSH = next((p for p in [r"C:\Program Files\Git\usr\bin\ssh.exe"] if os.path.exists(p)), "ssh")
MAXPTS = 1200

GLOBS = {
    "acp": ["acp/runs/s[123]_v*/metrics.jsonl"],
    "ssr": ["ssr/runs/r7/log.jsonl"],
    "hiking": ["hiking/runs/x2_parkour_v4/2026*/events.out.tfevents.*"],
}
# ACP lineage of the running tracker (docs/runs/ledger_acp.md "Lineage"): folder, first global iteration taken from it.
# A folder listed with None starts at its own first logged iteration (for the next segment, e.g. s2_v5).
ACP_LINEAGE = [("s1_v1", 0), ("s2_v1", 2250), ("s1_v2", 4000), ("s2_v2", 5500), ("s2_v3", 7000), ("s2_v4", 10356),
               ("s2_v5", None), ("s2_v6", None), ("s3_v1", None)]
# Ledger rows that belong to the plotted lineage (SSR: r7 from row 23 on; Hiking: run 4 from row 9 on).
LEDGER_FIRST_ROW = {"acp": 1, "ssr": 23, "hiking": 9}
# SSR checkpoint names in the scorer -> global iteration (docs/runs/ledger_ssr.md, dashboard/data/apparatus_results.json).
SSR_CKPT = {"r7_v12_last": 4300, "r7_v14_p250": 5050, "r8_v15_p100": 7110, "r9_v16_p300": 8260}

# ----------------------------------------------------------------------------------------------------------------- sync

def mask(s):
    return re.sub(r"\d+\.\d+\.\d+\.\d+", "<addr>", s)


def sync():
    t0 = time.time()
    CACHE.mkdir(parents=True, exist_ok=True)
    stf = CACHE / "sync_state.json"
    state = json.loads(stf.read_text()) if stf.exists() else {}
    if not NODES.exists():
        print("sync: %s missing (run bash scripts/node_check.sh); using the cache" % NODES); return
    nodes = [l.split() for l in NODES.read_text().splitlines() if len(l.split()) >= 4]
    globs = [g for gs in GLOBS.values() for g in gs]
    arg = base64.b64encode(json.dumps({"globs": globs, "offsets": state}).encode()).decode()
    script = (HERE / "remote_extract.py").read_bytes()
    data = None
    for name, ip, user, key in [n[:4] for n in nodes]:
        cmd = [SSH, "-i", os.path.expanduser(key), "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
               "-o", "LogLevel=ERROR", "%s@%s" % (user, ip), "python3 - " + arg]
        try:
            p = subprocess.run(cmd, input=script, capture_output=True, timeout=900)
        except Exception as ex:
            print("sync: node %s: %s" % (name, mask(str(ex))[:200])); continue
        if p.returncode == 0 and p.stdout:
            data = p.stdout; break
        print("sync: node %s rc %d: %s" % (name, p.returncode, mask(p.stderr.decode("utf-8", "replace"))[-300:]))
    if data is None:
        print("sync: no node reachable; using the cache"); return
    text = gzip.decompress(data).decode("utf-8", "replace")
    nb = 0; nf = 0; cur = None
    for line in io.StringIO(text):
        if line.startswith("##FILE "):
            if cur: cur.close()
            _, path, old, new, size, kind = line.split()
            old, new = int(old), int(new)
            lf = CACHE / (path.replace("/", "__") + (".jsonl" if kind == "tf" else ""))
            mode = "a" if old > 0 and old == state.get(path, 0) and lf.exists() else "w"
            if old != state.get(path, 0) and old == 0:
                print("sync: %s restarted from 0 (file replaced)" % path)
            cur = open(lf, mode, encoding="utf-8", newline="\n")
            if new > old: nf += 1; nb += new - old
            state[path] = new
        elif cur:
            cur.write(line)
    if cur: cur.close()
    stf.write_text(json.dumps(state, indent=0))
    print("sync: %d files with new data, %.1f MB new on the node, %.1f MB transferred, %.0f s" % (
        nf, nb / 1e6, len(data) / 1e6, time.time() - t0))

# ----------------------------------------------------------------------------------------------------------------- load

def flatten(d, pre="", out=None):
    out = {} if out is None else out
    for k, v in d.items():
        kk = pre + k
        if isinstance(v, dict):
            flatten(v, kk + ".", out)
        elif isinstance(v, list):
            for i, x in enumerate(v):
                if isinstance(x, (int, float)) and not isinstance(x, bool):
                    out["%s.%d" % (kk, i)] = x
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[kk] = v
    return out


def read_rows(f, merge=False):
    """{iteration: row}, last written wins (a restart from an earlier checkpoint replaces the repeated iterations)."""
    rows = {}
    if not f.exists():
        return rows
    with open(f, encoding="utf-8") as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            it = d.get("it", d.get("iter"))
            if it is None:
                continue
            r = flatten(d)
            if merge and it in rows:
                rows[it].update(r)
            else:
                rows[it] = r
    return rows


def load(track):
    files = sorted(CACHE.glob(track + "__runs__*"))
    if track == "acp":
        segs = []
        for folder, start in ACP_LINEAGE:
            r = read_rows(CACHE / ("acp__runs__%s__metrics.jsonl" % folder))
            if r:
                segs.append((folder, min(r) if start is None else start, r))
        rows = {}
        for i, (folder, start, r) in enumerate(segs):
            end = segs[i + 1][1] if i + 1 < len(segs) else 10 ** 9
            rows.update({k: v for k, v in r.items() if start <= k < end})
        return rows, [s[0] for s in segs]
    if track == "ssr":
        return read_rows(CACHE / "ssr__runs__r7__log.jsonl"), ["r7"]
    rows = {}
    used = []
    for f in sorted(files):          # timestamped run dirs: later dirs overwrite repeated iterations (fall-backs)
        r = read_rows(f, merge=True)
        if r:
            used.append(f.name.split("__")[3])
        rows.update(r)
    return rows, used

# --------------------------------------------------------------------------------------------------------- panel specs

def _weight_key(k):
    """Episode-count key for a finish-rate key (ACP logs episodes/<set> and probe/episodes/<set>), else None."""
    if k.startswith("probe/finish/"):
        return "probe/episodes/" + k[len("probe/finish/"):]
    if k.startswith("finish/"):
        return "episodes/" + k[len("finish/"):]
    return None


def mean_of(keys):
    """Mean over keys; finish rates are weighted by their episode counts and sets with 0 episodes are skipped
    (2026-11-13: an unweighted mean counted empty sets as 0 and showed a false Pico drop at 10,356)."""
    def f(r):
        num = den = 0.0
        for k in keys:
            v = r.get(k)
            if v is None or not math.isfinite(v):
                continue
            wk = _weight_key(k)
            if wk is not None and wk in r:
                w = r[wk] or 0
                if w <= 0:
                    continue
            else:
                w = 1.0
            num += w * v; den += w
        return num / den if den else None
    return f


def families(allkeys, prefix, fams, strip="", cap=8):
    """Fold keys `prefix<name>` into named families (regex on name), at most `cap` series; leftovers -> 'other'."""
    names = sorted(k[len(prefix):] for k in allkeys if k.startswith(prefix) and (not strip or not k.endswith(strip)))
    out = []; used = set()
    for label, rx in fams:
        ks = [prefix + n for n in names if re.search(rx, n) and n not in used]
        used.update(k[len(prefix):] for k in ks)
        if ks:
            out.append((label + (" (%d)" % len(ks) if len(ks) > 1 else ""), mean_of(ks), ", ".join(k[len(prefix):] for k in ks)))
    while len(out) > cap - 1 and len(out) + (1 if any(n not in used for n in names) else 0) > cap:
        used -= set(out.pop()[2].split(", "))       # fold the last families into 'other' (never a 9th hue)
    rest = [prefix + n for n in names if n not in used]
    if rest:
        out.append(("other (%d)" % len(rest), mean_of(rest), ", ".join(k[len(prefix):] for k in rest)))
    return out


def S(label, key, note=""):
    return (label, key, note)


def specs(track, keys):
    """Panels: (title, unit, [series]); series = (label, key-or-function, note)."""
    if track == "acp":
        fin = [("flat / blocks / edge", r"loco_flat|loco_blocks|stand_edge"), ("stairs", r"stairs|gst|gs2"),
               ("ramp", r"ramp"), ("4x4 turn-around", r"t4x4"), ("Pico captures", r"pico"), ("chains", r"chains"),
               ("turn in place", r"turn_place")]
        return [
            ("Reward per step", "", [S("reward_per_step", "reward_per_step")]),
            ("Return (PPO mean)", "", [S("return", "ppo/return_mean"), S("value estimate", "ppo/value_mean")]),
            ("Clip finish rate (phase 0.99)", "share", [S("finished", "end/finished_phase099")]),
            ("Episode ends", "share", [S("base pos > 0.5 m", "end/term_base_pos_gt_0.5m"), S("base rot > 1 rad", "end/term_base_rot_gt_1rad"),
                                       S("timeout", "end/timeout_before_phase099")]),
            ("Finish rate by terrain set (training)", "share", families(keys, "finish/", fin)),
            ("Probe finish by terrain set (planner-driven)", "share", families(keys, "probe/finish/", fin)),
            ("Stage-2 curriculum eta", "", [S("eta_obs", "stage2/eta_obs"), S("eta_rew", "stage2/eta_rew")]),
            ("CSI start phase (curriculum)", "phase", [S("mean", "csi/phase_mean"), S("p90", "csi/phase_p90")]),
            ("Tracking error, positions", "m", [S("base pos", "err/base_pos_m"), S("contact pos (base frame)", "err/contact_pos_base_m"),
                                                S("placement at onset", "err/contact_placement_at_onset_m")]),
            ("Tracking error, joints", "rad", [S("joint pos mean |err|", "err/joint_pos_rad_mean_abs")]),
            ("Contact label agreement", "share", [S("agreement", "contact/label_agreement")]),
            ("Speed, base vs reference", "m/s", [S("base", "speed/base_xy_mps"), S("reference", "speed/ref_xy_mps")]),
            ("Planner losses", "", [S("affordance", "planner/loss_affordance"), S("occupancy", "planner/loss_occupancy")]),
            ("Planner fed to tracker", "share", [S("share", "planner/fed_to_tracker_share")]),
            ("Value loss", "", [S("value loss", "ppo/value_loss")]),
            ("Policy loss", "", [S("policy loss", "ppo/policy_loss")]),
            ("KL", "", [S("kl", "ppo/kl")]),
            ("Action std", "", [S("action std", "ppo/action_std")]),
            ("Throughput", "env steps/s", [S("env steps/s", "env_steps_per_s")]),
            ("Sim-fault share (diverged ends)", "share", [S("diverged", "end/term_diverged")]),
            ("GPU utilisation", "%", [S("gpu util", "gpu_util_pct")]),
        ]
    if track == "ssr":
        ext = [("tight spots", r"tight"), ("log corridors", r"log_corr"), ("homes", r"homes"), ("street front", r"street"),
               ("stairs", r"^stairs"), ("wedges", r"wedges"), ("beams", r"beam"), ("blocks", r"blocks")]
        types = sorted({k.split(".")[1] for k in keys if k.startswith("levels_by_type.")})
        ext_s = []
        for label, rx in ext:
            ks = ["ext.%s.level" % n for n in sorted({k.split(".")[1] for k in keys if k.startswith("ext.") and k.endswith(".level")}) if re.search(rx, n)]
            if ks:
                ext_s.append((label, mean_of(ks), ", ".join(ks)))
        def ext_succ(r):
            s = sum(v for k, v in r.items() if k.startswith("ext.done.") and k.endswith("_succ"))
            f = sum(v for k, v in r.items() if k.startswith("ext.done.") and k.endswith("_fail"))
            return s / (s + f) if s + f > 0 else None
        return [
            ("Episode return", "", [S("ep_ret", "ep_ret")]),
            ("Reward by group (per step)", "", [S("locomotion r_l", "r_l"), S("foothold r_f", "r_f"), S("style r_s", "r_s")]),
            ("Episode length", "s", [S("ep_len_s", "ep_len_s")]),
            ("Terrain level by type", "level", [S(t, "levels_by_type." + t) for t in types] + [S("mean", "lvl_mean")]),
            ("Extended map level by type", "level", ext_s),
            ("Extended map route success", "share", [S("succ / (succ + fail)", ext_succ)]),
            ("Speed made good (vx_ratio)", "share of command", [S("vx_ratio", "gait.vx_ratio")]),
            ("Speed, command vs made good", "m/s", [S("commanded", "gait.cmd_speed"), S("made good", "gait.vx_mps")]),
            ("Style reward by group", "", [S(t, "style_by_group." + t) for t in types]),
            ("Discriminator output", "", [S("D(data)", "d_r"), S("D(policy)", "d_f")]),
            ("Progress along route", "share", [S("prog_mean", "prog_mean"), S("reached end", "done_share.reached_end")]),
            ("Episode ends", "share", [S("tilt", "done_share.tilt"), S("low", "done_share.low"), S("wall hit", "done_share.wall_hit"),
                                       S("side obstacle", "done_share.side_obstacle"), S("stuck", "done_share.stuck"),
                                       S("time limit", "done_share.time_limit")]),
            ("Sim-fault share (diverged ends)", "share", [S("diverged", "done_share.diverged")]),
            ("Gait phases", "share", [S("single support", "gait.single"), S("double support", "gait.double"), S("flight", "gait.flight")]),
            ("Value loss", "", [S("v", "v")]),
            ("Policy loss", "", [S("pi", "pi")]),
            ("KL", "", [S("kl", "kl")]),
            ("Action std", "", [S("std", "std")]),
            ("Throughput", "samples/s", [S("samples/s", "samples_per_s")]),
            ("Obstacle share of envs", "share", [S("obstacle", "obst_share"), S("omni commands", "omni_share")]),
        ]
    # hiking
    lv = [("stairs up", r"^pyramid_stairs(_high)?$"), ("stairs down", r"stairs_inv"), ("boxes", r"boxes"), ("slope", r"slope"),
          ("rough", r"perlin|rough"), ("gaps", r"gap"), ("stepping / beams", r"stone|beam|step")]
    term = sorted(k for k in keys if k.startswith("Episode_Termination/"))
    simf = sorted(k for k in keys if re.search(r"(?i)sim.?fault|nan|diverg", k))
    return [
        ("Mean reward (per episode)", "", [S("mean reward", "Train/mean_reward_0")]),
        ("Mean episode length", "steps", [S("mean", "Train/mean_episode_length"), S("median", "Train/median_episode_length")]),
        ("Terrain level by type", "level", families(keys, "Episode/Curriculum/walk_monitor/level/", lv, cap=7) +
         [S("mean (all)", "Episode/Curriculum/terrain_levels")]),
        ("Speed made good (speed_ratio)", "share of command", [S("speed_ratio", "Episode/Curriculum/walk_monitor/speed_ratio")]),
        ("Speed, command vs actual", "m/s", [S("commanded", "Episode/Curriculum/walk_monitor/cmd_speed"),
                                              S("actual", "Episode/Curriculum/walk_monitor/actual_speed")]),
        ("Turn: rate along command", "share", [S("rate along cmd", "Episode/Curriculum/level_keeper/turn/rate_along_cmd")]),
        ("Velocity tracking error", "", [S("xy", "Episode/Metrics/base_velocity/error_vel_xy"), S("yaw", "Episode/Metrics/base_velocity/error_vel_yaw")]),
        ("Style / discriminator reward", "", [S("discriminator reward", "Step/discriminator_reward"),
                                              S("style reward weighted", "Step/style_reward_weighted")]),
        ("Discriminator output", "", [S("D(reference)", "Train/discriminator_reference"), S("D(actor)", "Train/discriminator_actor")]),
        ("Discriminator loss", "", [S("discriminator", "Loss/discriminator_loss"), S("gradient penalty", "Loss/discriminator_gradient_penalty")]),
        ("Episode ends", "share", [S(k.split("/")[1].replace("_", " "), k) for k in term if not re.search(r"(?i)sim.?fault|nan", k)][:8]),
        ("Sim-fault share", "share", [S(k.split("/")[-1], k) for k in simf][:3]),
        ("Gait: air per touchdown", "s", [S("air / touchdown", "Episode/Curriculum/walk_monitor/air_per_touchdown_s")]),
        ("Gait: step rate", "Hz", [S("step rate", "Episode/Curriculum/walk_monitor/step_rate_hz")]),
        ("Value loss", "", [S("value loss", "Loss/value_loss")]),
        ("Policy (surrogate) loss", "", [S("surrogate", "Loss/surrogate_loss")]),
        ("Entropy", "", [S("entropy", "Loss/entropy")]),
        ("Action noise std", "", [S("mean noise std", "Policy/mean_noise_std")]),
        ("Throughput", "steps/s", [S("total fps", "Perf/total_fps")]),
        ("Learning rate", "", [S("lr", "Loss/learning_rate")]),
    ]

# ------------------------------------------------------------------------------------------------------------ assemble

def bucket(rows):
    its = sorted(rows)
    if not its:
        return [], {}
    lo, hi = its[0], its[-1]
    w = max(1, math.ceil((hi - lo + 1) / MAXPTS))
    bins = {}
    for it in its:
        bins.setdefault((it - lo) // w, []).append(it)
    xs = sorted(bins)
    return [(lo + b * w + (w - 1) / 2.0, bins[b]) for b in xs], w


def series_values(rows, bins, fn):
    out = []; n = 0
    for x, its in bins:
        v = []
        for it in its:
            try:
                y = fn(rows[it]) if callable(fn) else rows[it].get(fn)
            except Exception:
                y = None
            if y is not None and isinstance(y, (int, float)) and math.isfinite(y):
                v.append(y)
        if v:
            n += len(v); out.append(round(sum(v) / len(v), 6))
        else:
            out.append(None)
    return out, n


def parse_ledger(track):
    f = REPO / "docs" / "runs" / ("ledger_%s.md" % track)
    if not f.exists():
        return []
    lines = [l for l in f.read_text(encoding="utf-8").splitlines() if l.startswith("|")]
    head = [c.strip() for c in lines[0].strip("|").split("|")]
    col = {h: i for i, h in enumerate(head)}
    num = lambda s: int(s.replace(",", ""))
    out = []
    for l in lines[2:]:
        c = [x.strip() for x in l.strip().strip("|").split("|")]
        if len(c) < len(head) or not c[0].isdigit() or int(c[0]) < LEDGER_FIRST_ROW[track]:
            continue
        run, its, frm = c[col["Run / version"]], c[col["Iterations"]], c[col["Started from"]]
        ended = c[col.get("Ended how", len(c) - 2)]
        m = re.match(r"^\s*~?([\d,]+)\s*(->|-)", its)
        if m:
            start = num(m.group(1))
        elif frm.lower().startswith("scratch"):
            start = 0
        else:
            m = re.search(r"\(([\d,]+)\)", frm) or re.search(r"model_(\d+)", frm) or re.search(r"([\d,]{3,})", frm)
            if not m:
                continue
            start = num(m.group(1))
        failed = bool(re.search(r"(?i)discard|fell back|not started|crash|tripwire|must not|^fall|\bFALL\b", ended))
        change = c[col.get("What changed vs previous segment", 0)] if "What changed vs previous segment" in col else ""
        out.append(dict(n=int(c[0]), it=start, run=re.sub(r"[`*]", "", run)[:80], its=its[:40], ended=re.sub(r"[`*]", "", ended)[:120],
                        change=re.sub(r"[`*]", "", change)[:160], kept=not failed))
    return out


def ckpt_iter(track, name):
    if track == "ssr":
        if name in SSR_CKPT:
            return SSR_CKPT[name]
        m = re.search(r"_(\d{4,})$", name)
        return int(m.group(1)) if m else None
    m = re.search(r"_(\d+)$", name)
    return int(m.group(1)) if m else None


def frac(s):
    m = re.match(r"\s*(\d+)\s*/\s*(\d+)", s or "")
    return round(100.0 * int(m.group(1)) / int(m.group(2)), 1) if m and int(m.group(2)) else None


def scorer_panels(track):
    pts = {k: [] for k in ["leg1", "leg2", "leg3", "leg4", "leg5", "product", "gate"]}
    f = REPO / "eval" / "autoscore" / "results.csv"
    names = []
    if f.exists():
        for r in csv.DictReader(open(f, encoding="utf-8")):
            if r["track"] != track:
                continue
            it = ckpt_iter(track, r["checkpoint"])
            if it is None:
                continue
            names.append((it, r["checkpoint"]))
            for k in range(1, 6):
                v = frac(r["leg%d" % k])
                if v is not None:
                    pts["leg%d" % k].append([it, v, r["checkpoint"]])
            if r.get("product"):
                try:
                    pts["product"].append([it, round(100 * float(r["product"]), 1), r["checkpoint"]])
                except ValueError:
                    pass
            g = re.search(r"(\d+)/(\d+)", r.get("gate", ""))
            if g:
                pts["gate"].append([it, round(100.0 * int(g.group(1)) / int(g.group(2)), 1), r["checkpoint"]])
    panels = [dict(title="Desktop scorer: station legs (success %)", unit="%", kind="points", ymin=0, ymax=100,
                   series=[dict(label="leg %d" % k, pts=pts["leg%d" % k]) for k in range(1, 6)]),
              dict(title="Desktop scorer: leg product and hold-out gate (%)", unit="%", kind="points", ymin=0, ymax=100,
                   series=[dict(label="product of legs", pts=pts["product"]), dict(label="hold-out gate passed", pts=pts["gate"])])]
    tf = REPO / "eval" / "autoscore" / "train_results.csv"
    tp = {}
    if tf.exists():
        agg = {}
        for r in csv.DictReader(open(tf, encoding="utf-8")):
            if r["track"] != track:
                continue
            it = ckpt_iter(track, r["checkpoint"])
            if it is None:
                continue
            a = agg.setdefault((r["terrain_type"], it, r["checkpoint"]), [0, 0])
            a[0] += int(r["successes"] or 0); a[1] += int(r["trials"] or 0)
        for (t, it, ck), (k, m) in sorted(agg.items()):
            if m:
                tp.setdefault(t, []).append([it, round(100.0 * k / m, 1), ck])
    panels.append(dict(title="Desktop scorer: training-terrain success by type (%)", unit="%", kind="points", ymin=0, ymax=100,
                       series=[dict(label=t, pts=v) for t, v in sorted(tp.items())][:8],
                       empty="no rows in eval/autoscore/train_results.csv yet" if not tp else ""))
    return panels


def build():
    tracks = []
    missing = []
    for track, title in [("acp", "ACP (adaptive contact planning)"), ("ssr", "SSR"), ("hiking", "Hiking in the Wild (X2)")]:
        rows, used = load(track)
        keys = set()
        for r in rows.values():
            keys.update(r.keys())
        bins, w = bucket(rows)
        xs = [round(x, 1) for x, _ in bins]
        panels = []
        for ptitle, unit, ser in specs(track, keys):
            ss = []
            for label, fn, *note in ser:
                ys, n = series_values(rows, bins, fn)
                if n == 0:
                    missing.append("%s / %s / %s" % (track, ptitle, label)); continue
                ss.append(dict(label=label, y=ys, note=note[0] if note else ""))
            if not ss:
                panels.append(dict(title=ptitle, unit=unit, kind="line", series=[], empty="not logged by this run"))
                continue
            panels.append(dict(title=ptitle, unit=unit, kind="line", series=ss))
        its = sorted(rows)
        tracks.append(dict(id=track, title=title, x=xs, bin=w, first=its[0] if its else None, last=its[-1] if its else None,
                           n=len(its), folders=used if len(used) <= 8 else used[:1] + ["... %d run dirs ..." % (len(used) - 2)] + used[-1:], panels=panels, scorer=scorer_panels(track), markers=parse_ledger(track),
                           last_time=(rows[its[-1]].get("time") if its else None)))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(dict(generated=time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), tracks=tracks), separators=(",", ":"))
    OUT.write_text(TEMPLATE.replace("__DATA__", data), encoding="utf-8")
    print("page: %s (%.1f MB)" % (OUT.relative_to(REPO), OUT.stat().st_size / 1e6))
    for t in tracks:
        print("  %s: %d iterations %s..%s, %d panels, %d markers" % (t["id"], t["n"], t["first"], t["last"], len(t["panels"]), len(t["markers"])))
    if missing:
        print("not found in the logs: " + "; ".join(missing))


TEMPLATE = (HERE / "plots_template.html").read_text(encoding="utf-8") if (HERE / "plots_template.html").exists() else ""

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-sync", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    if not a.no_sync:
        sync()
    build()
    print("total %.0f s" % (time.time() - t0))
