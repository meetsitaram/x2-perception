#!/usr/bin/env python
"""Automatic scorer: no LLM involved. See eval/autoscore/README.md.

Every poll_minutes it lists each track's hand-over exports on the shared filesystem, picks the ones to score (config.json:
priority first, then the newest unscored per track), and runs, one process per core and at most max_cores at once, on the
default floor (solid, trimmed) with the strict stand rule and the end retry:
  fetch + hand-over check -> [ACP: hold-out gate; legs only if it passes] -> flat gate (4 tests) -> station legs 1-5
  -> full route (4 x 50) only when every leg is at least route_min_leg of 50.
Every run resumes from its partial.json / summary.json, so nothing finished is run again. One row per export goes to
results.csv; a full route also updates dashboard/data/apparatus_results.json and runs dashboard/prepare.py.

  python eval/autoscore/autoscore.py                       # the long-running scorer (start detached, see README)
  python eval/autoscore/autoscore.py --test --only acp:pair_s2_v3_007250 --trials 5 --root autotest
"""
import argparse, csv, datetime, json, math, os, pathlib, re, shutil, subprocess, sys, threading, time, traceback

REPO = pathlib.Path(__file__).resolve().parents[2]
HERE = REPO / "eval" / "autoscore"
PY = sys.executable
HIKING_PY = os.path.expanduser("~/.venvs/sm/Scripts/python.exe")
FLAT = ["flat_walk", "flat_stop_stand", "flat_turn_in_place", "flat_turn_and_return"]
GATE_SEEDS, TURN_SEEDS = [1001, 1005, 1036, 1019, 1022], [1001, 1005, 1036]
FLOOR = "solid, trimmed"
TRACK = {
    "acp": dict(root="/mnt/filesystem-g4/tracks/acp/eval", need=["DONE"], adapter="acp:ACPHandover", label="ACP",
                env=dict(ACP_GOAL_R="0.2", ACP_LOOKAHEAD="0.3"), hdir_env="ACP_HANDOVER_DIR", setting="goal 0.20 + lookahead 0.30"),
    "ssr": dict(root="/mnt/filesystem-g4/tracks/ssr/eval/export", need=["policy.onnx", "adapter_card.json", "obs_check.npz"],
                adapter="ssr:SSRHandover", label="SSR", env=dict(SSR_CMD="standard"), hdir_env="SSR_HANDOVER_DIR", setting="standard command"),
    "hiking": dict(root="/mnt/filesystem-g4/tracks/hiking/eval", need=["policy.onnx", "adapter_card.json", "obs_check.npz", "0-depth_encoder.onnx"],
                   adapter="hiking:HikingHandover", label="Hiking", env=dict(HIKING_CMD="card"), hdir_env="HIKING_HANDOVER_DIR", setting="card command"),
}
BASE_ENV = dict(X2_EXCLUDE_PAIRS="hip", STAND_RULE="strict", END_RETRY="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                PYTHONIOENCODING="utf-8")


def now():
    return datetime.datetime.now(datetime.timezone.utc)


def log(*a):
    print(now().strftime("%m-%d %H:%M:%S"), *a, flush=True)


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return 100 * (c - h) / d, 100 * (c + h) / d


def names(track, name):
    """(local hand-over dir, eval/out sub-folder) using the names the earlier queue scripts used."""
    if track == "acp":
        m = re.match(r"pair_(s\d+)_(v\d+)_0*(\d+)$", name)
        return ("acp_%s%s_%s" % m.groups() if m else "acp_" + name), name
    if track == "ssr":
        m = re.match(r"(.*?)_0*(\d+)$", name)
        loc = "%s_%s" % m.groups() if m else name
        return "ssr_" + loc, loc
    m = re.match(r"x2_parkour_(v\d+)_(\d+)$", name)
    return ("hiking_%s_%s" % m.groups() if m else "hiking_" + name), name


def run_env(track, hdir):
    e = dict(os.environ)
    for k in ("PHYS_FLOOR", "PHYS_FLOOR_TRIM", "LADDER_REDO"):
        e.pop(k, None)
    e.update(BASE_ENV); e.update(TRACK[track]["env"]); e[TRACK[track]["hdir_env"]] = str(hdir)
    return e


def sh(cmd, timeout, **kw):
    cwd = kw.pop("cwd", REPO)
    try:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace", **kw)
    except subprocess.TimeoutExpired:
        return None


class Export:
    def __init__(self, sc, track, name):
        self.sc, self.track, self.name, self.id = sc, track, name, "%s:%s" % (track, name)
        loc, sub = names(track, name)
        self.hdir = REPO / "eval" / "data" / "handover" / loc
        self.out = REPO / "eval" / "out" / track / sub / sc.root
        self.jobs = {}            # job -> "running" | "done" | "failed"
        self.started = time.time()
        self.notes = []
        o = sc.cfg.get("export_options", {}).get(self.id, {})
        self.order = o.get("order", ["gate", "flat", "leg1", "leg2", "leg3", "leg4", "leg5"])
        self.label = "%s %s, %s, strict, solid floor (trimmed), autoscore" % (TRACK[track]["label"], name, TRACK[track]["setting"])

    # ---- what is finished on disk (so a restart resumes) ----
    def on_disk(self, job):
        n = self.sc.trials
        if job == "fetch":
            f = self.hdir / "autoscore_fetch.json"
            return f.exists() and json.load(open(f)).get("ok")
        if job == "gate":
            return all((self.out / "gate" / ("s%d" % s) / "results.json").exists() for s in GATE_SEEDS)
        if job.startswith("flat_"):
            f = self.out / "flat" / "step1" / job / "results.json"
            return (self.out / "flat" / "step1" / job / "summary.json").exists() and len(json.load(open(f))) >= n
        if job.startswith("leg"):
            return (self.out / "legs" / job / "summary.json").exists()
        if job.startswith("route_"):
            return (self.out / "route" / ("s" + job[6:]) / "summary.json").exists()
        return False

    def fetch_ok(self):
        f = self.hdir / "autoscore_fetch.json"
        return f.exists() and json.load(open(f)).get("ok")

    def gate_result(self):
        r = {}
        for s in GATE_SEEDS:
            f = self.out / "gate" / ("s%d" % s) / "results.json"
            if f.exists():
                x = json.load(open(f))[0]; r[s] = bool(x.get("strict_success"))
        ok = len(r) == 5 and all(r[s] for s in TURN_SEEDS) and sum(r.values()) >= 4
        return ok, "%d/5 (turn-around %d/3)" % (sum(r.values()), sum(r.get(s, 0) for s in TURN_SEEDS))

    def legs_rows(self):
        out = {}
        for k in range(1, 6):
            f = self.out / "legs" / ("leg%d" % k) / "summary.json"
            if f.exists():
                s = json.load(open(f)); out[k] = s
        return out

    def route_jobs(self):
        return ["route_%d" % s for s in self.sc.cfg.get("route_seeds", [1000, 1050, 1100, 1150])]

    def route_eligible(self):
        L = self.legs_rows()
        need = math.ceil(self.sc.cfg.get("route_min_leg", 45) * self.sc.trials / 50)
        return len(L) == 5 and all(L[k]["strict_successes"] >= need for k in L) and not self.sc.test

    def wanted(self):
        """Jobs this export still needs, in order, given what is finished (None = not decidable yet)."""
        J = ["fetch"]
        if not self.state("fetch"):
            return J
        if not self.fetch_ok():
            return []
        for o in self.order:
            if o == "gate" and self.track == "acp":
                J.append("gate")
            elif o == "flat":
                J += FLAT
            elif o.startswith("leg"):
                if self.track == "acp" and not (self.state("gate") and self.gate_result()[0]):
                    continue
                J.append(o)
        if all(self.state(j) for j in J) and self.route_eligible():
            J += self.route_jobs()
        return J

    def state(self, job):
        st = self.jobs.get(job)
        if st in ("done", "failed"):
            return st
        if st is None and self.on_disk(job):
            self.jobs[job] = "done"
            return "done"
        return None

    def ready_jobs(self):
        J = self.wanted()
        out = []
        for j in J:
            st = self.state(j)
            if st or self.jobs.get(j) == "running":
                continue
            if j == "fetch":
                return [j]
            if j.startswith("leg") and self.track == "acp" and not self.state("gate"):
                continue
            out.append(j)
        return out

    def finished(self):
        J = self.wanted()
        return bool(J) and all(self.state(j) for j in J) or (self.state("fetch") and not self.fetch_ok())

    # ---- running one job (in a worker thread) ----
    def loop(self, job, cmd, timeout, logf, env, tries=60, cwd=None):
        hard = 0
        for i in range(tries):
            if self.sc.stopping:
                return False
            with open(logf, "a", encoding="utf-8") as lf:
                lf.write("\n=== %s attempt %d %s\n" % (job, i + 1, now().isoformat())); lf.flush()
                try:
                    r = subprocess.run(cmd, cwd=cwd or REPO, env=env, stdout=lf, stderr=subprocess.STDOUT, timeout=timeout)
                    rc = r.returncode
                except subprocess.TimeoutExpired:
                    rc = 124
            if rc == 0:
                return True
            if rc not in (3, 124):
                hard += 1
                if hard >= 3:
                    return False
        return False

    def run_job(self, job):
        env = run_env(self.track, self.hdir)
        (self.out / "logs").mkdir(parents=True, exist_ok=True)
        logf = self.out / "logs" / (job + ".log")
        A, N = TRACK[self.track]["adapter"], str(self.sc.trials)
        if job == "fetch":
            return self.fetch(logf)
        if job == "gate":
            for s in GATE_SEEDS:
                o = self.out / "gate" / ("s%d" % s)
                if (o / "results.json").exists():
                    continue
                e = dict(env, EVAL_REGRESSION="0")
                self.loop(job, [PY, "eval/harness.py", "--adapter", A, "--apparatus", "ours", "--trials", "1", "--seed0", str(s), "--protocol",
                                "nominal", "--out", str(o), "--label", "%s hold-out gate seed %d" % (self.name, s)], 900, logf, e, tries=2)
            return self.on_disk("gate")
        if job.startswith("flat_"):
            return self.loop(job, [PY, "eval/ladder.py", "--adapter", A, "--step", "1", "--trials", N, "--out", str(self.out / "flat"),
                                   "--tests", job, "--label", self.label], 570, logf, env)
        if job.startswith("leg"):
            return self.loop(job, [PY, "eval/legs.py", "--adapter", A, "--out", str(self.out / "legs"), "--legs", job[3:], "--trials", N,
                                   "--wall-limit", "520", "--label", self.label], 590, logf, env)
        if job.startswith("route_"):
            return self.loop(job, [PY, "eval/harness.py", "--adapter", A, "--apparatus", "ours", "--trials", "50", "--seed0", job[6:], "--protocol",
                                   "nominal", "--out", str(self.out / "route" / ("s" + job[6:])), "--wall-limit", "520", "--label", self.label],
                             590, logf, env)
        raise ValueError(job)

    def fetch(self, logf):
        t, d = TRACK[self.track], self.hdir
        d.mkdir(parents=True, exist_ok=True)
        rem = "%s/%s" % (t["root"], self.name)
        scp = ('T=$(bash eval/node.sh target %s); K="-i $HOME/.ssh/nebius_corpus -o IdentitiesOnly=yes -o BatchMode=yes"; '
               'timeout 300 scp -q $K "$T:%s/*" "%s/" 2>/dev/null; ' % (self.track, rem, d.as_posix()))
        if self.track == "acp":
            (d / "mlp").mkdir(exist_ok=True)
            scp += 'timeout 120 scp -q $K "$T:%s/pair_eval_*/obs_check_pair.npz" "%s/mlp/" 2>/dev/null; ' % (rem, d.as_posix())
        scp += "true"
        sh(["bash", "-c", scp], 700)
        if self.track == "hiking" and not (d / "run_params").exists():
            shutil.copytree(REPO / self.sc.cfg["hiking_run_params"], d / "run_params")
        env = run_env(self.track, d)
        if self.track == "acp":
            r = sh([PY, str(REPO / "eval" / "tools" / "acp_handover_check.py"), "."], 600, cwd=d, env=env)
            tail = r.stdout.strip().splitlines()[-3:] if r else ["TIMEOUT"]
            ok = bool(tail) and "ALL PASS" in tail[-1]
        elif self.track == "ssr":
            r = sh([PY, "eval/tools/ssr_check.py", str(d)], 900, env=env)
            tail = r.stdout.strip().splitlines()[-3:] if r else ["TIMEOUT"]
            ok = bool(tail) and tail[-1].startswith("PASS")
        else:
            r = sh([HIKING_PY, "eval/tools/hiking_check.py", str(d), str(REPO / self.sc.cfg["hiking_port"])], 900, env=env)
            tail = r.stdout.strip().splitlines()[-10:] if r else ["TIMEOUT"]
            ok = any(x.startswith("PASS") for x in tail)
        (d / "check_result.txt").write_text("\n".join(tail) + "\n", encoding="utf-8")
        md5 = ""
        if self.track == "acp" and (d / "planner_attn.onnx").exists():
            import hashlib
            md5 = hashlib.md5(open(d / "planner_attn.onnx", "rb").read()).hexdigest()[:8]
        json.dump(dict(ok=ok, when=now().isoformat(), check=tail[-1] if tail else "", planner_md5=md5,
                       files=sorted(p.name for p in d.iterdir())), open(d / "autoscore_fetch.json", "w"), indent=1)
        with open(logf, "a", encoding="utf-8") as lf:
            lf.write("fetch %s: check %s\n%s\n" % (self.id, "PASS" if ok else "FAILED", "\n".join(tail)))
        return True

    # ---- the row ----
    def collect(self):
        env = run_env(self.track, self.hdir)
        row = dict(track=self.track, checkpoint=self.name, floor=FLOOR, trials_per_test=self.sc.trials)
        flags = []
        if not self.fetch_ok():
            f = self.hdir / "autoscore_fetch.json"
            chk = json.load(open(f)).get("check", "") if f.exists() else "no fetch"
            flags.append("hand-over check FAILED: %s" % chk[:80])
        if (self.out / "flat" / "step1").exists():
            sh([PY, "eval/ladder.py", "--collect", "--step", "1", "--out", str(self.out / "flat")], 300, env=env)
        allflat = True
        for t in FLAT:
            f = self.out / "flat" / "step1" / t / "summary.json"
            if f.exists():
                s = json.load(open(f)); k = s.get("strict_successes", s.get("successes")); n = s["trials"]
                row[t] = "%d/%d" % (k, n); allflat &= k == n
            else:
                row[t] = ""; allflat = False
        row["flat_gate"] = ("PASSED" if allflat else "NOT PASSED") if any(row[t] for t in FLAT) else ""
        if self.track == "acp" and self.state("gate"):
            ok, txt = self.gate_result(); row["gate"] = ("PASS " if ok else "FAIL ") + txt
        else:
            row["gate"] = "" if self.track != "acp" else "not run"
        L = self.legs_rows()
        if len(L) == 5:
            sh([PY, "eval/legs.py", "--adapter", TRACK[self.track]["adapter"], "--out", str(self.out / "legs"), "--trials", str(self.sc.trials),
                "--label", self.label], 600, env=env)
        prod = 1.0
        for k in range(1, 6):
            s = L.get(k)
            if s:
                row["leg%d" % k] = "%d/%d" % (s["strict_successes"], s["trials"]); prod *= s["strict_successes"] / s["trials"]
            else:
                row["leg%d" % k] = ""
        row["product"] = "%.3f" % prod if len(L) == 5 else ""
        ft = {"trials_with_joint_limit": 0, "trials_with_trapped_foot": 0, "trials_with_limb_clamp": 0}; nt = 0
        secs, rk, rn, outc = {}, 0, 0, {}
        for s in list(L.values()) + [json.load(open(self.out / "route" / ("s" + j[6:]) / "summary.json")) for j in self.route_jobs()
                                     if self.state(j) == "done"]:
            nt += s["trials"]
            for key in ft:
                ft[key] += (s.get("flags") or {}).get(key, 0)
        for j in self.route_jobs():
            if self.state(j) == "done":
                s = json.load(open(self.out / "route" / ("s" + j[6:]) / "summary.json"))
                rk += s["strict_successes"]; rn += s["trials"]
                for a, x in (s.get("sections") or {}).items():
                    c = secs.setdefault(a, [0, 0]); c[0] += x["entered"]; c[1] += x["completed"]
                for a, x in (s.get("outcomes") or {}).items():
                    outc[a] = outc.get(a, 0) + x
        if rn:
            lo, hi = wilson(rk, rn)
            row["route"] = "%d/%d" % (rk, rn); row["route_ci95"] = "%.0f-%.0f%%" % (lo, hi)
            row["route_sections"] = " ".join("%s %d/%d" % (a, c[1], c[0]) for a, c in secs.items())
        else:
            row["route"] = "not run" if len(L) == 5 else ""; row["route_ci95"] = row["route_sections"] = ""
        if nt:
            flags.append("legs+route %d trials: joint-limit %d, trapped-foot %d, limb-clamp %d" % (
                nt, ft["trials_with_joint_limit"], ft["trials_with_trapped_foot"], ft["trials_with_limb_clamp"]))
        failed = [j for j, s in self.jobs.items() if s == "failed"]
        if failed:
            flags.append("FAILED jobs: " + " ".join(failed))
        if self.track == "acp" and self.state("gate") and not self.gate_result()[0]:
            flags.append("legs not run (hold-out gate failed)")
        row["flags"] = "; ".join(flags)
        row["finished_utc"] = now().strftime("%Y-%m-%d %H:%M")
        row["minutes"] = "%.0f" % ((time.time() - self.started) / 60)
        row["out"] = self.out.relative_to(REPO).as_posix()
        row["_route"] = (rk, rn, secs, outc)
        return row


COLS = ["track", "checkpoint", "floor", "trials_per_test"] + FLAT + ["flat_gate", "gate", "leg1", "leg2", "leg3", "leg4", "leg5", "product",
                                                                     "route", "route_ci95", "route_sections", "flags", "finished_utc", "minutes", "out"]


class Scorer:
    def __init__(self, a):
        self.a, self.test, self.root, self.stopping = a, a.test, a.root, False
        self.cfg = json.load(open(HERE / "config.json"))
        self.trials = a.trials or self.cfg.get("trials", 50)
        self.statef = HERE / ("state_test.json" if self.test else "state.json")
        self.csvf = HERE / ("results_test.csv" if self.test else "results.csv")
        self.st = json.load(open(self.statef)) if self.statef.exists() else dict(done={}, active=[], seen={})
        self.exports = {}
        self.threads = {}          # (id, job) -> thread
        self.lock = threading.Lock()
        self.last = []

    def save(self):
        self.st["active"] = list(self.exports)
        json.dump(self.st, open(self.statef, "w"), indent=1)
        run = ["%s %s" % (k[0], k[1]) for k in self.threads]
        json.dump(dict(updated=now().strftime("%Y-%m-%dT%H:%MZ"), pid=os.getpid(), running=run, active=list(self.exports),
                       done=list(self.st["done"])[-10:], last_rows=self.last[-5:]),
                  open(HERE / ("status_test.json" if self.test else "status.json"), "w"), indent=1)

    def heartbeat(self):
        if self.test:
            return
        run = ", ".join("%s %s" % k for k in list(self.threads)[:6]) or "idle"
        res = "; ".join(self.last[-3:]) or "see eval/autoscore/results.csv"
        sh(["bash", "eval/tools/heartbeat.sh", "autoscore running (eval/autoscore, pid %d, no LLM)" % os.getpid(),
            "since %s: %s" % (self.st.get("since", "?"), run), res[:400]], 90)

    def discover(self):
        found = {}
        for tr, tc in self.cfg["tracks"].items():
            if not tc.get("enabled", True):
                continue
            cmd = ('echo "NOW|$(date +%%s)"; cd %s && for d in %s; do [ -d "$d" ] || continue; echo "$d|$(ls -1 "$d" | tr \'\\n\' \' \')|'
                   '$(stat -c %%Y "$d"/* 2>/dev/null | sort -n | tail -1)"; done' % (TRACK[tr]["root"], tc.get("pattern", "*")))
            r = sh(["bash", "eval/node.sh", "run", tr, cmd], 90)
            if r is None or r.returncode != 0:
                log("discover %s: node not reachable" % tr); continue
            lines = r.stdout.strip().splitlines()
            nowr = int(lines[0].split("|")[1]) if lines and lines[0].startswith("NOW|") else int(time.time())
            for ln in lines[1:]:
                p = ln.split("|")
                if len(p) < 3 or not p[2].strip():
                    continue
                files = set(p[1].split())
                ok = all(f in files for f in TRACK[tr]["need"]) and nowr - int(p[2]) > 120
                if ok:
                    found["%s:%s" % (tr, p[0])] = int(p[2])
        self.st["seen"].update(found)
        return found

    def pick(self, found):
        """Activate exports: priority entries, then per track the newest unscored one (policy newest) or all (policy all)."""
        done, skip = self.st["done"], set()
        for tr, tc in self.cfg["tracks"].items():
            skip |= {"%s:%s" % (tr, n) for n in tc.get("skip", [])}
        for tr, tc in self.cfg["tracks"].items():
            mine = sorted([i for i in found if i.startswith(tr + ":")], key=lambda i: -found[i])
            pri = ["%s:%s" % (tr, n) for n in tc.get("priority", [])]
            for i in pri:
                if i in found and i not in done and i not in self.exports:
                    self.activate(i)
            if tc.get("policy", "newest") == "all":
                for i in mine:
                    if i not in done and i not in skip and i not in self.exports:
                        self.activate(i)
            else:
                busy = [i for i in self.exports if i.startswith(tr + ":") and i.split(":", 1)[1] not in tc.get("priority", [])]
                cand = [i for i in mine if i not in skip]
                if not busy and cand and cand[0] not in done and cand[0] not in self.exports:
                    self.activate(cand[0])

    def activate(self, i):
        tr, n = i.split(":", 1)
        self.exports[i] = Export(self, tr, n)
        log("activate %s" % i)

    def fill(self):
        cores = self.cfg.get("max_cores", 3)
        order = sorted(self.exports.values(), key=lambda e: (0 if e.name in self.cfg["tracks"].get(e.track, {}).get("priority", []) else 1, e.started))
        while len(self.threads) < cores:
            nxt = None
            for e in order:
                r = e.ready_jobs()
                if r:
                    nxt = (e, r[0]); break
            if not nxt:
                return
            e, j = nxt
            e.jobs[j] = "running"
            t = threading.Thread(target=self.worker, args=(e, j), daemon=True)
            self.threads[(e.id, j)] = t
            log("start %s %s" % (e.id, j))
            t.start()

    def worker(self, e, j):
        try:
            ok = e.run_job(j)
        except Exception:
            log("job %s %s crashed:\n%s" % (e.id, j, traceback.format_exc())); ok = False
        with self.lock:
            e.jobs[j] = "done" if ok and e.on_disk(j) else "failed"
            if self.stopping:
                e.jobs.pop(j, None)
            log("end %s %s: %s" % (e.id, j, e.jobs.get(j)))
            self.threads.pop((e.id, j), None)

    def finish(self, e):
        row = e.collect()
        rk, rn, secs, outc = row.pop("_route")
        new = not self.csvf.exists()
        with open(self.csvf, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLS)
            if new:
                w.writeheader()
            w.writerow({c: row.get(c, "") for c in COLS})
        s = "%s %s flat %s, gate %s, legs %s (product %s), route %s" % (e.track, e.name, row["flat_gate"] or "-", row["gate"] or "-",
             "/".join(row["leg%d" % k].split("/")[0] or "-" for k in range(1, 6)), row["product"] or "-", row["route"] or "-")
        self.last.append(s); log("ROW " + s)
        self.st["done"][e.id] = row["finished_utc"]
        if rn and not self.test:
            self.dashboard(e, row, rk, rn, secs, outc)

    def dashboard(self, e, row, rk, rn, secs, outc):
        f = REPO / "dashboard" / "data" / "apparatus_results.json"
        rows = json.load(open(f, encoding="utf-8"))
        rid = "auto-%s-%s-trim" % (e.track, e.name.replace("_", "-"))
        lo, hi = wilson(rk, rn)
        r = dict(id=rid, date=now().strftime("%Y-%m-%d"), track=TRACK[e.track]["label"],
                 checkpoint="%s; strict stand check; solid floor (trimmed); automatic scorer" % e.name, apparatus="ours",
                 simulator="MuJoCo CPU", trials=rn, success_pct=round(100.0 * rk / rn, 1),
                 note="%d of %d (%.0f-%.0f%%). Outcomes %s. Sections %s. Station legs %s (product %s). Automatic scorer, not yet reviewed." % (
                     rk, rn, lo, hi, ", ".join("%s %d" % kv for kv in outc.items()), row["route_sections"],
                     " / ".join(row["leg%d" % k].split("/")[0] for k in range(1, 6)), row["product"]))
        rows = [x for x in rows if x["id"] != rid] + [r]
        json.dump(rows, open(f, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        for i in range(6):
            p = sh([PY, "prepare.py"], 300, cwd=REPO / "dashboard")
            if p is not None and p.returncode == 0:
                log("dashboard row %s written, prepare.py ok" % rid); return
            txt = (p.stdout + p.stderr) if p else "timeout"
            log("prepare.py failed (try %d): %s" % (i + 1, txt.strip()[-300:]))
            if p is not None and "OSError" not in txt and "PermissionError" not in txt:
                return
            time.sleep(20)

    def main(self):
        self.st.setdefault("since", now().strftime("%Y-%m-%d %H:%MZ"))
        log("autoscore start pid %d test=%s root=%s trials=%d" % (os.getpid(), self.test, self.root, self.trials))
        if self.test:
            for i in self.a.only:
                self.activate(i)
        else:
            for i in self.st.get("active", []):
                if i not in self.st["done"]:
                    self.activate(i)
        next_poll, next_hb = 0, 0
        while True:
            if (HERE / "STOP").exists() and not self.stopping:
                log("STOP file found: no new jobs; waiting for running jobs to end their slice"); self.stopping = True
            if self.stopping and not self.threads:
                break
            try:
                self.cfg = json.load(open(HERE / "config.json"))
            except Exception as ex:
                log("config.json unreadable, keeping the last one: %s" % ex)
            if not self.test and time.time() >= next_poll and not self.stopping:
                self.pick(self.discover())
                next_poll = time.time() + 60 * self.cfg.get("poll_minutes", 10)
            with self.lock:
                for i, e in list(self.exports.items()):
                    if not any(k[0] == i for k in self.threads) and e.finished():
                        self.finish(e); del self.exports[i]
                if not self.stopping:
                    self.fill()
                self.save()
            if self.test and not self.exports:
                break
            if time.time() >= next_hb:
                self.heartbeat(); next_hb = time.time() + 600
            time.sleep(15)
        self.save(); self.heartbeat() if not self.test else None
        log("autoscore exit")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="score only --only exports, write results_test.csv, no dashboard, no route")
    ap.add_argument("--only", nargs="*", default=[])
    ap.add_argument("--trials", type=int, default=0)
    ap.add_argument("--root", default="auto", help="sub-folder of eval/out/<track>/<checkpoint>/ for this scorer's runs")
    a = ap.parse_args()
    Scorer(a).main()
