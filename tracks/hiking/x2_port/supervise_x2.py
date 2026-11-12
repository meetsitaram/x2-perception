"""Unattended training on the hiking node (Linux). One GPU job; stall detection; auto-resume from the newest checkpoint; heartbeat.

  python supervise_x2.py --task Instinct-Parkour-Target-Amp-X2-v0 --name x2_parkour --total 30000

Run directory: /mnt/filesystem-g4/tracks/hiking/runs/<name>/ (one timestamped folder per start or resume, checkpoints model_<it>.pt).
Supervisor files: runs/<name>/_supervisor/{supervisor.log, train.log, STOP}. Stop: touch runs/<name>/_supervisor/STOP.
Heartbeat: /mnt/filesystem-g4/status/hiking.json, rewritten every 10 s.
"""
import argparse
import glob
import json
import os
import pathlib
import re
import signal
import subprocess
import sys
import time

T = pathlib.Path("/mnt/filesystem-g4/tracks/hiking")
CODE = T / "code/hiking-in-the-wild"
PY = "/home/NODE_USER/hiking-env/bin/python"
HEART = pathlib.Path("/mnt/filesystem-g4/status/hiking.json")

ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--total", type=int, default=30000)
ap.add_argument("--num_envs", type=int, default=None)
ap.add_argument("--save_interval", type=int, default=250)
ap.add_argument("--stall_s", type=int, default=900)
ap.add_argument("--extra", nargs="*", default=[])
A = ap.parse_args()

RUN = T / "runs" / A.name
OUT = RUN / "_supervisor"
OUT.mkdir(parents=True, exist_ok=True)
STOP = OUT / "STOP"
ITER = re.compile(rb"Learning iteration (\d+)/(\d+)")
KEYS = {
    "reward": re.compile(rb"Mean reward:\s+(-?[\d.]+)"),
    "episode_length": re.compile(rb"Mean episode length:\s+(-?[\d.]+)"),
    "terrain_level": re.compile(rb"Curriculum/terrain_levels:\s+(-?[\d.]+)"),
}


def log(m):
    line = "SUPERVISOR %s %s" % (time.strftime("%m-%d %H:%M:%S"), m)
    print(line, flush=True)
    with open(OUT / "supervisor.log", "a") as f:
        f.write(line + "\n")


def gpu():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=30)
        u, t, util = [float(x) for x in r.stdout.strip().splitlines()[0].split(",")]
        return round(u / 1024, 1), round(t / 1024, 1), util
    except Exception:
        return None


def other_gpu_jobs(own_pid=None):
    """Other compute processes on the GPU (one GPU job at a time)."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=30)
        pids = [int(l.split(",")[0]) for l in r.stdout.strip().splitlines() if l.strip()]
    except Exception:
        return []
    return [p for p in pids if p != own_pid]


def latest():
    best = (-1, None)
    for f in glob.glob(str(RUN / "*" / "model_*.pt")):
        m = re.search(r"model_(\d+)\.pt$", f)
        if m and int(m.group(1)) > best[0] and os.path.getsize(f) > 1000:
            best = (int(m.group(1)), f)
    return best


def tail_stats(path):
    """Newest iteration number and metrics in the training log."""
    out = {}
    try:
        with open(path, "rb") as f:
            f.seek(max(0, os.path.getsize(path) - 60000))
            b = f.read()
    except OSError:
        return out
    m = ITER.findall(b)
    if m:
        out["iteration"] = int(m[-1][0])
    for k, rx in KEYS.items():
        m = rx.findall(b)
        if m:
            out[k] = float(m[-1])
    return out


def kill(pr):
    try:
        os.killpg(pr.pid, signal.SIGTERM)
        time.sleep(10)
        os.killpg(pr.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        pr.wait(timeout=60)
    except Exception:
        pass


def heartbeat(state, pr=None, extra=None):
    it, ck = latest()
    d = dict(track="hiking", t=time.time(), when=time.strftime("%Y-%m-%d %H:%M:%S"), state=state, task=A.task, run=str(RUN),
             total_iterations=A.total, checkpoint_iteration=it, checkpoint=ck, pid=pr.pid if pr else None, gpu=gpu())
    d.update(extra or {})
    tmp = HEART.with_suffix(".tmp")
    tmp.write_text(json.dumps(d))
    os.replace(tmp, HEART)


def housekeeping(path, pr, last):
    """Every 60 s a GPU row; every 5 min the metrics file; every 30 min the environment freeze and a backup manifest.
    Everything a restart needs is written straight to the shared filesystem; this records it and keeps the derived files fresh."""
    now = time.time()
    if now - last["gpu"] >= 60:
        last["gpu"] = now
        g, st = gpu(), tail_stats(path)
        new = not (RUN / "gpu.csv").exists()
        with open(RUN / "gpu.csv", "a") as f:
            if new:
                f.write("unix_time,iteration,gpu_mem_used_gb,gpu_mem_total_gb,gpu_util_percent,train_cpu_percent\n")
            try:
                cpu = subprocess.run(["ps", "-o", "%cpu=", "-p", str(pr.pid)], capture_output=True, text=True, timeout=10).stdout.strip()
            except Exception:
                cpu = ""
            f.write("%d,%s,%s,%s,%s,%s\n" % (now, st.get("iteration", ""), *(g or ("", "", "")), cpu))
    if now - last["metrics"] >= 300:
        last["metrics"] = now
        try:
            subprocess.run(["python3", str(T / "code/x2_port/log_metrics.py"), str(path), str(RUN / "metrics.jsonl")], timeout=240,
                           capture_output=True)
        except Exception as e:
            log("metrics parse failed %r" % (e,))
    if now - last["backup"] >= 1800:
        last["backup"] = now
        try:
            fz = subprocess.run([PY, "-m", "pip", "freeze"], capture_output=True, text=True, timeout=120).stdout
            if fz:
                (T / "env/freeze.txt").write_text(fz)
            cks = sorted(glob.glob(str(RUN / "*" / "model_*.pt")), key=os.path.getmtime)
            lines = ["backup manifest %s (everything below is on the shared filesystem)" % time.strftime("%Y-%m-%d %H:%M:%S"),
                     "checkpoints: %d, newest %s" % (len(cks), cks[-1] if cks else "none")]
            lines += ["  %s %d bytes" % (c, os.path.getsize(c)) for c in cks[-5:]]
            for d in ("code/x2_port", "env", "runs/_bench", "runs/%s/_supervisor" % A.name):
                n = sum(len(fs) for _, _, fs in os.walk(T / d))
                lines.append("%s: %d files" % (d, n))
            (T / "backup_manifest.txt").write_text("\n".join(lines) + "\n")
        except Exception as e:
            log("backup manifest failed %r" % (e,))


def main():
    STOP.unlink(missing_ok=True)
    (OUT / "supervisor.pid").write_text(str(os.getpid()))
    log("supervisor start pid %d task %s total %d" % (os.getpid(), A.task, A.total))
    tries = 0
    while tries < 200 and not STOP.exists():
        while other_gpu_jobs() and not STOP.exists():
            log("waiting: another GPU job is running (pids %s)" % other_gpu_jobs())
            heartbeat("waiting for GPU")
            time.sleep(60)
        it, ck = latest()
        if it >= A.total:
            log("DONE at iteration %d" % it)
            heartbeat("done")
            return
        args = [PY, str(CODE / "run_train.py"), "--headless", "--task=" + A.task, "--logroot", str(RUN),
                "--max_iterations", str(A.total - max(it, 0))]
        if A.num_envs:
            args += ["--num_envs", str(A.num_envs)]
        if ck:
            args += ["--resume", "--load_run", os.path.dirname(ck), "--checkpoint", os.path.basename(ck)]
        args += ["agent.save_interval=%d" % A.save_interval] + A.extra
        path = OUT / "train.log"
        env = dict(os.environ, PYTHONUNBUFFERED="1", OMNI_KIT_ACCEPT_EULA="YES", HIKING_ITER_OFFSET=str(max(it, 0)))
        with open(path, "ab") as f:
            pr = subprocess.Popen(args, cwd=CODE, stdout=f, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        log("start pid %d from iteration %d: %s" % (pr.pid, max(it, 0), " ".join(args[1:])))
        seen, t_seen, res = -1, time.time(), None
        last = dict(gpu=0.0, metrics=time.time() - 240, backup=0.0)
        while res is None:
            time.sleep(10)
            code = pr.poll()
            s = path.stat().st_size
            if s != seen:
                seen, t_seen = s, time.time()
            heartbeat("training", pr, dict(log_age_s=round(time.time() - t_seen), num_envs=A.num_envs or 4096, **tail_stats(path)))
            housekeeping(path, pr, last)
            if code is not None:
                res = "exit %d" % code
            elif STOP.exists():
                kill(pr)
                res = "stop"
            elif time.time() - t_seen > A.stall_s:
                log("STALLED: no output for %d s" % A.stall_s)
                kill(pr)
                res = "stall"
        log("run ended: %s, checkpoint %d" % (res, latest()[0]))
        if latest()[0] >= A.total - 1:
            log("DONE")
            heartbeat("done")
            return
        tries += 1
        time.sleep(30)
    heartbeat("stopped")
    log("supervisor exit (STOP or too many tries)")


if __name__ == "__main__":
    sys.exit(main())
