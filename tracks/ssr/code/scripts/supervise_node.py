"""Supervisor for the SSR track's node (Linux). Standard library only.

- One GPU job: refuses to start a training process while another compute process is on the GPU.
- Starts `python -m ssr.train`, resuming from runs/<name>/last.pt when it exists.
- A run whose log stops growing for --stall seconds is killed (whole process group) and resumed.
- Heartbeat every 30 s to /mnt/filesystem-g4/status/ssr.json. Stop with: touch <runs>/STOP

python scripts/supervise_node.py --name r1 --iters 20000 -- --rsi_frac 0.5
"""
import argparse
import glob
import json
import os
import pathlib
import signal
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
TRACK = ROOT.parent
RUNS = TRACK / "runs"
PY = str(TRACK / "env/venv/bin/python")
STATUS = pathlib.Path("/mnt/filesystem-g4/status/ssr.json")
sys.path.insert(0, "/mnt/filesystem-g4/tools")
from simcheck import alarm  # sim_fault_share in the heartbeat (TRACK_RULES)


def sh(cmd, timeout=30):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout.strip()
    except Exception as e:
        return 1, repr(e)


def gpu_procs():
    code, out = sh(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"])
    return None if code else [int(x) for x in out.split() if x.strip().isdigit()]


def last_rec(name):
    try:
        with open(RUNS / name / "log.jsonl", "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 20000))
            return json.loads(f.read().decode().strip().splitlines()[-1])
    except Exception:
        return {}


def newest_ckpt(name):
    c = sorted(glob.glob(str(RUNS / name / "ckpt_*.pt")))
    return c[-1] if c else ""


def heartbeat(**kw):
    tmp = STATUS.with_suffix(".tmp")
    tmp.write_text(json.dumps(dict(track="ssr", t=time.time(), when=time.strftime("%Y-%m-%d %H:%M:%S %Z"), **kw)))
    os.replace(tmp, STATUS)


def log(msg):
    line = "SUPERVISOR %s %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg)
    print(line, flush=True)
    with open(RUNS / "supervisor.log", "a") as f:
        f.write(line + "\n")


def kill(pr):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pr.pid, sig)
        except ProcessLookupError:
            return
        for _ in range(20):
            if pr.poll() is not None:
                return
            time.sleep(0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--num_envs", type=int, default=4096)
    ap.add_argument("--stall", type=int, default=600)
    ap.add_argument("--first_stall", type=int, default=1500, help="allowance before the first log line (kernel compilation)")
    ap.add_argument("extra", nargs="*")
    a = ap.parse_args()
    RUNS.mkdir(exist_ok=True)
    stop = RUNS / "STOP"
    stop.unlink(missing_ok=True)
    (RUNS / "supervisor.pid").write_text(str(os.getpid()))
    out_log = RUNS / f"{a.name}.log"
    restarts = 0
    while not stop.exists():
        procs = gpu_procs()
        if procs is None or procs:
            log("GPU not free or not answering (%r); waiting 60 s" % (procs,))
            heartbeat(run=a.name, state="waiting_for_gpu", gpu_procs=procs)
            time.sleep(60)
            continue
        last = RUNS / a.name / "last.pt"
        cmd = [PY, "-m", "ssr.train", "--num_envs", str(a.num_envs), "--iters", str(a.iters), "--name", a.name, "--out_root", str(RUNS), "--save_every", "50"] + a.extra
        if last.exists():
            cmd += ["--resume", str(last)]
        with open(out_log, "ab") as f:
            f.write(("\n=== %s start: %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), " ".join(cmd))).encode())
            f.flush()
            pr = subprocess.Popen(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, start_new_session=True,
                                  env=dict(os.environ, PYTHONUNBUFFERED="1"))
        log("start pid %d resume=%s restarts=%d" % (pr.pid, last.exists(), restarts))
        seen, seen_t, t0, first = out_log.stat().st_size, time.time(), time.time(), True
        res = None
        while res is None:
            time.sleep(30)
            size = out_log.stat().st_size
            if size != seen:
                if seen_t != t0:
                    first = False
                seen, seen_t = size, time.time()
            rec = last_rec(a.name)
            heartbeat(run=a.name, state="training", pid=pr.pid, iter=rec.get("it", -1), iters=a.iters, log_age_s=round(time.time() - seen_t),
                      ep_len_s=rec.get("ep_len_s"), r_l=rec.get("r_l"), r_f=rec.get("r_f"), r_s=rec.get("r_s"), style_by_group=rec.get("style_by_group"), omni_share=rec.get("omni_share"), critic_warmup=rec.get("critic_warmup"), lvl_mean=rec.get("lvl_mean"), gait=rec.get("gait"), restarts=restarts,
                      newest_ckpt=newest_ckpt(a.name), last=str(last), log=str(out_log),
                      **alarm.fields_from_shares(rec.get("done_share") or (rec.get("extra") or {}).get("done_share") or {}, sim_keys=("diverged",)))
            code = pr.poll()
            if code is not None:
                res = "done" if code == 0 else "crash(%d)" % code
            elif stop.exists():
                kill(pr)
                res = "stop"
            elif time.time() - seen_t > (a.first_stall if first else a.stall):
                log("STALLED: no log output for %d s; killing pid %d" % (time.time() - seen_t, pr.pid))
                kill(pr)
                res = "stall"
        log("ended: %s at iter %s" % (res, last_rec(a.name).get("it")))
        if res in ("done", "stop"):
            heartbeat(run=a.name, state=res, iter=last_rec(a.name).get("it", -1), newest_ckpt=newest_ckpt(a.name), last=str(last))
            return
        restarts += 1
        if restarts > 200:
            heartbeat(run=a.name, state="gave_up", restarts=restarts)
            return
        time.sleep(30)


if __name__ == "__main__":
    sys.exit(main())
