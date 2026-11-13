"""simcheck.alarm: put the simulator-fault share of episode ends into a track's heartbeat, with a WARN over 1 %.

    fields(counts, total=None)            -> dict to merge into the heartbeat dict before it is written
    fields_from_shares(shares, sim_keys)  -> the same from a {cause: share of episode ends} dict (SSR's done_share)
    write(path, counts, total=None)       -> read-modify-write of an existing heartbeat JSON (atomic replace)

counts: {cause: number of episode ends}; the causes in SIM_KEYS (or those given) are simulator faults. total: all episode
ends in the same window (default: the sum of counts, which is right only when counts lists every cause once).
Heartbeat fields: sim_fault_share (0-1, 4 decimals), sim_fault_n, episode_ends_n, sim_fault_warn ("WARN ..." or null).
The coordinator's dashboard refresh reads sim_fault_warn; a WARN means: stop and look before the next checkpoint is trusted.
"""
from __future__ import annotations

import json
import os
import time

WARN_SHARE = 0.01
SIM_KEYS = ("sim_fault", "diverged", "bad", "nan", "qvel", "depth", "speed_jump")


def fields(counts: dict, total: float | None = None, sim_keys=SIM_KEYS, warn_share: float = WARN_SHARE) -> dict:
    sim = float(sum(float(v) for k, v in counts.items() if k in sim_keys))
    tot = float(total if total is not None else sum(float(v) for v in counts.values()))
    share = sim / tot if tot > 0 else 0.0
    warn = None
    if share > warn_share:
        warn = (f"WARN sim_fault_share {share:.1%} > {warn_share:.0%} of episode ends "
                f"({int(sim)} of {int(tot)}; {', '.join(f'{k} {int(v)}' for k, v in counts.items() if k in sim_keys and v)})")
    return dict(sim_fault_share=round(share, 4), sim_fault_n=int(sim), episode_ends_n=int(tot), sim_fault_warn=warn)


def fields_from_shares(shares: dict, sim_keys=SIM_KEYS, warn_share: float = WARN_SHARE) -> dict:
    share = float(sum(float(v) for k, v in (shares or {}).items() if k in sim_keys))
    warn = f"WARN sim_fault_share {share:.1%} > {warn_share:.0%} of episode ends" if share > warn_share else None
    return dict(sim_fault_share=round(share, 4), sim_fault_warn=warn)


def write(path: str, counts: dict, total: float | None = None, **kw) -> dict:
    try:
        d = json.load(open(path))
    except Exception:
        d = {}
    f = fields(counts, total, **kw)
    d.update(f, sim_fault_t=time.time())
    tmp = path + ".simcheck.tmp"
    with open(tmp, "w") as fh:
        json.dump(d, fh)
    os.replace(tmp, path)
    return f
