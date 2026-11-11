"""Checks of the scoring code that the sanity adapters do not reach, with the robot placed by hand in the physics.

Side hits (each must count, in its class):
  4x4-side     the robot is set down moving toes-first into the front 4x4 of our apparatus
  wedge-side   ... into the 14 cm side face between two crossing-ramp cells (ours, x 4.2, y 1.22)
  stair-riser  ... into the first stair riser (ours, x 8.62)
Clean set-downs (none may count a side hit; on the 4x4 and the tread only the landing, the first 0.3 s, is tested, because
the stand fixture cannot balance on a 9 cm beam or a 24 cm tread and what follows is a fall, not a clean stand):
  4x4-top      soles set down across the top of the 4x4 (an edge landing, not a side hit)
  wedge-slope  set down on a ramp's sloped walking surface
  stair-tread  set down on the first stair tread
  flat         set down on the open deck (landings, no edge, no side contact)
Other:
  arm          pushed sideways into the lane's side rail: arm or trunk contacts with a wall must be logged
  normals      a 2 cm sphere pressed into each kind of face of the height field: the contact normal MuJoCo reports

    python eval/selftest.py        (CPU, under a minute; prints PASS / FAIL per check)
The robot is slid or set down, not walked: there is no policy yet. A real swing-foot stub is covered when a policy runs.
"""
import math
import pathlib
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import harness as Hn  # noqa: E402
import x2 as X  # noqa: E402
from adapters.stand import Stand  # noqa: E402
from apparatus import Apparatus  # noqa: E402

app = Apparatus("ours")
x = X.build(app)
tr = Hn.Trial(x, app, Stand(), x.model.body_mass.copy(), x.model.body_inertia.copy())
PROTO = dict(Hn.NOMINAL, start_xy=0.0, start_yaw_deg=0.0)
TOE = 0.139 + X.SOLE_R            # toe tip ahead of the ankle (the ankle is 0.019 m behind the pelvis origin at the default pose)


def place(px, py, yaw, lift, v=0.0, seconds=1.5, keep=None):
    """Set the robot down at (px, py) facing yaw, soles `lift` above the ground, moving at v m/s along its heading (or
    sideways if v is a pair (forward, left)). Runs the stand fixture for `seconds` and returns the trial result."""
    tr.reset(0, PROTO)
    m, d = tr.m, tr.d
    d.qpos[:2] = [px, py]
    d.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    d.qpos[2] = 2.5
    mujoco.mj_forward(m, d)
    d.qpos[2] -= min(tr._sole_gap(b) for b in x.foot_ids) - lift
    d.time = 1.0                                           # past the set-down window, so landings count
    for b in x.foot_ids:
        tr.stance[b] = False
    vf, vl = (v, 0.0) if np.isscalar(v) else v
    d.qvel[0] = vf * math.cos(yaw) - vl * math.sin(yaw)
    d.qvel[1] = vf * math.sin(yaw) + vl * math.cos(yaw)
    mujoco.mj_forward(m, d)
    tr.h0 = float(d.qpos[2] - tr.ground(px, py))
    cmd = tr.command()
    for _ in range(int(seconds / 0.02)):
        d.ctrl[:] = tr.adapter.act(tr.view, cmd)
        for _ in range(tr.sub):
            mujoco.mj_step(m, d)
            tr._contacts()
        tr._disturbance()
        if keep is not None:
            keep["t"].append(d.time); keep["qpos"].append(d.qpos.astype(np.float32).copy()); keep["wp"].append(0)
            keep["cmd"].append(np.zeros(3)); keep["load"].append([tr.load[b] for b in x.foot_ids])
            keep["h"].append(float(d.qpos[2] - tr.ground(d.qpos[0], d.qpos[1]))); keep["hold"].append(False)
    return tr._result(0, tr.rand, "selftest")


def line(name, ok, r):
    h = r["side_hits"]
    c = r["contacts"]
    print("%s %-12s side hits %s, scuffs %s; penalty %.2f; landings %d (edge %d); arm %d trunk %d" % (
        "PASS" if ok else "FAIL", name, {k: v["foot"] + v["shin_knee"] for k, v in h.items()}, {k: v["scuffs"] for k, v in h.items()},
        r["stumble_penalty"], c["steps"], c["edge_landings"], c["arm"], c["trunk"]))
    for e in [e for e in r["events"] if e.get("hit")][:3]:
        print("        %s %s on the %s: normal z %s, peak horizontal %.0f N, impulse %.1f N s, pelvis dv %.2f m/s, tilt +%.1f deg, in swing %s -> penalty %.2f" % (
            e["side"], e["kind"], e["cls"], e.get("nz", "-"), e["peak_h"], e["impulse"], e["dv"], e["dtilt"], e.get("swing", "-"), e["penalty"]))
    return ok


def only(r, cls):
    h = r["side_hits"]
    return h[cls]["foot"] >= 1 and all(h[c]["foot"] + h[c]["shin_knee"] == 0 for c in Hn.CLASSES if c != cls)


def none(r):
    return r["side_hit_count"] == 0 and all(v["scuffs"] == 0 for v in r["side_hits"].values())


def normals():
    """A free 2 cm sphere pressed 3 mm into the height field at a point of each kind of face: |normal z| of the contact."""
    spec = mujoco.MjSpec()
    z = app.grid
    zmin, top = float(z.min()), float(z.max() - z.min())
    hf = spec.add_hfield(name="t", nrow=app.ny, ncol=app.nx, size=[(app.nx - 1) * app.res / 2, (app.ny - 1) * app.res / 2, top, 0.2])
    hf.userdata = ((z - zmin) / top).T.astype(np.float32).ravel().tolist()
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname="t", pos=[app.x0 + (app.nx - 1) * app.res / 2, app.y0 + (app.ny - 1) * app.res / 2, zmin])
    b = spec.worldbody.add_body(name="s")
    b.add_freejoint()
    b.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.02, 0, 0])
    m = spec.compile()
    d = mujoco.MjData(m)

    def probe(pts):
        out = []
        for px, py, pz in pts:
            d.qpos[:3] = [px, py, pz]
            mujoco.mj_forward(m, d)
            if d.ncon:
                k = int(np.argmin([d.contact[i].dist for i in range(d.ncon)]))
                out.append(abs(float(d.contact[k].frame[2])))
        return np.array(out)
    R, P = 0.02, 0.003
    ys = np.linspace(0.45, 0.9, 10)
    cases = [
        ("4x4 side face (8.9 cm), mid-height", [(3.02 - R + P + 0.004, y, 0.045) for y in ys], True),
        ("4x4 top", [(3.05, y, 0.089 + R - P) for y in ys], False),
        ("stair riser (18.6 cm), mid-height", [(8.615 - R + P + 0.004, y, 0.093) for y in np.linspace(0.3, 0.8, 10)], True),
        ("stair tread", [(8.72, y, 0.186 + R - P) for y in np.linspace(0.3, 0.8, 10)], False),
        ("wedge side face (14 cm), mid-height", [(xx, 1.215 - R + P + 0.004, float(app.height(xx, 1.14)) + 0.075) for xx in np.linspace(4.0, 4.6, 10)], True),
        ("wedge sloped surface (15 deg)", [(xx, 0.6, float(app.height(xx, 0.6)) + R - P) for xx in np.linspace(5.0, 5.6, 10)], False),
        ("flat deck", [(1.2 + 0.013 * i, 1.2, R - P) for i in range(10)], False),
    ]
    ok = True
    print("normals: |normal z| of a 2 cm sphere's contact with the 2 cm height field (0 = horizontal normal, 1 = vertical)")
    for name, pts, side in cases:
        nz = probe(pts)
        good = len(nz) > 0 and (nz.max() < Hn.SIDE_NZ if side else nz.min() > 0.9)
        ok &= good
        print("  %s %-38s %2d contacts, |nz| %.2f .. %.2f  (%s)" % ("PASS" if good else "FAIL", name, len(nz), nz.min() if len(nz) else -1,
              nz.max() if len(nz) else -1, "must be under %.1f: counts as a side face" % Hn.SIDE_NZ if side else "must be over 0.9: a walking surface"))
    return ok


if __name__ == "__main__":
    ok = True
    # side hits: set down 3 cm above the ground with the toes 3 cm short of the face, moving at it at 0.6 m/s, so the
    # toes meet the face while the feet are still coming down (as a swing foot would)
    r = place(3.02 - TOE - 0.03 + 0.019, 0.69, 0.0, 0.03, v=0.6)
    ok &= line("4x4-side", only(r, "4x4"), r)
    rec = {k: [] for k in ("t", "qpos", "wp", "cmd", "load", "h", "hold")} if "--save" in sys.argv else None
    r = place(8.615 - TOE - 0.03 + 0.019, 0.55, 0.0, 0.03, v=0.6, keep=rec)
    ok &= line("stair-riser", only(r, "stair"), r)
    if rec is not None:                                    # a run folder render.py can read: the marked side hit
        import json
        out = pathlib.Path(sys.argv[sys.argv.index("--save") + 1])
        (out / "traj").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / "traj" / "0.npz", t=np.array(rec["t"]), qpos=np.array(rec["qpos"]), wp=np.array(rec["wp"]), cmd=np.array(rec["cmd"]),
                            load=np.array(rec["load"], np.float32), h=np.array(rec["h"], np.float32), hold=np.array(rec["hold"]))
        json.dump([r], open(out / "results.json", "w"), indent=1)
        json.dump(dict(apparatus="ours", simulator="MuJoCo %s (CPU), dt %.4f s" % (mujoco.__version__, X.SIM_DT), adapter="selftest",
                       checkpoint="none (stand fixture set down moving toes-first into the first stair riser at 0.6 m/s: a test of the side-hit marker)",
                       protocol="self-test, not a result", renders={"selftest": 0}), open(out / "summary.json", "w"), indent=1)
    r = place(4.2, 1.215 - TOE - 0.03 + 0.019, math.pi / 2, 0.03, v=0.6)
    ok &= line("wedge-side", only(r, "wedge"), r)
    # clean set-downs
    r = place(3.05 - X.SOLE_X, 0.69, 0.0, 0.03, seconds=0.3)
    ok &= line("4x4-top", none(r) and r["contacts"]["edge_landings"] >= 1, r)
    r = place(5.3, 0.6, 0.0, 0.03)
    ok &= line("wedge-slope", none(r), r)
    # the tread is 22 cm deep and the X2's foot 24 cm long: set down with the toes 3 cm short of the next riser, heels overhanging
    r = place(8.665, 0.55, 0.0, 0.03, seconds=0.3)
    ok &= line("stair-tread", none(r), r)
    r = place(1.2, 1.2, 0.0, 0.03)
    ok &= line("flat", none(r) and r["contacts"]["steps"] >= 2 and r["contacts"]["edge_landings"] == 0 and r["contacts"]["foot_side_contacts"] == 0, r)
    r = place(2.0, 0.40, 0.0, 0.002, v=(0.0, -1.2), seconds=2.0)
    walls = [e for e in r["events"] if e.get("on") == "wall"]
    ok &= line("arm", len(walls) >= 1, r)
    ok &= normals()
    print("ALL PASS" if ok else "SOME FAILED")
    sys.exit(0 if ok else 1)
