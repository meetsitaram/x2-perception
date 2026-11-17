"""A1 parity: the deploy core (deploy/acp/acp_deploy/core.py) against the eval's ACP + fine student (eval/adapters/acp_depth.py)
on the same simulator states. The core gets only what the robot has: the torso's orientation and gyro (the IMU), the joints,
the same depth frame and goal; the eval reads the pelvis and the camera pose from the simulator directly.

    PA_ROBOT_DIR=... ACP_HANDOVER_DIR=... ACPD_STUDENT=.../fine_v3/student.pt python3 deploy/acp/tests/test_a1_parity.py [seconds]

Done = camera pose error < 1e-5, scan error < 1e-4 m, action error < 1e-4.
"""
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "eval"))
sys.path.insert(0, os.path.join(REPO, "deploy", "acp"))
os.environ.setdefault("EVAL_REGRESSION", "0")
os.environ["ACPD_MODE"] = "student"
os.environ["ACPD_DEPTH_NOISE"] = "0"
os.environ["ACPD_ANCHOR"] = "0"

import harness as H  # noqa: E402
import x2 as X  # noqa: E402
from adapters.acp_depth import ACPDepth  # noqa: E402
from acp_deploy import core  # noqa: E402

WORST = dict(cam=0.0, plan=0.0, track=0.0, action=0.0, n=0)


class Checked(ACPDepth):
    """The eval adapter, with the deploy core run beside it on robot-available inputs every step."""

    def reset(self, view, rng):
        super().reset(view, rng)
        self.core = core.ACPCore(os.environ["ACP_HANDOVER_DIR"], os.environ["ACPD_STUDENT"])
        self.k_core = 0
        self.last_z = None

    def _estimate(self, z, view):
        self.last_z = z
        super()._estimate(z, view)

    def act(self, view, cmd):
        m, d = view.m, view.d
        self.last_z = None
        mujoco.mj_forward(m, d)                                # body poses consistent with qpos (mj_step leaves them a step behind)
        a = super().act(view, cmd)                             # the eval: pelvis + camera from the simulator
        tb = m.body("torso_link").id
        v6 = np.zeros(6)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_XBODY, tb, v6, 1)
        names = X.joint_names_of(X.robot_dir() / "mjcf" / "x2_ultra.xml") if not hasattr(self, "_names") else self._names
        self._names = names
        qn, qdn = dict(zip(names, view.q())), dict(zip(names, view.qd()))
        q_core, info = self.core.step(d.xmat[tb].reshape(3, 3).copy(), v6[:3].copy(), qn, qdn, self.goal_now,
                                      depth=self.last_z, cam_pitch=self.cam_off[0], cam_roll=self.cam_off[1])
        if self.last_z is not None:                            # the camera pose on frames the student ran
            ct = core.cam_T(info["R_pelvis"], [qn[n] for n in core.WAIST], *self.cam_off)
            e = float(np.abs(ct - self._cam_T(view)).max())
            WORST["cam"] = max(WORST["cam"], e)
        WORST["plan"] = max(WORST["plan"], float(np.abs(info["plan"] - self.est[187]).max()))
        WORST["track"] = max(WORST["track"], float(np.abs(info["track"] - self.est[63]).max()))
        WORST["action"] = max(WORST["action"], float(np.abs(info["action"] - self.debug["action"]).max()))
        WORST["n"] += 1
        return a


def main(seconds=12.0):
    ad = Checked()
    app = H.load_course("ours") if hasattr(H, "load_course") else None
    if app is None:
        from ladder import load_course
        app = load_course("ours")
    x = X.build(app, ad.plant(X.joint_names_of(X.robot_dir() / "mjcf" / "x2_ultra.xml")))
    tr = H.Trial(x, app, ad, x.model.body_mass.copy(), x.model.body_inertia.copy())
    r = tr.run(2007, H.RANDOM, seconds=seconds, record=False)
    print("[A1] %d steps, outcome %s, waypoints %s" % (WORST["n"], r.get("reason"), r.get("waypoints")))
    print("[A1] max error: camera pose %.2e  planner scan %.2e m  tracker scan %.2e m  action %.2e"
          % (WORST["cam"], WORST["plan"], WORST["track"], WORST["action"]))
    ok = WORST["cam"] < 1e-5 and WORST["plan"] < 1e-4 and WORST["track"] < 1e-4 and WORST["action"] < 1e-4
    print("A1 PARITY:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 12.0)
