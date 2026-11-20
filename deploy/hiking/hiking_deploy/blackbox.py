"""Black box for the deploy node, as SONIC's deploy_logger (gear-sonic agi_x2_deploy_onnx_ref): one folder per run, one CSV per
stream, a row every control step. Two differences, both from the 2026-11-19 ACP run whose logs died with a pulled plug: rows go
to a writer thread (the control step only appends to a queue, so a slow disk never delays it), and every second the files are
flushed AND fsync'd (SONIC's 1 Hz flush reaches only the OS cache; a power cut loses that). At most the last second is lost.

    run_<YYYYmmdd_HHMMSS>/tick.csv     t, state, step_ms, depth_age_ms, cmd vx vy wz, deadman, estop, assist_share
                          joint_pos.csv / joint_vel.csv / joint_effort.csv (measured torque) / target_pos.csv / kp.csv / kd.csv
                          imu.csv      t, torso quat (w x y z), gyro (x y z), pelvis gravity (x y z)
                          policy.csv   t + the policy's own channels (ACP: action, goal, contact targets, occupancy, scans)
"""
from __future__ import annotations

import os
import queue
import threading
import time


class BlackBox:
    def __init__(self, root, joints, policy_fields=()):
        self.dir = os.path.join(root, time.strftime("run_%Y%m%d_%H%M%S"))
        os.makedirs(self.dir, exist_ok=True)
        self.joints = list(joints)
        self.q = queue.Queue(maxsize=2000)          # 40 s at 50 Hz; full = the disk stalled: drop rows, never block control
        self.dropped = 0
        hdr = {
            "tick": "t,state,step_ms,depth_age_ms,vx,vy,wz,deadman,estop,assist_share",
            "joint_pos": "t," + ",".join(self.joints), "joint_vel": "t," + ",".join(self.joints),
            "joint_effort": "t," + ",".join(self.joints),
            "target_pos": "t," + ",".join(self.joints), "kp": "t," + ",".join(self.joints), "kd": "t," + ",".join(self.joints),
            "imu": "t,qw,qx,qy,qz,wx,wy,wz,gx,gy,gz",
            "policy": "t," + ",".join(policy_fields),
        }
        self.f = {}
        for k, h in hdr.items():
            self.f[k] = open(os.path.join(self.dir, k + ".csv"), "w", buffering=1 << 16)
            self.f[k].write(h + "\n")
        threading.Thread(target=self._run, daemon=True, name="blackbox").start()

    def log(self, rows):
        """rows: {stream: (t, values...)}; called from the control step, returns at once."""
        try:
            self.q.put_nowait(rows)
        except queue.Full:
            self.dropped += 1

    def _run(self):
        last = time.monotonic()
        while True:
            try:
                rows = self.q.get(timeout=0.2)
                for k, v in rows.items():
                    # the time column with fixed decimals: "%.6g" of a monotonic time past 10,000 s rounds it to 0.1 s
                    self.f[k].write(",".join(("%.4f" % x if c == 0 else "%.6g" % x) if isinstance(x, float) else str(x)
                                             for c, x in enumerate(v)) + "\n")
            except queue.Empty:
                pass
            now = time.monotonic()
            if now - last >= 1.0:
                last = now
                for f in self.f.values():
                    f.flush()
                    os.fsync(f.fileno())
