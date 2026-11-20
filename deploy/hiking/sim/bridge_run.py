"""Run the gwbc-x2 MuJoCo bridge unchanged (used at run time, not copied), with two additions:

1. Torso IMU in the body frame: the bridge's torso gyro comes from mj_objectVelocity(mjOBJ_BODY, local), the body's inertial
   frame (rotated on the X2); the real IMU and the Hiking adapter use the body frame (mjOBJ_XBODY). Local-frame calls are
   redirected to XBODY; world-frame calls are untouched.
2. Robot reset (sim only; the bridge has none, its scene_reset moves scene objects): a UDP datagram "reset" on 127.0.0.1:5691
   stands the robot up in its --init-pose where it fell (user: "reset only where it falls, or at least some flat terrain near
   that place"): same heading, on the nearest flat patch within 2.5 m (0.5 m square, floor within 1 cm, nothing taller),
   at rest, physics frozen until the next deploy command, as at start. The deploy node sends it after it has stopped
   commanding (pad: View / Back held 1 s). BRIDGE_SPAWN="x,y,yaw_deg" (env): the same reset once at start, at that place.

    python3 bridge_run.py /path/to/gwbc-x2/gear_sonic_deploy/scripts/x2_mujoco_ros_bridge.py [bridge args...]
"""
import importlib.util
import math
import os
import socket
import sys
import threading

import mujoco
import numpy as np

RESET_PORT = 5691
_object_velocity = mujoco.mj_objectVelocity


def _xbody_local(m, d, objtype, objid, res, flg_local):
    if flg_local and objtype == mujoco.mjtObj.mjOBJ_BODY:
        objtype = mujoco.mjtObj.mjOBJ_XBODY
    return _object_velocity(m, d, objtype, objid, res, flg_local)


mujoco.mj_objectVelocity = _xbody_local
_reset_requested = threading.Event()
_release_requested = threading.Event()             # "release": drop the gantry band (the deploy node sends it once its policy runs)
_spawn = [tuple(float(v) for v in os.environ["BRIDGE_SPAWN"].split(","))] if os.environ.get("BRIDGE_SPAWN") else []
if _spawn:
    _reset_requested.set()


def _listen():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", RESET_PORT))
    while True:
        msg = s.recv(64).strip()
        if msg == b"reset":
            _reset_requested.set()
        elif msg == b"release":
            _release_requested.set()


def _floor(m, d, x, y, groups):
    """Floor height under (x, y); None off the map (no hit, or only the catch floor: the scene's plane under the terrain)."""
    gid = np.zeros(1, np.int32)
    dist = mujoco.mj_ray(m, d, np.array([x, y, 5.0]), np.array([0.0, 0.0, -1.0]), groups, 1, -1, gid)
    if dist < 0 or (m.ngeom > 1 and m.geom_type[gid[0]] == mujoco.mjtGeom.mjGEOM_PLANE and m.nhfield > 0):
        return None
    return 5.0 - dist


def _flat_spot(m, d, x0, y0, radius=2.5, step=0.1, half=0.25, tol=0.01, clear=None):
    """Nearest (x, y, floor z) within radius whose 0.5 m square of floor (terrain group 4 and walls group 5) is flat and on the
    map; then within 15 m in coarser steps (fallen off the map: back onto its edge); else None (the caller uses the spawn)."""
    return (_search(m, d, x0, y0, radius, step, half, tol, clear=clear)
            or _search(m, d, x0, y0, 15.0, 0.25, half, tol, r0=radius, clear=clear))


def _search(m, d, x0, y0, radius, step, half, tol, r0=0.0, clear=None):
    terrain, both = np.zeros(6, np.uint8), np.zeros(6, np.uint8)
    terrain[4] = 1                      # the floor: terrain only (a wall or rail top is not a floor)
    both[4] = both[5] = 1               # anything standing on it (walls, props: group 5)
    offs = [(dx, dy) for dx in (-half, 0.0, half) for dy in (-half, 0.0, half)]
    for r in np.arange(r0, radius + 1e-9, step):
        n = max(1, int(round(2 * math.pi * r / step)))
        for k in range(n):
            a = 2 * math.pi * k / n
            x, y = x0 + r * math.cos(a), y0 + r * math.sin(a)
            z = [_floor(m, d, x + dx, y + dy, terrain) for dx, dy in offs]
            if None in z or max(z) - min(z) >= tol:
                continue
            top = [_floor(m, d, x + dx, y + dy, both) for dx, dy in offs]
            if all(t is not None and t - zz < 0.02 for t, zz in zip(top, z)):
                zc = float(np.median(z))
                if clear is None or clear(x, y, zc):
                    return x, y, zc
    return None


def _clearance_test(m, d, q0, quat, margin=0.15):
    """-> clear(x, y, floor_z): the robot in the stand pose there, and shifted by margin each way, touches nothing but the floor
    with its feet (no wall / prop contact, no shin or hand in the ground). Checked with MuJoCo's own collision on the real
    robot geometry (thin walls slipped between the old 0.25 m probe points; user 2026-11-15: "launched inside a wall")."""
    root = m.body("pelvis").id if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis") >= 0 else 1
    robot = {g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]] == m.body_rootid[root]}
    feet = {g for g in robot if m.body(m.geom_bodyid[g]).name.endswith("ankle_roll_link")}

    def ok_at(x, y, z):
        d.qpos[:] = q0
        d.qpos[0], d.qpos[1], d.qpos[2] = x, y, q0[2] + z + 0.01
        d.qpos[3:7] = quat
        mujoco.mj_kinematics(m, d)
        mujoco.mj_collision(m, d)
        for c in d.contact[:d.ncon]:
            g1, g2 = int(c.geom1), int(c.geom2)
            if (g1 in robot) == (g2 in robot):
                continue
            mine, other = (g1, g2) if g1 in robot else (g2, g1)
            if m.geom_group[other] != 4 or mine not in feet:     # a wall / prop, or a non-foot part in the ground
                return False
        return True

    def clear(x, y, z):
        return all(ok_at(x + dx, y + dy, z) for dx, dy in ((0, 0), (margin, 0), (-margin, 0), (0, margin), (0, -margin)))
    return clear


def _patch(cls):
    drain = cls._drain_pending_scene_reset

    def drain_and_reset(self):
        # runs on the sim thread once per cycle, frozen or stepping: the one safe place to write mj_data
        if _release_requested.is_set():
            _release_requested.clear()
            band = getattr(self, "elastic_band", None)
            if band is not None and band.enable:
                band.enable = False
                print("[bridge_run] gantry band released (the policy is running)", flush=True)
        if _reset_requested.is_set():
            _reset_requested.clear()
            x0, y0 = float(self.mj_data.qpos[0]), float(self.mj_data.qpos[1])
            w, qx, qy, qz = self.mj_data.qpos[3:7]
            yaw = math.atan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
            if _spawn:
                x0, y0, yaw = _spawn[0][0], _spawn[0][1], math.radians(_spawn[0][2])
                _spawn.clear()
            mujoco.mj_resetData(self.mj_model, self.mj_data)
            pose = str(getattr(self.args, "init_pose", "default")).replace("-", "_")
            if pose == "default":
                self._reset_to_default_pose()
            else:
                self._reset_to_named_pose(pose)
            q0 = self.mj_data.qpos.copy()                  # the stand pose at the origin, pelvis height above z = 0
            quat = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
            clear = _clearance_test(self.mj_model, self.mj_data, q0, quat)
            spot = _flat_spot(self.mj_model, self.mj_data, x0, y0, clear=clear) or (0.0, 0.0, 0.0)
            self.mj_data.qpos[:] = q0
            self.mj_data.qpos[0], self.mj_data.qpos[1] = spot[0], spot[1]
            self.mj_data.qpos[2] += spot[2]
            self.mj_data.qpos[3:7] = quat
            self.mj_data.ctrl[:] = 0.0
            mujoco.mj_forward(self.mj_model, self.mj_data)
            self._first_command_received = False        # frozen until the deploy commands again
            print("[bridge_run] reset to %r at (%.2f, %.2f, floor %.2f), yaw %.0f deg, %.2f m from where it fell; frozen until "
                  "the next command" % (pose, spot[0], spot[1], spot[2], math.degrees(yaw), math.hypot(spot[0] - x0, spot[1] - y0)),
                  flush=True)
        return drain(self)

    cls._drain_pending_scene_reset = drain_and_reset


if __name__ == "__main__":
    path = sys.argv[1]
    sys.argv = [path] + sys.argv[2:]
    spec = importlib.util.spec_from_file_location("x2_mujoco_ros_bridge", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    _patch(mod.X2MujocoRosBridge)
    threading.Thread(target=_listen, daemon=True).start()
    sys.exit(mod.main())
