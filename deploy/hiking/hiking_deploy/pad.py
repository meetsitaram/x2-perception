"""Pad -> Hiking deploy node over UDP JSON (50 Hz, 127.0.0.1:5690). Runs on the host in sim and on PC2 on the robot.
The node answers each pad with its state line, which the pad prints.

    python3 pad.py [--source gamepad|terminal|script] [--script FILE] [--port 5690]

gamepad, SONIC's layout (gwbc-x2 docs/x2/gamepad_cheatsheet.md; Xbox names, DualSense works via SDL):
  stand: hold L1+L2+R1+R2 ~3 s (rumble), then Y. Drive: hold L2 (deadman); left stick up = forward (0..vmax, forward only),
  left stick sideways = turn (yaw rate +-1.0); no sidestep (user, 2026-11-15: "side walk is not that good yet"); release L2 =
  stop (the policy keeps it standing). Right stick: the sim viewer's camera (orbits the robot, still following it).
  L1 / R1 while driving: vmax -/+ 0.1 m/s (0.1..0.8, starts at 0.5). E-stop: hold A+X and pump both triggers 3 times
  (damps; latched); clear: release, wait 5 s or more, repeat the gesture, then stand again.
  Sim only: View / Back held 1 s = reset the robot to the spawn (then stand again).
terminal (keys typed into this terminal; deadman always on, sim only): Enter = stand / walk, Up / Down = speed +-0.1 m/s,
  Left / Right = turn +-0.25 rad/s, , / . = sideways +-0.1 m/s, Space = stop, b = damp, e = e-stop on / off, r = reset (sim), q = quit.
script: lines "t_seconds key=value ..." (keys vx vy wz deadman y b estop); values hold until changed; y / b fire once.
Commands are rate-limited (0.5 m/s^2, 2 rad/s^2) and dead-zoned (0.15) as in the plan.
"""
from __future__ import annotations

import argparse
import json
import os
import select
import socket
import sys
import time

DEADZONE, VY_MAX, WZ_MAX = 0.15, 0.3, 1.0
PRESETS = (0.3, 0.5, 0.8)                         # [-1] = top speed
ACC_V, ACC_W = 0.5, 2.0
HZ = 50.0


def dz(v):
    return 0.0 if abs(v) < DEADZONE else (v - DEADZONE * (1 if v > 0 else -1)) / (1 - DEADZONE)


def limit(cur, tgt, rate, dt):
    return cur + max(-rate * dt, min(rate * dt, tgt - cur))


class Script:
    def __init__(self, path):
        self.rows = []
        for ln in open(path):
            ln = ln.split("#")[0].split()
            if ln:
                self.rows.append((float(ln[0]), dict((kv.split("=") + ["1"])[:2] for kv in ln[1:])))
        self.i, self.t0 = 0, time.monotonic()
        self.cur = dict(vx=0.0, vy=0.0, wz=0.0, deadman=False, estop=False)

    def poll(self, fire):
        t = time.monotonic() - self.t0
        while self.i < len(self.rows) and self.rows[self.i][0] <= t:
            for k, v in self.rows[self.i][1].items():
                if k in ("y", "b", "reset"):
                    fire.add(k)
                elif k in ("deadman", "estop"):
                    self.cur[k] = v in ("1", "true", "on")
                else:
                    self.cur[k] = float(v)
            self.i += 1
        return self.cur

    def done(self):
        return self.i >= len(self.rows) and time.monotonic() - self.t0 > self.rows[-1][0] + 1.0


class Terminal:
    """Raw keys from this terminal (cbreak, no echo)."""

    def __init__(self):
        import termios
        import tty
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        self.cur = dict(vx=0.0, vy=0.0, wz=0.0, deadman=True, estop=False)
        self.quit = False

    def restore(self):
        import termios
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    def poll(self, fire):
        c = self.cur
        while select.select([self.fd], [], [], 0)[0]:
            ch = os.read(self.fd, 64).decode(errors="ignore")
            for seq in ("\x1b[A", "\x1b[B", "\x1b[C", "\x1b[D", "\x1bOA", "\x1bOB", "\x1bOC", "\x1bOD"):
                while seq in ch:
                    ch = ch.replace(seq, "", 1)
                    k = seq[-1]
                    if k == "A": c["vx"] = round(min(c["vx"] + 0.1, PRESETS[-1]), 2)
                    elif k == "B": c["vx"] = round(max(c["vx"] - 0.1, 0.0), 2)
                    elif k == "D": c["wz"] = round(min(c["wz"] + 0.25, WZ_MAX), 2)
                    elif k == "C": c["wz"] = round(max(c["wz"] - 0.25, -WZ_MAX), 2)
            for k in ch:
                if k in "\r\n": fire.add("y")
                elif k == "b": fire.add("b")
                elif k == ",": c["vy"] = round(min(c["vy"] + 0.1, VY_MAX), 2)
                elif k == ".": c["vy"] = round(max(c["vy"] - 0.1, -VY_MAX), 2)
                elif k == " ": c.update(vx=0.0, vy=0.0, wz=0.0)
                elif k == "e": c["estop"] = not c["estop"]
                elif k == "r": fire.add("reset")
                elif k == "q": self.quit = True
        return c


class Gamepad:
    """SONIC's X2 pad layout (gwbc-x2 docs/x2/gamepad_cheatsheet.md), without its clips and dances."""
    IGNITE_S, ARMED_S = 3.0, 8.0     # as SONIC's start ritual (gwbc-x2 gear_sonic/scripts/pc2_pad_daemon.py)

    def __init__(self):
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        import pygame
        from pygame._sdl2 import controller
        self.pg = pygame
        pygame.init()
        controller.init()
        if controller.get_count() == 0:
            sys.exit("no gamepad found (plug one in, or use --source terminal)")
        self.c = controller.Controller(0)
        print("gamepad:", self.c.name, flush=True)
        self.prev, self.estop, self.estop_t = {}, False, -1e9
        self.vmax, self.hold_t, self.armed_until, self.pumps, self.chord = 0.5, None, 0.0, 0, False
        self.hold_pulses, self.need_release = 0, False
        self.back_t, self.back_fired = None, False

    def rumble(self, lo, hi, ms):
        try:
            self.c.rumble(lo, hi, ms)
        except Exception:
            pass

    ritual = True                    # the start ritual runs here (False when SONIC's pad daemon owns it: ZmqPad)

    def read(self):
        """-> (buttons by name, axis(name) in -1..1, triggers 0..1)."""
        pg, c = self.pg, self.c
        pg.event.pump()
        btn = {k: c.get_button(getattr(pg, "CONTROLLER_BUTTON_" + k))
               for k in ("A", "B", "X", "Y", "BACK", "LEFTSHOULDER", "RIGHTSHOULDER")}
        return btn, (lambda n: c.get_axis(getattr(pg, "CONTROLLER_AXIS_" + n)) / 32767.0)

    def poll(self, fire):
        now = time.monotonic()
        btn, ax = self.read()
        if btn is None:                  # no fresh pad data: all neutral, no deadman
            return dict(vx=0.0, vy=0.0, wz=0.0, deadman=False, estop=self.estop)
        edge = lambda k: btn[k] and not self.prev.get(k)  # noqa: E731
        l2, r2 = ax("TRIGGERLEFT") > 0.5, ax("TRIGGERRIGHT") > 0.5
        # start ritual, as SONIC's: hold L1+R1+L2+R2 -> one pulse per held second (3); the 3rd = ARMED (8 s window);
        # Y in the window = long strong buzz + start; A / B / X in the window = weak blip, disarmed; early release resets
        if not self.ritual:
            pass
        elif now < self.armed_until:
            if edge("Y"):
                fire.add("y")
                self.armed_until = 0.0
                self.rumble(1.0, 1.0, 800)
                self.need_release = True         # a new ritual only after the four are let go (no re-arm while held)
            elif any(edge(k) for k in ("A", "B", "X")):
                self.armed_until = 0.0
                self.rumble(0.3, 0.3, 100)
                print("start ritual: disarmed", flush=True)
        elif btn["LEFTSHOULDER"] and btn["RIGHTSHOULDER"] and l2 and r2 and not self.need_release:
            if self.hold_t is None:
                self.hold_t, self.hold_pulses = now, 0
            el = now - self.hold_t
            if self.hold_pulses < int(el) and self.hold_pulses < self.IGNITE_S:
                self.hold_pulses += 1
                self.rumble(0.6, 0.6, 150)
            if el >= self.IGNITE_S:
                self.armed_until = now + self.ARMED_S
                self.hold_t = None
                print("start ritual armed: press Y", flush=True)
        elif not (btn["LEFTSHOULDER"] or btn["RIGHTSHOULDER"] or l2 or r2):
            self.hold_t, self.hold_pulses, self.need_release = None, 0, False
        else:
            self.hold_t, self.hold_pulses = None, 0
        # reset (sim only): View / Back held 1 s
        if btn["BACK"]:
            self.back_t = self.back_t or now
            if now - self.back_t >= 1.0 and not self.back_fired:
                fire.add("reset")
                self.back_fired = True
                self.rumble(0.3, 0.3, 200)
        else:
            self.back_t, self.back_fired = None, False
        # e-stop: A+X chord, then pump both triggers (3 pumps); clear = the same gesture 5 s or more later
        if btn["A"] and btn["X"]:
            if not self.chord:
                self.chord, self.pumps = True, 0
            if l2 and r2 and not self.prev.get("both"):
                self.pumps += 1
                if self.pumps == 3:
                    if not self.estop:
                        self.estop, self.estop_t = True, now
                        self.rumble(1.0, 1.0, 600)
                        print("E-STOP", flush=True)
                    elif now - self.estop_t >= 5.0:
                        self.estop = False
                        print("E-STOP CLEARED", flush=True)
        else:
            self.chord = False
        # speed nudges while driving (L1 / R1 alone, deadman held)
        drive = l2 and not r2 and not (btn["A"] and btn["X"])
        if drive and edge("LEFTSHOULDER") and not btn["RIGHTSHOULDER"]:
            self.vmax = round(max(self.vmax - 0.1, 0.1), 2)
        if drive and edge("RIGHTSHOULDER") and not btn["LEFTSHOULDER"]:
            self.vmax = round(min(self.vmax + 0.1, PRESETS[-1]), 2)
        btn["both"] = l2 and r2
        self.prev = btn
        names = dict(A="A", B="B", X="X", Y="Y", BACK="View", LEFTSHOULDER="L1", RIGHTSHOULDER="R1")
        # raw inputs for the viewer's pad HUD (screen axes: x right, y up)
        self.hud = dict(lx=round(ax("LEFTX"), 2), ly=round(-ax("LEFTY"), 2), rx=round(ax("RIGHTX"), 2), ry=round(-ax("RIGHTY"), 2),
                        l2=round(ax("TRIGGERLEFT"), 2), r2=round(ax("TRIGGERRIGHT"), 2),
                        held=[names[k] for k in names if btn[k]], vmax=self.vmax, estop=self.estop,
                        ign="armed" if now < self.armed_until else ("%.1f" % (now - self.hold_t) if self.hold_t else ""))
        return dict(vx=max(0.0, -dz(ax("LEFTY"))) * self.vmax, vy=0.0,           # sidestep off (user, 2026-11-15)
                    wz=-dz(ax("LEFTX")) * WZ_MAX, deadman=drive and not (btn["LEFTSHOULDER"] and btn["RIGHTSHOULDER"]),
                    estop=self.estop)



class ZmqPad(Gamepad):
    """The robot's pad as SONIC's PC2 pad daemon publishes it (ZMQ pad_state on :5569, raw SDL joystick indices; vibration
    requests to its PULL :5570), with our layout and e-stop on top. The daemon keeps the Bluetooth pairing and owns the start
    ritual (it launched us), so the ritual is off here."""
    ritual = False
    AX = dict(LEFTX=0, LEFTY=1, TRIGGERLEFT=2, RIGHTX=3, RIGHTY=4, TRIGGERRIGHT=5)
    BTN = dict(A=0, B=1, X=2, Y=3, LEFTSHOULDER=4, RIGHTSHOULDER=5, BACK=6)

    def __init__(self, host="127.0.0.1", port=5569, rumble_port=5570):
        import zmq
        ctx = zmq.Context.instance()
        self.sub = ctx.socket(zmq.SUB)
        self.sub.setsockopt(zmq.SUBSCRIBE, b"pad_state")
        self.sub.connect("tcp://%s:%d" % (host, port))
        self.push = ctx.socket(zmq.PUSH)
        self.push.setsockopt(zmq.SNDTIMEO, 0)
        self.push.connect("tcp://%s:%d" % (host, rumble_port))
        self.zmq, self.last, self.last_t = zmq, None, 0.0
        print("pad: SONIC pad daemon at tcp://%s:%d" % (host, port), flush=True)
        self.prev, self.estop, self.estop_t = {}, False, -1e9
        self.vmax, self.hold_t, self.armed_until, self.pumps, self.chord = 0.5, None, 0.0, 0, False
        self.hold_pulses, self.need_release = 0, False
        self.back_t, self.back_fired = None, False

    def rumble(self, lo, hi, ms):
        try:
            self.push.send_string(json.dumps(dict(strength=max(lo, hi), ms=int(ms))))
        except Exception:
            pass

    def read(self):
        try:
            while True:
                parts = self.sub.recv_multipart(self.zmq.NOBLOCK)
                self.last, self.last_t = json.loads(parts[-1]), time.monotonic()
        except self.zmq.Again:
            pass
        if self.last is None or time.monotonic() - self.last_t > 0.5:      # the daemon publishes at >= 5 Hz
            return None, None
        b, a = self.last["buttons"], self.last["axes"]
        btn = {k: (b[i] if i < len(b) else 0) for k, i in self.BTN.items()}

        def ax(n):
            v = a[self.AX[n]] if self.AX[n] < len(a) else 0.0
            return (v + 1.0) / 2.0 if n.startswith("TRIGGER") else v      # triggers rest at -1 on the raw joystick
        return btn, ax

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=("gamepad", "terminal", "script", "zmq"), default="gamepad",
                    help="zmq: the robot's pad via SONIC's PC2 pad daemon (it keeps the Bluetooth pairing and the ritual)")
    ap.add_argument("--script")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5690)
    a = ap.parse_args()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setblocking(False)
    src = (Script(a.script) if a.source == "script" else Terminal() if a.source == "terminal"
           else ZmqPad() if a.source == "zmq" else Gamepad())
    tty = a.source == "terminal"
    if tty:
        print(" ".join(__doc__.split("terminal (")[1].split("script:")[0].split()))
    ev, out, node_state = {"y": 0, "b": 0, "reset": 0}, dict(vx=0.0, vy=0.0, wz=0.0), "no answer from the deploy node yet"
    t_next, n = time.monotonic(), 0
    try:
        while True:
            fire = set()
            cur = src.poll(fire)
            if (a.source == "script" and src.done()) or getattr(src, "quit", False):
                break
            tgt = dict(vx=cur["vx"], vy=cur["vy"], wz=cur["wz"])
            if not cur["deadman"] or cur["estop"]:
                tgt = dict(vx=0.0, vy=0.0, wz=0.0)
            dt = 1.0 / HZ
            out = dict(vx=limit(out["vx"], tgt["vx"], ACC_V, dt), vy=limit(out["vy"], tgt["vy"], ACC_V, dt),
                       wz=limit(out["wz"], tgt["wz"], ACC_W, dt))
            for k in fire:
                ev[k] += 1
            sock.sendto(json.dumps(dict(out, deadman=bool(cur["deadman"]), estop=bool(cur["estop"]), ev=ev, seq=n,
                                        hud=getattr(src, "hud", "%s target vx %.1f vy %+.1f wz %+.2f" % (a.source, tgt["vx"], tgt["vy"], tgt["wz"])))).encode(),
                        (a.host, a.port))
            try:
                while True:
                    node_state = sock.recv(4096).decode()
            except (BlockingIOError, ConnectionRefusedError):
                pass
            line = "%s | target vx %.1f vy %+.1f wz %+.2f%s" % (node_state.replace("|", "  "), tgt["vx"], tgt["vy"], tgt["wz"],
                                                               " E-STOP" if cur["estop"] else "")
            if tty:
                if n % 5 == 0 or fire:
                    sys.stdout.write("\r\x1b[K" + line[:200])
                    sys.stdout.flush()
            elif fire or n % 50 == 0:
                print("%s %s" % ("".join(sorted(fire)) or "-", line), flush=True)
            n += 1
            t_next += dt
            time.sleep(max(0.0, t_next - time.monotonic()))
    finally:
        if tty:
            src.restore()
            print()


if __name__ == "__main__":
    main()
