"""Writes routes.py.v4j, keep.py.v4j, extended.py.v4j next to the live files (x2_port/files_instinctlab/.../config/x2), by exact
string replacement (each must match once). The live files stay v4i's until continue_v4j.sh copies the .v4j files over them."""
import sys
D = sys.argv[1]


def patch(name, reps):
    s = open(f"{D}/{name}").read()
    for a, b in reps:
        assert s.count(a) == 1, (name, a[:60], s.count(a))
        s = s.replace(a, b)
    open(f"{D}/{name}.v4j", "w").write(s)
    print("wrote", name + ".v4j")


patch("extended.py", [(
"""    terms. Same observations and actions as run 4, so v4g's checkpoints continue on it.\"\"\"
    def __post_init__(self):
        super().__post_init__()
        _apply_extended(self)
""",
"""    terms. Same observations and actions as run 4, so v4g's checkpoints continue on it.\"\"\"
    def __post_init__(self):
        super().__post_init__()
        _apply_extended(self)
        from .staged import _opts
        if "g" in _opts():  # v4j: wall / prop contact graded by body part, torso ends only when resting on something (contact.py)
            from .contact import apply_v4j
            apply_v4j(self)
""")])

patch("keep.py", [(
"""        # posture watch (2026-11-12, user""",
"""        # v4j logs (option g, contact.py): episode ends and route arrivals per tile group, forces per body group, tunnelling,
        # route robots' speed near tall props
        from .staged import _opts
        if "g" in _opts() and "agroup_of_col" in _k:
            try:
                from .contact import log_v4j
                log_v4j(env, env_ids, out, _k, AGROUPS, _k["agroup_of_col"][t.terrain_types])
            except Exception as e:
                if not _k.get("v4j_warned"):
                    _k["v4j_warned"] = True
                    print("[X2 level_keeper] v4j log off: %r" % (e,), flush=True)
        # posture watch (2026-11-12, user""")])

patch("routes.py", [
("""MAX_WP = 12
""",
"""MAX_WP = 12
# v4j (2026-11-12, user: "there is option of side stepping in really tight spots"): on tight-spot and home tiles a share of the
# route robots walk the route sideways (facing +-90 deg off the bearing, |v| <= SIDE_V); and every route robot gets a lateral
# command for a waypoint off to the side within 1 m (vy = clip(lateral offset, +-ROUTE_VY)), so small side errors are stepped out.
SIDE_SHARE = float(os.environ.get("HIKING_X2_SIDESTEP_SHARE", "0"))
SIDE_V = float(os.environ.get("HIKING_X2_SIDESTEP_V", "0.3"))
ROUTE_VY = float(os.environ.get("HIKING_X2_ROUTE_VY", "0"))
SIDE_TYPES = ("tight_spots", "homes")


def side_law(d_xy, heading, side, is_last, pass_radius=0.2, vside=0.3):
    \"\"\"Sideways walking: side (n,) +1 = waypoint on the robot's left (walks left), -1 = right. Returns vx, vy, wz, arrived.\"\"\"
    dist = torch.norm(d_xy, dim=1)
    err = torch.atan2(d_xy[:, 1], d_xy[:, 0]) - heading
    err = torch.atan2(torch.sin(err), torch.cos(err))
    off = err - side * (math.pi / 2)
    off = torch.atan2(torch.sin(off), torch.cos(off))
    wz = torch.clamp(1.5 * off, -1.0, 1.0)
    s = torch.minimum(torch.clamp(dist, min=0.1), torch.full_like(dist, vside))
    s = torch.where(off.abs() > math.pi / 3, torch.zeros_like(s), s)
    vx, vy = s * torch.cos(err), s * torch.sin(err)
    arrived = dist < pass_radius
    stop = arrived & is_last
    z = torch.zeros_like(vx)
    return torch.where(stop, z, vx), torch.where(stop, z, vy), torch.where(stop, z, wz), arrived


def lateral_vy(d_xy, heading, vmax_y):
    \"\"\"Body-frame lateral offset of a waypoint within 1 m, clipped to +-vmax_y (0 farther away or behind).\"\"\"
    dist = torch.norm(d_xy, dim=1)
    c, s = torch.cos(heading), torch.sin(heading)
    lx = c * d_xy[:, 0] + s * d_xy[:, 1]
    ly = -s * d_xy[:, 0] + c * d_xy[:, 1]
    vy = torch.clamp(ly, -vmax_y, vmax_y)
    return torch.where((dist < 1.0) & (lx > -0.1), vy, torch.zeros_like(vy))
"""),
("""        self.route_done = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._in_reset = False
        self.table = {}
""",
"""        self.route_done = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.side = torch.zeros(n, device=self.device)
        self.side_cells = set()
        self._in_reset = False
        self.table = {}
"""),
("""                self.table = build_route_table(ext._CALLS, meta, cols, gen.num_rows, self.terrain.terrain_origins.tolist())
""",
"""                self.table = build_route_table(ext._CALLS, meta, cols, gen.num_rows, self.terrain.terrain_origins.tolist())
                self.side_cells = {(r_, c_) for tt in SIDE_TYPES for c_ in cols.get(tt, []) for r_ in range(gen.num_rows)}
"""),
("""        self.on_route[ids] = False
        self.route_done[ids] = False
""",
"""        self.on_route[ids] = False
        self.route_done[ids] = False
        self.side[ids] = 0.0
"""),
("""            yaw = yaw0 + (torch.rand(1).item() - 0.5) * 0.6
""",
"""            yaw = yaw0 + (torch.rand(1).item() - 0.5) * 0.6
            if SIDE_SHARE > 0 and (row, col) in self.side_cells and torch.rand(1).item() < SIDE_SHARE:
                sd = 1.0 if torch.rand(1).item() < 0.5 else -1.0
                self.side[e] = sd
                yaw -= sd * math.pi / 2   # first waypoint at +-90 deg in the body frame
"""),
("""        vx, wz, arrived = waypoint_law(cur[:, :2] - pos, self.robot.data.heading_w[r], vmax, is_last, self.cfg.pass_radius)
        self.route_done[r] |= arrived & is_last
        done = self.route_done[r]
        vx = torch.where(done, torch.zeros_like(vx), vx)
        wz = torch.where(done, torch.zeros_like(wz), wz)
        self.vel_command_b[r, 0] = vx
        self.vel_command_b[r, 1] = 0.0
        self.vel_command_b[r, 2] = wz
""",
"""        d_xy, hd = cur[:, :2] - pos, self.robot.data.heading_w[r]
        vx, wz, arrived = waypoint_law(d_xy, hd, vmax, is_last, self.cfg.pass_radius)
        vy = lateral_vy(d_xy, hd, ROUTE_VY) if ROUTE_VY > 0 else torch.zeros_like(vx)
        sd = self.side[r]
        if (sd != 0).any():
            svx, svy, swz, _ = side_law(d_xy, hd, sd, is_last, self.cfg.pass_radius, SIDE_V)
            m = sd != 0
            vx, vy, wz = torch.where(m, svx, vx), torch.where(m, svy, vy), torch.where(m, swz, wz)
        self.route_done[r] |= arrived & is_last
        done = self.route_done[r] | (arrived & is_last)
        vx = torch.where(done, torch.zeros_like(vx), vx)
        vy = torch.where(done, torch.zeros_like(vy), vy)
        wz = torch.where(done, torch.zeros_like(wz), wz)
        self.vel_command_b[r, 0] = vx
        self.vel_command_b[r, 1] = vy
        self.vel_command_b[r, 2] = wz
"""),
])
