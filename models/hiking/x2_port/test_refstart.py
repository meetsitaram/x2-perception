"""CPU test of option r in staged.py.next with stub scene objects (no Isaac). python test_refstart.py staged.py.next"""
import math, os, sys, types, torch

src = open(sys.argv[1]).read()
ns = {"os": os, "torch": torch}
exec(src[src.index("STEPS_PER_ITERATION"):src.index("# ---- a: staged rewards")], ns)
exec(src[src.index("# ---- r: reference-state starts ----"):src.index("# ---- f: fall penalty ----")], ns)
torch.manual_seed(0)


def euler(r, p, y):  # wxyz from roll, pitch, yaw (ZYX)
    cr, sr, cp, sp, cy, sy = (math.cos(r / 2), math.sin(r / 2), math.cos(p / 2), math.sin(p / 2), math.cos(y / 2), math.sin(y / 2))
    return torch.tensor([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy])


def rpy(q):
    w, x, y, z = q.tolist()
    return (math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)), math.asin(max(-1, min(1, 2 * (w * y - z * x)))),
            math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


# reference frame: yaw 1.2 rad, roll 0.05, pitch 0.1, moving 0.8 m/s along its own heading, pelvis 0.62 m above an origin at z 0.4
ref_q = euler(0.05, 0.1, 1.2).unsqueeze(0)
ref_lin = torch.tensor([[0.8 * math.cos(1.2), 0.8 * math.sin(1.2), 0.05]])
ref_ang = torch.tensor([[0.1, -0.2, 0.3]])
origin = torch.tensor([[5.0, 7.0, 0.4]])
ref_pos = torch.tensor([[9.0, 9.0, 0.4 + 0.62]])  # x, y far away: must not be used
robot_pos = torch.tensor([[5.05, 6.95, 1.1]]); robot_q = euler(0, 0, -0.4).unsqueeze(0)
pos, q, lin, ang = ns["reference_start_state"](ref_pos, ref_q, ref_lin, ref_ang, origin, robot_pos, robot_q, 0.03)
r, p, y = rpy(q[0])
assert abs(r - 0.05) < 1e-5 and abs(p - 0.1) < 1e-5 and abs(y + 0.4) < 1e-5, (r, p, y)
assert torch.allclose(pos, torch.tensor([[5.05, 6.95, 1.05]]), atol=1e-6), pos
assert torch.allclose(lin, torch.tensor([[0.8 * math.cos(-0.4), 0.8 * math.sin(-0.4), 0.05]]), atol=1e-6), lin  # still along the heading
assert abs(ang[0, 2].item() - 0.3) < 1e-6 and abs(ang[0, :2].norm().item() - ref_ang[0, :2].norm().item()) < 1e-6

# the event: share follows the schedule, only selected robots are written, joint state passed through
calls = []
n = 4000


class Ref:
    data = None

    def get_init_reference_state(self, ids):
        k = len(ids)
        return types.SimpleNamespace(base_pos_w=torch.zeros(k, 3) + 0.62, base_quat_w=ref_q.repeat(k, 1), base_lin_vel_w=ref_lin.repeat(k, 1),
                                     base_ang_vel_w=ref_ang.repeat(k, 1), joint_pos=torch.ones(k, 31) * 0.3, joint_vel=torch.ones(k, 31))


class Robot:
    data = types.SimpleNamespace(root_pos_w=torch.zeros(n, 3), root_quat_w=torch.tensor([[1.0, 0, 0, 0]]).repeat(n, 1))

    def write_root_pose_to_sim(self, x, env_ids):
        calls.append(("pose", len(env_ids), x.shape))

    def write_root_velocity_to_sim(self, x, env_ids):
        calls.append(("vel", len(env_ids), x.shape))

    def write_joint_state_to_sim(self, a, b, env_ids):
        calls.append(("joint", len(env_ids), a.shape))


scene = {"robot": Robot(), "motion_reference": Ref()}
env = types.SimpleNamespace(device="cpu", common_step_counter=0, scene=types.SimpleNamespace(__getitem__=None))
env.scene = type("S", (), {"__getitem__": lambda self, k: scene[k], "env_origins": torch.zeros(n, 3)})()
os.environ["HIKING_X2_STAGE"] = "1000,2000"
for off, want in ((500, 0.5), (1000, 0.5), (1500, 0.25), (2000, 0.0)):
    os.environ["HIKING_ITER_OFFSET"] = str(off)
    calls.clear()
    ns["reset_from_reference_staged"](env, torch.arange(n), early_share=0.5)
    k = calls[0][1] if calls else 0
    assert abs(k / n - want) < 0.03, (off, k / n)
    if calls:
        assert [c[0] for c in calls] == ["pose", "vel", "joint"] and calls[0][2] == (k, 7) and calls[1][2] == (k, 6) and calls[2][2] == (k, 31)
assert "r" in src[src.index("unknown = opts"):src.index("unknown = opts") + 40] and "HIKING_X2_REF_START" in src
print("option r: x, y, yaw kept; roll 0.05 / pitch 0.1 from the clip; height origin + 0.62 + 0.03; velocity follows the heading; "
      "share 0.5 / 0.5 / 0.25 / 0 at 500 / 1,000 / 1,500 / 2,000 with stage 1000,2000; only selected robots written. All checks passed.")
