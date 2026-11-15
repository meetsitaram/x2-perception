"""Build variants/staged.py.v6 from the live staged.py (= staged.py.v5): option t gets a speed-dependent cap (HIKING_X2_AIR_TD_CAP=ref)."""
import sys
X = "/mnt/filesystem-g4/tracks/hiking/code/x2_port"
src = open(f"{X}/files_instinctlab/tasks/parkour/config/x2/staged.py").read()

doc_old = "     (Isaac Lab's (air - threshold) form charges them). Beside the paper's `feet_air_time`, not instead of it. Not scheduled.\n"
doc_new = doc_old + (
    "     v6 (2026-11-13, HIKING_X2_AIR_TD_CAP=ref): the 0.30 s cap set the paid optimum at 1/(0.30+0.05) = 2.9 steps/s at every\n"
    "     speed (docs/findings/gait_vs_reference.md). With ref, the cap depends on the commanded planar speed v:\n"
    "     cap(v) = 1/f_ref(v) - 0.05, f_ref = linear in (0.3, 1.4), (0.5, 1.9), (0.7, 2.0) steps/s (X2 paper reference), flat outside;\n"
    "     the pay per second then peaks at the reference's swing (1/f_ref). Swings under 0.35 s pay exactly as before.\n")
assert src.count(doc_old) == 1
src = src.replace(doc_old, doc_new)

fn_anchor = "\n\n# ---- r: reference-state starts ----\n"
assert src.count(fn_anchor) == 1
fn = '''

_REF_V = (0.3, 0.5, 0.7)
_REF_F = (1.4, 1.9, 2.0)


def _ref_cadence(v: torch.Tensor) -> torch.Tensor:
    """Reference cadence (steps/s) of the X2 paper reference at planar speed v, linear between points, flat outside."""
    f = torch.full_like(v, _REF_F[0])
    for (v0, f0), (v1, f1) in zip(zip(_REF_V, _REF_F), zip(_REF_V[1:], _REF_F[1:])):
        seg = (v >= v0) & (v < v1)
        f = torch.where(seg, f0 + (f1 - f0) * (v - v0) / (v1 - v0), f)
    return torch.where(v >= _REF_V[-1], torch.full_like(v, _REF_F[-1]), f)


def feet_air_touchdown_refcap(env, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float = 0.05,
                              min_cmd: float = 0.1) -> torch.Tensor:
    """Option t, v6: as feet_air_touchdown_positive, cap = 1/f_ref(|v_cmd|) - threshold (paid optimum = reference swing)."""
    sensor = env.scene.sensors[sensor_cfg.name]
    first = sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    air = sensor.data.last_air_time[:, sensor_cfg.body_ids]
    cmd = env.command_manager.get_command(command_name)
    cap = (1.0 / _ref_cadence(torch.norm(cmd[:, :2], dim=1)) - threshold).unsqueeze(1)
    pay = (torch.minimum((air - threshold).clamp(min=0.0), cap) * first).sum(dim=1)
    moving = (torch.norm(cmd[:, :2], dim=1) > min_cmd) | (cmd[:, 2].abs() > min_cmd)
    return pay * moving
'''
src = src.replace(fn_anchor, fn + fn_anchor)

reg_old = "            r.feet_air_touchdown = RewTerm(func=feet_air_touchdown_positive, "
reg_new = ("            _td = feet_air_touchdown_refcap if os.environ.get(\"HIKING_X2_AIR_TD_CAP\", \"fixed\") == \"ref\" else feet_air_touchdown_positive\n"
           "            r.feet_air_touchdown = RewTerm(func=_td, ")
assert src.count(reg_old) == 1
src = src.replace(reg_old, reg_new)
open(sys.argv[1], "w").write(src)
print("wrote", sys.argv[1])
