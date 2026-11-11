"""Desktop-only, run in the `sm` virtual environment (torch, mujoco_warp) on the CPU, from gwbc-supermimic with
PYTHONPATH=. : (1) export the 2026-11-02 baseline tracker to numpy; (2) run the ORIGINAL rollout code
(scripts/acp/eval/pico_rollout_planner.py, unmodified except device "cpu" and a 50-step loop that records everything) on
our apparatus from two start states and dump what the eval adapter has to reproduce.

    python baseline_dump.py OUT_DIR
        -> OUT_DIR/acp3_v2_1000.npz   tracker weights, normalisers, joint order, default pose, action scale, gains, start pose
        -> OUT_DIR/obs_check.npz      per step and robot: qpos, qvel, planner input / output, contact input, tracker input, action

Nothing here goes near the GPU (CUDA is hidden), one thread, memory-capped, below-normal priority.
"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import memcap  # noqa: E402

memcap.guard(cap_gb=10, need_free_gb=0, low_priority=True)
import json  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
SM = "C:/Users/PC/projects/gwbc-supermimic"
CK = SM + "/logs_sm/acp3_v2/ckpt_001000.pt"
PL = SM + "/logs_sm/acp_artifacts/planner/planner_v2_hist1.pt"
B = "C:/Users/PC/projects/loco-manip-apparatus/build/"
STEPS = int(os.environ.get("CHECK_STEPS", "50"))

# ---- 1. export the tracker
from gear_sonic.supermimic.assets import build_x2  # noqa: E402
from gear_sonic.supermimic.motion import IL_TO_MJ  # noqa: E402

ck = torch.load(CK, map_location="cpu", weights_only=False)
x2 = build_x2(ck["cfg"]["plant"])
q = np.load(B + "stairs/routepass/route_on_v065.npy")
out = {}
for net in ("enc", "dec"):
    names = sorted((n for n in ck["pi"] if n.startswith(net + ".") and n.endswith(".weight")), key=lambda n: int(n.split(".")[1]))
    for k, n in enumerate(names):
        out["%s_w%d" % (net, k)] = ck["pi"][n].numpy()
        out["%s_b%d" % (net, k)] = ck["pi"][n[:-6] + "bias"].numpy()
for nm in ("goal", "prop", "scan", "contact", "ref"):
    out[nm + "_mean"], out[nm + "_var"] = ck["pi"][nm + "_norm.mean"].numpy(), ck["pi"][nm + "_norm.var"].numpy()
out.update(norm_eps=1e-2, joint_names=np.array(x2.joint_names), default_q=x2.default_q, action_scale=x2.action_scale, kp=x2.kp, kd=x2.kd,
           effort=x2.effort, armature=x2.armature, il_to_mj=np.array(IL_TO_MJ), start_q=q[0, 7:],
           source="gwbc-supermimic/logs_sm/acp3_v2/ckpt_001000.pt (the 2026-11-02 baseline tracker, plant %s)" % ck["cfg"]["plant"])
np.savez(os.path.join(OUT, "acp3_v2_1000.npz"), **out)
print("exported tracker:", {k: v.shape for k, v in out.items() if k.endswith("w0")}, flush=True)

# ---- 2. the original rollout code on two clips: the route start, and a start 0.35 m short of the front 4x4
n = 400
s0 = q[0].copy()
s1 = q[0].copy()
s1[:2] = [2.62, 0.69]
s1[3:7] = [1.0, 0.0, 0.0, 0.0]


def clip(s):
    return dict(root_trans_offset=np.tile(s[:3], (n, 1)).astype(np.float32), root_rot=np.tile(np.r_[s[4:7], s[3]], (n, 1)).astype(np.float32),
                dof=np.tile(s[7:], (n, 1)).astype(np.float32), fps=50)


joblib.dump({"route": clip(s0), "beam": clip(s1)}, os.path.join(OUT, "check.pkl"))
json.dump({"route": [], "beam": []}, open(os.path.join(OUT, "check_props.json"), "w"))
WPS = {"route": [[2.1336, 1.1938, 0.0], [2.6416, 1.1684, 0.0]],                  # the first waypoints of crowd/route_roam_h1.json
       "beam": [[3.302, 0.6858, 0.0, 1.0, -0.6], [3.302, 0.3302, 0.0]]}   # over the front 4x4, as a station with a facing (tests that path)
# (the 50 recorded steps keep the first waypoint: the switch to the next one lives in the original's main loop, replaced here)
json.dump(WPS, open(os.path.join(OUT, "check_roam.json"), "w"))
os.environ.update(GOAL_EVAL="1", GOAL_MAG="0.5", WALLS=B + "hmap/cad_walls.json", HMAP=B + "hmap/cad_20mm_walk.npz", PLAN_DECODE="mode",
                  STEP_MASK="0.03", PLAN_LATCH="0.3", ROAM_JSON=os.path.join(OUT, "check_roam.json"))
src = open(SM + "/scripts/acp/eval/pico_rollout_planner.py").read()
assert 'dev = "cuda"' in src
src = src.replace('dev = "cuda"', 'dev = "cpu"')
cut = src.index("with torch.no_grad():\n    while True:")
head = src[:cut]
sys.argv = ["pico_rollout_planner.py", CK, PL, os.path.join(OUT, "check.pkl"), os.path.join(OUT, "check_props.json"), os.path.join(OUT, "check_out.npy"), "1"]
import warp as wp  # noqa: E402

wp.set_device("cpu")                                  # warp's default device too, not only torch's
g = {"__name__": "__main__", "__file__": SM + "/scripts/acp/eval/pico_rollout_planner.py"}
exec(compile(head, "pico_rollout_planner.py(head)", "exec"), g)
sim, pi, planner, m = g["sim"], g["pi"], g["planner"], g["m"]
rec = {k: [] for k in ("qpos", "qvel", "x", "contacts", "action", "p_scan", "p_goal", "p_hist", "p_logits", "p_head", "a_prev")}
_fw = planner.forward


def fw(scan, goal, hist):
    o = _fw(scan, goal, hist)
    rec["p_scan"].append(scan.numpy().copy()); rec["p_goal"].append(goal.numpy().copy()); rec["p_hist"].append(hist.numpy().copy())
    rec["p_logits"].append(o[0].detach().numpy().copy()); rec["p_head"].append(torch.stack(o[1:], -1).detach().numpy().copy())
    return o


planner.forward = fw
_ci = sim.contact_input


def ci():
    c = _ci()
    rec["contacts"].append(c.numpy().copy())
    return c


sim.contact_input = ci
mj_to_il = g["mj_to_il"]
with torch.no_grad():
    for k in range(STEPS):
        rec["qpos"].append(sim.qpos.numpy().copy()); rec["qvel"].append(sim.qvel.numpy().copy()); rec["a_prev"].append(sim.a_prev.numpy().copy())
        x = sim.policy_input()
        a = pi(x)[:, mj_to_il].clamp(-20, 20)
        rec["x"].append(x.numpy().copy()); rec["action"].append(a.numpy().copy())
        sim.step(a)
        if k % 10 == 0:
            print("step", k, "root", sim.qpos[:, :3].numpy().round(3).tolist(), flush=True)
np.savez(os.path.join(OUT, "obs_check.npz"), names=np.array(m.names), waypoints=json.dumps(WPS), default_q=sim.default_q.numpy(), alpha=sim.alpha.numpy(),
         **{k: np.array(v) for k, v in rec.items()})
print("dumped", STEPS, "steps of", list(m.names), "->", os.path.join(OUT, "obs_check.npz"))
