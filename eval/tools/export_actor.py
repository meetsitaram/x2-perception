"""Run by a TRACK, in its own Python environment (needs torch), to hand a plain-MLP policy to the eval as numpy.

    python export_actor.py CKPT.pt OUT.npz --actor-prefix actor --norm-prefix an [--goal-r 0.5]
        (ACP layout: checkpoint["model"] holds actor.{0,2,4,6}.{weight,bias} and an.{mean,var})

The joint list, default pose, action scale and PD gains are NOT in a checkpoint: pass them with --meta meta.json
({"joint_names": [...31], "default_q": [...], "action_scale": [...], "kp": [...], "kd": [...]}, all in the order the
policy's joint observations use), written from the same objects the training run used. Policies with a CNN, a GRU or
several heads go out as ONNX instead (torch.onnx.export), with the adapter card describing every input and output.
"""
import argparse
import json

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("ckpt"); ap.add_argument("out")
ap.add_argument("--actor-prefix", default="actor"); ap.add_argument("--norm-prefix", default="an")
ap.add_argument("--meta", required=True); ap.add_argument("--goal-r", type=float, default=None)
a = ap.parse_args()
ck = torch.load(a.ckpt, map_location="cpu")
sd = ck["model"] if "model" in ck else ck
out, k = {}, 0
for name in sorted((n for n in sd if n.startswith(a.actor_prefix + ".") and n.endswith(".weight")), key=lambda n: int(n.split(".")[1])):
    out["actor_w%d" % k] = sd[name].numpy()
    out["actor_b%d" % k] = sd[name[:-6] + "bias"].numpy()
    k += 1
out["norm_mean"], out["norm_var"] = sd[a.norm_prefix + ".mean"].numpy(), sd[a.norm_prefix + ".var"].numpy()
meta = json.load(open(a.meta))
for key in ("joint_names", "default_q", "action_scale", "kp", "kd"):
    out[key] = np.array(meta[key])
if a.goal_r is not None:
    out["goal_r"] = a.goal_r
out["source"] = "%s, iteration %s" % (a.ckpt, ck.get("it", "?"))
np.savez(a.out, **out)
print("wrote", a.out, "layers", k, "actor in/out", out["actor_w0"].shape[1], out["actor_w%d" % (k - 1)].shape[0])
