"""Add the planner's definition and the recommended decode options to a pair's adapter_card.json.
    python tools/card_planner.py PAIR_DIR PLANNER.pt PLANNER.onnx"""
import json
import os
import sys

import torch

d, pt, onnx = sys.argv[1:4]
card = json.load(open(os.path.join(d, "adapter_card.json")))
ck = torch.load(pt, map_location="cpu")
card["pair"] = "tracker (actor.npz) + contact planner (%s); every control step: planner -> 27 contact-target numbers -> tracker" % os.path.basename(onnx)
card["label"] = ("interim: stage-1 tracker + first planner; the tracker has not been trained on planner contacts (no stage-2 co-training), "
                 "no stage 3 (no goal terrains, walls, rails or in-place turns in training)")
card["planner"] = dict(
    file=os.path.basename(onnx), architecture=ck["arch"], parameters=ck["parameters"], runs_every="control step (0.02 s), batch 1, no recurrent state",
    trained_on="hindsight contact labels from successful rollouts of tracker iteration %s on the stage-1 clip terrains; no SuperMimic data" % ck["tracker_iteration"],
    held_out_stats=ck["stats"],
    inputs=[
        dict(name="proprio_history", size=card["observation"][0]["size"],
             definition="exactly the tracker's proprio_history block (5 control steps, oldest first, 99 numbers each: base angular velocity, "
                        "projected gravity, joint position minus default, joint velocity, previous action); un-normalised, the graph normalises"),
        dict(name="goal_command", size=3, definition="exactly the tracker's goal_command (dx, dy, dpsi)"),
        dict(name="scan_17x11", size=187, grid="17 (x) by 11 (y), 0.1 m spacing; x from -0.8 to +0.8 m, y from -0.5 to +0.5 m; index = ix * 11 + iy",
             frame="body-yaw (heading) frame centred on the root (pelvis) xy", value="terrain height under the point minus the root height (m)",
             clip=[-2.0, 2.0], walls="training terrains had prop boxes (their tops are in the scan) and no walls or rails; tall columns are out of "
             "distribution for this planner")],
    outputs=[
        dict(name="contact_targets", shape=[1, 9, 3], bodies=card["observation"][2]["bodies"],
             definition="per body the scan point with the largest affordance weight (top-1 decode, so the target always lies on scanned "
                        "terrain): (x, y) of that grid point and its scanned height, i.e. z = terrain height minus root height",
             frame="body-yaw (heading) frame relative to the root, the same frame and units as the tracker's contact_targets",
             gating="already zeroed inside the graph where occupancy <= 0.5",
             use="flatten to 27 numbers (body order as listed, x y z each) and put them in the tracker's contact_targets slot unchanged"),
        dict(name="occupancy", shape=[1, 9], definition="probability that the body starts a NEW contact within 1.5 s")],
    only_feet="only the two feet were ever labelled in this round's data; the other seven bodies' outputs are zero in practice",
)
card["recommended_decode_options"] = dict(
    default="use the ONNX outputs as they are (this is how the pair was checked on the training terrains)",
    target_latch=dict(recommend="yes, as a labelled option", what="once a foot's target is within about 0.3 s of touchdown (foot within ~10 cm "
                      "above the target and descending, or the target unchanged within 5 cm for several steps), hold that target in the WORLD "
                      "frame until the foot's contact starts, re-expressing it in the body-yaw frame each step",
                      why="the planner has no memory and re-plans every step; in the earlier X2 work 13 % of steps jumped more than 5 cm and "
                          "the latch raised stair climbs from 31 to 39 of 48"),
    edge_mask=dict(recommend="yes, as a labelled option", what="before taking a target, reject scan points with a height step of more than 3 cm "
                   "within the foot's footprint (the X2 sole is about 22 x 13 cm; 24 cm with the toe cap) and take the nearest remaining scan "
                   "point to the planner's choice (the ONNX gives only the chosen point, not the weights)",
                   why="earlier work: planned-on-edge 47 % -> 18 %; note the X2 foot is longer than a 22 cm tread, so on stairs allow heel overhang"),
    report="score the pair with the options off and on, and say which was used; neither was used in training")
json.dump(card, open(os.path.join(d, "adapter_card.json"), "w"), indent=1)
print("card updated:", list(card.keys())[-4:])
