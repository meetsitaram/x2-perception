"""Export a checkpoint to one ONNX file: (hist 5x72, depth 36x36, gru 256) -> (action 21, gru' 256). Deterministic; normalizer inside.

python deploy/export_onnx.py --ckpt runs/v1/last.pt --out deploy/ssr_x2.onnx
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import onnxruntime as ort
import torch

from ssr import networks as N
from ssr.env import DEPTH_N, H, PROP_DIM


class Policy(torch.nn.Module):
    def __init__(self, enc, aux, actor):
        super().__init__()
        self.enc, self.aux, self.actor = enc, aux, actor

    def forward(self, hist, depth, gru):
        e = self.enc.step(hist, depth, gru, sample=False)
        o = torch.cat((hist[:, -1], self.aux.est(hist.flatten(1)), e["z"]), -1)
        return self.actor(o), e["h"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    enc, aux, actor = N.Encoder(), N.Aux(), N.MoEActor()
    enc.load_state_dict(ck["enc"]); aux.load_state_dict(ck["aux"]); actor.load_state_dict(ck["actor"])
    p = Policy(enc, aux, actor).eval()
    x = (torch.randn(1, H, PROP_DIM), torch.rand(1, DEPTH_N, DEPTH_N), torch.zeros(1, 256))
    torch.onnx.export(p, x, a.out, input_names=["hist", "depth", "gru"], output_names=["action", "gru_out"], opset_version=17,
                      dynamic_axes={k: {0: "b"} for k in ("hist", "depth", "gru", "action", "gru_out")}, dynamo=False)
    s = ort.InferenceSession(a.out, providers=["CPUExecutionProvider"])
    xs = (torch.randn(8, H, PROP_DIM), torch.rand(8, DEPTH_N, DEPTH_N), torch.randn(8, 256) * 0.3)
    with torch.no_grad():
        act, g = p(*xs)
    r = s.run(None, {"hist": xs[0].numpy(), "depth": xs[1].numpy(), "gru": xs[2].numpy()})
    err = max(float(np.abs(r[0] - act.numpy()).max()), float(np.abs(r[1] - g.numpy()).max()))
    assert err < 1e-3, err
    print("wrote", a.out, "max |torch - onnx|", err, "iter", ck["iter"])


if __name__ == "__main__":
    main()
