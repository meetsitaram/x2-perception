"""Export the fine-map depth student (TorchScript student.pt) to ONNX for the robot (PC2 runs onnxruntime, no torch), and
check it against the TorchScript model on a sequence of frames (the 13-frame memory included).

    python3 models/acp_depth/export_onnx.py runs/fine_v3/student.pt runs/fine_v3/student.onnx

ONNX graph: inputs depth (1,1,45,80) z-depth m (0 = no return), prop (1,99), cam_T (1,12), h (1,13,4,52,44) the previous
memory, first (1,) 1.0 on the first frame (the memory is then filled with the current grid, as the TorchScript h = empty
branch), else 0.0; outputs plan (1,187), track (1,63), h (1,13,4,52,44) the memory for the next frame, fine (1,52,44).
"""
from __future__ import annotations

import sys

import numpy as np
import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from fine_student import FineStudent  # noqa: E402


def _project_onnx(self, depth, cam_T):
    """FineStudent.project with index_add -> scatter_add (the same sums on a 1-D tensor; the ONNX exporter has no index_add)."""
    B = depth.shape[0]
    z = depth.reshape(B, -1)
    R, t = cam_T[:, :9].reshape(B, 3, 3), cam_T[:, 9:]
    P = self.rays[None] * (z / self.rays[:, 0][None])[..., None]
    P = torch.einsum("bij,bnj->bni", R, P) + t[:, None]
    i = torch.round((P[..., 0] - self.x0) / self.res).long()
    j = torch.round(P[..., 1] / self.res + (self.ny - 1) / 2).long()
    ok = (z > 0) & (i >= 0) & (i < self.nx) & (j >= 0) & (j < self.ny)
    idx = torch.where(ok, i * self.ny + j, torch.zeros_like(i)) + torch.arange(B, device=depth.device)[:, None] * (self.nx * self.ny)
    hz = torch.where(ok, P[..., 2], torch.full_like(z, -10.0))
    w = ok.float()
    n = B * self.nx * self.ny
    mx = torch.full((n,), -10.0).scatter_reduce(0, idx.flatten(), hz.flatten(), "amax")
    sm = torch.zeros(n).scatter_add(0, idx.flatten(), (P[..., 2] * w).flatten())
    cn = torch.zeros(n).scatter_add(0, idx.flatten(), w.flatten())
    seen = cn > 0
    mx = torch.where(seen, mx, torch.zeros_like(mx))
    mean = torch.where(seen, sm / cn.clamp(min=1.0), torch.zeros_like(sm))
    g = torch.stack(((mx + 0.7) / 0.3 * seen.float(), (mean + 0.7) / 0.3 * seen.float(), torch.log1p(cn), seen.float()), 0)
    return g.reshape(4, B, self.nx, self.ny).permute(1, 0, 2, 3)


class Graph(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, depth, prop, cam_T, h, first):
        m = self.m
        g = _project_onnx(m, depth, cam_T)
        f = first.view(-1, 1, 1, 1, 1)
        h = f * g[:, None].expand(g.shape[0], 13, 4, m.nx, m.ny) + (1.0 - f) * h
        h = torch.cat((h[:, 1:], g[:, None]), 1)
        past = [h[:, 12 - lag] for lag in m.lags]
        fine = m.decode(torch.cat([g] + past, 1), prop)
        plan, track = m.sample(fine)
        return plan, track, h, fine


def main(src, dst):
    ts = torch.jit.load(src, map_location="cpu").eval()
    sd = ts.state_dict()
    prop_dim = 99                                          # ACP proprioception (deploy core: w, g, q - q0, qd, a_prev)
    m = FineStudent(prop_dim)
    missing = m.load_state_dict(sd, strict=True)
    m.eval()
    g = Graph(m).eval()
    rng = np.random.default_rng(0)
    T = lambda a: torch.from_numpy(np.asarray(a, np.float32))
    cam = np.zeros(12, np.float32)
    R = np.array([[0.4695, 0, 0.8829], [0, 1, 0], [-0.8829, 0, 0.4695]], np.float32)   # 62 deg down, the trained pose
    cam[:9], cam[9:] = R.ravel(), [0.25, 0.0, 0.38]
    ex = (T(rng.uniform(0.4, 3.0, (1, 1, 45, 80))), T(rng.normal(0, 0.3, (1, prop_dim))), T(cam[None]),
          torch.zeros(1, 13, 4, m.nx, m.ny), torch.ones(1))
    torch.onnx.export(g, ex, dst, input_names=["depth", "prop", "cam_T", "h", "first"],
                      output_names=["plan", "track", "h_out", "fine"], opset_version=18, dynamo=False)
    import onnxruntime as ort
    so = ort.SessionOptions()            # onnxruntime's graph optimiser fails on this graph (an initializer check in its
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL      # constant folding, even at basic level): off
    s = ort.InferenceSession(dst, so, providers=["CPUExecutionProvider"])
    h_ts, h_ox, worst = torch.zeros(0), np.zeros((1, 13, 4, m.nx, m.ny), np.float32), 0.0
    for k in range(20):                                    # a walk: the depth and pose change, the memory fills
        depth = rng.uniform(0.4, 3.0, (1, 1, 45, 80)).astype(np.float32)
        depth[rng.uniform(size=depth.shape) < 0.05] = 0.0  # holes
        prop = rng.normal(0, 0.3, (1, prop_dim)).astype(np.float32)
        c = cam.copy()
        c[9] += 0.02 * k
        with torch.no_grad():
            p1, t1, h_ts, _ = ts(T(depth), T(prop), T(c[None]), h_ts)
        p2, t2, h_ox, _ = s.run(None, dict(depth=depth, prop=prop, cam_T=c[None], h=h_ox, first=np.array([1.0 if k == 0 else 0.0], np.float32)))
        worst = max(worst, float(np.abs(p1.numpy() - p2).max()), float(np.abs(t1.numpy() - t2).max()))
    print("prop_dim %d, ONNX %s: max |TorchScript - ONNX| over 20 frames = %.2e m (plan + track scans)" % (prop_dim, dst, worst))
    print("PARITY:", "PASS" if worst < 1e-4 else "FAIL")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
