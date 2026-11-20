"""ACP's tracker (actor.npz: normalisation, MLP with ELU, clip +-20) as one ONNX graph, so the robot runs it in onnxruntime on a
single thread as SONIC runs its actor, not numpy (whose OpenBLAS starts a thread per core: 55-118 ms per step on PC2 next to the
student thread, 2026-11-19). Same weights; checked against the numpy MLP.

    python3 models/acp_depth/export_tracker_onnx.py ACP_DIR/actor.npz ACP_DIR/tracker.onnx
"""
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def build(z):
    mean = np.asarray(z["norm_mean"], np.float32)
    std = (np.sqrt(np.asarray(z["norm_var"], np.float64)) + 1e-5).astype(np.float32)
    inits = [numpy_helper.from_array(mean, "mean"), numpy_helper.from_array(std, "std"),
             numpy_helper.from_array(np.array(-10, np.float32), "lo10"), numpy_helper.from_array(np.array(10, np.float32), "hi10"),
             numpy_helper.from_array(np.array(-20, np.float32), "lo20"), numpy_helper.from_array(np.array(20, np.float32), "hi20")]
    nodes = [helper.make_node("Sub", ["obs", "mean"], ["c"]), helper.make_node("Div", ["c", "std"], ["n"]),
             helper.make_node("Clip", ["n", "lo10", "hi10"], ["x0"])]
    i, x = 0, "x0"
    while f"actor_w{i}" in z:
        inits += [numpy_helper.from_array(np.asarray(z[f"actor_w{i}"], np.float32), f"w{i}"),
                  numpy_helper.from_array(np.asarray(z[f"actor_b{i}"], np.float32), f"b{i}")]
        nodes.append(helper.make_node("Gemm", [x, f"w{i}", f"b{i}"], [f"g{i}"], transB=1))
        x = f"g{i}"
        if f"actor_w{i + 1}" in z:
            nodes.append(helper.make_node("Elu", [x], [f"e{i}"], alpha=1.0))
            x = f"e{i}"
        i += 1
    nodes.append(helper.make_node("Clip", [x, "lo20", "hi20"], ["action"]))
    d_in, d_out = mean.shape[0], np.asarray(z[f"actor_w{i - 1}"]).shape[0]
    g = helper.make_graph(nodes, "acp_tracker", [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, d_in])],
                          [helper.make_tensor_value_info("action", TensorProto.FLOAT, [1, d_out])], inits)
    return helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)


def main():
    z = np.load(sys.argv[1], allow_pickle=True)
    m = build(z)
    onnx.checker.check_model(m)
    onnx.save(m, sys.argv[2])
    import onnxruntime as ort
    sys.path.insert(0, "deploy/acp")
    sys.path.insert(0, "deploy/hiking")
    from acp_deploy.core import Mlp
    mlp = Mlp(z)
    mean, std = np.asarray(z["norm_mean"], np.float64), np.sqrt(np.asarray(z["norm_var"], np.float64)) + 1e-5
    s = ort.InferenceSession(sys.argv[2], providers=["CPUExecutionProvider"])
    rng, err = np.random.default_rng(0), 0.0
    for _ in range(200):
        obs = mean + std * rng.normal(size=mean.shape) * 2
        ref = np.clip(mlp(np.clip((obs - mean) / std, -10, 10)), -20, 20)
        out = s.run(None, {"obs": obs[None].astype(np.float32)})[0][0]
        err = max(err, float(np.abs(out - ref).max()))
    print("%s: max |onnx - numpy| over 200 random obs = %.2e (%s)" % (sys.argv[2], err, "PASS" if err < 1e-3 else "FAIL"))


if __name__ == "__main__":
    main()
