"""g_theta (12-64-64-6, relu) in a real inference runtime.

Round 2 item B/C. The compiled numbers in HANDOFF fact 12 span 261 ns
(-ffast-math) to 1,515 ns (strict) for the same MLP, and neither is a
deployment condition: -ffast-math changes float semantics globally and would
not ship, while the strict build is what a naive g++ gives. What would
actually deploy is an inference runtime, so this measures ONNX Runtime.

Two regimes, because they answer different questions:

  batch 1, serially dependent -> the per-call frame (CKF calls this per jump,
      and jump k+1 depends on jump k, so consecutive calls cannot overlap)
  batch 64/256/1024          -> the batched-throughput frame (PLAN phase 3)

Threading is pinned to 1 intra/inter op: ACTS CKF already runs one thread per
event, so an inference call inside it does not get a thread pool to itself.

Watch for per-call framework overhead. ORT's session-run path can cost
microseconds for a model this small, which would dominate the arithmetic
entirely -- if so, that is itself the result, and a hand-written kernel
becomes the deployment assumption.

Usage:
    python3 bench_gtheta_onnx.py
"""
import statistics as st
import time

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

D_IN, H, H2, D_OUT = 12, 64, 64, 6
REPS_B1 = 20000
REPS_BATCH = 2000
WARMUP = 2000


def build_model(path="/tmp/gtheta.onnx"):
    """12-64-64-6 MLP, relu, float32 -- same shape as bench_learned.cpp."""
    rng = np.random.default_rng(0)
    inits, nodes = [], []
    dims = [(D_IN, H), (H, H2), (H2, D_OUT)]
    prev = "input"
    for i, (a, b) in enumerate(dims):
        w = (rng.standard_normal((a, b)) * 0.1).astype(np.float32)
        bias = np.zeros(b, dtype=np.float32)
        inits += [numpy_helper.from_array(w, f"W{i}"),
                  numpy_helper.from_array(bias, f"B{i}")]
        nodes.append(helper.make_node("Gemm", [prev, f"W{i}", f"B{i}"],
                                      [f"gemm{i}"]))
        prev = f"gemm{i}"
        if i < len(dims) - 1:                      # no activation on output
            nodes.append(helper.make_node("Relu", [prev], [f"act{i}"]))
            prev = f"act{i}"

    graph = helper.make_graph(
        nodes, "gtheta",
        inputs=[helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, ["N", D_IN])],
        outputs=[helper.make_tensor_value_info(
            prev, TensorProto.FLOAT, ["N", D_OUT])],
        initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return path, prev


def make_session(path, threads=1):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def bench_batch1_serial(sess, out_name, reps=REPS_B1):
    """Each call's input is perturbed by the previous call's output, so the
    runtime cannot pipeline consecutive calls -- the CKF condition."""
    x = np.zeros((1, D_IN), dtype=np.float32)
    base = np.random.default_rng(1).standard_normal((1, D_IN)).astype(np.float32)
    run = sess.run
    for _ in range(WARMUP):
        run([out_name], {"input": base})
    acc = np.float32(0.0)
    t0 = time.perf_counter_ns()
    for _ in range(reps):
        np.add(base, acc * np.float32(1e-9), out=x)
        acc = run([out_name], {"input": x})[0][0, 0]
    t1 = time.perf_counter_ns()
    return (t1 - t0) / reps, float(acc)


def bench_batched(sess, out_name, batch, reps=REPS_BATCH):
    x = np.random.default_rng(2).standard_normal((batch, D_IN)).astype(np.float32)
    run = sess.run
    for _ in range(min(WARMUP, 200)):
        run([out_name], {"input": x})
    t0 = time.perf_counter_ns()
    for _ in range(reps):
        run([out_name], {"input": x})
    t1 = time.perf_counter_ns()
    per_call = (t1 - t0) / reps
    return per_call, per_call / batch


def bench_iobinding_b1(sess, out_name):
    """Lower-overhead path: pre-bound buffers, no per-call dict/alloc."""
    x = np.random.default_rng(3).standard_normal((1, D_IN)).astype(np.float32)
    xo = ort.OrtValue.ortvalue_from_numpy(x)
    io = sess.io_binding()
    io.bind_ortvalue_input("input", xo)
    io.bind_output(out_name)
    for _ in range(WARMUP):
        sess.run_with_iobinding(io)
    t0 = time.perf_counter_ns()
    for _ in range(REPS_B1):
        sess.run_with_iobinding(io)
    t1 = time.perf_counter_ns()
    return (t1 - t0) / REPS_B1


def main():
    path, out_name = build_model()
    sess = make_session(path, threads=1)
    print(f"ONNX Runtime {ort.__version__}, providers={sess.get_providers()}, "
          f"1 intra/inter-op thread")
    print(f"model: {D_IN}-{H}-{H2}-{D_OUT} relu, float32\n")

    reps = 5
    b1 = [bench_batch1_serial(sess, out_name)[0] for _ in range(reps)]
    iob = [bench_iobinding_b1(sess, out_name) for _ in range(reps)]
    print(f"{'regime':28s} {'ns/call':>12s} {'ns/sample':>12s}")
    print(f"{'batch 1, serial (run)':28s} {st.median(b1):12.0f} "
          f"{st.median(b1):12.0f}")
    print(f"{'batch 1, io_binding':28s} {st.median(iob):12.0f} "
          f"{st.median(iob):12.0f}")
    for batch in (64, 256, 1024):
        r = [bench_batched(sess, out_name, batch) for _ in range(3)]
        pc = st.median([a for a, _ in r])
        ps = st.median([b for _, b in r])
        print(f"{'batch ' + str(batch):28s} {pc:12.0f} {ps:12.1f}")

    print("\n  compare (same CPU, C++ batch 1 serial, HANDOFF fact 12):")
    print("    g_theta relu  -ffast-math 261 ns   strict 1,515 ns")
    print("    helix+g_theta -ffast-math 456 ns   strict 1,706 ns")
    print("    ACTS jump: 2,570 ns (const 2T) / 3,360 ns (ODD map)")


if __name__ == "__main__":
    main()
