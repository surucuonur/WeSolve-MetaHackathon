"""Sanity: does the probe harness reproduce the starting point's known
0/2/2 differs on captured inputs? Runs muse-spark/kernels/combine/
attempt-1 verbatim. Scratch only; never graded.
"""
import sys
sys.path.insert(0, "/Users/onur/Desktop/Github/Hackathon-muse/bench/grader")
import mlx.core as mx
import slots
from probe_barriers import ref, differs

SRC = open("/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/kernels/combine/attempt-1/block-0.metal").read()
fn = mx.fast.metal_kernel(name="probe_sp", input_names=["y", "weights", "shared"],
                          output_names=["out"], source=SRC)
for rows in (1, 8, 32):
    tot = 0
    for layer in slots.load_inputs(rows):
        y, w, s = layer["y"], layer["weights"], layer["shared"]
        out = fn(inputs=[y, w, s], grid=(rows * 768, 1, 1), threadgroup=(256, 1, 1),
                 output_shapes=[(rows, 768)], output_dtypes=[mx.bfloat16])[0]
        mx.eval(out)
        tot += differs(out, ref(y, w, s))
    print(f"rows={rows}: starting-point differs={tot}", flush=True)
