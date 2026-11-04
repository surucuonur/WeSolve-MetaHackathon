"""Stress: large random inputs (prefill-sized) for naive vs volatile.
A contracting kernel should show a few differs per 1.5M elements;
a barriered one must show 0. Scratch only; never graded.
"""
import sys
sys.path.insert(0, "/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/round-2/kernels/combine/attempt-2/probes")
import mlx.core as mx
from probe_barriers import VARIANTS, build, run, ref, differs

fns = {n: build(VARIANTS[n]) for n in ("naive", "volatile")}
for seed, scale in [(10, 1.0), (11, 1.0), (12, 20.0), (13, 0.05), (14, 1.0)]:
    rows = 2048
    mx.random.seed(seed)
    y = (mx.random.normal((rows, 4, 768)) * scale).astype(mx.bfloat16)
    w = mx.random.uniform(0.0, 1.0, (rows, 4)).astype(mx.float32)
    w = w / w.sum(axis=-1, keepdims=True) * 2.5
    s = (mx.random.normal((rows, 768)) * scale).astype(mx.bfloat16)
    mx.eval(y, w, s)
    r = ref(y, w, s)
    print(f"seed={seed} scale={scale}: " +
          " ".join(f"{n}={differs(run(fn, y, w, s), r)}" for n, fn in fns.items()),
          flush=True)
