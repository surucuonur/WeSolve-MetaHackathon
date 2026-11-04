"""Probe variants on the grader's captured decode inputs (rows 1/8/32,
all 20 layers). The starting point differs from the eager reference by
0/2/2 elements there; a working contraction barrier must give 0.
Scratch only; never graded.
"""
import sys
sys.path.insert(0, "/Users/onur/Desktop/Github/Hackathon-muse/bench/grader")
sys.path.insert(0, "/Users/onur/Desktop/Github/Hackathon-muse/muse-spark/round-2/kernels/combine/attempt-2/probes")
import mlx.core as mx
import slots
from probe_barriers import VARIANTS, build, run, ref, differs

only = sys.argv[1:] or list(VARIANTS)
fns = {}
for name in only:
    try:
        fns[name] = build(VARIANTS[name])
        print(f"[{name}] compiled OK", flush=True)
    except Exception as e:
        print(f"[{name}] COMPILE FAIL: {str(e)[:300]}", flush=True)

for rows in (1, 8, 32):
    layers = slots.load_inputs(rows)
    tot = {n: 0 for n in fns}
    for layer in layers:
        y, w, s = layer["y"], layer["weights"], layer["shared"]
        r = ref(y, w, s)
        for n, fn in fns.items():
            tot[n] += differs(run(fn, y, w, s), r)
    print(f"rows={rows}: " + " ".join(f"{n}={tot[n]}" for n in fns), flush=True)
