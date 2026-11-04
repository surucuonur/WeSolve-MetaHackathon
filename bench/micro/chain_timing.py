"""Investigation: does timing independent calls hide an experts kernel's latency?

Until 2026-11-04 the grader timed a slot on 20 layers' calls with no data
between them, and waited for the GPU once: the GPU could overlap the calls.
In a decode step each layer's experts call waits for the layer before. This
times the reference experts and an attempt both ways, at 1, 8 and 32 rows:

  independent  as the grader did
  chained      each call's x is the previous call's output for its first
               expert, so it can't start until that call is done

  .venv/bin/python bench/micro/chain_timing.py muse-spark/kernels/experts/attempt-4

The output of 2026-11-04 is results/runs/2026-11-04-muse-round-1/console-chain.txt.
"""

import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "grader"))
import check_candidate as cc  # noqa: E402
import gpu_lock  # noqa: E402
import mlx.core as mx  # noqa: E402

import slots  # noqa: E402

REPEATS = 32


def main():
    gpu_lock.hold(f"chain_timing.py {sys.argv[1]}")
    weights = slots.load_expert_weights()
    names = slots.SLOTS["experts"][0]
    fns = {
        "reference": lambda **kw: {"y": slots.ref_experts(*[kw[n] for n in names])},
        "muse": cc.load(Path(sys.argv[1]), "experts")["experts"],
    }
    for rows in (1, 8, 32):
        layers = slots.load_inputs(rows, weights)
        for mode in ("independent", "chained"):
            for name, fn in fns.items():
                times = []
                for _ in range(REPEATS):
                    mx.synchronize()
                    t = time.perf_counter()
                    outs, x = [], layers[0]["x"]
                    for layer in layers:
                        args = {n: layer[n] for n in names}
                        if mode == "chained":
                            args["x"] = x
                        y = fn(**args)["y"]
                        x = y[:, 0, :]
                        outs.append(y)
                    mx.eval(outs)
                    times.append((time.perf_counter() - t) / len(layers) * 1e6)
                med = statistics.median(times[2:])  # the first two warm up
                print(f"rows {rows:2d} {mode:>11} {name:>9}: {med:7.1f} us per layer")


if __name__ == "__main__":
    main()
