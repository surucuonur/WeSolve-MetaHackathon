"""Is each slot's compiled reference bit-exact with its reference at the six G2 sizes?

An attempt's hand-off (`max_rows`, claude/adr/0011-decode-and-prefill.md) runs
the slot's reference under mx.compile, so G2 can only pass if compiling keeps
every output identical. This checks it with the grader's own G2 checks, at a
decode step's 1, 8 and 32 rows and a prefill chunk's 127, 1,016 and 2,048.

  .venv/bin/python bench/micro/compiled_exact.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "grader"))
import mlx.core as mx  # noqa: E402

import check_candidate as cc  # noqa: E402
import gpu_lock  # noqa: E402
import slots  # noqa: E402

CHECK = {"routing": cc.check_routing, "combine": cc.check_combine, "experts": cc.check_experts}


def compiled_fn(part):
    names, outs = slots.SLOTS[part]
    comp = mx.compile(slots.REFERENCE[part])

    def fn(**kw):
        out = comp(*[kw[n] for n in names])
        return dict(zip(outs, out if isinstance(out, (tuple, list)) else [out]))

    return fn


def main():
    gpu_lock.hold("compiled_exact.py")
    model = slots.load_model()
    weights = slots.model_expert_weights(model)
    sizes = [("rows", lambda r=r: slots.load_inputs(r, weights)) for r in cc.ROWS]
    sizes += [
        ("prefill rows", lambda r=r: slots.capture_prefill(model, r, weights))
        for r in slots.PROMPT_REQUESTS
    ]
    for kind, inputs in sizes:
        layers = inputs()
        for part in CHECK:
            res = [CHECK[part](compiled_fn(part), layer) for layer in layers]
            key = "rows_differ" if part == "routing" else "differ"
            print(
                f"{kind} {layers[0]['x'].shape[0]:5d} {part:8}: "
                f"exact {all(cc.passes(part, r) for r in res)}, {key} {sum(r.get(key, 0) for r in res)}",
                flush=True,
            )
        del layers


if __name__ == "__main__":
    main()
