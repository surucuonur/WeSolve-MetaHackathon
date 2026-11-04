"""G2's checks at row counts the grader doesn't test.

check_candidate.py tests a slot at 1, 8 and 32 rows of a decode step and at
127, 1,016 and 2,048 rows of a prefill. A kernel that branches on the row
count, as MLX's own kernels do, can be exact at those six and differ in
between. This applies check_candidate's checks and pass rule to every layer's
slot inputs in the first N tokens of a one-request prefill, for each N in
SIZES: either side of MLX's switches (16 rows, where the experts' sort
starts, and 60, where gather_qmm_rhs takes over) and inside each row-count
regime an attempt names.

  .venv/bin/python bench/grader/check_sizes.py <out.json> <slot>=<attempt dir> ...
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import check_candidate as cc  # noqa: E402
import gpu_lock  # noqa: E402

import slots  # noqa: E402

SIZES = (2, 3, 5, 15, 16, 17, 31, 33, 40, 59, 60, 61, 64, 65, 100, 128, 129, 200,
         256, 257, 500, 768, 769, 1024, 1500)
CHECKS = {
    "routing": cc.check_routing,
    "combine": cc.check_combine,
    "experts": cc.check_experts,
}


def main():
    out_path = Path(sys.argv[1])
    attempts = [a.partition("=")[::2] for a in sys.argv[2:]]
    gpu_lock.hold("check_sizes.py " + " ".join(sys.argv[2:]))
    model = slots.load_model()
    weights = slots.model_expert_weights(model)
    fns = []
    for slot, folder in attempts:
        fns += [(f"{slot} {folder}", part, fn) for part, fn in cc.load(Path(folder), slot).items()]
    report = {name: {} for name, _, _ in fns}
    for rows in SIZES:
        layers = slots.capture_prefill(model, 1, weights, tokens=rows)
        for name, part, fn in fns:
            per_layer = [CHECKS[part](fn, layer) for layer in layers]
            ok = all(cc.passes(part, s) for s in per_layer)
            keys = [k for k in per_layer[0] if k not in ("example", "error")]
            agg = {k: (max if k.startswith("max") else sum)(s.get(k, 0) for s in per_layer) for k in keys}
            report[name][rows] = agg | {"pass": ok}
            print(f"{name} rows {rows:5d}: {'pass' if ok else 'FAIL'} {json.dumps(agg)}", flush=True)
        del layers
    out_path.write_text(json.dumps(report, indent=1) + "\n")
    failed = {n: [r for r, v in rs.items() if not v["pass"]] for n, rs in report.items()}
    print("sizes that fail:", json.dumps(failed))


if __name__ == "__main__":
    main()
