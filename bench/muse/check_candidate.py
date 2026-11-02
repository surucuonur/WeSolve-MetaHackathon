"""Check one Muse Spark attempt: compile it, test it on captured inputs, time it.

Runs in its own process (kernel_loop.py gives it a time limit), on the
inputs slots.save_inputs captured from a real decode step. Writes
result.json into the attempt folder and prints it.

  .venv/bin/python bench/muse/check_candidate.py muse-spark/kernels/<slot>/attempt-N <slot>
"""

import json
import statistics
import sys
import time
import traceback
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import client  # noqa: E402
import slots  # noqa: E402

ROWS = (1, 8, 32)
REPEATS = 30


def parts(slot):
    return ["routing", "combine"] if slot == "routing+combine" else [slot]


def load(attempt, slot):
    spec = json.loads((attempt / "spec.json").read_text())
    sources = [
        client.read_verbatim(p)
        for p in sorted(attempt.glob("block-*.metal"), key=lambda p: int(p.stem[6:]))
    ]
    specs = spec if slot == "routing+combine" else {slot: spec}
    return {p: slots.build(specs[p], sources, p) for p in parts(slot)}


def ulp_bf16(r):
    """bfloat16's unit in the last place at |r|, floored at |r| = 0.001 (brief, G2)."""
    exp = np.floor(np.log2(np.maximum(np.abs(r), 1e-3)))
    return 2.0 ** (exp - 7)


def check_routing(fn, layer):
    ref_inds, ref_w = slots.ref_routing(layer["x"], layer["w_router"], layer["bias"])
    got = fn(x=layer["x"], w_router=layer["w_router"], bias=layer["bias"])
    inds, w = got["inds"], got["weights"]
    rows = layer["x"].shape[0]
    if inds.shape != (rows, slots.K) or w.shape != (rows, slots.K):
        return {
            "error": f"shapes inds {inds.shape}, weights {w.shape}; want ({rows}, 4)"
        }
    if inds.dtype != mx.uint32 or w.dtype != mx.float32:
        return {
            "error": f"dtypes inds {inds.dtype}, weights {w.dtype}; want uint32, float32"
        }
    inds, w = np.array(inds), np.array(w)
    ref_inds, ref_w = np.array(ref_inds), np.array(ref_w)
    if inds.max() >= slots.E:
        return {"error": f"expert index {int(inds.max())} out of range (60 experts)"}
    scores = np.array(mx.sigmoid(layer["x"].astype(mx.float32) @ layer["w_router"].T))
    biased = np.sort(scores + np.array(layer["bias"]), axis=-1)[:, ::-1]
    wrong, ties, worst, example = 0, 0, 0.0, None
    for r in range(rows):
        if set(inds[r]) != set(ref_inds[r]):
            if biased[r, 3] - biased[r, 4] < 1e-6:
                ties += 1
            else:
                wrong += 1
                example = example or (
                    f"row {r}: experts {sorted(inds[r].tolist())}, "
                    f"want {sorted(ref_inds[r].tolist())}"
                )
            continue
        ref = dict(zip(ref_inds[r], ref_w[r]))
        for e, v in zip(inds[r], w[r]):
            err = abs(v - ref[e]) / max(abs(ref[e]), 1e-12)
            if err > worst:
                worst = err
                if err > 1e-4:
                    example = f"row {r} expert {e}: weight {v:.7g}, want {ref[e]:.7g}"
    out = {
        "rows_wrong_experts": wrong,
        "near_tie_rows": ties,
        "max_rel_err": float(worst),
    }
    return out | ({"example": example} if example else {})


def check_combine(fn, layer):
    ref = slots.ref_combine(layer["y"], layer["weights"], layer["shared"])
    got = fn(y=layer["y"], weights=layer["weights"], shared=layer["shared"])["out"]
    if got.shape != ref.shape or got.dtype != mx.bfloat16:
        return {"error": f"out {got.shape} {got.dtype}; want {ref.shape} bfloat16"}
    r = np.array(ref.astype(mx.float32))
    g = np.array(got.astype(mx.float32))
    with np.errstate(over="ignore", invalid="ignore"):
        over = np.nan_to_num(np.abs(g - r) / ulp_bf16(r), nan=1e9, posinf=1e9)
    out = {"max_ulp": float(min(over.max(), 1e9)), "bad": int((over > 2).sum())}
    if out["bad"]:
        i, j = np.unravel_index(over.argmax(), over.shape)
        out["example"] = f"row {i} column {j}: {g[i, j]:.6g}, want {r[i, j]:.6g}"
    return out


def passes(part, s):
    if "error" in s:
        return False
    if part == "routing":
        return s["rows_wrong_experts"] == 0 and s["max_rel_err"] <= 1e-4
    return s["bad"] == 0


def timed(fn, layers, names):
    """Median µs per layer: all 20 layers' calls, then one wait for the GPU."""
    for _ in range(3):
        mx.eval([fn(**{n: layer[n] for n in names}) for layer in layers])
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        mx.eval([fn(**{n: layer[n] for n in names}) for layer in layers])
        times.append((time.perf_counter() - t0) / len(layers))
    return round(statistics.median(times) * 1e6, 1)


def short_error(e):
    """The exception, keeping only the compiler's error lines from Metal builds."""
    text = "".join(traceback.format_exception_only(type(e), e)).strip()
    lines = text.splitlines()
    if len(lines) > 1 and "Unable to build metal library" in lines[0]:
        keep = [lines[0]]
        for i, line in enumerate(lines):
            if "error:" in line:
                keep += lines[i : i + 3]
        text = "\n".join(keep)
    return text[:4000]


def main():
    attempt, slot = Path(sys.argv[1]), sys.argv[2]
    result = {"slot": slot, "g1": False, "g2": False, "beats_compiled": False}
    try:
        fns = load(attempt, slot)
        stats = {}
        for rows in ROWS:
            layers = slots.load_inputs(rows)
            for part, fn in fns.items():
                check = check_routing if part == "routing" else check_combine
                per_layer = [check(fn, layer) for layer in layers]
                errors = [s["error"] for s in per_layer if "error" in s]
                if errors:
                    stats[f"{part} rows {rows}"] = {"error": errors[0]}
                    continue
                agg = {
                    k: (max if k.startswith("max") else sum)(s[k] for s in per_layer)
                    for k in per_layer[0]
                    if k != "example"
                }
                agg["pass"] = all(passes(part, s) for s in per_layer)
                examples = [
                    f"layer {i}, {s['example']}"
                    for i, s in enumerate(per_layer)
                    if "example" in s
                ]
                if examples:
                    agg["worst_example"] = examples[0]
                stats[f"{part} rows {rows}"] = agg
        result["g1"] = True
        result["correctness"] = stats
        result["g2"] = all(s.get("pass") for s in stats.values())
        if result["g2"]:
            timing, beats = {}, True
            for rows in ROWS:
                layers = slots.load_inputs(rows)
                row = {"reference": 0.0, "compiled": 0.0, "candidate": 0.0}
                for part, fn in fns.items():
                    names = slots.SLOTS[part][0]
                    ref = slots.REFERENCE[part]
                    wrap = lambda f: lambda **kw: f(
                        *[kw[n] for n in names]
                    )  # noqa: E731
                    row["reference"] += timed(wrap(ref), layers, names)
                    row["compiled"] += timed(wrap(mx.compile(ref)), layers, names)
                    row["candidate"] += timed(fn, layers, names)
                timing[f"rows {rows}"] = {k: round(v, 1) for k, v in row.items()}
                beats &= row["candidate"] < row["compiled"]
            result["timing_us_per_layer"] = timing
            result["beats_compiled"] = beats
    except Exception as e:  # report the failure to Muse Spark as G1
        result["error"] = short_error(e)
    (attempt / "result.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
