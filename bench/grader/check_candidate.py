"""Grade one kernel attempt: compile it, test it on captured inputs, time it.

The independent grader for the Muse agent's kernels (claude/muse-agent-brief.md).
It runs on the inputs `slots.py capture` recorded from a real decode step,
prints its verdict as JSON, and writes it to --out when given. It never
writes into the attempt folder, which holds only the Muse agent's files.

  .venv/bin/python bench/grader/check_candidate.py <attempt folder> <slot> [--out FILE]
      [--baseline [SLOT=]FOLDER ...]

--baseline times an earlier, already graded attempt in the same run as the
candidate, and adds `beats_baseline`. From round 2 on, the bar is the previous
round's final attempt for the slot (claude/adr/0009-rounds-build-on-each-other.md).
The two are timed in alternating repeats, so drift in the GPU's clocks reaches
both alike, and the candidate beats the baseline at a row count only if it is
faster in at least WINS_NEEDED of the REPEATS paired repeats: a one-sided sign
test at p < 0.01. The same kernel timed twice differs by up to 5% between
medians, so comparing medians alone would pass on noise. FOLDER alone is an
attempt for the same slot. For routing+combine, give either one
routing+combine attempt or `routing=FOLDER` and `combine=FOLDER`.
"""

import argparse
import json
import math
import statistics
import sys
import time
import traceback
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu_lock  # noqa: E402
import slots  # noqa: E402

ROWS = (1, 8, 32)
REPEATS = 30
WINS_NEEDED = next(  # 22 of 30
    k
    for k in range(REPEATS + 1)
    if sum(math.comb(REPEATS, i) for i in range(k, REPEATS + 1)) / 2**REPEATS < 0.01
)


def parts(slot):
    return ["routing", "combine"] if slot == "routing+combine" else [slot]


def load(attempt, slot):
    spec = json.loads((attempt / "spec.json").read_text())
    sources = [
        p.read_text()
        for p in sorted(attempt.glob("block-*.metal"), key=lambda p: int(p.stem[6:]))
    ]
    specs = spec if slot == "routing+combine" else {slot: spec}
    return {p: slots.build(specs[p], sources, p) for p in parts(slot)}


def baselines(values, slot):
    """The --baseline attempts' functions by part; they must cover the slot's parts."""
    fns = {}
    for value in values:
        name, sep, folder = value.rpartition("=")
        name = name if sep else slot
        if name not in slots.SLOTS:
            sys.exit(f"--baseline {value}: unknown slot {name!r}")
        try:
            fns |= load(Path(folder), name)
        except Exception as e:
            sys.exit(f"--baseline {value}: can't load it: {short_error(e)}")
    if sorted(fns) != sorted(parts(slot)):
        sys.exit(f"--baseline must cover {parts(slot)}, not {sorted(fns)}")
    return fns


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


def check_experts(fn, layer):
    names = slots.SLOTS["experts"][0]
    args = [layer[n] for n in names]
    ref = slots.ref_experts(*args)
    got = fn(**{n: layer[n] for n in names})["y"]
    if got.shape != ref.shape or got.dtype != mx.bfloat16:
        return {"error": f"y {got.shape} {got.dtype}; want {ref.shape} bfloat16"}
    r = np.array(ref.astype(mx.float32)).reshape(ref.shape[0], -1)
    g = np.array(got.astype(mx.float32)).reshape(ref.shape[0], -1)
    with np.errstate(over="ignore", invalid="ignore"):
        over = np.nan_to_num(np.abs(g - r) / ulp_bf16(r), nan=1e9, posinf=1e9)
    out = {"max_ulp": float(min(over.max(), 1e9)), "bad": int((over > 2).sum())}
    if out["bad"]:
        i, j = np.unravel_index(over.argmax(), over.shape)
        k, d = divmod(j, slots.D)
        out["example"] = (
            f"row {i} expert slot {k} column {d}: {g[i, j]:.6g}, want {r[i, j]:.6g}"
        )
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


def paired_times(fns, layers, names):
    """Per-repeat µs per layer of each function, in rotating order within each repeat."""
    calls = [
        lambda f=f: mx.eval([f(**{n: layer[n] for n in names}) for layer in layers])
        for f in fns
    ]
    for _ in range(3):
        for call in calls:
            call()
    times = [[] for _ in fns]
    for i in range(REPEATS):
        for j in [(i + k) % len(fns) for k in range(len(fns))]:
            t0 = time.perf_counter()
            calls[j]()
            times[j].append((time.perf_counter() - t0) / len(layers) * 1e6)
    return times


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
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("attempt", type=Path)
    ap.add_argument("slot", choices=tuple(slots.SLOTS))
    ap.add_argument("--out", type=Path, help="also write the verdict here")
    ap.add_argument(
        "--baseline",
        action="append",
        default=[],
        metavar="[SLOT=]FOLDER",
        help="an earlier attempt to time in the same run (repeat for routing+combine)",
    )
    args = ap.parse_args()
    attempt, slot = args.attempt, args.slot
    gpu_lock.hold(f"check_candidate.py {attempt} {slot}")
    base = baselines(args.baseline, slot) if args.baseline else None
    result = {"slot": slot, "g1": False, "g2": False, "beats_compiled": False}
    if base:
        result |= {
            "baseline": args.baseline,
            "beats_baseline": False,
            "wins_needed": f"{WINS_NEEDED} of {REPEATS} paired repeats",
        }
    try:
        fns = load(attempt, slot)
        weights = slots.load_expert_weights() if slot == "experts" else None
        stats = {}
        for rows in ROWS:
            layers = slots.load_inputs(rows, weights)
            for part, fn in fns.items():
                check = {
                    "routing": check_routing,
                    "combine": check_combine,
                    "experts": check_experts,
                }[part]
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
            timing, beats, beats_base = {}, True, True
            for rows in ROWS:
                layers = slots.load_inputs(rows, weights)
                row = {"reference": 0.0, "compiled": 0.0, "candidate": 0.0}
                base_t, cand_t = [0.0] * REPEATS, [0.0] * REPEATS
                for part, fn in fns.items():
                    names = slots.SLOTS[part][0]
                    ref = slots.REFERENCE[part]
                    wrap = lambda f: lambda **kw: f(
                        *[kw[n] for n in names]
                    )  # noqa: E731
                    row["reference"] += timed(wrap(ref), layers, names)
                    row["compiled"] += timed(wrap(mx.compile(ref)), layers, names)
                    if base:
                        b, c = paired_times([base[part], fn], layers, names)
                        base_t = [t + u for t, u in zip(base_t, b)]
                        cand_t = [t + u for t, u in zip(cand_t, c)]
                    else:
                        row["candidate"] += timed(fn, layers, names)
                if base:
                    row["candidate"] = statistics.median(cand_t)
                    row["baseline"] = statistics.median(base_t)
                    row["wins_vs_baseline"] = sum(c < b for b, c in zip(base_t, cand_t))
                    beats_base &= row["wins_vs_baseline"] >= WINS_NEEDED
                timing[f"rows {rows}"] = {k: round(v, 1) for k, v in row.items()}
                beats &= row["candidate"] < row["compiled"]
            result["timing_us_per_layer"] = timing
            result["beats_compiled"] = beats
            if base:
                result["beats_baseline"] = beats_base
    except Exception as e:  # report the failure to the Muse agent as G1
        result["error"] = short_error(e)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
