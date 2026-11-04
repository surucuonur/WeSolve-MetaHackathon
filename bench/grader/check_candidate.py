"""Grade one kernel attempt: compile it, test it on captured inputs, time it.

The independent grader for the Muse agent's kernels (claude/muse-agent-brief.md).
G2 tests it on the inputs `slots.py capture` recorded from a real decode step
(1, 8 and 32 rows), and on every layer's inputs in a prefill, recorded as
it runs (127, 1,016 and 2,048 rows): every output must be identical to the
reference's, bit for bit, since only then are the model's tokens identical
(claude/adr/0010-exact-at-every-size.md). The timing runs at the same six
sizes and chains the 20 layers, so each call waits for the one before, as in
the model. The grader prints its verdict as JSON, and writes it to --out when
given. It writes into an attempt folder only when --out points there, as
the Muse agent's builders do with grade.json; load() reads only spec.json and
the block-N.metal files.

  .venv/bin/python bench/grader/check_candidate.py <attempt folder> <slot> [--out FILE]
      [--baseline [SLOT=]FOLDER ...]

The candidate and the compiled reference are timed in alternating repeats, so
drift in the GPU's clocks reaches both alike. At each size the candidate is
"faster" when it wins at least WINS_NEEDED of the REPEATS paired repeats (a
one-sided sign test at p < 0.01), "slower" when it loses that many, and
"same" otherwise; the same kernel timed twice differs by up to 5% between
medians, so comparing medians alone would pass on noise. `beats_compiled`
needs both phases, decode and prefill: faster at every size of one, slower
at none of the other (claude/adr/0011-decode-and-prefill.md).

--baseline times an earlier, already graded attempt in the same run and adds
`beats_baseline`, judged the same way. From round 2 on, the bar is the
previous round's final attempt for the slot
(claude/adr/0009-rounds-build-on-each-other.md). FOLDER alone is an attempt
for the same slot. For routing+combine, give either one routing+combine
attempt or `routing=FOLDER` and `combine=FOLDER`.
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
PHASES = {"decode": "rows ", "prefill": "prefill rows "}  # phase -> size label prefix
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


def differences(g, r):
    """Elements of *g* that aren't *r*'s bit for bit, and the largest gap in units
    in the last place (floored at |r| = 0.001); "at" is the flat index of it."""
    differ = (g != r) & ~(np.isnan(g) & np.isnan(r))
    out = {"differ": int(differ.sum()), "max_ulp": 0.0}
    if out["differ"]:
        with np.errstate(over="ignore", invalid="ignore"):
            over = np.nan_to_num(np.abs(g - r) / ulp_bf16(r), nan=1e9, posinf=1e9)
        out |= {"max_ulp": float(min(over.max(), 1e9)), "at": int(over.argmax())}
    return out


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
    # Exact, in order: combine sums the 4 pairs in this order.
    differ = (inds != ref_inds).any(axis=1) | (w != ref_w).any(axis=1)
    same_order = (inds == ref_inds).all(axis=1)
    rel = np.abs(w - ref_w)[same_order] / np.maximum(np.abs(ref_w[same_order]), 1e-12)
    out = {
        "rows_differ": int(differ.sum()),
        "rows_other_experts": int(
            sum(set(a) != set(b) for a, b in zip(inds.tolist(), ref_inds.tolist()))
        ),
        "max_rel_err": float(rel.max()) if rel.size else 0.0,
    }
    if out["rows_differ"]:
        r = int(differ.argmax())
        out["example"] = (
            f"row {r}: experts {inds[r].tolist()}, weights {w[r].tolist()}; "
            f"want {ref_inds[r].tolist()}, {ref_w[r].tolist()}"
        )
    return out


def check_combine(fn, layer):
    ref = slots.ref_combine(layer["y"], layer["weights"], layer["shared"])
    got = fn(y=layer["y"], weights=layer["weights"], shared=layer["shared"])["out"]
    if got.shape != ref.shape or got.dtype != mx.bfloat16:
        return {"error": f"out {got.shape} {got.dtype}; want {ref.shape} bfloat16"}
    r = np.array(ref.astype(mx.float32))
    g = np.array(got.astype(mx.float32))
    out = differences(g, r)
    if out["differ"]:
        i, j = np.unravel_index(out.pop("at"), g.shape)
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
    out = differences(g, r)
    if out["differ"]:
        i, j = np.unravel_index(out.pop("at"), g.shape)
        k, d = divmod(j, slots.D)
        out["example"] = (
            f"row {i} expert slot {k} column {d}: {g[i, j]:.6g}, want {r[i, j]:.6g}"
        )
    return out


def passes(part, s):
    if "error" in s:
        return False
    return s["rows_differ" if part == "routing" else "differ"] == 0


@mx.compile
def link(a, prev):
    """*a* unchanged, computed after *prev*: one small fused kernel."""
    return a + (prev.reshape(-1)[:1] * 0).astype(a.dtype)


def chained(fn, layers, names):
    """All 20 layers' calls, each one waiting for the one before, as in a decode step.

    A call's smallest input is linked to the previous call's last output, the
    same for every function timed. Without the wait, the GPU overlaps the
    calls and hides a slow call's latency; mx.depends orders only the graph.
    """
    small = min(names, key=lambda n: layers[0][n].size)
    outs = []
    for layer in layers:
        args = {n: layer[n] for n in names}
        if outs:
            args[small] = link(args[small], outs[-1][-1])
        out = fn(**args)
        if isinstance(out, dict):
            out = list(out.values())
        outs.append(list(out) if isinstance(out, (tuple, list)) else [out])
    return outs


def timed(fn, layers, names):
    """Median µs per layer: the 20 layers' chained calls, then one wait for the GPU."""
    for _ in range(3):
        mx.eval(chained(fn, layers, names))
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        mx.eval(chained(fn, layers, names))
        times.append((time.perf_counter() - t0) / len(layers))
    return round(statistics.median(times) * 1e6, 1)


def paired_times(fns, layers, names):
    """Per-repeat µs per layer of each function, in rotating order within each repeat."""
    calls = [lambda f=f: mx.eval(chained(f, layers, names)) for f in fns]
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


def compare(wins):
    """One size's verdict from the candidate's wins in REPEATS paired repeats."""
    if wins >= WINS_NEEDED:
        return "faster"
    if REPEATS - wins >= WINS_NEEDED:
        return "slower"
    return "same"


def judge(verdicts):
    """Each phase's verdict from its sizes' verdicts, and whether the candidate
    beats the bar: faster at every size of one phase, slower at no size of the other.

    A phase is "faster" when every size is, "slower" when any size is, and
    "not slower" otherwise.
    """
    phases = {}
    for phase, prefix in PHASES.items():
        sizes = [v for size, v in verdicts.items() if size.startswith(prefix)]
        phases[phase] = (
            "faster" if all(v == "faster" for v in sizes)
            else "slower" if "slower" in sizes
            else "not slower"
        )
    beats = "faster" in phases.values() and "slower" not in phases.values()
    return phases, beats


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
        model = slots.load_model()
        weights = slots.model_expert_weights(model) if slot == "experts" else None
        stats = {}
        sizes = [("rows", lambda r=r: slots.load_inputs(r, weights)) for r in ROWS]
        sizes += [
            ("prefill rows", lambda r=r: slots.capture_prefill(model, r, weights))
            for r in slots.PROMPT_REQUESTS
        ]
        for kind, inputs in sizes:
            layers = inputs()
            size = f"{kind} {layers[0]['x'].shape[0]}"
            for part, fn in fns.items():
                check = {
                    "routing": check_routing,
                    "combine": check_combine,
                    "experts": check_experts,
                }[part]
                per_layer = [check(fn, layer) for layer in layers]
                errors = [s["error"] for s in per_layer if "error" in s]
                if errors:
                    stats[f"{part} {size}"] = {"error": errors[0]}
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
                stats[f"{part} {size}"] = agg
            del layers
        result["g1"] = True
        result["correctness"] = stats
        result["g2"] = all(s.get("pass") for s in stats.values())
        if result["g2"]:
            timing = {}
            bars = ("compiled", "baseline") if base else ("compiled",)
            verdicts = {bar: {} for bar in bars}
            for kind, inputs in sizes:
                layers = inputs()
                size = f"{kind} {layers[0]['x'].shape[0]}"
                row = {"reference": 0.0}
                times = {name: [0.0] * REPEATS for name in ("candidate", *bars)}
                for part, fn in fns.items():
                    names = slots.SLOTS[part][0]
                    ref = slots.REFERENCE[part]
                    wrap = lambda f: lambda **kw: f(  # noqa: E731
                        *[kw[n] for n in names]
                    )
                    row["reference"] += timed(wrap(ref), layers, names)
                    together = {"candidate": fn, "compiled": wrap(mx.compile(ref))}
                    if base:
                        together["baseline"] = base[part]
                    paired = paired_times(list(together.values()), layers, names)
                    for name, t in zip(together, paired):
                        times[name] = [s + u for s, u in zip(times[name], t)]
                del layers
                row |= {name: statistics.median(t) for name, t in times.items()}
                for bar in bars:
                    wins = sum(c < b for c, b in zip(times["candidate"], times[bar]))
                    verdicts[bar][size] = compare(wins)
                    row |= {f"wins_vs_{bar}": wins, f"vs_{bar}": verdicts[bar][size]}
                timing[size] = {
                    k: round(v, 1) if isinstance(v, float) else v for k, v in row.items()
                }
            result["timing_us_per_layer"] = timing
            for bar in bars:
                result[f"phases_vs_{bar}"], result[f"beats_{bar}"] = judge(verdicts[bar])
    except Exception as e:  # report the failure to the Muse agent as G1
        result["error"] = short_error(e)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
