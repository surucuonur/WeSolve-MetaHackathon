"""Tables for the decode-step breakdown, from bench/micro/breakdown.py's runs.

Prints markdown to stdout:

  .venv/bin/python bench/micro/breakdown_tables.py results/runs/<date>-breakdown
"""

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

PEAK_BANDWIDTH = 200e9
# Operations that only change how an array is viewed launch no kernel.
VIEW_OPS = {
    "Reshape",
    "Broadcast",
    "Transpose",
    "Slice",
    "Squeeze",
    "ExpandDims",
    "Flatten",
    "Unflatten",
    "AsStrided",
    "BroadcastAxes",
    "Depends",
    "CacheView",
}
RUNNER = [
    ("schedule", "vLLM scheduler: pick the step's requests"),
    ("execute_model", "runner: turn the scheduler output into a batch"),
    ("start_forward", "runner: inputs and attention metadata for the step"),
    ("target_forward", "runner: call the model"),
    ("submit", "mx.async_eval: MLX encodes the step's GPU work"),
    ("submit_wait", "inside mx.async_eval: waiting for the GPU to catch up"),
    ("sample_tokens", "runner: sampling entry"),
    ("sampling", "runner: deferred sampling bookkeeping"),
    ("sampler", "sampler graph (argmax)"),
    ("wait_gpu", "waiting for the previous step's tokens (GPU)"),
    ("update_from_output", "vLLM scheduler: record the step's tokens"),
    ("process_outputs", "vLLM output processing (detokenize, finish checks)"),
]
PIECES = [
    ("embed", "embedding lookup"),
    ("mask", "attention mask"),
    ("layer", "layer glue: two residual adds"),
    ("input_norm", "input norm"),
    ("attention", "attention glue"),
    ("qkv_prep", "q/k/v prep glue"),
    ("q_proj", "q projection"),
    ("k_proj", "k projection"),
    ("v_proj", "v projection"),
    ("q_norm", "q norm"),
    ("k_norm", "k norm"),
    ("rope", "RoPE glue"),
    ("rope_kernel", "RoPE kernels"),
    ("metadata", "attention metadata"),
    ("contiguous", "q/k/v contiguous copies"),
    ("kv_write", "cache write"),
    ("attn_kernel", "attention kernel"),
    ("o_proj", "o projection"),
    ("post_norm", "post-attention norm"),
    ("moe", "MoE glue"),
    ("router", "router: cast, product, sigmoid"),
    ("topk", "top-4: bias, argpartition, gather"),
    ("weights", "weight normalisation"),
    ("experts", "routed experts glue: activation, sort"),
    ("expert_gate_proj", "expert gate products"),
    ("expert_up_proj", "expert up products"),
    ("expert_down_proj", "expert down products"),
    ("combine", "combine: weight, sum, cast, add shared"),
    ("shared_expert", "shared expert glue: activation"),
    ("shared_gate_proj", "shared gate product"),
    ("shared_up_proj", "shared up product"),
    ("shared_down_proj", "shared down product"),
    ("final_norm", "final norm"),
    ("lm_head", "output layer"),
]
MODEL_SPANS = {name for name, _ in PIECES}
SETTINGS = [(1, 128), (8, 128), (32, 128), (32, 1024)]


def kernels(ops):
    return sum(
        n for op, n in ops.items() if op not in VIEW_OPS and not op.startswith("View ")
    )


def load(folder):
    runs, headers = defaultdict(list), {}
    for path in sorted(folder.glob("runs-*.jsonl")):
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if r["kind"] == "process":
                headers[r["engine"]] = r
            elif r["kind"] == "run":
                runs[(r["engine"], r["mode"], r["rows"], r["prompt"])].append(r)
    return runs, headers


def period(runs, key, pick=statistics.median):
    """The repeats' median step periods, combined by *pick*: the median, or min
    for the least disturbed repeat (paging only ever adds time)."""
    rs = runs.get(key, [])
    return pick([r["period_median_ms"] for r in rs]) if rs else None


def mean_period(runs, key):
    """Mean step period, comparable with the span sums, which are means."""
    steps = [p for r in runs.get(key, []) for p in r["period_ms"]]
    return statistics.mean(steps) if steps else None


def spans(runs, key):
    """Per-step span averages, averaged over the repeats."""
    out = defaultdict(lambda: defaultdict(float))
    rs = runs.get(key, [])
    for r in rs:
        for name, s in r["spans"].items():
            for f, v in s.items():
                out[name][f] += v / len(rs)
    return out


def fmt(x, digits=2):
    return "" if x is None else f"{x:.{digits}f}"


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |"]
    lines.append(
        "|" + "|".join("---:" if i else "---" for i in range(len(header))) + "|"
    )
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def step_periods(runs):
    rows = []
    for engine in ("plain", "vllm"):
        for r, p in SETTINGS:
            cells = []
            for mode in ("bare", "off", "cpu"):
                rs = runs.get((engine, mode, r, p), [])
                med = [x["period_median_ms"] for x in rs]
                cells.append(
                    f"{statistics.median(med):.2f} ({min(med):.2f}–{max(med):.2f})"
                    if med
                    else ""
                )
            rows.append([engine, r, p, *cells])
    return table(["Engine", "Rows", "Prompt", "bare ms", "off ms", "cpu ms"], rows)


def runner_cpu(runs):
    rows = []
    cols = [(e, r, p) for e in ("plain", "vllm") for r, p in SETTINGS]
    data = {c: spans(runs, (c[0], "cpu", c[1], c[2])) for c in cols}
    periods = {c: mean_period(runs, (c[0], "cpu", c[1], c[2])) for c in cols}
    for name, what in RUNNER:
        cells = []
        for c in cols:
            span = data[c].get("submit" if name == "submit_wait" else name)
            if span is None:
                cells.append("")
            elif name == "submit":  # its CPU time is the encoding
                cells.append(fmt(span["cpu_ms"]))
            elif name == "submit_wait":
                cells.append(fmt(span["self_ms"] - span["cpu_ms"]))
            else:
                cells.append(fmt(span["self_ms"]))
        rows.append([f"`{name}`", what, *cells])
    model = [
        sum(v["self_ms"] for k, v in data[c].items() if k in MODEL_SPANS) for c in cols
    ]
    rows.append(
        [
            "model pieces",
            "building the model's graph (sum of the pieces below)",
            *map(fmt, model),
        ]
    )
    traced = [sum(v["self_ms"] for v in data[c].values()) for c in cols]
    rows.append(
        [
            "(not in a span)",
            "step period minus all spans",
            *[fmt(periods[c] - t) if periods[c] else "" for c, t in zip(cols, traced)],
        ]
    )
    rows.append(
        ["step period", "mean of the steady steps", *map(fmt, periods.values())]
    )
    header = ["Span", "What it is", *[f"{e} {r}×{p}" for e, r, p in cols]]
    return table(header, rows)


def model_cpu(runs):
    cols = [(e, r, p) for r, p in ((1, 128), (32, 128)) for e in ("plain", "vllm")]
    data = {c: spans(runs, (c[0], "cpu", c[1], c[2])) for c in cols}
    rows = []
    for name, what in PIECES:
        if not any(name in data[c] for c in cols):
            continue
        rows.append(
            [
                f"`{name}`",
                what,
                *[
                    fmt(data[c][name]["self_ms"] * 1e3, 0) if name in data[c] else ""
                    for c in cols
                ],
            ]
        )
    total = [
        sum(v["self_ms"] for k, v in data[c].items() if k in MODEL_SPANS) * 1e3
        for c in cols
    ]
    rows.append(["total", "", *[fmt(t, 0) for t in total]])
    return table(
        [
            "Piece",
            "What it is",
            *[f"{e} {r} row{'s' if r > 1 else ''} µs" for e, r, p in cols],
        ],
        rows,
    )


def census(runs):
    cols = [(e, r, p) for r, p in ((1, 128), (32, 128)) for e in ("plain", "vllm")]
    data = {}
    for c in cols:
        rs = runs.get((c[0], "sync", c[1], c[2]), [])
        data[c] = rs[0]["census"] if rs else {}
    rows = []
    for name, what in [
        ("(outside)", "outside any span"),
        *[(n, w) for n, w in RUNNER],
        *PIECES,
    ]:
        counts = [kernels(data[c].get(name, {})) for c in cols]
        if not any(counts):
            continue
        ops = defaultdict(int)
        for c in cols:
            for op, n in data[c].get(name, {}).items():
                if op not in VIEW_OPS and not op.startswith("View "):
                    ops[op] = max(ops[op], n)
        kinds = ", ".join(
            f"{op} {n}" for op, n in sorted(ops.items(), key=lambda kv: -kv[1])
        )
        rows.append([f"`{name}`", *counts, kinds])
    rows.append(
        ["total", *[sum(kernels(v) for v in data[c].values()) for c in cols], ""]
    )
    return table(
        [
            "Piece",
            *[f"{e} {r} row{'s' if r > 1 else ''}" for e, r, p in cols],
            "Kernel operations (most in any column)",
        ],
        rows,
    )


def serialized(runs, headers, setting):
    r_, p_ = setting
    rows = []
    for name, what in [*RUNNER, *PIECES]:
        cells = []
        for engine in ("plain", "vllm"):
            rs = runs.get((engine, "sync", r_, p_), [])
            cal = headers.get(engine, {}).get("calibration", {})
            if not rs or name not in rs[0]["spans"]:
                cells += ["", "", ""]
                continue
            s = rs[0]["spans"][name]
            gpu = max(
                0.0, s["wait_ms"] - s["work_syncs"] * cal.get("one_kernel_wait_ms", 0.0)
            )
            cells += [fmt(s["encode_ms"]), fmt(gpu), fmt(s["work_syncs"], 0)]
        if any(cells):
            rows.append([f"`{name}`", *cells])
    header = [
        "Piece",
        "plain encode ms",
        "plain GPU ms",
        "plain syncs",
        "vllm encode ms",
        "vllm GPU ms",
        "vllm syncs",
    ]
    return table(header, rows)


def gain(runs, engine, piece, rows, pick=statistics.median):
    """What the step saves without *piece*, at 128-token prompts.

    At 1 row the knock-out's bookkeeping is a visible part of the step, so the
    control run (same bookkeeping, piece still runs) is the reference; at 32
    rows it is the run with every wrapper idle.
    """
    ko = period(runs, (engine, f"ko:{piece}", rows, 128), pick)
    ref = period(runs, (engine, f"kc:{piece}" if rows == 1 else "off", rows, 128), pick)
    return ref - ko if ko is not None and ref is not None else None


# Knock-out groups, each with every span that runs inside it.
ATTENTION_SPANS = {
    "attention",
    "qkv_prep",
    "q_proj",
    "k_proj",
    "v_proj",
    "q_norm",
    "k_norm",
    "rope",
    "rope_kernel",
    "metadata",
    "contiguous",
    "kv_write",
    "attn_kernel",
    "o_proj",
}
EXPERT_SPANS = {"experts", "expert_gate_proj", "expert_up_proj", "expert_down_proj"}
SHARED_SPANS = {
    "shared_expert",
    "shared_gate_proj",
    "shared_up_proj",
    "shared_down_proj",
}
ROUTING_SPANS = {"router", "topk", "weights"}
GROUPS = [
    ("attention", "attention, all of it", ATTENTION_SPANS),
    ("qkv_proj", "q, k and v projections", {"q_proj", "k_proj", "v_proj"}),
    ("qk_norm", "q and k norms", {"q_norm", "k_norm"}),
    ("rope", "RoPE: glue and kernels", {"rope", "rope_kernel"}),
    ("contiguous", "q/k/v copies before the kernel", {"contiguous"}),
    ("kv_write", "cache write", {"kv_write"}),
    ("attn_kernel", "attention kernel", {"attn_kernel"}),
    ("o_proj", "o projection", {"o_proj"}),
    ("layer_norms", "input and post-attention norms", {"input_norm", "post_norm"}),
    (
        "moe",
        "mixture of experts, all of it",
        {"moe", "combine", *ROUTING_SPANS, *EXPERT_SPANS, *SHARED_SPANS},
    ),
    ("routing", "router, top-4 selection, weights", ROUTING_SPANS),
    ("experts", "routed experts", EXPERT_SPANS),
    ("combine", "weighting and adding the outputs", {"combine"}),
    ("shared_expert", "shared expert", SHARED_SPANS),
    ("lm_head", "output layer", {"lm_head"}),
]


def pieces(runs, rows):
    engines = ("plain", "vllm")
    census = {}
    for e in engines:
        rs = runs.get((e, "sync", rows, 128), [])
        census[e] = rs[0]["census"] if rs else None
    cpu = {e: spans(runs, (e, "cpu", rows, 128)) for e in engines}
    out = []
    for name, what, members in GROUPS:
        cells = []
        for e in engines:
            if census[e] is None:
                cells.append("")
            else:
                cells.append(sum(kernels(census[e].get(s, {})) for s in members))
        for e in engines:
            us = sum(cpu[e][s]["self_ms"] for s in members if s in cpu[e]) * 1e3
            cells.append(fmt(us, 0) if cpu[e] else "")
        cells += [fmt(gain(runs, e, name, rows)) for e in engines]
        out.append([f"`{name}`", what, *cells])
    header = [
        "Piece",
        "What it is",
        "Kernels plain",
        "Kernels vLLM",
        "CPU µs plain",
        "CPU µs vLLM",
        "Saved ms plain",
        "Saved ms vLLM",
    ]
    return table(header, out)


def knockouts(runs, pick=statistics.median):
    names = sorted(
        {m[3:] for (_, m, _, _) in runs if m.startswith("ko:")},
        key=lambda k: list(KO_ORDER).index(k) if k in KO_ORDER else 99,
    )
    rows = []
    for k in names:
        cells = []
        for engine in ("plain", "vllm"):
            off1, off32 = (period(runs, (engine, "off", r, 128), pick) for r in (1, 32))
            gain1, gain32 = (gain(runs, engine, k, r, pick) for r in (1, 32))
            cells += [
                fmt(gain1),
                f"{gain1 / off1 * 100:.0f}%" if gain1 is not None and off1 else "",
                fmt(gain32),
                f"{gain32 / off32 * 100:.0f}%" if gain32 is not None and off32 else "",
            ]
        rows.append([f"`{k}`", *cells])
    header = [
        "Removed",
        "plain 1 row ms",
        "%",
        "plain 32 rows ms",
        "%",
        "vllm 1 row ms",
        "%",
        "vllm 32 rows ms",
        "%",
    ]
    return table(header, rows)


KO_ORDER = [
    "attention",
    "qkv_proj",
    "qk_norm",
    "rope",
    "contiguous",
    "kv_write",
    "attn_kernel",
    "o_proj",
    "layer_norms",
    "moe",
    "routing",
    "experts",
    "combine",
    "shared_expert",
    "lm_head",
]


def machine(runs):
    rows = []
    for engine in ("plain", "vllm"):
        rs = [r for rs in runs.values() for r in rs if r["engine"] == engine]
        if not rs:
            continue
        pages = [
            r["machine_after"]["swapouts"] - r["machine_before"]["swapouts"] for r in rs
        ]
        secs = sum(r["machine_after"]["time"] - r["machine_before"]["time"] for r in rs)
        docker = sum(
            r["machine_before"]["docker_vm"] or r["machine_after"]["docker_vm"]
            for r in rs
        )
        swap = [r["machine_before"]["swap_used_mb"] for r in rs]
        rows.append(
            [
                engine,
                len(rs),
                f"{secs / 60:.0f}",
                f"{statistics.median(pages):.0f}",
                max(pages),
                f"{sum(pages) / secs:.0f}",
                docker,
                f"{min(swap) / 1024:.1f}–{max(swap) / 1024:.1f}",
            ]
        )
    return table(
        [
            "Engine",
            "Runs",
            "Minutes",
            "Swap-outs per run (median, pages)",
            "Most in one run",
            "Pages/s",
            "Runs with Docker's VM up",
            "Swap used (GB)",
        ],
        rows,
    )


def repeats(runs):
    rows = []
    for engine in ("plain", "vllm"):
        for mode in ("bare", "off", "cpu"):
            for r_, p_ in SETTINGS:
                rs = sorted(
                    runs.get((engine, mode, r_, p_), []), key=lambda r: r["rep"]
                )
                if not rs:
                    continue
                meds = [r["period_median_ms"] for r in rs]
                spread = (max(meds) - min(meds)) / statistics.median(meds) * 100
                cells = ", ".join(
                    f"{r['period_median_ms']:.2f} "
                    f"({r['machine_after']['swapouts'] - r['machine_before']['swapouts']})"
                    for r in rs
                )
                rows.append([engine, mode, r_, p_, cells, f"{spread:.1f}%"])
    return table(
        [
            "Engine",
            "Mode",
            "Rows",
            "Prompt",
            "Each repeat's median ms (swap-outs in pages)",
            "Spread",
        ],
        rows,
    )


def census_totals(runs):
    rows = []
    for engine in ("plain", "vllm"):
        for r_, p_ in SETTINGS:
            rs = runs.get((engine, "sync", r_, p_), [])
            if not rs:
                continue
            ops = defaultdict(int)
            for counts in rs[0]["census"].values():
                for op, n in counts.items():
                    ops[op] += n
            kern = {op: n for op, n in ops.items() if op not in VIEW_OPS}
            kern = {op: n for op, n in kern.items() if not op.startswith("View ")}
            views = sum(ops.values()) - sum(kern.values())
            kinds = ", ".join(
                f"{op} {n}" for op, n in sorted(kern.items(), key=lambda kv: -kv[1])
            )
            rows.append([engine, r_, p_, sum(kern.values()), views, kinds])
    return table(
        ["Engine", "Rows", "Prompt", "Kernel operations", "View operations", "By type"],
        rows,
    )


def serialized_totals(runs, headers):
    rows = []
    for engine in ("plain", "vllm"):
        cal = headers.get(engine, {}).get("calibration", {})
        for r_, p_ in SETTINGS:
            rs = runs.get((engine, "sync", r_, p_), [])
            if not rs:
                continue
            s = rs[0]["spans"].values()
            enc = sum(v["encode_ms"] for v in s)
            wait = sum(v["wait_ms"] for v in s)
            syncs = sum(v["work_syncs"] for v in s)
            build = sum(v["self_ms"] for v in s)
            gpu = wait - syncs * cal.get("one_kernel_wait_ms", 0.0)
            rows.append(
                [
                    engine,
                    r_,
                    p_,
                    fmt(rs[0]["period_median_ms"]),
                    fmt(build),
                    fmt(enc),
                    fmt(wait),
                    fmt(syncs, 0),
                    fmt(gpu),
                    fmt(period(runs, (engine, "off", r_, p_))),
                ]
            )
    return table(
        [
            "Engine",
            "Rows",
            "Prompt",
            "Serialized step ms",
            "CPU build",
            "Encoding",
            "Waiting",
            "Syncs with work",
            "GPU (waiting less the sync cost)",
            "Pipelined step (off)",
        ],
        rows,
    )


def fixes(folder):
    """repeat_fixes.py's results: kernel operations, exactness, step periods."""
    out = []
    for path in sorted(folder.glob("fixes-*.jsonl")):
        recs = [json.loads(line) for line in path.read_text().splitlines()]
        checks = {(r["variant"], r["rows"]): r for r in recs if r["kind"] == "check"}
        runs = [r for r in recs if r["kind"] == "run"]
        names = list(dict.fromkeys(v for v, _ in checks))
        settings = sorted({r["rows"] for r in runs})
        rows = []
        for name in names:
            cells = []
            for n in (1, 32):
                c = checks.get((name, n))
                if c is None:
                    cells.append("")
                elif name == "base":
                    cells.append(str(c["kernels"]))
                else:
                    same = "identical" if c["exact"] else "DIFFERENT"
                    cells.append(f"{c['kernels']}, outputs {same}")
            for n in settings:
                meds, base = (
                    [
                        r["period_median_ms"]
                        for r in runs
                        if r["rows"] == n and r["variant"] == v
                    ]
                    for v in (name, "base")
                )
                m = statistics.median(meds)
                change = statistics.median(meds) - statistics.median(base)
                cells.append(
                    f"{m:.3f} [{min(meds):.3f}–{max(meds):.3f}]"
                    + ("" if name == "base" else f", {change:+.3f}")
                )
            rows.append([f"`{name}`", *cells])
        header = [
            "Change",
            "Kernel operations, 1 row",
            "Kernel operations, 32 rows",
            *[f"{n} row{'s' if n > 1 else ''}: ms [range], change" for n in settings],
        ]
        out += [f"\n{path.stem.split('-', 1)[1]}:\n", table(header, rows)]
    return "\n".join(out)


def distinct(runs):
    rows = []
    for engine in ("plain", "vllm"):
        for r, p in SETTINGS:
            rs = runs.get((engine, "sync", r, p), [])
            if rs and rs[0].get("distinct_experts"):
                d = rs[0]["distinct_experts"]
                rows.append([engine, r, p, f"{statistics.mean(d):.1f}", min(d), max(d)])
    return table(
        ["Engine", "Rows", "Prompt", "Distinct experts per layer", "Min", "Max"], rows
    )


# What each knock-out piece must read and compute, from the model's shapes.
PEAK_COMPUTE = 5.2e12  # FLOP/s, the M1 Pro's 16-core GPU
LAYERS, EXPERTS, TOP_K = 20, 60, 4
HIDDEN, Q_HEADS = 768, 12 * 64
KV_BYTES = (
    2 * 4 * 64 * 2
)  # keys and values, 4 heads × 64 × bfloat16, per token per layer
INT4_BYTES = 0.5 + 2 * 2 / 32  # 4-bit weight, bfloat16 scale and bias per group of 32
ROUTER_BYTES = 4  # the router weight is float32
PROJ = ("gate_proj", "up_proj", "down_proj")


def demands(w, rows, ctx, distinct):
    """Per step: bytes each piece must read (weights it uses, and the cache for
    the attention kernel) and its arithmetic. Activations are not counted."""
    expert = sum(w[f"expert_{n}"] for n in PROJ) / (EXPERTS * LAYERS)
    shared = sum(w[f"shared_{n}"] for n in PROJ)
    qkv = w["q_proj"] + w["k_proj"] + w["v_proj"]
    out = {
        "qkv_proj": (qkv, 2 * rows * qkv / INT4_BYTES),
        "attn_kernel": (ctx * LAYERS * KV_BYTES, 4 * Q_HEADS * ctx * LAYERS),
        "o_proj": (w["o_proj"], 2 * rows * w["o_proj"] / INT4_BYTES),
        "routing": (w["router"], 2 * rows * w["router"] / ROUTER_BYTES),
        "experts": (
            sum(distinct) * expert,
            2 * rows * TOP_K * LAYERS * expert / INT4_BYTES,
        ),
        "shared_expert": (shared, 2 * rows * shared / INT4_BYTES),
        "lm_head": (w["embed"], 2 * rows * w["embed"] / INT4_BYTES),
    }
    total = lambda names: tuple(sum(out[n][i] for n in names) for i in (0, 1))
    out["attention"] = total(("qkv_proj", "attn_kernel", "o_proj"))
    out["moe"] = total(("routing", "experts", "shared_expert"))
    out["step"] = total(("attention", "moe", "lm_head"))
    return out, rows * TOP_K * LAYERS * expert


def bandwidth(runs, headers):
    """Achieved bandwidth and compute of each piece where it runs: what it must
    read and compute, over what the step saves without it."""
    w = headers["plain"]["weight_bytes"]
    names = ["attention", "qkv_proj", "attn_kernel", "o_proj", "moe", "routing"]
    names += ["experts", "shared_expert", "lm_head", "step"]
    labels = {n: what for n, what, _ in GROUPS} | {"step": "whole step (bare)"}
    out = []
    for rows in (1, 32):
        # At 1 row vLLM's step is CPU-bound: its savings are CPU time, not GPU.
        engines = ("plain",) if rows == 1 else ("plain", "vllm")
        lines = []
        for name in names:
            cells = [labels[name]]
            for engine in engines:
                sync = runs[(engine, "sync", rows, 128)][0]
                ctx = statistics.mean(
                    r["ctx_mean"] for r in runs[(engine, "off", rows, 128)]
                )
                need, pairs = demands(w, rows, ctx, sync["distinct_experts"])
                nbytes, flops = need[name]
                if name == "step":
                    saved = period(runs, (engine, "bare", rows, 128))
                else:
                    saved = gain(runs, engine, name, rows)
                if engine == "plain":
                    cells += [fmt(nbytes / 1e6, 1), fmt(flops / 1e9, 2)]
                    cells.append(fmt(nbytes / PEAK_BANDWIDTH * 1e3, 2))
                if saved is None:
                    cells += ["", "", ""]
                    continue
                cells += [
                    fmt(saved),
                    f"{nbytes / saved / 1e6:.0f}",
                    f"{flops / (saved / 1e3) / PEAK_COMPUTE * 100:.0f}",
                ]
            lines.append(cells)
        header = ["Piece", "MB to read", "GFLOP", "ms at peak bandwidth"]
        for engine in engines:
            name = {"plain": "plain", "vllm": "vLLM"}[engine]
            header += [f"Saved ms {name}", f"GB/s {name}", f"% peak compute {name}"]
        out += [
            f"\n{rows} row{'s' if rows > 1 else ''}, 128-token prompts "
            f"(bytes from plain MLX's step; routed experts read once per pair: "
            f"{pairs / 1e6:.0f} MB):\n",
            table(header, lines),
        ]
    return "\n".join(out)


def step_bandwidth(runs, headers):
    """The whole step at every setting: what it must read, over its bare period."""
    w = headers["plain"]["weight_bytes"]
    lines = []
    for engine in ("plain", "vllm"):
        for rows, prompt in SETTINGS:
            sync = runs[(engine, "sync", rows, prompt)][0]
            ctx = statistics.mean(
                r["ctx_mean"] for r in runs[(engine, "off", rows, prompt)]
            )
            need, _ = demands(w, rows, ctx, sync["distinct_experts"])
            nbytes, flops = need["step"]
            bound = max(nbytes / PEAK_BANDWIDTH, flops / PEAK_COMPUTE) * 1e3
            ms = period(runs, (engine, "bare", rows, prompt))
            gbs = nbytes / ms / 1e6
            lines.append(
                [
                    engine,
                    rows,
                    prompt,
                    f"{statistics.mean(sync['distinct_experts']):.1f}",
                    fmt(nbytes / 1e6, 0),
                    fmt(ms),
                    f"{gbs:.0f} ({gbs / PEAK_BANDWIDTH * 1e11:.0f}%)",
                    f"{flops / (ms / 1e3) / PEAK_COMPUTE * 100:.0f}%",
                    fmt(bound),
                    f"{ms / bound:.1f}",
                ]
            )
    header = ["Engine", "Rows", "Prompt", "Distinct experts / layer", "MB to read"]
    header += ["Step ms (bare)", "GB/s (% of peak)", "% of peak compute"]
    header += ["Lower bound ms", "Times the bound"]
    return table(header, lines)


def main():
    folder = Path(sys.argv[1])
    runs, headers = load(folder)
    out = [
        "# Decode-step breakdown tables",
        f"Generated by `bench/micro/breakdown_tables.py` from `{folder}`.",
    ]
    for engine, h in sorted(headers.items()):
        out.append(
            f"\n{engine}: calibration {json.dumps({k: round(v, 4) for k, v in h['calibration'].items()})}; free memory at start {h.get('free_percent')}%"
        )
    out += ["\n## Machine during the runs", machine(runs)]
    out += [
        "\n## Step period by mode (median of the repeats, range in brackets)",
        step_periods(runs),
    ]
    out += ["\n## Each repeat", repeats(runs)]
    out += [
        "\n## Time per step in each runner span, cpu mode (ms, mean of the steady steps)",
        runner_cpu(runs),
    ]
    out += [
        "\n## CPU time per step to build each piece's graph, cpu mode (µs)",
        model_cpu(runs),
    ]
    out += [
        "\n## Kernel operations per step, one counted step in sync mode",
        census(runs),
    ]
    out += ["\n## Operations per step by type", census_totals(runs)]
    out += [
        "\n## Serialized step against the sum of its pieces, sync mode (ms per step)",
        serialized_totals(runs, headers),
    ]
    for setting in ((1, 128), (32, 128), (32, 1024)):
        out += [
            f"\n## Serialized cost per step, sync mode, {setting[0]} rows × {setting[1]}-token prompts",
            serialized(runs, headers, setting),
        ]
    out += [
        "\n## What removing a piece saves per step (1 row: control minus knock-out; 32 rows: off minus knock-out)",
        knockouts(runs),
    ]
    out += [
        "\n## The same, from each mode's fastest repeat (the least disturbed by paging)",
        knockouts(runs, min),
    ]
    for rows in (1, 32):
        out += [
            f"\n## Each piece at {rows} row{'s' if rows > 1 else ''}, 128-token prompts: "
            "kernel operations (sync mode), CPU time to build it (cpu mode), "
            "and what the step saves without it (knock-outs)",
            pieces(runs, rows),
        ]
    out += ["\n## Distinct experts in the counted step", distinct(runs)]
    out += [
        "\n## Achieved bandwidth and compute of each piece where it runs "
        "(what it must read and compute, over what the step saves without it; "
        f"peaks {PEAK_BANDWIDTH / 1e9:.0f} GB/s and {PEAK_COMPUTE / 1e12:.1f} TFLOP/s)",
        bandwidth(runs, headers),
    ]
    out += [
        "\n## The whole step's achieved bandwidth and compute "
        "(what it must read and compute, over its bare period)",
        step_bandwidth(runs, headers),
    ]
    if any(folder.glob("fixes-*.jsonl")):
        out += [
            "\n## Work repeated in every step: exact changes timed by repeat_fixes.py "
            "(median of the repeats' median step)",
            fixes(folder),
        ]
    print("\n".join(out))


if __name__ == "__main__":
    main()
