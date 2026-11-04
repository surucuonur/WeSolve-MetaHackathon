"""Investigation: what removing work repeated in every decode step saves.

Three changes, each exact (the same numbers, fewer kernels), applied at runtime:

  bias_f32     each layer's expert_bias held in float32 instead of bfloat16, so
               the router's bias add stops converting it in every step
               (bfloat16 -> float32 is exact)
  lhs_indices  gather_qmm is given the row indices it otherwise builds with an
               arange in every call (MLX ops.cpp indices_or_default), made once
               per shape. Only for unsorted pairs (decode below 16 rows): MLX
               takes its sorted kernels only when no row indices are given
               (ops.cpp:5824-5825), so passing them there changes the kernel
  qkv_view     vllm only: sdpa_forward takes queries.squeeze(0) instead of
               queries[0] (likewise keys and values). In Python an integer
               index is a gather, a copy; squeeze is a view of the same values

Each variant's steady decode step period is measured as in breakdown.py's bare
mode, in shuffled order over repeats. Before timing, each variant's outputs are
compared with the unchanged model's, and one decode step's graph is exported
to count its operations.

Run from the repo root, one engine at a time:
  .venv/bin/python bench/micro/repeat_fixes.py plain results/runs/<date>-breakdown
  .venv/bin/python bench/micro/repeat_fixes.py vllm results/runs/<date>-breakdown
"""

import __future__
import argparse
import inspect
import json
import math
import random
import re
import statistics
import tempfile
import time
from collections import Counter
from pathlib import Path

import breakdown as bd  # same folder; sets the vLLM environment before importing it
import mlx.core as mx
from mlx_lm.models import mobilemoe

SETTINGS = [(1, 128), (8, 128), (32, 128)]
REPEATS = 3
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
ORIGINAL_SWITCH = mobilemoe.TransposedQuantizedSwitchLinear.__call__
ROWS = {}  # shape -> the row indices gather_qmm would build for it


def row_indices(shape):
    """reshape(arange(n), shape) as MLX builds it when lhs_indices is None."""
    if shape not in ROWS:
        ROWS[shape] = mx.arange(math.prod(shape), dtype=mx.uint32).reshape(shape)
        mx.eval(ROWS[shape])
    return ROWS[shape]


def switch_with_rows(self, x, indices, sorted_indices=False):
    if sorted_indices:
        return ORIGINAL_SWITCH(self, x, indices, sorted_indices)
    return mx.gather_qmm(
        x,
        self["weight"],
        self["scales"],
        self.get("biases"),
        lhs_indices=row_indices(x.shape[:-2]),
        rhs_indices=indices,
        transpose=False,
        group_size=self.group_size,
        bits=self.bits,
        mode=self.mode,
        sorted_indices=sorted_indices,
    )


def sdpa_with_views():
    """vllm-metal's sdpa_forward with [0] on q, k and v replaced by squeeze(0)."""
    from vllm_metal.attention.impls import sdpa

    src = inspect.getsource(sdpa.sdpa_forward)
    for name in ("queries", "keys", "values"):
        old = f"{name}[0].transpose(1, 0, 2)"
        if src.count(old) != 1:
            raise RuntimeError(f"sdpa_forward changed: {old!r} found {src.count(old)}x")
        src = src.replace(old, f"{name}.squeeze(0).transpose(1, 0, 2)")
    scope = {}
    flags = __future__.annotations.compiler_flag  # as sdpa.py compiles it
    code = compile(src, sdpa.__file__, "exec", flags=flags, dont_inherit=True)
    exec(code, sdpa.__dict__, scope)
    return scope["sdpa_forward"]


class Variants:
    def __init__(self, engine, model):
        self.engine = engine
        self.moes = [layer.feed_forward for layer in model.model.layers]
        self.bias = [m.expert_bias for m in self.moes]
        self.bias_f32 = [b.astype(mx.float32) for b in self.bias]
        mx.eval(self.bias_f32)
        self.names = ["base", "bias_f32", "lhs_indices"]
        if engine == "vllm":
            from vllm_metal.attention.impls import sdpa_wrapper

            self.wrapper = sdpa_wrapper
            self.sdpa = sdpa_wrapper.sdpa_forward
            self.sdpa_views = sdpa_with_views()
            self.names.append("qkv_view")
        self.names.append("all")

    def apply(self, name):
        on = set(self.names[1:-1]) if name == "all" else {name}
        for moe, b, b32 in zip(self.moes, self.bias, self.bias_f32):
            moe.expert_bias = b32 if "bias_f32" in on else b
        mobilemoe.TransposedQuantizedSwitchLinear.__call__ = (
            switch_with_rows if "lhs_indices" in on else ORIGINAL_SWITCH
        )
        if self.engine == "vllm":
            self.wrapper.sdpa_forward = (
                self.sdpa_views if "qkv_view" in on else self.sdpa
            )


def count_ops(arrs):
    with tempfile.NamedTemporaryFile(suffix=".dot") as f:
        mx.export_to_dot(f.name, *arrs)
        ops = Counter(re.findall(r'\[label ="([^"]+)"', Path(f.name).read_text()))
    kernels = {
        k: v for k, v in ops.items() if k not in VIEW_OPS and not k.startswith("View ")
    }
    return {"kernels": sum(kernels.values()), "ops": dict(kernels)}


def plain_checks(model, variants):
    """Logits after 8 decode steps, and one decode step's graph, per variant."""
    from mlx_lm.models.cache import make_prompt_cache

    out = {}
    for rows in (1, 32):
        logits = {}
        for name in variants.names:
            variants.apply(name)
            mx.random.seed(rows)
            ids = mx.random.randint(0, 100000, (rows, 128))
            cache = make_prompt_cache(model)
            y = bd.prefill(model, ids, cache)
            for _ in range(8):
                y = bd.greedy(model(y, cache=cache))
                mx.eval(y)
            step = model(y, cache=cache)
            out[(name, rows)] = count_ops([step])
            logits[name] = step
            mx.eval(step)
        for name in variants.names[1:]:
            out[(name, rows)]["exact"] = mx.array_equal(
                logits["base"], logits[name]
            ).item()
    variants.apply("base")
    return out


def vllm_checks(llm, variants):
    """Greedy tokens for a batch, and one steady decode step's graph, per variant."""
    from vllm import SamplingParams
    from vllm_metal.v1.model_runner import MetalModelRunner

    rng = random.Random(7)
    out = {}
    for rows in (1, 32):
        prompts = [
            {"prompt_token_ids": [rng.randrange(1000, 100000) for _ in range(128)]}
            for _ in range(rows)
        ]
        params = SamplingParams(max_tokens=24, temperature=0, ignore_eos=True)
        tokens = {}
        for name in variants.names:
            variants.apply(name)
            seen, found = [0], {}
            real_submit = MetalModelRunner._submit_paged_forward_outputs

            def submit(self, *args, **kwargs):
                m = bd.REC.meta or {}
                if m.get("rows") == rows and m.get("prefill") == 0:
                    seen[0] += 1
                    if seen[0] == 10:
                        real_eval = mx.async_eval

                        def export(*a):
                            found.update(count_ops(bd.arrays(a)))
                            return real_eval(*a)

                        mx.async_eval = export
                        try:
                            return real_submit(self, *args, **kwargs)
                        finally:
                            mx.async_eval = real_eval
                return real_submit(self, *args, **kwargs)

            MetalModelRunner._submit_paged_forward_outputs = submit
            try:
                bd.REC.reset(rows)
                res = llm.generate(prompts, params, use_tqdm=False)
                bd.REC.end_step()
            finally:
                MetalModelRunner._submit_paged_forward_outputs = real_submit
            tokens[name] = [list(r.outputs[0].token_ids) for r in res]
            out[(name, rows)] = dict(found)
        for name in variants.names[1:]:
            out[(name, rows)]["exact"] = tokens[name] == tokens["base"]
            out[(name, rows)].update(divergence(tokens[name], tokens["base"]))
    variants.apply("base")
    return out


def divergence(got, want):
    """How many sequences' greedy tokens differ, and the first step where any does."""
    first = [
        next((i for i, (a, b) in enumerate(zip(g, w)) if a != b), None)
        for g, w in zip(got, want)
    ]
    differ = [f for f in first if f is not None]
    return {"rows_differ": len(differ), "first_differ_step": min(differ, default=None)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("engine", choices=("plain", "vllm"))
    ap.add_argument("out", type=Path)
    ap.add_argument("--tokens", type=int, default=256, help="output tokens per run")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    log = args.out / f"fixes-{args.engine}.jsonl"

    if args.engine == "plain":
        from mlx_lm.utils import load_model

        model, _ = load_model(Path(bd.MODEL))
        mx.eval(model.parameters())
        variants = Variants("plain", model)
        checks = plain_checks(model, variants)
        run = lambda r, p, s: bd.run_plain(model, r, p, args.tokens, s)  # noqa: E731
    else:
        from vllm import LLM

        bd.install_vllm_step_hooks()
        llm = LLM(
            model=bd.MODEL,
            trust_remote_code=True,
            max_model_len=2048,
            max_num_seqs=32,
            max_num_batched_tokens=2048,
            kv_cache_memory_bytes=512 << 20,  # 32 rows x 384 tokens need 240 MiB
            enable_prefix_caching=False,
            generation_config="vllm",
        )
        run = lambda r, p, s: bd.run_vllm(llm, r, p, args.tokens, s)  # noqa: E731
        run(1, 16, 0)  # warm-up; captures the runner
        variants = Variants("vllm", bd.RUNNER[0].model)
        checks = vllm_checks(llm, variants)

    time_variants(args.engine, variants, checks, run, log, tokens=args.tokens)


def prompt_ms():
    """The last run's time from its first forward pass to its first one with no
    prompt tokens left: until every prompt was processed (vLLM's steps only)."""
    steps = [m for m, _ in bd.REC.steps if "t" in m]
    done = next((m["t"] for m in steps if m.get("prefill") == 0), None)
    return (done - steps[0]["t"]) * 1e3 if steps and done is not None else None


def time_variants(engine, variants, checks, run, log, settings=SETTINGS, repeats=REPEATS, tokens=None):
    """Log the checks, then time every variant at *settings* (rows, prompt tokens)
    in *repeats* shuffled repeats. Each run also records its wall time, prompt_ms()
    and *tokens*, the output tokens per request that *run* generates."""
    with log.open("a") as f:
        for (name, rows), c in checks.items():
            f.write(
                json.dumps({"kind": "check", "variant": name, "rows": rows, **c}) + "\n"
            )
            print(
                f"{engine} check {name:>11} rows {rows:2d}: {c.get('kernels')} kernels"
                f"{'' if name == 'base' else ', exact ' + str(c.get('exact'))}",
                flush=True,
            )

    plan = []
    for rep in range(repeats):
        block = [(v, r, p) for v in variants.names for r, p in settings]
        random.Random(100 + rep).shuffle(block)
        plan += [(rep, *b) for b in block]
    bd.REC.mode = "bare"
    for rep, name, rows, prompt in plan:
        variants.apply(name)
        bd.REC.reset(rows)
        before = bd.machine()
        t0 = time.perf_counter()
        run(rows, prompt, rows * 10000 + prompt)
        wall = (time.perf_counter() - t0) * 1e3
        record = bd.summarize()
        del record["spans"]
        record.update(
            wall_ms=wall,
            prompt_ms=prompt_ms(),
            tokens=tokens,
            kind="run",
            engine=engine,
            variant=name,
            rows=rows,
            prompt=prompt,
            rep=rep,
            machine_before=before,
            machine_after=bd.machine(),
        )
        with log.open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(
            f"{engine} rep{rep} {name:>11} rows {rows:2d} prompt {prompt}: "
            f"{record['period_median_ms']:7.3f} ms/step",
            flush=True,
        )
    variants.apply("base")

    print(f"\n{engine}: median step ms over {repeats} repeats (change against base)")
    runs = [json.loads(line) for line in log.read_text().splitlines()]
    for rows, prompt in settings:
        base = None
        for name in variants.names:
            meds = [
                r["period_median_ms"]
                for r in runs
                if r["kind"] == "run"
                and r["variant"] == name
                and r["rows"] == rows
                and r["prompt"] == prompt
                and r["engine"] == engine
            ]
            med = statistics.median(meds)
            base = med if name == "base" else base
            change = (
                ""
                if name == "base"
                else f" ({med - base:+.3f} ms, {(med / base - 1) * 100:+.1f}%)"
            )
            print(
                f"  rows {rows:2d} prompt {prompt:4d} {name:>11}: "
                f"{med:7.3f} [{min(meds):.3f}-{max(meds):.3f}]{change}"
            )


if __name__ == "__main__":
    main()
