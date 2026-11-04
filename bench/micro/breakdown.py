"""Investigation: break a MobileMoE-S decode step into pieces and time each one.

Each process runs one engine on the int4 model and appends one JSON line per
run to runs-<engine>.jsonl in the output folder:

  plain   the mlx_lm port with its own KV cache, pipelined like mlx_lm generate
  vllm    vLLM in process: vllm-metal's model runner, paged attention, sampler

Wrappers installed at runtime give each piece of the step a named span (the
model's layers, attention and expert parts, and in vLLM the runner's own
steps). Every run uses one mode:

  bare        only the step-boundary hooks: the step period as served
  off         all wrappers installed but idle: the baseline for knock-outs
  cpu         spans timed, nothing else changed: CPU time of each piece
  sync        each span evaluates its inputs on entry and its outputs on
              exit, so its kernels are encoded and run on their own: encoding
              (CPU) and GPU time per piece. One step also exports each piece's
              graph (mx.export_to_dot) to count its operations.
  ko:<piece>  as off, but the piece (or group of pieces) is not run: each call
              returns the output it gave the first time it ran with inputs
              of that shape, tied to its current inputs with mx.depends so
              they are still computed. What the step gains without it.
  kc:<piece>  the control for ko:<piece>: the same bookkeeping and ties, but
              the piece still runs. At 1 row the step is CPU-bound, so the
              gain from removing a piece is the kc time minus the ko time.

The model and engine code are not changed; outputs of knock-out runs are
meaningless, only their timing counts.

Run from the repo root with .venv/bin/python, one engine at a time:
  .venv/bin/python bench/micro/breakdown.py plain results/runs/<date>-breakdown
  .venv/bin/python bench/micro/breakdown.py vllm results/runs/<date>-breakdown
"""

import argparse
import json
import os
import random
import re
import statistics
import subprocess
import tempfile
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
# vllm-metal's startup check that the KV cache fits refuses 1 GiB whenever this
# Mac pages; the runs record swap-outs instead
os.environ.setdefault("VLLM_METAL_KV_COMMIT_PROBE", "0")

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import numpy as np  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402
from mlx_lm.models import mobilemoe  # noqa: E402
from mlx_lm.models.cache import KVCache, make_prompt_cache  # noqa: E402
from mlx_lm.models.rope_utils import Llama3RoPE  # noqa: E402
from mlx_lm.models.switch_layers import SwitchGLU  # noqa: E402

MODEL = "models/MobileMoE-S-QAT-mlx-bfloat16"
GRID = [(1, 128), (8, 128), (32, 128), (32, 1024)]  # (decode rows, prompt tokens)
KNOCKOUT_ROWS = (1, 32)
KNOCKOUTS = {  # name -> the spans it removes
    "attention": {"attention"},
    "qkv_proj": {"q_proj", "k_proj", "v_proj"},
    "qk_norm": {"q_norm", "k_norm"},
    "rope": {"rope", "rope_kernel"},
    "kv_write": {"kv_write"},  # vllm only: the plain cache would stop growing
    "contiguous": {"contiguous"},  # vllm only
    "attn_kernel": {"attn_kernel"},
    "o_proj": {"o_proj"},
    "layer_norms": {"input_norm", "post_norm"},
    "moe": {"moe"},
    "routing": {"router", "topk", "weights"},
    "experts": {"experts"},
    "combine": {"combine"},
    "shared_expert": {"shared_expert"},
    "lm_head": {"lm_head"},
}
VLLM_ONLY = {"kv_write", "contiguous"}
SKIP = 5  # steady steps dropped at the start of each run
PREFILL_TOKENS = 2048  # plain engine: prompt tokens per forward pass, as vLLM's budget
# Spans that can wait for the GPU also record their CPU (thread) time.
CPU_TIMED = {"submit", "wait_gpu"}
CENSUS_STEP = 10  # the steady step whose operations are counted in sync mode
MAIN_THREAD = threading.get_ident()


def tie(out, deps):
    """*out* with every array made to depend on *deps* (no kernel)."""
    if isinstance(out, mx.array):
        return mx.depends(out, deps)
    if isinstance(out, (list, tuple)):
        return type(out)(tie(o, deps) for o in out)
    return out


def first_shape(args):
    """Shape of the first array argument: it sets the shape of a piece's output."""
    for a in args:
        if isinstance(a, mx.array):
            return a.shape
    return ()


def arrays(*objs):
    """Every mx.array inside call arguments or a return value."""
    found, todo = [], list(objs)
    while todo:
        o = todo.pop()
        if isinstance(o, mx.array):
            found.append(o)
        elif isinstance(o, nn.Module):  # a dict subclass: never walk weights
            continue
        elif isinstance(o, (list, tuple)):
            todo.extend(o)
        elif isinstance(o, dict):
            todo.extend(o.values())
    return found


class Frame:
    __slots__ = ("name", "start", "child", "sync")

    def __init__(self, name):
        self.name, self.start, self.child, self.sync = (
            name,
            time.perf_counter(),
            0.0,
            0.0,
        )


class Recorder:
    """Named spans, nested, summed per engine step."""

    def __init__(self):
        self.mode = "bare"
        self.knockout = set()  # spans that replay instead of running
        self.control = False  # kc: replay bookkeeping, but run the piece
        self.work_sync_ms = 0.0  # syncs shorter than this had nothing to run
        self.reset(0)

    def reset(self, rows):
        self.rows = rows
        self.stack = []
        self.meta = None
        self.stats = None  # name -> [calls, self_ms, encode_ms, wait_ms, syncs, cpu_ms]
        self.steps = []
        self.census = defaultdict(Counter)
        self.census_on = self.census_done = False
        self.steady_seen = 0
        self.distinct = []
        self.replays = {}
        self.nth = Counter()

    def begin_step(self, **meta):
        self.end_step()
        self.meta = dict(meta)
        self.stats = defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0, 0.0])
        self.nth.clear()

    def end_step(self):
        if self.meta is not None:
            self.steps.append((self.meta, dict(self.stats)))
        if self.census_on:
            self.census_on, self.census_done = False, True
        self.meta = self.stats = None

    def maybe_census(self, rows, prefill):
        """In sync mode, count the operations of one steady step."""
        if self.mode != "sync" or self.census_done or rows != self.rows or prefill:
            return
        self.steady_seen += 1
        self.census_on = self.steady_seen == CENSUS_STEP

    def replay(self, name, fn, args, kwargs):
        """Knock-out: the first output for inputs of this shape, without running.

        Keyed by the first array argument's shape, not every input's: the
        plain engine's attention reads a key/value cache that grows each step.
        """
        deps = arrays(args, kwargs)
        key = (name, self.nth[name], first_shape(args))
        self.nth[name] += 1
        if self.control or key not in self.replays:
            out = fn(*args, **kwargs)
            if self.control:
                return tie(out, deps)
            mx.eval(arrays(out))
            self.replays[key] = out
            return out
        return tie(self.replays[key], deps)

    def call(self, name, fn, args, kwargs):
        stats = self.stats
        if name in self.knockout and stats is not None:
            return self.replay(name, fn, args, kwargs)
        if (
            stats is None
            or self.mode not in ("cpu", "sync")
            or threading.get_ident() != MAIN_THREAD
        ):
            return fn(*args, **kwargs)
        if self.mode == "sync":  # the parent's pending work is the parent's
            self._sync(self.stack[-1] if self.stack else None, arrays(args, kwargs))
        cpu = time.thread_time() if name in CPU_TIMED else None
        frame = Frame(name)
        self.stack.append(frame)
        try:
            out = fn(*args, **kwargs)
            if self.mode == "sync":
                self._sync(frame, arrays(out))
            return out
        finally:
            self.stack.pop()
            total = time.perf_counter() - frame.start
            if self.stack:
                self.stack[-1].child += total
            s = stats[name]
            s[0] += 1
            s[1] += (total - frame.child - frame.sync) * 1e3
            if cpu is not None:
                s[5] += (time.thread_time() - cpu) * 1e3

    def _sync(self, frame, arrs):
        if not arrs:
            return
        name = frame.name if frame else "(outside)"
        if self.census_on:
            with tempfile.NamedTemporaryFile(suffix=".dot") as f:
                mx.export_to_dot(f.name, *arrs)
                ops = re.findall(r'\[label ="([^"]+)"', Path(f.name).read_text())
            self.census[name].update(ops)
        t0 = time.perf_counter()
        mx.async_eval(arrs)
        t1 = time.perf_counter()
        mx.eval(arrs)
        t2 = time.perf_counter()
        s = self.stats[name]
        s[2] += (t1 - t0) * 1e3
        s[3] += (t2 - t1) * 1e3
        s[4] += (t2 - t0) * 1e3 > self.work_sync_ms
        if frame is not None:
            frame.sync += t2 - t0


REC = Recorder()
ROLES = {}  # id(module) -> span name, for module classes shared by several pieces
ROLE_MODULES = defaultdict(list)
STEP_PATCHES, PIECE_PATCHES = [], []


def patch(owner, attr, make, into):
    """Replace owner.attr with make(original); remembered in *into*."""
    raw = vars(owner).get(attr, None) if isinstance(owner, type) else None
    if raw is None:
        raw = getattr(owner, attr)
    new = classmethod(make(raw.__func__)) if isinstance(raw, classmethod) else make(raw)
    setattr(owner, attr, new)
    into.append((owner, attr, raw))


def unpatch(patches):
    while patches:
        owner, attr, raw = patches.pop()
        setattr(owner, attr, raw)


def spanned(name):
    """Time calls as span *name*, a string or a function of the call's arguments."""

    def make(fn):
        def wrapper(*args, **kwargs):
            label = name(*args) if callable(name) else name
            return REC.call(label, fn, args, kwargs)

        return wrapper

    return make


def role(module, *args, **kwargs):
    return ROLES.get(id(module), type(module).__name__)


def build_roles(model):
    ROLES.clear()
    ROLE_MODULES.clear()

    def add(module, name):
        ROLES[id(module)] = name
        ROLE_MODULES[name].append(module)

    for layer in model.model.layers:
        attn = getattr(layer.self_attn, "_inner", layer.self_attn)
        for n in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"):
            add(getattr(attn, n), n)
        add(layer.input_layernorm, "input_norm")
        add(layer.post_attention_layernorm, "post_norm")
        moe = layer.feed_forward
        add(moe.router, "router")
        for n in ("gate_proj", "up_proj", "down_proj"):
            add(getattr(moe.shared_expert, n), "shared_" + n)
            add(getattr(moe.experts, n), "expert_" + n)
    add(model.model.norm, "final_norm")
    add(model.model.embed_tokens, "embed")


def weight_bytes():
    """Bytes of the weights behind each role, summed over the 20 layers."""
    return {
        name: sum(v.nbytes for m in mods for _, v in tree_flatten(m.parameters()))
        for name, mods in ROLE_MODULES.items()
    }


# The port's MoE.__call__ (mobilemoe.py:217-232), split into spans. Same
# operations in the same order; check_moe() compares the outputs.
ORIGINAL_MOE = mobilemoe.MoE.__call__


def _router(moe, x):
    return mx.sigmoid(moe.router(x.astype(mx.float32)))


def _topk(moe, scores):
    k = moe.top_k
    biased = scores + moe.expert_bias
    inds = mx.argpartition(-biased, kth=k - 1, axis=-1)[..., :k]
    return inds, mx.take_along_axis(scores, inds, axis=-1)


def _weights(moe, scores):
    if moe.norm_topk_prob:
        scores = scores / (scores.sum(axis=-1, keepdims=True) + 1e-20)
    return scores * moe.routed_scaling_factor


def _combine(y, scores, dtype):
    return (y * scores[..., None]).sum(axis=-2).astype(dtype)


def moe_call(self, x):
    scores = REC.call("router", _router, (self, x), {})
    inds, scores = REC.call("topk", _topk, (self, scores), {})
    scores = REC.call("weights", _weights, (self, scores), {})
    if REC.census_on:
        REC.distinct.append(int(np.unique(np.array(inds)).size))
    y = self.experts(x, inds)
    y = REC.call("combine", _combine, (y, scores, x.dtype), {})
    return REC.call("combine", mx.add, (y, self.shared_expert(x)), {})


def install_model_hooks(engine):
    p = PIECE_PATCHES
    patch(mobilemoe.DecoderLayer, "__call__", spanned("layer"), p)
    patch(mobilemoe.MoE, "__call__", lambda fn: spanned("moe")(moe_call), p)
    patch(SwitchGLU, "__call__", spanned("experts"), p)
    patch(mobilemoe.MLP, "__call__", spanned("shared_expert"), p)
    patch(mobilemoe.TransposedQuantizedSwitchLinear, "__call__", spanned(role), p)
    patch(nn.QuantizedLinear, "__call__", spanned(role), p)
    patch(nn.RMSNorm, "__call__", spanned(role), p)
    patch(mobilemoe.QKNorm, "__call__", spanned(role), p)
    patch(Llama3RoPE, "__call__", spanned("rope_kernel"), p)
    patch(nn.QuantizedEmbedding, "__call__", spanned("embed"), p)
    patch(nn.QuantizedEmbedding, "as_linear", spanned("lm_head"), p)
    patch(mobilemoe, "create_attention_mask", spanned("mask"), p)
    if engine == "plain":
        patch(mobilemoe.Attention, "__call__", spanned("attention"), p)
        patch(KVCache, "update_and_fetch", spanned("kv_write"), p)
        patch(mobilemoe, "scaled_dot_product_attention", spanned("attn_kernel"), p)
        return
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.engine.output_processor import OutputProcessor
    from vllm_metal.attention.impls import sdpa, sdpa_wrapper
    from vllm_metal.metal import get_ops
    from vllm_metal.v1.decode_pipeline import DecodePipeline
    from vllm_metal.v1.model_runner import MetalModelRunner
    from vllm_metal.v1.sampling_batch import SamplingBatch

    wrapper = sdpa_wrapper.SDPAPagedAttentionWrapper
    patch(wrapper, "__call__", spanned("attention"), p)
    patch(sdpa, "prepare_sdpa_qkv", spanned("qkv_prep"), p)
    patch(sdpa, "apply_attention_rope", spanned("rope"), p)
    patch(sdpa, "_kernel_metadata", spanned("metadata"), p)
    patch(mx, "contiguous", spanned("contiguous"), p)
    ops = get_ops()
    patch(ops, "reshape_and_cache", spanned("kv_write"), p)
    patch(ops, "paged_attention_primitive", spanned("attn_kernel"), p)
    patch(MetalModelRunner, "execute_model", spanned("execute_model"), p)
    patch(MetalModelRunner, "_target_forward", spanned("target_forward"), p)
    patch(MetalModelRunner, "_submit_paged_forward_outputs", spanned("submit"), p)
    patch(MetalModelRunner, "sample_tokens", spanned("sample_tokens"), p)
    patch(MetalModelRunner, "_submit_deferred_decode_sample", spanned("sampling"), p)
    patch(SamplingBatch, "native_decode_tokens", spanned("sampler"), p)
    patch(DecodePipeline, "_resolve_pending", spanned("wait_gpu"), p)
    patch(Scheduler, "schedule", spanned("schedule"), p)
    patch(Scheduler, "update_from_output", spanned("update_from_output"), p)
    patch(OutputProcessor, "process_outputs", spanned("process_outputs"), p)


RUNNER = []


def install_vllm_step_hooks():
    """Step boundaries and step metadata; installed once, before vLLM starts."""
    from vllm.v1.engine.core import EngineCore
    from vllm_metal.v1.model_runner import MetalModelRunner

    def step_boundary(fn):
        def wrapper(self, *args, **kwargs):
            REC.begin_step()
            return fn(self, *args, **kwargs)

        return wrapper

    def start_forward(fn):
        def wrapper(self, batch, prefill_reqs, decode_reqs, scheduler_output):
            if not RUNNER:
                RUNNER.append(self)
            prefill = sum(len(r.token_ids) for r in prefill_reqs)
            if REC.meta is not None:
                REC.meta.update(
                    t=time.perf_counter(),
                    rows=len(decode_reqs),
                    prefill=prefill,
                    ctx=sum(len(state.token_ids) for _, state in decode_reqs),
                )
                REC.maybe_census(len(decode_reqs), prefill)
            args = (self, batch, prefill_reqs, decode_reqs, scheduler_output)
            return REC.call("start_forward", fn, args, {})

        return wrapper

    patch(EngineCore, "step_with_batch_queue", step_boundary, STEP_PATCHES)
    patch(MetalModelRunner, "_start_paged_forward", start_forward, STEP_PATCHES)


def greedy(logits):
    return mx.argmax(logits[:, -1, :], axis=-1)[:, None]


def to_list(a):
    return a.tolist()


def prefill(model, ids, cache):
    """Run the prompts *ids* into *cache* and return the first new tokens.

    The prompts run in chunks of at most PREFILL_TOKENS tokens, like vLLM's
    token budget, and only the last position is scored, as in mlx_lm generate:
    scoring all 32 x 1,024 positions at once needs 8.4 GB of logits.
    """
    chunk = max(1, PREFILL_TOKENS // ids.shape[0])
    end = ids.shape[1] - 1
    for i in range(0, end, chunk):
        model(ids[:, i : min(i + chunk, end)], cache=cache)
        mx.eval([c.state for c in cache])
    return greedy(model(ids[:, -1:], cache=cache))


def run_plain(model, rows, prompt, tokens, seed):
    mx.clear_cache()  # the previous run's buffers
    mx.random.seed(seed)
    cache = make_prompt_cache(model)
    y = prefill(model, mx.random.randint(0, 100000, (rows, prompt)), cache)
    mx.eval(y)
    for _ in range(tokens - 1):
        REC.begin_step(t=time.perf_counter(), rows=rows, prefill=0)
        REC.meta["ctx"] = rows * cache[0].offset
        REC.maybe_census(rows, 0)
        logits = REC.call("target_forward", model, (y,), {"cache": cache})
        ny = REC.call("sampling", greedy, (logits,), {})
        REC.call("submit", mx.async_eval, (ny,), {})
        REC.call("wait_gpu", to_list, (y,), {})
        y = ny
    REC.end_step()
    y.tolist()


def run_vllm(llm, rows, prompt, tokens, seed):
    from vllm import SamplingParams

    rng = random.Random(seed)
    prompts = [
        {"prompt_token_ids": [rng.randrange(1000, 100000) for _ in range(prompt)]}
        for _ in range(rows)
    ]
    params = SamplingParams(max_tokens=tokens, temperature=0, ignore_eos=True)
    llm.generate(prompts, params, use_tqdm=False)
    REC.end_step()


def summarize():
    """Per-step averages over the run's steady decode steps."""
    steps = REC.steps
    ok = {
        i
        for i, (m, _) in enumerate(steps)
        if m.get("rows") == REC.rows and m.get("prefill") == 0 and "t" in m
    }
    keep = sorted(i for i in ok if i - 1 in ok and i + 1 in ok)[SKIP:]
    if not keep:
        raise RuntimeError("no steady decode steps")
    periods = [(steps[i + 1][0]["t"] - steps[i][0]["t"]) * 1e3 for i in keep]
    totals = defaultdict(lambda: [0.0] * 6)
    for i in keep:
        for name, s in steps[i][1].items():
            totals[name] = [a + b for a, b in zip(totals[name], s)]
    n = len(keep)
    fields = ("calls", "self_ms", "encode_ms", "wait_ms", "work_syncs", "cpu_ms")
    q = statistics.quantiles(periods, n=10)
    return {
        "steps": n,
        "period_ms": periods,
        "period_median_ms": statistics.median(periods),
        "period_p10_ms": q[0],
        "period_p90_ms": q[-1],
        "ctx_mean": statistics.mean(steps[i][0].get("ctx", 0) for i in keep),
        "spans": {
            name: {f: v / n for f, v in zip(fields, tot)}
            for name, tot in sorted(totals.items())
        },
    }


def calibrate():
    """Cost of one sync with nothing to run and with one tiny kernel, and of one span."""
    x = mx.ones((1, 1, 768), mx.bfloat16)
    mx.eval(x)
    noop, one, one_encode, one_wait = [], [], [], []
    for _ in range(300):
        t0 = time.perf_counter()
        mx.async_eval([x])
        mx.eval([x])
        noop.append((time.perf_counter() - t0) * 1e3)
        y = x * 1.0001
        t0 = time.perf_counter()
        mx.async_eval([y])
        t1 = time.perf_counter()
        mx.eval([y])
        t2 = time.perf_counter()
        one.append((t2 - t0) * 1e3)
        one_encode.append((t1 - t0) * 1e3)
        one_wait.append((t2 - t1) * 1e3)
    REC.reset(0)
    REC.mode = "cpu"
    REC.begin_step()
    n = 20000
    t0 = time.perf_counter()
    for _ in range(n):
        REC.call("calibration", int, (), {})
    span_us = (time.perf_counter() - t0) / n * 1e6
    REC.end_step()
    REC.reset(0)
    noop_ms, one_ms = statistics.median(noop), statistics.median(one)
    REC.work_sync_ms = (noop_ms + one_ms) / 2
    return {
        "noop_sync_ms": noop_ms,
        "one_kernel_sync_ms": one_ms,
        "one_kernel_encode_ms": statistics.median(one_encode),
        "one_kernel_wait_ms": statistics.median(one_wait),
        "work_sync_threshold_ms": REC.work_sync_ms,
        "span_overhead_us": span_us,
    }


def check_moe(model):
    """The split MoE gives bit-identical outputs to the port's, in every mode."""
    for rows in (1, 8, 32):
        moe = model.model.layers[3].feed_forward
        x = mx.random.normal((1, rows, moe.router.weight.shape[1])).astype(mx.bfloat16)
        expect = ORIGINAL_MOE(moe, x)
        mx.eval(expect)
        for mode in ("off", "cpu", "sync"):
            REC.reset(rows)
            REC.mode = mode
            REC.begin_step()
            got = moe(x)
            mx.eval(got)
            REC.end_step()
            if not mx.array_equal(expect, got).item():
                raise AssertionError(f"split MoE differs from the port's ({mode})")
    REC.reset(0)


def machine():
    vm = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    swap = subprocess.run(
        ["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True
    ).stdout
    docker = subprocess.run(["pgrep", "-x", "qemu-system-aarch64"], capture_output=True)
    return {
        "time": time.time(),
        "swapouts": int(re.search(r"Swapouts:\s+(\d+)", vm).group(1)),
        "swap_used_mb": float(re.search(r"used = ([\d.]+)M", swap).group(1)),
        "docker_vm": docker.returncode == 0,
    }


def free_percent():
    out = subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout
    m = re.search(r"free percentage: (\d+)%", out)
    return int(m.group(1)) if m else None


def plan(engine, quick):
    grid = GRID[::2] if quick else GRID
    pieces = [k for k in KNOCKOUTS if engine == "vllm" or k not in VLLM_ONLY]
    if quick:
        pieces = pieces[:2]
    runs = []
    for rep in range(1 if quick else 3):
        block = [(mode, r, p) for mode in ("bare", "off", "cpu") for r, p in grid]
        block += [(f"ko:{k}", r, 128) for k in pieces for r in KNOCKOUT_ROWS]
        block += [(f"kc:{k}", 1, 128) for k in pieces]
        if rep == 0:
            block += [("sync", r, p) for r, p in grid]
        random.Random(rep).shuffle(block)
        runs += [(rep, *b) for b in block]
    return runs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("engine", choices=("plain", "vllm"))
    ap.add_argument("out", type=Path)
    ap.add_argument("--quick", action="store_true", help="a short smoke run")
    ap.add_argument("--tokens", type=int, default=256, help="output tokens per run")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    log = args.out / f"runs-{args.engine}.jsonl"
    header = {"kind": "process", "engine": args.engine, "free_percent": free_percent()}

    if args.engine == "plain":
        from mlx_lm.utils import load_model

        model, _ = load_model(Path(MODEL))
        mx.eval(model.parameters())
        build_roles(model)
        run = lambda r, p, s: run_plain(model, r, p, args.tokens, s)  # noqa: E731
    else:
        from vllm import LLM

        install_vllm_step_hooks()
        llm = LLM(
            model=MODEL,
            trust_remote_code=True,
            max_model_len=2048,
            max_num_seqs=32,
            max_num_batched_tokens=2048,
            kv_cache_memory_bytes=1 << 30,
            enable_prefix_caching=False,
            generation_config="vllm",
        )
        run = lambda r, p, s: run_vllm(llm, r, p, args.tokens, s)  # noqa: E731
        run(1, 16, 0)  # warm-up; captures the runner
        model = RUNNER[0].model
        build_roles(model)

    install_model_hooks(args.engine)
    check_moe(model)
    header.update(calibration=calibrate(), weight_bytes=weight_bytes())
    with log.open("a") as f:
        f.write(json.dumps(header) + "\n")

    for rep, mode, rows, prompt in plan(args.engine, args.quick):
        if mode == "bare":
            unpatch(PIECE_PATCHES)
        REC.reset(rows)
        knock = mode[:3] in ("ko:", "kc:")
        REC.mode = "off" if knock else mode
        REC.knockout = KNOCKOUTS[mode[3:]] if knock else set()
        REC.control = mode.startswith("kc:")
        before = machine()
        run(rows, prompt, rows * 10000 + prompt)
        record = summarize()
        record.update(
            kind="run",
            engine=args.engine,
            mode=mode,
            rows=rows,
            prompt=prompt,
            rep=rep,
            machine_before=before,
            machine_after=machine(),
        )
        if mode == "sync":
            record["census"] = {k: dict(v) for k, v in REC.census.items()}
            record["distinct_experts"] = REC.distinct
        if mode == "bare":
            install_model_hooks(args.engine)
        REC.mode, REC.knockout, REC.control = "bare", set(), False
        with log.open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(
            f"{args.engine} rep{rep} {mode:>16} rows {rows:2d} prompt {prompt:4d}: "
            f"{record['period_median_ms']:7.2f} ms/step ({record['steps']} steps)",
            flush=True,
        )
    with log.open("a") as f:
        f.write(json.dumps({"kind": "end", "free_percent": free_percent()}) + "\n")


if __name__ == "__main__":
    main()
