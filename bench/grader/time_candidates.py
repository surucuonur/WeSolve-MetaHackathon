"""Muse Spark's kernels in the real decode step, against two baselines.

Variants, applied at runtime to MoE.__call__ in every layer:

  base          the port's MoE, unchanged
  compiled      the same slot's reference code under mx.compile (our baseline)
  muse          Muse Spark's best attempt for the slot
  claude_fixed  a separately labelled Claude version, only if one exists

G2 end to end: each variant's greedy tokens over 24 generated tokens, at 1
and 32 requests, must equal base's. G3: the steady decode step period, as
repeat_fixes.py measures it (3 repeats, shuffled), at 1, 8 and 32 requests.

  .venv/bin/python bench/grader/time_candidates.py plain results/runs/<date>-muse <slot> <attempt dir>
  .venv/bin/python bench/grader/time_candidates.py vllm  results/runs/<date>-muse <slot> <attempt dir>
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "micro"))
import breakdown as bd  # noqa: E402  sets the vLLM environment before importing it
import gpu_lock  # noqa: E402
import mlx.core as mx  # noqa: E402
import repeat_fixes as rf  # noqa: E402
from check_candidate import load  # noqa: E402
from mlx_lm.models import mobilemoe  # noqa: E402

import slots  # noqa: E402

TOKENS = 24


def as_routing(fn):
    def routing(x, w_router, bias):
        out = fn(x=x, w_router=w_router, bias=bias)
        return out["inds"], out["weights"]

    return routing


def as_combine(fn):
    return lambda y, weights, shared: fn(y=y, weights=weights, shared=shared)["out"]


def as_experts(fn):
    names = slots.SLOTS["experts"][0]
    return lambda *args: fn(**dict(zip(names, args)))["y"]


def implementation(parts):
    """make_moe_call keyword arguments for the swapped parts."""
    wrap = {"routing": as_routing, "combine": as_combine, "experts": as_experts}
    return {p: wrap[p](fn) for p, fn in parts.items()}


class Variants:
    def __init__(self, slot, attempts):
        self.original = mobilemoe.MoE.__call__
        part_names = ["routing", "combine"] if slot == "routing+combine" else [slot]
        compiled = {p: mx.compile(slots.REFERENCE[p]) for p in part_names}
        self.calls = {
            "base": self.original,
            "compiled": slots.make_moe_call(**compiled),
        }
        for name, attempt in attempts.items():
            self.calls[name] = slots.make_moe_call(
                **implementation(load(attempt, slot))
            )
        self.names = list(self.calls)

    def apply(self, name):
        mobilemoe.MoE.__call__ = self.calls[name]


def plain_checks(model, variants):
    """Greedy tokens over TOKENS steps and one step's kernel count, per variant."""
    from mlx_lm.models.cache import make_prompt_cache

    out = {}
    for rows in (1, 32):
        tokens = {}
        for name in variants.names:
            variants.apply(name)
            mx.random.seed(rows)
            ids = mx.random.randint(0, 100000, (rows, 128))
            cache = make_prompt_cache(model)
            seq = [bd.prefill(model, ids, cache)]
            for _ in range(TOKENS - 1):
                seq.append(bd.greedy(model(seq[-1], cache=cache)))
                mx.eval(seq[-1])
            tokens[name] = mx.concatenate(seq, axis=1).tolist()
            out[(name, rows)] = rf.count_ops([model(seq[-1], cache=cache)])
        for name in variants.names[1:]:
            out[(name, rows)]["exact"] = tokens[name] == tokens["base"]
            out[(name, rows)].update(rf.divergence(tokens[name], tokens["base"]))
    variants.apply("base")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("engine", choices=("plain", "vllm"))
    ap.add_argument("out", type=Path)
    ap.add_argument("slot", choices=tuple(slots.SLOTS))
    ap.add_argument("muse", type=Path, help="Muse Spark's attempt folder")
    ap.add_argument("--claude-fixed", type=Path, help="Claude's labelled version")
    ap.add_argument("--tokens", type=int, default=256, help="output tokens per run")
    args = ap.parse_args()
    gpu_lock.hold(f"time_candidates.py {args.engine} {args.slot} {args.muse}")
    args.out.mkdir(parents=True, exist_ok=True)
    log = args.out / f"muse-{args.engine}.jsonl"
    attempts = {"muse": args.muse}
    if args.claude_fixed:
        attempts["claude_fixed"] = args.claude_fixed

    if args.engine == "plain":
        from mlx_lm.utils import load_model

        model, _ = load_model(Path(bd.MODEL))
        mx.eval(model.parameters())
        variants = Variants(args.slot, attempts)
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
        variants = Variants(args.slot, attempts)
        checks = rf.vllm_checks(llm, variants)

    rf.time_variants(args.engine, variants, checks, run, log)


if __name__ == "__main__":
    main()
