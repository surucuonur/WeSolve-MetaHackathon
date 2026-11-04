"""The Muse agent's kernels in the real decode step, against two baselines.

Variants, applied at runtime to MoE.__call__ in every layer:

  base                the port's MoE, unchanged
  compiled-<slot>     the slot's reference code under mx.compile (our baseline)
  muse-<slot>         the Muse agent's attempt for the slot
  compiled-all        with several slots: all their references compiled at once
  muse-all            with several slots: all the attempts at once
  <name>              each --variant NAME:SLOT=FOLDER[+SLOT=FOLDER...]: other
                      attempts, timed in the same run but outside muse-all,
                      such as an earlier round's or one not submitted; the
                      FOLDER `compiled` is that slot's compiled reference

G2 end to end: each variant's greedy tokens over 24 generated tokens, at 1
and 32 requests, must equal base's. G3: the steady decode step period, as
repeat_fixes.py measures it (3 repeats, shuffled), at 1, 8 and 32 requests.

  .venv/bin/python bench/grader/time_candidates.py plain results/runs/<date>-muse <slot>=<attempt dir> ...
  .venv/bin/python bench/grader/time_candidates.py vllm  results/runs/<date>-muse <slot>=<attempt dir> ...
      [--variant NAME:SLOT=FOLDER[+SLOT=FOLDER...] ...]
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
from check_candidate import load, parts  # noqa: E402
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


def attempt_arg(value):
    """SLOT=FOLDER -> (slot, folder)."""
    slot, sep, folder = value.partition("=")
    if not sep or slot not in slots.SLOTS:
        raise argparse.ArgumentTypeError(
            f"{value!r}: give SLOT=FOLDER, SLOT one of {', '.join(slots.SLOTS)}"
        )
    return slot, Path(folder)


def variant_arg(value):
    """NAME:SLOT=FOLDER[+SLOT=FOLDER...] -> (name, [(slot, folder), ...])."""
    name, sep, spec = value.partition(":")
    if not sep or not name:
        raise argparse.ArgumentTypeError(
            f"{value!r}: give NAME:SLOT=FOLDER[+SLOT=FOLDER...]"
        )
    return name, [attempt_arg(a) for a in spec.split("+")]


def implementations(attempts):
    """make_moe_call keyword arguments for (slot, folder) pairs; `compiled` is the reference."""
    fns = {}
    for slot, folder in attempts:
        if str(folder) == "compiled":
            fns |= {p: mx.compile(slots.REFERENCE[p]) for p in parts(slot)}
        else:
            fns |= implementation(load(folder, slot))
    return fns


class Variants:
    def __init__(self, attempts, extra=()):
        """*attempts*: (slot, folder) pairs whose slots share no part; *extra*:
        (name, attempts) pairs, each its own variant."""
        self.original = mobilemoe.MoE.__call__
        self.calls = {"base": self.original}
        compiled, muse = {}, {}
        for slot, folder in attempts:
            ref = {p: mx.compile(slots.REFERENCE[p]) for p in parts(slot)}
            fns = implementation(load(folder, slot))
            self.calls[f"compiled-{slot}"] = slots.make_moe_call(**ref)
            self.calls[f"muse-{slot}"] = slots.make_moe_call(**fns)
            compiled |= ref
            muse |= fns
        if len(attempts) > 1:
            self.calls["compiled-all"] = slots.make_moe_call(**compiled)
            self.calls["muse-all"] = slots.make_moe_call(**muse)
        for name, more in extra:
            if name in self.calls:
                raise SystemExit(f"--variant {name}: that name is taken")
            self.calls[name] = slots.make_moe_call(**implementations(more))
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
    ap.add_argument(
        "attempts", nargs="+", type=attempt_arg, metavar="SLOT=FOLDER",
        help="the Muse agent's attempt folder per slot",
    )
    ap.add_argument(
        "--variant", type=variant_arg, action="append", default=[],
        metavar="NAME:SLOT=FOLDER[+SLOT=FOLDER...]",
        help="another set of attempts to time, outside muse-all",
    )
    ap.add_argument("--tokens", type=int, default=256, help="output tokens per run")
    args = ap.parse_args()
    taken = [p for slot, _ in args.attempts for p in parts(slot)]
    if len(taken) != len(set(taken)):
        ap.error(f"two attempts replace the same part: {sorted(taken)}")
    gpu_lock.hold(
        f"time_candidates.py {args.engine} "
        + " ".join(f"{s}={f}" for s, f in args.attempts)
    )
    args.out.mkdir(parents=True, exist_ok=True)
    log = args.out / f"muse-{args.engine}.jsonl"

    if args.engine == "plain":
        from mlx_lm.utils import load_model

        model, _ = load_model(Path(bd.MODEL))
        mx.eval(model.parameters())
        variants = Variants(args.attempts, args.variant)
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
        variants = Variants(args.attempts, args.variant)
        checks = rf.vllm_checks(llm, variants)

    rf.time_variants(args.engine, variants, checks, run, log)


if __name__ == "__main__":
    main()
