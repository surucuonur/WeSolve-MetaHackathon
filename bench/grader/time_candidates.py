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
and 32 requests, must equal base's. G3 (vllm): each run's steady decode step
period, its prompt time (until every prompt was processed) and its
whole-request time, as repeat_fixes.py measures them (G3_REPEATS repeats,
shuffled), at 1, 8 and 32 requests of 128-token prompts and 32 of 1,024. A muse-*
variant passes against base and against its compiled-* counterpart when
one phase, decode or prompt, is faster in every setting, neither is slower
in any, and no request is slower; faster and slower mean the two ranges over
the repeats don't overlap (claude/adr/0011-decode-and-prefill.md). With 5
repeats each, two identical engines' ranges miss each other by chance 1 time
in 252 per direction; with 3 it was 1 in 20, and the "never slower" checks,
8 per phase, would fail an unchanged phase about a third of the time. The
verdict goes to g3-vllm.json.

  .venv/bin/python bench/grader/time_candidates.py plain results/runs/<date>-muse <slot>=<attempt dir> ...
  .venv/bin/python bench/grader/time_candidates.py vllm  results/runs/<date>-muse <slot>=<attempt dir> ...
      [--variant NAME:SLOT=FOLDER[+SLOT=FOLDER...] ...]
"""

import argparse
import json
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
G3_SETTINGS = "1x128,8x128,32x128,32x1024"
G3_REPEATS = 5
MEASURES = {"decode": "period_median_ms", "prompt": "prompt_ms", "request": "wall_ms"}


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


def against(c, b):
    """Two (min, max) ranges of a time: "faster" when c's lies below b's, "slower" above."""
    return "faster" if c[1] < b[0] else "slower" if c[0] > b[1] else "same"


def g3(log, names, settings, out):
    """Each muse-* variant against base and its compiled-* counterpart, from *log*'s vLLM runs."""
    runs = [r for r in map(json.loads, log.open()) if r["kind"] == "run" and r["engine"] == "vllm"]

    def ranges(name, rows, prompt):
        rs = [r for r in runs if (r["variant"], r["rows"], r["prompt"]) == (name, rows, prompt)]
        return {m: (min(r[k] for r in rs), max(r[k] for r in rs)) for m, k in MEASURES.items()}

    verdict = {}
    for name in [n for n in names if n.startswith("muse-")]:
        bars = ("base", "compiled-" + name.removeprefix("muse-"))
        if bars[1] not in names:  # a --variant named muse-*: no compiled counterpart to judge against
            print(f"G3 {name}: not judged, no {bars[1]} variant")
            continue
        verdict[name] = {}
        for bar in bars:
            settings_v = {
                f"{rows}x{prompt}": {
                    m: against(c, b)
                    for (m, c), b in zip(ranges(name, rows, prompt).items(), ranges(bar, rows, prompt).values())
                }
                for rows, prompt in settings
            }
            phases = {
                m: "faster" if all(v[m] == "faster" for v in settings_v.values())
                else "slower" if any(v[m] == "slower" for v in settings_v.values())
                else "not slower"
                for m in MEASURES
            }
            passes = "faster" in (phases["decode"], phases["prompt"]) and "slower" not in phases.values()
            verdict[name][bar] = {"pass": passes, "phases": phases, "settings": settings_v}
            print(f"G3 {name} vs {bar}: {'pass' if passes else 'fail'} {phases}")
        verdict[name]["g3"] = all(v["pass"] for v in verdict[name].values())
    out.write_text(json.dumps(verdict, indent=1) + "\n")


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
    ap.add_argument("--repeats", type=int, default=G3_REPEATS, help="shuffled repeats per variant and setting")
    ap.add_argument(
        "--settings", default=G3_SETTINGS, metavar="ROWSxPROMPT,...",
        help="request counts and prompt lengths to time (G3: the default)",
    )
    ap.add_argument(
        "--max-model-len", type=int, default=2048,
        help="vLLM's tokens per request, prompt and output (G3: the default; the model allows 8,192)",
    )
    args = ap.parse_args()
    settings = [tuple(int(n) for n in s.split("x")) for s in args.settings.split(",")]
    if args.engine == "vllm" and max(p for _, p in settings) + args.tokens > args.max_model_len:
        ap.error(f"a prompt of {max(p for _, p in settings)} and {args.tokens} output tokens exceed --max-model-len")
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
            max_model_len=args.max_model_len,
            max_num_seqs=32,
            max_num_batched_tokens=2048,
            # 32 rows x 384 tokens need 240 MiB; 32 x 1,280 need 800 MiB
            kv_cache_memory_bytes=(1 << 30) if max(p for _, p in settings) > 128 else 512 << 20,
            enable_prefix_caching=False,
            generation_config="vllm",
        )
        run = lambda r, p, s: bd.run_vllm(llm, r, p, args.tokens, s)  # noqa: E731
        run(1, 16, 0)  # warm-up; captures the runner
        variants = Variants(args.attempts, args.variant)
        checks = rf.vllm_checks(llm, variants)

    rf.time_variants(args.engine, variants, checks, run, log, settings, args.repeats, args.tokens)
    if args.engine == "vllm":
        g3(log, variants.names, settings, args.out / "g3-vllm.json")


if __name__ == "__main__":
    main()
