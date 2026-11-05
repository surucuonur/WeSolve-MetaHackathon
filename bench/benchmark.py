"""Benchmark WeSolve: vLLM as shipped, with MLX's compiled code, and with WeSolve's kernels.

Times whole requests (prompt reading and generation together) and each phase,
on this Mac, with the README's workloads, and prints the README's tables. Every
variant must give exactly the unchanged model's tokens. The timing is
bench/grader/time_candidates.py's, which Claude's G3 uses:

  base          vLLM (or MLX) as shipped
  compiled-all  the experts and routing code under mx.compile, no new kernel
  muse-all      WeSolve: round 4's experts and routing kernels

Setup first (README, Setup): the virtual environment and the converted model in
models/. Quit other heavy apps; the Mac must not be short of memory.

  .venv/bin/python bench/benchmark.py              # vLLM, 4 workloads x 5 runs (about 25 minutes)
  .venv/bin/python bench/benchmark.py --quick      # 1 and 32 requests of 128 tokens, 3 runs (about 6 minutes)
  .venv/bin/python bench/benchmark.py --mlx        # also MLX on its own, no server (about 10 more minutes)
  .venv/bin/python bench/benchmark.py --report results/runs/<folder>   # tables from an earlier run
"""

import argparse
import json
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNELS = ROOT / "muse-spark/round-4/kernels"
ATTEMPTS = [f"experts={KERNELS / 'experts/attempt-1'}", f"routing={KERNELS / 'routing/attempt-1'}"]
MODEL = ROOT / "models/MobileMoE-S-QAT-mlx-bfloat16"
LABELS = ("as shipped", "+ compiled MLX code", "+ WeSolve")


def run(engine, out, settings, repeats):
    cmd = [sys.executable, str(ROOT / "bench/grader/time_candidates.py"), engine, str(out), *ATTEMPTS,
           "--settings", settings, "--repeats", str(repeats)]
    print(f"\n$ {' '.join(cmd)}\n", flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def report(folder):
    """The README's tables from a time_candidates.py run folder."""
    for engine in ("vllm", "plain"):
        log = folder / f"muse-{engine}.jsonl"
        if not log.exists():
            continue
        records = [json.loads(line) for line in log.open()]
        runs = [r for r in records if r.get("kind") == "run"]
        exact = all(r.get("exact", True) for r in records if r.get("kind") == "check")
        workloads = sorted({(r["rows"], r["prompt"]) for r in runs})
        name = "vLLM" if engine == "vllm" else "MLX alone"
        # A run with more slots than WeSolve's two times it as muse-experts-routing
        # (results/runs/2026-11-04-muse-round-4/); this script's runs as muse-all.
        pair = "experts-routing" if any(r["variant"] == "muse-experts-routing" for r in runs) else "all"
        variants = dict(zip(("base", f"compiled-{pair}", f"muse-{pair}"), LABELS))
        wesolve = f"muse-{pair}"

        def med(variant, w, key):
            values = [r[key] for r in runs if r["variant"] == variant and (r["rows"], r["prompt"]) == w]
            return statistics.median(values) if values else None

        repeats = max(r["rep"] for r in runs) + 1
        print(f"\n## {name}: {folder} ({repeats} run{'s' if repeats > 1 else ''} per cell, medians)")
        print(f"Tokens identical to the unchanged model in every variant: {'yes' if exact else 'NO'}\n")
        print("Whole request, prompt reading included, generated tokens per second:\n")
        print("| Requests × prompt | " + " | ".join(f"{name} {label}" for label in variants.values()) + " | WeSolve vs as shipped |")
        print("|---" * (len(variants) + 2) + "|")
        for w in workloads:
            tps = {v: w[0] * runs[0]["tokens"] / med(v, w, "wall_ms") * 1000 for v in variants}
            cells = " | ".join(f"{tps[v]:,.0f}" for v in variants)
            print(f"| {w[0]} × {w[1]:,} | {cells} | {100 * (tps[wesolve] / tps['base'] - 1):+.1f}% |")
        print("\nGeneration only, decode steps, tokens per second:\n")
        print("| Requests × prompt | " + " | ".join(f"{name} {label}" for label in variants.values()) + " | WeSolve vs as shipped |")
        print("|---" * (len(variants) + 2) + "|")
        for w in workloads:
            tps = {v: w[0] / med(v, w, "period_median_ms") * 1000 for v in variants}
            cells = " | ".join(f"{tps[v]:,.0f}" for v in variants)
            print(f"| {w[0]} × {w[1]:,} | {cells} | {100 * (tps[wesolve] / tps['base'] - 1):+.1f}% |")
        if engine == "vllm":  # the plain engine doesn't time prompt reading on its own
            print("\nPrompt reading, ms:\n")
            print("| Requests × prompt | " + " | ".join(f"{name} {label}" for label in variants.values()) + " | WeSolve vs as shipped |")
            print("|---" * (len(variants) + 2) + "|")
            for w in workloads:
                ms = {v: med(v, w, "prompt_ms") for v in variants}
                cells = " | ".join(f"{ms[v]:,.0f}" for v in variants)
                print(f"| {w[0]} × {w[1]:,} | {cells} | {100 * (ms[wesolve] / ms['base'] - 1):+.1f}% time |")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quick", action="store_true", help="1 and 32 requests of 128 tokens, 3 runs each")
    ap.add_argument("--mlx", action="store_true", help="also time MLX on its own (no server)")
    ap.add_argument("--settings", help="requests x prompt tokens, e.g. 1x128,32x1024 (default: the README's four)")
    ap.add_argument("--repeats", type=int, help="runs per variant and workload (default 5, or 3 with --quick)")
    ap.add_argument("--out", type=Path, help="results folder (default results/runs/<date>-benchmark)")
    ap.add_argument("--report", type=Path, metavar="FOLDER", help="only print the tables of an earlier run")
    args = ap.parse_args()
    if args.report:
        report(args.report)
        return
    if not MODEL.is_dir():
        sys.exit(f"{MODEL} is missing: convert the model first (README, Setup: bench/convert_qat.py)")
    settings = args.settings or ("1x128,32x128" if args.quick else "1x128,8x128,32x128,32x1024")
    repeats = args.repeats or (3 if args.quick else 5)
    out = args.out or ROOT / f"results/runs/{datetime.now(timezone.utc):%Y-%m-%d-%H%M}-benchmark"
    run("vllm", out, settings, repeats)
    if args.mlx:
        run("plain", out, settings, repeats)
    report(out)


if __name__ == "__main__":
    main()
