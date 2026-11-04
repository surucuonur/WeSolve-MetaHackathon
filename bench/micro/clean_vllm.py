"""vLLM as shipped, timed from outside: no hooks, no kernel swaps, none of our code.

The 2026-11-04 engine comparison times every version, the unchanged engine too,
with our step hooks installed and vLLM's engine running in the script's own
process. This sends the same requests to a vLLM that loads only the MobileMoE-S
port (the model can't run without it), so the comparison's unchanged engine can
be checked against vLLM itself. Three layouts:

  in-process   VLLM_ENABLE_V1_MULTIPROCESSING=0, as in the comparison
  default      vLLM's default: the engine in its own background process
  hooked       the comparison's unchanged engine: in-process, with breakdown's
               step hooks, each request run through breakdown.run_vllm

Machine load drifts over hours on this Mac, so run the layouts back to back,
alternating, and compare them with each other rather than with older runs.

Each setting is timed with 256 output tokens and with 1 (the prompt pass alone),
so prompt time is the 1-token run and the decode step is
(256-token run - 1-token run) / 255. Records go to <run>/clean-vllm-<layout>.jsonl.

  .venv/bin/python bench/micro/clean_vllm.py results/runs/<run> --layout in-process
"""

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "grader"))
import gpu_lock  # noqa: E402

MODEL = "models/MobileMoE-S-QAT-mlx-bfloat16"
SETTINGS = [(1, 128), (8, 128), (32, 128), (32, 1024)]
REPEATS = 3


def swapouts():
    """The machine's pages swapped out so far, from vm_stat."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    return int(next(l for l in out.splitlines() if l.startswith("Swapouts")).split()[-1].rstrip("."))


def run(llm, rows, prompt, tokens, seed):
    """One batch of requests, built and timed as breakdown.run_vllm does."""
    from vllm import SamplingParams

    t0 = time.perf_counter()
    rng = random.Random(seed)
    prompts = [
        {"prompt_token_ids": [rng.randrange(1000, 100000) for _ in range(prompt)]}
        for _ in range(rows)
    ]
    params = SamplingParams(max_tokens=tokens, temperature=0, ignore_eos=True)
    llm.generate(prompts, params, use_tqdm=False)
    return (time.perf_counter() - t0) * 1e3


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--layout", choices=["in-process", "default", "hooked"], required=True)
    ap.add_argument("--tokens", type=int, default=256, help="output tokens per run")
    args = ap.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
    # vllm-metal's startup check that the KV cache fits; on this Mac it refuses
    # 1 GiB whenever anything pages, so the comparison ran with it off too
    os.environ.setdefault("VLLM_METAL_KV_COMMIT_PROBE", "0")
    if args.layout == "default":
        os.environ.pop("VLLM_ENABLE_V1_MULTIPROCESSING", None)
    else:
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    gpu_lock.hold(f"clean_vllm.py {args.layout}")
    from vllm import LLM

    if args.layout == "hooked":
        import breakdown as bd

        bd.install_vllm_step_hooks()

        def batch(llm, rows, prompt, tokens, seed):  # as repeat_fixes.time_variants times it
            bd.REC.reset(rows)
            t0 = time.perf_counter()
            bd.run_vllm(llm, rows, prompt, tokens, seed)
            return (time.perf_counter() - t0) * 1e3
    else:
        batch = run

    llm = LLM(  # the comparison's settings
        model=MODEL,
        trust_remote_code=True,
        max_model_len=2048,
        max_num_seqs=32,
        max_num_batched_tokens=2048,
        kv_cache_memory_bytes=1 << 30,
        enable_prefix_caching=False,
        generation_config="vllm",
    )
    batch(llm, 1, 16, args.tokens, 0)  # warm-up, as in the comparison

    plan = []
    for rep in range(REPEATS):
        block = [(r, p, t) for r, p in SETTINGS for t in (args.tokens, 1)]
        random.Random(100 + rep).shuffle(block)
        plan += [(rep, *b) for b in block]
    args.out.mkdir(parents=True, exist_ok=True)
    log = args.out / f"clean-vllm-{args.layout}.jsonl"
    records = []
    for rep, rows, prompt, tokens in plan:
        before = swapouts()
        wall = batch(llm, rows, prompt, tokens, rows * 10000 + prompt)
        record = dict(
            layout=args.layout, rows=rows, prompt=prompt, tokens=tokens, rep=rep,
            wall_ms=wall, swapouts=swapouts() - before, time=time.time(),
        )
        records.append(record)
        with log.open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"{args.layout} rep{rep} {rows:2d} x {prompt:4d}, {tokens:3d} tokens: {wall:9.1f} ms", flush=True)

    for rows, prompt in SETTINGS:
        wall = {
            t: statistics.median(r["wall_ms"] for r in records if (r["rows"], r["prompt"], r["tokens"]) == (rows, prompt, t))
            for t in (args.tokens, 1)
        }
        step = (wall[args.tokens] - wall[1]) / (args.tokens - 1)
        print(
            f"{rows:2d} x {prompt:4d}: {rows * args.tokens / wall[args.tokens] * 1e3:6.1f} tokens/s, "
            f"prompt {wall[1]:7.1f} ms, step {step:6.3f} ms"
        )


if __name__ == "__main__":
    main()
